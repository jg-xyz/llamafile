#!/bin/bash
# bonsai-turso.sh — Query a Turso (libsql) database using Bonsai llamafile
#
# HOW IT WORKS:
#   This script implements the "tool use" (function calling) pattern for AI models.
#   Instead of the model hallucinating database answers, we give it real tools it can
#   call to fetch actual data, then let it reason over the results.
#
#   The flow:
#     1. We define "tools" (functions the model is allowed to call) as JSON schemas.
#        The schemas tell the model what each tool does and what arguments it takes.
#     2. We send the user's prompt + tool definitions to the llamafile server.
#     3. The model either:
#        a) Returns a final answer  → we print it and exit.
#        b) Returns tool_calls      → it wants us to run database operations.
#     4. For each tool call, we execute the real Turso HTTP API and send results back.
#     5. We loop back to step 2 so the model can reason over the new data.
#     6. Repeat until the model produces a final answer (or we hit MAX_TOOL_ROUNDS).
#
#   Turso's own MCP server (`turso mcp start`) exposes these same operations via the
#   MCP protocol (JSON-RPC over stdio/SSE). This script bypasses the MCP transport
#   layer and calls Turso's HTTP API directly — same semantics, simpler dependencies.
#
# Usage:
#   ./bonsai-turso.sh --db-url <turso-db-url> --token <auth-token> --prompt "question"
#
# Example:
#   ./bonsai-turso.sh \
#     --db-url https://mydb-org.turso.io \
#     --token eyJhb... \
#     --prompt "how many users signed up last week?"
#
# Requirements: curl, jq, llamafile server running (or set LLAMAFILE_URL)

set -euo pipefail

# LLAMAFILE_URL: where the llamafile server is listening.
# Start the server with: ./bonsai.llamafile (it opens :8080 by default)
# Override via environment: LLAMAFILE_URL=http://myserver:8080 ./bonsai-turso.sh ...
LLAMAFILE_URL="${LLAMAFILE_URL:-http://localhost:8080}"
DB_URL=""
AUTH_TOKEN=""
PROMPT=""

# Safety cap: if the model keeps calling tools without converging on an answer,
# stop after this many rounds to avoid runaway API usage.
MAX_TOOL_ROUNDS=10

usage() {
  grep '^#' "$0" | sed 's/^# \?//'
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --db-url)        DB_URL="$2";       shift 2 ;;
    --token)         AUTH_TOKEN="$2";   shift 2 ;;
    --prompt)        PROMPT="$2";       shift 2 ;;
    --llamafile-url) LLAMAFILE_URL="$2"; shift 2 ;;
    -h|--help)       usage ;;
    *) echo "Unknown argument: $1" >&2; usage ;;
  esac
done

[[ -z "$DB_URL" ]]     && { echo "error: --db-url required" >&2; exit 1; }
[[ -z "$AUTH_TOKEN" ]] && { echo "error: --token required" >&2; exit 1; }
[[ -z "$PROMPT" ]]     && { echo "error: --prompt required" >&2; exit 1; }

command -v jq   >/dev/null || { echo "error: jq not found" >&2; exit 1; }
command -v curl >/dev/null || { echo "error: curl not found" >&2; exit 1; }

# Confirm the llamafile server is up before doing anything.
# The /health endpoint returns 200 when the model is loaded and ready.
if ! curl -sf "${LLAMAFILE_URL}/health" >/dev/null 2>&1; then
  echo "error: llamafile server not reachable at ${LLAMAFILE_URL}" >&2
  echo "  Start it with: ./bonsai.llamafile" >&2
  exit 1
fi

# ─── TOOL DEFINITIONS ─────────────────────────────────────────────────────────
#
# Tools are described as JSON Schema. The model reads these descriptions to decide
# WHEN and HOW to call each tool. Good descriptions are critical — vague descriptions
# cause the model to misuse tools or skip them entirely.
#
# Each tool has:
#   name        — identifier the model uses to request the tool
#   description — natural language explanation of what the tool does
#   parameters  — JSON Schema describing the arguments the model must provide
#   required    — which parameters are mandatory (model will always include these)
#
# Turso is SQLite-compatible, so tools mirror SQLite introspection commands.
TOOLS='[
  {
    "type": "function",
    "function": {
      "name": "list_tables",
      "description": "List all tables in the Turso database",
      "parameters": {
        "type": "object",
        "properties": {}
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "describe_table",
      "description": "Get the schema for a specific table",
      "parameters": {
        "type": "object",
        "properties": {
          "table_name": {
            "type": "string",
            "description": "Name of the table to describe"
          }
        },
        "required": ["table_name"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "execute_sql",
      "description": "Execute a SQL query. Use SELECT for reads. Avoid destructive operations unless explicitly asked.",
      "parameters": {
        "type": "object",
        "properties": {
          "sql": {
            "type": "string",
            "description": "The SQL statement to execute"
          },
          "params": {
            "type": "array",
            "items": {},
            "description": "Optional positional parameters for the query"
          }
        },
        "required": ["sql"]
      }
    }
  }
]'

# ─── TOOL EXECUTION ───────────────────────────────────────────────────────────
#
# When the model requests a tool, we run it here against Turso's HTTP API.
# The Turso v2 pipeline endpoint accepts batched SQL statements as JSON and
# returns results as JSON — no SQLite client library needed, just curl.
#
# All results are returned as raw JSON. The model receives this JSON as the
# tool result and uses it to form its next response or tool call.
execute_tool() {
  local name="$1"
  local args="$2"  # JSON string from the model, e.g. {"table_name":"users"}

  case "$name" in
    list_tables)
      # SQLite stores table/view names in sqlite_master (same as standard SQLite).
      # Single quotes inside the shell single-quoted string require escaping via '"'"'.
      curl -sf -X POST "${DB_URL}/v2/pipeline" \
        -H "Authorization: Bearer ${AUTH_TOKEN}" \
        -H "Content-Type: application/json" \
        -d '{"requests":[{"type":"execute","stmt":{"sql":"SELECT name, type FROM sqlite_master WHERE type IN ('"'"'table'"'"','"'"'view'"'"') ORDER BY name"}}]}'
      ;;

    describe_table)
      local table
      table=$(echo "$args" | jq -r '.table_name')
      # PRAGMA table_info returns column definitions: cid, name, type, notnull, dflt_value, pk.
      # We use parameterized query (args array) to avoid SQL injection from model output.
      curl -sf -X POST "${DB_URL}/v2/pipeline" \
        -H "Authorization: Bearer ${AUTH_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "$(jq -n --arg t "$table" \
          '{"requests":[{"type":"execute","stmt":{"sql":"PRAGMA table_info(?)", "args":[{"type":"text","value":$t}]}}]}')"
      ;;

    execute_sql)
      local sql params_json
      sql=$(echo "$args" | jq -r '.sql')
      # Convert the model's params array to Turso's typed-value format.
      # Turso requires each param to declare its type; we default everything to "text"
      # since the model doesn't know SQLite types and "text" is safely coercible.
      params_json=$(echo "$args" | jq -r '.params // [] | map({"type":"text","value":tostring})')
      curl -sf -X POST "${DB_URL}/v2/pipeline" \
        -H "Authorization: Bearer ${AUTH_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "$(jq -n --arg s "$sql" --argjson p "$params_json" \
          '{"requests":[{"type":"execute","stmt":{"sql":$s,"args":$p}}]}')"
      ;;

    *)
      # Return a recognizable error so the model can self-correct rather than hang.
      echo '{"error":"unknown tool"}'
      ;;
  esac
}

# ─── CONVERSATION STATE ───────────────────────────────────────────────────────
#
# The OpenAI chat completions API is stateless — every request must include the
# full conversation history. We maintain the `messages` array ourselves, appending
# each turn (user → assistant → tool results → assistant → ...) as we go.
#
# Message roles:
#   user      — human input
#   assistant — model output (may include tool_calls instead of content)
#   tool      — our response to a tool call (indexed by tool_call_id)
messages=$(jq -n --arg p "$PROMPT" '[{"role":"user","content":$p}]')
round=0

# ─── TOOL-CALL LOOP ───────────────────────────────────────────────────────────
while [[ $round -lt $MAX_TOOL_ROUNDS ]]; do

  # Send the full conversation + available tools to the model.
  # temperature:0.3 — slightly deterministic for data queries (less creative = fewer hallucinations)
  # max_tokens:2048  — enough headroom for multi-step reasoning and SQL generation
  response=$(curl -sf "${LLAMAFILE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(jq -n \
      --argjson messages "$messages" \
      --argjson tools "$TOOLS" \
      '{"model":"bonsai","messages":$messages,"tools":$tools,"max_tokens":2048,"temperature":0.3}')")

  # finish_reason tells us why the model stopped generating:
  #   "stop"       — produced a complete answer, we're done
  #   "tool_calls" — wants to call one or more tools before answering
  #   "length"     — hit max_tokens, response may be truncated
  finish_reason=$(echo "$response" | jq -r '.choices[0].finish_reason')

  # Extract the assistant's message and append it to conversation history.
  # This is required — subsequent requests must include the assistant's tool_calls
  # so the model knows what it asked for when we send back the tool results.
  assistant_msg=$(echo "$response" | jq '.choices[0].message')
  messages=$(echo "$messages" | jq --argjson msg "$assistant_msg" '. + [$msg]')

  if [[ "$finish_reason" == "tool_calls" ]]; then
    ((round++))

    # The model may request multiple tools in a single response.
    # We execute each one and append the results before looping back.
    # Process substitution (<()) avoids a subshell for the while loop so
    # our `messages` variable updates are visible outside the loop.
    while IFS= read -r tool_call; do
      tool_id=$(echo "$tool_call"   | jq -r '.id')       # unique ID linking call to result
      tool_name=$(echo "$tool_call" | jq -r '.function.name')
      tool_args=$(echo "$tool_call" | jq -r '.function.arguments')  # JSON string

      # Log tool activity to stderr so stdout stays clean for the final answer.
      echo "[tool] ${tool_name}: ${tool_args}" >&2
      tool_result=$(execute_tool "$tool_name" "$tool_args")
      echo "[result] ${tool_result}" >&2

      # Append the tool result with role "tool" and the matching tool_call_id.
      # The model uses tool_call_id to correlate which result belongs to which call.
      messages=$(echo "$messages" | jq \
        --arg id "$tool_id" \
        --arg name "$tool_name" \
        --arg result "$tool_result" \
        '. + [{"role":"tool","tool_call_id":$id,"name":$name,"content":$result}]')
    done < <(echo "$assistant_msg" | jq -c '.tool_calls[]')

  else
    # finish_reason is "stop" (or "length") — model has a final answer.
    # Print only the content to stdout; debug output went to stderr.
    echo "$response" | jq -r '.choices[0].message.content'
    exit 0
  fi
done

# If we get here, the model kept calling tools without converging.
# Could indicate an infinite loop in tool calls or a model reasoning failure.
echo "error: exceeded max tool rounds (${MAX_TOOL_ROUNDS})" >&2
exit 1
