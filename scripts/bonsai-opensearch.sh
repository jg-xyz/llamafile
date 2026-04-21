#!/bin/bash
# bonsai-opensearch.sh — Query an OpenSearch cluster using Bonsai llamafile
#
# HOW IT WORKS:
#   OpenSearch stores data in "indices" (like tables) and is queried using a JSON
#   query language called Query DSL — very different from SQL. This script teaches
#   the model to build Query DSL by giving it tools that map to OpenSearch REST
#   endpoints, then executes those queries and feeds results back for reasoning.
#
#   The flow:
#     1. We define tools as JSON schemas (what the model can call and with what args).
#     2. We send the user's prompt + tool definitions to the llamafile server.
#     3. The model either:
#        a) Returns a final answer  → print it and exit.
#        b) Returns tool_calls      → it wants to run search/inspection operations.
#     4. For each tool call, we hit the OpenSearch REST API and return the raw JSON.
#     5. We loop back so the model can reason over the results and decide next steps.
#
#   OpenSearch has an MCP server plugin that exposes these same operations via the
#   MCP protocol. This script calls the REST API directly — same semantics, no
#   Node.js or plugin required.
#
# Usage:
#   ./bonsai-opensearch.sh --host <opensearch-url> --index <index-name> --prompt "question"
#
# Example:
#   ./bonsai-opensearch.sh \
#     --host https://localhost:9200 \
#     --user admin:password \
#     --index logs-2026 \
#     --insecure \
#     --prompt "find all ERROR level log entries from the auth service today"
#
# Requirements: curl, jq, llamafile server running (or set LLAMAFILE_URL)

set -euo pipefail

# LLAMAFILE_URL: where the llamafile server is listening.
# Start the server with: ./bonsai.llamafile (opens :8080 by default)
LLAMAFILE_URL="${LLAMAFILE_URL:-http://localhost:8080}"
OS_HOST=""
OS_USER=""    # optional: user:password for HTTP basic auth (OpenSearch default security)
OS_INDEX=""   # the "default" index — model can query others via the search tool
PROMPT=""
MAX_TOOL_ROUNDS=10

# CURL_OPTS accumulates flags that apply to every OpenSearch request.
# Start with -sf (silent + fail on HTTP errors). More flags added below.
CURL_OPTS=(-sf)

usage() {
  grep '^#' "$0" | sed 's/^# \?//'
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --host)          OS_HOST="$2";       shift 2 ;;
    --user)          OS_USER="$2";       shift 2 ;;
    --index)         OS_INDEX="$2";      shift 2 ;;
    --prompt)        PROMPT="$2";        shift 2 ;;
    --llamafile-url) LLAMAFILE_URL="$2"; shift 2 ;;
    # --insecure disables TLS certificate verification. Use for self-signed certs
    # in dev/local OpenSearch instances that lack a trusted CA cert.
    --insecure)      CURL_OPTS+=(-k);    shift ;;
    -h|--help)       usage ;;
    *) echo "Unknown argument: $1" >&2; usage ;;
  esac
done

[[ -z "$OS_HOST" ]]  && { echo "error: --host required" >&2; exit 1; }
[[ -z "$OS_INDEX" ]] && { echo "error: --index required" >&2; exit 1; }
[[ -z "$PROMPT" ]]   && { echo "error: --prompt required" >&2; exit 1; }

command -v jq   >/dev/null || { echo "error: jq not found" >&2; exit 1; }
command -v curl >/dev/null || { echo "error: curl not found" >&2; exit 1; }

# Add basic auth if credentials were provided.
# OpenSearch's default security plugin requires authentication even for reads.
[[ -n "$OS_USER" ]] && CURL_OPTS+=(-u "$OS_USER")

# Wrapper so all OpenSearch calls share the same flags (auth, TLS, etc.)
# without repeating them at every call site.
os_curl() {
  curl "${CURL_OPTS[@]}" "$@"
}

if ! curl -sf "${LLAMAFILE_URL}/health" >/dev/null 2>&1; then
  echo "error: llamafile server not reachable at ${LLAMAFILE_URL}" >&2
  echo "  Start it with: ./bonsai.llamafile" >&2
  exit 1
fi

# ─── TOOL DEFINITIONS ─────────────────────────────────────────────────────────
#
# We use `jq -n` to build the JSON so we can inject the $OS_INDEX shell variable
# into the "default" field. Plain single-quoted strings can't expand variables.
#
# OpenSearch tool set mirrors what the OpenSearch MCP plugin exposes:
#   search      — the main query tool; builds and runs Query DSL
#   list_indices — discover what data exists in the cluster
#   get_mapping  — inspect field types before querying (avoids type mismatch errors)
#   count        — quick document count without fetching data
#
# The "description" field is what the model reads to decide when to use each tool.
# Specific descriptions produce better tool selection than generic ones.
TOOLS=$(jq -n --arg idx "$OS_INDEX" '[
  {
    "type": "function",
    "function": {
      "name": "search",
      "description": "Search an OpenSearch index using a query DSL body",
      "parameters": {
        "type": "object",
        "properties": {
          "index": {
            "type": "string",
            "description": "Index or index pattern to search (default: the configured index)",
            "default": $idx
          },
          "query": {
            "type": "object",
            "description": "OpenSearch query DSL object (e.g. {\"match\":{\"message\":\"error\"}})"
          },
          "size": {
            "type": "integer",
            "description": "Max number of hits to return (default 10)"
          },
          "sort": {
            "type": "array",
            "description": "Sort specification array"
          },
          "aggs": {
            "type": "object",
            "description": "Aggregations to compute alongside results"
          }
        },
        "required": ["query"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "list_indices",
      "description": "List all indices in the OpenSearch cluster with doc counts and sizes",
      "parameters": {
        "type": "object",
        "properties": {
          "pattern": {
            "type": "string",
            "description": "Optional glob pattern to filter index names"
          }
        }
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_mapping",
      "description": "Get field mappings for an index to understand its schema",
      "parameters": {
        "type": "object",
        "properties": {
          "index": {
            "type": "string",
            "description": "Index name"
          }
        },
        "required": ["index"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "count",
      "description": "Count documents matching a query without returning them",
      "parameters": {
        "type": "object",
        "properties": {
          "index": {
            "type": "string",
            "description": "Index or index pattern"
          },
          "query": {
            "type": "object",
            "description": "OpenSearch query DSL object"
          }
        },
        "required": ["index", "query"]
      }
    }
  }
]')

# ─── TOOL EXECUTION ───────────────────────────────────────────────────────────
execute_tool() {
  local name="$1"
  local args="$2"  # JSON string from the model

  case "$name" in
    search)
      local index body
      # Model may omit "index"; fall back to the configured default index.
      index=$(echo "$args" | jq -r --arg d "$OS_INDEX" '.index // $d')
      # Build a clean request body from model-provided fields.
      # Provide sensible defaults: sort by @timestamp desc (common in log indices),
      # empty aggs object (no aggregations unless model requested them).
      body=$(echo "$args" | jq '{
        query: .query,
        size:  (.size // 10),
        sort:  (.sort // [{"@timestamp":{"order":"desc"}}]),
        aggs:  (.aggs // {})
      }')
      # ?pretty makes the response human-readable in debug output.
      os_curl -X GET "${OS_HOST}/${index}/_search?pretty" \
        -H "Content-Type: application/json" \
        -d "$body"
      ;;

    list_indices)
      local pattern
      pattern=$(echo "$args" | jq -r '.pattern // "*"')
      # _cat/indices returns compact index metadata. We request JSON format
      # and limit to useful fields: name, doc count, size, health status.
      os_curl "${OS_HOST}/_cat/indices/${pattern}?h=index,docs.count,store.size,status&format=json"
      ;;

    get_mapping)
      local index
      index=$(echo "$args" | jq -r '.index')
      # Returns the full field mapping for the index. The model uses this to learn
      # field names and types before constructing queries (e.g. "message" is text,
      # "@timestamp" is date). Prevents type mismatch query errors.
      os_curl "${OS_HOST}/${index}/_mapping?pretty"
      ;;

    count)
      local index query
      index=$(echo "$args" | jq -r '.index')
      # Extract just the query DSL; _count doesn't accept size/sort/aggs.
      query=$(echo "$args" | jq '{query:.query}')
      os_curl -X GET "${OS_HOST}/${index}/_count" \
        -H "Content-Type: application/json" \
        -d "$query"
      ;;

    *)
      echo '{"error":"unknown tool"}'
      ;;
  esac
}

# ─── SYSTEM PROMPT ────────────────────────────────────────────────────────────
#
# The system prompt is injected as the first message with role "system".
# It sets the model's persona and operating constraints for this session.
# Key elements:
#   - Tells the model what kind of data store it's working with (OpenSearch)
#   - Names the default index so it doesn't have to discover it
#   - Gives query-building hints (range filters for time, aggs for counts)
#     to steer the model toward idiomatic OpenSearch queries
SYSTEM_PROMPT="You are a data analyst with access to an OpenSearch cluster. \
The default index is '${OS_INDEX}'. Use the search tool to answer questions about the data. \
When building queries, prefer range filters for time-based queries and use aggregations for counts/stats."

# ─── CONVERSATION STATE ───────────────────────────────────────────────────────
#
# Messages array carries the full conversation history on every API call.
# OpenSearch queries often require multiple steps (list indices → get mapping →
# search), so the model may call several tools before answering.
messages=$(jq -n \
  --arg sys "$SYSTEM_PROMPT" \
  --arg p "$PROMPT" \
  '[{"role":"system","content":$sys},{"role":"user","content":$p}]')

# ─── TOOL-CALL LOOP ───────────────────────────────────────────────────────────
round=0
while [[ $round -lt $MAX_TOOL_ROUNDS ]]; do

  # temperature:0.2 — more deterministic than default; helps with precise Query DSL
  # generation where small deviations can produce invalid JSON or wrong field names.
  response=$(curl -sf "${LLAMAFILE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(jq -n \
      --argjson messages "$messages" \
      --argjson tools "$TOOLS" \
      '{"model":"bonsai","messages":$messages,"tools":$tools,"max_tokens":2048,"temperature":0.2}')")

  finish_reason=$(echo "$response" | jq -r '.choices[0].finish_reason')
  assistant_msg=$(echo "$response" | jq '.choices[0].message')
  # Always append the assistant message — even tool_calls responses must be in history
  # so the model remembers what it requested when results come back.
  messages=$(echo "$messages" | jq --argjson msg "$assistant_msg" '. + [$msg]')

  if [[ "$finish_reason" == "tool_calls" ]]; then
    ((round++))
    while IFS= read -r tool_call; do
      tool_id=$(echo "$tool_call"   | jq -r '.id')
      tool_name=$(echo "$tool_call" | jq -r '.function.name')
      tool_args=$(echo "$tool_call" | jq -r '.function.arguments')

      # Tool activity to stderr; final answer to stdout. Lets callers do:
      #   answer=$(./bonsai-opensearch.sh ...) 2>debug.log
      echo "[tool] ${tool_name}: ${tool_args}" >&2
      tool_result=$(execute_tool "$tool_name" "$tool_args")
      # Truncate debug preview — full result still goes to the model.
      echo "[result] $(echo "$tool_result" | head -c 500)..." >&2

      # Append tool result with matching tool_call_id.
      # The model correlates results to requests via this ID, which is important
      # when multiple tools are called in a single round.
      messages=$(echo "$messages" | jq \
        --arg id "$tool_id" \
        --arg name "$tool_name" \
        --arg result "$tool_result" \
        '. + [{"role":"tool","tool_call_id":$id,"name":$name,"content":$result}]')
    done < <(echo "$assistant_msg" | jq -c '.tool_calls[]')
  else
    echo "$response" | jq -r '.choices[0].message.content'
    exit 0
  fi
done

echo "error: exceeded max tool rounds (${MAX_TOOL_ROUNDS})" >&2
exit 1
