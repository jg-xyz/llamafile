#!/bin/bash
# bonsai-postgres.sh — Query a PostgreSQL database using Bonsai llamafile
#
# HOW IT WORKS:
#   PostgreSQL has rich schema introspection via information_schema and pg_catalog.
#   This script teaches the model to use that introspection before querying, so it
#   knows the actual column names and types rather than guessing. Good schema context
#   is the single biggest factor in model query quality.
#
#   The flow:
#     1. We define tools as JSON schemas (what the model can call and with what args).
#     2. We send the user's prompt + tool definitions to the llamafile server.
#     3. The model either:
#        a) Returns a final answer  → print it and exit.
#        b) Returns tool_calls      → it wants to inspect schema or run queries.
#     4. For each tool call, we run psql and return the results as JSON.
#     5. We loop back so the model can reason over results and ask follow-up questions.
#
#   The MCP server for PostgreSQL (@modelcontextprotocol/server-postgres) exposes
#   these same operations over JSON-RPC/stdio. This script calls psql directly —
#   same semantics, no Node.js required.
#
#   Safety: execute_query wraps all SELECT statements in a read-only transaction
#   (BEGIN READ ONLY ... ROLLBACK). The database cannot be mutated even if the
#   model generates INSERT/UPDATE/DELETE — the transaction type blocks it.
#
# Usage:
#   ./bonsai-postgres.sh --conn <connection-string> --prompt "question"
#
# Example:
#   ./bonsai-postgres.sh \
#     --conn "postgresql://user:pass@localhost:5432/mydb" \
#     --prompt "which customers made more than 5 orders in the last 30 days?"
#
# Requirements: psql, curl, jq, python3, llamafile server running (or set LLAMAFILE_URL)

set -euo pipefail

# LLAMAFILE_URL: where the llamafile server is listening.
# Start the server with: ./bonsai.llamafile (opens :8080 by default)
LLAMAFILE_URL="${LLAMAFILE_URL:-http://localhost:8080}"
PG_CONN=""
PROMPT=""
MAX_TOOL_ROUNDS=10

# Default schema to inspect. Override via PG_SCHEMA env var or --schema flag.
# "public" is the PostgreSQL default; multi-tenant apps often use per-tenant schemas.
SCHEMA="${PG_SCHEMA:-public}"

usage() {
  grep '^#' "$0" | sed 's/^# \?//'
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --conn)          PG_CONN="$2";       shift 2 ;;
    --schema)        SCHEMA="$2";        shift 2 ;;
    --prompt)        PROMPT="$2";        shift 2 ;;
    --llamafile-url) LLAMAFILE_URL="$2"; shift 2 ;;
    -h|--help)       usage ;;
    *) echo "Unknown argument: $1" >&2; usage ;;
  esac
done

[[ -z "$PG_CONN" ]] && { echo "error: --conn required" >&2; exit 1; }
[[ -z "$PROMPT" ]]  && { echo "error: --prompt required" >&2; exit 1; }

command -v psql    >/dev/null || { echo "error: psql not found" >&2; exit 1; }
command -v jq      >/dev/null || { echo "error: jq not found" >&2; exit 1; }
command -v curl    >/dev/null || { echo "error: curl not found" >&2; exit 1; }
command -v python3 >/dev/null || { echo "error: python3 not found" >&2; exit 1; }

if ! curl -sf "${LLAMAFILE_URL}/health" >/dev/null 2>&1; then
  echo "error: llamafile server not reachable at ${LLAMAFILE_URL}" >&2
  echo "  Start it with: ./bonsai.llamafile" >&2
  exit 1
fi

# ─── DATABASE HELPER ──────────────────────────────────────────────────────────
#
# pg_query runs a SQL string via psql and converts the output to JSON.
# We use CSV output mode because:
#   - psql's native --json flag has inconsistent behavior with NULLs and arrays
#   - CSV is stable, predictable, and Python's csv.DictReader handles it cleanly
#   - The model receives consistent JSON regardless of column types
#
# psql flags:
#   --tuples-only  suppress column headers and row counts from psql's own output
#   --no-align     disable column padding (required for --csv)
#   --csv          output rows as comma-separated values with a header row
#   --command      run this SQL string and exit (non-interactive mode)
pg_query() {
  local sql="$1"
  psql "$PG_CONN" \
    --tuples-only \
    --no-align \
    --csv \
    --command="$sql" 2>&1 | \
    python3 -c "
import sys, csv, json
reader = csv.DictReader(sys.stdin)
print(json.dumps(list(reader), indent=2))
" 2>/dev/null || echo '{"error":"query failed"}'
}

# ─── TOOL DEFINITIONS ─────────────────────────────────────────────────────────
#
# We use `jq -n` to build the JSON so we can interpolate the $SCHEMA shell variable
# into the "default" field of schema-scoped tools.
#
# Tool set mirrors the PostgreSQL MCP server's interface:
#   list_tables    — discover what tables exist before writing queries
#   describe_table — get columns, types, nullability, defaults, constraints
#   get_indexes    — understand what's indexed (guides efficient query design)
#   execute_query  — run a read-only SELECT (safeguarded against mutation)
#   explain_query  — get the query plan (useful for verifying the model's SQL)
#
# Ordering matters in the description: the system prompt tells the model to
# inspect schema first, so list_tables/describe_table are listed first.
TOOLS=$(jq -n --arg schema "$SCHEMA" '[
  {
    "type": "function",
    "function": {
      "name": "list_tables",
      "description": "List all tables and views in the database schema",
      "parameters": {
        "type": "object",
        "properties": {
          "schema": {
            "type": "string",
            "description": "Schema name to list tables from",
            "default": $schema
          }
        }
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "describe_table",
      "description": "Get columns, types, and constraints for a specific table",
      "parameters": {
        "type": "object",
        "properties": {
          "table": {
            "type": "string",
            "description": "Table name"
          },
          "schema": {
            "type": "string",
            "description": "Schema name",
            "default": $schema
          }
        },
        "required": ["table"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_indexes",
      "description": "Get index definitions for a table (useful for understanding query performance)",
      "parameters": {
        "type": "object",
        "properties": {
          "table": {
            "type": "string",
            "description": "Table name"
          }
        },
        "required": ["table"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "execute_query",
      "description": "Execute a read-only SQL query (SELECT only). For safety, all queries run in a read-only transaction.",
      "parameters": {
        "type": "object",
        "properties": {
          "sql": {
            "type": "string",
            "description": "SELECT statement to execute. No INSERT/UPDATE/DELETE/DROP."
          },
          "limit": {
            "type": "integer",
            "description": "Automatically appended LIMIT if not already present (default: 100)"
          }
        },
        "required": ["sql"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "explain_query",
      "description": "Get the query execution plan for a SQL query to understand performance",
      "parameters": {
        "type": "object",
        "properties": {
          "sql": {
            "type": "string",
            "description": "SQL query to explain"
          }
        },
        "required": ["sql"]
      }
    }
  }
]')

# ─── TOOL EXECUTION ───────────────────────────────────────────────────────────
execute_tool() {
  local name="$1"
  local args="$2"  # JSON string from the model

  case "$name" in
    list_tables)
      local schema
      schema=$(echo "$args" | jq -r --arg s "$SCHEMA" '.schema // $s')
      # information_schema.tables is the portable, standard way to list tables.
      # pg_total_relation_size includes indexes and toast tables in the size figure,
      # giving a complete picture of how much space each table uses.
      pg_query "
        SELECT
          table_name,
          table_type,
          pg_size_pretty(pg_total_relation_size(
            quote_ident(table_schema)||'.'||quote_ident(table_name)
          )) AS size
        FROM information_schema.tables
        WHERE table_schema = '${schema}'
        ORDER BY table_name
      "
      ;;

    describe_table)
      local table schema
      table=$(echo "$args"  | jq -r '.table')
      schema=$(echo "$args" | jq -r --arg s "$SCHEMA" '.schema // $s')
      # Join information_schema.columns with table_constraints + key_column_usage
      # to surface PRIMARY KEY / FOREIGN KEY / UNIQUE constraints alongside each column.
      # COALESCE converts NULL (no constraint) to empty string for cleaner model output.
      pg_query "
        SELECT
          c.column_name,
          c.data_type,
          c.character_maximum_length,
          c.is_nullable,
          c.column_default,
          COALESCE(
            (SELECT string_agg(tc.constraint_type, ', ')
             FROM information_schema.table_constraints tc
             JOIN information_schema.key_column_usage kcu
               ON tc.constraint_name = kcu.constraint_name
               AND tc.table_schema   = kcu.table_schema
             WHERE kcu.table_name   = c.table_name
               AND kcu.column_name  = c.column_name
               AND tc.table_schema  = c.table_schema),
            ''
          ) AS constraints
        FROM information_schema.columns c
        WHERE c.table_schema = '${schema}'
          AND c.table_name   = '${table}'
        ORDER BY c.ordinal_position
      "
      ;;

    get_indexes)
      local table
      table=$(echo "$args" | jq -r '.table')
      # pg_indexes exposes the CREATE INDEX DDL for each index, which shows
      # exactly which columns are indexed and in what order. The model uses this
      # to write queries that can use existing indexes rather than forcing seq scans.
      pg_query "
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE tablename = '${table}'
        ORDER BY indexname
      "
      ;;

    execute_query)
      local sql limit
      sql=$(echo "$args"   | jq -r '.sql')
      limit=$(echo "$args" | jq -r '.limit // 100')

      # Strip trailing semicolon — we're about to embed this in a larger statement.
      sql=$(echo "$sql" | sed 's/[[:space:]]*;[[:space:]]*$//')

      # Reject anything that isn't a SELECT or a CTE (WITH ... SELECT).
      # This is a belt-and-suspenders check; BEGIN READ ONLY already prevents
      # mutations at the database level, but this catches mistakes earlier
      # and returns a clear error message to the model so it can self-correct.
      if ! echo "$sql" | grep -iqE '^\s*(SELECT|WITH)'; then
        echo '{"error":"only SELECT queries are permitted"}'
        return
      fi

      # Auto-append LIMIT if the model forgot to include one.
      # Prevents accidentally fetching millions of rows on large tables.
      if ! echo "$sql" | grep -iqE 'LIMIT\s+[0-9]'; then
        sql="${sql} LIMIT ${limit}"
      fi

      # BEGIN READ ONLY starts a transaction that PostgreSQL itself enforces as
      # read-only at the engine level — no INSERT/UPDATE/DELETE/DDL can succeed
      # regardless of what SQL the model generates. ROLLBACK ensures clean state.
      pg_query "BEGIN READ ONLY; ${sql}; ROLLBACK"
      ;;

    explain_query)
      local sql
      sql=$(echo "$args" | jq -r '.sql')
      # FORMAT JSON returns a structured plan rather than text, which the model
      # can parse more reliably. ANALYZE false means we don't actually execute
      # the query — we get the estimated plan with no side effects or wait time.
      pg_query "EXPLAIN (FORMAT JSON, ANALYZE false) ${sql}" | \
        python3 -c "
import sys, json
data = json.load(sys.stdin)
# psql wraps EXPLAIN JSON output in an extra array layer; unwrap it.
print(json.dumps(data[0] if data else {}, indent=2))
"
      ;;

    *)
      echo '{"error":"unknown tool"}'
      ;;
  esac
}

# ─── SYSTEM PROMPT ────────────────────────────────────────────────────────────
#
# Instructing the model to inspect schema before querying is critical.
# Without this, it tends to guess column names and produce queries that fail.
# "Never mutate data" reinforces the technical safety guarantee from BEGIN READ ONLY
# at the model-reasoning level so it doesn't even attempt writes.
SYSTEM_PROMPT="You are a PostgreSQL expert and data analyst. \
Schema in use: '${SCHEMA}'. \
Always inspect the schema before writing queries. \
Use explain_query to verify query plans when performance matters. \
Only generate SELECT queries — never mutate data."

# ─── CONVERSATION STATE ───────────────────────────────────────────────────────
#
# Conversation begins with system context + user prompt.
# A typical multi-step session looks like:
#   user → list_tables → describe_table(orders) → describe_table(customers)
#   → execute_query(JOIN ...) → assistant final answer
# Each step's results are in the history when the model decides the next step.
messages=$(jq -n \
  --arg sys "$SYSTEM_PROMPT" \
  --arg p "$PROMPT" \
  '[{"role":"system","content":$sys},{"role":"user","content":$p}]')

# ─── TOOL-CALL LOOP ───────────────────────────────────────────────────────────
round=0
while [[ $round -lt $MAX_TOOL_ROUNDS ]]; do

  # temperature:0.2 — low randomness improves SQL precision. Higher values cause
  # the model to vary column names or SQL syntax in ways that break queries.
  response=$(curl -sf "${LLAMAFILE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(jq -n \
      --argjson messages "$messages" \
      --argjson tools "$TOOLS" \
      '{"model":"bonsai","messages":$messages,"tools":$tools,"max_tokens":2048,"temperature":0.2}')")

  finish_reason=$(echo "$response" | jq -r '.choices[0].finish_reason')
  assistant_msg=$(echo "$response" | jq '.choices[0].message')
  messages=$(echo "$messages" | jq --argjson msg "$assistant_msg" '. + [$msg]')

  if [[ "$finish_reason" == "tool_calls" ]]; then
    ((round++))
    while IFS= read -r tool_call; do
      tool_id=$(echo "$tool_call"   | jq -r '.id')
      tool_name=$(echo "$tool_call" | jq -r '.function.name')
      tool_args=$(echo "$tool_call" | jq -r '.function.arguments')

      echo "[tool] ${tool_name}: ${tool_args}" >&2
      tool_result=$(execute_tool "$tool_name" "$tool_args")
      # Preview first 500 chars in debug output; full result goes to the model.
      echo "[result] $(echo "$tool_result" | head -c 500)" >&2

      messages=$(echo "$messages" | jq \
        --arg id "$tool_id" \
        --arg name "$tool_name" \
        --arg result "$tool_result" \
        '. + [{"role":"tool","tool_call_id":$id,"name":$name,"content":$result}]')
    done < <(echo "$assistant_msg" | jq -c '.tool_calls[]')
  else
    # Model produced its final answer — print to stdout and exit cleanly.
    echo "$response" | jq -r '.choices[0].message.content'
    exit 0
  fi
done

echo "error: exceeded max tool rounds (${MAX_TOOL_ROUNDS})" >&2
exit 1
