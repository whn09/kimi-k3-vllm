#!/bin/bash
# Correctness gate through the router: one greedy completion, one chat, one
# stream. Run anywhere that can reach the router (default: on B300-1).
set -euo pipefail
cd "$(dirname "$0")"
source ./env_common.sh
URL="${URL:-http://${PREFILL_HEAD}:${ROUTER_PORT}}"

echo "== completion (greedy)"
curl -sS -m 600 "$URL/v1/completions" -H 'content-type: application/json' -d "{
  \"model\": \"$SERVED_MODEL_NAME\", \"prompt\": \"The capital of France is\",
  \"max_tokens\": 32, \"temperature\": 0}" | python3 -c 'import json,sys; r=json.load(sys.stdin); print(repr(r["choices"][0]["text"])); print(r["usage"])'

echo "== chat"
curl -sS -m 600 "$URL/v1/chat/completions" -H 'content-type: application/json' -d "{
  \"model\": \"$SERVED_MODEL_NAME\",
  \"messages\": [{\"role\": \"user\", \"content\": \"What is 17 * 23? Answer with the number only.\"}],
  \"max_tokens\": 512, \"temperature\": 0}" | python3 -c 'import json,sys; r=json.load(sys.stdin); m=r["choices"][0]["message"]; print("content:", repr(m.get("content"))); print("reasoning chars:", len(m.get("reasoning_content") or m.get("reasoning") or "")); print(r["usage"])'

echo "== stream"
curl -sS -N -m 600 "$URL/v1/completions" -H 'content-type: application/json' -d "{
  \"model\": \"$SERVED_MODEL_NAME\", \"prompt\": \"Count from 1 to 10:\",
  \"max_tokens\": 40, \"temperature\": 0, \"stream\": true}" | grep -c '^data: {' | sed 's/^/stream chunks: /'
