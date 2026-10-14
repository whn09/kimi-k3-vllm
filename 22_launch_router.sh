#!/bin/bash
# HOST-side launcher: vllm-router in PD mode. Run on B300-1.
#
#   DECODE_MODE=deepep_v2 bash 22_launch_router.sh   # --decode = both decode nodes
#   DECODE_MODE=tp        bash 22_launch_router.sh   # --decode = decode head only
#
# Clients then talk to http://<B300-1>:$ROUTER_PORT/v1/... only.
set -euo pipefail
cd "$(dirname "$0")"
source ./env_common.sh

NAME="${NAME:-k3v-router}"
docker rm -f "$NAME" >/dev/null 2>&1 || true

if [[ "$DECODE_MODE" == tp ]]; then
    DECODES="$DECODE_HEAD"          # the headless node serves nothing
else
    DECODES="$DECODE_IPS"           # hybrid-lb: one API server per node
fi

# NIXL: vllm-router (Rust). Mooncake: vLLM's own example proxy, because the
# MooncakeConnector needs the PROXY to mint transfer_id and hand decode the
# prefill's bootstrap address -- vllm-router's PD mode speaks the NIXL protocol
# only. That makes the router a confound BETWEEN connectors, never between
# decode arms of the same connector.
if [[ "$KV_CONNECTOR" == mooncake ]]; then
    ROUTE_ARGS=(--prefill "http://${PREFILL_HEAD}:${PREFILL_PORT}" "$MOONCAKE_BOOTSTRAP_PORT")
else
    ROUTE_ARGS=(--prefill "http://${PREFILL_HEAD}:${PREFILL_PORT}")
fi
for ip in $DECODES; do ROUTE_ARGS+=(--decode "http://${ip}:${DECODE_PORT}"); done

# A node that is not up yet fails registration; probe first so the symptom is
# not just a router that vanished.
for url in "${PREFILL_HEAD}:${PREFILL_PORT}" $(for ip in $DECODES; do echo "${ip}:${DECODE_PORT}"; done); do
    curl -sf -m 5 "http://${url}/health" >/dev/null \
      || { echo "FATAL: http://${url}/health not ready -- start 20_/21_ and wait for 'Application startup complete'"; exit 1; }
done

echo "router : 0.0.0.0:${ROUTER_PORT}  kv=$KV_CONNECTOR decode-mode=$DECODE_MODE"
printf '  %s\n' "${ROUTE_ARGS[*]}"

if [[ "$KV_CONNECTOR" == mooncake ]]; then
    docker run -d --name "$NAME" --net=host --entrypoint python3 "$IMAGE" \
        /vllm-workspace/examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py \
        "${ROUTE_ARGS[@]}" --host 0.0.0.0 --port "$ROUTER_PORT"
else
    docker run -d --name "$NAME" --net=host \
        -v "$HOST_MODEL_DIR/Kimi-K3:$MODEL_PATH:ro" \
        "$IMAGE" \
        vllm-router --policy "${ROUTER_POLICY:-round_robin}" \
            --vllm-pd-disaggregation \
            "${ROUTE_ARGS[@]}" \
            --host 0.0.0.0 --port "$ROUTER_PORT" \
            --intra-node-data-parallel-size 1
fi

echo "launched '$NAME'  ->  docker logs -f $NAME"
