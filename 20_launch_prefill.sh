#!/bin/bash
# HOST-side launcher: the PREFILL instance, TP=16 / EP=16 across B300-1 + B300-2.
# Same command on both hosts except NODE_RANK:
#
#   B300-1:  NODE_RANK=0 bash 20_launch_prefill.sh
#   B300-2:  NODE_RANK=1 bash 20_launch_prefill.sh
#
# This is the vLLM recipe's prefill unchanged apart from EFA plumbing: TP16 with
# --enable-expert-parallel and NO a2a backend. At DP=1 vLLM never runs an a2a
# anyway (the MoE output is all-reduced), so prefill is the same in every decode
# arm and only the decode side varies.
#
# Only node 0 serves HTTP (:$PREFILL_PORT); node 1 runs --headless workers.
set -euo pipefail
cd "$(dirname "$0")"
source ./env_common.sh

NAME="${NAME:-k3v-prefill}"
remove_container "$NAME"
require_image
build_cache_args
build_gdr_args
build_common_envs
build_kv_envs

ARGS=(
    vllm serve "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --trust-remote-code
    --gpu-memory-utilization "$PREFILL_GPU_UTIL"
    --port "$PREFILL_PORT"
    --kv-transfer-config "$(kv_config kv_producer)"
    --tensor-parallel-size 16
    --enable-expert-parallel
    --nnodes "$NNODES" --node-rank "$NODE_RANK"
    --master-addr "$PREFILL_HEAD" --master-port "$PREFILL_MASTER_PORT"
    --max-num-batched-tokens "$PREFILL_MAX_BATCHED"
    --no-disable-hybrid-kv-cache-manager
    --no-enable-flashinfer-autotune
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
)
(( NODE_RANK > 0 )) && ARGS+=(--headless)
TP_ENVS=()
if (( NNODES > 1 )); then ARGS+=("${MULTINODE_TP_ARGS[@]}"); TP_ENVS+=("${MULTINODE_TP_ENVS[@]}"); fi   # see env_common.sh
# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS:-}" ]] && ARGS+=($EXTRA_ARGS)

docker run -d --name "$NAME" \
    --init --gpus all --privileged \
    --net=host --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --device=/dev/infiniband ${GDR_ARGS[@]+"${GDR_ARGS[@]}"} \
    --shm-size=600g \
    -v "$HOST_MODEL_DIR/Kimi-K3:$MODEL_PATH:ro" \
    "${CACHE_ARGS[@]}" \
    "${COMMON_ENVS[@]}" "${KV_ENVS[@]}" ${TP_ENVS[@]+"${TP_ENVS[@]}"} \
    -e VLLM_NIXL_SIDE_CHANNEL_HOST="$PREFILL_HEAD" \
    -e VLLM_NIXL_SIDE_CHANNEL_PORT="$PREFILL_NIXL_PORT" \
    -e VLLM_MOONCAKE_BOOTSTRAP_PORT="$MOONCAKE_BOOTSTRAP_PORT" \
    -e VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT=1 \
    -e K3V_ROLE=prefill -e K3V_NODE_RANK="$NODE_RANK" -e K3V_KV_CONNECTOR="$KV_CONNECTOR" \
    "$IMAGE" "${ARGS[@]}"

echo "launched '$NAME'  node-rank=$NODE_RANK/$NNODES  tp=16 ep=16  head=$PREFILL_HEAD:$PREFILL_PORT  ->  docker logs -f $NAME"
(( NODE_RANK > 0 )) && echo "  (headless: never binds :$PREFILL_PORT)"
exit 0
