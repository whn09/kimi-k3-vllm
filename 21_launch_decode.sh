#!/bin/bash
# HOST-side launcher: the DECODE instance across B300-3 + B300-4, in one of three
# arms (DECODE_MODE, see env_common.sh). Same command on both hosts except
# NODE_RANK:
#
#   B300-3:  DECODE_MODE=deepep_v2 NODE_RANK=0 bash 21_launch_decode.sh
#   B300-4:  DECODE_MODE=deepep_v2 NODE_RANK=1 bash 21_launch_decode.sh
#
#   deepep_v2 / agrs : TP1 x DP16 (--data-parallel-hybrid-lb). EACH node serves
#                      HTTP on :$DECODE_PORT for its 8 local DP ranks, so the
#                      router gets both nodes as --decode.
#   tp               : TP16 x DP1. Only node 0 serves HTTP; node 1 is headless,
#                      so the router gets only the decode head.
set -euo pipefail
cd "$(dirname "$0")"
source ./env_common.sh

NAME="${NAME:-k3v-decode}"
remove_container "$NAME"
require_image
build_cache_args
build_gdr_args
build_common_envs
build_kv_envs

MY_IP=$(ip route get 1.1.1.1 | sed -n 's/.* src \([0-9.]*\).*/\1/p')

ARGS=(
    vllm serve "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --trust-remote-code
    --gpu-memory-utilization "$DECODE_GPU_UTIL"
    --port "$DECODE_PORT"
    --kv-transfer-config "$(kv_config kv_consumer)"
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --enable-expert-parallel
    --no-enable-prefix-caching
    --no-disable-hybrid-kv-cache-manager
    --enable-auto-tool-choice
    --tool-call-parser kimi_k3
    --reasoning-parser kimi_k3
)
EXTRA_ENVS=()

case "$DECODE_MODE" in
    deepep_v2|agrs)
        if [[ "$DECODE_MODE" == deepep_v2 ]]; then
            A2A=deepep_v2
            build_deepep_envs
            EXTRA_ENVS+=("${DEEPEP_ENVS[@]}")
        else
            A2A=allgather_reducescatter
        fi
        ARGS+=(
            --tensor-parallel-size 1
            --data-parallel-size 16
            --data-parallel-size-local 8
            --data-parallel-start-rank $(( NODE_RANK * 8 ))
            --data-parallel-address "$DECODE_HEAD"
            --data-parallel-rpc-port "$DECODE_DP_RPC_PORT"
            --data-parallel-hybrid-lb
            --all2all-backend "$A2A"
            --max-num-seqs "$DECODE_SEQS_PER_RANK"
            --max-num-batched-tokens "${DECODE_MAX_BATCHED:-$(( DECODE_SEQS_PER_RANK * 4 ))}"
        )
        # Each node's API server owns its own DP ranks' NIXL listeners.
        NIXL_HOST="$MY_IP"
        ;;
    deepep_v2_1n_tp*)
        # One self-contained instance per decode node, EP8 over this node's 8
        # GPUs, its own API server, no peer node: the a2a never leaves NVLink.
        # Attention is TP${N} x DP(8/N). TP1 x DP8 does NOT fit: TP1 replicates
        # K3's ~93 GB of non-expert weights on every GPU and EP8 holds 1/8 of the
        # experts (~192 GB) -> ~285 GB > 267.7 GiB, OOM in weight load
        # (2026-10-15). TP2 -> ~238 GB, TP4 -> ~215 GB.
        # Admission is kept at the EP16 rows' total: 2 instances x 512 = 1024,
        # i.e. DECODE_SEQS_PER_RANK x TP per DP rank.
        T1N="${DECODE_MODE#deepep_v2_1n_tp}"
        D1N=$(( 8 / T1N ))
        build_deepep_1n_envs
        EXTRA_ENVS+=("${DEEPEP_ENVS[@]}")
        ARGS+=(
            --tensor-parallel-size "$T1N"
            --data-parallel-size "$D1N"
            --data-parallel-size-local "$D1N"
            --data-parallel-address "$MY_IP"
            --data-parallel-rpc-port "$DECODE_DP_RPC_PORT"
            --all2all-backend deepep_v2
            --max-num-seqs $(( DECODE_SEQS_PER_RANK * T1N ))
            --max-num-batched-tokens "${DECODE_MAX_BATCHED:-$(( DECODE_SEQS_PER_RANK * T1N * 4 ))}"
        )
        NIXL_HOST="$MY_IP"
        ;;
    tp)
        ARGS+=(
            --tensor-parallel-size 16
            --nnodes "$NNODES" --node-rank "$NODE_RANK"
            --master-addr "$DECODE_HEAD" --master-port "$DECODE_MASTER_PORT"
            --max-num-seqs "$DECODE_TOTAL_SEQS"
            --max-num-batched-tokens "${DECODE_MAX_BATCHED:-$(( DECODE_TOTAL_SEQS * 4 ))}"
        )
        (( NODE_RANK > 0 )) && ARGS+=(--headless)
        if (( NNODES > 1 )); then ARGS+=("${MULTINODE_TP_ARGS[@]}"); EXTRA_ENVS+=("${MULTINODE_TP_ENVS[@]}"); fi   # see env_common.sh
        NIXL_HOST="$DECODE_HEAD"
        ;;
    *)
        echo "ERROR: DECODE_MODE must be deepep_v2 | deepep_v2_1n_tp{2,4} | agrs | tp, got '$DECODE_MODE'" >&2
        exit 1 ;;
esac
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
    "${COMMON_ENVS[@]}" "${KV_ENVS[@]}" \
    ${EXTRA_ENVS[@]+"${EXTRA_ENVS[@]}"} \
    -e VLLM_NIXL_SIDE_CHANNEL_HOST="$NIXL_HOST" \
    -e VLLM_NIXL_SIDE_CHANNEL_PORT="$DECODE_NIXL_PORT" \
    -e K3V_ROLE=decode -e K3V_DECODE_MODE="$DECODE_MODE" -e K3V_NODE_RANK="$NODE_RANK" -e K3V_KV_CONNECTOR="$KV_CONNECTOR" \
    -e K3V_DECODE_TOTAL_SEQS="$DECODE_TOTAL_SEQS" \
    "$IMAGE" "${ARGS[@]}"

echo "launched '$NAME'  mode=$DECODE_MODE  node-rank=$NODE_RANK/$NNODES  total-seqs=$DECODE_TOTAL_SEQS  ->  docker logs -f $NAME"
[[ "$DECODE_MODE" == deepep_v2_1n_tp* ]] && echo "  1-node instance (TP${T1N} x DP${D1N}, EP8): gin=3 hca=${IB_HCA:-$(pick_ib_hca)} direct, deepep-sms=$DEEPEP_NUM_SMS(0=auto)"
[[ "$DECODE_MODE" == deepep_v2 ]] && echo "  gin=$GIN_TYPE hca=${EFA_HCA:-<none>} deepep-sms=$DEEPEP_NUM_SMS"
exit 0
