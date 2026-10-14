#!/bin/bash
# Shared config for the Kimi-K3 vLLM PD kit. Sourced by the host launchers and
# by the laptop-side drivers, so keep it side-effect free.

# ---- paths (host) ----
NVME="${NVME:-/opt/dlami/nvme}"
HOST_MODEL_DIR="${HOST_MODEL_DIR:-$NVME/models}"
HOST_CACHE_DIR="${HOST_CACHE_DIR:-$NVME/cache-vllm}"
SCRIPT_DIR_HOST="${SCRIPT_DIR_HOST:-/home/ubuntu/kimi-k3-vllm}"
MODEL_PATH="${MODEL_PATH:-/models/Kimi-K3}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-moonshotai/Kimi-K3}"

# ---- image ----
# Built by 01_build_push.sh on B300-1, pushed to ECR, pulled on the others.
# The tag carries every pin (vllm sha, DeepEP sha, NCCL, EFA) so two images can
# never be confused in a comparison.
# Not AWS_REGION: a laptop-side AWS_REGION would silently retarget the registry.
K3V_REGION="${K3V_REGION:-ap-south-2}"
ECR_REGISTRY="${ECR_REGISTRY:-579019700964.dkr.ecr.${K3V_REGION}.amazonaws.com}"
ECR_REPO="${ECR_REPO:-kimi-k3-vllm-b300}"
IMAGE_TAG="${IMAGE_TAG:-vllm77154ca-deepep6de427b-nccl2312-efa1.50.0}"
IMAGE="${IMAGE:-${ECR_REGISTRY}/${ECR_REPO}:${IMAGE_TAG}}"

# ---- JIT / autotune caches persisted on the host ----
# "container path=host subdir"
CACHE_MOUNTS=(
    "/root/.cache/vllm=vllm"                # torch.compile + cudagraph artefacts
    "/root/.cache/flashinfer=flashinfer"
    "/root/.cache/deep_gemm=deep_gemm"
    "/root/.cache/tvm-ffi=tvm-ffi"
    "/root/.triton=triton"
    "/root/.nv/ComputeCache=nv_compute"
    "/root/.deep_ep=deep_ep"                # DeepEP v2 JIT (EP_JIT_CACHE_DIR)
)
build_cache_args() {
    local entry cpath hsub
    CACHE_ARGS=()
    for entry in "${CACHE_MOUNTS[@]}"; do
        cpath="${entry%%=*}"; hsub="${entry#*=}"
        mkdir -p "$HOST_CACHE_DIR/$hsub"
        CACHE_ARGS+=(-v "$HOST_CACHE_DIR/$hsub:$cpath")
    done
}

# GIN type 5 has the GPU ring the NIC doorbell over MMIO, which needs
# --privileged AND /dev/gdrdrv; without them the GIN plugin loads and quietly
# gives up, and DeepEP aborts at nccl.cu:101 blaming "the network".
build_gdr_args() {
    GDR_ARGS=()
    if [[ -c /dev/gdrdrv ]]; then
        GDR_ARGS+=(--device=/dev/gdrdrv)
    else
        echo "WARN: /dev/gdrdrv missing -- GIN type 5 will not initialise." >&2
        echo "      sudo mknod /dev/gdrdrv c \$(awk '/gdrdrv/{print \$1}' /proc/devices) 0 && sudo chmod 666 /dev/gdrdrv" >&2
    fi
}

# ---- cluster ----
# PRIMARY (enp71s0) addresses. `hostname -I` lists all 18 interfaces on a b300
# and its first entry is not reliably the ENA one; refresh from
# `ssh B300-N ip route get 1.1.1.1` after every stop/start.
PRIMARY_IFACE="${PRIMARY_IFACE:-enp71s0}"
B300_1_IP="${B300_1_IP:-172.31.10.219}"   # 2026-10-14, ap-south-2
B300_2_IP="${B300_2_IP:-172.31.7.243}"
B300_3_IP="${B300_3_IP:-172.31.0.212}"
B300_4_IP="${B300_4_IP:-172.31.9.249}"

ip_of_host() {
    case "$1" in
        B300-1) echo "$B300_1_IP" ;;
        B300-2) echo "$B300_2_IP" ;;
        B300-3) echo "$B300_3_IP" ;;
        B300-4) echo "$B300_4_IP" ;;
        *) echo "ERROR: no IP known for host alias '$1'" >&2; return 1 ;;
    esac
}

# 2P2D: prefill is ONE TP=16 instance on B300-1+2, decode is ONE 16-GPU instance
# on B300-3+4. *_HEAD is the rank-0 node of each side.
PREFILL_HEAD="${PREFILL_HEAD:-$B300_1_IP}"
DECODE_HEAD="${DECODE_HEAD:-$B300_3_IP}"
DECODE_IPS="${DECODE_IPS:-$B300_3_IP $B300_4_IP}"

PREFILL_PORT="${PREFILL_PORT:-8001}"
DECODE_PORT="${DECODE_PORT:-8002}"
ROUTER_PORT="${ROUTER_PORT:-30000}"
PREFILL_MASTER_PORT="${PREFILL_MASTER_PORT:-29501}"
DECODE_MASTER_PORT="${DECODE_MASTER_PORT:-29502}"
DECODE_DP_RPC_PORT="${DECODE_DP_RPC_PORT:-13345}"
# NIXL handshake listener. The prefill SCHEDULER (node 0) serves the metadata of
# all 16 TP ranks on one port, so the host is the prefill HEAD on both prefill
# nodes. Under DP each rank listens on PORT + dp_index.
PREFILL_NIXL_PORT="${PREFILL_NIXL_PORT:-5557}"
DECODE_NIXL_PORT="${DECODE_NIXL_PORT:-5600}"
# Mooncake bootstrap server, served by the prefill head (VLLM_MOONCAKE_BOOTSTRAP_PORT).
MOONCAKE_BOOTSTRAP_PORT="${MOONCAKE_BOOTSTRAP_PORT:-8998}"

NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:-0}"

# ---- decode arm: THE axis this kit exists to measure ----
#   deepep_v2  TP1 x DP16, EP16, --all2all-backend deepep_v2 (hybrid: NVLink +
#              EFA GIN type 5). The recipe's decode shape plus DeepEP v2.
#   agrs       TP1 x DP16, EP16, --all2all-backend allgather_reducescatter
#              (vLLM's default a2a: all-gather tokens over NCCL, reduce-scatter
#              back). Same shape as deepep_v2 -- only the a2a differs.
#   tp         TP16 x DP1, EP16. vLLM never runs an a2a at DP=1 (see
#              ParallelConfig.use_sequence_parallel_moe): every rank sees every
#              token and the MoE output is all-reduced over 16 ranks.
DECODE_MODE="${DECODE_MODE:-deepep_v2}"

# Matched admission. The DP arms admit DECODE_SEQS_PER_RANK per DP rank, i.e.
# 16x that in total; the TP arm is one engine, so it gets the same TOTAL.
# Without this the TP arm is capped at 1/16 of the concurrency and the bench
# measures the admission wall instead of the MoE path.
DECODE_SEQS_PER_RANK="${DECODE_SEQS_PER_RANK:-8}"
DECODE_TOTAL_SEQS=$(( DECODE_SEQS_PER_RANK * 16 ))

# ---- per-side serving knobs (recipe defaults) ----
PREFILL_GPU_UTIL="${PREFILL_GPU_UTIL:-0.95}"
DECODE_GPU_UTIL="${DECODE_GPU_UTIL:-0.95}"
PREFILL_MAX_BATCHED="${PREFILL_MAX_BATCHED:-16384}"

# ---- NCCL / GIN ----
# Type 5 (EFA-GDA) needs NCCL >= 2.31 (image), NCCL_SYM_GIN_KERNELS_ENABLE=0
# (else nccl.cu:108 "GIN strong signals are required"), and NCCL_IB_HCA=rdmap
# because b300 lists 2 ConnectX-7 `ibp*` next to the 16 EFA `rdmap*` rails --
# `rdmap` is a prefix and covers all 16.
GIN_TYPE="${GIN_TYPE:-5}"
pick_efa_hca() {
    local devs=""
    [[ -d /sys/class/infiniband ]] && devs="$(ls /sys/class/infiniband 2>/dev/null)"
    grep -q '^rdmap' <<<"$devs" || return 0
    grep -qv '^rdmap' <<<"$devs" || return 0
    echo rdmap
}
EFA_HCA="${EFA_HCA-$(pick_efa_hca)}"

# docker -e flags common to both sides.
build_common_envs() {
    COMMON_ENVS=(
        -e PYTHONUNBUFFERED=1
        -e FI_PROVIDER=efa
        -e FI_EFA_USE_DEVICE_RDMA=1
        -e NCCL_SOCKET_IFNAME="$PRIMARY_IFACE"
        -e GLOO_SOCKET_IFNAME="$PRIMARY_IFACE"
        -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
        -e VLLM_SSM_CONV_STATE_LAYOUT=DS
        # 1.5 TB of weights + a cold JIT outlast the 600 s default.
        -e VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
    )
    [[ -n "${NCCL_DEBUG_SUBSYS:-}" ]] && COMMON_ENVS+=(-e NCCL_DEBUG_SUBSYS="$NCCL_DEBUG_SUBSYS")
    return 0
}

# Extra -e flags for DeepEP v2 (decode deepep_v2 arm only).
build_deepep_envs() {
    DEEPEP_ENVS=(
        -e NCCL_GIN_TYPE="$GIN_TYPE"
        -e NCCL_SYM_GIN_KERNELS_ENABLE=0
        -e VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE=1
    )
    [[ -n "$EFA_HCA" ]] && DEEPEP_ENVS+=(-e NCCL_IB_HCA="$EFA_HCA")
    return 0
}

# ---- KV transfer ----
#   nixl      NixlConnector, LIBFABRIC backend (EFA has no UCX `rc`). The recipe's
#             connector, and the one whose P TP16 -> D TP1 path the recipe uses.
#   mooncake  MooncakeConnector over the EFA wheel, protocol=efa. vLLM's Mooncake
#             Mamba path is only validated on GDN, not K3's KDA.
KV_CONNECTOR="${KV_CONNECTOR:-nixl}"
kv_config() {   # kv_config <kv_producer|kv_consumer>
    case "$KV_CONNECTOR" in
        nixl)
            printf '{"kv_connector":"NixlConnector","kv_role":"%s","kv_load_failure_policy":"fail","kv_connector_extra_config":{"backends":["%s"]}}' \
                "$1" "${NIXL_BACKEND:-LIBFABRIC}" ;;
        mooncake)
            printf '{"kv_connector":"MooncakeConnector","kv_role":"%s","kv_load_failure_policy":"fail","kv_connector_extra_config":{"mooncake_protocol":"efa"}}' \
                "$1" ;;
        *) echo "ERROR: KV_CONNECTOR must be nixl | mooncake" >&2; return 1 ;;
    esac
}
# Mooncake MR registration serialises on the EFA provider's per-domain lock;
# unbounded fan-out cost 138 s of K3 startup on sglang, 8 threads 20.7 s.
# expandable_segments remaps VA under a registered MR => every fi_read EINVAL,
# so it must stay off under PD.
build_kv_envs() {
    KV_ENVS=(-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False)
    [[ "$KV_CONNECTOR" == mooncake ]] && KV_ENVS+=(-e MC_MAX_CONCURRENT_REG_MR="${MC_MAX_CONCURRENT_REG_MR:-8}")
    return 0
}

# `docker rm -f` returns before the workers release the GPUs; an immediate
# relaunch then OOMs during weight load. Remove, then wait for the GPUs to drain.
remove_container() {
    local name="$1" _
    for _ in $(seq 1 60); do
        docker inspect "$name" >/dev/null 2>&1 || break
        docker rm -f "$name" >/dev/null 2>&1 || true
        sleep 2
    done
    if docker inspect "$name" >/dev/null 2>&1; then
        echo "ERROR: could not remove container '$name'" >&2; exit 1
    fi
    for _ in $(seq 1 60); do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
        (( used < 2000 )) && return 0
        sleep 2
    done
    echo "WARN: a GPU still has ${used} MiB in use -- another tenant? check 'docker ps' and nvidia-smi" >&2
}

require_image() {
    docker image inspect "$IMAGE" >/dev/null 2>&1 && return 0
    echo "image $IMAGE not local -- pulling" >&2
    aws ecr get-login-password --region "$K3V_REGION" \
        | docker login --username AWS --password-stdin "$ECR_REGISTRY" >/dev/null
    docker pull "$IMAGE"
}
