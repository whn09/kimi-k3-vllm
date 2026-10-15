#!/bin/bash
# LAPTOP-side helpers shared by 93_compare.sh and 94_sm_sweep.sh. Sourced after
# env_common.sh. Every value a remote script needs is forwarded in $FWD, so a
# laptop-side override always reaches the hosts.
P0="${P0:-B300-1}"; P1="${P1:-B300-2}"; D0="${D0:-B300-3}"; D1="${D1:-B300-4}"
KIT="$SCRIPT_DIR_HOST"
SSH=(ssh -o ConnectTimeout=10 -o BatchMode=yes -o LogLevel=ERROR -o ServerAliveInterval=30)
FWD="${DECODE_1N_SEQS_PER_DP:+DECODE_1N_SEQS_PER_DP=$DECODE_1N_SEQS_PER_DP }KV_CONNECTOR=$KV_CONNECTOR ISL=${ISL:-1024} OSL=${OSL:-2048} ${CONCS:+CONCS='$CONCS'} DECODE_SEQS_PER_RANK=$DECODE_SEQS_PER_RANK IMAGE=$IMAGE"

rsh() { local h="$1"; shift; "${SSH[@]}" "$h" "cd $KIT && $FWD $*"; }

healthy() {   # healthy <ip:port>   (probed from P0, which can reach every node)
    "${SSH[@]}" "$P0" "curl -sf -m 5 http://$1/health >/dev/null"
}

wait_healthy() {   # wait_healthy <timeout_s> <container-host> <container> <ip:port>...
    local t="$1" ch="$2" cn="$3"; shift 3
    local deadline=$(( $(date +%s) + t )) ep st
    for ep in "$@"; do
        until healthy "$ep"; do
            st=$("${SSH[@]}" "$ch" "docker inspect -f '{{.State.Status}}' $cn 2>/dev/null || echo gone")
            if [[ "$st" != running ]]; then
                echo "FATAL: $cn on $ch is '$st' while waiting for $ep. Last log lines:"
                "${SSH[@]}" "$ch" "docker logs --tail 60 $cn 2>&1"
                return 1
            fi
            (( $(date +%s) > deadline )) && { echo "FATAL: $ep not healthy after ${t}s"; return 1; }
            sleep 20
        done
        echo "  healthy: $ep"
    done
}

# Both sides refuse to start on a host that already has someone on its GPUs: a
# colleague's job on one GPU looks like an OOM or "memory capacity is
# unbalanced", far from the cause.
check_gpus_free() {
    local h used
    for h in "$@"; do
        used=$("${SSH[@]}" "$h" "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1")
        others=$("${SSH[@]}" "$h" "docker ps --format '{{.Names}}' | grep -v '^k3v-' || true")
        if [[ -n "$others" ]] || { (( used > 2000 )) && ! "${SSH[@]}" "$h" "docker ps --format '{{.Names}}' | grep -q '^k3v-'"; }; then
            echo "FATAL: $h is busy (max GPU ${used} MiB; other containers: ${others:-none}) -- not launching on it"
            return 1
        fi
    done
}

ensure_prefill() {
    echo "### prefill (TP16) on $P0+$P1"
    if healthy "${PREFILL_HEAD}:${PREFILL_PORT}"; then
        echo "  already healthy -- reusing"; return 0
    fi
    rsh "$P0" "NODE_RANK=0 bash 20_launch_prefill.sh"
    rsh "$P1" "NODE_RANK=1 bash 20_launch_prefill.sh"
    wait_healthy 5400 "$P0" k3v-prefill "${PREFILL_HEAD}:${PREFILL_PORT}"
}

# launch_decode <mode> [extra env assignments...]; then waits for every decode
# API server the router will be given.
launch_decode() {
    local mode="$1"; shift
    rsh "$P0" "docker rm -f k3v-router >/dev/null 2>&1 || true"
    rsh "$D1" "docker rm -f k3v-decode >/dev/null 2>&1 || true"
    rsh "$D0" "$* DECODE_MODE=$mode NODE_RANK=0 bash 21_launch_decode.sh"
    rsh "$D1" "$* DECODE_MODE=$mode NODE_RANK=1 bash 21_launch_decode.sh"
    local eps=()
    if [[ "$mode" == tp ]]; then eps=("${DECODE_HEAD}:${DECODE_PORT}")
    else for ip in $DECODE_IPS; do eps+=("${ip}:${DECODE_PORT}"); done; fi
    wait_healthy 5400 "$D0" k3v-decode "${eps[@]}"
    rsh "$P0" "DECODE_MODE=$mode bash 22_launch_router.sh"
    sleep 15
}

# The SM count DeepEP really uses, from patches/deepep_v2_num_sms.py's log line.
# All 16 ranks must agree; anything else is reported, not averaged away.
resolved_sms() {
    local v
    # Two log formats: the legacy patch ("DeepEP v2 num_sms=N", per rank) and
    # PR #61707 ("DeepEP v2 dispatch/combine uses N SMs", info_once per process).
    # grep -o, never per line: concurrent ranks glue records onto one line.
    v=$( { "${SSH[@]}" "$D0" "docker logs k3v-decode 2>&1"; "${SSH[@]}" "$D1" "docker logs k3v-decode 2>&1"; } \
        | grep -o -E 'DeepEP v2 num_sms=[0-9]+|DeepEP v2 dispatch/combine uses [0-9]+ SMs' \
        | sed -E 's/.*(num_sms=|uses )([0-9]+).*/\2/' | sort | uniq -c)
    if [[ $(wc -l <<<"$v") -eq 1 && -n "$v" ]]; then awk '{print $2}' <<<"$v"
    else echo "mixed[$(tr -s ' \n' ' ' <<<"$v")]"; fi
}

save_decode_logs() {   # save_decode_logs <label>
    mkdir -p results/logs
    local h
    for h in "$D0" "$D1"; do
        "${SSH[@]}" "$h" "docker logs k3v-decode 2>&1" > "results/logs/decode_$1_${h}_$(date +%Y%m%d-%H%M%S).log" || true
    done
}

# Decode can only be "full" if every DP rank's KV pool holds seqs_per_rank whole
# sequences; otherwise running caps below the admission limit and the excess
# shows up as TTFT, not as a lower tok/s. Warn up front.
kv_capacity_check() {
    local need min
    need=$(( DECODE_SEQS_PER_RANK * (${ISL:-1024} + ${OSL:-2048}) ))
    min=$( { "${SSH[@]}" "$D0" "docker logs k3v-decode 2>&1"; "${SSH[@]}" "$D1" "docker logs k3v-decode 2>&1"; } \
        | sed -n 's/.*GPU KV cache size: \([0-9,]*\) tokens.*/\1/p' | tr -d , | sort -n | head -1)
    if [[ -z "$min" ]]; then
        echo "  WARN: no 'GPU KV cache size' line in the decode logs -- cannot check residency"
    elif (( min < need )); then
        echo "  WARN: smallest decode KV pool is $min tokens < $need needed for $DECODE_SEQS_PER_RANK x $(( ${ISL:-1024} + ${OSL:-2048} )) -- decode cannot fill"
    else
        echo "  KV ok: smallest decode pool $min tokens >= $need needed per rank"
    fi
}
