#!/bin/bash
# LAPTOP-side driver: the decode comparison, all arms back to back.
#
#   bash 93_compare.sh                         # ARMS="deepep_v2 agrs tp"
#   ARMS="tp" CONCS="32 128" bash 93_compare.sh
#
# Prefill (B300-1+2) is launched once and kept up; per arm only decode
# (B300-3+4) and the router are relaunched, so the prefill side is byte-identical
# across arms. Results are pulled after EVERY arm -- a terminated host takes its
# unpulled results with it.
set -euo pipefail
cd "$(dirname "$0")"
source ./env_common.sh

ARMS="${ARMS:-deepep_v2 agrs tp}"
P0="${P0:-B300-1}"; P1="${P1:-B300-2}"; D0="${D0:-B300-3}"; D1="${D1:-B300-4}"
KIT="$SCRIPT_DIR_HOST"
SSH=(ssh -o ConnectTimeout=10 -o BatchMode=yes -o LogLevel=ERROR -o ServerAliveInterval=30)
# Forwarded to every remote script so a laptop-side override reaches the hosts.
FWD="KV_CONNECTOR=${KV_CONNECTOR:-nixl} DEEPEP_NUM_SMS=${DEEPEP_NUM_SMS:-12} ISL=${ISL:-1024} OSL=${OSL:-1024} CONCS='${CONCS:-16 32 64 128}' DECODE_SEQS_PER_RANK=${DECODE_SEQS_PER_RANK:-8} IMAGE=$IMAGE"

rsh() { local h="$1"; shift; "${SSH[@]}" "$h" "cd $KIT && $FWD $*"; }

healthy() {   # healthy <ip:port>   (probed from P0, which can reach every node)
    "${SSH[@]}" "$P0" "curl -sf -m 5 http://$1/health >/dev/null"
}

wait_healthy() {   # wait_healthy <timeout_s> <container-host> <container> <ip:port>...
    local t="$1" ch="$2" cn="$3"; shift 3
    local deadline=$(( $(date +%s) + t )) ep
    for ep in "$@"; do
        until healthy "$ep"; do
            local st
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

echo "### prefill (TP16) on $P0+$P1"
if healthy "${PREFILL_HEAD}:${PREFILL_PORT}"; then
    echo "  already healthy -- reusing"
else
    rsh "$P0" "NODE_RANK=0 bash 20_launch_prefill.sh"
    rsh "$P1" "NODE_RANK=1 bash 20_launch_prefill.sh"
    wait_healthy 5400 "$P0" k3v-prefill "${PREFILL_HEAD}:${PREFILL_PORT}"
fi

for arm in $ARMS; do
    echo "### decode arm: $arm on $D0+$D1"
    rsh "$P0" "docker rm -f k3v-router >/dev/null 2>&1 || true"
    rsh "$D1" "docker rm -f k3v-decode >/dev/null 2>&1 || true"
    rsh "$D0" "DECODE_MODE=$arm NODE_RANK=0 bash 21_launch_decode.sh"
    rsh "$D1" "DECODE_MODE=$arm NODE_RANK=1 bash 21_launch_decode.sh"
    if [[ "$arm" == tp ]]; then
        eps=("${DECODE_HEAD}:${DECODE_PORT}")
    else
        eps=(); for ip in $DECODE_IPS; do eps+=("${ip}:${DECODE_PORT}"); done
    fi
    wait_healthy 5400 "$D0" k3v-decode "${eps[@]}"

    rsh "$P0" "DECODE_MODE=$arm bash 22_launch_router.sh"
    sleep 15
    rsh "$P0" "bash 90_smoke_test.sh" || { echo "FATAL: smoke test failed on arm $arm"; exit 1; }
    rsh "$P0" "DECODE_MODE=$arm bash 91_bench.sh"
    HOSTS="$P0" bash sync.sh results
    # Keep each arm's decode boot log: it is the proof of which a2a ran.
    mkdir -p results/logs
    for h in "$D0" "$D1"; do
        "${SSH[@]}" "$h" "docker logs k3v-decode 2>&1" > "results/logs/decode_${arm}_${h}_$(date +%Y%m%d-%H%M%S).log" || true
    done
done

echo "### done. Tables: python3 gen_compare_table.py results/"
