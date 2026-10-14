#!/bin/bash
# LAPTOP-side driver: the decode comparison, all arms back to back.
#
#   bash 93_compare.sh                         # ARMS="deepep_v2 agrs tp"
#   ARMS="agrs tp" bash 93_compare.sh          # lower priority than 94_sm_sweep.sh
#
# Prefill (B300-1+2) is launched once and kept up; per arm only decode
# (B300-3+4) and the router are relaunched, so the prefill side is byte-identical
# across arms. Results are pulled after EVERY arm -- a terminated host takes its
# unpulled results with it.
set -euo pipefail
cd "$(dirname "$0")"
source ./env_common.sh

ARMS="${ARMS:-deepep_v2 agrs tp}"
source ./lib_drive.sh
check_gpus_free "$P0" "$P1" "$D0" "$D1"
ensure_prefill

for arm in $ARMS; do
    echo "### decode arm: $arm on $D0+$D1"
    launch_decode "$arm" "DEEPEP_NUM_SMS=${DEEPEP_NUM_SMS:-12}"
    kv_capacity_check
    rs=unknown; [[ "$arm" == deepep_v2 ]] && rs=$(resolved_sms)
    rsh "$P0" "bash 90_smoke_test.sh" || { echo "FATAL: smoke test failed on arm $arm"; exit 1; }
    rsh "$P0" "DECODE_MODE=$arm DEEPEP_NUM_SMS=${DEEPEP_NUM_SMS:-12} RESOLVED_SMS=$rs bash 91_bench.sh"
    HOSTS="$P0" bash sync.sh results
    save_decode_logs "$arm"
done

python3 gen_compare_table.py results/ | tee results/compare.md
