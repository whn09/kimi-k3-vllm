#!/bin/bash
# LAPTOP-side driver: DeepEP v2 decode, SM=12 vs DeepEP's own pick ("auto"),
# with decode driven to its admission cap. THE priority experiment.
#
#   bash 94_sm_sweep.sh                       # SMS_LIST="12 0", REPS=2
#   SMS_LIST="12 0 24" REPS=3 bash 94_sm_sweep.sh
#   CONCS="1024" bash 94_sm_sweep.sh          # only the full-decode point
#
# Order is INTERLEAVED (12, auto, 12, auto, ...), never 12,12,auto,auto: a slow
# drift of the machines (thermal, a neighbour's job, cache state) then lands on
# both arms equally, and the spread between replicates of the SAME arm is the
# noise floor any 12-vs-auto gap has to beat.
#
# Prefill (TP16) is booted once and shared by every run. Decode is rebooted per
# run because the SM count is fixed at ElasticBuffer creation.
# Results are pulled after EVERY run.
set -euo pipefail
cd "$(dirname "$0")"
source ./env_common.sh
source ./lib_drive.sh

SMS_LIST="${SMS_LIST:-12 0}"
REPS="${REPS:-2}"

echo "plan: sms=[$SMS_LIST] x reps=$REPS  seqs/rank=$DECODE_SEQS_PER_RANK (side cap $DECODE_TOTAL_SEQS)  ISL/OSL=${ISL:-1024}/${OSL:-2048}  concs=[${CONCS:-<91_bench default>}]"
check_gpus_free "$P0" "$P1" "$D0" "$D1"
ensure_prefill

for rep in $(seq 1 "$REPS"); do
    for sms in $SMS_LIST; do
        label="deepep_v2_sms${sms}_r${rep}"
        echo "### $label"
        launch_decode deepep_v2 "DEEPEP_NUM_SMS=$sms"
        kv_capacity_check
        rs=$(resolved_sms)
        echo "  DeepEP resolved num_sms=$rs (requested ${sms}; 0 = auto)"
        if [[ "$rs" == mixed* || -z "$rs" ]]; then
            echo "FATAL: ranks disagree on / never logged num_sms: '$rs'"; exit 1
        fi
        if (( sms > 0 )) && [[ "$rs" != "$sms" ]]; then
            echo "FATAL: asked for $sms SMs, DeepEP is using $rs -- the patch is not in effect"; exit 1
        fi
        rsh "$P0" "bash 90_smoke_test.sh" >/dev/null || { echo "FATAL: smoke test failed on $label"; exit 1; }
        rsh "$P0" "DECODE_MODE=deepep_v2 DEEPEP_NUM_SMS=$sms RESOLVED_SMS=$rs REP=$rep bash 91_bench.sh"
        HOSTS="$P0" bash sync.sh results
        save_decode_logs "$label"
    done
done

python3 gen_sm_table.py results/ | tee results/sm_sweep.md
