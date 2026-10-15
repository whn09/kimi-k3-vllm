#!/bin/bash
# HOST-side: `vllm bench serve` against the router, one run per concurrency.
# Run on B300-1 after 90_smoke_test.sh passes.
#
#   DECODE_MODE=deepep_v2 bash 91_bench.sh
#   DECODE_MODE=tp CONCS="256 1024" ISL=1024 OSL=2048 bash 91_bench.sh
#
# Writes per concurrency: <tag>.json (client, vllm bench serve), <tag>.log, and
# <tag>.decode.json (decode-side tok/s over the window where decode was FULL).
#
# DECODE_MODE here is a LABEL: the decode containers live on B300-3/4, which this
# host cannot inspect. 93_compare.sh launches and benches each arm with the same
# value so the label cannot drift from what was served.
set -euo pipefail
cd "$(dirname "$0")"
source ./env_common.sh

# Decode-saturation defaults: short ISL so prefill (TP16, shared by every arm)
# never starves decode, long OSL so the full-decode window dominates the ramp.
ISL="${ISL:-1024}"
OSL="${OSL:-2048}"
# 16 / 32 / 64 running per DP rank. The top point equals the admission cap, so
# it is the "decode full" row; decode_sampler.py proves whether it got there.
CONCS="${CONCS:-$(( DECODE_TOTAL_SEQS / 4 )) $(( DECODE_TOTAL_SEQS / 2 )) $DECODE_TOTAL_SEQS}"
NPROMPT_MULT="${NPROMPT_MULT:-2}"      # num_prompts = conc * this (>= 64): two full waves
REP="${REP:-1}"                        # replicate index, stamped into the name
# The SM count DeepEP actually resolved (94_sm_sweep.sh reads it from the
# decode log). DEEPEP_NUM_SMS=0 is a request for "auto", not a number.
RESOLVED_SMS="${RESOLVED_SMS:-unknown}"
WARMUP="${WARMUP:-1}"
URL_HOST="${URL_HOST:-$PREFILL_HEAD}"
# Image generation from the image actually benched: "...:vllm77154ca-...-pr61707"
# -> "pr61707" (last dash field), so runs of two images never share a table row.
IMG="${IMAGE##*-}"
OUT="$SCRIPT_DIR_HOST/results"
mkdir -p "$OUT"
if [[ "$DECODE_MODE" == tp ]]; then DECODE_URLS="http://${DECODE_HEAD}:${DECODE_PORT}"
else DECODE_URLS=$(for ip in $DECODE_IPS; do printf 'http://%s:%s ' "$ip" "$DECODE_PORT"; done); fi

run() {   # run <conc> <num_prompts> <tag>
    docker run --rm --net=host \
        -v "$HOST_MODEL_DIR/Kimi-K3:$MODEL_PATH:ro" -v "$OUT:/results" \
        "$IMAGE" \
        vllm bench serve \
            --backend openai --base-url "http://${URL_HOST}:${ROUTER_PORT}" \
            --endpoint /v1/completions \
            --model "$SERVED_MODEL_NAME" --tokenizer "$MODEL_PATH" --trust-remote-code \
            --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" \
            --random-range-ratio 0 --ignore-eos \
            --num-prompts "$2" --max-concurrency "$1" \
            --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
            --save-result --result-dir /results --result-filename "$3.json"
}

if [[ "$WARMUP" == 1 ]]; then
    echo "== warmup (not saved as a result)"
    run 16 32 "warmup_${DECODE_MODE}_$(date +%s)" >/dev/null 2>&1 || true
    rm -f "$OUT"/warmup_*.json
fi

for c in $CONCS; do
    n=$(( c * NPROMPT_MULT )); (( n < 64 )) && n=64
    # Every axis that can differ between arms goes in the name.
    tag="k3v_pd2p2d_kv-${KV_CONNECTOR}_dec-${DECODE_MODE}_sms${DEEPEP_NUM_SMS}_seqs${DECODE_TOTAL_SEQS}_isl${ISL}_osl${OSL}_c${c}_n${n}_r${REP}_img-${IMG}_$(date +%Y%m%d-%H%M%S)"
    echo "== $tag  (resolved sms=$RESOLVED_SMS)"
    python3 decode_sampler.py --out "$OUT/$tag.decode.json" --target "$c" $DECODE_URLS &
    spid=$!
    run "$c" "$n" "$tag" 2>&1 | tee "$OUT/$tag.log" | grep -E "Successful|Benchmark duration|Output token throughput|Mean TTFT|Median TTFT|Mean TPOT|Median TPOT|P99 TPOT|Median ITL|P99 ITL" || true
    kill -INT "$spid"; wait "$spid" || true
    # A bench that dies before sending a request still "finishes" -- in 10 s, with
    # the sampler reporting max_running=0. No result JSON = a failed run, stop.
    if [[ ! -s "$OUT/$tag.json" ]]; then
        echo "FATAL: vllm bench serve wrote no $tag.json -- tail of its log:" >&2
        tail -5 "$OUT/$tag.log" >&2
        exit 1
    fi
    python3 - "$OUT/$tag.decode.json" <<PY
import json, sys
p = sys.argv[1]; d = json.load(open(p))
d.update(decode_mode="$DECODE_MODE", requested_sms="$DEEPEP_NUM_SMS", resolved_sms="$RESOLVED_SMS",
         rep=$REP, seqs_per_rank=$DECODE_SEQS_PER_RANK, image="$IMAGE")
json.dump(d, open(p, "w"))
PY
done
