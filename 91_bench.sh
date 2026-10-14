#!/bin/bash
# HOST-side: `vllm bench serve` against the router, one run per concurrency.
# Run on B300-1 after 90_smoke_test.sh passes.
#
#   DECODE_MODE=deepep_v2 bash 91_bench.sh
#   DECODE_MODE=tp CONCS="32 128" ISL=1024 OSL=1024 bash 91_bench.sh
#
# DECODE_MODE here is a LABEL: the decode containers live on B300-3/4, which this
# host cannot inspect. 93_compare.sh launches and benches each arm with the same
# value so the label cannot drift from what was served.
set -euo pipefail
cd "$(dirname "$0")"
source ./env_common.sh

ISL="${ISL:-1024}"
OSL="${OSL:-1024}"
CONCS="${CONCS:-16 32 64 128}"
NPROMPT_MULT="${NPROMPT_MULT:-4}"      # num_prompts = conc * this (>= 64)
WARMUP="${WARMUP:-1}"
URL_HOST="${URL_HOST:-$PREFILL_HEAD}"
OUT="$SCRIPT_DIR_HOST/results"
mkdir -p "$OUT"

run() {   # run <conc> <num_prompts> <tag>
    docker run --rm --net=host \
        -v "$HOST_MODEL_DIR/Kimi-K3:$MODEL_PATH:ro" -v "$OUT:/results" \
        "$IMAGE" \
        vllm bench serve \
            --backend openai --base-url "http://${URL_HOST}:${ROUTER_PORT}" \
            --endpoint /v1/completions \
            --model "$SERVED_MODEL_NAME" --tokenizer "$MODEL_PATH" --trust-remote-code \
            --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" \
            --random-range-ratio 1.0 --ignore-eos \
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
    tag="k3v_pd2p2d_kv-${KV_CONNECTOR}_dec-${DECODE_MODE}_seqs${DECODE_TOTAL_SEQS}_isl${ISL}_osl${OSL}_c${c}_n${n}_$(date +%Y%m%d-%H%M%S)"
    echo "== $tag"
    run "$c" "$n" "$tag" 2>&1 | tee "$OUT/$tag.log" | grep -E "Successful|Benchmark duration|Output token throughput|Mean TTFT|Median TTFT|Mean TPOT|Median TPOT|P99 TPOT|Median ITL|P99 ITL"
done
