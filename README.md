# kimi-k3-vllm

Kimi-K3 PD-disaggregated serving on **4 × p6-b300 (EFA)** with **vLLM + DeepEP v2**,
the vLLM sibling of [kimi-k3-sglang](https://github.com/whn09/kimi-k3-sglang).

Question this kit answers: **on the decode side, how much does DeepEP v2 over EFA
buy relative to vLLM's default all-gather/reduce-scatter a2a and to plain TP?**

## Topology (2P2D)

| side | hosts | layout | MoE path |
|---|---|---|---|
| prefill | B300-1 (rank 0) + B300-2 | TP16, EP16, DP1 | all-reduce (no a2a; identical in every arm) |
| decode `deepep_v2` | B300-3 + B300-4 | TP1 × DP16, EP16, hybrid-lb | DeepEP v2 hybrid (NVLink + EFA GIN type 5) |
| decode `agrs` | B300-3 + B300-4 | TP1 × DP16, EP16, hybrid-lb | `allgather_reducescatter` over NCCL |
| decode `tp` | B300-3 (rank 0) + B300-4 | TP16, EP16, DP1 | all-reduce |
| router | B300-1 | `vllm-router --vllm-pd-disaggregation` | — |

KV transfer: `NixlConnector`, **LIBFABRIC** backend (EFA has no UCX `rc`).

**Why prefill is not on deepep_v2:** vLLM only enables an a2a when DP > 1
(`ParallelConfig.use_sequence_parallel_moe` / `FusedMoEParallelConfig.use_all2all_kernels`).
At TP16/DP1, `--all2all-backend deepep_v2` is accepted and silently does nothing.
For the same reason the `tp` decode arm *is* the "pure TP" baseline.

**Matched admission:** the DP arms admit `DECODE_SEQS_PER_RANK` (default 64) per
rank = 1024 in total; the `tp` arm gets `--max-num-seqs 1024`. Otherwise TP would
be capped at 1/16 of the concurrency. agrs / tp: `ARMS="agrs tp" bash 93_compare.sh`
(lower priority).

## Image

`Dockerfile` = pinned `vllm/vllm-openai:nightly-77154ca…` (CUDA 13.0) +
EFA 1.50.0 (libfabric, aws-ofi-nccl w/ `ncclGinPlugin_v14`) + gdrcopy +
`nvidia-nccl-cu13==2.31.2` + **amazon-contributing/DeepEP@6de427b built for
sm_100/sm_103** (replaces the nightly's deep_ep 2.0.0, which has no sm_103
cubins and links NCCL 2.30.7) + `vllm-router`.

```bash
bash 01_build_push.sh          # on B300-1: build + push to ECR ap-south-2
bash sync.sh pull-image        # on the laptop: pull on all 4, prints image ids
```

## Priority experiment: decode SM=12 vs auto (`94_sm_sweep.sh`)

vLLM never passes `num_sms` to `ElasticBuffer.dispatch`, so DeepEP picks
`get_theoretical_num_sms()`, which is **max(theory, 64)** whenever
`prefer_overlap_with_compute` is off -- and vLLM's `VLLM_DEEPEP_V2_PREFER_OVERLAP`
defaults to 0. `patches/deepep_v2_num_sms.py` adds `VLLM_DEEPEP_V2_NUM_SMS`
(0 = DeepEP's pick); the decode launcher sets it from `DEEPEP_NUM_SMS` (default 12).

```bash
bash 94_sm_sweep.sh                    # SMS_LIST="12 0" REPS=2, interleaved 12,auto,12,auto
CONCS=1024 REPS=3 bash 94_sm_sweep.sh  # only the decode-full point, 3 reps
python3 gen_sm_table.py results/
```

- **Decode is driven to its cap:** 64 seqs per DP rank (1024 per side, vs the
  recipe's 8/rank = 128), ISL 1024 / OSL 2048 so the shared TP16 prefill cannot
  starve it, concurrency 256 / 512 / 1024, two waves of requests each.
- **"Full" is measured, not assumed:** `decode_sampler.py` polls both decode
  nodes' `/metrics` every 5 s, and the headline number is decode-side tok/s over
  the window where summed `num_requests_running` >= 90% of the target. A run
  that never got there prints `NOT FULL`.
- **The SM count actually used** is read back from every rank's log (the patch
  logs `DeepEP v2 num_sms=N`); a rank disagreement or a requested-vs-used
  mismatch aborts the sweep.
- **Noise floor:** each arm's own max-min spread across replicates is printed
  next to the 12-vs-auto delta; the verdict only says "beats noise" when the
  delta exceeds both.
- Each run gets a fresh decode boot (SM count is fixed at ElasticBuffer
  creation, ~8-10 min warm-cache); prefill is booted once and shared.

## Run one arm by hand

```bash
bash 00_download_models.sh                                  # every host, ~5 min
# prefill
ssh B300-1 'cd ~/kimi-k3-vllm && NODE_RANK=0 bash 20_launch_prefill.sh'
ssh B300-2 'cd ~/kimi-k3-vllm && NODE_RANK=1 bash 20_launch_prefill.sh'
# decode (DECODE_MODE = deepep_v2 | agrs | tp)
ssh B300-3 'cd ~/kimi-k3-vllm && DECODE_MODE=deepep_v2 NODE_RANK=0 bash 21_launch_decode.sh'
ssh B300-4 'cd ~/kimi-k3-vllm && DECODE_MODE=deepep_v2 NODE_RANK=1 bash 21_launch_decode.sh'
# wait for /health on B300-1:8001 and B300-3/4:8002, then
ssh B300-1 'cd ~/kimi-k3-vllm && DECODE_MODE=deepep_v2 bash 22_launch_router.sh'
ssh B300-1 'cd ~/kimi-k3-vllm && bash 90_smoke_test.sh'
ssh B300-1 'cd ~/kimi-k3-vllm && DECODE_MODE=deepep_v2 bash 91_bench.sh'
```

## GIN type 5 checklist (decode `deepep_v2`)

`NCCL >= 2.31` (image) + `NCCL_GIN_TYPE=5` + `NCCL_SYM_GIN_KERNELS_ENABLE=0` +
`NCCL_IB_HCA=rdmap` (b300 also lists 2 ConnectX-7 `ibp*`) + `--privileged` +
`--device=/dev/gdrdrv`. All set by `env_common.sh`. To prove the backend, relaunch
with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GIN` and look for
`Loaded gin plugin Libfabric_GDAKI (v14)` and
`Skipping plugin Libfabric index 3 type 2`.
