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

**Matched admission:** the DP arms admit `DECODE_SEQS_PER_RANK` (8) per rank =
128 in total; the `tp` arm gets `--max-num-seqs 128`. Otherwise TP would be
capped at 1/16 of the concurrency.

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
