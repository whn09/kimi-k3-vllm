<!-- markdownlint-disable -->

## Overview

Add `VLLM_DEEPEP_V2_NUM_SMS` so the SM count used by DeepEP v2 dispatch/combine can be set. Today vLLM never passes `num_sms`, and DeepEP's own pick never goes below 64 SMs under vLLM's default `VLLM_DEEPEP_V2_PREFER_OVERLAP=0`.

## Claims

- New env var `VLLM_DEEPEP_V2_NUM_SMS` (default `0` = current behaviour, unchanged).
- On a decode-saturated PD deployment over EFA, `VLLM_DEEPEP_V2_NUM_SMS=12` gives **+10.4% to +12.5% decode throughput** and **-11% to -12% TPOT** vs the default (which resolves to 64 SMs).

## Validation

**Setup:** Kimi-K3, 4x p6-b300 (8x B300 each, 16x EFA), PD 2P2D. Prefill: TP16/EP16 on 2 nodes. Decode: TP1 x DP16, EP16, `--all2all-backend deepep_v2` (hybrid: NVLink + EFA, NCCL GIN), 64 seqs per DP rank, `FULL_DECODE_ONLY` graphs, NixlConnector. `vllm bench serve` random dataset, ISL 1024 / OSL 2048 (fixed), 2 waves of requests per point. 2 replicates per arm, interleaved (12, default, 12, default).

Decode throughput is measured on the decode servers (`vllm:generation_tokens_total`) over the window where summed `vllm:num_requests_running` >= 90% of the target concurrency; every run reached the target. Values are mean [min-max] over the 2 replicates.

| concurrency | decode tok/s, default (64 SMs) | decode tok/s, `NUM_SMS=12` | Δ | TPOT p50 default (ms) | TPOT p50 `NUM_SMS=12` (ms) | Δ |
|---|---|---|---|---|---|---|
| 256 | 3601 [3598-3604] | 4048 [4040-4056] | +12.4% | 70.72 | 62.87 | -11.1% |
| 512 | 5840 [5839-5841] | 6571 [6518-6624] | +12.5% | 87.26 | 76.74 | -12.1% |
| 1024 | 10240 [10239-10241] | 11303 [11102-11505] | +10.4% | 99.53 | 87.87 | -11.7% |

The largest replicate-to-replicate spread of either arm is 3.6% (tok/s) / 1.4% (TPOT), below every delta.

**Validated on this exact diff.** The PR's diff was applied with `patch` onto vLLM `77154ca` (the base of the table above; 3 of the 4 files are byte-identical to this PR's base, `envs.py` differs by a 1-line offset), same hardware and config, one decode boot per setting, c=1024:

| `VLLM_DEEPEP_V2_NUM_SMS` | log line (`get_handle`) | decode tok/s | TPOT p50 |
|---|---|---|---|
| `12` | `DeepEP v2 dispatch/combine uses 12 SMs` | 11507 | 86.44 ms |
| unset (`0`, default) | `DeepEP v2 dispatch/combine uses 64 SMs` | 9956 | 98.73 ms |

Both reproduce the table above within its replicate spread (12: 11303 [11102-11505] / 87.87 ms; default: 10240 / 99.53 ms), and the default resolves to the same 64 SMs as before the change. All requests completed; greedy completions through the PD router were correct in both runs.

Reproduce (decode side):

```bash
VLLM_DEEPEP_V2_NUM_SMS=12 vllm serve moonshotai/Kimi-K3 --trust-remote-code \
  --tensor-parallel-size 1 --data-parallel-size 16 --data-parallel-size-local 8 \
  --data-parallel-hybrid-lb --enable-expert-parallel --all2all-backend deepep_v2 \
  --max-num-seqs 64 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}' ...
```

## Details

- **Root cause.** `DeepEPV2PrepareAndFinalize` calls `ElasticBuffer.dispatch()` without `num_sms`, so DeepEP uses `get_theoretical_num_sms()`, which ends with `num_sms if self.prefer_overlap_with_compute else max(num_sms, 64)`. vLLM builds the buffer with `prefer_overlap_with_compute=VLLM_DEEPEP_V2_PREFER_OVERLAP` (default off), so dispatch/combine always takes >= 64 SMs. When the a2a is network-bound (EFA), the extra SMs don't speed it up; they only slow down the overlapping compute.
- **Design.** The count is resolved once in `DeepEPV2All2AllManager.get_handle()` (env value, or the theoretical count when 0), stored as `_num_sms` and returned by `max_sms_used()`. That means DBO's compute/comm SM split sees the same number the kernels use. It is then passed to `DeepEPV2PrepareAndFinalize` and on to `dispatch(num_sms=...)`. `combine()` keeps `num_sms=0`, which DeepEP defines as "reuse the dispatch handle's SM count".
- **Default unchanged.** With the env unset, `get_handle()` calls `get_theoretical_num_sms()` with the same `num_experts`/`num_topk` dispatch would use, and that function is cached, so the value passed explicitly is the one DeepEP would have picked.
- **Why not `VLLM_DEEPEP_V2_PREFER_OVERLAP=1`?** That also drops the 64-SM floor, but it changes how DeepEP sizes the buffer and schedules overlap, and the theoretical count is still not tunable per deployment. A direct knob is simpler to reason about and benchmark. (Not measured in this PR.)
- **Scope: one value per server process.** The knob applies to every dispatch the server issues, prefill and decode batches alike. In a PD deployment, set it separately for the prefill and decode instances; the numbers above are decode-only. Prefill moves far more bytes per step and is more likely to want a larger count. Per-phase selection (e.g. by token count) is left out to keep this change minimal.
- **Tests.** No new test: exercising the path needs DeepEP v2 with NCCL GIN on multiple GPUs, and a mock-buffer test would only check that an argument is forwarded. CI does not cover multi-node deepep_v2; validation was done on the hardware above.

---

<details>
<summary> Pull Request Checklist </summary>

- [x] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [x] AI assistance was used during the creation of this PR.

- [x] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [x] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI.
- [x] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [x] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>
