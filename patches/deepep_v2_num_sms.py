"""Expose DeepEP v2's SM count to vLLM as VLLM_DEEPEP_V2_NUM_SMS.

vLLM calls ElasticBuffer.dispatch() without num_sms, so DeepEP resolves it with
get_theoretical_num_sms(), which returns max(theory, 64) whenever
prefer_overlap_with_compute is False -- and vLLM's VLLM_DEEPEP_V2_PREFER_OVERLAP
defaults to 0. On EFA decode, 12 SMs beats that. 0/unset keeps upstream
behaviour. combine() is left at num_sms=0, which DeepEP documents as "reuse the
dispatch handle's SM count", and max_sms_used() reports the same value so the
DBO SM partitioning agrees with what the kernels actually take.

Applied at image build: python3 deepep_v2_num_sms.py <vllm package dir>
Every replacement asserts it matched exactly once, so a moved base image fails
the build instead of silently shipping an unpatched file.
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])


def patch(rel: str, old: str, new: str) -> None:
    p = root / rel
    s = p.read_text()
    if new in s:
        print(f"  SKIP  {rel} (already patched)")
        return
    n = s.count(old)
    assert n == 1, f"{rel}: expected 1 match, found {n}:\n{old}"
    p.write_text(s.replace(old, new))
    print(f"  patched {rel}")


ENV = 'int(__import__("os").environ.get("VLLM_DEEPEP_V2_NUM_SMS", "0"))'

patch(
    "model_executor/layers/fused_moe/prepare_finalize/deepep_v2.py",
    """            do_cpu_sync=do_cpu_sync,
            async_with_compute_stream=False,
        )
""",
    f"""            do_cpu_sync=do_cpu_sync,
            async_with_compute_stream=False,
            num_sms={ENV},
        )
""",
)
patch(
    "distributed/device_communicators/all2all.py",
    """        if self._num_sms is None:
            self._num_sms = handle.get_theoretical_num_sms(
                num_experts=num_experts,
                num_topk=kwargs["num_topk"],
            )
        return handle
""",
    f"""        if self._num_sms is None:
            self._num_sms = {ENV} or handle.get_theoretical_num_sms(
                num_experts=num_experts,
                num_topk=kwargs["num_topk"],
            )
            logger.info("DeepEP v2 num_sms=%d (VLLM_DEEPEP_V2_NUM_SMS=%s)",
                        self._num_sms,
                        __import__("os").environ.get("VLLM_DEEPEP_V2_NUM_SMS", "<unset>"))
        return handle
""",
)
