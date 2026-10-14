#!/usr/bin/env python3
"""Decode-arm comparison table from `vllm bench serve` result JSONs.

    python3 gen_compare_table.py results/ > results/compare.md

Generated, never hand-copied. Every axis is parsed from the filename that
91_bench.sh stamps; a row whose filename does not parse is reported, not
dropped silently. Latest run wins per (mode, seqs, isl, osl, conc).
"""
import json
import re
import sys
from pathlib import Path

PAT = re.compile(
    r"k3v_pd2p2d_kv-(?P<kv>[a-z]+)_dec-(?P<mode>[a-z_0-9]+)_sms(?P<sms>\d+)_seqs(?P<seqs>\d+)_isl(?P<isl>\d+)"
    r"_osl(?P<osl>\d+)_c(?P<conc>\d+)_n(?P<n>\d+)(?:_r(?P<rep>\d+))?_(?P<ts>[\d-]+)\.json$"
)
ORDER = {"deepep_v2": 0, "agrs": 1, "tp": 2}


def main(root: str) -> None:
    rows, bad = {}, []
    for p in sorted(Path(root).glob("k3v_*.json")):
        if p.name.endswith(".decode.json"):
            continue
        m = PAT.search(p.name)
        if not m:
            bad.append(p.name)
            continue
        d = json.loads(p.read_text())
        k = (m["kv"], m["mode"] if m["mode"] != "deepep_v2" else f"deepep_v2/sm{m['sms']}", int(m["seqs"]), int(m["isl"]), int(m["osl"]), int(m["conc"]))
        if k not in rows or m["ts"] > rows[k][0]:
            rows[k] = (m["ts"], int(m["n"]), d)

    base = {}  # (kv, seqs, isl, osl, conc) -> tp's median TPOT
    for (kv, mode, seqs, isl, osl, conc), (_, _, d) in rows.items():
        if mode == "tp":
            base[(kv, seqs, isl, osl, conc)] = d["median_tpot_ms"]

    print("| kv | decode arm | ISL/OSL | conc | ok/n | out tok/s | TPOT p50 (ms) | TPOT p99 | "
          "ITL p50 | TTFT p50 (ms) | dur (s) | TPOT vs tp |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in sorted(rows, key=lambda k: (k[0], k[3], k[4], k[5], ORDER.get(k[1], 9))):
        kv, mode, seqs, isl, osl, conc = k
        _, n, d = rows[k]
        ref = base.get((kv, seqs, isl, osl, conc))
        rel = f"{d['median_tpot_ms'] / ref - 1:+.1%}" if ref else "n/a"
        print(f"| {kv} | {mode} | {isl}/{osl} | {conc} | {d['completed']}/{n} | "
              f"{d['output_throughput']:.1f} | {d['median_tpot_ms']:.2f} | "
              f"{d['p99_tpot_ms']:.2f} | {d['median_itl_ms']:.2f} | "
              f"{d['median_ttft_ms']:.0f} | {d['duration']:.1f} | {rel} |")
    if bad:
        print(f"\nUNPARSED ({len(bad)}): " + ", ".join(bad), file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
