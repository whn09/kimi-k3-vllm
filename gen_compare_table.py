#!/usr/bin/env python3
"""Decode-arm comparison table: deepep_v2 vs agrs (allgather_reducescatter) vs tp.

    python3 gen_compare_table.py results/ > results/compare.md

Generated, never hand-copied. Every axis is parsed from the filename that
91_bench.sh stamps; a file whose name does not parse is reported, not dropped
silently. Replicates of one (image, kv, arm, SM, isl, osl, conc) are shown as
mean [min-max].

Headline = decode-side tok/s over the window where decode was FULL
(decode_sampler.py, <tag>.decode.json); TPOT is the time axis. "vs deepep_v2"
compares each arm with the deepep_v2 SM12 row at the same (kv, isl, osl, conc).
deepep_v2 rows from the legacy-patch image (sms1) and from the PR #61707 image
are both valid baselines -- the PR run reproduced the legacy numbers within
their spread -- so they are pooled for that comparison, and the image column
says which runs a row holds.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

PAT = re.compile(
    r"k3v_pd2p2d_kv-(?P<kv>[a-z]+)_dec-(?P<mode>[a-z_0-9]+?)(?:_sms(?P<sms>\d+))?_seqs(?P<seqs>\d+)"
    r"_isl(?P<isl>\d+)_osl(?P<osl>\d+)_c(?P<conc>\d+)_n(?P<n>\d+)"
    r"(?:_r(?P<rep>\d+))?(?:_img-(?P<img>[a-z0-9]+))?_(?P<ts>[\d-]+)\.json$"
)
ORDER = {"deepep_v2": 0, "deepep_v2_1n_tp2": 1, "deepep_v2_1n_tp4": 1, "agrs": 2, "tp": 3}


def fmt(xs, nd=0):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n/a"
    if len(xs) == 1:
        return f"{xs[0]:.{nd}f}"
    return f"{mean(xs):.{nd}f} [{min(xs):.{nd}f}-{max(xs):.{nd}f}]"


def main(root):
    rows, bad = defaultdict(list), []
    for p in sorted(Path(root).glob("k3v_*.json")):
        if p.name.endswith(".decode.json"):
            continue
        m = PAT.search(p.name)
        if not m:
            bad.append(p.name)
            continue
        mode = m["mode"]
        sms = int(m["sms"]) if m["sms"] is not None else None
        # Only the SM12 deepep_v2 runs are the baseline; the auto-SM ones belong
        # to the SM sweep table.
        if mode == "deepep_v2" and sms != 12:
            continue
        if mode.startswith("deepep_v2_1n"):
            mode = f"{mode}/sm{'auto' if sms == 0 else sms}"
        dpath = p.with_name(p.name[:-5] + ".decode.json")
        dec = json.loads(dpath.read_text()) if dpath.exists() else {}
        key = (m["kv"], int(m["isl"]), int(m["osl"]), int(m["conc"]), mode)
        rows[key].append((m["img"] or "sms1", json.loads(p.read_text()), dec))

    print("| kv | ISL/OSL | conc | decode arm | image(s) | runs | decode-full tok/s | "
          "max running / target | TPOT p50 (ms) | ITL p50 (ms) | TTFT p50 (ms) | ok | vs deepep_v2 (tok/s / TPOT) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(rows, key=lambda k: (k[0], k[1], k[2], k[3], ORDER.get(k[4].split("/")[0], 9))):
        kv, isl, osl, conc, mode = key
        rs = rows[key]
        sat = [d.get("sat_gen_tok_s") for _, _, d in rs]
        full = all(s is not None for s in sat)
        tpot = [c["median_tpot_ms"] for _, c, _ in rs]
        base = rows.get((kv, isl, osl, conc, "deepep_v2"))
        rel = "—"
        if mode != "deepep_v2" and base:
            bs = [d.get("sat_gen_tok_s") for _, _, d in base if d.get("sat_gen_tok_s")]
            bt = [c["median_tpot_ms"] for _, c, _ in base]
            rel_s = f"{mean(sat) / mean(bs) - 1:+.1%}" if full and bs else "n/a"
            rel = f"{rel_s} / {mean(tpot) / mean(bt) - 1:+.1%}"
        label = {"deepep_v2": "deepep_v2 (SM12, EP16 cross-node)"}.get(mode, mode)
        if mode.startswith("deepep_v2_1n"):
            label = f"2 x 1-node EP8, {mode.split('/')[0].rsplit('_', 1)[1].upper()} attn ({mode.split('/')[1]})"
        imgs = "+".join(sorted({i for i, _, _ in rs}))
        mx = max(d.get("max_running", 0) or 0 for _, _, d in rs)
        ok = "/".join(f"{c['completed']}" for _, c, _ in rs)
        print(f"| {kv} | {isl}/{osl} | {conc} | {label} | {imgs} | {len(rs)} | "
              f"{fmt(sat) if full else 'NOT FULL'} | {mx:.0f} / {conc} | {fmt(tpot, 2)} | "
              f"{fmt([c['median_itl_ms'] for _, c, _ in rs], 2)} | "
              f"{fmt([c['median_ttft_ms'] for _, c, _ in rs])} | {ok} | {rel} |")
    if bad:
        print(f"\nUNPARSED ({len(bad)}): " + ", ".join(bad), file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
