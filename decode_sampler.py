#!/usr/bin/env python3
"""Sample decode-side Prometheus counters while a benchmark runs.

    python3 decode_sampler.py --out X.decode.json --target 1024 \
        http://172.31.0.212:8002 http://172.31.9.249:8002 &
    ... benchmark ...
    kill -INT $!            # writes X.decode.json on exit

Why: the client-side `vllm bench serve` throughput averages over the ramp-up
(every request's prefill arrives at once) and the drain, so it under-reports
what decode sustains once it is FULL. This polls every decode API server
(each hybrid-lb node exposes only its own DP engines, so all of them must be
summed), and reports generation tok/s over the window where the summed
`vllm:num_requests_running` is >= SAT_FRAC x target -- i.e. decode actually
saturated. If that window is empty, decode was never full and the row says so
instead of quoting a number.
"""
import argparse
import json
import re
import signal
import sys
import time
import urllib.request

GEN = re.compile(r'^vllm:generation_tokens_total\{[^}]*\}\s+([0-9.e+]+)$', re.M)
RUN = re.compile(r'^vllm:num_requests_running\{[^}]*\}\s+([0-9.e+]+)$', re.M)
WAIT = re.compile(r'^vllm:num_requests_waiting\{[^}]*\}\s+([0-9.e+]+)$', re.M)
KVU = re.compile(r'^vllm:kv_cache_usage_perc\{[^}]*\}\s+([0-9.e+]+)$', re.M)


def scrape(urls):
    gen = run = wait = 0.0
    kv = []
    for u in urls:
        txt = urllib.request.urlopen(u + "/metrics", timeout=5).read().decode()
        gen += sum(float(x) for x in GEN.findall(txt))
        run += sum(float(x) for x in RUN.findall(txt))
        wait += sum(float(x) for x in WAIT.findall(txt))
        kv += [float(x) for x in KVU.findall(txt)]
    return gen, run, wait, (max(kv) if kv else None)


def summarize(samples, target, sat_frac):
    sat = [s for s in samples if s["running"] >= sat_frac * target]
    out = {
        "target_running": target,
        "sat_frac": sat_frac,
        "n_samples": len(samples),
        "max_running": max((s["running"] for s in samples), default=0),
        "max_waiting": max((s["waiting"] for s in samples), default=0),
        "max_kv_usage": max((s["kv"] or 0 for s in samples), default=0),
        "sat_samples": len(sat),
    }
    if len(sat) >= 2:
        dt = sat[-1]["t"] - sat[0]["t"]
        out["sat_window_s"] = dt
        out["sat_gen_tok_s"] = (sat[-1]["gen"] - sat[0]["gen"]) / dt if dt > 0 else None
        out["sat_mean_running"] = sum(s["running"] for s in sat) / len(sat)
    else:
        out["sat_window_s"] = 0
        out["sat_gen_tok_s"] = None   # decode never full: do not quote a number
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, required=True, help="expected running reqs when full")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--sat-frac", type=float, default=0.9)
    ap.add_argument("urls", nargs="+")
    a = ap.parse_args()

    samples, t0 = [], time.time()
    stop = False

    def _stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    while not stop:
        try:
            g, r, w, kv = scrape(a.urls)
            samples.append({"t": time.time() - t0, "gen": g, "running": r, "waiting": w, "kv": kv})
        except Exception as e:  # a missed scrape is a gap, not a failure
            print(f"sampler: scrape failed: {e}", file=sys.stderr)
        time.sleep(a.interval)

    res = summarize(samples, a.target, a.sat_frac)
    res["samples"] = samples
    with open(a.out, "w") as f:
        json.dump(res, f)
    print(f"sampler: max_running={res['max_running']:.0f}/{a.target} "
          f"sat_window={res['sat_window_s']:.0f}s sat_gen_tok_s={res['sat_gen_tok_s']}")


if __name__ == "__main__":
    main()
