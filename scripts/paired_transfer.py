"""Pair on the board seed instead of averaging it away.

Two arms measured on the *same* board seeds share that board's difficulty, and
board difficulty is the dominant variance term here: the within-training-seed
sd across boards is 0.29 to 0.34, while the real between-training-seed sd is
under 0.10. An unpaired test throws that shared term into the error; a paired
one cancels it. The question this script answers is how many board seeds a
paired test needs to reach the conclusion twelve unpaired ones reached.
"""
import sys, statistics, math, json, re
from pathlib import Path

PATTERN = re.compile(r"train(?P<train>\d+)_seed(?P<seed>\d+)")

def by_board(run_dir, metric="score", agent="research_agent"):
    out = {}
    for job in sorted((Path(run_dir) / "jobs").iterdir()):
        m = PATTERN.search(job.name)
        stats = job / "official_stats.json"
        if not m or not stats.is_file():
            continue
        mine = json.loads(stats.read_text())["by_agent"].get(agent)
        if mine is None:
            continue
        rounds = float(mine.get("rounds") or 0) or 1.0
        value = float(mine.get(metric, 0)) / rounds if metric != "score" else float(mine["score"]) / rounds
        out.setdefault(int(m.group("seed")), []).append(value)
    return {seed: statistics.fmean(v) for seed, v in out.items()}

def paired_t(a, b):
    """b minus a over the boards both were measured on."""
    common = sorted(set(a) & set(b))
    diffs = [b[s] - a[s] for s in common]
    if len(diffs) < 2:
        return float("nan"), float("nan"), 0
    mean = statistics.fmean(diffs)
    se = statistics.stdev(diffs) / math.sqrt(len(diffs))
    return mean, (mean / se if se else float("nan")), len(diffs)

def unpaired_t(a, b):
    va, vb = list(a.values()), list(b.values())
    se = math.sqrt(statistics.variance(va) / len(va) + statistics.variance(vb) / len(vb))
    d = statistics.fmean(vb) - statistics.fmean(va)
    return d, (d / se if se else float("nan"))

CASES = [("EPS0 vs RAINBOW, frozen R02_9", "runs/rep_ln_rain_r029", "runs/rep_ln_eps0_r029"),
         ("EPS0 vs RAINBOW, coin_collector", "runs/rep_ln_rain_coin", "runs/rep_ln_eps0_coin"),
         ("EPS0 vs RAINBOW, frozen RAINBOW", "runs/rep_ln_rain_rbow", "runs/rep_ln_eps0_rbow")]
print("%-34s %22s %22s" % ("", "UNPAIRED (12 boards)", "PAIRED"))
print("%-34s %10s %10s %6s %10s %6s %10s %6s"
      % ("case", "delta", "t", "", "t @12", "", "t @6", ""))
for label, a_run, b_run in CASES:
    a, b = by_board(a_run), by_board(b_run)
    du, tu = unpaired_t(a, b)
    dp, tp, n = paired_t(a, b)
    half = sorted(set(a) & set(b))[:6]
    a6 = {s: a[s] for s in half}; b6 = {s: b[s] for s in half}
    _, tp6, n6 = paired_t(a6, b6)
    print("%-34s %10.4f %10.2f %6s %10.2f %6s %10.2f %6s"
          % (label, du, tu, "*" if abs(tu) > 2.3 else "",
             tp, "*" if abs(tp) > 2.3 else "", tp6, "*" if abs(tp6) > 2.3 else ""))
print()
print("* = |t| > 2.3.  Paired @6 uses half the compute of unpaired @12.")
