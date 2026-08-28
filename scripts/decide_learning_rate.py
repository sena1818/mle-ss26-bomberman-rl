#!/usr/bin/env python3
"""Execute the pre-registered step-size decision, and record it.

``run_m4_line.sh`` used to point at ``compare_runs.py`` here, which cannot do
this job: it does not treat the learning rate as a comparison dimension and it
does not compute the pre-registered rule.  A gate a human has to eyeball is not
a gate -- the whole reason the step size is settled before the increments is
that docs/05 section 0.20 published a conclusion that an untuned base inverted.

The rule, from docs/06 section 8:

    On the POOLED VALIDATION curve of each arm, take
        mean(last two evaluated checkpoints) - mean(first two)
    Compare each candidate arm's final pooled validation score against the
    anchor's.  A candidate wins only if it beats the anchor by more than the
    4.7 resolution of the five-seed spread (docs/01 section 7.21).  If neither
    candidate wins, KEEP the anchor's rate.  Ties go to the anchor.

Writes ``learning_rate_decision.json`` next to the anchor run, which is what
the SLURM wrapper requires before it will run any increment.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_lib import Experiment, RUNS_ROOT, resolved_runtime_config, write_json  # noqa: E402

RESOLUTION = 4.7  # docs/01 section 7.21: the five-seed spread of the control
DOWNSTREAM_ARMS = ("m4_r07_a06_e09_t02opp_opponents", "m4_r07_a03_e09_t02_no_shaping",
                   "m4_r07_a06_e10_t02_bc", "m4_r08_a06_e09_t02_dueling")
# The arms this decision is between, and the rate each is supposed to carry.
# Checked rather than assumed: a decision computed over the wrong runs is worse
# than no decision, because it looks like one.
EXPECTED_RATES = {"anchor": 2.5e-4, "lr1e4": 1e-4, "lr5e4": 5e-4}


def curve(run_dir: Path) -> list[tuple[int | None, float]]:
    summary = json.loads((run_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
    records = summary["evaluation_suites"]["primary"]["validation_checkpoint_curve"]
    return [(row["checkpoint_round"], row["metrics"]["score"]["mean"])
            for row in records if row["metrics"]["score"]["count"]]


def learning_rate_of(run_dir: Path) -> float:
    """The rate the run actually used, read from its own snapshot."""
    experiment = Experiment.load(run_dir / "experiment_config.snapshot.json")
    return resolved_runtime_config(experiment)["config"]["learning_rate"]


def describe(run_dir: Path) -> dict:
    points = curve(run_dir)
    if len(points) < 2:
        raise SystemExit(f"{run_dir.name}: needs at least two evaluated checkpoints, has {len(points)}")
    window = min(2, len(points) // 2)
    early = statistics.fmean(value for _, value in points[:window])
    late = statistics.fmean(value for _, value in points[-window:])
    return {
        "run": run_dir.name,
        "learning_rate": learning_rate_of(run_dir),
        "final_pooled_validation_score": late,
        "early_pooled_validation_score": early,
        "trend": late - early,
        "curve": [{"round": one, "score": value} for one, value in points],
    }


def verify(anchor_dir: Path) -> int:
    """Refuse unless the decision exists AND the downstream configs carry it.

    Printing the two numbers and continuing -- which is what the SLURM wrapper
    did -- is not a gate.  A decision naming 1e-4 while all four downstream
    configs say 5e-4 passed it, and so did four configs that named nothing and
    silently fell back to the route default.  Both entry points call this, so
    there is one implementation of the rule rather than two that can drift.
    """
    path = anchor_dir / "learning_rate_decision.json"
    if not path.is_file():
        print(f"REFUSED: no step-size decision at {path}", file=sys.stderr)
        print("  Produce it with decide_learning_rate.py ... --apply", file=sys.stderr)
        return 1
    try:
        decision = json.loads(path.read_text(encoding="utf-8"))
        decided = float(decision["decided_learning_rate"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"REFUSED: {path} is not a readable decision ({exc})", file=sys.stderr)
        return 1

    configured = {}
    for arm in DOWNSTREAM_ARMS:
        config_path = Path("experiments") / f"{arm}.json"
        if not config_path.is_file():
            print(f"REFUSED: missing downstream config {config_path}", file=sys.stderr)
            return 1
        # The RESOLVED rate, not the raw field: an arm that names nothing falls
        # back to the route default, and that fallback has to be checked too.
        configured[arm] = resolved_runtime_config(
            Experiment.load(config_path))["config"]["learning_rate"]

    wrong = {arm: rate for arm, rate in configured.items() if rate != decided}
    if wrong:
        print(f"REFUSED: the decision is {decided:.1e} but these arms resolve to something else:",
              file=sys.stderr)
        for arm, rate in sorted(wrong.items()):
            print(f"    {rate:.1e}  {arm}", file=sys.stderr)
        print("  Set agent.learning_rate in every downstream config and commit.", file=sys.stderr)
        return 1
    print(f"step size verified: decision and all {len(DOWNSTREAM_ARMS)} downstream arms are {decided:.1e}")
    return 0


def check_identity(anchor: dict, candidates: list[dict]) -> None:
    """Refuse to decide between runs that are not the arms of this comparison.

    A decision computed over a mislabelled or half-finished set of runs still
    writes a decision file, and everything downstream then trusts it.
    """
    seen = {"anchor": anchor["learning_rate"],
            **{one["run"].split("_")[1]: one["learning_rate"] for one in candidates}}
    for label, expected in EXPECTED_RATES.items():
        actual = seen.get(label)
        if actual is None:
            raise SystemExit(f"missing the {label} arm; this comparison needs all of "
                             f"{sorted(EXPECTED_RATES)}")
        if abs(actual - expected) > 1e-12:
            raise SystemExit(f"the {label} arm resolves to {actual:.1e}, expected {expected:.1e}; "
                             "these are not the arms this decision is defined over")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--anchor", required=True, help="the anchor run directory")
    parser.add_argument("--candidate", action="append", default=[],
                        help="a step-size arm; repeat the flag")
    parser.add_argument("--apply", action="store_true",
                        help="write the decision file the SLURM wrapper requires")
    parser.add_argument("--verify", action="store_true",
                        help="check an existing decision against the downstream configs and exit")
    arguments = parser.parse_args()

    if arguments.verify:
        path = Path(arguments.anchor)
        raise SystemExit(verify(path if path.is_dir() else RUNS_ROOT / path.name))

    def resolve(name: str) -> Path:
        path = Path(name)
        return path if path.is_dir() else RUNS_ROOT / path.name

    anchor = describe(resolve(arguments.anchor))
    candidates = [describe(resolve(one)) for one in arguments.candidate]
    check_identity(anchor, candidates)

    print(f"{'run':34s} {'lr':>9s} {'final':>8s} {'trend':>8s} {'vs anchor':>10s}")
    print(f"{anchor['run']:34s} {anchor['learning_rate']:9.1e} "
          f"{anchor['final_pooled_validation_score']:8.2f} {anchor['trend']:+8.2f} {'--':>10s}")
    ranked = []
    for candidate in candidates:
        margin = candidate["final_pooled_validation_score"] - anchor["final_pooled_validation_score"]
        candidate["margin_over_anchor"] = margin
        candidate["distinguishable"] = margin > RESOLUTION
        ranked.append(candidate)
        print(f"{candidate['run']:34s} {candidate['learning_rate']:9.1e} "
              f"{candidate['final_pooled_validation_score']:8.2f} {candidate['trend']:+8.2f} "
              f"{margin:+10.2f}{'  WINS' if candidate['distinguishable'] else ''}")

    winners = [one for one in ranked if one["distinguishable"]]
    if winners:
        chosen = max(winners, key=lambda one: one["margin_over_anchor"])
        rate, reason = chosen["learning_rate"], (
            f"{chosen['run']} beats the anchor by {chosen['margin_over_anchor']:+.2f}, "
            f"above the {RESOLUTION} resolution")
    else:
        chosen, rate = anchor, anchor["learning_rate"]
        reason = (f"no candidate beats the anchor by more than {RESOLUTION}; the pre-registered "
                  "rule keeps the anchor's rate")

    print(f"\nDECISION: {rate:.1e}\n  {reason}")
    decision = {
        "decided_learning_rate": rate,
        "rule": (f"argmax over candidates whose final pooled VALIDATION score beats the anchor's "
                 f"by more than {RESOLUTION}; otherwise keep the anchor"),
        "resolution": RESOLUTION,
        "reason": reason,
        "anchor": anchor,
        "candidates": ranked,
    }
    if arguments.apply:
        path = resolve(arguments.anchor) / "learning_rate_decision.json"
        write_json(path, decision)
        print(f"\nwrote {path}")
        print("Now set agent.learning_rate to this value in the four downstream configs and COMMIT,")
        print("then re-run the line with --lr-settled.")
    else:
        print("\n(dry run; pass --apply to record the decision)")


if __name__ == "__main__":
    main()
