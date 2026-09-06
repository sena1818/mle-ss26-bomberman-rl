#!/usr/bin/env python3
"""Assemble the archive a teammate needs to write the report, and nothing else.

The full run directories are 59 GB, almost all of it per-action logs, and every
diagnostic that reads them has already been run -- their outputs are in
``diagnostics/`` and their conclusions are in ``docs/``. So this ships the
record and the reasoning rather than the raw telemetry: the narrative documents,
the experiment configs (whose ``_design_note`` and ``_predeclared_design_numbers``
are where a decision's justification was written down *before* it was measured),
the finished diagnostics, one consolidated results table, the code, the
submitted agent, and the handful of checkpoints the documents actually quote.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
from pathlib import Path

from experiment_lib import ROOT, git_provenance, write_json

IGNORED = shutil.ignore_patterns("__pycache__", "*.pyc", "artifacts", "logs", ".pytest_cache")
# Checkpoints the documents quote by name: the four ensemble members are the
# submission itself, and the rest are arms other sections compare against or
# frozen opponents the evaluation pool seats.
QUOTED_MODELS = "frozen_opponents"


def _leaderboard(results: list[dict]) -> str:
    """The report boards, best first -- the one table a reader starts from."""
    rows = [r for r in results if r.get("boards") == "report" and r.get("members") != "?"]
    rows.sort(key=lambda r: -r["mean_over_pools"]["score"])
    lines = [
        "# 对手代理池：报数棋盘 4007–4012 的排行榜",
        "",
        "五张对手桌等权，配对到棋盘种子。这组棋盘从未参与任何选择。",
        "完整逐桌数据见 `tournament_pool_results.json`。",
        "",
        "| 模型 | 成员 | score | coins | kills | suicides | coins_share |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        m = row["mean_over_pools"]
        lines.append(
            f"| {row['arm']} | {row['members']} | {m['score']:.4f} | {m['coins']:.4f} | "
            f"{m['kills']:.4f} | {m['suicides']:.4f} | {m['coins_share']:.4f} |")
    lines += ["", "选种棋盘 4001–4006 的同一张表在 JSON 里 (`boards: selection`)；",
              "两组棋盘的判定符号、量级、显著性一致，这是评估口径复现自己的证据。"]
    return "\n".join(lines) + "\n"


def _run_index() -> str:
    """Every archived run, generated from its own snapshot rather than by hand."""
    try:
        output = subprocess.check_output(
            ["python3", str(ROOT / "scripts" / "run_ledger.py"),
             "--runs-root", str(ROOT.parent / "run_archives"), "--markdown"],
            text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return "# 归档 run 索引\n\n生成失败；在仓库里跑 `scripts/run_ledger.py --runs-root ../run_archives --markdown`。\n"
    return "# 归档 run 索引\n\n从每个 run 自己的快照与 provenance 生成，不会与磁盘漂移。\n\n" + output


def build(destination: Path, submitted: str) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    shutil.copy2(ROOT / "HANDOVER.md", destination / "HANDOVER.md")
    shutil.copytree(ROOT / "docs", destination / "docs", ignore=IGNORED)
    shutil.copytree(ROOT / "experiments", destination / "experiments", ignore=IGNORED)
    shutil.copytree(ROOT / "diagnostics", destination / "diagnostics", ignore=IGNORED)
    shutil.copytree(ROOT / QUOTED_MODELS, destination / "models", ignore=IGNORED)
    shutil.copytree(ROOT / "agent_code" / submitted, destination / "submitted_agent" / submitted,
                    ignore=IGNORED)

    code = destination / "code"
    code.mkdir()
    shutil.copytree(ROOT / "agent_code" / "research_agent", code / "research_agent", ignore=IGNORED)
    shutil.copytree(ROOT / "scripts", code / "scripts", ignore=IGNORED)
    for name in ("CONTEXT.md", "settings.py"):
        shutil.copy2(ROOT / name, code / name)
    # The opponents the evaluation pool and the training tables are built from.
    opponents = code / "opponent_agents"
    opponents.mkdir()
    for name in ("rule_based_agent", "coin_collector_agent", "peaceful_agent",
                 "random_agent", "evader_agent", "rule_based_noisy_agent",
                 "frozen_agent", "frozen_agent_b", "frozen_agent_c"):
        source = ROOT / "agent_code" / name
        if source.is_dir():
            shutil.copytree(source, opponents / name, ignore=IGNORED)

    results = destination / "results"
    results.mkdir()
    pool_results = ROOT / "results_index" / "tournament_pool_results.json"
    if pool_results.is_file():
        shutil.copy2(pool_results, results / "tournament_pool_results.json")
        (results / "leaderboard.md").write_text(
            _leaderboard(json.loads(pool_results.read_text(encoding="utf-8"))), encoding="utf-8")
    (results / "run_index.md").write_text(_run_index(), encoding="utf-8")

    write_json(destination / "MANIFEST.json", {
        "submitted_agent": submitted,
        "contents": {
            "docs": "完整实验记录，按时间顺序，后节更正前节",
            "experiments": "实验配置；_design_note 与 _predeclared_design_numbers 是跑之前写下的预测",
            "diagnostics": "离线诊断的产物（逐动作日志已在服务器上消费完毕）",
            "results": "对手代理池的全部测量与排行榜",
            "code": "agent 实现、实验基础设施、以及所有对手 agent",
            "submitted_agent": "可直接放进 agent_code/ 运行",
            "models": "文档引用过的 checkpoint",
        },
        "deliberately_excluded": (
            "逐动作日志与框架日志，服务器上约 59 GB。所有依赖它们的诊断已经跑完，"
            "产物在 diagnostics/，结论在 docs/。需要重跑时按 run 名单独拉取。"),
        **git_provenance(),
    })
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--destination", type=Path, default=ROOT.parent / "bomberman_handover")
    parser.add_argument("--submitted", default="psycho_killer")
    parser.add_argument("--zip", action="store_true", help="Also write <destination>.zip")
    args = parser.parse_args()

    package = build(args.destination, args.submitted)
    total = sum(path.stat().st_size for path in package.rglob("*") if path.is_file())
    count = sum(1 for path in package.rglob("*") if path.is_file())
    print(f"{package}: {count} files, {total / 1024 / 1024:.1f} MiB")
    for child in sorted(package.iterdir()):
        size = (sum(p.stat().st_size for p in child.rglob("*") if p.is_file())
                if child.is_dir() else child.stat().st_size)
        print(f"  {child.name + ('/' if child.is_dir() else ''):22s} {size / 1024:9.0f} KiB")
    if args.zip:
        archive = shutil.make_archive(str(package), "zip", package.parent, package.name)
        print(f"\n{archive}  {Path(archive).stat().st_size / 1024 / 1024:.1f} MiB")


if __name__ == "__main__":
    main()
