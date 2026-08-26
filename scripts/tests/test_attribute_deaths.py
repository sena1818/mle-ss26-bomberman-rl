"""Tests for splitting deaths into self-inflicted and opponent-inflicted.

The counting looks trivial and is not.  ``evaluate_explosions`` iterates over
every live explosion, so one death can produce two log lines when blasts
overlap; counting lines gave a survival rate below zero before this was fixed.
The overlap case is pinned here because nothing else would catch it.
"""

from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import attribute_deaths  # noqa: E402


def make_job(lines: list[str], suicides: int | None = None, compressed: bool = False) -> Path:
    job = Path(tempfile.mkdtemp())
    text = "".join(line + "\n" for line in lines)
    if compressed:
        with gzip.open(job / "framework_game.log.gz", "wt", encoding="utf-8") as handle:
            handle.write(text)
    else:
        (job / "framework_game.log").write_text(text, encoding="utf-8")
    if suicides is not None:
        (job / "official_stats.json").write_text(
            json.dumps({"by_agent": {"research_agent": {"suicides": suicides}}}), encoding="utf-8")
    return job


ROUND = "INFO: STARTING ROUND #{}"
OWN = "INFO: Agent <research_agent> blown up by own bomb"
BY = "INFO: Agent <research_agent> blown up by agent <rule_based_agent_{}>'s bomb"
OTHER_VICTIM = "INFO: Agent <rule_based_agent_0> blown up by own bomb"


class AttributionTest(unittest.TestCase):
    def test_a_round_with_both_kinds_of_line_is_one_death(self):
        job = make_job([ROUND.format(1), OWN, BY.format(2)])
        result = attribute_deaths.attribute_job(job, "research_agent")
        self.assertEqual(result["rounds"], 1)
        self.assertEqual(result["caught_in_overlapping_blasts"], 1)
        self.assertEqual(result["died_to_own_bomb_only"], 0)
        self.assertEqual(result["killed_by_others_only"], 0)
        self.assertEqual(result["survived"], 0)

    def test_categories_always_add_up_to_the_round_count(self):
        job = make_job([
            ROUND.format(1), OWN,
            ROUND.format(2), BY.format(1),
            ROUND.format(3), OWN, BY.format(0),
            ROUND.format(4),
        ])
        r = attribute_deaths.attribute_job(job, "research_agent")
        total = (r["died_to_own_bomb_only"] + r["killed_by_others_only"]
                 + r["caught_in_overlapping_blasts"] + r["survived"])
        self.assertEqual(total, r["rounds"])
        self.assertEqual(r["survived"], 1)

    def test_other_agents_deaths_are_ignored(self):
        job = make_job([ROUND.format(1), OTHER_VICTIM, ROUND.format(2), OTHER_VICTIM])
        r = attribute_deaths.attribute_job(job, "research_agent")
        self.assertEqual(r["survived"], 2)
        self.assertEqual(r["died_to_own_bomb_only"], 0)

    def test_own_bomb_deaths_match_the_official_suicide_definition(self):
        # The framework raises KILLED_SELF for an overlapping blast too, so the
        # acceptance test has to count those in, or it would fire spuriously.
        job = make_job([ROUND.format(1), OWN, ROUND.format(2), OWN, BY.format(1)], suicides=2)
        r = attribute_deaths.attribute_job(job, "research_agent")
        self.assertEqual(r["own_bomb_deaths"], attribute_deaths.official_suicides(job, "research_agent"))

    def test_two_opponents_sharing_one_kill_are_each_credited_a_half(self):
        job = make_job([ROUND.format(1), BY.format(0), BY.format(1)])
        r = attribute_deaths.attribute_job(job, "research_agent")
        self.assertEqual(r["killed_by_others_only"], 1)
        self.assertEqual(set(r["by_killer"].values()), {0.5})

    def test_a_compressed_log_reads_the_same(self):
        lines = [ROUND.format(1), OWN, ROUND.format(2)]
        plain = attribute_deaths.attribute_job(make_job(lines), "research_agent")
        packed = attribute_deaths.attribute_job(make_job(lines, compressed=True), "research_agent")
        self.assertEqual(plain["died_to_own_bomb_only"], packed["died_to_own_bomb_only"])
        self.assertEqual(plain["survived"], packed["survived"])

    def test_a_missing_log_yields_no_rounds_rather_than_an_error(self):
        r = attribute_deaths.attribute_job(Path(tempfile.mkdtemp()), "research_agent")
        self.assertEqual(r["rounds"], 0)
        self.assertEqual(r["survived"], 0)


if __name__ == "__main__":
    unittest.main()
