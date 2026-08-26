"""Tests for the n-step return window.

Two invariants matter more than the arithmetic.  First, every transition must
be emitted exactly once, or the update count per round stops matching the step
count and the terminal-transition regression check goes blind.  Second, an
n-step target has to be bootstrapped with ``gamma ** n``, not ``gamma``; getting
that wrong quietly rescales every value estimate.
"""

from __future__ import annotations

import unittest

import numpy as np

from agent_code.research_agent.learners.base import Transition
from agent_code.research_agent.runtime.transitions import NStepAssembler


DISCOUNT = 0.95


def step(index: int, reward: float, *, terminal: bool = False) -> Transition:
    return Transition(
        state=np.full(2, float(index), dtype=np.float32),
        action_index=index % 6,
        reward=reward,
        next_state=None if terminal else np.full(2, float(index + 1), dtype=np.float32),
        next_legal_mask=None if terminal else np.ones(6, dtype=bool),
        terminal=terminal,
    )


class NStepAssemblerTest(unittest.TestCase):
    def test_one_step_emits_immediately_and_unchanged(self):
        assembler = NStepAssembler(1, DISCOUNT)
        emitted = assembler.push(step(0, 1.0))
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0].reward, 1.0)
        self.assertEqual(emitted[0].n_step, 1)
        self.assertFalse(emitted[0].terminal)
        self.assertEqual(assembler.pending_count(), 0)

    def test_a_full_window_returns_the_discounted_sum_and_the_last_next_state(self):
        assembler = NStepAssembler(3, DISCOUNT)
        self.assertEqual(assembler.push(step(0, 1.0)), [])
        self.assertEqual(assembler.push(step(1, 2.0)), [])
        emitted = assembler.push(step(2, 4.0))
        self.assertEqual(len(emitted), 1)
        self.assertAlmostEqual(emitted[0].reward, 1.0 + DISCOUNT * 2.0 + DISCOUNT ** 2 * 4.0, places=6)
        self.assertEqual(emitted[0].n_step, 3)
        # The head's state, but the tail's successor: that is what makes the
        # bootstrap term gamma**3 rather than gamma.
        np.testing.assert_array_equal(emitted[0].state, np.full(2, 0.0, dtype=np.float32))
        np.testing.assert_array_equal(emitted[0].next_state, np.full(2, 3.0, dtype=np.float32))

    def test_a_terminal_step_drains_every_suffix_as_a_complete_return(self):
        assembler = NStepAssembler(5, DISCOUNT)
        assembler.push(step(0, 1.0))
        assembler.push(step(1, 2.0))
        emitted = assembler.push(step(2, -1.0, terminal=True))
        self.assertEqual([transition.n_step for transition in emitted], [3, 2, 1])
        self.assertTrue(all(transition.terminal for transition in emitted))
        self.assertTrue(all(transition.next_state is None for transition in emitted))
        self.assertAlmostEqual(emitted[0].reward, 1.0 + DISCOUNT * 2.0 + DISCOUNT ** 2 * -1.0, places=6)
        self.assertAlmostEqual(emitted[-1].reward, -1.0, places=6)
        self.assertEqual(assembler.pending_count(), 0)

    def test_a_truncated_round_flushes_shorter_bootstrapped_returns(self):
        assembler = NStepAssembler(5, DISCOUNT)
        assembler.push(step(0, 1.0))
        assembler.push(step(1, 2.0))
        emitted = assembler.flush()
        self.assertEqual([transition.n_step for transition in emitted], [2, 1])
        # Truncation is not termination: both still bootstrap.
        self.assertFalse(any(transition.terminal for transition in emitted))
        np.testing.assert_array_equal(emitted[0].next_state, np.full(2, 2.0, dtype=np.float32))
        np.testing.assert_array_equal(emitted[1].next_state, np.full(2, 2.0, dtype=np.float32))

    def test_every_pushed_transition_is_emitted_exactly_once(self):
        for n_step in (1, 2, 3, 5, 8):
            for length in (1, 3, 7, 12):
                for terminal in (False, True):
                    with self.subTest(n_step=n_step, length=length, terminal=terminal):
                        assembler = NStepAssembler(n_step, DISCOUNT)
                        emitted = []
                        for index in range(length):
                            last = index == length - 1
                            emitted += assembler.push(step(index, 1.0, terminal=terminal and last))
                        emitted += assembler.flush()
                        self.assertEqual(len(emitted), length)
                        heads = [int(transition.state[0]) for transition in emitted]
                        self.assertEqual(sorted(heads), list(range(length)))

    def test_the_window_never_reaches_beyond_n_steps(self):
        assembler = NStepAssembler(3, DISCOUNT)
        for index in range(10):
            assembler.push(step(index, 1.0))
            self.assertLessEqual(assembler.pending_count(), 3)

    def test_only_one_step_transitions_may_be_pushed(self):
        assembler = NStepAssembler(3, DISCOUNT)
        with self.assertRaises(ValueError):
            assembler.push(Transition(np.zeros(2), 0, 0.0, None, None, True, n_step=2))

    def test_a_non_positive_window_is_refused(self):
        with self.assertRaises(ValueError):
            NStepAssembler(0, DISCOUNT)


if __name__ == "__main__":
    unittest.main()
