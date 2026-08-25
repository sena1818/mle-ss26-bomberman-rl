# Bomberman RL Experiment Context

This glossary fixes the language used to describe Bomberman RL experiments. It separates an agent design from its training objective and from the tasks used to train it.

## Experiment dimensions

**Route (R)**:
The learning-agent design: state representation, value/policy model, and learning algorithm. A route is not a reward version or a curriculum.
_Avoid_: experiment, task

**Reward version (A)**:
The versioned mapping from game events to training feedback. It changes what a route is trained to optimise, but never changes the official evaluation score.
_Avoid_: score version, global reward switch

**Exploration schedule (E)**:
The training-time `epsilon` value and its schedule over rounds. It is a separate dimension from the route because it bounds what the agent can physically observe: with a zero-slack four-step bomb escape, `epsilon` caps the achievable escape success rate at `(1 - 0.75 * epsilon)^4`, independently of how good the greedy policy is.
_Avoid_: route hyperparameter, tuning knob

**Curriculum (C)**:
The ordered or mixed distribution of training tasks used by one experiment. `C` means curriculum, not the `classic` scenario.
_Avoid_: Classic version

**Task (T)**:
One concrete environment condition, including scenario and opponent setup, such as solo Coin Heaven or solo Classic. A curriculum may contain several tasks.
_Avoid_: route, reward

**Baseline**:
A frozen reference experiment kept unchanged so that later changes have a fair comparison.
_Avoid_: current best model

**Ablation**:
A controlled comparison in which one declared experiment dimension changes while the others remain fixed.
_Avoid_: tuning run

## Outcomes

**Training reward**:
The numeric feedback used to update an agent during training. It may include auxiliary signals and is scoped to one reward version.
_Avoid_: official score

**Official score**:
The score awarded by the unmodified game rules: one point per collected coin and five points per killed opponent. It is the primary evaluation outcome.
_Avoid_: training reward

**Capability gate**:
A pre-declared evaluation condition that a candidate must satisfy before it is used as the input to a harder curriculum stage. A gate must always contain at least one *activity* metric (`bomb_rate`, `crates_per_round`, `wait_fraction`), because survival metrics alone cannot separate "played well" from "froze".
_Avoid_: promotion by a single lucky game

**Feasibility threshold (`p*`)**:
The escape success rate a reward version requires before dropping a bomb has positive expected value: `p* = (D - G) / (D + w_safe)`. Computed before a run from pilot-averaged trajectory statistics, and compared afterwards against the measured `p_hat`. It is a mean-field design approximation, not a proof of optimality: when `p*` exceeds `p_ref(epsilon)` by well beyond the model's error, expect the learned policy to degenerate towards waiting.
_Avoid_: survival rate, suicide rate

**Activity metric**:
An evaluation metric that measures whether the greedy policy does anything at all: `bomb_rate`, `wait_fraction`, `crates_per_round`, `distinct_cells`. Most come from the unmodified framework's own stats (`bombs`/`crates`/`moves`/`invalid` at job level only; `wait_fraction` as a residual); `distinct_cells` and an exact safe-bomb rate need extra instrumentation in our own agent artifact.
_Avoid_: training reward, auxiliary score
