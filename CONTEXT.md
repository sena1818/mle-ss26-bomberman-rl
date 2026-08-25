# Bomberman RL Experiment Context

This glossary fixes the language used to describe Bomberman RL experiments. It separates an agent design from its training objective and from the tasks used to train it.

## Experiment dimensions

**Route (R)**:
The learning-agent design: state representation, value/policy model, and learning algorithm. A route is not a reward version or a curriculum.
_Avoid_: experiment, task

**Reward version (A)**:
The versioned mapping from game events to training feedback. It changes what a route is trained to optimise, but never changes the official evaluation score.
_Avoid_: score version, global reward switch

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
A pre-declared evaluation condition that a candidate must satisfy before it is used as the input to a harder curriculum stage.
_Avoid_: promotion by a single lucky game
