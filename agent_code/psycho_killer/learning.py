"""Backward-compatible imports; learner adapters now live in ``learners/``."""

from .learners.online_q import OnlineQLearner


def choose_action_index(model, state, legal_mask, epsilon, generator):
    """Compatibility helper delegated to the shared online-Q learner."""
    learner = OnlineQLearner.__new__(OnlineQLearner)
    learner.model = model
    return learner.select_action(state, legal_mask, epsilon, generator)


def update_q_learning(model, config, state, action_index, reward, next_state, next_legal_mask):
    """Compatibility helper delegated to the shared online-Q learner."""
    from .learners.base import Transition

    return OnlineQLearner(config, model).observe(Transition(
        state=state,
        action_index=action_index,
        reward=reward,
        next_state=next_state,
        next_legal_mask=next_legal_mask,
        terminal=next_state is None,
    ))


__all__ = ("OnlineQLearner", "choose_action_index", "update_q_learning")
