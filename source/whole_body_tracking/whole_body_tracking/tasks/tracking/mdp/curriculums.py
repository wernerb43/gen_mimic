"""Curriculum terms for the tracking MDP."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def decay_imitation_reward_weights(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    hold_steps: int,
    decay_steps: int,
    term_prefixes: tuple[str, ...] = ("motion_",),
) -> dict[str, float]:
    """Hold imitation reward weights constant, then linearly decay them to zero.

    Weights stay at their initial values for ``hold_steps``, then linearly decay
    to zero over the next ``decay_steps``. Total steps to reach zero weight is
    ``hold_steps + decay_steps``.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        hold_steps: Number of steps to hold initial weights before starting decay.
        decay_steps: Number of steps over which to linearly decay weights to zero.
        term_prefixes: Prefixes that identify which reward terms are imitation rewards.

    Returns:
        A dict mapping term names to their current weights (for logging).
    """
    reward_mgr = env.reward_manager

    # Lazily capture initial weights on first call
    if not hasattr(reward_mgr, "_imitation_initial_weights"):
        initial_weights: dict[str, float] = {}
        for name in reward_mgr._term_names:
            if any(name.startswith(prefix) for prefix in term_prefixes):
                cfg = reward_mgr.get_term_cfg(name)
                initial_weights[name] = cfg.weight
        reward_mgr._imitation_initial_weights = initial_weights

    initial_weights = reward_mgr._imitation_initial_weights
    step = env.common_step_counter

    if step <= hold_steps:
        scale = 1.0
    else:
        progress = min((step - hold_steps) / decay_steps, 1.0)
        scale = 1.0 - progress

    result: dict[str, float] = {}
    for name, init_w in initial_weights.items():
        new_weight = init_w * scale
        cfg = reward_mgr.get_term_cfg(name)
        cfg.weight = new_weight
        reward_mgr.set_term_cfg(name, cfg)
        result[name] = new_weight

    return result
