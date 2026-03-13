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

    Weights stay at their initial values for ``hold_steps`` learning iterations,
    then linearly decay to zero over the next ``decay_steps`` iterations. An
    iteration is ``common_step_counter / num_envs``.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        hold_steps: Number of iterations to hold initial weights before starting decay.
        decay_steps: Number of iterations over which to linearly decay weights to zero.
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
    iteration = env.common_step_counter / env.num_envs

    if iteration <= hold_steps:
        scale = 1.0
    else:
        progress = min((iteration - hold_steps) / decay_steps, 1.0)
        scale = 1.0 - progress

    result: dict[str, float] = {}
    for name, init_w in initial_weights.items():
        new_weight = init_w * scale
        cfg = reward_mgr.get_term_cfg(name)
        cfg.weight = new_weight
        reward_mgr.set_term_cfg(name, cfg)
        result[name] = new_weight

    return result
