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


def ramp_target_pos_variance(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    reward_term_name: str,
    reward_threshold: float,
    scale_increment: float,
    ema_alpha: float,
) -> dict[str, float]:
    """Increase target position Gaussian std when the policy achieves high reward.

    Tracks an exponential moving average of the mean (unweighted) target position
    reward across all environments.  When the EMA exceeds ``reward_threshold``
    the std scale is bumped up by ``scale_increment`` (clamped to [0, 1]).
    When the EMA is below the threshold the scale stays where it is, giving the
    policy time to adapt before the next increase.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        command_name: Name of the command term whose ``target_pos_std_scale`` to modify.
        reward_term_name: Name of the reward term to track (the unweighted value is used).
        reward_threshold: EMA reward level above which the variance is increased.
        scale_increment: How much to increase the std scale each time the threshold is exceeded.
        ema_alpha: Smoothing factor for the exponential moving average (smaller = smoother).

    Returns:
        A dict with the current std scale and EMA reward (for logging).
    """
    import torch

    command_term = env.command_manager.get_term(command_name)
    reward_mgr = env.reward_manager

    # Get the unweighted reward for the target position term this step
    term_idx = reward_mgr._term_names.index(reward_term_name)
    term_cfg = reward_mgr._term_cfgs[term_idx]
    # _step_reward stores weight*value; divide out the weight to get raw reward
    if term_cfg.weight != 0.0:
        raw_reward = reward_mgr._step_reward[:, term_idx] / term_cfg.weight
    else:
        raw_reward = torch.zeros(env.num_envs, device=env.device)
    mean_reward = raw_reward.mean().item()

    # Initialize EMA state on first call
    if not hasattr(command_term, "_target_pos_reward_ema"):
        command_term._target_pos_reward_ema = mean_reward

    # Update EMA
    ema = command_term._target_pos_reward_ema
    ema = ema * (1.0 - ema_alpha) + mean_reward * ema_alpha
    command_term._target_pos_reward_ema = ema

    # Increase scale when performance is above threshold
    scale = command_term.target_pos_std_scale
    if ema >= reward_threshold:
        scale = min(scale + scale_increment, 1.0)
    command_term.target_pos_std_scale = scale

    return {"target_pos_std_scale": scale, "target_pos_reward_ema": ema}
