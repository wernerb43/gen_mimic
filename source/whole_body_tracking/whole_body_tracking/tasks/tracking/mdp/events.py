from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)


def apply_force_in_direction_during_contact(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    direction: tuple[float, float, float],
    command_name: str,
    motion_index: int,
    force_duration_s: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Apply an external force in the body frame of specified bodies, triggered by the contact phase.

    When an env enters the contact phase, a force magnitude is sampled and persisted
    for ``force_duration_s`` seconds. The force continues even after the phase window
    ends, ensuring the impulse has time to take effect.

    Args:
        force_range: Min / max force magnitude (N).
        direction: 3-tuple giving the desired force direction in the body's local frame
            (will be normalised).
        command_name: Name of the command term to read phase info from.
        motion_index: Which motion's phase window to use.
        force_duration_s: How long (seconds) the force persists once triggered.
        asset_cfg: Asset and body specification (use ``body_names`` to pick links).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)
    num_bodies = len(asset_cfg.body_ids) if asset_cfg.body_ids != slice(None) else asset.num_bodies
    num_envs = env.scene.num_envs

    # Lazily initialise persistent state on first call
    state = apply_force_in_direction_during_contact.__dict__
    if "_steps_remaining" not in state or state["_steps_remaining"].shape[0] != num_envs:
        state["_steps_remaining"] = torch.zeros(num_envs, device=asset.device, dtype=torch.long)
        state["_magnitudes"] = torch.zeros(num_envs, device=asset.device)

    steps_remaining = state["_steps_remaining"]
    cached_magnitudes = state["_magnitudes"]

    duration_steps = max(1, int(force_duration_s / env.cfg.sim.dt))

    # Compute current phase (0-1) for each env
    motion_cfg = command.motion_configs[motion_index]
    time_step_total = command.motion_loaders[motion_index].time_step_total
    phase = command.time_steps.float() / max(time_step_total, 1)

    # Determine which envs just entered the contact phase (trigger new force)
    in_phase = (phase >= motion_cfg.target_phase_start) & (phase <= motion_cfg.target_phase_end)
    correct_motion = command.which_motion == motion_index
    newly_triggered = in_phase & correct_motion & (steps_remaining == 0)

    # Sample and store magnitudes for newly triggered envs
    if newly_triggered.any():
        new_ids = torch.where(newly_triggered)[0]
        cached_magnitudes[new_ids] = torch.empty(len(new_ids), device=asset.device).uniform_(*force_range)
        steps_remaining[new_ids] = duration_steps

    # Normalise direction (body-frame)
    dir_t = torch.tensor(direction, device=asset.device, dtype=torch.float32)
    dir_t = dir_t / dir_t.norm()

    # Build force for all envs (active = steps_remaining > 0)
    active = steps_remaining > 0
    magnitudes = torch.where(active, cached_magnitudes, torch.zeros_like(cached_magnitudes))

    forces = dir_t.unsqueeze(0).unsqueeze(0) * magnitudes.unsqueeze(-1).unsqueeze(-1)
    forces = forces.expand(num_envs, num_bodies, 3).clone()
    torques = torch.zeros_like(forces)

    # Decrement counters
    steps_remaining[active] -= 1

    asset.set_external_force_and_torque(
        forces, torques, env_ids=torch.arange(num_envs, device=asset.device), body_ids=asset_cfg.body_ids
    )
