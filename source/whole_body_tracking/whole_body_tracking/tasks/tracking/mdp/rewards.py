from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_error_magnitude

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand, TargetPositionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


def target_position_error_exp(env: ManagerBasedRLEnv, target_command_name: str, motion_command_name: str, std: float) -> torch.Tensor:
    """Exponential reward for target position tracking error."""
    target_command: TargetPositionCommand = env.command_manager.get_term(target_command_name)
    motion_command: MotionCommand = env.command_manager.get_term(motion_command_name)
    error = torch.sum(torch.square(target_command.target_position_w - target_command.source_body_pos_w), dim=-1)
    reward = torch.exp(-error / std**2)
    
    phase = motion_command.time_steps/motion_command.motion.time_step_total

    active = (phase >= target_command.target_phase_start[0]) & (phase <= target_command.target_phase_end[0])
    # print(active)
    return reward * active.float()

def target_orientation_error_exp(env: ManagerBasedRLEnv, target_command_name: str, motion_command_name: str, std: float) -> torch.Tensor:
    """Exponential reward for target orientation tracking error."""
    target_command: TargetPositionCommand = env.command_manager.get_term(target_command_name)
    motion_command: MotionCommand = env.command_manager.get_term(motion_command_name)
    error = quat_error_magnitude(target_command.target_orientation_w, target_command.source_body_quat_w) ** 2
    reward = torch.exp(-error / std**2)
    
    phase = motion_command.time_steps/motion_command.motion.time_step_total

    active = (phase >= target_command.target_phase_start[0]) & (phase <= target_command.target_phase_end[0])

    return reward * active.float()


def target_orientation_axis_alignment_error_exp(
    env: ManagerBasedRLEnv,
    target_command_name: str,
    motion_command_name: str,
    std: float,
    axis: str,
) -> torch.Tensor:
    """Exponential reward for alignment of a single axis between source and target orientations.

    The reward is invariant to rotations around the chosen axis.
    """
    target_command: TargetPositionCommand = env.command_manager.get_term(target_command_name)
    motion_command: MotionCommand = env.command_manager.get_term(motion_command_name)

    axis_map = {
        "x": torch.tensor([1.0, 0.0, 0.0], device=target_command.device),
        "y": torch.tensor([0.0, 1.0, 0.0], device=target_command.device),
        "z": torch.tensor([0.0, 0.0, 1.0], device=target_command.device),
    }
    if axis not in axis_map:
        raise ValueError(f"Invalid axis '{axis}'. Expected one of: 'x', 'y', 'z'.")

    axis_vec = axis_map[axis].expand(target_command.num_envs, 3)

    target_axis_w = quat_apply(target_command.target_orientation_w, axis_vec)
    source_axis_w = quat_apply(target_command.source_body_quat_w, axis_vec)

    dot = torch.sum(target_axis_w * source_axis_w, dim=-1).clamp(-1.0, 1.0)
    error = 1.0 - dot
    reward = torch.exp(-(error**2) / std**2)

    phase = motion_command.time_steps / motion_command.motion.time_step_total
    active = (phase >= target_command.target_phase_start[0]) & (phase <= target_command.target_phase_end[0])

    return reward * active.float()


def multi_motion_target_position_error_exp(env: ManagerBasedRLEnv, target_command_name: str, std: float, motion_to_reward: int) -> torch.Tensor:
    """Exponential reward for target position tracking error for a specific motion."""
    command = env.command_manager.get_term(target_command_name)

    which_motion = command.which_motion
    correct_motion = (which_motion == motion_to_reward)

    # Get source body positions based on the current motion
    source_body_positions = torch.zeros(command.num_envs, 3, device=command.device)
    for i in range(len(command.motion_configs)):
        mask = command.which_motion == i
        if torch.any(mask):
            source_body_positions[mask] = command.robot.data.body_pos_w[mask, command.source_body_indices[i]]

    error = torch.sum(torch.square(command.target_position_w - source_body_positions), dim=-1)
    reward = torch.exp(-error / std**2)
    
    # Gather time_step_total for each motion in the batch
    time_step_totals = torch.tensor(
        [loader.time_step_total for loader in command.motion_loaders],
        device=command.device,
        dtype=torch.float32
    )
    time_step_total = time_step_totals[which_motion]
    phase = command.time_steps / time_step_total

    active = (phase >= command.target_phase_start) & (phase <= command.target_phase_end)
    return reward * active.float() * correct_motion.float()


def multi_motion_target_orientation_error_exp(env: ManagerBasedRLEnv, target_command_name: str, std: float, motion_to_reward: int) -> torch.Tensor:
    """Exponential reward for target orientation tracking error for a specific motion."""
    command = env.command_manager.get_term(target_command_name)
    
    which_motion = command.which_motion
    correct_motion = (which_motion == motion_to_reward)
    
    # Get source body orientations based on the current motion
    source_body_quats = torch.zeros(command.num_envs, 4, device=command.device)
    source_body_quats[:, 0] = 1.0  # Initialize with identity quaternion
    for i in range(len(command.motion_configs)):
        mask = command.which_motion == i
        if torch.any(mask):
            source_body_quats[mask] = command.robot.data.body_quat_w[mask, command.source_body_indices[i]]
    
    error = quat_error_magnitude(command.target_orientation_w, source_body_quats) ** 2
    reward = torch.exp(-error / std**2)
    
    # Gather time_step_total for each motion in the batch
    time_step_totals = torch.tensor(
        [loader.time_step_total for loader in command.motion_loaders],
        device=command.device,
        dtype=torch.float32
    )
    time_step_total = time_step_totals[which_motion]
    phase = command.time_steps / time_step_total

    active = (phase >= command.target_phase_start) & (phase <= command.target_phase_end)
    return reward * active.float() * correct_motion.float()

def multi_motion_target_orientation_axis_alignment_error_exp(
    env: ManagerBasedRLEnv,
    target_command_name: str,
    std: float,
    axis: str,
    motion_to_reward: int,
) -> torch.Tensor:
    """Exponential reward for alignment of a single axis between source and target orientations.

    The reward is invariant to rotations around the chosen axis.
    """
    command = env.command_manager.get_term(target_command_name)

    which_motion = command.which_motion
    correct_motion = (which_motion == motion_to_reward)

    # Get source body orientations based on the current motion
    source_body_quats = torch.zeros(command.num_envs, 4, device=command.device)
    source_body_quats[:, 0] = 1.0  # Initialize with identity quaternion
    for i in range(len(command.motion_configs)):
        mask = command.which_motion == i
        if torch.any(mask):
            source_body_quats[mask] = command.robot.data.body_quat_w[mask, command.source_body_indices[i]]

    axis_map = {
        "x": torch.tensor([1.0, 0.0, 0.0], device=command.device),
        "y": torch.tensor([0.0, 1.0, 0.0], device=command.device),
        "z": torch.tensor([0.0, 0.0, 1.0], device=command.device),
    }
    if axis not in axis_map:
        raise ValueError(f"Invalid axis '{axis}'. Expected one of: 'x', 'y', 'z'.")

    axis_vec = axis_map[axis].expand(command.num_envs, 3)

    target_axis_w = quat_apply(command.target_orientation_w, axis_vec)
    source_axis_w = quat_apply(source_body_quats, axis_vec)

    dot = torch.sum(target_axis_w * source_axis_w, dim=-1).clamp(-1.0, 1.0)
    error = 1.0 - dot
    reward = torch.exp(-(error**2) / std**2)

    time_step_totals = torch.tensor(
        [loader.time_step_total for loader in command.motion_loaders],
        device=command.device,
        dtype=torch.float32
    )
    time_step_total = time_step_totals[which_motion]
    phase = command.time_steps / time_step_total

    active = (phase >= command.target_phase_start) & (phase <= command.target_phase_end)
    return reward * active.float() * correct_motion.float()


def action_rate_l2_clamped(env: ManagerBasedRLEnv, max_value: float = 1.0) -> torch.Tensor:
    """Penalize the rate of change of actions using L2 squared kernel, clamped to a max value."""
    raw = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    return torch.clamp(raw, min=-max_value, max=max_value)