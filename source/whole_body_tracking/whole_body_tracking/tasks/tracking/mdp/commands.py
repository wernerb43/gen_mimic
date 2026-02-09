from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""The position command: commanded position of a single body (e.g., end effector or base for throwing/catching)"""
class TargetPositionCommand(CommandTerm):
    cfg: TargetPositionCommandCfg

    def __init__(self, cfg: TargetPositionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        print('body names:')
        print(self.robot.body_names)
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.target_body_index = self.robot.body_names.index(self.cfg.target_body_name)
        
        # Target position in world frame (this is the one we want to use for plotting and for rewards)
        self.target_position_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Target position in the body frame (this is the one we want to send as observation)
        self.target_position_b = torch.zeros(self.num_envs, 3, device=self.device)

        # Target time window in motion timesteps (frame indices)
        self.target_phase_start = torch.zeros(self.num_envs, device=self.device)
        self.target_phase_end = torch.zeros(self.num_envs, device=self.device)
        
        # Initialize metrics
        self.metrics["error_target_pos"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Returns the target position in body frame as the command observation."""
        # Transform target position from world frame to body frame
        anchor_pos_w = self.robot.data.body_pos_w[:, self.robot_anchor_body_index]
        anchor_quat_w = self.robot.data.body_quat_w[:, self.robot_anchor_body_index]
        anchor_quat_inv = quat_inv(anchor_quat_w)
        self.target_position_b = quat_apply(anchor_quat_inv, self.target_position_w - anchor_pos_w)
        return self.target_position_b

    @property
    def target_body_pos_w(self) -> torch.Tensor:
        """Current position of the target body in world frame."""
        return self.robot.data.body_pos_w[:, self.target_body_index]
    
    def _update_metrics(self):
        """Update tracking error: distance between target body position and target position."""
        self.metrics["error_target_pos"] = torch.norm(self.target_body_pos_w - self.target_position_w, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample target position with random values within specified range."""
        if len(env_ids) == 0:
            return
        
        # Sample target position within specified range
        range_list = [self.cfg.target_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=self.device)
        
        # Set target position in world frame relative to robot anchor body
        anchor_pos_w = self.robot.data.body_pos_w[env_ids, self.robot_anchor_body_index]
        self.target_position_w[env_ids] = rand_samples + anchor_pos_w

        # Sample target time window (motion timesteps)
        t_start = sample_uniform(
            self.cfg.target_phase_start_range[0],
            self.cfg.target_phase_start_range[1],
            (len(env_ids),),
            device=self.device,
        )
        t_end = sample_uniform(
            self.cfg.target_phase_end_range[0],
            self.cfg.target_phase_end_range[1],
            (len(env_ids),),
            device=self.device,
        )
        t_end = torch.maximum(t_end, t_start)
        self.target_phase_start[env_ids] = t_start
        self.target_phase_end[env_ids] = t_end

    def _update_command(self):
        """Update command each step - no changes needed for static targets."""
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Enable/disable debug visualization."""
        if debug_vis:
            if not hasattr(self, "target_visualizer"):
                # Create sphere marker for target position visualization
                marker_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/target_position",
                    markers={
                        "sphere": sim_utils.SphereCfg(
                            radius=0.05,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
                        )
                    },
                )
                self.target_visualizer = VisualizationMarkers(marker_cfg)
            self.target_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_visualizer"):
                self.target_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Callback for debug visualization - draw sphere at target position."""
        if not self.robot.is_initialized:
            return
        # Visualize target position as a sphere (no orientation needed)
        # Use identity rotation for the visualization
        identity_quat = torch.zeros(self.num_envs, 4, device=self.device)
        identity_quat[:, 0] = 1.0
        self.target_visualizer.visualize(self.target_position_w, identity_quat)
        print('Visualizing target at position:', self.target_position_w.cpu().numpy())

@configclass
class TargetPositionCommandCfg(CommandTermCfg):
    """Configuration for the target position command."""

    class_type: type = TargetPositionCommand

    asset_name: str = MISSING
    """Name of the robot asset in the scene."""

    anchor_body_name: str = MISSING
    """Name of the anchor body used as reference for sampling target positions."""

    target_body_name: str = MISSING
    """Name of the body/link that should reach the target position (e.g., 'left_hand', 'right_hand')."""

    target_range: dict[str, tuple[float, float]] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
    """Range for sampling target positions (x, y, z) in meters."""

    target_phase_start_range: tuple[float, float] = (0.0, 0.0)
    """Range for sampling target phase window start in motion timesteps."""

    target_phase_end_range: tuple[float, float] = (1.0, 1.0)
    """Range for sampling target phase window end in motion timesteps."""




class MultiTargetMotionCommand(CommandTerm):
    cfg: MultiTargetMotionCommandCfg

    def __init__(self, cfg: MultiTargetMotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        # NOTE: command_manager is not available during construction.
        # If you need to reference another command term, resolve it lazily after managers are initialized.
        self.env = env

        self.motions = []
        for motion_file in self.cfg.motion_files:
            self.motions.append(MotionLoader(motion_file, self.body_indexes, device=self.device))
        self.which_motion = torch.zeros(self.num_envs, dtype=torch.long, device=self.device) # which motion to track for each env

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # self.bin_count = int(self.motions[0].time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1

        self.bin_counts = [int(self.motions[i].time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1 for i in range(len(self.motions))]
        self.bin_failed_counts = [torch.zeros(self.bin_counts[i], dtype=torch.float, device=self.device) for i in range(len(self.motions))]
        self._current_bin_failed = [torch.zeros(self.bin_counts[i], dtype=torch.float, device=self.device) for i in range(len(self.motions))]

        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
    

    def _gather_motion_tensor(self, getter):
        """Gather per-env data from the active motion for each env (vectorized per motion)."""
        out = None
        for i, motion in enumerate(self.motions):
            mask = self.which_motion == i
            if torch.any(mask):
                data = getter(motion, mask)
                if out is None:
                    out = torch.zeros((self.num_envs,) + data.shape[1:], device=self.device, dtype=data.dtype)
                out[mask] = data
        return out


    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)
    
    @property
    def joint_pos(self) -> torch.Tensor:
        return self._gather_motion_tensor(lambda motion, mask: motion.joint_pos[self.time_steps[mask]])

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._gather_motion_tensor(lambda motion, mask: motion.joint_vel[self.time_steps[mask]])
    
    @property
    def body_pos_w(self) -> torch.Tensor:
        body_pos = self._gather_motion_tensor(lambda motion, mask: motion.body_pos_w[self.time_steps[mask]])
        return body_pos + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._gather_motion_tensor(lambda motion, mask: motion.body_quat_w[self.time_steps[mask]])
    
    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._gather_motion_tensor(lambda motion, mask: motion.body_lin_vel_w[self.time_steps[mask]])

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._gather_motion_tensor(lambda motion, mask: motion.body_ang_vel_w[self.time_steps[mask]])

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        anchor_pos = self._gather_motion_tensor(
            lambda motion, mask: motion.body_pos_w[self.time_steps[mask], self.motion_anchor_body_index]
        )
        return anchor_pos + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self._gather_motion_tensor(
            lambda motion, mask: motion.body_quat_w[self.time_steps[mask], self.motion_anchor_body_index]
        )

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self._gather_motion_tensor(
            lambda motion, mask: motion.body_lin_vel_w[self.time_steps[mask], self.motion_anchor_body_index]
        )

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self._gather_motion_tensor(
            lambda motion, mask: motion.body_ang_vel_w[self.time_steps[mask], self.motion_anchor_body_index]
        )
    
    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]
    
    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        """Adaptive sampling that tracks failures per motion separately."""
        episode_failed = self._env.termination_manager.terminated[env_ids]
        
        # Update failure counts for each motion
        if torch.any(episode_failed):
            for i in range(len(self.motions)):
                # Get environments that use this motion and failed
                motion_mask = self.which_motion[env_ids] == i
                failed_mask = episode_failed & motion_mask
                
                if torch.any(failed_mask):
                    # Calculate bin indices for failed environments using this motion
                    current_bin_index = torch.clamp(
                        (self.time_steps[env_ids[failed_mask]] * self.bin_counts[i]) // max(self.motions[i].time_step_total, 1), 
                        0, self.bin_counts[i] - 1
                    )
                    fail_bins = current_bin_index
                    self._current_bin_failed[i] = torch.bincount(fail_bins, minlength=self.bin_counts[i])
        
        # Compute sampling probabilities for all motions
        sampling_probs_per_motion = []
        for i in range(len(self.motions)):
            sampling_probabilities = self.bin_failed_counts[i] + self.cfg.adaptive_uniform_ratio / float(self.bin_counts[i])
            
            # Apply temporal smoothing kernel
            sampling_probabilities = torch.nn.functional.pad(
                sampling_probabilities.unsqueeze(0).unsqueeze(0),
                (0, self.cfg.adaptive_kernel_size - 1),
                mode="replicate",
            )
            sampling_probabilities = torch.nn.functional.conv1d(
                sampling_probabilities, self.kernel.view(1, 1, -1)
            ).view(-1)
            
            # Normalize
            sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()
            sampling_probs_per_motion.append(sampling_probabilities)
        
        # Vectorized sampling: sample bins for each environment based on its motion
        sampled_bins = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
        for i in range(len(self.motions)):
            motion_mask = self.which_motion[env_ids] == i
            if torch.any(motion_mask):
                num_samples = motion_mask.sum().item()
                sampled = torch.multinomial(sampling_probs_per_motion[i], num_samples, replacement=True)
                sampled_bins[motion_mask] = sampled
        
        # Vectorized conversion from bins to timesteps
        motion_indices = self.which_motion[env_ids]
        bin_counts_for_envs = torch.tensor([self.bin_counts[m] for m in motion_indices.tolist()], device=self.device)
        timestep_totals = torch.tensor([self.motions[m].time_step_total for m in motion_indices.tolist()], device=self.device)
        
        random_offsets = torch.rand(len(env_ids), device=self.device)
        new_timesteps = (
            (sampled_bins.float() + random_offsets) / bin_counts_for_envs.float() 
            * (timestep_totals.float() - 1)
        ).long()
        self.time_steps[env_ids] = new_timesteps
        
        # Vectorized metrics computation
        for i in range(len(self.motions)):
            motion_mask = self.which_motion[env_ids] == i
            if torch.any(motion_mask):
                probs = sampling_probs_per_motion[i]
                H = -(probs * (probs + 1e-12).log()).sum()
                H_norm = H / math.log(self.bin_counts[i])
                pmax, imax = probs.max(dim=0)
                
                # Get env_ids indices for this motion
                env_indices = torch.where(motion_mask)[0]
                actual_env_ids = torch.tensor([env_ids[j] for j in env_indices.tolist()], device=self.device)
                
                self.metrics["sampling_entropy"][actual_env_ids] = H_norm
                self.metrics["sampling_top1_prob"][actual_env_ids] = pmax
                self.metrics["sampling_top1_bin"][actual_env_ids] = imax.float() / self.bin_counts[i]

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        
        # print('target_position command', self.env.command_manager.get_command("target_position"))
                # Choose which motion to track for each env based on y-position
        y_pos = self.env.command_manager.get_command("target_position")[env_ids, 1]
        which = torch.full((len(env_ids),), 2, dtype=torch.long, device=self.device)
        which[y_pos < 0.0] = 0
        which[y_pos > 0.3] = 1
        self.which_motion[env_ids] = which
        # print("Chosen motions for envs:", self.which_motion[env_ids])
        

        # print("target positions as seen in multi action cmd", )

        # Choose which motion to track for each env FIRST
        # self.which_motion[env_ids] = torch.randint(0, len(self.motions), (len(env_ids),), device=self.device)
        
        # Then do adaptive sampling based on chosen motions
        self._adaptive_sampling(env_ids)

        # Set the root position to the body position in the motion
        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        """Update command each step: increment time steps, handle motion completion, and update adaptive tracking."""
        self.time_steps += 1
        
        # Check which environments need to reset (motion finished) - vectorized
        motion_indices = self.which_motion
        timestep_limits = torch.tensor([self.motions[m].time_step_total for m in motion_indices.tolist()], device=self.device)
        env_ids_to_reset = torch.where(self.time_steps >= timestep_limits)[0]
        
        if len(env_ids_to_reset) > 0:
            self.time_steps[env_ids_to_reset] = 0
            self._resample_command(env_ids_to_reset)

        # Update relative body poses
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        # Update adaptive failure tracking per motion - vectorized
        for i in range(len(self.motions)):
            self.bin_failed_counts[i] = (
                self.cfg.adaptive_alpha * self._current_bin_failed[i] 
                + (1 - self.cfg.adaptive_alpha) * self.bin_failed_counts[i]
            )
            self._current_bin_failed[i].zero_()


@configclass
class MultiTargetMotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MultiTargetMotionCommand

    asset_name: str = MISSING

    motion_files: list[str] = []
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)




class MultiMotionCommand(CommandTerm):
    cfg: MultiMotionCommandCfg

    def __init__(self, cfg: MultiMotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motions = []
        for motion_file in self.cfg.motion_files:
            self.motions.append(MotionLoader(motion_file, self.body_indexes, device=self.device))
        self.which_motion = torch.zeros(self.num_envs, dtype=torch.long, device=self.device) # which motion to track for each env

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # self.bin_count = int(self.motions[0].time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1

        self.bin_counts = [int(self.motions[i].time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1 for i in range(len(self.motions))]
        self.bin_failed_counts = [torch.zeros(self.bin_counts[i], dtype=torch.float, device=self.device) for i in range(len(self.motions))]
        self._current_bin_failed = [torch.zeros(self.bin_counts[i], dtype=torch.float, device=self.device) for i in range(len(self.motions))]

        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.between_motion_pause_length = 1.0
        self.between_motion_pause_time = torch.zeros(self.num_envs, device=self.device)

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
    

    def _gather_motion_tensor(self, getter):
        """Gather per-env data from the active motion for each env (vectorized per motion)."""
        out = None
        for i, motion in enumerate(self.motions):
            mask = self.which_motion == i
            if torch.any(mask):
                data = getter(motion, mask)
                if out is None:
                    out = torch.zeros((self.num_envs,) + data.shape[1:], device=self.device, dtype=data.dtype)
                out[mask] = data
        return out

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)
    
    @property
    def joint_pos(self) -> torch.Tensor:
        return self._gather_motion_tensor(lambda motion, mask: motion.joint_pos[torch.where(self.time_steps[mask] >= motion.time_step_total, 0, self.time_steps[mask])])

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._gather_motion_tensor(lambda motion, mask: motion.joint_vel[torch.where(self.time_steps[mask] >= motion.time_step_total, 0, self.time_steps[mask])])
    
    @property
    def body_pos_w(self) -> torch.Tensor:
        body_pos = self._gather_motion_tensor(lambda motion, mask: motion.body_pos_w[torch.where(self.time_steps[mask] >= motion.time_step_total, 0, self.time_steps[mask])])
        return body_pos + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._gather_motion_tensor(lambda motion, mask: motion.body_quat_w[torch.where(self.time_steps[mask] >= motion.time_step_total, 0, self.time_steps[mask])])
    
    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._gather_motion_tensor(lambda motion, mask: motion.body_lin_vel_w[torch.where(self.time_steps[mask] >= motion.time_step_total, 0, self.time_steps[mask])])

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._gather_motion_tensor(lambda motion, mask: motion.body_ang_vel_w[torch.where(self.time_steps[mask] >= motion.time_step_total, 0, self.time_steps[mask])])

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        anchor_pos = self._gather_motion_tensor(
            lambda motion, mask: motion.body_pos_w[torch.where(self.time_steps[mask] >= motion.time_step_total, 0, self.time_steps[mask]), self.motion_anchor_body_index]
        )
        return anchor_pos + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self._gather_motion_tensor(
            lambda motion, mask: motion.body_quat_w[torch.where(self.time_steps[mask] >= motion.time_step_total, 0, self.time_steps[mask]), self.motion_anchor_body_index]
        )

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self._gather_motion_tensor(
            lambda motion, mask: motion.body_lin_vel_w[torch.where(self.time_steps[mask] >= motion.time_step_total, 0, self.time_steps[mask]), self.motion_anchor_body_index]
        )

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self._gather_motion_tensor(
            lambda motion, mask: motion.body_ang_vel_w[torch.where(self.time_steps[mask] >= motion.time_step_total, 0, self.time_steps[mask]), self.motion_anchor_body_index]
        )
    
    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]
    
    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        """Adaptive sampling that tracks failures per motion separately."""
        episode_failed = self._env.termination_manager.terminated[env_ids]
        
        # Update failure counts for each motion
        if torch.any(episode_failed):
            for i in range(len(self.motions)):
                # Get environments that use this motion and failed
                motion_mask = self.which_motion[env_ids] == i
                failed_mask = episode_failed & motion_mask
                
                if torch.any(failed_mask):
                    # Calculate bin indices for failed environments using this motion
                    current_bin_index = torch.clamp(
                        (self.time_steps[env_ids[failed_mask]] * self.bin_counts[i]) // max(self.motions[i].time_step_total, 1), 
                        0, self.bin_counts[i] - 1
                    )
                    fail_bins = current_bin_index
                    self._current_bin_failed[i] = torch.bincount(fail_bins, minlength=self.bin_counts[i])
        
        # Compute sampling probabilities for all motions
        sampling_probs_per_motion = []
        for i in range(len(self.motions)):
            sampling_probabilities = self.bin_failed_counts[i] + self.cfg.adaptive_uniform_ratio / float(self.bin_counts[i])
            
            # Apply temporal smoothing kernel
            sampling_probabilities = torch.nn.functional.pad(
                sampling_probabilities.unsqueeze(0).unsqueeze(0),
                (0, self.cfg.adaptive_kernel_size - 1),
                mode="replicate",
            )
            sampling_probabilities = torch.nn.functional.conv1d(
                sampling_probabilities, self.kernel.view(1, 1, -1)
            ).view(-1)
            
            # Normalize
            sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()
            sampling_probs_per_motion.append(sampling_probabilities)
        
        # Vectorized sampling: sample bins for each environment based on its motion
        sampled_bins = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
        for i in range(len(self.motions)):
            motion_mask = self.which_motion[env_ids] == i
            if torch.any(motion_mask):
                num_samples = motion_mask.sum().item()
                sampled = torch.multinomial(sampling_probs_per_motion[i], num_samples, replacement=True)
                sampled_bins[motion_mask] = sampled
        
        # Vectorized conversion from bins to timesteps
        motion_indices = self.which_motion[env_ids]
        bin_counts_for_envs = torch.tensor([self.bin_counts[m] for m in motion_indices.tolist()], device=self.device)
        timestep_totals = torch.tensor([self.motions[m].time_step_total for m in motion_indices.tolist()], device=self.device)
        
        random_offsets = torch.rand(len(env_ids), device=self.device)
        new_timesteps = (
            (sampled_bins.float() + random_offsets) / bin_counts_for_envs.float() 
            * (timestep_totals.float() - 1)
        ).long()
        self.time_steps[env_ids] = new_timesteps
        
        # Vectorized metrics computation
        for i in range(len(self.motions)):
            motion_mask = self.which_motion[env_ids] == i
            if torch.any(motion_mask):
                probs = sampling_probs_per_motion[i]
                H = -(probs * (probs + 1e-12).log()).sum()
                H_norm = H / math.log(self.bin_counts[i])
                pmax, imax = probs.max(dim=0)
                
                # Get env_ids indices for this motion
                env_indices = torch.where(motion_mask)[0]
                actual_env_ids = torch.tensor([env_ids[j] for j in env_indices.tolist()], device=self.device)
                
                self.metrics["sampling_entropy"][actual_env_ids] = H_norm
                self.metrics["sampling_top1_prob"][actual_env_ids] = pmax
                self.metrics["sampling_top1_bin"][actual_env_ids] = imax.float() / self.bin_counts[i]

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        
        # Choose which motion to track for each env FIRST
        self.which_motion[env_ids] = torch.randint(0, len(self.motions), (len(env_ids),), device=self.device)
        
        # Then do adaptive sampling based on chosen motions
        self._adaptive_sampling(env_ids)

        # Set the root position to the body position in the motion
        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        """Update command each step: increment time steps, handle motion completion, and update adaptive tracking."""
        self.time_steps += 1
        
        # Check which environments need to reset (motion finished) - vectorized
        motion_indices = self.which_motion

        timestep_limits = torch.tensor([self.motions[m].time_step_total for m in motion_indices.tolist()], device=self.device)
        env_ids_at_end_of_motion = torch.where(self.time_steps >= timestep_limits)[0]
        self.between_motion_pause_time[env_ids_at_end_of_motion] += self._env.cfg.sim.dt
        env_ids_to_reset = env_ids_at_end_of_motion[self.between_motion_pause_time[env_ids_at_end_of_motion] >= self.between_motion_pause_length]
        self.between_motion_pause_time[env_ids_to_reset] = 0.0

        if len(env_ids_to_reset) > 0:
            self.time_steps[env_ids_to_reset] = 0
            self._resample_command(env_ids_to_reset)

        # Update relative body poses
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        # Update adaptive failure tracking per motion - vectorized
        for i in range(len(self.motions)):
            self.bin_failed_counts[i] = (
                self.cfg.adaptive_alpha * self._current_bin_failed[i] 
                + (1 - self.cfg.adaptive_alpha) * self.bin_failed_counts[i]
            )
            self._current_bin_failed[i].zero_()


@configclass
class MultiMotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MultiMotionCommand

    asset_name: str = MISSING

    motion_files: list[str] = []
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]

"""The motion command: joint positions and velocities of target motion"""
class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)

        # Set the root position to the body position in the motion
        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
