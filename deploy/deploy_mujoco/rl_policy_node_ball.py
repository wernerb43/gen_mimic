import argparse
import os
import threading
import numpy as np
import rclpy
import yaml
import sys
import math
import os

# use python onnx package only (no onnxruntime)
import onnx
import json
import ast

# add onnxruntime for inference
try:
    import onnxruntime as ort
except Exception:
    ort = None
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64

import xml.etree.ElementTree as ET

# add typing imports for explicit annotation to avoid List[None] inference
from typing import List, Optional, Tuple
import threading


class RLPolicyNode(Node):
    def __init__(self, config_file="g1_29dof.yaml"):
        super().__init__("rl_policy_node")
        self.config_file = config_file
        self.sensor_sub = self.create_subscription(
            Float32MultiArray, "sensor_data", self.sensor_callback, 10
        )
        self.robot_position_sub = self.create_subscription(
            Float32MultiArray, "robot_position", self.robot_position_callback, 10
        )
        self.time_sub = self.create_subscription(
            Float64, "sim_time", self.time_callback, 10
        )
        # Subscribe to control enable flag published by the simulator (Float64: 0.0/1.0)
        self.control_enable_sub = self.create_subscription(
            Float64, "control_enable", self.control_enable_callback, 10
        )

        self.base_lin_vel_sub = self.create_subscription(
            Float32MultiArray, "base_lin_vel", self.base_lin_vel_callback, 10
        )

        self.ball_position_sub = self.create_subscription(
            Float32MultiArray, "ball_position", self.ball_position_callback, 10
        )

        self.action_pub = self.create_publisher(Float32MultiArray, "action", 10)
        self.sim_time = 0.0

        # Load minimal config (only keys present in YAML)
        self.load_config()
        # Load policy if desired (may be TorchScript); it's optional — node works without it.
        self.load_policy()
        self.load_policy_metadata()
        # Load motion trajectory from npz file if provided
        self.load_motion_trajectory()

        # timer uses control_dt from YAML (simulation_dt * control_decimation)
        self.timer = self.create_timer(self.control_dt, self.policy_step)

        self.cmd_lock = threading.Lock()
        # Initialize command from config (vx, vy, vyaw, height)
        if len(self.cmd_init) == 3:
            self.cmd = np.concatenate([self.cmd_init.copy(), np.array([0.75], dtype=np.float32)])
        else:
            self.cmd = self.cmd_init.copy()

        # Lightweight state used by simplified node

        self.robot_position = np.zeros(3, dtype=np.float32)
        self.base_lin_vel = np.zeros(3, dtype=np.float32)
        self.qj = np.zeros(29, dtype=np.float32)
        self.dqj = np.zeros(29, dtype=np.float32)
        self.omega = np.zeros(3, dtype=np.float32)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.action = np.zeros(29, dtype=np.float32)
        # Policy enable flag (wait until simulator publishes control_enable)
        self.control_enabled = False
        # whether policy stepping has actually started (for one-time print)
        self._policy_started = False
        # Track last cache frame per velocity index to detect wraps
        
        # Motion playback control (policy mode)
        self._play_motion_once = False
        self._initial_yaw = None
        
        # Keyboard input handling for motion playback
        self._kb_request_play = False
        self._term_settings = None
        self._keyboard_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self._keyboard_thread.start()

    def load_config(self):
        G1_RL_ROOT_DIR = os.getenv("G1_RL_ROOT_DIR")
        with open(f"{G1_RL_ROOT_DIR}/deploy/configs/{self.config_file}") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # Keep only keys present in the YAML
        self.policy_path = config["policy_path"].replace(
            "{G1_RL_ROOT_DIR}", G1_RL_ROOT_DIR
        )
        self.use_gpu = config.get("use_gpu", False)

        self.xml_path = config.get("xml_path", "").replace(
            "{G1_RL_ROOT_DIR}", G1_RL_ROOT_DIR
        )
        self.robot_xml_path = config.get("robot_xml_path", self.xml_path).replace(
            "{G1_RL_ROOT_DIR}", G1_RL_ROOT_DIR
        )

        # timing
        self.simulation_dt = float(config.get("simulation_dt", 0.002))
        self.control_decimation = int(config.get("control_decimation", 10))
        # control_dt key exists in YAML; fall back to simulation_dt * control_decimation if not present
        self.control_dt = float(
            config.get("control_dt", self.simulation_dt * self.control_decimation)
        )

        # command scales and initial command
        self.cmd_scale = np.array(
            config.get("cmd_scale", [1.0, 1.0, 1.0]), dtype=np.float32
        )
        self.cmd_init = np.array(
            config.get("cmd_init", [0.0, 0.0, 0.0]), dtype=np.float32
        )
        self.max_cmd = np.array(
            config.get("max_cmd", [1.0, 1.0, 1.0]), dtype=np.float32
        )

        # motion trajectory npz file
        self.motion_npz_path = config.get("motion_npz_path", "")
        if self.motion_npz_path:
            self.motion_npz_path = self.motion_npz_path.replace("{G1_RL_ROOT_DIR}", G1_RL_ROOT_DIR)

    def control_enable_callback(self, msg: Float64):
        try:
            enabled = float(msg.data) != 0.0
        except Exception:
            enabled = False
        if enabled and not getattr(self, "control_enabled", False):
            self.get_logger().info("Received control_enable = True; enabling policy.")
            print(f"[RLPolicyNode] control_enable received at sim_time={getattr(self,'sim_time',-1):.3f}; enabling policy", flush=True)
        self.control_enabled = enabled

    def load_policy_metadata(self):
        """
        Reads and parses embedded ONNX metadata from the model file.
        Returns parsed_meta dict.
        """
        try:
            onnx_model = self.onnx_model
            meta_from_model = {}
            for prop in getattr(onnx_model, "metadata_props", []) or []:
                k = getattr(prop, "key", None)
                v = getattr(prop, "value", None)
                if k is not None:
                    meta_from_model[k] = v
        except Exception as e:
            raise RuntimeError(f"Failed to read metadata_props from ONNX model: {e}")

        if not meta_from_model:
            raise RuntimeError(
                "ONNX model does not contain embedded metadata (metadata_props). "
                "External fallback is disabled; please embed 'onnx_metadata' into the model."
            )

        parsed_meta = None
        if "onnx_metadata" in meta_from_model:
            raw = meta_from_model["onnx_metadata"]
            try:
                parsed_meta = json.loads(raw)
            except Exception:
                try:
                    parsed_meta = ast.literal_eval(raw)
                except Exception as e:
                    raise RuntimeError(f"Failed to parse 'onnx_metadata' property: {e}")
        else:
            parsed_meta = {}
            for k, raw in meta_from_model.items():
                if raw is None:
                    parsed_meta[k] = None
                    continue
                try:
                    parsed_meta[k] = json.loads(raw)
                except Exception:
                    try:
                        parsed_meta[k] = ast.literal_eval(raw)
                    except Exception:
                        parsed_meta[k] = raw

        # --- NEW: Convert string metadata entries to list by splitting on comma ---
        for k, v in parsed_meta.items():
            if isinstance(v, str):
                parsed_meta[k] = [item.strip() for item in v.split(",") if item.strip()]

        # Populate node metadata fields (existing logic)
        self.onnx_metadata = parsed_meta
        self.joint_names = parsed_meta.get("joint_names", [])
        action_scale = parsed_meta.get("action_scale", None)
        if action_scale is None:
            raise RuntimeError(
                "Embedded ONNX metadata missing required 'action_scale' entry (no fallback)."
            )
        try:
            self.action_scale = np.array(action_scale, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(
                f"Failed to convert 'action_scale' from metadata to numpy array: {e}"
            )

        default_joint_pos = parsed_meta.get("default_joint_pos", None)
        if default_joint_pos is None:
            raise RuntimeError(
                "Embedded ONNX metadata missing required 'default_joint_pos' entry (no fallback)."
            )
        try:
            self.default_joint_pos = np.array(default_joint_pos, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(
                f"Failed to convert 'default_joint_pos' from metadata to numpy array: {e}"
            )

        # --- Load p_gain and d_gain from metadata ---
        joint_stiffness = parsed_meta.get("joint_stiffness", None)
        joint_damping = parsed_meta.get("joint_damping", None)
        if joint_stiffness is None or joint_damping is None:
            raise RuntimeError(
                "Embedded ONNX metadata missing required 'joint_stiffness' or 'joint_damping' entry."
            )
        try:
            # These are in ONNX order, will be converted to XML order below
            onnx_p_gain = np.array(joint_stiffness, dtype=np.float32)
            onnx_d_gain = np.array(joint_damping, dtype=np.float32)
            print("[RLPolicyNode] Using gains from ONNX metadata")
        except Exception as e:
            raise RuntimeError(
                f"Failed to convert joint_stiffness/damping from metadata to numpy array: {e}"
            )

        # Infer joint dimension from default_joint_pos
        self.num_joints = len(default_joint_pos)
        self.qj_shape = (self.num_joints,)
        self.dqj_shape = (self.num_joints,)

        self.observation_names = parsed_meta.get("observation_names", [])
        print('obs names', self.observation_names)
        self.command_names = parsed_meta.get("command_names", [])

        # --- Parse joint order from XML and create mapping to ONNX metadata joint order ---
        self.xml_joint_names = self._parse_joint_names_from_xml(self.robot_xml_path)
        self.xml_to_onnx_joint_indices, self.onnx_to_xml_joint_indices = (
            self._create_joint_order_mappings(self.xml_joint_names, self.joint_names)
        )
        
        # Convert ONNX-order gains to XML/MuJoCo order
        self.p_gain = self.isaac_to_mujoco(onnx_p_gain)
        self.d_gain = self.isaac_to_mujoco(onnx_d_gain)
        
        # Create onnxruntime session for inference (use GPU if requested and available)
        if ort is None:
            raise RuntimeError(
                "onnxruntime is required for model inference but not available."
            )
        providers = []
        try:
            avail = ort.get_available_providers()
        except Exception:
            avail = []
        if self.use_gpu and "CUDAExecutionProvider" in avail:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        try:
            self.onnx_sess = ort.InferenceSession(self.policy_path, providers=providers)
        except Exception as e:
            raise RuntimeError(f"Failed to create onnxruntime InferenceSession: {e}")

        # Cache input/output names for fast inference
        self.onnx_input_names = [inp["name"] for inp in self.onnx_inputs]
        self.onnx_output_names = [out["name"] for out in self.onnx_outputs]
        # determine if recurrent model by presence of h_in/c_in inputs
        input_names_set = set(self.onnx_input_names)
        self.is_recurrent_onnx = "h_in" in input_names_set and "c_in" in input_names_set

        self.get_logger().info(
            "Successfully loaded ONNX model and created onnxruntime session."
        )
        # print the loaded metadata
        print("default_joint_pos:", self.isaac_to_mujoco(self.default_joint_pos))
        print(f"mujoco joint names: {self.xml_joint_names}")
        print(f"onnx joint names: {self.joint_names}")
        print(f"p_gain: {self.p_gain}")
        print(f"d_gain: {self.d_gain}")
        print(f"observation_names: {self.observation_names}")
        print(f"command_names: {self.command_names}")

    def load_policy(self):
        """
        Strict ONNX loader with onnxruntime inference support.
        Requires the ONNX model to include embedded metadata_props.
        """
        # Load and validate ONNX model (existing logic)
        self.onnx_model = None
        self.onnx_inputs = []
        self.onnx_outputs = []
        try:
            self.onnx_model = onnx.load(self.policy_path)
            onnx.checker.check_model(self.onnx_model)
            graph = self.onnx_model.graph
            # extract input metadata (name and shape)
            for inp in graph.input:
                name = inp.name
                shape = None
                if (
                    getattr(inp, "type", None)
                    and getattr(inp.type, "tensor_type", None)
                    and getattr(inp.type.tensor_type, "shape", None)
                ):
                    dims = []
                    for d in inp.type.tensor_type.shape.dim:
                        if getattr(d, "dim_value", 0) > 0:
                            dims.append(int(d.dim_value))
                        else:
                            dims.append(-1)
                    shape = tuple(dims)
                self.onnx_inputs.append({"name": name, "shape": shape})
            # extract outputs
            for out in graph.output:
                name = out.name
                shape = None
                if (
                    getattr(out, "type", None)
                    and getattr(out.type, "tensor_type", None)
                    and getattr(out.type.tensor_type, "shape", None)
                ):
                    dims = []
                    for d in out.type.tensor_type.shape.dim:
                        if getattr(d, "dim_value", 0) > 0:
                            dims.append(int(d.dim_value))
                        else:
                            dims.append(-1)
                    shape = tuple(dims)
                self.onnx_outputs.append({"name": name, "shape": shape})
            print("inputs names and shapes:", self.onnx_inputs)
            print("output names and shapes:", self.onnx_outputs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load or validate ONNX model at '{self.policy_path}': {e}"
            )

    def load_motion_trajectory(self):
        """
        Load motion trajectory from npz file. The npz file should contain:
        - fps: output fps of the motion
        - joint_pos: shape (T, num_joints)
        - joint_vel: shape (T, num_joints)
        - body_pos_w: shape (T, num_bodies, 3)
        - body_quat_w: shape (T, num_bodies, 4)
        - body_lin_vel_w: shape (T, num_bodies, 3)
        - body_ang_vel_w: shape (T, num_bodies, 3)
        """
        self.motion_trajectory = None
        self.motion_frame_idx = 0
        self._play_motion_once = False
        
        if not self.motion_npz_path or self.motion_npz_path == "":
            self.get_logger().info("No motion trajectory file specified, using zero placeholders")
            return
        
        try:
            if not os.path.exists(self.motion_npz_path):
                self.get_logger().warn(f"Motion npz file not found: {self.motion_npz_path}")
                return
            
            self.motion_trajectory = np.load(self.motion_npz_path)
            motion_fps = self.motion_trajectory["fps"]
            num_frames = self.motion_trajectory["joint_pos"].shape[0]
            num_bodies = self.motion_trajectory["body_pos_w"].shape[1]
            
            self.get_logger().info(
                f"Loaded motion trajectory from {self.motion_npz_path}: "
                f"{num_frames} frames at {motion_fps} fps, {num_bodies} bodies"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to load motion trajectory: {e}")
            self.motion_trajectory = None

    def _compute_traj_velocities(self, pos_traj, dt):
        """
        Compute per-frame joint velocities for a position trajectory pos_traj (T, num_joints).
        Uses the same indexing rules previously used in obs_command:
          - central difference (next - prev) / (2*dt)
          - for t == T-1: use next == pos_traj[T-1] (duplicate) and prev == pos_traj[T-2] to avoid wrap discontinuity
          - for t == 0 prev uses pos_traj[-1] (wrap) as in original lookup logic
        Returns vel_traj shape (T, num_joints) dtype float32.
        """
        pos = np.asarray(pos_traj, dtype=np.float32)
        T = pos.shape[0]
        if T <= 1:
            return np.zeros_like(pos, dtype=np.float32)
        vel = np.zeros_like(pos, dtype=np.float32)
        dt_val = float(dt)
        vel[1:, :] = (pos[1:, :] - pos[:-1, :]) / dt_val
        vel[0, :] = vel[1, :] if T > 1 else 0.0
        return vel.astype(np.float32)

    def _find_closest_cached_velocity(self, vel_x: float, vel_y: float = 0.0):
        """Return the cached velocity key nearest to (vel_x, vel_y)."""
        if self.velocity_grid is None or len(self.velocity_grid) == 0:
            return None
        query = np.array([vel_x, vel_y], dtype=np.float32)
        diffs = self.velocity_grid - query[None, :]
        dists = np.linalg.norm(diffs, axis=1)
        idx = int(np.argmin(dists))
        vx, vy = float(self.velocity_grid[idx, 0]), float(self.velocity_grid[idx, 1])
        return (round(vx, 3), round(vy, 3), idx)

    def _parse_joint_names_from_xml(self, xml_path):
        """
        Parse joint names from MuJoCo XML file in the order they appear.
        Returns a list of joint names.
        """
        joint_names = []
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            # MuJoCo joints are typically under <joint> tags
            for joint in root.iter("joint"):
                name = joint.attrib.get("name", None)
                if name is not None:
                    joint_names.append(name)
            # pop the floating base
            joint_names.pop(0)
        except Exception as e:
            self.get_logger().warning(f"Failed to parse joint names from XML: {e}")
        return joint_names

    def _create_joint_order_mappings(self, xml_joint_names, onnx_joint_names):
        """
        Create mappings between XML joint order and ONNX metadata joint order.
        Returns:
            xml_to_onnx: list where xml_to_onnx[i] = index in onnx_joint_names for xml_joint_names[i]
            onnx_to_xml: list where onnx_to_xml[i] = index in xml_joint_names ````````for onnx_joint_names[i]
        """
        xml_to_onnx = []
        for name in xml_joint_names:
            try:
                idx = onnx_joint_names.index(name)
            except ValueError:
                idx = -1
            xml_to_onnx.append(idx)
        onnx_to_xml = []
        for name in onnx_joint_names:
            try:
                idx = xml_joint_names.index(name)
            except ValueError:
                idx = -1
            onnx_to_xml.append(idx)
        return xml_to_onnx, onnx_to_xml

    def _run_onnx_inference(self):
        """Prepare inputs and run the ONNX model via onnxruntime.
        Returns the first output ndarray.
        """
        if not hasattr(self, "onnx_sess") or self.onnx_sess is None:
            raise RuntimeError("ONNX session not initialized")

        # Prepare input feed
        input_feed = {}
        # If model expects 'obs' input, use construct_observations(); otherwise use first input
        for inp in self.onnx_inputs:
            name = inp["name"]
            shape = inp["shape"]  # tuple or None
            if name == "obs" or len(self.onnx_inputs) == 1:
                obs_np = self.construct_observations().astype(np.float32)
                expected = None
                if shape is not None:
                    if len(shape) == 1:
                        expected = shape[0]
                    elif len(shape) >= 2:
                        expected = shape[-1]  # assume shape [B, N]
                if expected is None or expected <= 0:
                    arr = obs_np.reshape(1, -1).astype(np.float32)
                else:
                    vals = obs_np[:expected]
                    if vals.size < expected:
                        pad = np.zeros((expected - vals.size,), dtype=np.float32)
                        vals = np.concatenate([vals, pad], axis=0)
                    arr = vals.reshape(1, expected).astype(np.float32)
                input_feed[name] = arr
            elif name == "time_step":
                # Provide a default time step input if required
                arr = np.array([[self.sim_time / self.control_dt]], dtype=np.float32)
                input_feed[name] = arr

        # Run inference
        outputs = self.onnx_sess.run(None, input_feed)
        # Return first output ndarray
        return outputs[0]


    def policy_step(self):
        # Do not run policy until simulator signals control_enable
        if not getattr(self, "control_enabled", False):
            # Keep alive but skip heavy processing
            return
        # One-time print when policy actually starts running
        if not getattr(self, '_policy_started', False):
            print(f"[RLPolicyNode] Policy stepping started at sim_time={getattr(self,'sim_time',-1):.3f}", flush=True)
            self._policy_started = True
        
        # Handle motion playback: increment frame only if playback active
        if self.motion_trajectory is not None and self._play_motion_once:
            num_frames = self.motion_trajectory["joint_pos"].shape[0]
            if num_frames > 0:
                if self.motion_frame_idx < num_frames - 1:
                    self.motion_frame_idx += 1
                else:
                    # Reached end: reset to first frame and stop playback
                    self.motion_frame_idx = 1
                    self._play_motion_once = False
        
        if hasattr(self, "onnx_sess") and self.onnx_sess is not None:
            try:
                out0 = self._run_onnx_inference()  # numpy ndarray
                # Prepare action, p gain, d gain arrays from metadata
                action = out0.astype(np.float32).ravel()
                self.action = action.copy()
                p_gain = self.p_gain.astype(np.float32)
                d_gain = self.d_gain.astype(np.float32)
                # Concatenate [action, p_gain, d_gain]
                msg = Float32MultiArray()
                joint_pos = self.default_joint_pos + self.action * self.action_scale

                msg.data = np.concatenate(
                    [
                        self.isaac_to_mujoco(joint_pos),
                        self.isaac_to_mujoco(p_gain),
                        self.isaac_to_mujoco(d_gain),
                    ]
                ).tolist()
                self.action_pub.publish(msg)
                return
            except Exception as e:
                print(
                    f"ONNX inference failed: {e} — falling back to IK/default publish"
                )
                exit(0)

    def sensor_callback(self, msg):
        # Sensor data is [qj, dqj, quat, omega]
        received = np.array(msg.data, dtype=np.float32)
        qj = received[: self.num_joints]
        dqj = received[self.num_joints : 2 * self.num_joints]
        quat = received[2 * self.num_joints : 2 * self.num_joints + 4]
        omega = received[2 * self.num_joints + 4 : 2 * self.num_joints + 7]
        # Store as numpy arrays
        self.qj = np.array(qj, dtype=np.float32)
        self.dqj = np.array(dqj, dtype=np.float32)
        self.quat = np.array(quat, dtype=np.float32)
        self.omega = np.array(omega, dtype=np.float32)

    def robot_position_callback(self, msg):
        received = np.array(msg.data, dtype=np.float32)
        if len(received) == 3:
            self.robot_position = np.array(received, dtype=np.float32)
        else:
            self.get_logger().warn(
                f"Received robot position with unexpected length: {len(received)}"
            )

    def time_callback(self, msg):
        self.sim_time = msg.data

    def base_lin_vel_callback(self, msg):
        received = np.array(msg.data, dtype=np.float32)
        if len(received) == 3:
            self.base_lin_vel = np.array(received, dtype=np.float32)

    def ball_position_callback(self, msg):
        received = np.array(msg.data, dtype=np.float32)
        if len(received) == 3:
            self.ball_position = np.array(received, dtype=np.float32)

    def isaac_to_mujoco(self, arr):
        """
        Convert a joint array from ONNX/IsaacLab order to MuJoCo/XML order.
        arr: numpy array in ONNX joint order (length = num_joints)
        Returns: array in XML joint order (length = num_joints)
        """
        out = np.zeros_like(arr)
        for xml_idx, onnx_idx in enumerate(self.xml_to_onnx_joint_indices):
            if onnx_idx >= 0 and onnx_idx < len(arr):
                out[xml_idx] = arr[onnx_idx]
        return out

    def mujoco_to_isaac(self, arr):
        """
        Convert a joint array from MuJoCo/XML order to ONNX/IsaacLab order.
        arr: numpy array in XML joint order (length = num_joints)
        Returns: array in ONNX joint order (length = num_joints)
        """
        out = np.zeros_like(arr)
        for onnx_idx, xml_idx in enumerate(self.onnx_to_xml_joint_indices):
            if xml_idx >= 0 and xml_idx < len(arr):
                out[onnx_idx] = arr[xml_idx]
        return out


    def obs_command_imitate(self):
        # Return motion trajectory command: [joint_pos (motion), joint_vel (motion)]
        # This is the desired joint configuration and velocities from the motion being replayed
        # Size: 2 * num_joints (e.g., 58 for 29-DOF robot)
        if self.motion_trajectory is not None:
            try:
                joint_pos = self.motion_trajectory["joint_pos"][self.motion_frame_idx]
                joint_vel = self.motion_trajectory["joint_vel"][self.motion_frame_idx]
                cmd = np.concatenate([joint_pos, joint_vel]).astype(np.float32)
                return cmd
            except Exception as e:
                self.get_logger().warn(f"Error extracting motion command: {e}")
        return np.zeros(2 * self.num_joints, dtype=np.float32)

    def obs_command_target_position(self):
        # Transform ball position from world frame to robot body frame
        # 1. Compute relative position in world frame
        ball_pos_relative_w = self.ball_position - self.robot_position
        
        # 2. Rotate to body frame using inverse quaternion
        q_inv = self._quat_conjugate(self.quat)
        ball_pos_body = self._quat_rotate(q_inv, ball_pos_relative_w)
        
        return ball_pos_body.astype(np.float32)

    def obs_motion_anchor_pos_b(self):
        # Return motion anchor body position error in robot base frame
        # This is: motion_anchor_pos_w - robot_anchor_pos_w, transformed to robot base frame
        if self.motion_trajectory is not None:
            # try:
                # Anchor body is typically the first body (index 0, pelvis/base)
            motion_anchor_pos = self.motion_trajectory["body_pos_w"][self.motion_frame_idx, 0]  # (3,)
            
            # Robot anchor position is the robot's current position
            robot_anchor_pos = self.robot_position  # (3,)
            
            # Position difference in world frame
            pos_error_w = motion_anchor_pos - robot_anchor_pos
            
            # Transform to robot base frame using quaternion
            # q_inv rotates from world to body frame
            q_inv = self._quat_conjugate(self.quat)
            pos_error_b = self._quat_rotate(q_inv, pos_error_w)
            
            return pos_error_b.astype(np.float32)
            # except Exception as e:
            #     self.get_logger().warn(f"Error computing motion anchor position: {e}")
        return np.zeros(3, dtype=np.float32)

    def obs_motion_anchor_ori_b(self):
        # Return motion anchor body orientation error in robot base frame as 6D encoding
        # Format: [r00, r01, r10, r11, r20, r21] - first two columns of 3x3 rotation matrix
        if self.motion_trajectory is not None:
            try:
                # Get motion anchor orientation (first body, pelvis/base)
                motion_anchor_quat = self.motion_trajectory["body_quat_w"][self.motion_frame_idx, 0]  # (4,) wxyz
                
                # Robot anchor orientation is the robot's current quaternion
                robot_anchor_quat = self.quat  # (4,) wxyz
                
                # Compute relative quaternion: quat_error = quat_robot_inv * quat_motion
                q_inv = self._quat_conjugate(robot_anchor_quat)
                q_rel = self._quat_multiply(q_inv, motion_anchor_quat)
                
                # Convert quaternion to rotation matrix
                rot_mat = self._quat_to_rotation_matrix(q_rel)
                
                # Extract 6D encoding: [r00, r01, r10, r11, r20, r21]
                ori_6d = np.array([
                    rot_mat[0, 0], rot_mat[0, 1],
                    rot_mat[1, 0], rot_mat[1, 1],
                    rot_mat[2, 0], rot_mat[2, 1]
                ], dtype=np.float32)
                
                return ori_6d
            except Exception as e:
                self.get_logger().warn(f"Error computing motion anchor orientation: {e}")
        return np.zeros(6, dtype=np.float32)

    def _quat_conjugate(self, q: np.ndarray) -> np.ndarray:
        """Compute quaternion conjugate. Input/output format: [w, x, y, z]"""
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)
    
    def _quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions. Input/output format: [w, x, y, z]"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dtype=np.float32)
    
    def _quat_rotate(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate vector v by quaternion q. q format: [w, x, y, z]"""
        # v' = q * v * q_inv where v is treated as [0, vx, vy, vz]
        v_quat = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)
        q_inv = self._quat_conjugate(q)
        result = self._quat_multiply(q, self._quat_multiply(v_quat, q_inv))
        return result[1:4]  # Return [x, y, z]
    
    def _quat_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to 3x3 rotation matrix. q format: [w, x, y, z]"""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ], dtype=np.float32)

    def obs_base_lin_vel(self):
        return self.base_lin_vel

    def obs_base_ang_vel(self):
        return self.omega

    def obs_joint_pos(self):
        joint_pos = self.mujoco_to_isaac(self.qj)
        return joint_pos - self.default_joint_pos

    def obs_joint_vel(self):
        joint_vel = self.mujoco_to_isaac(self.dqj)
        return joint_vel

    def obs_actions(self):
        return self.action

    def obs_projected_gravity(self):
        qw = self.quat[0]
        qx = self.quat[1]
        qy = self.quat[2]
        qz = self.quat[3]

        gravity_orientation = np.zeros(3)

        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

        return gravity_orientation

    def _keyboard_loop(self):
        """Keyboard input loop for motion playback control."""
        try:
            import sys
            import select
            import tty
            import termios
            
            self._term_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            while True:
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch.lower() == 'p':
                        self._play_motion_once = True
                    elif ch.lower() == 'q':
                        break
        except Exception as e:
            self.get_logger().warn(f"Keyboard input error: {e}")
        finally:
            if self._term_settings is not None:
                try:
                    import sys
                    import termios
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._term_settings)
                except Exception:
                    pass

    def construct_observations(self):
        """
        Construct the observation vector using observation function names from metadata.
        Each function is called as obs_${name} and must return a numpy array.
        Returns a concatenated numpy array of all observations.
        """
        obs_list = []
        for name in self.observation_names:
            func_name = f"obs_{name}"
            obs_func = getattr(self, func_name, None)
            if obs_func is None:
                raise RuntimeError(
                    f"Observation function '{func_name}' not found in RLPolicyNode."
                )
            obs_val = obs_func()
            if obs_val is None:
                raise RuntimeError(f"Observation function '{func_name}' returned None.")
            obs_list.append(np.asarray(obs_val, dtype=np.float32))

        print(obs_list)
        observations = np.concatenate(obs_list, axis=-1)
        return observations


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser(
        description="Launch MujocoSimNode and RLPolicyNode with a given config file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config file name (e.g., g1_21dof.yaml)",
    )
    args = parser.parse_args()
    node = RLPolicyNode(config_file=args.config)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()