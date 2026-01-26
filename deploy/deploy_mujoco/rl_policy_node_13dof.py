# Copyright (c) 2025 Lizhi Yang AMBER LAB

import argparse
import os
import threading
import numpy as np
import rclpy
import yaml
import sys
import math
import termios
import tty
import select
import os

try:
    import matplotlib

    if not os.environ.get("DISPLAY") and os.name != "nt":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

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

try:
    from inputs import get_gamepad
except ImportError:
    get_gamepad = None
    print("inputs library not found. Xbox controller input will not work.")

import xml.etree.ElementTree as ET

# --- ADDED: attempt to import G1 analytical IK solver (non-fatal) ---
try:
    # adjust path to where your build installs the python module
    sys.path.append("/home/yangl/BeyondMimic/g1_ctrl/analytical_kinematic_solver/build")
    import g1_kinematics  # type: ignore

    G1_KINEMATICS_AVAILABLE = True
except Exception:
    g1_kinematics = None
    G1_KINEMATICS_AVAILABLE = False

# add typing imports for explicit annotation to avoid List[None] inference
from typing import List, Optional, Tuple


class RLPolicyNode(Node):
    def __init__(self, config_file="g1_21dof.yaml"):
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
        self.heightmap_sub = self.create_subscription(
            Float32MultiArray, "height_map", self.heightmap_callback, 10
        )
        self.action_pub = self.create_publisher(Float32MultiArray, "action", 10)
        self.sim_time = 0.0

        # Load minimal config (only keys present in YAML)
        self.load_config()
        # Load policy if desired (may be TorchScript); it's optional — node works without it.
        self.load_policy()
        self.load_policy_metadata()

        # timer uses control_dt from YAML (simulation_dt * control_decimation)
        self.timer = self.create_timer(self.control_dt, self.policy_step)

        self.cmd_lock = threading.Lock()
        # Initialize command from config (vx, vy, vyaw, height)
        # If cmd_init has 3 elements, add height=0.75 as 4th element
        if len(self.cmd_init) == 3:
            self.cmd = np.concatenate([self.cmd_init.copy(), np.array([0.75], dtype=np.float32)])
        else:
            self.cmd = self.cmd_init.copy()

        # joystick input thread if available
        if get_gamepad is not None:
            self.joystick_thread = threading.Thread(
                target=self.joystick_loop, daemon=True
            )
            self.joystick_thread.start()

        # Keyboard input thread
        self.keyboard_thread = threading.Thread(
            target=self.keyboard_loop, daemon=True
        )
        self.keyboard_thread.start()

        # Lightweight state used by simplified node
        self.height_map = np.full(
            (self.N_grid_points,), float(self.height_offset), dtype=np.float32
        )
        self.robot_position = np.zeros(3, dtype=np.float32)
        self.qj = np.zeros(29, dtype=np.float32)
        self.dqj = np.zeros(29, dtype=np.float32)
        self.omega = np.zeros(3, dtype=np.float32)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.action = np.zeros(29, dtype=np.float32)
        self.action_obs = np.zeros(13, dtype=np.float32)
        # Policy enable flag (wait until simulator publishes control_enable)
        self.control_enabled = False
        # whether policy stepping has actually started (for one-time print)
        self._policy_started = False
        # Track last cache frame per velocity index to detect wraps
        # Plotting state
        self.plot_enabled = False
        self.plot_max_points = 1000
        self.plot_update_interval = 0.05
        self._last_plot_update_time = 0.0
        self._last_plot_sample_time: Optional[float] = None
        self.plot_time_hist: List[float] = []
        self.plot_cmd_joint0_hist: List[float] = []
        self.plot_act_joint0_hist: List[float] = []
        self.plot_cmd_vel0_hist: List[float] = []
        self.plot_act_vel0_hist: List[float] = []
        if MATPLOTLIB_AVAILABLE:
            try:
                plt.ion()
                self.fig, (self.ax_pos, self.ax_vel) = plt.subplots(
                    2, 1, sharex=True, figsize=(8, 6)
                )
                self.pos_cmd_line, = self.ax_pos.plot([], [], label="cmd joint0")
                self.pos_act_line, = self.ax_pos.plot([], [], label="act joint0")
                self.ax_pos.set_ylabel("Joint0 pos (rad)")
                self.ax_pos.legend(loc="upper right")
                self.vel_cmd_line, = self.ax_vel.plot([], [], label="cmd vel0")
                self.vel_act_line, = self.ax_vel.plot([], [], label="act vel0")
                self.ax_vel.set_ylabel("Joint0 vel (rad/s)")
                self.ax_vel.set_xlabel("Time (s)")
                self.ax_vel.legend(loc="upper right")
                self.plot_enabled = True
            except Exception as exc:
                print(
                    f"[RLPolicyNode] Failed to initialize command plot: {exc}",
                    flush=True,
                )
                self.plot_enabled = False

        # --- ADDED: IK solver availability and simple defaults ---
        # Prefer robot_xml_path as the MuJoCo model used by the IK solver if available
        self.ik_model_path = getattr(self, "robot_xml_path", None)
        self.ik_enabled = G1_KINEMATICS_AVAILABLE and (
            self.ik_model_path is not None and os.path.isfile(self.ik_model_path)
        )
        if self.ik_enabled:
            self.get_logger().info(
                f"G1 analytical kinematics available, model: {self.ik_model_path}"
            )
        else:
            # keep node alive even if IK not installed
            if not G1_KINEMATICS_AVAILABLE:
                self.get_logger().warning(
                    "G1 kinematics module not available; IK fallbacks disabled."
                )
            else:
                self.get_logger().warning(
                    "IK model path not found; IK fallbacks disabled."
                )

        # IK candidate placeholders (ONNX/Isaac order)
        # These will be filled each control step (policy_step) and are used by obs_ik_* functions.
        self.ik_traj = None  # ndarray shape (T, num_joints) in ONNX order or None
        self.ik_joint_pos = (
            self.isaac_to_mujoco(self.default_joint_pos)
            if hasattr(self, "default_joint_pos")
            else np.zeros(getattr(self, "num_joints", 29), dtype=np.float32)
        )

        # IK cache (pre-generated grid) variables
        # explicit typing: entries are Optional[ (pos_traj, vel_traj) ]
        self.ik_cache: List[
            Optional[
                Tuple[
                    np.ndarray,
                    np.ndarray,
                ]
            ]
        ] = []  # list of (pos_traj, vel_traj) or None
        self.velocity_grid = None  # numpy array shape (G,2) for (vx, vy)
        self.velocity_to_index = {}  # map (vx,vy) -> index in ik_cache
        self.velocity_granularity = 0.05  # resolution for cache grid
        self.lin_vel_x_range = (-self.max_cmd[0], self.max_cmd[0])
        self.lin_vel_y_range = (-self.max_cmd[1], self.max_cmd[1])

        # --- NEW: IK cache file path ---
        self.ik_cache_path = os.path.join(
            os.path.dirname(self.robot_xml_path),
            f"ik_cache_{self.velocity_granularity:.2f}.npz"
        ) if self.ik_enabled else None

        # Pre-generate or load the IK cache if IK available
        if self.ik_enabled:
            try:
                loaded = self._load_ik_cache()
                # loaded = False  # TODO change back when done
                if not loaded:
                    self._pre_generate_ik_cache()
                    self._save_ik_cache()
            except Exception as e:
                self.get_logger().warning(f"IK cache load/generation failed: {e}")
                # leave cache empty and continue

    def load_config(self):
        G1_RL_ROOT_DIR = os.getenv("G1_RL_ROOT_DIR")
        with open(f"{G1_RL_ROOT_DIR}/deploy/configs/{self.config_file}") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # Keep only keys present in the YAML
        self.policy_path = config["policy_path"].replace(
            "{G1_RL_ROOT_DIR}", G1_RL_ROOT_DIR
        )
        self.use_gpu = config.get("use_gpu", False)
        self.use_height_map = config.get("use_height_map", False)

        self.xml_path = config.get("xml_path", "").replace(
            "{G1_RL_ROOT_DIR}", G1_RL_ROOT_DIR
        )
        self.robot_xml_path = config.get("robot_xml_path", self.xml_path).replace(
            "{G1_RL_ROOT_DIR}", G1_RL_ROOT_DIR
        )

        self.grid_size_x = float(config.get("grid_size_x", 1.0))
        self.grid_size_y = float(config.get("grid_size_y", 1.0))
        self.resolution = float(config.get("resolution", 0.1))
        self.forward_offset = float(config.get("forward_offset", 0.0))
        self.grid_points_x = int(config.get("grid_points_x", 1))
        self.grid_points_y = int(config.get("grid_points_y", 1))
        self.N_grid_points = int(
            config.get("N_grid_points", self.grid_points_x * self.grid_points_y)
        )
        self.height_offset = float(config.get("height_offset", 0.5))

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

        # --- NEW: Load p_gain and d_gain from metadata ---
        joint_stiffness = parsed_meta.get("joint_stiffness", None)
        joint_damping = parsed_meta.get("joint_damping", None)
        if joint_stiffness is None or joint_damping is None:
            raise RuntimeError(
                "Embedded ONNX metadata missing required 'joint_stiffness' or 'joint_damping' entry."
            )
        try:
            self.p_gain = np.array(joint_stiffness, dtype=np.float32)
            self.d_gain = np.array(joint_damping, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(
                f"Failed to convert joint_stiffness/damping from metadata to numpy array: {e}"
            )

        # Infer joint dimension from default_joint_pos
        self.num_joints = len(default_joint_pos)
        self.qj_shape = (self.num_joints,)
        self.dqj_shape = (self.num_joints,)

        self.observation_names = parsed_meta.get("observation_names", [])
        self.command_names = parsed_meta.get("command_names", [])

        # --- Parse joint order from XML and create mapping to ONNX metadata joint order ---
        self.xml_joint_names = self._parse_joint_names_from_xml(self.robot_xml_path)
        self.xml_to_onnx_joint_indices, self.onnx_to_xml_joint_indices = (
            self._create_joint_order_mappings(self.xml_joint_names, self.joint_names)
        )
        
        # Identify wrist and waist joint indices in ONNX order to zero out
        self.zero_joint_indices = []
        zero_joint_patterns = ['waist_roll', 'waist_pitch']
        for i, name in enumerate(self.joint_names):
            if any(pattern in name for pattern in zero_joint_patterns):
                self.zero_joint_indices.append(i)
        print(f"Zero joint indices (ONNX order): {self.zero_joint_indices}")

        self.free_joint_indices = []
        # free_joint_patterns = ['left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint', 'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint']
        free_joint_patterns = []
        for i, name in enumerate(self.joint_names):
            if any(pattern in name for pattern in free_joint_patterns):
                self.free_joint_indices.append(i)
        print(f"Free joint indices (ONNX order): {self.free_joint_indices}")

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
        print(f"p_gain: {self.isaac_to_mujoco(self.p_gain)}")
        print(f"d_gain: {self.isaac_to_mujoco(self.d_gain)}")
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

    def _save_ik_cache(self):
        """Save IK cache, velocity grid, and velocity_to_index to disk."""
        if self.ik_cache_path is None:
            return
        try:
            # Save as npz: ik_cache (object array), velocity_grid, velocity_to_index (as keys/indices)
            ik_cache_arr = np.array(self.ik_cache, dtype=object)
            np.savez(
                self.ik_cache_path,
                ik_cache=ik_cache_arr,
                velocity_grid=self.velocity_grid,
                velocity_to_index_keys=np.array(list(self.velocity_to_index.keys())),
                velocity_to_index_vals=np.array(list(self.velocity_to_index.values())),
            )
            print(f"\nIK cache saved to {self.ik_cache_path}")
        except Exception as e:
            print(f"\nFailed to save IK cache: {e}")

    def _load_ik_cache(self):
        """Try to load IK cache from disk. Returns True if loaded, False otherwise."""
        if self.ik_cache_path is None or not os.path.isfile(self.ik_cache_path):
            return False
        try:
            print(f"Loading IK cache from {self.ik_cache_path}")
            data = np.load(self.ik_cache_path, allow_pickle=True)
            raw_cache = list(data["ik_cache"])
            coerced_cache = []
            for entry in raw_cache:
                pos = vel = None
                if isinstance(entry, (list, tuple)) and len(entry) == 2:
                    pos, vel = entry
                elif isinstance(entry, np.ndarray) and entry.dtype == object and entry.size == 2:
                    pos, vel = entry[0], entry[1]
                else:
                    raise ValueError(
                        "IK cache entries must be (position, velocity) pairs."
                    )
                pos_arr = np.asarray(pos, dtype=np.float32)
                vel_arr = np.asarray(vel, dtype=np.float32)
                if pos_arr.shape != vel_arr.shape:
                    raise ValueError(
                        "IK cache entry has mismatched position and velocity shapes."
                    )
                coerced_cache.append((pos_arr, vel_arr))
            self.ik_cache = coerced_cache
            self.velocity_grid = data["velocity_grid"].astype(np.float32, copy=False)
            keys = data["velocity_to_index_keys"]
            vals = data["velocity_to_index_vals"]
            self.velocity_to_index = {tuple(map(float, k)): int(v) for k, v in zip(keys, vals)}
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to load IK cache: {e}") from e

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

    def _pre_generate_ik_cache(self):
        """Pre-generate IK trajectories for a grid of (vx, vy) and store ONNX-ordered trajectories and velocities."""
        # build grid
        vx_min, vx_max = self.lin_vel_x_range
        vy_min, vy_max = self.lin_vel_y_range
        step = self.velocity_granularity
        vx_samples = np.arange(vx_min, vx_max + 1e-8, step, dtype=np.float32)
        vy_samples = np.arange(vy_min, vy_max + 1e-8, step, dtype=np.float32)
        combos = []
        for vx in vx_samples:
            for vy in vy_samples:
                key = (round(float(vx), 3), round(float(vy), 3))
                combos.append(key)
        G = len(combos)
        self.velocity_grid = np.array([[c[0], c[1]] for c in combos], dtype=np.float32)
        self.velocity_to_index = {combos[i]: i for i in range(G)}
        # Now each entry will be (pos_traj, vel_traj)
        self.ik_cache = [None] * G

        # --- Progress bar setup ---
        print(f"Generating IK cache for {G} velocity combinations...")
        bar_width = 40
        last_percent = -1

        # Generate trajectories (store entire traj in ONNX order and precompute velocities)
        for i, (vx, vy) in enumerate(combos):
            try:
                traj_data = self.solve_ik_velocity(vx, vy, 0.0, step_period=0.5)
                if traj_data is None:
                    # fallback: single-frame default joints
                    T = 1
                    pos = np.tile(self.default_joint_pos.astype(np.float32), (T, 1))
                else:
                    joint_traj = traj_data
                    pos = joint_traj.astype(np.float32)  # (T, num_joints) in ONNX order

                # compute per-frame velocities using helper
                dt = float(getattr(self, "control_dt", 0.02))
                vel = self._compute_traj_velocities(pos, dt)

                # store tuple (pos_traj, vel_traj)
                self.ik_cache[i] = (pos, vel)
            except Exception:
                T = 1
                pos = np.tile(self.default_joint_pos.astype(np.float32), (T, 1))
                vel = np.zeros_like(pos, dtype=np.float32)
                self.ik_cache[i] = (pos, vel)
            # --- Progress bar update ---
            percent = int((i + 1) * 100 / G)
            if percent != last_percent:
                filled = int(bar_width * percent / 100)
                bar = "[" + "#" * filled + "-" * (bar_width - filled) + "]"
                print(f"\r{bar} {percent}% ({i+1}/{G})", end="", flush=True)
                last_percent = percent
        print("\nIK cache generation complete.")

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

    # --- NEW: helper that calls G1 IK and returns joint trajectory in ONNX order ---
    def solve_ik_velocity(
        self,
        vel_x: float,
        vel_y: float = 0.0,
        vel_yaw: float = 0.0,
        step_period: float = 0.5,
    ):
        """
        Use g1_kinematics.solve_kinematics_multi_step to compute a trajectory
        for the requested velocity. Returns joint position trajectory in ONNX/Isaac order:
            traj_onnx: ndarray shape (T, num_joints)
        If IK is unavailable or fails, returns None.
        """
        if not self.ik_enabled:
            return None
        try:
            # Build SolverInput similar to commands_ik_cached
            solver_input = g1_kinematics.SolverInput()
            # target step length based on velocity * step_period
            solver_input.target_step_length = [
                float(vel_x * step_period),
                float(vel_y * step_period),
                0.0,
            ]
            is_stationary = math.hypot(float(vel_x), float(vel_y)) < 0.01
            solver_input.target_step_height = 0.0 if is_stationary else 0.15
            # default base and joint pose approximations (use metadata default_joint_pos if present)
            base_pos_global = [0.0, 0.0, float(getattr(self, "pelvis_height", 0.75))]
            base_quat = [1.0, 0.0, 0.0, 0.0]
            if hasattr(self, "default_joint_pos"):
                joint0 = (
                    list(self.default_joint_pos.tolist())
                    if isinstance(self.default_joint_pos, np.ndarray)
                    else list(self.default_joint_pos)
                )
            else:
                joint0 = [0.0] * getattr(self, "num_joints", 29)
            solver_input.base0_pos = base_pos_global
            solver_input.base0_quat = base_quat
            solver_input.joint0_pos = self.isaac_to_mujoco(joint0)
            solver_input.dt = float(getattr(self, "control_dt", 0.02))
            standing_T = 300
            solver_input.T = standing_T if is_stationary else int(step_period / solver_input.dt)
            solver_input.foot = 2 if vel_y >= 0.0 else 3  # left or right foot swing first
            foot_width = 0.2
            # anchor/swing foot positions relative to base
            if vel_y >= 0.0:
                anchor = [
                    base_pos_global[0],
                    base_pos_global[1] - foot_width / 2.0,
                    0.0,
                ]
                swing = [base_pos_global[0], base_pos_global[1] + foot_width / 2.0, 0.0]
            else:
                anchor = [
                    base_pos_global[0],
                    base_pos_global[1] + foot_width / 2.0,
                    0.0,
                ]
                swing = [base_pos_global[0], base_pos_global[1] - foot_width / 2.0, 0.0]
            solver_input.anchor_foot_wpos = anchor
            solver_input.swing_foot_wpos = swing

            # Call IK solver
            traj = g1_kinematics.solve_kinematics_multi_step(
                "/home/yangl/BeyondMimic/g1_ctrl/unitree_robots/g1/scene_29dof.xml", solver_input, solver_input.dt * solver_input.T
            )
            trajectory_np = traj.as_numpy().T.astype(np.float32)  # shape (36, T)

            # Extract joint portion (MuJoCo/XML order rows 7:36)
            if trajectory_np.shape[0] >= 36:
                mujoco_joints = trajectory_np[7:36, :]  # (29, T)
            else:
                # handle shorter outputs robustly
                n_j = getattr(
                    self,
                    "num_joints",
                    mujoco_joints.shape[0] if "mujoco_joints" in locals() else 29,
                )
                mujoco_joints = np.zeros(
                    (n_j, trajectory_np.shape[1]), dtype=np.float32
                )

            # convert to ONNX/Isaac order per time-step using existing mapping helpers
            T = mujoco_joints.shape[1]
            traj_onnx = np.zeros((T, len(self.joint_names)), dtype=np.float32)
            for t in range(T):
                # mujoco_to_isaac expects array in XML order -> returns ONNX order
                traj_onnx[t, :] = self.mujoco_to_isaac(mujoco_joints[:, t])
            return traj_onnx
        except Exception as e:
            self.get_logger().warning(f"IK solver failed: {e}")
            return None

    def policy_step(self):
        """
        Compute or lookup an IK candidate (cached) and expose it as observations,
        then run ONNX inference (which may include the IK candidate via metadata).
        IK is an observation candidate, not only a fallback.
        """
        # Do not run policy until simulator signals control_enable
        if not getattr(self, "control_enabled", False):
            # Keep alive but skip heavy processing
            return
        # One-time print when policy actually starts running
        if not getattr(self, '_policy_started', False):
            print(f"[RLPolicyNode] Policy stepping started at sim_time={getattr(self,'sim_time',-1):.3f}", flush=True)
            self._policy_started = True
        # Build velocity command and fetch nearest cached IK candidate
        # Run ONNX inference (the policy may reference obs_ik_* in its observation_names)
        if hasattr(self, "onnx_sess") and self.onnx_sess is not None:
            try:
                out0 = self._run_onnx_inference()  # numpy ndarray
                # Prepare action, p gain, d gain arrays from metadata
                action = out0.astype(np.float32).ravel()
                # Zero out wrist and waist actions
                # if hasattr(self, 'zero_joint_indices'):
                #     action[self.zero_joint_indices] = 0.0
                # Store the 13-element action for observation
                self.action_obs = action.copy()
                
                p_gain = self.p_gain.astype(np.float32)
                d_gain = self.d_gain.astype(np.float32)
                # Concatenate [action, p_gain, d_gain]
                msg = Float32MultiArray()
                
                # Map 13-element policy action to 29-element ONNX joint array
                # Policy controls these joints in order:
                # [left_hip_pitch, right_hip_pitch, waist_yaw, left_hip_roll, right_hip_roll,
                #  left_hip_yaw, right_hip_yaw, left_knee, right_knee,
                #  left_ankle_pitch, right_ankle_pitch, left_ankle_roll, right_ankle_roll]
                # ONNX indices: [0, 1, 2, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18]
                policy_to_onnx_indices = [0, 1, 2, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18]
                
                scaled_action = np.zeros(29, dtype=np.float32)
                for policy_idx, onnx_idx in enumerate(policy_to_onnx_indices):
                    scaled_action[onnx_idx] = action[policy_idx] * self.action_scale[policy_idx]
                
                joint_pos = self.default_joint_pos + scaled_action
                if hasattr(self, 'free_joint_indices'):
                    joint_pos[self.free_joint_indices] = self.default_joint_pos[self.free_joint_indices]
                # print("joint pos mujoco", self.isaac_to_mujoco(joint_pos))
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
                # ONNX inference failed; fall back to IK-first-frame or default
                print(
                    f"ONNX inference failed: {e} — falling back to IK/default publish"
                )
                exit(0)

    def joystick_loop(self):
        # Xbox controller axes: left stick (ABS_X, ABS_Y), right stick (ABS_RX)
        # Typical value range: -32768 to 32767
        vx, vy, vyaw = 0.0, 0.0, 0.0
        while True:
            try:
                events = get_gamepad()
                for event in events:
                    if event.ev_type == "Absolute":
                        if event.code == "ABS_X":  # Left stick horizontal
                            vy = -event.state / 32768.0  # normalize to [-1, 1]
                        elif event.code == "ABS_Y":  # Left stick vertical
                            vx = -event.state / 32768.0  # invert so up is positive
                        elif event.code == "ABS_RX":  # Right stick horizontal
                            vyaw = event.state / 32768.0
                        # D-pad up/down for height control (ABS_HAT0Y: -1=up, 1=down, 0=neutral)
                        elif event.code == "ABS_HAT0Y":
                            with self.cmd_lock:
                                if event.state == -1:  # D-pad up
                                    self.cmd[3] = min(0.75, self.cmd[3] + 0.01)
                                elif event.state == 1:  # D-pad down
                                    self.cmd[3] = max(0.55, self.cmd[3] - 0.01)
                with self.cmd_lock:
                    self.cmd[0] = vx * self.cmd_scale[0] * self.max_cmd[0]
                    self.cmd[1] = vy * self.cmd_scale[1] * self.max_cmd[1]
                    self.cmd[2] = -vyaw * self.cmd_scale[2] * self.max_cmd[2]
                    # cmd[3] (height) is only modified by D-pad, keep existing value
            except Exception:
                # Avoid crashing the thread
                pass

    def keyboard_loop(self):
        """
        Reads keyboard input from stdin (non-blocking) and updates commands.
        WASD: Linear Velocity (X/Y)
        Arrow Keys: Height (Up/Down) and Yaw (Left/Right)
        """
        def get_key():
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                # Use setcbreak to preserve signals (Ctrl+C) and output formatting
                tty.setcbreak(fd)
                rlist, _, _ = select.select([fd], [], [], 0.1)
                if rlist:
                    key = os.read(fd, 1)
                    if key == b'\x1b':  # Escape sequence for arrows
                        # Check if there are more characters for escape sequence
                        rlist2, _, _ = select.select([fd], [], [], 0.05)
                        if rlist2:
                            key += os.read(fd, 2)
                    return key.decode('utf-8', errors='ignore')
                return None
            except KeyboardInterrupt:
                return None
            except Exception as e:
                print(f"Error reading key: {e}", flush=True)
                return None
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        print("[RLPolicyNode] Keyboard control enabled: WASD for move, Arrows for Height/Yaw", flush=True)
        
        # Internal state for smooth control
        target_vx, target_vy, target_vyaw = 0.0, 0.0, 0.0
        
        while True:
            try:
                key = get_key()
                if key:
                    with self.cmd_lock:
                        # WASD for Linear Velocity
                        if key == 'w':
                            self.cmd[0] = min(self.max_cmd[0], self.cmd[0] + 0.05)
                        elif key == 's':
                            self.cmd[0] = max(-self.max_cmd[0], self.cmd[0] - 0.05)
                        elif key == 'a':
                            self.cmd[1] = min(self.max_cmd[1], self.cmd[1] + 0.05)
                        elif key == 'd':
                            self.cmd[1] = max(-self.max_cmd[1], self.cmd[1] - 0.05)
                        
                        # Arrow Keys for Height and Yaw
                        elif key == '\x1b[A':  # Up Arrow
                            self.cmd[3] = min(0.75, self.cmd[3] + 0.01)
                        elif key == '\x1b[B':  # Down Arrow
                            self.cmd[3] = max(0.52, self.cmd[3] - 0.01)
                        elif key == '\x1b[D':  # Left Arrow
                            self.cmd[2] = min(self.max_cmd[2], self.cmd[2] + 0.05)
                        elif key == '\x1b[C':  # Right Arrow
                            self.cmd[2] = max(-self.max_cmd[2], self.cmd[2] - 0.05)
                        
                        # Space to stop
                        elif key == ' ':
                            self.cmd[0] = 0.0
                            self.cmd[1] = 0.0
                            self.cmd[2] = 0.0
                            
                        # print(f"Cmd: vx={self.cmd[0]:.2f}, vy={self.cmd[1]:.2f}, yaw={self.cmd[2]:.2f}, h={self.cmd[3]:.2f}", flush=True)

            except Exception as e:
                print(f"Keyboard input error: {e}")
                exit()

    def pd_control(self, target_q, q, kp, target_dq, dq, kd):
        """Calculates torques from position commands"""
        return (target_q - q) * kp + (target_dq - dq) * kd

    # Remove all pinocchio/cbf/obstacle xml parsing methods

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

    def heightmap_callback(self, msg):
        received = np.array(msg.data, dtype=np.float32)
        self.height_map[: received.shape[0]] = received

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

    def obs_command_vel(self):
        # if self.sim_time > 10 and self.sim_time <= 14:
        #     self.cmd[0] = 0.6
        # elif self.sim_time > 10 and self.sim_time <= 15:
        #     self.cmd[0] = 0.4
        #     self.cmd[2] = 0.2
        # if self.sim_time > 8:
        #     self.cmd[0] = 0.75
        # if self.sim_time > 11:
        #     self.cmd[0] = 0.3
        # if self.sim_time > 14:
        #     self.cmd[0] = 0.0
        # if self.sim_time > 17:
        #     self.cmd[0] = -0.3
        # if self.sim_time > 20:
        #     self.cmd[0] = -0.5
        # if self.sim_time > 23:
        #     self.cmd[0] = -0.3
        # if self.sim_time > 26:
        #     self.cmd[0] = 0.0
        # self.cmd[2:] = 0
        # if self.cmd[0] < 0.1 and self.cmd[0] > -0.1:
        #     self.cmd[0] = 0.0
        # if self.cmd[1] < 0.1 and self.cmd[1] > -0.1:
        #     self.cmd[1] = 0.0
        # if self.cmd[2] < 0.05 and self.cmd[2] > -0.05:
        #     self.cmd[2] = 0.0
        # print("cmd:", self.cmd)
        # Return all 4 command elements: [vx, vy, vyaw, height]
        # self.cmd[0] = 0.0
        # self.cmd[1] = 0.0 
        # self.cmd[2] = 0.0
        # self.cmd[3] = 0.75
        print("cmd:", self.cmd)
        return self.cmd[:4]

    def _compute_command_cache(
        self, vel: np.ndarray, step: int
    ) -> Optional[np.ndarray]:
        if not (self.ik_enabled and self.velocity_grid is not None):
            return None
        found = self._find_closest_cached_velocity(
            float(vel[0]), float(vel[1]) if len(vel) > 1 else 0.0
        )
        if found is None:
            return None
        _, _, idx = found
        if idx is None or not (0 <= idx < len(self.ik_cache)):
            return None
        entry = self.ik_cache[idx]
        if entry is None:
            return None
        pos_traj = entry[0] if len(entry) >= 1 else None
        vel_traj = entry[1] if len(entry) >= 2 else None
        if pos_traj is None or pos_traj.size == 0:
            return None
        T = pos_traj.shape[0]
        if T <= 0:
            return None
        frame = step % T
        print("frame:", frame)

        ik_pos = pos_traj[frame]
        if vel_traj is not None and vel_traj.shape[0] == T:
            ik_vel = vel_traj[frame]
        else:
            ik_vel = np.zeros_like(ik_pos)
        # command_vec = np.concatenate([ik_pos, ik_vel], axis=-1)
        command_vec = ik_pos
        max_abs = float(np.max(np.abs(command_vec))) if command_vec.size > 0 else 0.0
        if max_abs > 1.0:
            print(
                f"[RLPolicyNode] Large command magnitude detected (|cmd|_max={max_abs:.2f}) at sim_time={self.sim_time:.3f}, frame={frame}",
                flush=True,
            )
        self._record_command_stats(command_vec)
        return command_vec

    def _record_command_stats(self, command_vec: Optional[np.ndarray]) -> None:
        if not (self.plot_enabled and command_vec is not None):
            return
        if command_vec.size < 2 * self.num_joints:
            return
        t = float(getattr(self, "sim_time", 0.0))
        if (
            self._last_plot_sample_time is not None
            and abs(t - self._last_plot_sample_time) < 1e-6
        ):
            return
        self._last_plot_sample_time = t
        cmd_pos = command_vec[: self.num_joints]
        cmd_vel = command_vec[self.num_joints : 2 * self.num_joints]
        actual_pos = self.mujoco_to_isaac(self.qj)
        actual_vel = self.mujoco_to_isaac(self.dqj)
        joint0_cmd = float(cmd_pos[0]) if cmd_pos.size > 0 else 0.0
        joint0_act = float(actual_pos[0]) if actual_pos.size > 0 else 0.0
        joint0_cmd_vel = float(cmd_vel[0]) if cmd_vel.size > 0 else 0.0
        joint0_act_vel = float(actual_vel[0]) if actual_vel.size > 0 else 0.0
        self.plot_time_hist.append(t)
        self.plot_cmd_joint0_hist.append(joint0_cmd)
        self.plot_act_joint0_hist.append(joint0_act)
        self.plot_cmd_vel0_hist.append(joint0_cmd_vel)
        self.plot_act_vel0_hist.append(joint0_act_vel)
        if len(self.plot_time_hist) > self.plot_max_points:
            self.plot_time_hist = self.plot_time_hist[-self.plot_max_points :]
            self.plot_cmd_joint0_hist = self.plot_cmd_joint0_hist[
                -self.plot_max_points :
            ]
            self.plot_act_joint0_hist = self.plot_act_joint0_hist[
                -self.plot_max_points :
            ]
            self.plot_cmd_vel0_hist = self.plot_cmd_vel0_hist[-self.plot_max_points :]
            self.plot_act_vel0_hist = self.plot_act_vel0_hist[-self.plot_max_points :]
        if (
            t - self._last_plot_update_time
        ) >= self.plot_update_interval:
            self._last_plot_update_time = t
            self.pos_cmd_line.set_data(self.plot_time_hist, self.plot_cmd_joint0_hist)
            self.pos_act_line.set_data(self.plot_time_hist, self.plot_act_joint0_hist)
            self.vel_cmd_line.set_data(self.plot_time_hist, self.plot_cmd_vel0_hist)
            self.vel_act_line.set_data(self.plot_time_hist, self.plot_act_vel0_hist)
            self.ax_pos.relim()
            self.ax_pos.autoscale_view()
            self.ax_vel.relim()
            self.ax_vel.autoscale_view()
            self.fig.canvas.draw_idle()
            plt.pause(0.001)

    def obs_command(self):
        # Find the closest cached IK trajectory to the current command velocity
        vel = self.obs_command_vel()
        step = int(self.sim_time / self.control_dt)
        return self._compute_command_cache(vel, step)

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

    def obs_base_ang_vel(self):
        # print("omega:", self.omega)
        return self.omega

    def obs_joint_pos(self):
        # print("qj:", self.qj)
        joint_pos = self.mujoco_to_isaac(self.qj)
        # Zero out wrist and waist joint positions
        if hasattr(self, 'zero_joint_indices'):
            joint_pos[self.zero_joint_indices] = self.default_joint_pos[self.zero_joint_indices]    
        if hasattr(self, 'free_joint_indices'):
            joint_pos[self.free_joint_indices] = self.default_joint_pos[self.free_joint_indices]
        joint_pos -= self.default_joint_pos
        return joint_pos
    
    def obs_joint_pos_specify(self):
        # Extract the same 13 leg joints that the policy controls
        # ONNX indices: [0, 1, 2, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18]
        full_joint_pos = self.obs_joint_pos()
        policy_joint_indices = [0, 1, 2, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18]
        return np.array([full_joint_pos[i] for i in policy_joint_indices], dtype=np.float32)

    def obs_joint_vel(self):
        # print("dqj:", self.dqj)
        joint_vel = self.mujoco_to_isaac(self.dqj)
        # Zero out wrist and waist joint velocities
        if hasattr(self, 'zero_joint_indices'):
            joint_vel[self.zero_joint_indices] = 0.0
        if hasattr(self, 'free_joint_indices'):
            joint_vel[self.free_joint_indices] = 0.0
        return joint_vel

    def obs_joint_vel_specify(self):
        # Extract the same 13 leg joints that the policy controls
        # ONNX indices: [0, 1, 2, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18]
        full_joint_vel = self.obs_joint_vel()
        policy_joint_indices = [0, 1, 2, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18]
        return np.array([full_joint_vel[i] for i in policy_joint_indices], dtype=np.float32)

    def obs_actions(self):
        return self.action_obs

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
        return np.concatenate(obs_list, axis=-1)


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