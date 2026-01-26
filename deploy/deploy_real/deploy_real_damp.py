# Copyright (c) 2025 Lizhi Yang AMBER LAB

import atexit
import os
import subprocess
import tempfile
import textwrap
import threading
import time
import xml.etree.ElementTree as ET
from enum import Enum
import yaml
import onnx
import json
import ast
try:
    import onnxruntime as ort
except Exception:
    ort = None

import numpy as np

# ROS2 imports
import rclpy
import tf2_ros
from common.command_helper import (
    MotorMode,
    create_damping_cmd,
    create_zero_cmd,
    init_cmd_go,
    init_cmd_hg,
)
from common.remote_controller import KeyMap, RemoteController
# from common.rotation_helper import get_gravity_orientation, transform_imu_data
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import GridCells
# from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
# from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from sensor_msgs.msg import JointState  # add this import
# from std_msgs.msg import Float32, Float32MultiArray
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)

# from deploy.deploy_mujoco.deploy_mujoco_sync import remove_wrist
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__LowCmd_,
    unitree_go_msg_dds__LowState_,
    unitree_go_msg_dds__SportModeState_,
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC

# import pinocchio as pin


class PolicyModes(Enum):
    Walk = 0
    # Stair = 1
    # VariedHeight = 2
    # StairVision = 3


class ControllerState(Enum):
    ZERO_TORQUE = 0
    DAMPING = 1
    DEFAULT_POSE = 2
    POLICY = 3


class G1JointIndex:
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11
    WaistYaw = 12
    WaistRoll = 13        # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistA = 13           # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistPitch = 14       # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistB = 14           # NOTE: INVALID for g1 23dof/29dof with waist locked
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20   # NOTE: INVALID for g1 23dof
    LeftWristYaw = 21     # NOTE: INVALID for g1 23dof
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27  # NOTE: INVALID for g1 23dof
    RightWristYaw = 28    # NOTE: INVALID for g1 23dof



def load_robot_description_text(urdf_path: str) -> str:
    """Return full URDF XML text. If the path ends with .xacro and xacro is installed, expand it."""
    # if urdf_path.endswith(".xacro"):
    #     try:
    #         import xacro
    #         return xacro.process_file(urdf_path).toxml()
    #     except Exception as e:
    #         print(f"Warning: xacro expansion failed for {urdf_path}: {e}. Trying to read as XML file.")
    with open(urdf_path) as f:
        return f.read()


class Controller(Node):
    def __init__(self, config_file="g1_21dof.yaml") -> None:
        # Initialize ROS2 node
        super().__init__("g1_controller")
        self.config_file = config_file

        # --- Use RLPolicyNode-style config loading ---
        self.load_config()
        self.load_policy()
        self.load_policy_metadata()

        self._rsp_params_file = None
        try:
            robot_description_xml = load_robot_description_text(self.urdf_path)

            # Build a minimal ROS 2 params YAML with a block scalar for the URDF XML
            yaml_body = "robot_state_publisher:\n  ros__parameters:\n    robot_description: |\n" + textwrap.indent(
                robot_description_xml, "      "
            )

            tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml")
            tmp.write(yaml_body)
            tmp.flush()
            tmp.close()
            self._rsp_params_file = tmp.name
            self.get_logger().info(f"robot_state_publisher params file: {self._rsp_params_file}")
        except Exception as e:
            self.get_logger().error(f"Failed to prepare robot_state_publisher params file: {e}")

        # --- Start robot_state_publisher WITH params file ---
        self.rsp_proc = None
        try:
            cmd = ["ros2", "run", "robot_state_publisher", "robot_state_publisher"]
            if self._rsp_params_file:
                cmd += ["--ros-args", "--params-file", self._rsp_params_file]
            self.rsp_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            atexit.register(self._stop_rsp)
            self.get_logger().info("robot_state_publisher started")
        except Exception as e:
            self.get_logger().error(f"Failed to start robot_state_publisher: {e}")

        # --- Publish a static world -> base frame (optional, helps RViz Fixed Frame) ---
        base_frame = "base_link"
        # world_frame = getattr(self.config, "world_frame", "world")
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "world"
        t.child_frame_id = "base_link"
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.static_broadcaster.sendTransform(t)
        t1 = TransformStamped()
        t1.header.stamp = self.get_clock().now().to_msg()
        t1.header.frame_id = base_frame
        t1.child_frame_id = "pelvis_link"
        t1.transform.translation.x = 0.0
        t1.transform.translation.y = 0.0
        t1.transform.translation.z = 0.0
        t1.transform.rotation.x = 0.0
        t1.transform.rotation.y = 0.0
        t1.transform.rotation.z = 0.0
        t1.transform.rotation.w = 1.0
        self.static_broadcaster.sendTransform(t1)
        t2 = TransformStamped()
        t2.header.stamp = self.get_clock().now().to_msg()
        t2.header.frame_id = "map"
        t2.child_frame_id = "world"
        t2.transform.translation.x = 0.0
        t2.transform.translation.y = 0.0
        t2.transform.translation.z = 0.0
        t2.transform.rotation.x = 0.0
        t2.transform.rotation.y = 0.0
        t2.transform.rotation.z = 0.0
        t2.transform.rotation.w = 1.0
        self.static_broadcaster.sendTransform(t2)

        # ROS2 publishers and subscribers
        self.cmd_vel_publisher = self.create_publisher(Twist, "/cmd_vel_raw", 10)
        self.cmd_vel_subscriber = self.create_subscription(
            Twist, "/cmd_vel_filtered", self.cmd_vel_filtered_callback, 10
        )
        if self.use_height_map:
            self.height_map_subscriber = self.create_subscription(
                GridCells, self.height_map_topic, self.height_map_callback, 10
            )
        # Store filtered velocity for history updates
        self.filtered_cmd = np.zeros(3, dtype=np.float32)
        self.filtered_cmd_received = False  # Flag to track if we've received filtered commands

        # Start ROS2 spinning in a separate thread
        self.ros_thread = threading.Thread(target=self._spin_ros, daemon=True)
        self.ros_running = True
        self.ros_thread.start()

        self.remote_controller = RemoteController()
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result["name"]:
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)
        print("Successfully released mode.")

        # Initialize state machine
        self.current_state = ControllerState.ZERO_TORQUE
        self.prev_button_states = {}
        for key in KeyMap.__dict__:
            if not key.startswith("_"):
                self.prev_button_states[getattr(KeyMap, key)] = 0

        # --- State variables for observation ---
        self.qj = np.zeros(self.num_joints, dtype=np.float32)
        self.dqj = np.zeros(self.num_joints, dtype=np.float32)
        self.omega = np.zeros(3, dtype=np.float32)
        self.action = np.zeros(self.num_joints, dtype=np.float32)
        # Initialize command (vx, vy, vyaw, height)
        if len(self.cmd_init) == 3:
            self.cmd = np.concatenate([self.cmd_init.copy(), np.array([0.75], dtype=np.float32)])
        else:
            self.cmd = self.cmd_init.copy()
        self.counter = 0

        if self.use_height_map:
            self.height_map_obs = np.ones(self.N_grid_points, dtype=np.float32) * 0.0
            self.height_map_obs = self.height_map_obs.view(self.grid_points_y, self.grid_points_x)

        if self.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.odom_state = unitree_go_msg_dds__SportModeState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 5  # TODO might be 1 on new firmware?

            self.lowcmd_publisher_ = ChannelPublisher(self.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(self.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

            # self.odom_subscriber = ChannelSubscriber("rt/lf/odommodestate", SportModeState_)
            # self.odom_subscriber.Init(self.OdomStateHandler, 10)

            self.odom_pose = np.zeros(3, dtype=np.float32)
            self.init_odom_flag = True

        elif self.msg_type == "go":
            # h1 uses the go msg type
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(self.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(self.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        if self.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif self.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.weak_motor)

        self.dof_idx_full = np.arange(self.num_joints)

        self.policy_modes = PolicyModes
        self.current_mode = self.policy_modes.Walk
        self.joint_states_pub = self.create_publisher(JointState, "/joint_states", 10)

    def load_config(self):
        G1_RL_ROOT_DIR = os.getenv("G1_RL_ROOT_DIR")
        with open(f"{G1_RL_ROOT_DIR}/deploy/configs/{self.config_file}") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.policy_path = config["policy_path"].replace("{G1_RL_ROOT_DIR}", G1_RL_ROOT_DIR)
        self.use_gpu = config.get("use_gpu", False)
        self.use_height_map = config.get("use_height_map", False)
        self.xml_path = config.get("xml_path", "").replace("{G1_RL_ROOT_DIR}", G1_RL_ROOT_DIR)
        self.robot_xml_path = config.get("robot_xml_path", self.xml_path).replace("{G1_RL_ROOT_DIR}", G1_RL_ROOT_DIR)
        self.grid_size_x = float(config.get("grid_size_x", 1.0))
        self.grid_size_y = float(config.get("grid_size_y", 1.0))
        self.resolution = float(config.get("resolution", 0.1))
        self.forward_offset = float(config.get("forward_offset", 0.0))
        self.grid_points_x = int(config.get("grid_points_x", 1))
        self.grid_points_y = int(config.get("grid_points_y", 1))
        self.N_grid_points = int(config.get("N_grid_points", self.grid_points_x * self.grid_points_y))
        self.height_offset = float(config.get("height_offset", 0.5))
        self.simulation_dt = float(config.get("simulation_dt", 0.002))
        self.control_decimation = int(config.get("control_decimation", 10))
        self.control_dt = float(config.get("control_dt", self.simulation_dt * self.control_decimation))
        self.cmd_scale = np.array(config.get("cmd_scale", [1.0, 1.0, 1.0]), dtype=np.float32)
        self.cmd_init = np.array(config.get("cmd_init", [0.0, 0.0, 0.0]), dtype=np.float32)
        self.max_cmd = np.array(config.get("max_cmd", [1.0, 1.0, 1.0]), dtype=np.float32)
        self.urdf_path = config.get("urdf_path", "").replace("{G1_RL_ROOT_DIR}", G1_RL_ROOT_DIR)
        self.lowcmd_topic = config.get("lowcmd_topic", "rt/low_cmd")
        self.lowstate_topic = config.get("lowstate_topic", "rt/low_state")
        self.msg_type = config.get("msg_type", "hg")
        self.imu_type = config.get("imu_type", "pelvis")
        self.weak_motor = config.get("weak_motor", False)

    def load_policy(self):
        self.onnx_model = None
        self.onnx_inputs = []
        self.onnx_outputs = []
        try:
            self.onnx_model = onnx.load(self.policy_path)
            onnx.checker.check_model(self.onnx_model)
            graph = self.onnx_model.graph
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
        except Exception as e:
            raise RuntimeError(
                f"Failed to load or validate ONNX model at '{self.policy_path}': {e}"
            )

    def load_policy_metadata(self):
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

        for k, v in parsed_meta.items():
            if isinstance(v, str):
                parsed_meta[k] = [item.strip() for item in v.split(",") if item.strip()]

        self.onnx_metadata = parsed_meta
        self.joint_names = parsed_meta.get("joint_names", [])
        self.joint_names_mujoco = self._parse_joint_names_from_xml(self.robot_xml_path)
        self.mujoco_to_isaac_joint_indices, self.isaac_to_mujoco_joint_indices = (
            self._create_joint_order_mappings(self.joint_names_mujoco, self.joint_names)
        )
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
            self.default_joint_pos_mujoco = self.isaac_to_mujoco(self.default_joint_pos)
        except Exception as e:
            raise RuntimeError(
                f"Failed to convert 'default_joint_pos' from metadata to numpy array: {e}"
            )

        joint_stiffness = parsed_meta.get("joint_stiffness", None)
        joint_damping = parsed_meta.get("joint_damping", None)
        if joint_stiffness is None or joint_damping is None:
            raise RuntimeError(
                "Embedded ONNX metadata missing required 'joint_stiffness' or 'joint_damping' entry."
            )
        try:
            self.p_gain = np.array(joint_stiffness, dtype=np.float32)
            self.d_gain = np.array(joint_damping, dtype=np.float32)
            self.p_gain_mujoco = self.isaac_to_mujoco(self.p_gain)
            print("p_gain_mujoco:", self.p_gain_mujoco)
            self.d_gain_mujoco = self.isaac_to_mujoco(self.d_gain)
            print("d_gain_mujoco:", self.d_gain_mujoco)
        except Exception as e:
            raise RuntimeError(
                f"Failed to convert joint_stiffness/damping from metadata to numpy array: {e}"
            )

        self.num_joints = len(default_joint_pos)
        self.qj_shape = (self.num_joints,)
        self.dqj_shape = (self.num_joints,)

        self.observation_names = parsed_meta.get("observation_names", [])
        self.command_names = parsed_meta.get("command_names", [])

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

        self.onnx_input_names = [inp["name"] for inp in self.onnx_inputs]
        self.onnx_output_names = [out["name"] for out in self.onnx_outputs]
        input_names_set = set(self.onnx_input_names)
        self.is_recurrent_onnx = "h_in" in input_names_set and "c_in" in input_names_set

        self.get_logger().info(
            "Successfully loaded ONNX model and created onnxruntime session."
        )
        
        # Identify wrist and waist joint indices in ONNX order to zero out
        self.zero_joint_indices = []
        zero_joint_patterns = ['waist_roll', 'waist_pitch']
        self.gain_add_indicies = []
        gain_add_patterns = []
        for i, name in enumerate(self.joint_names):
            if any(pattern in name for pattern in zero_joint_patterns):
                self.zero_joint_indices.append(i)
            if any(pattern in name for pattern in gain_add_patterns):
                self.gain_add_indicies.append(i)
        print(f"Zero joint indices (ONNX order): {self.zero_joint_indices}")
        print(f"Gain add indices (ONNX order): {self.gain_add_indicies}")
        
        # Map zero joint indices to MuJoCo/hardware indices for motor disabling
        self.zero_motor_indices = []
        self.gain_add_motor_indices = []
        for idx in self.zero_joint_indices:
            # Convert ONNX index to MuJoCo index
            mujoco_idx = self.isaac_to_mujoco_joint_indices[idx]
            if mujoco_idx >= 0:
                self.zero_motor_indices.append(mujoco_idx)
        for idx in self.gain_add_indicies:
            mujoco_idx = self.isaac_to_mujoco_joint_indices[idx]
            if mujoco_idx >= 0:
                self.gain_add_motor_indices.append(mujoco_idx)
        print(f"Zero motor indices (MuJoCo/hardware order): {self.zero_motor_indices}")
        print(f"Gain add motor indices (MuJoCo/hardware order): {self.gain_add_motor_indices}")
        
        print(f"mujoco joint names: {self.joint_names_mujoco}")
        print(f"p_gain: {self.isaac_to_mujoco(self.p_gain)}")
        print(f"d_gain: {self.isaac_to_mujoco(self.d_gain)}")
        print(f"observation names:", self.observation_names)
        print(f"command names:", self.command_names)

    def _stop_rsp(self):
        try:
            if self.rsp_proc and self.rsp_proc.poll() is None:
                self.rsp_proc.terminate()
                try:
                    self.rsp_proc.wait(timeout=2.0)
                except Exception:
                    self.rsp_proc.kill()
        except Exception:
            pass
        # Clean up the temporary params file
        try:
            if self._rsp_params_file:
                os.remove(self._rsp_params_file)
        except Exception:
            pass

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def OdomStateHandler(self, msg: SportModeState_):
        self.odom_state = msg
        # print("position: ", self.odom_state.position[0]-self.odom_pose[0], self.odom_state.position[1]-self.odom_pose[1], self.odom_state.position[2]-self.odom_pose[2])

    def send_cmd(self, cmd: LowCmdGo | LowCmdHG):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the L1 signal...")
        while self.current_state == ControllerState.ZERO_TORQUE:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            self.update_state_machine()
            time.sleep(self.control_dt)

    def damping_state(self):
        print("Enter damping state.")
        print("Press R1 to go to default pose, or L1 to stay in damping")
        while self.current_state == ControllerState.DAMPING:
            create_damping_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            self.update_state_machine()
            time.sleep(self.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Press R1 to go to policy mode, L1 to go to damping")

        # Move to default position first
        self.move_to_default_pos()
        print("default pos:", self.default_joint_pos_mujoco)
        kps= [100., 100., 100., 150., 40., 40.,
                100., 100., 100., 150., 40., 40.,
                300., 300., 300.,
                100., 100., 50., 50., 20., 20., 20.,
                100., 100., 50., 50., 20., 20., 20.]
        kds= [6., 6., 6., 4., 2., 2.,
                6., 6., 6., 4., 2., 2.,
                3., 3., 3.,
                2., 2., 2., 2., 1., 1., 1.,
                2., 2., 2., 2., 1., 1., 1.]
        # Then hold default position and wait for button press
        while self.current_state == ControllerState.DEFAULT_POSE:
            for i in range(self.num_joints):
                motor_idx = self.dof_idx_full[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.default_joint_pos_mujoco[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
                self.low_cmd.motor_cmd[motor_idx].mode = 1  # 1:Enable, 0:Disable
            self.send_cmd(self.low_cmd)
            self.update_state_machine()
            time.sleep(self.control_dt)
        

    def policy_state(self):
        print("Enter policy state.")
        print(
            "Press L1 to go to damping, A to go to Walk, rest to be implemented"
        )
        while self.current_state == ControllerState.POLICY:
            self.run()
            # if self.counter == 20:
                # self.current_state = ControllerState.DAMPING
            self.policy_mode_listener()
            self.update_state_machine()

    def check_button_press(self, button_key):
        """Check if a button was just pressed (transition from 0 to 1)"""
        current_state = self.remote_controller.button[button_key]
        was_pressed = self.prev_button_states[button_key] == 0 and current_state == 1
        self.prev_button_states[button_key] = current_state
        return was_pressed

    def update_state_machine(self):
        """Update the state machine based on button presses"""
        # Select button always goes to damping
        if self.check_button_press(KeyMap.L1):
            if self.current_state != ControllerState.DAMPING:
                print("Transitioning to DAMPING state")
                self.current_state = ControllerState.DAMPING
            return

        # State-specific transitions
        if self.current_state == ControllerState.ZERO_TORQUE:
            if self.check_button_press(KeyMap.L1):
                print("Transitioning to DAMPING state")
                self.current_state = ControllerState.DAMPING

        elif self.current_state == ControllerState.DAMPING:
            if self.check_button_press(KeyMap.R1):
                print("Transitioning to DEFAULT_POSE state")
                self.current_state = ControllerState.DEFAULT_POSE

        elif self.current_state == ControllerState.DEFAULT_POSE:
            if self.check_button_press(KeyMap.R1):
                print("Transitioning to POLICY state")
                self.current_state = ControllerState.POLICY
            elif self.check_button_press(KeyMap.L1):
                print("Transitioning to DAMPING state")
                self.current_state = ControllerState.DAMPING

        elif self.current_state == ControllerState.POLICY:
            if self.check_button_press(KeyMap.L1):
                print("Transitioning to DAMPING state")
                self.current_state = ControllerState.DAMPING

    def policy_mode_listener(self):
        # Listen for remote button presses to switch policy mode

        # A is VariedHeight, X is Stair, B is Obstacle
        if self.check_button_press(KeyMap.A):
            self.current_mode = self.policy_modes.Walk
            print("Current mode: Walk, A for Walk, rest to be implemented")
        elif self.check_button_press(KeyMap.X):
            print("Current mode: not implemented yet, A for Walk, rest to be implemented")
        elif self.check_button_press(KeyMap.B):
            print("Current mode: not implemented yet, A for Walk, rest to be implemented")
        elif self.check_button_press(KeyMap.Y):
            if self.use_height_map:
                print("Current mode: not implemented yet, A for Walk, rest to be implemented")
            else:
                print("Height map not enabled in config")
                print("Current mode: not implemented yet, A for Walk, rest to be implemented")

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 4s
        total_time = 4
        num_step = int(total_time / self.control_dt)

        kps = self.p_gain_mujoco
        kds = self.d_gain_mujoco
        dof_size = self.num_joints

        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[self.dof_idx_full[i]].q

        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = self.dof_idx_full[j]
                # Check if this motor should be disabled (wrist/waist)
                if hasattr(self, 'zero_motor_indices') and j in self.zero_motor_indices:
                    # Disable motor (mode = 0)
                    self.low_cmd.motor_cmd[motor_idx].q = 0
                    self.low_cmd.motor_cmd[motor_idx].qd = 0
                    self.low_cmd.motor_cmd[motor_idx].kp = 0
                    self.low_cmd.motor_cmd[motor_idx].kd = 0
                    self.low_cmd.motor_cmd[motor_idx].tau = 0
                    self.low_cmd.motor_cmd[motor_idx].mode = 0  # 0:Disable
                else:
                    # Normal control for active motors
                    target_pos = self.default_joint_pos_mujoco[j]
                    self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                    self.low_cmd.motor_cmd[motor_idx].qd = 0
                    self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                    self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                    self.low_cmd.motor_cmd[motor_idx].tau = 0
                    self.low_cmd.motor_cmd[motor_idx].mode = 1  # 1:Enable

            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)

    def update_history_buffers(
        self,
        omega,
        gravity_orientation,
        cmd,
        qj_isaac,
        dqj_isaac,
        action,
        phase,
        obstacle_obs=None,
    ):
        """Update history buffers by shifting old data and adding new observations"""
        # Shift existing data (move older entries forward, newest at the end)
        if self.history_length > 1:
            self.omega_history[:-1] = self.omega_history[1:].clone()
            self.gravity_orientation_history[:-1] = self.gravity_orientation_history[1:].clone()
            self.cmd_history[:-1] = self.cmd_history[1:].clone()
            self.qj_isaac_history[:-1] = self.qj_isaac_history[1:].clone()
            self.dqj_isaac_history[:-1] = self.dqj_isaac_history[1:].clone()
            self.action_history[:-1] = self.action_history[1:].clone()
            self.phase_history[:-1] = self.phase_history[1:].clone()
            # if self.obstacle_observation:
            self.obstacle_history[:-1] = self.obstacle_history[1:].clone()

        # Add new observations at the end (newest)
        self.omega_history[-1] = omega
        self.gravity_orientation_history[-1] = gravity_orientation
        # Always store 4-dim cmd (pad if needed)
        # cmd_pad = torch.zeros(self.max_cmd_dim, dtype=cmd.dtype, device=cmd.device)
        # cmd_pad[: cmd.shape[0]] = cmd
        self.cmd_history[-1] = cmd
        self.qj_isaac_history[-1] = qj_isaac
        self.dqj_isaac_history[-1] = dqj_isaac
        self.action_history[-1] = action
        self.phase_history[-1] = phase
        # if (
        #     getattr(self, "obstacle_history", None) is not None
        #     and obstacle_obs is not None
        # ):
        self.obstacle_history[-1] = obstacle_obs

    def run(self):
        start_time = time.time()
        if self.init_odom_flag:
            self.init_odom_flag = False
            self.odom_pose[0] = self.odom_state.position[0]
            self.odom_pose[1] = self.odom_state.position[1]
            self.odom_pose[2] = self.odom_state.position[2]
        # Get the current joint position and velocity
        self.cmd[0] = self.remote_controller.ly
        self.cmd[1] = self.remote_controller.lx * -1
        self.cmd[2] = self.remote_controller.rx * -1
        if self.check_button_press(KeyMap.up):
            self.cmd[3] = min(0.75, self.cmd[3] + 0.01)
            print(f"Height increased to {self.cmd[3]:.2f}")
        elif self.check_button_press(KeyMap.down):
            self.cmd[3] = max(0.4, self.cmd[3] - 0.01)
            print(f"Height decreased to {self.cmd[3]:.2f}")

        # Publish raw velocity command to ROS2
        max_cmd = self.max_cmd.copy()  
        raw_cmd_scaled = self.cmd[:3] * max_cmd
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = float(raw_cmd_scaled[0])
        cmd_vel_msg.linear.y = float(raw_cmd_scaled[1])
        cmd_vel_msg.angular.z = float(raw_cmd_scaled[2])
        self.cmd_vel_publisher.publish(cmd_vel_msg)

        # Use filtered velocity if available, otherwise fall back to raw command
        if self.filtered_cmd_received:
            self.cmd[:3] = self.filtered_cmd.copy()

        #TODO implement history buffer
        # self.update_history_buffers(
        #     omega_tensor,
        #     gravity_orientation_tensor,
        #     cmd_for_history,
        #     qj_isaac_tensor,
        #     dqj_isaac_tensor,
        #     self.action_tensor,
        #     phase_tensor,
        #     self.obstacle_info,
        # )

        # Run the policy network
        out0 = self._run_onnx_inference()
        action = out0.astype(np.float32).ravel()
        self.action = action.copy()
        # Zero out wrist and waist actions
        if hasattr(self, 'zero_joint_indices'):
            action[self.zero_joint_indices] = 0.0
        joint_pos = self.default_joint_pos + self.action * self.action_scale
        joint_pos_mujoco = self.isaac_to_mujoco(joint_pos)
        # print("default joint pos:", self.default_joint_pos)
        # print("action:", self.action)
        # print("action_scale:", self.action_scale)
        # print("joint_pos_mujoco:", joint_pos_mujoco_test)
        # joint_pos_obs = [
        #     -0.50667626, 0.09009623, -0.05003734, 0.5851157, -0.3889048, 0.07695626,
        #     -0.43107846, -0.01222508, 0.11346863, 0.76132303, -0.39926296, -0.08412777,
        #     0.02849789, 0.05217453, 0.19677635, 0.46041033, 0.51166236, -0.15512648,
        #     0.55312645, 0.00994195, -0.00559423, -0.00548745, 0.41059622, -0.44231993,
        #     0.15539043, 0.61511827, -0.06598981, 0.04594062, 0.02837695
        # ]
        # self.action = self.mujoco_to_isaac(np.array(joint_pos_mujoco))
        # Build low cmd
        for step in range(1):
            for i in range(len(self.dof_idx_full)):
                motor_idx = self.dof_idx_full[i]
                # Check if this motor should be disabled (wrist/waist)
                if hasattr(self, 'zero_motor_indices') and i in self.zero_motor_indices:
                    # Disable motor (mode = 0)
                    self.low_cmd.motor_cmd[motor_idx].q = 0
                    self.low_cmd.motor_cmd[motor_idx].qd = 0
                    self.low_cmd.motor_cmd[motor_idx].kp = 0
                    self.low_cmd.motor_cmd[motor_idx].kd = 0
                    self.low_cmd.motor_cmd[motor_idx].tau = 0
                    self.low_cmd.motor_cmd[motor_idx].mode = 0  # 0:Disable
                else:
                    # Normal control for active motors
                    self.low_cmd.motor_cmd[motor_idx].q = joint_pos_mujoco[i]
                    self.low_cmd.motor_cmd[motor_idx].qd = 0
                    self.low_cmd.motor_cmd[motor_idx].kp = self.p_gain_mujoco[i]
                    self.low_cmd.motor_cmd[motor_idx].kd = self.d_gain_mujoco[i]
                    self.low_cmd.motor_cmd[motor_idx].tau = 0
                    self.low_cmd.motor_cmd[motor_idx].mode = 1  # 1:Enable

            # send the command
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)
        # joint_state_msg = JointState()
        # joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        # joint_state_msg.name = self.joint_names_mujoco[1:]
        # joint_state_msg.position = self.qj.tolist()
        # joint_state_msg.velocity = self.dqj.tolist()
        # self.joint_states_pub.publish(joint_state_msg)

        end_time = time.time()
        # time.sleep(self.control_dt*10)

    def _parse_joint_names_from_xml(self, xml_path):
        joint_names = []
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for joint in root.iter("joint"):
                name = joint.attrib.get("name", None)
                if name is not None:
                    joint_names.append(name)
            joint_names.pop(0)
        except Exception as e:
            self.get_logger().warning(f"Failed to parse joint names from XML: {e}")
        return joint_names

    def _create_joint_order_mappings(self, xml_joint_names, onnx_joint_names):
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

    def isaac_to_mujoco(self, arr):
        # Logic matches rl_policy_node.py
        out = np.zeros_like(arr)
        for xml_idx, onnx_idx in enumerate(self.mujoco_to_isaac_joint_indices):
            if onnx_idx >= 0 and onnx_idx < len(arr):
                out[xml_idx] = arr[onnx_idx]
        return out

    def mujoco_to_isaac(self, arr):
        # Logic matches rl_policy_node.py
        out = np.zeros_like(arr)
        for onnx_idx, xml_idx in enumerate(self.isaac_to_mujoco_joint_indices):
            if xml_idx >= 0 and xml_idx < len(arr):
                out[onnx_idx] = arr[xml_idx]
        return out

    def obs_command_vel(self):
        return self.cmd[:4]

    def obs_projected_gravity(self):
        quat = self.low_state.imu_state.quaternion
        qw = quat[0]
        qx = quat[1]
        qy = quat[2]
        qz = quat[3]

        gravity_orientation = np.zeros(3)

        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

        return gravity_orientation

    def obs_base_ang_vel(self):
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)
        return ang_vel[0,:]

    def obs_joint_pos(self):
        for i in range(len(self.dof_idx_full)):
            self.qj[i] = self.low_state.motor_state[self.dof_idx_full[i]].q
        joint_pos = self.mujoco_to_isaac(self.qj) - self.default_joint_pos
        # Zero out wrist and waist joint positions
        if hasattr(self, 'zero_joint_indices'):
            joint_pos[self.zero_joint_indices] = 0.0
        return joint_pos

    def obs_joint_vel(self):
        for i in range(len(self.dof_idx_full)):
            self.dqj[i] = self.low_state.motor_state[self.dof_idx_full[i]].dq
        joint_vel = self.mujoco_to_isaac(self.dqj)
        # Zero out wrist and waist joint velocities
        if hasattr(self, 'zero_joint_indices'):
            joint_vel[self.zero_joint_indices] = 0.0
        return joint_vel

    def obs_actions(self):
        return self.action

    def construct_observations(self):
        obs_list = []
        for name in self.observation_names:
            func_name = f"obs_{name}"
            obs_func = getattr(self, func_name, None)
            if obs_func is None:
                raise RuntimeError(f"Observation function '{func_name}' not found in Controller.")
            obs_val = obs_func()
            if obs_val is None:
                raise RuntimeError(f"Observation function '{func_name}' returned None.")
            obs_list.append(np.asarray(obs_val, dtype=np.float32))
        return np.concatenate(obs_list, axis=-1)

    def _run_onnx_inference(self):
        input_feed = {}
        self.counter += 1
        for inp in self.onnx_inputs:
            name = inp["name"]
            shape = inp["shape"]
            if name == "obs" or len(self.onnx_inputs) == 1:
                obs_np = self.construct_observations().astype(np.float32)
                expected = None
                if shape is not None:
                    if len(shape) == 1:
                        expected = shape[0]
                    elif len(shape) >= 2:
                        expected = shape[-1]
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
                arr = np.array([[self.counter]], dtype=np.float32)
                input_feed[name] = arr
        # print("inputs are:", input_feed)
        outputs = self.onnx_sess.run(None, input_feed)
        return outputs[0]

    def cmd_vel_filtered_callback(self, msg: Twist):
        """Callback for filtered velocity commands"""
        self.filtered_cmd[0] = msg.linear.x  # forward/backward
        self.filtered_cmd[1] = msg.linear.y  # left/right
        self.filtered_cmd[2] = msg.angular.z  # rotation
        self.filtered_cmd_received = True
        print(
            f"Received filtered cmd: [{self.filtered_cmd[0]:.3f}, {self.filtered_cmd[1]:.3f},"
            f" {self.filtered_cmd[2]:.3f}]"
        )

    def height_map_callback(self, msg: GridCells):
        """Callback for height map information"""
        if self.use_height_map:
            # Find the average cell.z
            heightmap_mean_z = np.mean([cell.z for cell in msg.cells])
            for i, cell in enumerate(msg.cells):
                x = round(cell.x / self.resolution) + self.grid_points_x // 2
                y = round(cell.y / self.resolution) + self.grid_points_y // 2
                if 0 <= x < self.grid_points_x and 0 <= y < self.grid_points_y:
                    self.height_map_obs[y, x] = -(cell.z - heightmap_mean_z)

            # print("Received height map with shape:", self.height_map_obs.shape)
        else:
            print("Height map callback called but height map is not enabled in config.")

    def _spin_ros(self):
        """Spin ROS2 in a separate thread to ensure callbacks are processed"""
        while self.ros_running:
            try:
                rclpy.spin_once(self, timeout_sec=0.1)
            except Exception as e:
                print(f"ROS2 spinning error: {e}")
                time.sleep(0.01)

    def main_loop(self):
        """Main control loop that runs the state machine"""
        print("Starting main control loop...")
        print("Waiting for filtered velocity commands...")

        # Wait a moment for ROS2 connections to establish
        time.sleep(1.0)

        try:
            while True:
                if self.current_state == ControllerState.ZERO_TORQUE:
                    self.zero_torque_state()
                elif self.current_state == ControllerState.DAMPING:
                    self.damping_state()
                elif self.current_state == ControllerState.DEFAULT_POSE:
                    self.default_pos_state()
                elif self.current_state == ControllerState.POLICY:
                    self.policy_state()
                else:
                    print(f"Unknown state: {self.current_state}")
                    break
        except KeyboardInterrupt:
            print("\nShutting down controller...")
            self.ros_running = False
            # Send zero torque command before exit
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
        except Exception as e:
            print(f"Error in main loop: {e}")
            self.ros_running = False
            # Send zero torque command before exit
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            raise


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument(
        "config",
        type=str,
        help="config file name in the configs folder",
        default="g1.yaml",
    )
    args = parser.parse_args()

    # Initialize ROS2
    rclpy.init()

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config_file=args.config)

    try:
        # Run the main control loop with state machine
        controller.main_loop()
    finally:
        # Stop ROS2 spinning thread
        controller.ros_running = False
        if controller.ros_thread.is_alive():
            controller.ros_thread.join(timeout=1.0)
        # Cleanup ROS2
        controller.destroy_node()
        rclpy.shutdown()
        controller.ros_running = False
        if controller.ros_thread.is_alive():
            controller.ros_thread.join(timeout=1.0)
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
