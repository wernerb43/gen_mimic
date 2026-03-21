import argparse
import os

# Update the viewer at 30Hz
import time

import json
import ast
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
import onnx
import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64



class MujocoSimNode(Node):
    def __init__(self, config_file):

        super().__init__("mujoco_sim_node")
        self.config_file = config_file
        self.publisher = self.create_publisher(Float32MultiArray, "sensor_data", 10)
        self.robot_position_pub = self.create_publisher(
            Float32MultiArray, "robot_position", 10
        )
        self.ball_orientation_pub = self.create_publisher(
            Float32MultiArray, "ball_orientation", 10
        )
        self.time_pub = self.create_publisher(Float64, "sim_time", 10)
        
        self.height_map_pub = self.create_publisher(Float32MultiArray, "height_map", 10)
        self.base_lin_vel_pub = self.create_publisher(
            Float32MultiArray, "base_lin_vel", 10
        )
        # Publisher to signal RL policy to enable control (Float64: 0.0/1.0)
        self.control_enable_pub = self.create_publisher(Float64, "control_enable", 10)

        self.torso_pose_pub = self.create_publisher(
            Float32MultiArray, "torso_pose", 10
        )
        self.ball_position_pub = self.create_publisher(
            Float32MultiArray, "ball_position", 10
        )

        self.action_sub = self.create_subscription(
            Float32MultiArray, "action", self.action_callback, 10
        )
        self.action = None
        self.timer = self.create_timer(0.0, self.sim_step)  # As fast as possible
        self.num_joints = 29
        self.load_config()
        self.init_sim()
        # control enable published once after hold period
        self._control_enable_sent = False
        # duration to hold robot upright before enabling policy (seconds)
        self._hold_upright_duration = 4.0
        # one-time prints
        self._hold_printed = False
        self._released_printed = False
        
        ##################################################### Ball spawn configuration
        self.ball_spawn_time = 0.0  # Spawn ball at t=0.0s
        # Nominal ball offset from torso_link (matches target_pos_means in multi_catching_env_cfg.py)
        self.ball_offset = np.array([0.4807, -0.0159, 0.0493])  # x, y, z offset from torso_link
        self.ball_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion (w, x, y, z)

        # Randomization ranges for target position (offset from torso_link, matches training)
        self.target_x_range = (0.4807, 0.4807)   # forward offset from torso_link
        self.target_y_range = (-0.1, 0.1)         # lateral offset from torso_link
        self.target_z_range = (-0.1, 0.1)         # vertical offset from torso_link
        #####################################################

        self.ball_spawned = False

        # Subscribe to ros commands for randomizing and resetting the ball target
        # (published by the RL policy node when user presses 't' or 'r')
        self.throw_ball_sub = self.create_subscription(
            Float32MultiArray, "throw_ball_command", self.throw_ball_callback, 10
        )
        self.reset_ball_sub = self.create_subscription(
            Float32MultiArray, "reset_ball_command", self.reset_ball_callback, 10
        )

    # self.sim_time = 0.0

    def load_config(self):
        G1_RL_ROOT_DIR = os.getenv("G1_RL_ROOT_DIR")
        with open(f"{G1_RL_ROOT_DIR}/deploy/configs/{self.config_file}") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.simulation_dt = config["simulation_dt"]

        self.xml_path = config["xml_path"].replace("{G1_RL_ROOT_DIR}", G1_RL_ROOT_DIR)
        self.robot_xml_path = config.get("robot_xml_path", self.xml_path).replace(
            "{G1_RL_ROOT_DIR}", G1_RL_ROOT_DIR
        )
        policy_path = config["policy_path"].replace("{G1_RL_ROOT_DIR}", G1_RL_ROOT_DIR)

        # Load default_joint_pos from ONNX metadata and convert to MuJoCo joint order
        self.default_joint_pos = self._load_default_joint_pos_from_onnx(policy_path, self.robot_xml_path)


    def _load_default_joint_pos_from_onnx(self, policy_path, robot_xml_path):
        """Load default_joint_pos from ONNX metadata and convert from Isaac to MuJoCo joint order."""
        onnx_model = onnx.load(policy_path)
        meta = {}
        for prop in onnx_model.metadata_props:
            meta[prop.key] = prop.value
        if "onnx_metadata" in meta:
            try:
                parsed = json.loads(meta["onnx_metadata"])
            except Exception:
                parsed = ast.literal_eval(meta["onnx_metadata"])
        else:
            parsed = {}
            for k, v in meta.items():
                try:
                    parsed[k] = json.loads(v)
                except Exception:
                    try:
                        parsed[k] = ast.literal_eval(v)
                    except Exception:
                        parsed[k] = v

        onnx_joint_names = parsed.get("joint_names", [])
        if isinstance(onnx_joint_names, str):
            onnx_joint_names = [s.strip() for s in onnx_joint_names.split(",") if s.strip()]
        default_joint_pos_isaac = np.array(parsed["default_joint_pos"], dtype=np.float64)

        # Parse MuJoCo XML joint order
        xml_joint_names = []
        tree = ET.parse(robot_xml_path)
        for joint in tree.getroot().iter("joint"):
            name = joint.attrib.get("name")
            if name:
                xml_joint_names.append(name)
        xml_joint_names.pop(0)  # remove floating base joint

        # Convert Isaac order -> MuJoCo order
        default_joint_pos_mujoco = np.zeros(len(xml_joint_names), dtype=np.float64)
        for xml_idx, xml_name in enumerate(xml_joint_names):
            if xml_name in onnx_joint_names:
                onnx_idx = onnx_joint_names.index(xml_name)
                default_joint_pos_mujoco[xml_idx] = default_joint_pos_isaac[onnx_idx]
        self.get_logger().info(f"Loaded default_joint_pos from ONNX metadata ({len(default_joint_pos_mujoco)} joints)")
        return default_joint_pos_mujoco

    def init_sim(self):
        self.m = mujoco.MjModel.from_xml_path(self.xml_path)
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = self.simulation_dt
        self.target_dof_pos = np.zeros(self.num_joints)
        self.target_dof_p = np.zeros(self.num_joints)
        self.target_dof_d = np.zeros(self.num_joints)

        # Initialize robot joints to default pose from ONNX metadata (in MuJoCo joint order)
        self.d.qpos[7:7 + self.num_joints] = self.default_joint_pos
        self.target_dof_pos = self.default_joint_pos.copy()
        
        # Find torso_link body ID (anchor body used in Isaac training)
        try:
            self.torso_body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
            self.get_logger().info(f"torso_link found: body_id={self.torso_body_id}")
        except Exception as e:
            self.get_logger().warning(f"torso_link body not found in model: {e}, falling back to root body")
            self.torso_body_id = 1  # fallback to root

        # Find ball body and joint indices
        try:
            self.ball_body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "ball")
            ball_joint_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
            self.ball_joint_qpos_adr = self.m.jnt_qposadr[ball_joint_id]
            self.ball_joint_qvel_adr = self.m.jnt_dofadr[ball_joint_id]
            self.ball_gravcomp_default = float(self.m.body_gravcomp[self.ball_body_id])
            self.get_logger().info(f"Ball found: body_id={self.ball_body_id}, qpos_adr={self.ball_joint_qpos_adr}")
        except Exception as e:
            self.get_logger().warning(f"Ball body/joint not found in model: {e}")
            self.ball_body_id = None
            self.ball_gravcomp_default = 0.0

        # Launch the viewer in passive mode
        self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
        self.viewer_sync_interval = 1.0 / 30.0  # 30Hz
        self._last_viewer_sync_time = 0.0



    def throw_ball_callback(self, msg):
        """Randomize ball position within target range (offset from torso_link) and hold it in place."""
        if self.ball_body_id is None:
            return
        # Randomize offset from torso_link (matches target_pos_means convention in training)
        torso_pos = self.d.xpos[self.torso_body_id]
        offset = np.array([
            np.random.uniform(*self.target_x_range),
            np.random.uniform(*self.target_y_range),
            np.random.uniform(*self.target_z_range),
        ])
        new_pos = torso_pos + offset
        # Place ball at randomized position with gravity compensation (static target)
        self.m.body_gravcomp[self.ball_body_id] = 1.0
        self.d.qpos[self.ball_joint_qpos_adr:self.ball_joint_qpos_adr + 3] = new_pos
        self.d.qpos[self.ball_joint_qpos_adr + 3:self.ball_joint_qpos_adr + 7] = [1, 0, 0, 0]
        self.d.qvel[self.ball_joint_qvel_adr:self.ball_joint_qvel_adr + 6] = 0.0
        self.ball_spawned = True
        # print(f"Target randomized at position: [{x:.3f}, {y:.3f}, {z:.3f}]")

    def reset_ball_callback(self, msg):
        if self.ball_body_id is None:
            return
        self.m.body_gravcomp[self.ball_body_id] = self.ball_gravcomp_default
        torso_pos = self.d.xpos[self.torso_body_id]
        ball_pos = torso_pos + self.ball_offset
        self.d.qpos[self.ball_joint_qpos_adr:self.ball_joint_qpos_adr + 3] = ball_pos
        self.d.qpos[self.ball_joint_qpos_adr + 3:self.ball_joint_qpos_adr + 7] = [1, 0, 0, 0]
        self.d.qvel[self.ball_joint_qvel_adr:self.ball_joint_qvel_adr + 6] = 0.0
        self.ball_spawned = True
        print(f"Reset ball to torso_link + offset: {ball_pos}")


    def action_callback(self, msg):
        # Expect msg.data to be [position, p, d] 
        arr = np.array(msg.data, dtype=np.float32)
        n = arr.shape[0] // 3
        self.target_dof_pos = arr[:n]
        self.target_dof_p = arr[n : 2 * n]
        self.target_dof_d = arr[2 * n : 3 * n]

    def sim_step(self):
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.viewer.cam.trackbodyid = 1  # ID of the robot's main body
        start_time = time.time()  # Start timing at the very beginning of the step
        
        # Spawn ball at prescribed time and position (offset from torso_link)
        if not self.ball_spawned and self.d.time >= self.ball_spawn_time and self.ball_body_id is not None:
            torso_pos = self.d.xpos[self.torso_body_id]
            ball_pos = torso_pos + self.ball_offset
            # Set ball position (qpos for free joint: x, y, z, qw, qx, qy, qz)
            self.d.qpos[self.ball_joint_qpos_adr:self.ball_joint_qpos_adr+3] = ball_pos
            self.d.qpos[self.ball_joint_qpos_adr+3:self.ball_joint_qpos_adr+7] = [1, 0, 0, 0]  # identity quaternion
            # Zero initial velocity
            self.d.qvel[self.ball_joint_qvel_adr:self.ball_joint_qvel_adr+6] = 0.0
            self.ball_spawned = True
            self.get_logger().info(f"Ball spawned at torso_link + {self.ball_offset} = {ball_pos} at t={self.d.time:.3f}s")
        
        # Hold robot upright for initial period: fix base pose and zero base velocities
        if self.d.time < self._hold_upright_duration:
            # print when hold starts (only once)
            if not self._hold_printed:
                print(f"[MujocoSimNode] Holding robot upright for {self._hold_upright_duration}s", flush=True)
                self._hold_printed = True
            # Force pelvis/base position and orientation to initial values
            # keep base at current z (pelvis height) and neutral orientation
            self.d.qpos[0] = 0.0
            self.d.qpos[1] = 0.0
            # preserve current pelvis height
            # if not set, leave as-is
            # set neutral orientation quaternion (w,x,y,z) = (1,0,0,0)
            try:
                self.d.qpos[3] = 1.0
                self.d.qpos[4] = 0.0
                self.d.qpos[5] = 0.0
                self.d.qpos[6] = 0.0
            except Exception:
                pass
            # zero base linear and angular velocities
            try:
                self.d.qvel[0] = 0.0
                self.d.qvel[1] = 0.0
                self.d.qvel[2] = 0.0
                self.d.qvel[3] = 0.0
                self.d.qvel[4] = 0.0
                self.d.qvel[5] = 0.0
            except Exception:
                pass
        # Apply action if available
        # if self.action is not None:
            # self.target_dof_pos = self.action * self.action_scale + self.default_angles
        self.d.ctrl[: self.num_joints] = self.target_dof_pos

        mujoco.mj_step(self.m, self.d)

        # Prepare sensor data
        qj = self.d.sensordata[: self.num_joints]  # [self.joint_ids]
        dqj = self.d.sensordata[
            self.num_joints : 2 * self.num_joints
        ]  # [self.joint_ids]
        tqj = self.d.sensordata[2 * self.num_joints : 3 * self.num_joints]
        quat = self.d.sensordata[3 * self.num_joints : 3 * self.num_joints + 4]
        omega = self.d.sensordata[3 * self.num_joints + 4 : 3 * self.num_joints + 7]
        # robot_pos = self.d.sensordata[3 * self.num_joints + 7 : 3 * self.num_joints + 10]
        # robot_vel = self.d.sensordata[3 * self.num_joints + 10 : 3 * self.num_joints + 13]
        # qj = (qj - self.default_angles) * self.dof_pos_scale
        # dqj = dqj * self.dof_vel_scale
        # omega = omega * self.ang_vel_scale
        # Compose sensor data
        sensor_data = np.concatenate([qj, dqj, quat, omega])
        msg = Float32MultiArray()
        msg.data = sensor_data.tolist()
        self.publisher.publish(msg)
        robot_position = np.array(
            self.d.xpos[1]
        )  # Get the robot's base position in world coordinates
        robot_position_msg = Float32MultiArray()
        robot_position_msg.data = robot_position.tolist()
        self.robot_position_pub.publish(robot_position_msg)

        # Publish torso_link pose (position + quaternion) for anchor-frame observations
        torso_pos = self.d.xpos[self.torso_body_id]
        torso_quat = self.d.xquat[self.torso_body_id]
        torso_pose_msg = Float32MultiArray()
        torso_pose_msg.data = np.concatenate([torso_pos, torso_quat]).tolist()
        self.torso_pose_pub.publish(torso_pose_msg)
        tmsg = Float64()
        tmsg.data = self.d.time  # Use actual mujoco sim time
        self.time_pub.publish(tmsg)

        # Publish a single control_enable flag when hold period ends
        if (not self._control_enable_sent) and (self.d.time >= self._hold_upright_duration):
            flag = Float64()
            flag.data = 1.0
            self.control_enable_pub.publish(flag)
            self._control_enable_sent = True
            print(f"[MujocoSimNode] Published control_enable=1.0 at sim_time={self.d.time:.3f}", flush=True)

        # Publish base linear velocity (from first 3 elements of qvel)
        base_lin_vel = self.d.qvel[0:3]
        base_lin_vel_msg = Float32MultiArray()
        base_lin_vel_msg.data = base_lin_vel.tolist()
        self.base_lin_vel_pub.publish(base_lin_vel_msg)

        # Publish ball position to RL policy
        if self.ball_body_id is not None:
            ball_pos = self.d.xpos[self.ball_body_id]
            ball_pos_msg = Float32MultiArray()
            ball_pos_msg.data = ball_pos.tolist()
            self.ball_position_pub.publish(ball_pos_msg)
            
            ball_quat = self.d.xquat[self.ball_body_id]
            self.ball_orientation = ball_quat
            ball_quat_msg = Float32MultiArray()
            ball_quat_msg.data = ball_quat.tolist()
            self.ball_orientation_pub.publish(ball_quat_msg)

        # --- Synchronize simulation time with real time ---
        sim_dt = self.simulation_dt
        elapsed = time.time() - start_time
        if elapsed < sim_dt:
            time.sleep(sim_dt - elapsed)
        if (
            self.viewer.is_running()
            and (time.time() - self._last_viewer_sync_time) >= self.viewer_sync_interval
        ):
            self.viewer.sync()
            self._last_viewer_sync_time = time.time()
        # --- end synchronization ---


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser(
        description="Launch MujocoSimNode and RLPolicyNode with a given config file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config file name (e.g., g1_29dof.yaml)",
    )
    args = parser.parse_args()
    node = MujocoSimNode(config_file=args.config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Attempt to close the mujoco viewer gracefully
        if hasattr(node, "viewer") and node.viewer is not None:
            try:
                node.viewer.close()
            except Exception:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
