# Copyright (c) 2025 Lizhi Yang AMBER LAB

import argparse
import os

# Update the viewer at 30Hz
import time

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float32MultiArray, Float64



class MujocoSimNode(Node):
    def __init__(self, config_file="g1_21dof.yaml"):

        super().__init__("mujoco_sim_node")
        self.config_file = config_file
        self.publisher = self.create_publisher(Float32MultiArray, "sensor_data", 10)
        self.robot_position_pub = self.create_publisher(
            Float32MultiArray, "robot_position", 10
        )
        self.time_pub = self.create_publisher(Float64, "sim_time", 10)
        self.foot_forces_pub = self.create_publisher(
            Float32MultiArray, "foot_forces", 10
        )
        self.ankle_heights_pub = self.create_publisher(
            Float32MultiArray, "ankle_heights", 10
        )
        self.height_map_pub = self.create_publisher(Float32MultiArray, "height_map", 10)
        # Publisher to signal RL policy to enable control (Float64: 0.0/1.0)
        self.control_enable_pub = self.create_publisher(Float64, "control_enable", 10)
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

    # self.sim_time = 0.0

    def load_config(self):
        G1_RL_ROOT_DIR = os.getenv("G1_RL_ROOT_DIR")
        with open(f"{G1_RL_ROOT_DIR}/deploy/configs/{self.config_file}") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.simulation_dt = config["simulation_dt"]
        # self.num_actions = config["num_actions"]
        # self.num_obs = config["num_obs"]
        # self.joint_ids = config.get("joint_ids", list(range(self.num_actions)))
        # self.default_angles = np.array(config["default_angles"], dtype=np.float32)[
        #     self.joint_ids
        # ]
        # self.dof_pos_scale = config["dof_pos_scale"]
        # self.dof_vel_scale = config["dof_vel_scale"]
        # self.ang_vel_scale = config["ang_vel_scale"]
        # self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
        # self.action_scale = config["action_scale"]
        # self.policy_path = config["policy_path"].replace(
        #     "{G1_RL_ROOT_DIR}", G1_RL_ROOT_DIR
        # )
        self.xml_path = config["xml_path"].replace("{G1_RL_ROOT_DIR}", G1_RL_ROOT_DIR)
        self.use_height_map = config["use_height_map"]

        self.grid_size_x = float(config["grid_size_x"])
        self.grid_size_y = float(config["grid_size_y"])
        self.resolution = float(config["resolution"])
        self.forward_offset = float(config["forward_offset"])
        self.grid_points_x = int(config["grid_points_x"])
        self.grid_points_y = int(config["grid_points_y"])
        self.N_grid_points = int(config["N_grid_points"])
        self.height_offset = float(config.get("height_offset", 0.5))
        # self.isaac_mujoco_conversion = config.get("isaac_mujoco_conversion", False)

    def init_sim(self):
        self.m = mujoco.MjModel.from_xml_path(self.xml_path)
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = self.simulation_dt
        self.target_dof_pos = np.zeros(self.num_joints)
        self.target_dof_p = np.zeros(self.num_joints)
        self.target_dof_d = np.zeros(self.num_joints)

        # --- Initialize robot joints to provided values ---
        initial_joint_values = np.array([
            -0.413, 0.0, 0.0, 0.807, -0.374, 0.0, -0.413, 0.0, 0.0, 0.807,
            -0.374, 0.0, 0.0, 0.0, 0.0, 0.498, 0.3, 0.0, 0.501, 0.0,
            0.0, 0.0, 0.498, -0.3, 0.0, 0.501, 0.0, 0.0, 0.0
        ], dtype=np.float64)
        self.d.qpos[7:7 + self.num_joints] = initial_joint_values
        # Optionally, also set target positions to match initial
        self.target_dof_pos = initial_joint_values.copy()

        # Find foot body IDs
        self.left_foot_id = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"
        )
        self.right_foot_id = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link"
        )

        # Launch the viewer in passive mode
        self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
        # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
        self.viewer_sync_interval = 1.0 / 30.0  # 30Hz
        self._last_viewer_sync_time = 0.0

        # Initialize height map visualization if enabled
        if self.use_height_map:
            # Initialize visualization spheres for height map points
            for i in range(self.N_grid_points):
                mujoco.mjv_initGeom(
                    self.viewer.user_scn.geoms[i],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([0.02, 0, 0]),
                    pos=np.array([0, 0, 0]),
                    mat=np.eye(3).flatten(),
                    rgba=np.array([1, 0, 0, 1]),
                )
                self.viewer.user_scn.ngeom += 1

    def action_callback(self, msg):
        # Expect msg.data to be [position, p, d] repeated for each joint
        arr = np.array(msg.data, dtype=np.float32)
        n = arr.shape[0] // 3
        self.target_dof_pos = arr[:n]
        self.target_dof_p = arr[n : 2 * n]
        self.target_dof_d = arr[2 * n : 3 * n]

    def get_foot_forces(self):
        """Extract contact forces for both feet"""
        left_force = np.zeros(3)
        right_force = np.zeros(3)

        # Iterate through all contacts
        for i in range(self.d.ncon):
            contact = self.d.contact[i]
            geom1_body = self.m.geom_bodyid[contact.geom1]
            geom2_body = self.m.geom_bodyid[contact.geom2]

            # Check if contact involves foot bodies
            if geom1_body == self.left_foot_id or geom2_body == self.left_foot_id:
                # Get contact force
                c_array = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.m, self.d, i, c_array)
                force = c_array[:3]  # First 3 elements are force
                left_force += np.abs(force)  # Sum absolute forces

            if geom1_body == self.right_foot_id or geom2_body == self.right_foot_id:
                # Get contact force
                c_array = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.m, self.d, i, c_array)
                force = c_array[:3]  # First 3 elements are force
                right_force += np.abs(force)  # Sum absolute forces

        return left_force, right_force

    def get_ankle_heights(self):
        """Extract world Z positions of ankle links"""
        left_height = self.d.xpos[self.left_foot_id][2]  # Z coordinate
        right_height = self.d.xpos[self.right_foot_id][2]  # Z coordinate
        return left_height, right_height

    def get_height_map(self):
        """Generate a height map around the robot's base"""

        hit_points = np.zeros((self.N_grid_points, 3), dtype=np.float32)
        self.hit_points_viz = np.zeros((self.N_grid_points, 3), dtype=np.float32)

        base_pos = self.d.qpos[:3].copy()
        base_quat = self.d.qpos[3:7].copy()

        # Height map needs to be in the yaw frame
        # Get the yaw of the quat
        r = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
        base_yaw = r.as_euler("zyx", degrees=False)[0]

        # 2D rotation matrix for the yaw
        cos_yaw = np.cos(base_yaw)
        sin_yaw = np.sin(base_yaw)
        R_2d = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])

        # Starting position for the grid (offset forward by 0.2m and centered)
        offset = np.array(
            [-self.forward_offset + self.grid_size_x / 2, self.grid_size_y / 2]
        )
        grid_start = base_pos[:2] - R_2d @ offset

        # Cast rays from each point in the grid
        point_index = 0
        for i in range(self.grid_points_y):  # Note that we are going in row-major order
            for j in range(self.grid_points_x):
                # Calculate the x, y coordinates for this grid point
                # The IsaacLab grid config defaults to "xy" which means the inner loop is x
                offset = np.array([(j * self.resolution), (i * self.resolution)])
                xy = grid_start + R_2d @ offset

                # Ray parameters
                ray_origin = np.array([xy[0], xy[1], base_pos[2]])
                ray_direction = np.array([0.0, 0.0, -1.0])
                geomid = np.zeros(1, dtype=np.int32)
                geom_group = np.zeros(6, dtype=np.int32)
                geom_group[2] = 1  # Only include group 2

                # Cast the ray
                distance = mujoco.mj_ray(
                    self.m,
                    self.d,
                    ray_origin.astype(np.float64),
                    ray_direction.astype(np.float64),
                    geom_group,
                    1,  # flg_static
                    -1,  # bodyid
                    geomid,
                )

                if distance != -1.0:
                    hit_points[point_index, :] = np.array(
                        [xy[0], xy[1], -distance + base_pos[2]]
                    )  # Location in world frame
                    self.hit_points_viz[point_index, :] = np.array(
                        [xy[0], xy[1], -distance + base_pos[2]]
                    )  # Store for visualization

                    # Print info for the center point (used for ground height)
                    # if i == grid_points_y // 2 and j == grid_points_x // 2:
                    #     ground_height = -distance + base_pos[2]
                    #     print("Ground height: ", ground_height)
                    #     if geomid[0] != -1:
                    #         geom_name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, geomid[0])
                    #         print("Hit object name: ", geom_name)
                else:
                    # If no hit, use a default position at ground level
                    self.hit_points_viz[point_index, :] = np.array([xy[0], xy[1], 0.0])
                # else:
                #     # If no hit, set to center
                #     hit_points[point_index, :] = hit_points[18, :]

                point_index += 1

        ground_heights = hit_points[:, 2]
        ground_heights = ground_heights.reshape((self.N_grid_points,))

        return ground_heights

    def sim_step(self):
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.viewer.cam.trackbodyid = 1  # ID of the robot's main body
        start_time = time.time()  # Start timing at the very beginning of the step
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
        #     self.target_dof_pos = self.action * self.action_scale + self.default_angles
        self.d.ctrl[: self.num_joints] = self.target_dof_pos
        self.d.ctrl[self.num_joints :] = 0.0  # Ensure the rest of the controls are zero

        # Update PD gains for each joint from self.target_dof_p and self.target_dof_d
        # for i in range(self.num_joints):
        #     self.m.jnt_stiffness[i] = self.target_dof_p[i]
        #     self.m.dof_damping[i] = self.target_dof_d[i]

        # fix robot
        # self.d.qpos[0] = 0
        # self.d.qpos[1] = 0
        # self.d.qpos[2] = 0.78  # Set pelvis height
        # self.d.qpos[3] = 1.0  # Set pelvis orientation
        # self.d.qpos[4] = 0.0  # Set pelvis orientation
        # self.d.qpos[5] = 0.0  # Set pelvis orientation
        # self.d.qpos[6] = 0.0  # Set pelvis orientation
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

        # Publish foot forces
        left_force, right_force = self.get_foot_forces()
        foot_forces_msg = Float32MultiArray()
        foot_forces_msg.data = np.concatenate([left_force, right_force]).tolist()
        self.foot_forces_pub.publish(foot_forces_msg)

        # Publish ankle heights
        left_height, right_height = self.get_ankle_heights()
        ankle_heights_msg = Float32MultiArray()
        ankle_heights_msg.data = [left_height, right_height]
        self.ankle_heights_pub.publish(ankle_heights_msg)

        if self.use_height_map:
            ground_heights = self.get_height_map()
            base_pos = self.d.qpos[:3].copy()
            self.height_obs = base_pos[2] - ground_heights - self.height_offset
            self.height_obs = np.ones_like(self.height_obs) * 0.23
            # self.height_obs = torch.ones_like(self.height_obs) * -1.21
            # Normalize height map points: subtract mean, divide by stddev
            # mean = np.mean(self.height_obs)
            # std = np.std(self.height_obs) if np.std(self.height_obs) > 1e-5 else 1.0
            # print(std)
            # std = np.max(np.std(self.height_obs), 1e-7)
            # self.height_obs = (self.height_obs - mean) / std
            # self.height_obs = np.zeros_like(self.height_obs)
            # print("Height map points (normalized): ", self.height_obs)

            msg = Float32MultiArray()
            msg.data = self.height_obs.tolist()
            self.height_map_pub.publish(msg)

            # Update visualization spheres
            for i in range(self.N_grid_points):
                if hasattr(self, "hit_points_viz"):
                    self.viewer.user_scn.geoms[i].pos = self.hit_points_viz[i, :]
        # Track joint error statistics (RL policy node performs plotting)

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
        help="Config file name (e.g., g1_21dof.yaml)",
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
