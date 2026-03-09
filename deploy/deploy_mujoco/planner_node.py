import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64, Bool

GRAVITY = 9.81  # m/s^2
NUM_JOINTS = 29


class Planner(Node):
    """
    Subscribes to ball_position from the MuJoCo sim node, estimates ball velocity
    via forward differencing, solves for the ballistic trajectory intersection with
    the plane x_body = 0.3 m in front of the robot, and publishes the intersection
    point (in robot body frame) and time-to-intercept.
    """

    def __init__(self):
        super().__init__("planner_node")

        # Subscribers
        self.ball_position_sub = self.create_subscription(
            Float32MultiArray, "ball_position", self.ball_position_callback, 10
        )
        self.robot_position_sub = self.create_subscription(
            Float32MultiArray, "robot_position", self.robot_position_callback, 10
        )

        self.motion_in_progress_sub = self.create_subscription(
            Bool, "motion_in_progress", self.motion_in_progress_callback, 10
        )

        self.sensor_sub = self.create_subscription(
            Float32MultiArray, "sensor_data", self.sensor_callback, 10
        )
        self.time_sub = self.create_subscription(
            Float64, "sim_time", self.time_callback, 10
        )

        # Publisher: [x_body, y_body, z_body] - intercept estimate or default position
        self.estimate_pub = self.create_publisher(
            Float32MultiArray, "ball_intercept_estimate", 10
        )

        self.play_motion_pub = self.create_publisher(
            Float32MultiArray, "play_motion_command", 10
        )

        # State
        self.ball_pos = None
        self.prev_ball_pos = None
        self.prev_time = None
        self.sim_time = 0.0

        self.robot_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # x y z
        self.robot_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # w,x,y,z
        self.default_ball_position = np.array([0.0, 0.0, 1.1], dtype=np.float32)  # fallback if no data yet

        # Intercept plane: x = 0.3 in robot body frame
        self.intercept_x = 0.3

        # the time it takes between the start of a motion and the 'catch' of the motion (ideally this is fixed for)
        self.motion_time = 1.0

        # Estimation at 50 Hz (matches control_dt = 0.02)
        self.timer = self.create_timer(0.02, self.estimate_step)

        self.play_motion = False
        self.motion_in_progress = False

        self.last_intercept_location = np.copy(self.default_ball_position)

        

    # ------------------------------------------------------------------ callbacks

    def ball_position_callback(self, msg):
        received = np.array(msg.data, dtype=np.float32)
        if len(received) == 3:
            self.ball_pos = received.copy()

    def sensor_callback(self, msg):
        received = np.array(msg.data, dtype=np.float32)
        # sensor_data layout: [qj(29), dqj(29), quat(4), omega(3)]
        if len(received) >= 2 * NUM_JOINTS + 4:
            self.robot_quat = received[2 * NUM_JOINTS : 2 * NUM_JOINTS + 4].copy()

    def robot_position_callback(self, msg):
        received = np.array(msg.data, dtype=np.float32)
        if len(received) == 3:
            self.robot_position = received.copy()

    def motion_in_progress_callback(self, msg):
        self.motion_in_progress = msg.data


    def time_callback(self, msg):
        self.sim_time = msg.data

    # ------------------------------------------------------------------ estimation

    def estimate_step(self):
        if self.ball_pos is None:
            # Publish default position if no ball data yet
            msg = Float32MultiArray()
            default_body = self.default_ball_position
            msg.data = [float(default_body[0]), float(default_body[1]), float(default_body[2])]
            self.estimate_pub.publish(msg)
            play_msg = Float32MultiArray()
            play_msg.data = [0.0]
            self.play_motion_pub.publish(play_msg)
            return

        # --- forward-difference velocity estimation ---
        if self.prev_ball_pos is not None and self.prev_time is not None:
            dt = self.sim_time - self.prev_time
            if dt < 1e-6:
                return
            ball_vel = (self.ball_pos - self.prev_ball_pos) / dt
        else:
            # First tick – store and wait for next
            self.prev_ball_pos = self.ball_pos.copy()
            self.prev_time = self.sim_time
            # Publish default position
            msg = Float32MultiArray()
            default_body = self.default_ball_position
            msg.data = [float(default_body[0]), float(default_body[1]), float(default_body[2])]
            self.estimate_pub.publish(msg)
            play_msg = Float32MultiArray()
            play_msg.data = [0.0]
            self.play_motion_pub.publish(play_msg)
            return

        # Update previous state for next tick
        self.prev_ball_pos = self.ball_pos.copy()
        self.prev_time = self.sim_time

        # --- solve for plane intersection ---
        # Transform ball position and velocity from world frame to robot body frame
        ball_pos_body = self._world_to_body(self.ball_pos)
        ball_vel_body = self._world_to_body_vec(ball_vel)
        
        # Gravity vector in robot body frame
        gravity_w = np.array([0.0, 0.0, -GRAVITY], dtype=np.float32)
        gravity_body = self._world_to_body_vec(gravity_w)

        # Forward direction in body frame is just the x-axis
        forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Ballistic trajectory: p(t) = p0 + v0*t + 0.5*g*t^2
        # Plane condition:  forward . p(t) = intercept_x
        # => A*t^2 + B*t + C = 0
        A = 0.5 * np.dot(forward, gravity_body)
        B = np.dot(forward, ball_vel_body)
        C = np.dot(forward, ball_pos_body) - self.intercept_x
        # print(f"Planner: A={A:.4f}, B={B:.4f}, C={C:.4f}") #TODO why is this quadratic?
        # print("forward:", forward)

        t_intercept = self._solve_quadratic_smallest_positive(A, B, C)

        # Decide what to publish based on time to intercept
        msg = Float32MultiArray()
        play_msg = Float32MultiArray()
        
        if t_intercept is None or t_intercept < 0:
            # No valid intercept or ball has passed - publish default position (in world frame)
            default_world = self.default_ball_position
            msg.data = [float(default_world[0]), float(default_world[1]), float(default_world[2])]
            play_msg.data = [0.0]
        elif t_intercept < self.motion_time:  
            print(f"Valid intercept in {t_intercept:.2f} seconds - publishing intercept location and play command")
            # Ball is approaching and will intercept soon - compute intercept in body frame, then transform to world
            intercept_body = (
                ball_pos_body
                + ball_vel_body * t_intercept
                + 0.5 * gravity_body * t_intercept ** 2
            )
            # Transform intercept from body frame to world frame
            intercept_world = self._body_to_world(intercept_body)
            msg.data = [float(intercept_world[0]), float(intercept_world[1]), float(intercept_world[2])]
            play_msg.data = [1.0]
            self.last_intercept_location = intercept_world
        elif self.motion_in_progress:
            # motion is in progress, but the ball has passed, so we should publish the last intercept location to try to complete the motion
            intercept_world = self.last_intercept_location
            msg.data = [float(intercept_world[0]), float(intercept_world[1]), float(intercept_world[2])]
            play_msg.data = [1.0]
        else:
            # Ball is too far away - publish default position (in world frame)
            default_world = self.default_ball_position
            msg.data = [float(default_world[0]), float(default_world[1]), float(default_world[2])]
            play_msg.data = [0.0]
        
        self.estimate_pub.publish(msg)
        self.play_motion_pub.publish(play_msg)

    # ------------------------------------------------------------------ helpers
    def _world_to_body(self, point_world):
        """Transform a point from world frame to robot body frame."""
        point_relative = point_world - self.robot_position
        q_inv = self._quat_conjugate(self.robot_quat)
        return self._quat_rotate_vec(q_inv, point_relative)
    
    def _world_to_body_vec(self, vec_world):
        """Transform a vector (not a point) from world frame to robot body frame."""
        q_inv = self._quat_conjugate(self.robot_quat)
        return self._quat_rotate_vec(q_inv, vec_world)
    
    def _body_to_world(self, point_body):
        """Transform a point from robot body frame to world frame."""
        point_world_relative = self._quat_rotate_vec(self.robot_quat, point_body)
        return point_world_relative + self.robot_position
    
    @staticmethod
    def _quat_conjugate(q):
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)

    @staticmethod
    def _quat_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float32,
        )

    @classmethod
    def _quat_rotate_vec(cls, q, v):
        v_quat = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)
        q_inv = cls._quat_conjugate(q)
        result = cls._quat_multiply(q, cls._quat_multiply(v_quat, q_inv))
        return result[1:4]

    @staticmethod
    def _solve_quadratic_smallest_positive(A, B, C):
        """Solve A*t^2 + B*t + C = 0 for the smallest positive t."""
        if abs(A) < 1e-10:
            # Degenerate to linear
            if abs(B) < 1e-10:
                return None
            t = -C / B
            return t if t > 1e-6 else None

        discriminant = B ** 2 - 4 * A * C
        if discriminant < 0:
            return None

        sqrt_d = np.sqrt(discriminant)
        t1 = (-B - sqrt_d) / (2 * A)
        t2 = (-B + sqrt_d) / (2 * A)
        # print(f"quadratic solutions: t1={t1:.4f}, t2={t2:.4f}")

        candidates = [t for t in (t1, t2) if t > 1e-6]
        return min(candidates) if candidates else None


def main(args=None):
    rclpy.init(args=args)
    node = Planner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
