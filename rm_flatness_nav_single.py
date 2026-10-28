#!/usr/bin/env python3
"""Merged reference; uses an already-running robomaster_ros AP driver.
Do not run alongside the launch navigation node. Generated from the five modules.
"""

# ==================== flatness.py ====================
"""Differential-flatness trajectory generation for a unicycle-type robot.

The flat output is the planar position z(t) = [x(t), y(t)]^T. Every smooth curve
for z(t) gives the linear speed v and turn rate w in closed form (Eq. 1), so no
numerical optimiser is needed. Each component is a cubic over horizon T (Eq. 2).
"""
import math
import numpy as np

EPS = 1e-6
V_MAX = 0.16         # m/s cruise limit (Eq. 3)
STANDOFF = 0.20      # m, final distance from the tag
T_MIN = 4.0          # s
T_GAIN = 1.6


def flat_inversion(xd, yd, xdd, ydd):
    """Eq. (1): recover (v, w) from the first and second derivatives of z(t)."""
    speed_sq = xd * xd + yd * yd
    v = math.sqrt(speed_sq)
    if speed_sq < EPS:
        return v, 0.0
    w = (xd * ydd - yd * xdd) / speed_sq
    return v, w


def solve_flat_polynomial(z0, zdot0, zf, zdotf, T):
    """Eq. (2): cubic coefficients matching the four boundary values."""
    A = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0],
                  [1.0, T, T ** 2, T ** 3],
                  [0.0, 1.0, 2.0 * T, 3.0 * T ** 2]])
    b = np.array([z0, zdot0, zf, zdotf], dtype=float)
    return np.linalg.solve(A, b)


def eval_cubic(alpha, t):
    """Position, velocity and acceleration of the cubic at time t."""
    a0, a1, a2, a3 = alpha
    return (a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3,
            a1 + 2.0 * a2 * t + 3.0 * a3 * t ** 2,
            2.0 * a2 + 6.0 * a3 * t)


def time_horizon(d0, standoff=STANDOFF, v_max=V_MAX, t_min=T_MIN, gain=T_GAIN):
    """Eq. (3): T = max(4.0, 1.6 (d0 - 0.20) / v_max)."""
    return max(t_min, gain * (d0 - standoff) / v_max)


class FlatTrajectory:
    """Straight-ahead cubic pursuit in the robot frame at centering.

    x(0)=0, x'(0)=0.06 m/s, x(T)=d0-0.20, x'(T)=0;
    y(0)=0, y'(0)=0, y(T)=0, y'(T)=0.
    """

    def __init__(self, d0, v0=0.06, standoff=STANDOFF, v_max=V_MAX):
        self.d0 = d0
        self.T = time_horizon(d0, standoff, v_max)
        goal = d0 - standoff
        self.alpha_x = solve_flat_polynomial(0.0, v0, goal, 0.0, self.T)
        self.alpha_y = solve_flat_polynomial(0.0, 0.0, 0.0, 0.0, self.T)

    def sample(self, t):
        """(v, w, x, y) at time t since trajectory start, clamped to [0,T]."""
        t = min(max(t, 0.0), self.T)
        x, xd, xdd = eval_cubic(self.alpha_x, t)
        y, yd, ydd = eval_cubic(self.alpha_y, t)
        v, w = flat_inversion(xd, yd, xdd, ydd)
        if xd < 0.0:
            v = 0.0
        return v, w, x, y

    def finished(self, t):
        return t >= self.T

# ==================== wheel_voltage.py ====================
"""Wheel-rate kinematics (Eq. 4) and armature voltage model (Eq. 5)."""

WHEEL_RADIUS = 0.05
CHASSIS_RADIUS = 0.20
K_V = 0.38
K_A = 0.06


def wheel_rates(vx, wz, r=WHEEL_RADIUS, lc=CHASSIS_RADIUS):
    """Eq. (4): (front-right, front-left, rear-left, rear-right)."""
    w1 = (vx + lc * wz) / r
    w2 = (vx - lc * wz) / r
    return (w1, w2, w2, w1)


class MotorVoltageModel:
    """Eq. (5): V_i = Kv*w_i + Ka*(w_i(t)-w_i(t-dt))/dt."""

    def __init__(self, kv=K_V, ka=K_A):
        self.kv = kv
        self.ka = ka
        self._prev = None

    def reset(self):
        self._prev = None

    def step(self, omegas, dt):
        if self._prev is None or dt <= 0.0:
            accel = (0.0,) * len(omegas)
        else:
            accel = tuple((w - p) / dt for w, p in zip(omegas, self._prev))
        self._prev = tuple(omegas)
        return tuple(self.kv * w + self.ka * a for w, a in zip(omegas, accel))

# ==================== state_machine.py ====================
"""Supervisory FSM: SEARCHING -> ALIGNING -> CHASING -> ARRIVED.

Bearing > 0 means the tag is LEFT of the camera axis (ROS REP-103 convention).
Friend's FSM; tag loss always returns to SEARCHING after lost_timeout.
"""
from dataclasses import dataclass
from typing import Optional

SEARCHING, ALIGNING, CHASING, ARRIVED = 'SEARCHING', 'ALIGNING', 'CHASING', 'ARRIVED'


@dataclass
class Params:
    lost_timeout: float = 0.45
    search_rate: float = 0.30
    align_tol: float = 0.18
    realign_tol: float = 0.30
    settle_time: float = 0.30
    align_gain: float = 0.8
    align_min_rate: float = 0.12
    align_max_rate: float = 0.40
    chase_bearing_gain: float = 0.8
    chase_max_w: float = 0.40
    brake_distance: float = 0.20
    v_max: float = 0.16


@dataclass
class Observation:
    now: float
    visible: bool = False
    bearing: float = 0.0
    distance: float = float('inf')
    tof: Optional[float] = None
    observed_at: Optional[float] = None


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


class NavStateMachine:
    def __init__(self, params: Params = None):
        self.p = params or Params()
        self.state = SEARCHING
        self.last_seen = None
        self.last_dir = 1.0
        self.centered_since = None
        self.traj = None
        self.traj_t0 = None
        self.arrival_reason = ''
        self.last_distance = float('inf')

    def _brake_hit(self, obs):
        if obs.visible and obs.distance <= self.p.brake_distance:
            return 'camera'
        if obs.tof is not None and obs.tof <= self.p.brake_distance:
            return 'tof'
        return ''

    def _enter_search(self):
        self.state = SEARCHING
        self.traj = None
        self.centered_since = None

    def update(self, obs: Observation):
        p = self.p
        if self.state == ARRIVED:
            return 0.0, 0.0

        if obs.visible:
            self.last_seen = obs.now if obs.observed_at is None else obs.observed_at
            self.last_distance = obs.distance
            if abs(obs.bearing) > 1e-3:
                self.last_dir = 1.0 if obs.bearing > 0 else -1.0

        reason = self._brake_hit(obs)
        if reason:
            self.state, self.arrival_reason, self.traj = ARRIVED, reason, None
            return 0.0, 0.0

        lost = self.last_seen is None or (obs.now - self.last_seen) > p.lost_timeout
        if lost:
            if self.state != SEARCHING:
                self._enter_search()
            return 0.0, p.search_rate * self.last_dir

        if self.state == SEARCHING:
            self.state = ALIGNING
            self.centered_since = None

        if self.state == ALIGNING:
            if not obs.visible:
                return 0.0, 0.0
            if abs(obs.bearing) <= p.align_tol:
                if self.centered_since is None:
                    self.centered_since = obs.now
                if obs.now - self.centered_since >= p.settle_time:
                    self.traj = FlatTrajectory(obs.distance, v_max=p.v_max)
                    self.traj_t0 = obs.now
                    self.state = CHASING
                    return self._chase(obs)
                return 0.0, 0.0
            self.centered_since = None
            w = _clip(p.align_gain * obs.bearing, -p.align_max_rate, p.align_max_rate)
            if abs(w) < p.align_min_rate:
                w = p.align_min_rate * (1.0 if w >= 0 else -1.0)
            return 0.0, w

        if obs.visible and abs(obs.bearing) > p.realign_tol:
            self.state, self.traj, self.centered_since = ALIGNING, None, None
            return 0.0, 0.0
        return self._chase(obs)

    def _chase(self, obs):
        t = obs.now - self.traj_t0
        if self.traj.finished(t):
            # Elapsed trajectory time is not evidence of the 20 cm standoff.
            # Replan with the same cubic if the observed target is still farther.
            if obs.visible:
                self.traj = FlatTrajectory(obs.distance, v_max=self.p.v_max)
                self.traj_t0 = obs.now
                t = 0.0
            else:
                self._enter_search()
                return 0.0, self.p.search_rate * self.last_dir
        v, w_ff, _, _ = self.traj.sample(t)
        w = w_ff
        if obs.visible:
            w += self.p.chase_bearing_gain * obs.bearing
        return min(v, self.p.v_max), _clip(w, -self.p.chase_max_w, self.p.chase_max_w)

# ==================== tag_detector.py ====================
"""AprilTag detection (OpenCV aruco AprilTag dictionaries) and floor-plane geometry."""
import math
from typing import Optional
import cv2
import numpy as np

FAMILIES = {
    'tag16h5': 'DICT_APRILTAG_16h5', 'tag25h9': 'DICT_APRILTAG_25h9',
    'tag36h10': 'DICT_APRILTAG_36h10', 'tag36h11': 'DICT_APRILTAG_36h11',
}

R_OPTICAL_TO_BASE = np.array([[0.0, 0.0, 1.0],
                              [-1.0, 0.0, 0.0],
                              [0.0, -1.0, 0.0]])


def camera_rotation(pitch_down):
    """Rotation optical -> base for camera pitched down by pitch_down radians."""
    c, s = math.cos(pitch_down), math.sin(pitch_down)
    ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return ry @ R_OPTICAL_TO_BASE


def ground_range_bearing(p_cam, r_base_cam):
    """Horizontal ground distance and bearing; positive bearing = left."""
    p = r_base_cam @ np.asarray(p_cam, dtype=float)
    return math.hypot(p[0], p[1]), math.atan2(p[1], p[0])


class TagDetection:
    def __init__(self, tag_id, corners, tvec, rvec):
        self.tag_id, self.corners, self.tvec, self.rvec = tag_id, corners, tvec, rvec


class TagDetector:
    def __init__(self, family='tag36h11', tag_id=3, tag_size=0.15):
        name = FAMILIES[family]
        self.tag_id, self.size = tag_id, tag_size
        self._dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
        self._params = (cv2.aruco.DetectorParameters_create()
                        if hasattr(cv2.aruco, 'DetectorParameters_create')
                        else cv2.aruco.DetectorParameters())
        self._params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self._new_api = hasattr(cv2.aruco, 'ArucoDetector')
        if self._new_api:
            self._detector = cv2.aruco.ArucoDetector(self._dict, self._params)
        s = tag_size / 2.0
        self._obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32)

    def detect(self, gray, camera_matrix, dist_coeffs=None, scales=(1.0, 2.0)) -> Optional[TagDetection]:
        """First successful pose of configured tag ID across the image scales."""
        for scale in scales:
            img = gray if scale == 1.0 else cv2.resize(gray, None, fx=scale, fy=scale,
                                                        interpolation=cv2.INTER_CUBIC)
            det = self._detect_once(img, camera_matrix, dist_coeffs, scale)
            if det is not None:
                return det
        return None

    def _detect_once(self, gray, camera_matrix, dist_coeffs, scale):
        if self._new_api:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self._dict, parameters=self._params)
        if ids is None:
            return None
        for c, i in zip(corners, ids.flatten()):
            if self.tag_id is not None and int(i) != self.tag_id:
                continue
            img = (c.reshape(4, 2) / scale).astype(np.float32)
            dist = np.zeros(5) if dist_coeffs is None else dist_coeffs
            ok, rvec, tvec = cv2.solvePnP(self._obj, img, camera_matrix, dist,
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                return TagDetection(int(i), img, tvec.reshape(3), rvec.reshape(3))
        return None

# ==================== nav_node.py ====================
"""RoboMaster EP AprilTag search, alignment, flat-output pursuit, standoff brake,
and 1 Hz motor-voltage logging. Uses robomaster_ros (not direct RoboMaster SDK).

Subscribes: camera/image_color, camera/camera_info, range_0, TF
Publishes: cmd_vel; action: move_arm; feedback: arm_position, odom
"""
import csv
import math
import os
import time

import cv2
import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from action_msgs.msg import GoalStatus
from robomaster_msgs.action import MoveArm

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, Range


CSV_HEADER = ['t_s', 'V1_V', 'V2_V', 'V3_V', 'V4_V', 'state',
              'v_mps', 'w_radps', 'distance_m', 'bearing_rad', 'tof_m',
              'w1_radps', 'w2_radps', 'w3_radps', 'w4_radps']


def quat_to_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class FlatnessNav(Node):
    def __init__(self):
        super().__init__('flatness_nav')
        d = self.declare_parameter
        self.tag_family = d('tag_family', 'tag36h11').value
        self.tag_id = d('tag_id', 3).value
        self.tag_size = d('tag_size', 0.15).value
        self.image_topic = d('image_topic', 'camera/image_color').value
        self.info_topic = d('info_topic', 'camera/camera_info').value
        self.tof_topic = d('tof_topic', 'range_0').value
        self.use_tof = d('use_tof', True).value
        self.base_frame = d('base_frame', 'base_link').value
        self.camera_frame = d('camera_frame', 'camera_optical_link').value
        self.control_rate = d('control_rate', 20.0).value
        self.csv_path = d('csv_path', os.path.expanduser('~/ros2_ws/log/assessment2/motor_voltage_log.csv')).value
        self.fast_path = d('fast_csv_path', os.path.expanduser('~/ros2_ws/log/assessment2/motor_voltage_fast.csv')).value
        self.set_arm = d('set_arm_pose', True).value
        self.arm_x, self.arm_z = 0.105, 0.080  # User-approved ToF clearance: SDK X/Y mm -> ROS X/Z metres
        self.enable_motion = d('enable_motion', True).value
        self.detect_fresh = d('detection_fresh_s', 0.20).value
        fsm_params = Params()
        for name in Params.__dataclass_fields__:
            setattr(fsm_params, name, d(f'fsm.{name}', getattr(fsm_params, name)).value)
        self.fsm = NavStateMachine(fsm_params)
        self.voltage = MotorVoltageModel(d('kv', 0.38).value, d('ka', 0.06).value)
        self.r = d('wheel_radius', 0.05).value
        self.lc = d('chassis_radius', 0.20).value

        self.detector = TagDetector(self.tag_family, self.tag_id, self.tag_size)
        self.bridge = CvBridge()
        self.K, self.dist = None, None
        self.det_time, self.bearing, self.distance = -1e9, 0.0, float('inf')
        self.tof = None
        self.r_bc, self.tf_ok = None, False
        self.image_time = self.info_time = self.tof_time = self.odom_time = -1e9
        self.arm_time = -1e9
        self.arm_position = None
        self.arm_sent = self.arm_succeeded = self.arm_verified = False
        self.arm_failed = False
        self.odom_stopped = False
        self.zero_sent = False
        self.started = False
        self.camera_translation = np.zeros(3)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.arm_client = ActionClient(self, MoveArm, 'move_arm')
        self.create_subscription(PointStamped, 'arm_position', self.on_arm, qos_profile_sensor_data)
        self.create_subscription(Odometry, 'odom', self.on_odom, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, self.info_topic, self.on_info, qos_profile_sensor_data)
        self.create_subscription(Image, self.image_topic, self.on_image, qos_profile_sensor_data)
        if self.use_tof:
            self.create_subscription(Range, self.tof_topic, self.on_range, qos_profile_sensor_data)

        self.t_start = self.now()
        self.prev_tick = self.t_start
        self.next_log = 0.0
        self.volts, self.omegas = (0.0,) * 4, (0.0,) * 4
        self.v_cmd = self.w_cmd = 0.0
        self.stop_ticks = 0
        self.finished = False
        for path in (self.csv_path, self.fast_path):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._csv_file = open(self.csv_path, 'w', newline='')
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(CSV_HEADER)
        self._fast_file = open(self.fast_path, 'w', newline='')
        self._fast = csv.writer(self._fast_file)
        self._fast.writerow(['t_s', 'v_mps', 'w_radps', 'V1_V', 'V2_V', 'V3_V', 'V4_V', 'state'])
        self.create_timer(1.0 / self.control_rate, self.tick)
        self.get_logger().info(
            f'Tag {self.tag_family} id={self.tag_id} size={self.tag_size} m. '
            f'Table header: t(s) V1 V2 V3 V4 state. CSV -> {os.path.abspath(self.csv_path)}')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_arm(self, msg):
        if msg.header.frame_id != 'arm_base_link':
            return
        self.arm_position = msg.point
        self.arm_time = self.now()
        # Driver feedback is quantised to millimetres. Action success alone
        # does not guarantee arrival (see MoveArm.action).
        was_verified = self.arm_verified
        self.arm_verified = (self.arm_succeeded and
                             abs(msg.point.x - self.arm_x) <= 0.001 and
                             abs(msg.point.z - self.arm_z) <= 0.001)
        if self.arm_verified and not was_verified:
            self.get_logger().info('Arm feedback verified: X=105 mm, Y=80 mm (ROS X/Z)')
        elif self.arm_succeeded and not self.arm_verified:
            self.get_logger().error(
                f'Arm target not reached: feedback x={msg.point.x:.3f}, z={msg.point.z:.3f} m; holding zero',
                throttle_duration_sec=5.0)

    def on_odom(self, msg):
        self.odom_time = self.now()
        t = msg.twist.twist
        self.odom_stopped = (abs(t.linear.x) <= 0.005 and abs(t.linear.y) <= 0.005
                             and abs(t.angular.z) <= 0.03)

    def send_arm(self):
        if self.arm_sent or not self.arm_client.server_is_ready():
            return
        self.arm_sent = True
        # MoveArm uses float32 but this installed driver truncates metres*1000
        # with int(). The nearest float32 for 0.105/0.080 lies just below the
        # exact millimetre. Encode the next float32 above it so the driver's
        # integer targets are exactly 105 and 80 mm, not 104 and 79 mm.
        x_wire = float(np.nextafter(np.float32(self.arm_x), np.float32(np.inf)))
        z_wire = float(np.nextafter(np.float32(self.arm_z), np.float32(np.inf)))
        goal = MoveArm.Goal(x=x_wire, z=z_wire, relative=False)
        self.arm_client.send_goal_async(goal).add_done_callback(self.on_arm_goal)

    def on_arm_goal(self, future):
        try:
            handle = future.result()
            if not handle.accepted:
                raise RuntimeError('move_arm goal rejected')
            handle.get_result_async().add_done_callback(self.on_arm_result)
        except Exception as exc:
            self.arm_failed = True
            self.get_logger().error(str(exc))

    def on_arm_result(self, future):
        try:
            self.arm_succeeded = future.result().status == GoalStatus.STATUS_SUCCEEDED
            self.arm_failed = not self.arm_succeeded
            self.get_logger().info('Arm action complete; waiting for position feedback at X=105, Y=80 mm')
        except Exception as exc:
            self.arm_failed = True
            self.get_logger().error(str(exc))

    def interfaces_ready(self, now):
        # Reuse the assignment's loss timeout for required live interfaces.
        fresh = self.fsm.p.lost_timeout
        return (self.K is not None and self.tf_ok and
                now - self.image_time <= fresh and now - self.info_time <= fresh and
                now - self.odom_time <= fresh and
                (not self.use_tof or now - self.tof_time <= fresh) and
                self.cmd_pub.get_subscription_count() > 0)

    def on_info(self, msg):
        k = np.array(msg.k, dtype=float).reshape(3, 3)
        if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
            self.K = None
            return
        self.K = k
        self.info_time = self.now()
        self.camera_frame = msg.header.frame_id
        self.dist = np.array(msg.d, dtype=float) if len(msg.d) else None

    def on_range(self, msg):
        if math.isnan(msg.range) or msg.range < 0:
            self.tof = None
            self.tof_time = -1e9
            return
        self.tof_time = self.now()
        self.tof = None if msg.range >= msg.max_range - 1e-3 else float(msg.range)

    def update_camera_rotation(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, self.camera_frame, rclpy.time.Time())
            tf_age = self.now() - rclpy.time.Time.from_msg(tf.header.stamp).nanoseconds * 1e-9
            if abs(tf_age) > self.fsm.p.lost_timeout:
                self.tf_ok = False
                return
            self.r_bc, self.tf_ok = quat_to_matrix(tf.transform.rotation), True
            p = tf.transform.translation
            self.camera_translation = np.array([p.x, p.y, p.z])
        except tf2_ros.TransformException:
            self.tf_ok = False

    def process_frame(self, frame):
        if self.K is None or not self.tf_ok:
            return
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        det = self.detector.detect(gray, self.K, self.dist)
        if det is None:
            return
        p_base = self.r_bc @ det.tvec + self.camera_translation
        self.distance = math.hypot(p_base[0], p_base[1])
        self.bearing = math.atan2(p_base[1], p_base[0])
        self.det_time = self.image_time

    def on_image(self, msg):
        try:
            age = self.now() - rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
            if age > self.fsm.p.lost_timeout or age < -self.fsm.p.lost_timeout:
                return
            if msg.header.frame_id != self.camera_frame:
                return
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.image_time = rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
            self.update_camera_rotation()
            self.process_frame(frame)
        except Exception as exc:
            self.get_logger().warn(f'frame skipped: {exc}', throttle_duration_sec=2.0)

    def tick(self):
        if self.finished:
            self.cmd_pub.publish(Twist())
            return
        now = self.now()
        dt = max(now - self.prev_tick, 1e-3)
        self.prev_tick = now
        visible = (now - self.det_time) <= self.detect_fresh
        obs = Observation(now=now, visible=visible, bearing=self.bearing if visible else 0.0,
                          distance=self.distance if visible else float('inf'), tof=self.tof,
                          observed_at=self.det_time)
        ready = self.interfaces_ready(now)
        if ready and self.zero_sent and self.odom_stopped and self.enable_motion and self.set_arm:
            self.send_arm()
        arm_ready = (not self.set_arm or (self.arm_verified and now - self.arm_time <= self.fsm.p.lost_timeout))
        if not (ready and arm_ready and self.zero_sent and self.enable_motion):
            v, w = 0.0, 0.0
            if self.started and self.fsm.state != ARRIVED:
                self.fsm._enter_search()
            self.get_logger().info('Holding zero: waiting for live camera/calibration/TF/ToF/odom, '
                                   'cmd_vel subscriber and verified arm pose (or enable_motion=false)',
                                   throttle_duration_sec=5.0)
        else:
            if not self.started:
                self.started = True
                self.get_logger().info('Required interfaces ready; starting assignment FSM')
            v, w = self.fsm.update(obs)
        self.v_cmd, self.w_cmd = v, w
        tw = Twist()
        tw.linear.x, tw.angular.z = float(v), float(w)
        self.cmd_pub.publish(tw)
        if v == 0.0 and w == 0.0:
            self.zero_sent = True
        self.omegas = wheel_rates(v, w, self.r, self.lc)
        self.volts = self.voltage.step(self.omegas, dt)

        t_rel = now - self.t_start
        self._fast.writerow([f'{t_rel:.3f}', f'{v:.4f}', f'{w:.4f}',
                             *[f'{x:.4f}' for x in self.volts], self.fsm.state])
        self._fast_file.flush()
        if self.fsm.state == ARRIVED:
            self.stop_ticks += 1
            if self.stop_ticks == 1:
                self.log_row(t_rel, f'BRAKING ({self.fsm.arrival_reason})', visible)
            if self.stop_ticks == 3:
                self.log_row(t_rel, f'ARRIVED ({self.fsm.arrival_reason})', visible)
                self.get_logger().info(f'Stop requested at {t_rel:.1f} s ({self.fsm.arrival_reason}); '
                                       f'zero commands sent; log saved to {self.csv_path}')
                self.finished = True
                self._csv_file.close()
                self._fast_file.close()
            return
        if t_rel >= self.next_log:
            self.log_row(math.floor(t_rel + 1e-6), self.fsm.state, visible)
            self.next_log += 1.0

    def log_row(self, t, state, visible):
        row = [f'{t:.1f}' if isinstance(t, float) else t,
               *[f'{x:.3f}' for x in self.volts], state,
               f'{self.v_cmd:.3f}', f'{self.w_cmd:.3f}',
               f'{self.distance:.3f}' if visible else '',
               f'{self.bearing:.3f}' if visible else '',
               f'{self.tof:.3f}' if self.tof is not None else '',
               *[f'{x:.3f}' for x in self.omegas]]
        self._csv.writerow(row)
        self._csv_file.flush()
        self.get_logger().info(f'{row[0]:>5} s | V1..V4 = '
                               + ' '.join(f'{x:7.3f}' for x in self.volts) + f' V | {state}')

    def shutdown(self):
        try:
            if rclpy.ok():
                for _ in range(3):
                    self.cmd_pub.publish(Twist())
        except Exception:
            if rclpy.ok():
                raise
        finally:
            for f in (self._csv_file, self._fast_file):
                if not f.closed:
                    f.close()


def main(args=None):
    rclpy.init(args=args)
    node = FlatnessNav()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
        t_end = time.time() + 1.0
        while rclpy.ok() and time.time() < t_end:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
