"""RoboMaster EP Core AP-mode ROS navigation for AprilTag ID 3.

OpenCV draws every visible tag's ID and box.
Only ID 3 drives the FSM.
Above 15% width, approach straight; width >=17% arrives; ToF <=20 cm brakes.
"""

import csv
import math
import os
import time
from datetime import datetime

import cv2
import numpy as np
import rclpy
import tf2_ros

from action_msgs.msg import GoalStatus
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from robomaster_msgs.action import MoveArm
from sensor_msgs.msg import CameraInfo, Image, Range

from .state_machine import (
    SEARCHING,
    CHASING,
    ARRIVED,
    HOLD,
    Params,
    Observation,
    NavStateMachine
)

from .search_cycle import SearchCycle
from .tracking_log import TrackingLog
from .tag_detector import TagDetector
from .wheel_voltage import MotorVoltageModel, wheel_rates


CSV_COLUMNS = [
    't_s', 'V1_V', 'V2_V', 'V3_V', 'V4_V',
    'state', 'v_mps', 'w_radps',
    'camera_distance_m', 'bearing_rad', 'tof_m',
    'bbox_width_fraction', 'bbox_height_fraction',
    'w1_radps', 'w2_radps', 'w3_radps', 'w4_radps'
]


def quaternion_rotation(q):
    x, y, z, w = q.x, q.y, q.z, q.w

    return np.array([
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w)
        ],
        [
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w)
        ],
        [
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y)
        ]
    ], dtype=float)


class FlatnessNav(Node):

    def __init__(self):
        super().__init__('flatness_nav')

        param = self.declare_parameter

        self.target_id = int(param('tag_id', 3).value)
        self.tag_family = str(
            param('tag_family', 'tag36h11').value
        )
        self.tag_size = float(
            param('tag_size', 0.15).value
        )

        self.use_motion = bool(
            param('enable_motion', True).value
        )
        self.set_arm_pose = True

        self.arm_x = 0.105
        self.arm_z = 0.080
        self.arm_pose_tolerance = 0.003

        self.rate = float(
            param('control_rate', 20.0).value
        )

        self.base_frame = str(
            param('base_frame', 'base_link').value
        )
        self.camera_frame = str(
            param('camera_frame', 'camera_optical_link').value
        )

        self.camera_topic = str(
            param('image_topic', 'camera/image_color').value
        )
        self.info_topic = str(
            param('camera_info_topic', 'camera/camera_info').value
        )
        self.tof_topic = str(
            param('tof_topic', 'range_0').value
        )

        # Stream health is independent of target-observation freshness.
        self.camera_timeout = float(param('camera_timeout_s', 1.0).value)

        self.process_interval = float(
            param('min_image_interval_s', 0.08).value
        )

        # Bench-test switch: treat the arm pose check as passed so the
        # search logic can be tested while the arm is not reaching
        # X=105 / Y=80. Camera TF and distance are NOT valid in that case.
        self.skip_arm_check = bool(
            param('skip_arm_check', False).value
        )
        self.arm_max_attempts = int(
            param('arm_max_attempts', 3).value
        )

        slow_path = os.path.expanduser(
            '~/ros2_ws/log/assessment2/motor_voltage_log.csv'
        )
        fast_path = os.path.expanduser(
            '~/ros2_ws/log/assessment2/motor_voltage_fast.csv'
        )

        self.csv_path = os.path.expanduser(
            str(param('csv_path', slow_path).value)
        )
        self.fast_path = os.path.expanduser(
            str(param('fast_csv_path', fast_path).value)
        )

        fsm_config = Params()

        for name in Params.__dataclass_fields__:
            setattr(
                fsm_config,
                name,
                param(
                    f'fsm.{name}',
                    getattr(fsm_config, name)
                ).value
            )

        self.fsm = NavStateMachine(fsm_config)
        self.detector = TagDetector(
            self.tag_family,
            self.target_id,
            self.tag_size
        )

        self.search_cycle = SearchCycle()
        self.odom_pose = None
        self.actual_velocity = (0.0, 0.0)
        default_output = os.path.expanduser('~/ros2_ws/log/assessment2/run_' +
                                         datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        self.tracking = TrackingLog(str(param('output_dir', default_output).value),
                                    bool(param('generate_plots', True).value))
        self.get_logger().info(f'Tracking CSV and plots: {self.tracking.directory}')

        self.voltage = MotorVoltageModel()
        self.bridge = CvBridge()

        self.K = None
        self.distortion = None

        self.camera_time = -1e9
        self.last_processed = -1e9
        self.detection_time = -1e9

        self.target_distance = float('inf')
        self.target_bearing = 0.0
        self.center_x_error = float("inf")
        self.center_y_error = float("inf")
        self.bbox_w = 0.0
        self.bbox_h = 0.0

        self.tof_distance = None
        self.tof_time = -1e9
        self.odom_time = -1e9
        self.arm_time = -1e9
        self.arm_feedback = None
        self.arm_feedback_frame = None

        self.arm_succeeded = False
        self.arm_verified = False
        self.arm_sent = False
        self.arm_failed = False
        self.arm_attempts = 0
        self.arm_retry_at = 0.0
        self.odom_stopped = False

        self.tf_ok = False
        self.r_base_camera = None
        self.t_base_camera = np.zeros(3)

        self.was_ready = False
        self.zero_sent = False

        self.v_cmd = 0.0
        self.w_cmd = 0.0
        self.volts = (0.0,) * 4
        self.omegas = (0.0,) * 4

        self.last_state = ''
        self.stop_ticks = 0
        self.completed = False

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )

        self.annotated_pub = self.create_publisher(
            Image,
            'camera/annotated',
            1
        )

        self.arm_client = ActionClient(
            self,
            MoveArm,
            'move_arm'
        )

        self.create_subscription(
            Image,
            self.camera_topic,
            self.on_image,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        )

        self.create_subscription(
            CameraInfo,
            self.info_topic,
            self.on_camera_info,
            qos_profile_sensor_data
        )

        self.create_subscription(
            Range,
            self.tof_topic,
            self.on_tof,
            qos_profile_sensor_data
        )

        self.create_subscription(
            Odometry,
            'odom',
            self.on_odom,
            qos_profile_sensor_data
        )

        self.create_subscription(
            PointStamped,
            'arm_position',
            self.on_arm,
            qos_profile_sensor_data
        )

        for file in (self.csv_path, self.fast_path):
            os.makedirs(
                os.path.dirname(os.path.abspath(file)),
                exist_ok=True
            )

        self.slow_file = open(
            self.csv_path,
            'w',
            newline=''
        )

        self.fast_file = open(
            self.fast_path,
            'w',
            newline=''
        )

        self.slow_csv = csv.writer(self.slow_file)
        self.fast_csv = csv.writer(self.fast_file)

        self.slow_csv.writerow(CSV_COLUMNS)

        self.fast_csv.writerow([
            't_s', 'v_mps', 'w_radps',
            'V1_V', 'V2_V', 'V3_V', 'V4_V',
            'state', 'camera_distance_m', 'tof_m',
            'bbox_w', 'bbox_h'
        ])

        self.begin = self.now()
        self.last_tick = self.begin
        self.next_log = 0.0

        self.create_timer(
            1.0 / self.rate,
            self.tick
        )

        self.get_logger().info(
            f'AP navigation ready. '
            f'OpenCV detects ALL tags; '
            f'chasing ID {self.target_id}. '
            f'Search rate={fsm_config.search_rate:.2f} rad/s. '
            f'Arm X={self.arm_x:.3f}, Z={self.arm_z:.3f} m. '
            f'ARRIVED: width >= {fsm_config.arrival_width_min:.0%}. '
            f'BRAKE: ToF <= {fsm_config.stop_distance:.2f} m. '
            f'lost_timeout={fsm_config.lost_timeout:.2f} s. '
            f'Chase speed limit={fsm_config.v_max:.2f} m/s.'
        )

    def now(self):
        return (
            self.get_clock().now().nanoseconds * 1e-9
        )

    def on_camera_info(self, msg):
        K = np.array(
            msg.k,
            dtype=float
        ).reshape(3, 3)

        if (
            K[0, 0] <= 0
            or K[1, 1] <= 0
            or not np.isfinite(K).all()
        ):
            self.K = None
            return

        self.K = K

        self.distortion = (
            np.array(msg.d, dtype=float)
            if msg.d
            else None
        )

        self.camera_frame = (
            msg.header.frame_id or self.camera_frame
        )

    def on_tof(self, msg):
        now = self.now()

        if (
            not math.isfinite(msg.range)
            or msg.range < msg.min_range
        ):
            self.tof_distance = None
            self.tof_time = -1e9
            return

        self.tof_time = now

        self.tof_distance = (
            None
            if msg.range >= msg.max_range - 1e-3
            else float(msg.range)
        )
        if self.tof_distance is None:
            self.tof_time = -1e9

    def on_odom(self, msg):
        self.odom_time = self.now()

        t = msg.twist.twist
        position = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w*q.z + q.x*q.y),
                         1 - 2 * (q.y*q.y + q.z*q.z))
        if all(math.isfinite(value) for value in
               (position.x, position.y, yaw, t.linear.x, t.angular.z)):
            self.odom_pose = (position.x, position.y, yaw)
            self.actual_velocity = (t.linear.x, t.angular.z)
        else:
            self.odom_pose = None
            self.odom_time = -1e9

        self.odom_stopped = (
            abs(t.linear.x) < 0.005
            and abs(t.linear.y) < 0.005
            and abs(t.angular.z) < 0.03
        )

    def on_arm(self, msg):
        self.arm_feedback = (msg.point.x, msg.point.z)
        self.arm_feedback_frame = msg.header.frame_id
        if msg.header.frame_id != 'arm_base_link':
            return

        was_verified = self.arm_verified
        self.arm_time = self.now()

        pose_ok = (
            math.isfinite(msg.point.x)
            and math.isfinite(msg.point.z)
            and abs(msg.point.x - self.arm_x) <= self.arm_pose_tolerance + 1e-9
            and abs(msg.point.z - self.arm_z) <= self.arm_pose_tolerance + 1e-9
        )
        bypassed = self.skip_arm_check and not pose_ok
        self.arm_verified = pose_ok or self.skip_arm_check
        if self.arm_verified and not was_verified:
            self.detection_time = -1e9
            if bypassed:
                self.get_logger().warn(
                    'ARM CHECK BYPASSED (skip_arm_check:=true): '
                    f'measured X={msg.point.x * 1000:.1f}, Y={msg.point.z * 1000:.1f} mm, '
                    f'required X={self.arm_x * 1000:.0f}, Y={self.arm_z * 1000:.0f} mm. '
                    'Camera TF/distance are not valid for the assessment.')
            else:
                self.get_logger().info(
                    f'ARM READY: X={self.arm_x * 1000:.0f} mm, '
                    f'Y={self.arm_z * 1000:.0f} mm verified from feedback; '
                    f'measured X={msg.point.x * 1000:.1f}, Y={msg.point.z * 1000:.1f} mm (ROS Z)')
        if was_verified and not self.arm_verified:
            self.get_logger().error('Arm pose changed; stopping chassis')

    def send_arm(self):
        if (
            self.arm_sent
            or self.arm_failed
            or not self.arm_client.server_is_ready()
        ):
            return

        self.arm_sent = True
        self.arm_attempts += 1
        self.get_logger().info(
            f'ARM SETUP (attempt {self.arm_attempts}/{self.arm_max_attempts}): '
            f'commanding X={self.arm_x * 1000:.0f} mm, '
            f'Y={self.arm_z * 1000:.0f} mm (ROS Z)')

        x = float(
            np.nextafter(
                np.float32(self.arm_x),
                np.float32(np.inf)
            )
        )

        z = float(
            np.nextafter(
                np.float32(self.arm_z),
                np.float32(np.inf)
            )
        )

        goal = MoveArm.Goal(
            x=x,
            z=z,
            relative=False
        )

        self.arm_client.send_goal_async(
            goal
        ).add_done_callback(self.on_arm_goal)

    def on_arm_goal(self, future):
        try:
            handle = future.result()

            if not handle.accepted:
                self.arm_retry_or_fail('goal rejected')
            else:
                handle.get_result_async().add_done_callback(
                    self.on_arm_result
                )

        except Exception as exc:
            self.arm_retry_or_fail(f'action failed: {exc}')

    def on_arm_result(self, future):
        try:
            self.arm_succeeded = (
                future.result().status
                == GoalStatus.STATUS_SUCCEEDED
            )

            if not self.arm_succeeded:
                self.arm_retry_or_fail('aborted / did not succeed')
            else:
                self.get_logger().info(
                    'Arm action complete; '
                    'verifying arm_position feedback'
                )

        except Exception as exc:
            self.arm_failed = True
            self.get_logger().error(
                f'Arm result failed: {exc}'
            )

    def arm_retry_or_fail(self, why):
        if self.arm_feedback is None:
            measured = 'no arm_position feedback'
        else:
            x, z = self.arm_feedback
            measured = f'measured X={x * 1000:.1f}, Y={z * 1000:.1f} mm'
        required = (f'required X={self.arm_x * 1000:.0f}, '
                    f'Y={self.arm_z * 1000:.0f} mm (+/-{self.arm_pose_tolerance * 1000:.0f})')

        if self.arm_verified:
            self.get_logger().warn(
                f'Arm action {why}, but feedback is within tolerance '
                f'({measured}); continuing')
        elif self.arm_attempts < self.arm_max_attempts:
            self.get_logger().warn(
                f'Arm action {why}; {measured}; {required}. '
                f'Retrying in 1 s (attempt {self.arm_attempts}/{self.arm_max_attempts} used)')
            self.arm_sent = False
            self.arm_retry_at = self.now() + 1.0
        else:
            self.arm_failed = True
            self.get_logger().error(
                f'Arm action {why} after {self.arm_attempts} attempts; '
                f'{measured}; {required}. Chassis stays stopped.')

    def camera_transform(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                rclpy.time.Time()
            )

            stamp = (
                rclpy.time.Time.from_msg(
                    trans.header.stamp
                ).nanoseconds * 1e-9
            )

            if (
                stamp > 0
                and abs(self.now() - stamp)
                > self.fsm.p.lost_timeout
            ):
                self.tf_ok = False
                return

            self.r_base_camera = quaternion_rotation(
                trans.transform.rotation
            )

            pos = trans.transform.translation

            self.t_base_camera = np.array(
                [pos.x, pos.y, pos.z],
                dtype=float
            )

            self.tf_ok = True

        except tf2_ros.TransformException:
            self.tf_ok = False

    def on_image(self, msg):
        stamp = (
            rclpy.time.Time.from_msg(
                msg.header.stamp
            ).nanoseconds * 1e-9
        )

        now = self.now()

        if not math.isfinite(stamp) or stamp <= 0 or abs(now - stamp) > self.camera_timeout:
            self.get_logger().warn(
                f'Dropping stale camera frame: age {now - stamp:.3f} s '
                f'(camera timeout {self.camera_timeout:.3f} s)',
                throttle_duration_sec=1.0)
            return

        # Every accepted frame counts, including frames skipped by throttling.
        self.camera_time = stamp

        if stamp - self.last_processed < self.process_interval:
            return

        self.last_processed = stamp

        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

            self.camera_transform()

            detections = self.detector.detect_all(
                frame,
                self.K,
                self.distortion
            )

            target = next(
                (
                    d
                    for d in detections
                    if d.tag_id == self.target_id
                ),
                None
            )

            # Pixel coverage is usable even when PnP or TF cannot produce a pose.
            if target is not None:
                self.bbox_w = target.bbox_w_fraction
                self.bbox_h = target.bbox_h_fraction
                height, width = frame.shape[:2]
                midpoint = (target.corners.min(axis=0) + target.corners.max(axis=0)) / 2
                self.center_x_error = float(midpoint[0] - width // 2)
                self.center_y_error = float(midpoint[1] - height // 2)
                focal = float(self.K[0, 0]) if self.K is not None else float(width)
                self.target_bearing = math.atan2(width / 2 - midpoint[0], focal)
                if self.arm_verified:
                    self.detection_time = stamp
                self.fsm.check_target_width(True, self.bbox_w,
                                           self.center_x_error, self.center_y_error,
                                           tof=(self.tof_distance if now - self.tof_time <= self.fsm.p.lost_timeout else None))
                width_stop = self.fsm.state in (ARRIVED, HOLD)
            else:
                width_stop = self.fsm.state in (ARRIVED, HOLD)
                self.detection_time = -1e9

            if width_stop or (target is None and self.fsm.state == CHASING):
                self.cmd_pub.publish(Twist())
                self.v_cmd = self.w_cmd = 0.0

            if (
                self.arm_verified
                and target is not None
                and target.tvec is not None
                and self.tf_ok
            ):
                p = (
                    self.r_base_camera @ target.tvec
                    + self.t_base_camera
                )

                if (
                    np.isfinite(p).all()
                    and p[0] > 0
                ):
                    self.target_distance = float(
                        math.hypot(p[0], p[1])
                    )

                    self.bbox_w = target.bbox_w_fraction
                    self.bbox_h = target.bbox_h_fraction
                    self.detection_time = stamp

            visible = (
                now - self.detection_time
            ) <= self.fsm.p.lost_timeout

            overlay = self.detector.annotate(
                frame,
                detections,
                self.target_id,
                self.fsm.state if self.arm_verified else 'ARM SETUP',
                (
                    self.target_distance
                    if visible
                    else None
                ),
                self.tof_distance,
                self.fsm.p.arrival_width_min
            )

            annotated = self.bridge.cv2_to_imgmsg(
                overlay,
                encoding='bgr8'
            )

            annotated.header = msg.header

            self.annotated_pub.publish(
                annotated
            )

        except Exception as exc:
            self.get_logger().warn(
                f'OpenCV/TF frame processing error: {exc}',
                throttle_duration_sec=2.0
            )

    def interfaces_ready(self, now):
        fresh = self.fsm.p.lost_timeout

        return (
            self.K is not None
            and self.tf_ok
            and 0 <= now - self.camera_time <= self.camera_timeout
            and now - self.odom_time <= fresh
            and self.cmd_pub.get_subscription_count() > 0
        )

    def readiness_details(self, now):
        fresh = self.fsm.p.lost_timeout
        checks = {
            'camera calibration missing': self.K is not None,
            'camera TF unavailable': self.tf_ok,
            f'camera stale/missing (frame age={now - self.camera_time:.3f}s, limit={self.camera_timeout:.3f}s)': 0 <= now - self.camera_time <= self.camera_timeout,
            'odometry stale/missing': now - self.odom_time <= fresh,
            'cmd_vel driver subscriber missing': self.cmd_pub.get_subscription_count() > 0,
            'motion disabled': self.use_motion,
            'initial zero not sent': self.zero_sent,
            'arm pose not verified': self.arm_verified,
            'arm feedback stale/missing': now - self.arm_time <= fresh,
        }
        reasons = [reason for reason, passed in checks.items() if not passed]
        if self.arm_feedback is None:
            arm = 'no arm_position messages received'
        else:
            x, z = self.arm_feedback
            arm = (f'arm measured X={x:.4f}, Z={z:.4f} m, '
                   f'frame={self.arm_feedback_frame!r}; '
                   f'required X={self.arm_x:.4f}, Z={self.arm_z:.4f} m (+/-0.003 m)')
        return '; '.join(reasons + [arm])

    def tick(self):
        if self.completed:
            self.cmd_pub.publish(Twist())
            return

        now = self.now()

        dt = max(
            now - self.last_tick,
            1e-3
        )

        self.last_tick = now

        visible = (
            now - self.detection_time
        ) <= 0.20

        tof = (
            self.tof_distance
            if now - self.tof_time
            <= self.fsm.p.lost_timeout
            else None
        )

        obs = Observation(
            now=now,
            visible=visible,
            bearing=(
                self.target_bearing
                if visible
                else 0.0
            ),
            distance=(
                self.target_distance
                if visible
                else float('inf')
            ),
            tof=tof,
            observed_at=(
                self.detection_time
                if visible
                else None
            ),
            bbox_w_fraction=(
                self.bbox_w
                if visible
                else 0.0
            ),
            bbox_h_fraction=(
                self.bbox_h
                if visible
                else 0.0
            ),
            center_x_error=self.center_x_error,
            center_y_error=self.center_y_error
        )

        # As in the independent test, collect initial arm feedback before
        # sending the absolute target. Camera/ToF/odometry do not gate this.
        if (not self.arm_sent and not self.arm_failed
                and now >= self.arm_retry_at
                and now - self.arm_time <= self.fsm.p.lost_timeout):
            self.cmd_pub.publish(Twist())
            self.zero_sent = True
            self.send_arm()

        arm_ready = (
            self.arm_verified
            and now - self.arm_time <= self.fsm.p.lost_timeout
        )

        ready = self.interfaces_ready(now)

        if (
            ready
            and arm_ready
            and self.zero_sent
            and self.use_motion
        ):
            self.was_ready = True
            v, w = self.fsm.update(obs)
            if self.fsm.state == SEARCHING and v == 0.0 and w != 0.0:
                if self.odom_pose is not None:
                    was_advancing = self.search_cycle.forward_until is not None
                    v, w = self.search_cycle.command(self.odom_pose[2], w, now)
                    if v > 0.0 and not was_advancing:
                        self.get_logger().info(
                            'Search completed 360 degrees: forward at 0.1 m/s for 5 seconds')
                else:
                    v, w = 0.0, 0.0
                    self.search_cycle.reset()
            else:
                self.search_cycle.reset()

        else:
            self.search_cycle.reset()
            v, w = 0.0, 0.0

            # Say WHY the robot is stopped (previously invisible).
            self.get_logger().warn(
                'NOT READY: ' + self.readiness_details(now),
                throttle_duration_sec=1.0
            )

            if (
                self.was_ready
                and self.fsm.state
                not in (ARRIVED, HOLD)
            ):
                self.fsm._enter_search()

        self.v_cmd = float(v)
        self.w_cmd = float(w)

        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)

        self.cmd_pub.publish(twist)

        if v == 0 and w == 0:
            self.zero_sent = True

        self.omegas = wheel_rates(v, w)

        self.volts = self.voltage.step(
            self.omegas,
            dt
        )

        elapsed = now - self.begin

        def f(x):
            if x is None or not math.isfinite(x):
                return ''
            return f'{x:.4f}'

        self.fast_csv.writerow([
            f'{elapsed:.3f}',
            f(v),
            f(w),
            *[f(z) for z in self.volts],
            self.fsm.state,
            (
                f(obs.distance)
                if visible
                else ''
            ),
            f(tof),
            f(obs.bbox_w_fraction),
            f(obs.bbox_h_fraction)
        ])

        self.fast_file.flush()
        self.tracking.record(elapsed, self.fsm.state, self.odom_pose,
                             self.actual_velocity, (v, w), self.bbox_w,
                             self.fsm.traj if self.fsm.state == CHASING else None,
                             now - self.fsm.traj_t0 if self.fsm.traj_t0 is not None else 0.0)

        if self.fsm.state != self.last_state:
            self.get_logger().info(
                f'{self.last_state or "START"} '
                f'-> {self.fsm.state}; '
                f'camera={f(obs.distance)} '
                f'ToF={f(tof)} '
                f'W={obs.bbox_w_fraction:.2f} '
                f'H={obs.bbox_h_fraction:.2f}'
            )

            if self.fsm.state == HOLD:
                self.get_logger().warn(
                    'Safety HOLD without ARRIVED: '
                    + self.fsm.hold_reason
                )

                self.log_slow(
                    elapsed,
                    'HOLD (safety stop)',
                    obs
                )

            if self.fsm.state == ARRIVED:
                self.get_logger().info(
                    'ARRIVED confirmed: '
                    + self.fsm.arrival_reason
                )

                self.log_slow(
                    elapsed,
                    'BRAKING (width stop)',
                    obs
                )

            self.last_state = self.fsm.state

        if self.fsm.state == ARRIVED:
            self.stop_ticks += 1

            if self.stop_ticks == 3:
                self.log_slow(
                    elapsed,
                    'ARRIVED (confirmed)',
                    obs
                )

                self.completed = True
                self.slow_file.close()
                self.fast_file.close()
                self.finish_tracking()

            return

        if elapsed >= self.next_log:
            self.log_slow(
                round(elapsed, 2),
                self.fsm.state,
                obs
            )

            self.next_log = math.floor(elapsed) + 1.0

    def log_slow(self, elapsed, state, obs):
        def f(x):
            if x is None or not math.isfinite(x):
                return ''
            return f'{x:.3f}'

        row = [
            f(elapsed),
            *[f(z) for z in self.volts],
            state,
            f(self.v_cmd),
            f(self.w_cmd),
            (
                f(obs.distance)
                if obs.visible
                else ''
            ),
            (
                f(obs.bearing)
                if obs.visible
                else ''
            ),
            f(obs.tof),
            f(obs.bbox_w_fraction),
            f(obs.bbox_h_fraction),
            *[f(z) for z in self.omegas]
        ]

        self.slow_csv.writerow(row)
        self.slow_file.flush()

        self.get_logger().info(
            ' '.join(row[:6]) + ' ' + state
        )

    def finish_tracking(self):
        try:
            self.tracking.finish()
        except Exception as exc:
            self.get_logger().error(f'Could not generate tracking plots: {exc}')

    def shutdown(self):
        try:
            if rclpy.ok():
                for _ in range(3):
                    self.cmd_pub.publish(Twist())

        finally:
            self.finish_tracking()
            for file in (
                self.slow_file,
                self.fast_file
            ):
                if not file.closed:
                    file.close()


def main(args=None):
    rclpy.init(args=args)

    node = FlatnessNav()

    try:
        while rclpy.ok():
            rclpy.spin_once(
                node,
                timeout_sec=0.10
            )

        deadline = time.monotonic() + 0.4

        while (
            rclpy.ok()
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(
                node,
                timeout_sec=0.05
            )

    except (
        KeyboardInterrupt,
        ExternalShutdownException
    ):
        pass

    finally:
        node.shutdown()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()