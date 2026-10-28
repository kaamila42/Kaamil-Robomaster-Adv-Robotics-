# Verification record — 2026-10-25

## Completed in this session

- Inspected the supplied Downloads merged reference and existing workspace before
  replacing the package. Backed up the complete old package outside src, with
  backups/COLCON_IGNORE so colcon does not discover duplicate packages.
- Inspected installed driver source, action definition, launch arguments, camera
  calibration loading, EP camera URDF, chassis braking and velocity conversion.
- Moved previous package build/install artifacts into the backup and rebuilt
  cleanly to remove stale driver_node/planner_node executables.
- Imported installed driver, camera, arm, chassis and ToF modules and navigation
  using ROS Jazzy plus ~/rm_venv. OpenCV 4.6.0 has the required AprilTag ArUco API.
- Built with `python3 -m colcon build --symlink-install --packages-select
  rm_flatness_nav`. Package builds successfully; nav_node uses rm_venv Python.
- `ros2 launch rm_flatness_nav assessment2.launch.py --show-args` succeeds.
- Evaluated the complete launch description with process execution mocked.
  Resolved robomaster_driver, robot_state_publisher, joint_state_publisher,
  nav_node and image_view. Asserted final driver parameters: conn_type=ap,
  camera.enabled=true, camera.video.raw=1, arm.enabled=true, tof.enabled=true.
  This evaluation processes the actual driver includes and xacro description;
  no driver process, SDK connection or camera connection was started.
- Seven offline regression tests pass with real numerical algorithms, OpenCV,
  ROS message classes and CvBridge; ROS transport is mocked for node tests:
  1. Cubic endpoint positions/velocities, zero lateral polynomial, exact horizon,
     speed limits across distances, flat inversion including zero speed.
  2. Wheel rates and steady/acceleration/braking voltage examples.
  3. Alignment, settle, chase, remembered search direction, exact detection-time
     loss handling (no blind chasing beyond timeout).
  4. Camera/ToF 20 cm latch and no false ARRIVED on trajectory expiry.
  5. Synthetic tag36h11 ID 3 recognition, calibrated depth, wrong-ID rejection,
     bearing sign and horizontal projection.
  6. Node sensor gating, exact 105/80 mm arm goal after float32/driver conversion,
     incorrect-pose rejection, correct-pose feedback, chase, stale-sensor stop,
     ToF arrival, one-shot arm goal and both CSV outputs. CSVs were real temporary
     files; they were read back to verify braking and final ARRIVED records.
  7. Merged file imports and uses the same equations/FSM.

Repeat offline checks after sourcing ROS, venv and workspace:

```bash
python3 ~/ros2_ws/src/rm_flatness_nav/test/test_assessment.py
python3 ~/ros2_ws/src/rm_flatness_nav/test/test_launch_resolution.py
```

## Not physically verified

This execution environment refuses socket creation (`Operation not permitted`)
for ROS DDS/daemon discovery. `ros2 topic list --no-daemon` produced transport
errors; its local /rosout and /parameter_events entries are not evidence of a
robot. Action discovery also failed with PermissionError. No physical robot
motion or arm action was issued, and no hardware connection was opened. No
conflicting driver/navigation processes were visible to this session.

Still unverified: AP reachability; live robot topics/actions; actual video and
calibration delivery; physical tag detection; live optical TF and camera mounting
accuracy; ToF index 0 as the front sensor and its physical placement; arm
reachability and feedback at 105/80 mm; zero-command braking on the robot; actual
20 cm stopping accuracy and full-run telemetry. The node's readiness checks and
offline tests do not establish physical stopping performance.

## Non-moving hardware checks to run from the laptop terminal

Start `assessment2.launch.py enable_motion:=false` as described in README.
In another sourced terminal:

```bash
ros2 topic info /cmd_vel --verbose
ros2 topic hz /camera/image_color
ros2 topic echo /camera/camera_info --once
ros2 topic echo /range_0 --once
ros2 topic echo /odom --once
ros2 topic echo /arm_position --once
ros2 action info /move_arm
ros2 run tf2_ros tf2_echo base_link camera_optical_link
```

Inspect the image viewer, tag size and sensor placement. Confirm /cmd_vel is zero
and odometry is stationary. Stop this preflight launch before starting the active
launch. The active launch first commands and verifies the required arm pose, then
begins searching. Do not launch a second driver or the merged node concurrently.

## User-approved arm update — 2026-10-26

Changed the target and feedback verification to X=105 mm, Y=80 mm
(ROS x=0.105, z=0.080 m). The user reports Y=30 mm obstructs ToF and
Y=80 mm leaves it clear. The offline regression now rejects the old 30 mm
pose and checks the encoded integer hardware target is exactly 105/80 mm.
Hardware reachability, clearance and braking remain unverified.

## Live preflight attempt — 2026-10-26

Ran the real launch with enable_motion=false and view_camera=false. Found and
fixed the existing installed driver's system-Python entrypoint (it could not
import robomaster from rm_venv) by rebuilding the existing robomaster_ros package
under rm_venv. No additional driver or SDK connection was created.

The retry still reported DDS socket permission errors. No live sensor or
physical braking verification was possible. Preflight CSVs were created and
read back; every recorded chassis command was zero. They are not navigation
results. Full retry output: ~/ros2_ws/log/assessment2/preflight_20261026.log.

Fixed a shutdown error exposed by SIGINT: do not publish after the ROS context
is shut down, and close both CSV files even during shutdown.
