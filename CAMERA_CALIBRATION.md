# RoboMaster camera calibration

Use a flat printed checkerboard with **9 by 6 inner corners** (10 by 7 squares).
Measure one square side accurately. The example below assumes **25 mm** squares;
replace `0.025` with your measured size in metres. Keep the camera resolution
and zoom the same as navigation (the launch currently uses 640x360).

From `/home/kaamil/ros2_ws`, source ROS and this workspace in both terminals:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

In terminal 1, run the camera with chassis motion disabled:

```bash
ros2 launch rm_flatness_nav assessment2.launch.py enable_motion:=false view_camera:=false
```

In terminal 2, start the calibration tool (no rebuild needed for module execution):

```bash
~/rm_venv/bin/python3 -m rm_flatness_nav.calibrate_camera --columns 9 --rows 6 --square-size 0.025
```

The tool needs a graphical desktop for its preview window. Move the board rather
than the robot. Show the entire board in sharp focus, varying position, distance,
and tilt. Cover the center, edges, and corners of the image.

- Press **Space** to capture each view with detected corners. Very similar views
  are rejected. Capture at least 15 varied views; 20–30 is preferable.
- Press **C** to calculate and save. The tool prints overall and per-view
  reprojection errors in pixels. An overall RMS above 1 pixel is rejected; quit
  and recapture sharper, more varied views. Low error alone does not prove that
  the calibration is accurate: validate using a known distance afterward.
- Press **Q** to quit without saving.

The tool saves raw capture images beside the output for inspection. Existing
calibration files are backed up before replacement. Default output:
`src/rm_flatness_nav/config/camera_calibration.yaml`.

Stop the first launch and restart navigation with the new calibration:

```bash
ros2 launch rm_flatness_nav assessment2.launch.py enable_motion:=true camera_calibration_file:=/home/kaamil/ros2_ws/src/rm_flatness_nav/config/camera_calibration.yaml
```

First validate distance with motion disabled using the same command with
`enable_motion:=false`. Measure the black AprilTag square side (excluding its
white margin) and set `tag_size:=<metres>` correctly. Camera calibration determines
intrinsics and distortion; it does not calibrate the camera-to-base transform or
arm geometry. The navigation camera distance is measured in the robot base ground
plane, while ToF measures along its own beam, so the two values need not match
exactly. Large residual disagreement needs a pose/transform check as well.

Optional: after `colcon build --packages-select rm_flatness_nav --symlink-install`
and sourcing `install/setup.bash`, the tool is also available as
`ros2 run rm_flatness_nav calibrate_camera --square-size 0.025`.
