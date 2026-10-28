"""Capture ROS checkerboard images and write a driver-compatible calibration.

Run with --help. Board dimensions count INNER corners, not squares.
"""
import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import yaml


def board_points(columns, rows, square_size):
    points = np.zeros((columns * rows, 3), np.float32)
    points[:, :2] = np.mgrid[:columns, :rows].T.reshape(-1, 2) * square_size
    return points


def calibrate(image_points, image_size, columns, rows, square_size):
    if len(image_points) < 15:
        raise ValueError('Capture at least 15 different checkerboard views first.')
    objects = [board_points(columns, rows, square_size) for _ in image_points]
    rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(
        objects, image_points, image_size, None, None)
    if not (np.isfinite(rms) and np.isfinite(matrix).all()
            and np.isfinite(distortion).all() and matrix[0, 0] > 0 and matrix[1, 1] > 0):
        raise ValueError('Calibration failed: non-finite or invalid camera parameters.')
    errors = []
    for obj, measured, rotation, translation in zip(objects, image_points, rotations, translations):
        projected, _ = cv2.projectPoints(obj, rotation, translation, matrix, distortion)
        errors.append(float(np.sqrt(np.mean(np.sum((measured - projected) ** 2, axis=2)))))
    return float(rms), matrix, distortion, errors


def calibration_yaml(image_size, matrix, distortion):
    def entry(array):
        array = np.asarray(array, dtype=float)
        return dict(rows=array.shape[0], cols=array.shape[1], data=array.ravel().tolist())
    projection = np.zeros((3, 4))
    projection[:, :3] = matrix
    return dict(image_width=int(image_size[0]), image_height=int(image_size[1]),
                camera_name='robomaster_camera', distortion_model='plumb_bob',
                camera_matrix=entry(matrix),
                distortion_coefficients=entry(np.asarray(distortion).reshape(1, -1)),
                rectification_matrix=entry(np.eye(3)), projection_matrix=entry(projection))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--columns', type=int, default=9, help='Inner corner columns (default 9)')
    parser.add_argument('--rows', type=int, default=6, help='Inner corner rows (default 6)')
    parser.add_argument('--square-size', type=float, required=True, help='Measured square side in METRES')
    parser.add_argument('--topic', default='/camera/image_color')
    parser.add_argument('--output', type=Path, default=Path('src/rm_flatness_nav/config/camera_calibration.yaml'))
    args = parser.parse_args(argv)
    if args.columns < 3 or args.rows < 3 or not np.isfinite(args.square_size) or args.square_size <= 0:
        parser.error('Use at least 3x3 inner corners and a positive square size.')

    # ROS is needed only for live capture; calibration helpers also work offline.
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge

    rclpy.init(args=[])
    node = rclpy.create_node('calibrate_robomaster_camera')
    bridge = CvBridge()
    latest = [None, 0.]

    def receive(message):
        latest[:] = [message, time.monotonic()]

    node.create_subscription(Image, args.topic, receive,
                             QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
    image_points = []
    image_size = None
    last_captured = None
    output = args.output.expanduser().resolve()
    samples = output.parent / (output.stem + '_samples_' + time.strftime('%Y%m%d_%H%M%S'))
    print(f'Listening on {args.topic}. Keep the robot stationary and motion disabled.')
    print('Use a flat checkerboard. Move/tilt it across the image at different distances.')
    print('SPACE: capture detected corners | C: calibrate and save | Q: quit')
    print(f'Output: {output}\nSamples: {samples}')
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.01)
            message, received = latest
            if message is None:
                continue
            frame = bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
            size = (frame.shape[1], frame.shape[0])
            if image_size is not None and image_size != size:
                raise ValueError('Camera resolution changed. Restart and recapture all samples.')
            fresh = time.monotonic() - received < 1.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, (args.columns, args.rows),
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
            preview = frame.copy()
            if found:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, .001))
                cv2.drawChessboardCorners(preview, (args.columns, args.rows), corners, found)
            text = f'{len(image_points)} views | {"board found" if found else "no board"} | SPACE capture, C save, Q quit'
            if not fresh:
                text = 'CAMERA STALE: cannot capture'
            cv2.putText(preview, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 255, 255), 1)
            cv2.imshow('RoboMaster camera calibration', preview)
            key = cv2.waitKey(10) & 0xff
            if key in (ord('q'), 27):
                break
            if key == ord(' ') and found and fresh and message is not last_captured:
                # Avoid repeatedly sampling an unchanged board pose.
                if image_points and min(float(np.sqrt(np.mean((corners - old) ** 2)))
                                        for old in image_points) < 8.:
                    print('Move or tilt the board more before capturing another view.')
                    continue
                image_size = size
                samples.mkdir(parents=True, exist_ok=True)
                path = samples / f'view_{len(image_points):03d}.png'
                if not cv2.imwrite(str(path), frame):
                    raise OSError(f'Could not save {path}')
                image_points.append(corners.copy())
                last_captured = message
                print(f'Captured view {len(image_points)} ({size[0]}x{size[1]})')
            elif key in (ord('c'), ord('C')):
                try:
                    rms, matrix, distortion, errors = calibrate(image_points, image_size,
                        args.columns, args.rows, args.square_size)
                except (ValueError, cv2.error) as exc:
                    print(exc)
                    continue
                print(f'RMS reprojection error: {rms:.3f} px; per-view errors: {errors}')
                if rms > 1.:
                    print('RMS exceeds 1 pixel. Recapture sharper, varied board views; calibration not saved.')
                    continue
                if output.exists():
                    backup = output.with_name(output.name + f'.{time.time_ns()}.bak')
                    backup.write_bytes(output.read_bytes())
                    print(f'Previous calibration backed up to {backup}')
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(yaml.safe_dump(calibration_yaml(image_size, matrix, distortion), sort_keys=False))
                print(f'Saved {output}. Use the same camera resolution and zoom when navigating.')
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
