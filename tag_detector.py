"""OpenCV AprilTag detection, bounding boxes, and Tag 3 pose."""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


FAMILIES = {
    'tag16h5': 'DICT_APRILTAG_16h5',
    'tag25h9': 'DICT_APRILTAG_25h9',
    'tag36h10': 'DICT_APRILTAG_36h10',
    'tag36h11': 'DICT_APRILTAG_36h11'
}


@dataclass
class Detection:
    tag_id: int
    corners: np.ndarray
    tvec: Optional[np.ndarray] = None
    bbox_w_fraction: float = 0.0
    bbox_h_fraction: float = 0.0


class TagDetector:

    def __init__(
        self,
        family='tag36h11',
        target_id=3,
        tag_size=0.15
    ):
        if family not in FAMILIES:
            raise ValueError(
                f'Unsupported AprilTag family: {family}'
            )

        self.target_id = int(target_id)

        dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, FAMILIES[family])
        )

        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            parameters = cv2.aruco.DetectorParameters_create()
        else:
            parameters = cv2.aruco.DetectorParameters()

        parameters.cornerRefinementMethod = (
            cv2.aruco.CORNER_REFINE_APRILTAG
        )

        self.dictionary = dictionary
        self.parameters = parameters

        if hasattr(cv2.aruco, 'ArucoDetector'):
            self.detector = cv2.aruco.ArucoDetector(
                dictionary,
                parameters
            )
        else:
            self.detector = None

        s = float(tag_size) / 2.0

        self.obj = np.array([
            [-s,  s, 0],
            [ s,  s, 0],
            [ s, -s, 0],
            [-s, -s, 0]
        ], dtype=np.float32)

    def detect_all(self, frame, K, distortion=None):
        if frame.ndim == 2:
            gray = frame
        else:
            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

        height, width = gray.shape[:2]
        result = {}

        for scale in (1.0, 2.0):
            if scale == 1.0:
                img = gray
            else:
                img = cv2.resize(
                    gray,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC
                )

            if self.detector is not None:
                polygons, ids, _ = (
                    self.detector.detectMarkers(img)
                )
            else:
                polygons, ids, _ = cv2.aruco.detectMarkers(
                    img,
                    self.dictionary,
                    parameters=self.parameters
                )

            if ids is not None:
                for shape, ident in zip(
                    polygons,
                    ids.flatten()
                ):
                    ident = int(ident)

                    if ident in result:
                        continue

                    corners = (
                        shape.reshape(4, 2) / scale
                    ).astype(np.float32)

                    width_frac = float(
                        (
                            corners[:, 0].max()
                            - corners[:, 0].min()
                        ) / width
                    )

                    height_frac = float(
                        (
                            corners[:, 1].max()
                            - corners[:, 1].min()
                        ) / height
                    )

                    detection = Detection(
                        ident,
                        corners,
                        None,
                        width_frac,
                        height_frac
                    )

                    if (
                        ident == self.target_id
                        and K is not None
                    ):
                        coeff = (
                            np.zeros(5, dtype=float)
                            if distortion is None
                            else distortion
                        )

                        ok, _, tvec = cv2.solvePnP(
                            self.obj,
                            corners,
                            np.asarray(K, dtype=float),
                            np.asarray(coeff, dtype=float),
                            flags=cv2.SOLVEPNP_IPPE_SQUARE
                        )

                        if (
                            ok
                            and np.isfinite(tvec).all()
                            and tvec[2, 0] > 0
                        ):
                            detection.tvec = tvec.reshape(3)

                    result[ident] = detection

            if self.target_id in result:
                break

        return list(result.values())

    def annotate(
        self,
        frame,
        detections,
        target_id=3,
        state='SEARCHING',
        camera_distance=None,
        tof=None,
        arrival_width=0.17
    ):
        out = frame.copy()
        h, w = out.shape[:2]

        cv2.drawMarker(
            out,
            (w // 2, h // 2),
            (240, 240, 240),
            cv2.MARKER_CROSS,
            22,
            2
        )

        for item in detections:
            target = item.tag_id == target_id

            color = (
                (40, 210, 70)
                if target
                else (10, 190, 240)
            )

            poly = np.round(
                item.corners
            ).astype(np.int32)

            cv2.polylines(
                out,
                [poly],
                True,
                color,
                3 if target else 2
            )

            x0 = int(np.clip(
                poly[:, 0].min(), 0, w - 1
            ))

            y0 = int(np.clip(
                poly[:, 1].min(), 0, h - 1
            ))

            x1 = int(np.clip(
                poly[:, 0].max(), 0, w - 1
            ))

            y1 = int(np.clip(
                poly[:, 1].max(), 0, h - 1
            ))

            cv2.rectangle(
                out,
                (x0, y0),
                (x1, y1),
                color,
                1
            )

            suffix = ' TARGET' if target else ''
            label = f'AprilTag {item.tag_id}{suffix}'

            cv2.putText(
                out,
                label,
                (x0, max(25, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA
            )

            if target:
                midpoint = ((x0 + x1) // 2, (y0 + y1) // 2)
                cv2.drawMarker(out, midpoint, color, cv2.MARKER_CROSS, 16, 2)
                cv2.line(out, (w // 2, h // 2), midpoint, color, 1)
                coverage = (
                    f'Frame coverage '
                    f'W:{item.bbox_w_fraction:.0%} '
                    f'H:{item.bbox_h_fraction:.0%}  Arrive W>={arrival_width:.0%}; Brake ToF<=0.20m'
                )

                cv2.putText(
                    out,
                    coverage,
                    (10, max(65, h - 45)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA
                )

        def fmt(d):
            if d is None or not np.isfinite(d):
                return '--'
            return f'{d:.2f}m'

        text = (
            f'{state}  Target:{target_id}  '
            f'Camera:{fmt(camera_distance)}  '
            f'ToF:{fmt(tof)}'
        )

        cv2.rectangle(
            out,
            (0, 0),
            (w, 36),
            (10, 10, 10),
            -1
        )

        cv2.putText(
            out,
            text,
            (8, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        return out