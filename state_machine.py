"""Search, align and pursue Tag 3; separate safety holds from arrival."""

import math
from dataclasses import dataclass
from typing import Optional

from .flatness import FlatTrajectory


SEARCHING = 'SEARCHING'
ALIGNING = 'ALIGNING'
CHASING = 'CHASING'
HOLD = 'HOLD'
ARRIVED = 'ARRIVED'


@dataclass
class Params:
    lost_timeout: float = 0.45
    search_rate: float = 0.15
    align_tol: float = 0.18
    realign_tol: float = 0.30
    settle_time: float = 0.30
    align_gain: float = 0.8
    align_min_rate: float = 0.10
    align_max_rate: float = 0.24
    chase_bearing_gain: float = 0.8
    chase_max_w: float = 0.30
    stop_distance: float = 0.20
    tof_min_range: float = 0.10
    v_max: float = 0.16
    min_chase_speed: float = 0.14
    alignment_width_max: float = 0.15
    arrival_width_min: float = 0.17
    center_tolerance: float = 5.0  # Euclidean image-center error in pixels


@dataclass
class Observation:
    now: float
    visible: bool = False
    bearing: float = 0.0
    distance: float = float('inf')
    tof: Optional[float] = None
    observed_at: Optional[float] = None
    bbox_w_fraction: float = 0.0
    bbox_h_fraction: float = 0.0
    # Signed offsets from the crosshair, in pixels.
    center_x_error: float = float("inf")
    center_y_error: float = float("inf")


def clip(x, lo, hi):
    return max(lo, min(hi, x))


class NavStateMachine:

    def __init__(self, params=None):
        self.p = params or Params()

        self.state = SEARCHING
        self.last_seen = None
        self.last_dir = 1.0

        self.centered_since = None
        self.traj = None
        self.traj_t0 = None

        self.last_distance = float('inf')
        self.arrival_reason = ''
        self.hold_reason = ''

    def _enter_search(self):
        if self.state == ARRIVED:
            return
        self.state = SEARCHING
        self.traj = None
        self.centered_since = None

    def target_width_accepted(self, obs):
        return (
            obs.visible
            and obs.bbox_w_fraction >= self.p.arrival_width_min
        )

    def check_target_width(self, visible, width, center_x=float("inf"), center_y=float("inf"), tof=None):
        """Width declares arrival; ToF independently brakes into a safety hold."""
        if self.state == ARRIVED:
            return True
        if visible and math.isfinite(width) and width >= self.p.arrival_width_min:
            self.state = ARRIVED
            self.arrival_reason = (f'Target width {width:.1%} >= '
                                   f'{self.p.arrival_width_min:.1%}')
        elif tof is not None and math.isfinite(tof) and tof > 0:
            if tof <= self.p.stop_distance:
                self.state = HOLD
                self.hold_reason = (f'ToF {tof:.3f} m <= '
                                    f'{self.p.stop_distance:.2f} m braking distance')
            else:
                if self.state == HOLD:
                    self.hold_reason = ''
                    self._enter_search()
                return False
        else:
            # Missing range cannot release a previously established brake.
            return self.state == HOLD
        self.traj = None
        return True

    def update(self, obs):
        p = self.p

        if self.state == ARRIVED:
            return 0.0, 0.0

        if obs.visible and obs.observed_at is not None:
            self.last_seen = obs.observed_at
            self.last_distance = obs.distance

            if abs(obs.bearing) > 1e-3:
                self.last_dir = (
                    1.0 if obs.bearing > 0.0 else -1.0
                )

        if self.check_target_width(obs.visible, obs.bbox_w_fraction,
                                   obs.center_x_error, obs.center_y_error, tof=obs.tof):
            return 0.0, 0.0

        # Final approach ignores crosshair alignment and pose-derived distance.
        if obs.visible and obs.bbox_w_fraction > p.alignment_width_max:
            self.state = CHASING
            self.traj = None
            return min(p.min_chase_speed, p.v_max), 0.0

        lost = (
            self.last_seen is None
            or obs.now - self.last_seen > p.lost_timeout
        )

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

                if (
                    obs.now - self.centered_since
                    >= p.settle_time
                ):
                    self.traj = FlatTrajectory(
                        max(
                            obs.distance,
                            p.stop_distance + 1e-3
                        ),
                        v_max=p.v_max
                    )

                    self.traj_t0 = obs.now
                    self.state = CHASING

                    return self._chase(obs)

                return 0.0, 0.0

            self.centered_since = None

            w = clip(
                p.align_gain * obs.bearing,
                -p.align_max_rate,
                p.align_max_rate
            )

            if abs(w) < p.align_min_rate:
                w = (
                    p.align_min_rate
                    if w >= 0
                    else -p.align_min_rate
                )

            return 0.0, w

        if self.state == CHASING:
            # Stop translation and rotation as soon as the target observation expires.
            if not obs.visible:
                return 0.0, 0.0
            if (
                obs.visible
                and abs(obs.bearing) > p.realign_tol
            ):
                self.state = ALIGNING
                self.traj = None
                self.centered_since = None
                return 0.0, 0.0

            return self._chase(obs)

        return 0.0, 0.0

    def _chase(self, obs):
        if self.traj is None:
            self._enter_search()
            return 0.0, self.p.search_rate * self.last_dir

        t = obs.now - self.traj_t0

        if self.traj.finished(t):
            if (
                obs.visible
                and obs.distance > self.p.stop_distance
            ):
                self.traj = FlatTrajectory(
                    obs.distance,
                    v_max=self.p.v_max
                )

                self.traj_t0 = obs.now
                t = 0.0

            else:
                self._enter_search()
                return 0.0, self.p.search_rate * self.last_dir

        flat_v, flat_w, _, _ = self.traj.sample(t)

        v = clip(
            flat_v,
            self.p.min_chase_speed,
            self.p.v_max
        )

        if obs.visible:
            remaining = max(
                0.0,
                obs.distance - self.p.stop_distance
            )

            v = min(
                v,
                max(
                    self.p.min_chase_speed,
                    0.45 * remaining
                )
            )

        w = flat_w

        if obs.visible:
            w += self.p.chase_bearing_gain * obs.bearing

        return v, clip(
            w,
            -self.p.chase_max_w,
            self.p.chase_max_w
        )