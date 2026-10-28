"""Cubic flat-output trajectory using the assignment boundary conditions."""

import math
import numpy as np

EPS = 1e-6
V_MAX = 0.16
STANDOFF = 0.20
T_MIN = 4.0
T_GAIN = 1.6


def flat_inversion(xd, yd, xdd, ydd):
    speed_sq = xd * xd + yd * yd
    v = math.sqrt(speed_sq)

    if speed_sq < EPS:
        return v, 0.0

    w = (xd * ydd - yd * xdd) / speed_sq
    return v, w


def solve_flat_polynomial(z0, zdot0, zf, zdotf, T):
    A = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, T, T ** 2, T ** 3],
        [0.0, 1.0, 2.0 * T, 3.0 * T ** 2]
    ])

    b = np.array([z0, zdot0, zf, zdotf], dtype=float)

    return np.linalg.solve(A, b)


def eval_cubic(alpha, t):
    a0, a1, a2, a3 = alpha

    position = a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3
    velocity = a1 + 2.0 * a2 * t + 3.0 * a3 * t ** 2
    acceleration = 2.0 * a2 + 6.0 * a3 * t

    return position, velocity, acceleration


def time_horizon(
    d0,
    standoff=STANDOFF,
    v_max=V_MAX,
    t_min=T_MIN,
    gain=T_GAIN
):
    return max(t_min, gain * (d0 - standoff) / v_max)


class FlatTrajectory:

    def __init__(
        self,
        d0,
        v0=0.06,
        standoff=STANDOFF,
        v_max=V_MAX
    ):
        self.d0 = d0
        self.T = time_horizon(d0, standoff, v_max)

        goal = d0 - standoff

        self.alpha_x = solve_flat_polynomial(
            0.0, v0, goal, 0.0, self.T
        )

        self.alpha_y = solve_flat_polynomial(
            0.0, 0.0, 0.0, 0.0, self.T
        )

    def sample(self, t):
        t = min(max(t, 0.0), self.T)

        x, xd, xdd = eval_cubic(self.alpha_x, t)
        y, yd, ydd = eval_cubic(self.alpha_y, t)

        v, w = flat_inversion(xd, yd, xdd, ydd)

        if xd < 0.0:
            v = 0.0

        return v, w, x, y

    def finished(self, t):
        return t >= self.T