"""Wheel kinematics and estimated motor voltages."""

WHEEL_RADIUS = 0.05
CHASSIS_RADIUS = 0.20

K_V = 0.38
K_A = 0.06


def wheel_rates(
    vx,
    wz,
    r=WHEEL_RADIUS,
    lc=CHASSIS_RADIUS
):
    w1 = (vx + lc * wz) / r
    w2 = (vx - lc * wz) / r

    w3 = w2
    w4 = w1

    return w1, w2, w3, w4


class MotorVoltageModel:

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
            accel = tuple(
                (w - p) / dt
                for w, p in zip(omegas, self._prev)
            )

        self._prev = tuple(omegas)

        return tuple(
            self.kv * w + self.ka * a
            for w, a in zip(omegas, accel)
        )