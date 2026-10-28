"""Alternate an odometry-measured search turn with a timed forward move."""
import math


class SearchCycle:
    def __init__(self, forward_duration=5.0):
        self.forward_duration = forward_duration
        self.reset()

    def reset(self):
        self.previous = None
        self.rotation = 0.0
        self.forward_until = None

    def command(self, yaw, direction, now, speed=0.1):
        if self.forward_until is not None:
            if now < self.forward_until:
                return speed, 0.0
            # Start a new turn from the heading reached after translation.
            self.reset()
        if self.previous is not None:
            self.rotation += math.atan2(math.sin(yaw - self.previous),
                                        math.cos(yaw - self.previous))
        self.previous = yaw
        if abs(self.rotation) >= 2 * math.pi:
            self.forward_until = now + self.forward_duration
            return speed, 0.0
        return 0.0, direction
