"""Record odometry against the planned cubic and export plots without a GUI."""
import csv
import math
from pathlib import Path

COLUMNS = ('t_s', 'state', 'actual_x_m', 'actual_y_m', 'actual_yaw_rad',
           'planned_x_m', 'planned_y_m', 'planned_yaw_rad',
           'ref_v_mps', 'actual_v_mps', 'ref_w_radps', 'actual_w_radps',
           'error_x_m', 'error_y_m', 'error_yaw_rad', 'error_v_mps',
           'error_w_radps', 'tag_width_fraction')


class TrackingLog:
    def __init__(self, directory, plots=True):
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.file = (self.directory / 'tracking.csv').open('w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(COLUMNS)
        self.plots = plots
        self.reference = None
        self.previous = None
        self.trajectory = None
        self.anchor = None
        self.closed = False

    def record(self, t, state, pose, actual, command, width, trajectory=None, trajectory_t=0):
        if self.closed or pose is None:
            return
        x, y, yaw = pose
        v, w = command
        if self.reference is None:
            self.reference = pose
        if self.previous is not None:
            pt, pv, pw = self.previous
            dt = max(0., t - pt)
            rx, ry, rh = self.reference
            if abs(pw) < 1e-9:
                rx += pv * math.cos(rh) * dt
                ry += pv * math.sin(rh) * dt
            else:
                rx += pv / pw * (math.sin(rh + pw * dt) - math.sin(rh))
                ry -= pv / pw * (math.cos(rh + pw * dt) - math.cos(rh))
            self.reference = rx, ry, rh + pw * dt
        if trajectory is not None:
            if trajectory is not self.trajectory:
                self.anchor = pose
            _, _, px, py = trajectory.sample(trajectory_t)
            ax, ay, ah = self.anchor
            self.reference = (ax + math.cos(ah)*px - math.sin(ah)*py,
                              ay + math.sin(ah)*px + math.cos(ah)*py, ah)
        self.trajectory = trajectory
        rx, ry, rh = self.reference
        av, aw = actual
        row = (t, state, x, y, yaw, rx, ry, rh, v, av, w, aw,
               rx-x, ry-y, math.atan2(math.sin(rh-yaw), math.cos(rh-yaw)),
               v-av, w-aw, width)
        self.writer.writerow(row)
        self.file.flush()
        self.previous = t, v, w

    def finish(self):
        if self.closed:
            return
        self.file.close()
        self.closed = True
        if self.plots:
            plot_tracking(self.directory)


def plot_tracking(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(directory)
    with (directory / 'tracking.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return
    def values(key):
        return [float(row[key]) for row in rows]
    t = values('t_s')
    # Export each requested quantity separately as well as the comparisons.
    for filename, prefix, title in (
            ('reference_position.png', 'planned', 'Reference position'),
            ('actual_position.png', 'actual', 'Actual position (odometry)')):
        fig, axes = plt.subplots(2, 1, sharex=True)
        fig.suptitle(title)
        for ax, coordinate in zip(axes, ('x', 'y')):
            ax.plot(t, values(f'{prefix}_{coordinate}_m'))
            ax.set_ylabel(f'{coordinate} (m)')
            ax.grid()
        axes[-1].set_xlabel('Time (s)')
        fig.tight_layout(); fig.savefig(directory/filename); plt.close(fig)
    fig, axes = plt.subplots(2, 1, sharex=True)
    fig.suptitle('Reference velocity (command)')
    for ax, key, label in zip(axes, ('ref_v_mps', 'ref_w_radps'),
                              ('v (m/s)', 'w (rad/s)')):
        ax.plot(t, values(key))
        ax.set_ylabel(label)
        ax.grid()
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout(); fig.savefig(directory/'reference_velocity.png'); plt.close(fig)
    fig, ax = plt.subplots()
    ax.plot(values('planned_x_m'), values('planned_y_m'), '--', label='Planned reference')
    ax.set(xlabel='x (m)', ylabel='y (m)', title='Planned reference path')
    ax.axis('equal'); ax.grid(); ax.legend()
    fig.tight_layout(); fig.savefig(directory/'planned_path.png'); plt.close(fig)
    fig, ax = plt.subplots()
    ax.plot(values('planned_x_m'), values('planned_y_m'), '--', label='Planned reference')
    ax.plot(values('actual_x_m'), values('actual_y_m'), label='Actual odometry')
    ax.set(xlabel='x (m)', ylabel='y (m)', title='Planned and actual robot path')
    ax.axis('equal'); ax.grid(); ax.legend()
    fig.tight_layout(); fig.savefig(directory/'path.png'); plt.close(fig)
    fig, axes = plt.subplots(2, 1, sharex=True)
    for ax, ref, actual, label in zip(axes, ('ref_v_mps', 'ref_w_radps'),
            ('actual_v_mps', 'actual_w_radps'), ('v (m/s)', 'w (rad/s)')):
        ax.plot(t, values(ref), label='Reference command')
        ax.plot(t, values(actual), label='Actual odometry')
        ax.set_ylabel(label); ax.grid(); ax.legend()
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout(); fig.savefig(directory/'velocity.png'); plt.close(fig)
    fig, axes = plt.subplots(3, 1, sharex=True)
    for key in ('error_x_m', 'error_y_m'):
        axes[0].plot(t, values(key), label=key)
    axes[0].set_ylabel('Position error (m)')
    axes[1].plot(t, values('error_yaw_rad'), label='Heading error (rad)')
    for key in ('error_v_mps', 'error_w_radps'):
        axes[2].plot(t, values(key), label=key)
    axes[2].set_xlabel('Time (s)')
    for ax in axes:
        ax.grid(); ax.legend()
    fig.tight_layout(); fig.savefig(directory/'tracking_errors.png'); plt.close(fig)
