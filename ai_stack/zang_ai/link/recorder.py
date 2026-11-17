"""Record what the robot really did: STATE at its native rate while commands play. Used by the sim-to-real tools."""
import threading
import time

import numpy as np


class StateRecorder:
    """Collects (host time, robot ms, q_meas, q_cmd) from a connected RobotLink until stopped."""

    def __init__(self, link):
        self.link = link
        self.rows = []
        self._prev = link.on_state
        self._lock = threading.Lock()

    def __enter__(self):
        def cb(s):
            with self._lock:
                self.rows.append((time.perf_counter(), s.t_robot_ms, np.array(s.q_meas), np.array(s.q_cmd)))
            if self._prev:
                self._prev(s)
        self.link.on_state = cb
        return self

    def __exit__(self, *exc):
        self.link.on_state = self._prev

    def arrays(self, t0=None):
        """-> t (s, host clock from t0), robot_ms, q_meas (N, 15) rad, q_cmd (N, 15) rad."""
        with self._lock:
            rows = list(self.rows)
        if not rows:
            return np.zeros(0), np.zeros(0), np.zeros((0, 15)), np.zeros((0, 15))
        t = np.array([r[0] for r in rows])
        return (t - (t0 if t0 is not None else t[0]), np.array([r[1] for r in rows]),
                np.stack([r[2] for r in rows]), np.stack([r[3] for r in rows]))


def estimate_delay(t, cmd, meas, max_lag=0.3, dt=0.005):
    """Delay (s) that best aligns meas with cmd (cross-correlation of the velocity profiles)."""
    tt = np.arange(t[0], t[-1], dt)
    c = np.gradient(np.interp(tt, t, cmd))
    m = np.gradient(np.interp(tt, t, meas))
    lags = np.arange(0, int(max_lag / dt))
    score = [np.dot(c[:len(c) - k], m[k:]) for k in lags]
    return float(lags[int(np.argmax(score))] * dt)
