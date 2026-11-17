"""Physics twin: plays a model trajectory on the floating-base MuJoCo model the way the robot will.

  * command path: 50 Hz frames, linear ramp in between (what the firmware's trajectory player does),
    servo latency, servo internal speed limit, then the MJCF position actuator (kp, force limit, friction,
    armature from zang_description). The DS-R003G is NOT identified yet: config/motion.json servo_model and
    the MJCF actuator values are placeholders; replace them after a bench identification (rl/README.md).
  * closed parallelogram loops are enforced by the MJCF joint equalities.
  * measured per frame: joint tracking, base tilt, centre of pressure (= ZMP for flat feet) and its margin to
    the support polygon, foot slip, actuator load vs force limit, self-contacts, fall.
"""
import mujoco
import numpy as np

from .. import paths
from ..robot import hull_margin, _hull


class PhysicsTwin:
    def __init__(self, robot):
        self.r = robot
        self.m = mujoco.MjModel.from_xml_path(paths.SCENE_XML)
        self.d = mujoco.MjData(self.m)
        sm = robot.motion["servo_model"]
        self.latency = sm["latency_ms"] / 1000.0
        self.servo_vmax = np.radians(sm["vmax_dps"])
        m = self.m
        self.floor = m.geom("floor").id
        self.feet = [robot.feet["left"], robot.feet["right"]]
        self.force_lim = np.abs(m.actuator_forcerange[robot.act, 1])
        self.checks = robot.motion["physics_checks"]

    def reset(self, q0=None):
        m, d, r = self.m, self.d, self.r
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        q0 = r.home if q0 is None else q0
        r.qpos(q0, d.qpos[:7].copy(), d.qpos)
        d.ctrl[r.act] = q0
        mujoco.mj_forward(m, d)

    def simulate(self, Q, fps=50, settle=0.5):
        """Q (N, 15) commanded frames -> dict of per-frame arrays (see module doc)."""
        m, d, r = self.m, self.d, self.r
        Q = np.atleast_2d(Q)
        self.reset(Q[0])
        for _ in range(int(settle / m.opt.timestep)):
            mujoco.mj_step(m, d)
        n_sub = max(1, int(round(1.0 / fps / m.opt.timestep)))
        dt = m.opt.timestep
        lag = int(round(self.latency / dt))
        N = len(Q)
        out = {k: np.zeros(s) for k, s in (("q", (N, r.n)), ("base", (N, 7)), ("tilt", N), ("cop", (N, 2)),
                                            ("margin", N), ("load", (N, r.n)), ("slip", N), ("self_contacts", N))}
        foot0 = np.array([d.xpos[b, :2].copy() for b in self.feet])
        target = Q[0].copy()
        hist = [Q[0].copy()] * (lag + 1)
        for k in range(N):
            q_next = Q[min(k + 1, N - 1)]
            load = np.zeros(r.n)
            for s in range(n_sub):
                u = (s + 1) / n_sub
                hist.append(Q[k] + (q_next - Q[k]) * u)  # firmware: linear ramp between stream frames
                cmd = hist.pop(0)                          # servo latency
                target += np.clip(cmd - target, -self.servo_vmax * dt, self.servo_vmax * dt)
                d.ctrl[r.act] = target
                mujoco.mj_step(m, d)
                load = np.maximum(load, np.abs(d.actuator_force[r.act]) / self.force_lim)
            out["q"][k] = d.qpos[r.qadr]
            out["base"][k] = d.qpos[:7]
            up = d.xmat[r.pelvis].reshape(3, 3)[:, 2]
            out["tilt"][k] = np.degrees(np.arccos(np.clip(up[2], -1, 1)))
            out["cop"][k], out["margin"][k], out["self_contacts"][k] = self._contacts()
            out["load"][k] = load
            out["slip"][k] = 1000 * max(np.linalg.norm(d.xpos[b, :2] - foot0[i]) for i, b in enumerate(self.feet))
        out["cmd"] = Q
        return out

    def _contacts(self):
        m, d = self.m, self.d
        f6 = np.zeros(6)
        pts, fz, n_self = [], [], 0
        for i in range(d.ncon):
            c = d.contact[i]
            if self.floor in (c.geom1, c.geom2):
                mujoco.mj_contactForce(m, d, i, f6)
                if f6[0] > 1e-6:
                    pts.append(c.pos[:2].copy())
                    fz.append(f6[0])
            elif c.dist < -1e-4:
                n_self += 1
        if not fz:
            return np.zeros(2), -100.0, n_self
        pts, fz = np.array(pts), np.array(fz)
        cop = (pts * fz[:, None]).sum(0) / fz.sum()
        return cop, 1000.0 * hull_margin(_hull(pts), cop), n_self


def report(twin, sim):
    c = twin.checks
    err = np.degrees(np.sqrt(np.mean((sim["q"] - sim["cmd"]) ** 2, axis=0)))
    fallen = bool((sim["tilt"] > 30).any() or (sim["base"][:, 2] < 0.6 * sim["base"][0, 2]).any())
    rep = {
        "max_tilt_deg": round(float(sim["tilt"].max()), 2),
        "min_zmp_margin_mm": round(float(sim["margin"].min()), 1),
        "max_slip_mm": round(float(sim["slip"].max()), 2),
        "max_load_pct": round(float(sim["load"].max() * 100), 1),
        "tracking_rms_deg": round(float(err.max()), 2),
        "worst_tracking_joint": twin.r.names[int(np.argmax(err))],
        "self_contact_frames": int((sim["self_contacts"] > 0).sum()),
        "fallen": fallen,
    }
    checks = {
        "upright": rep["max_tilt_deg"] <= c["max_tilt_deg"] and not fallen,
        "zmp": rep["min_zmp_margin_mm"] >= c["min_zmp_margin_mm"],
        "feet_planted": rep["max_slip_mm"] <= c["max_slip_mm"],
        "servo_load": rep["max_load_pct"] <= c["max_torque_pct"],
        "tracking": rep["tracking_rms_deg"] <= c["max_tracking_rms_deg"],
        "no_self_contact": rep["self_contact_frames"] == 0,
    }
    rep["checks"] = checks
    rep["pass"] = all(checks.values())
    return rep
