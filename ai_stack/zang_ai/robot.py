"""Zang robot model: the joint contract (config/robot.json) bound to the MuJoCo model (zang_description).

Three angle spaces, never mixed:
  * model space   q[15] radians, wire order of robot.json (what sim, safety, ZLink and the firmware use)
  * anatomical    degrees a human reads: arm pitch + = forward/up, roll + = outward, elbow + = flexed toward
                  the midline, gaze + = turn to the robot's left, sway + = weight onto the robot's left foot.
                  The co-speech planner works here (it was written for Zing in these units).
  * servo ticks   firmware only (calibration in NVS). Python never sees ticks.

The anatomical signs are DERIVED from the MJCF at load (move a joint, see where the hand goes), so a CAD
re-export cannot silently flip a gesture.
"""
import json
from functools import cached_property

import mujoco
import numpy as np

from . import paths

ANAT = ["l_pitch", "l_roll", "l_elbow", "r_pitch", "r_roll", "r_elbow", "gaze", "sway"]
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "elbow")
SIDE = {"l": "left", "r": "right"}


def load_json(name):
    with open(paths.config(name), encoding="utf-8") as fh:
        return json.load(fh)


class RobotModel:
    def __init__(self, scene=paths.SCENE_XML):
        self.cfg = load_json("robot.json")
        self.motion = load_json("motion.json")
        J = self.cfg["joints"]
        assert [j["index"] for j in J] == list(range(len(J))), "robot.json joints must be listed in index order"
        self.names = [j["name"] for j in J]
        self.n = len(J)
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.lower = np.array([j["lower"] for j in J])
        self.upper = np.array([j["upper"] for j in J])
        self.home = np.array([j["home"] for j in J])
        self.group = [j["group"] for j in J]
        gl = self.cfg["group_limits"]
        self.vmax = np.array([gl[g]["vmax"] for g in self.group])
        self.amax = np.array([gl[g]["amax"] for g in self.group])

        self.m = mujoco.MjModel.from_xml_path(scene)
        self.d = mujoco.MjData(self.m)  # scratch data for kinematic queries (not thread-safe: one per RobotModel)
        m = self.m
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in self.names])
        self.dadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in self.names])
        self.act = np.array([m.actuator(n).id for n in self.names])
        self._mimic = self._parse_mimic()
        self.pelvis = m.body("pelvis").id
        self.feet = {s: m.body(f"{s}_foot").id for s in ("left", "right")}
        self.base_home = self._base_home()
        self.anat_sign = self._derive_signs()
        self.anat_lo, self.anat_hi = self._anat_envelope()

    # ---- model pose ---------------------------------------------------------------------------------
    def _parse_mimic(self):
        """Passive pins follow servos through joint equalities: q1 = c0 + c1*q2 (+ higher terms, unused)."""
        m, out = self.m, []
        for e in range(m.neq):
            if m.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            j1, j2 = m.eq_obj1id[e], m.eq_obj2id[e]
            c0, c1 = m.eq_data[e][0], m.eq_data[e][1]
            out.append((m.jnt_qposadr[j1], m.jnt_qposadr[j2], c0, c1))
        return out

    def _base_home(self):
        """Floating base pose at the 'home' keyframe (feet on the floor)."""
        key = self.m.key("home").id
        return self.m.key_qpos[key][:7].copy()

    def qpos(self, q, base=None, out=None):
        """model q[15] (+ base [x y z qw qx qy qz]) -> full qpos with the passive linkage pins filled in."""
        qp = np.zeros(self.m.nq) if out is None else out
        qp[:7] = self.base_home if base is None else base
        qp[self.qadr] = q
        for a1, a2, c0, c1 in self._mimic:
            qp[a1] = c0 + c1 * qp[a2]
        return qp

    def kinematics(self, q, base=None, d=None):
        d = d or self.d
        self.qpos(q, base, d.qpos)
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        return d

    # ---- anatomical <-> model -----------------------------------------------------------------------
    def _hand_tip(self, d, side):
        """A point near the fingertips: the hand body's geom AABB point farthest from the elbow axis."""
        b = self.m.body(f"{side}_hand").id
        return d.xpos[b] + d.xmat[b].reshape(3, 3) @ self._tip_local[side]

    @cached_property
    def _tip_local(self):
        out = {}
        for side in ("left", "right"):
            b = self.m.body(f"{side}_hand").id
            pts = []
            for g in range(self.m.ngeom):
                if self.m.geom_bodyid[g] == b and self.m.geom_group[g] == 3:
                    R = np.zeros(9)
                    mujoco.mju_quat2Mat(R, self.m.geom_quat[g])
                    mid = self.m.geom_dataid[g]
                    a, n = self.m.mesh_vertadr[mid], self.m.mesh_vertnum[mid]
                    pts.append(self.m.geom_pos[g] + self.m.mesh_vert[a:a + n] @ R.reshape(3, 3).T)
            pts = np.concatenate(pts)
            out[side] = pts[np.argmin(pts[:, 2])]  # lowest point of the hand with the arm hanging
        return out

    def _derive_signs(self):
        """Sign per anatomical channel such that + means the documented anatomical direction."""
        sign = {}
        d = mujoco.MjData(self.m)
        dq = 0.05
        for s, side in SIDE.items():
            outward = 1.0 if side == "left" else -1.0
            for k, j in enumerate(ARM_JOINTS):
                q = self.home.copy()
                if j == "elbow":  # judge elbow flexion with the upper arm forward (forearm then swings sideways)
                    q[self.idx[f"{side}_shoulder_pitch"]] = -0.6 if sign.get(f"{s}_pitch", -1) < 0 else 0.6
                self.kinematics(q, d=d)
                p0 = self._hand_tip(d, side).copy()
                q[self.idx[f"{side}_{j}"]] += dq
                self.kinematics(q, d=d)
                delta = self._hand_tip(d, side) - p0
                if j == "shoulder_pitch":
                    sign[f"{s}_pitch"] = 1.0 if delta[0] > 0 else -1.0       # forward = +x
                elif j == "shoulder_roll":
                    sign[f"{s}_roll"] = 1.0 if delta[1] * outward > 0 else -1.0
                else:
                    sign[f"{s}_elbow"] = 1.0 if delta[1] * outward < 0 else -1.0  # toward the midline
        # gaze: + turns the chest to the robot's left (+yaw about z)
        q = self.home.copy()
        self.kinematics(q, d=d)
        c = self.m.body("chest").id
        x0 = d.xmat[c].reshape(3, 3)[:, 0].copy()
        q[self.idx["waist_yaw"]] += dq
        self.kinematics(q, d=d)
        sign["gaze"] = 1.0 if np.cross(x0, d.xmat[c].reshape(3, 3)[:, 0])[2] > 0 else -1.0
        # sway: + puts the CoM over the left foot (hip roll and ankle roll counter-rotate; feet stay flat)
        sign["sway"] = 1.0
        com0 = self._com_rel_feet(self.home, d)
        com1 = self._com_rel_feet(self.home + self._sway_vec(1.0, 1.0) * np.radians(1.0), d)
        sign["sway"] = 1.0 if com1[1] > com0[1] else -1.0
        return sign

    def _sway_vec(self, s, sign):
        v = np.zeros(self.n)
        for side in ("left", "right"):
            v[self.idx[f"{side}_hip_roll"]] = sign * s
            v[self.idx[f"{side}_ankle_roll"]] = -sign * s
        return v

    def _com_rel_feet(self, q, d):
        self.kinematics(q, d=d)
        mid = 0.5 * (d.xpos[self.feet["left"]] + d.xpos[self.feet["right"]])
        return d.subtree_com[self.pelvis] - mid

    def _anat_envelope(self):
        e = self.motion["cospeech_envelope_deg"]
        lo = np.array([e["arm_pitch"][0], e["arm_roll"][0], e["elbow"][0]] * 2 + [e["gaze"][0], -1.0])
        hi = np.array([e["arm_pitch"][1], e["arm_roll"][1], e["elbow"][1]] * 2 + [e["gaze"][1], 1.0])
        return lo, hi

    def anat_to_q(self, A):
        """Anatomical (N, 8) degrees (sway in -1..1) -> model q (N, 15) radians. Unlisted joints stay at home."""
        A = np.atleast_2d(A)
        Q = np.repeat(self.home[None], len(A), 0)
        for s, side in SIDE.items():
            for k, j in enumerate(("pitch", "roll", "elbow")):
                col = ANAT.index(f"{s}_{j}")
                Q[:, self.idx[f"{side}_{ARM_JOINTS[k]}"]] += self.anat_sign[f"{s}_{j}"] * np.radians(A[:, col])
        Q[:, self.idx["waist_yaw"]] += self.anat_sign["gaze"] * np.radians(A[:, ANAT.index("gaze")])
        sway_rad = np.radians(self.motion["sway"]["deg"]) * A[:, ANAT.index("sway")]
        Q += sway_rad[:, None] * self._sway_vec(1.0, self.anat_sign["sway"])[None]
        return Q

    def q_to_anat(self, Q):
        Q = np.atleast_2d(Q)
        A = np.zeros((len(Q), len(ANAT)))
        for s, side in SIDE.items():
            for k, j in enumerate(("pitch", "roll", "elbow")):
                jj = self.idx[f"{side}_{ARM_JOINTS[k]}"]
                A[:, ANAT.index(f"{s}_{j}")] = self.anat_sign[f"{s}_{j}"] * np.degrees(Q[:, jj] - self.home[jj])
        A[:, ANAT.index("gaze")] = self.anat_sign["gaze"] * np.degrees(Q[:, self.idx["waist_yaw"]] - self.home[self.idx["waist_yaw"]])
        hr = self.idx["left_hip_roll"]
        A[:, ANAT.index("sway")] = self.anat_sign["sway"] * np.degrees(Q[:, hr] - self.home[hr]) / self.motion["sway"]["deg"]
        return A

    # ---- balance (quasi-static) ----------------------------------------------------------------------
    @cached_property
    def _sole_pts(self):
        """Per foot: sole contact points in the foot body frame (2D hull of the lowest collision vertices)."""
        d = mujoco.MjData(self.m)
        self.kinematics(self.home, d=d)
        out = {}
        for side, b in self.feet.items():
            pts = []
            for g in range(self.m.ngeom):
                if self.m.geom_bodyid[g] != b or self.m.geom_group[g] != 3:
                    continue
                mid = self.m.geom_dataid[g]
                a, n = self.m.mesh_vertadr[mid], self.m.mesh_vertnum[mid]
                v = d.geom_xpos[g] + self.m.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T
                pts.append(v)
            pts = np.concatenate(pts)
            low = pts[pts[:, 2] < pts[:, 2].min() + 0.0015]
            hull = _hull(low[:, :2])
            Rb, pb = d.xmat[b].reshape(3, 3), d.xpos[b]
            z = np.full((len(hull), 1), low[:, 2].min())
            out[side] = (np.hstack([hull, z]) - pb) @ Rb  # world -> body frame
        return out

    def support(self, d):
        """Support polygon (2D hull, world xy) of both soles for the current kinematics in d."""
        pts = [d.xpos[b] + self._sole_pts[s] @ d.xmat[b].reshape(3, 3).T for s, b in self.feet.items()]
        return _hull(np.concatenate(pts)[:, :2])

    def com_margin(self, Q, d=None):
        """Static balance for frames Q (N, 15): CoM (N, 3) m and signed margin to the support polygon (N,) mm."""
        d = d or self.d
        Q = np.atleast_2d(Q)
        com, margin = np.zeros((len(Q), 3)), np.zeros(len(Q))
        for i, q in enumerate(Q):
            self.kinematics(q, d=d)
            com[i] = d.subtree_com[self.pelvis]
            margin[i] = 1000.0 * hull_margin(self.support(d), com[i, :2])
        return com, margin

    # ---- viewer data -----------------------------------------------------------------------------------
    def body_poses(self, Q, base=None, d=None):
        """Frames -> world pose of every body: (N, nbody, 7) [x y z qw qx qy qz]. base: (N, 7) or None (home)."""
        d = d or self.d
        Q = np.atleast_2d(Q)
        out = np.zeros((len(Q), self.m.nbody, 7))
        for i, q in enumerate(Q):
            self.kinematics(q, None if base is None else base[i], d)
            out[i, :, :3] = d.xpos
            out[i, :, 3:] = d.xquat
        return out


def _hull(P):
    """2D convex hull (Andrew's monotone chain), counter-clockwise."""
    P = np.unique(np.round(np.asarray(P, float), 7), axis=0)
    if len(P) < 3:
        return P

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and cross(h[-2], h[-1], p) <= 0:
                h.pop()
            h.append(p)
        return h

    pts = [tuple(p) for p in P]
    return np.array(half(pts)[:-1] + half(pts[::-1])[:-1])


def hull_margin(hull, q):
    """Signed distance (m) from point q to a CCW convex polygon; positive inside."""
    if len(hull) < 3:
        return -1.0
    d = np.inf
    for a, b in zip(hull, np.roll(hull, -1, 0)):
        e = b - a
        L = np.hypot(*e)
        if L < 1e-9:
            continue
        n = np.array([-e[1], e[0]]) / L  # inward normal for CCW
        d = min(d, float(np.dot(q - a, n)))
    return d
