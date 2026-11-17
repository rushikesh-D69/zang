"""Zang humanoid kinematics: FK, centre of mass, analytic leg/arm IK, whole-body IK, numeric fallback.

Geometry is read from ../zang_description/zang.urdf at load time, so this module follows the
robot model whenever it is regenerated. Units: metres and radians. Frames follow REP-103
(x forward, y left, z up). The maths is derived in IK_MATH.md.

    from zang_ik import Zang
    z = Zang()
    q = z.leg_ik("left", sole_pos=[0.0, 0.07, -0.21])               # pelvis frame
    q = z.whole_body_ik(pelvis_pos=[0, 0, 0.23],
                        feet={"left": ([0, 0.07, 0], 0.0), "right": ([0, -0.07, 0], 0.0)})
"""
import cmath
import csv
import math
import os
import struct
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_URDF = os.path.join(HERE, "..", "zang_description", "zang.urdf")
SIDES = ("left", "right")
# rest pose: knees bent forward; must match HOME in zang_description/tools/build.py (MJCF keyframe "home")
HOME = {f"{s}_{j}": -0.5 for s in SIDES for j in ("hip_pitch", "knee_pitch")}
EX, EY, EZ = np.eye(3)


class IKError(ValueError):
    pass


def rot(axis, q):
    """Rotation matrix for angle q about a unit axis (Rodrigues)."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * K @ K


def rpy(roll=0.0, pitch=0.0, yaw=0.0):
    """R = Rz(yaw) Ry(pitch) Rx(roll), the URDF/ROS convention."""
    return rot(EZ, yaw) @ rot(EY, pitch) @ rot(EX, roll)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def solve_trig(a, b, c):
    """All theta with a*cos(theta) + b*sin(theta) = c."""
    r = math.hypot(a, b)
    if r < 1e-12 or abs(c) > r * (1 + 1e-9):
        return []
    phi, delta = math.atan2(b, a), math.acos(max(-1.0, min(1.0, c / r)))
    return [wrap(phi + delta), wrap(phi - delta)]


def two_link(T, L1, L2):
    """Planar 2-link in complex form: link1 + link2 = T, |link1| = L1, |link2| = L2.
    Returns [(link1, link2, branch)] with branch +1 / -1 = link1 rotated CCW / CW from T."""
    t = abs(T)
    if t > L1 + L2 + 1e-9 or t < abs(L1 - L2) - 1e-9 or t < 1e-12:
        return []
    beta = math.acos(max(-1.0, min(1.0, (L1 * L1 + t * t - L2 * L2) / (2 * L1 * t))))
    out = []
    for branch in (1, -1):
        l1 = L1 * cmath.exp(1j * (cmath.phase(T) + branch * beta))
        out.append((l1, T - l1, branch))
    return out


def _stl_vertices(path):
    data = open(path, "rb").read()
    n = struct.unpack("<I", data[80:84])[0]
    rec = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", 9), ("a", "<u2")]))
    return rec["v"].reshape(-1, 3).astype(float)


class Zang:
    def __init__(self, urdf=DEFAULT_URDF):
        root = ET.parse(urdf).getroot()
        self.urdf_dir = os.path.dirname(os.path.abspath(urdf))
        self.links, self.joints = {}, {}
        for l in root.findall("link"):
            ine = l.find("inertial")
            I = ine.find("inertia").attrib
            self.links[l.get("name")] = dict(
                mass=float(ine.find("mass").get("value")),
                com=np.array(ine.find("origin").get("xyz").split(), float),
                inertia=np.array([[float(I["ixx"]), float(I["ixy"]), float(I["ixz"])],
                                  [float(I["ixy"]), float(I["iyy"]), float(I["iyz"])],
                                  [float(I["ixz"]), float(I["iyz"]), float(I["izz"])]]),
                meshes=[v.find("geometry/mesh").get("filename") for v in l.findall("visual")])
        for j in root.findall("joint"):
            o, lim, mim = j.find("origin"), j.find("limit"), j.find("mimic")
            assert np.allclose(np.array(o.get("rpy", "0 0 0").split(), float), 0), "joint frames must be unrotated"
            self.joints[j.get("name")] = dict(
                parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                xyz=np.array(o.get("xyz").split(), float), axis=np.array(j.find("axis").get("xyz").split(), float),
                lower=float(lim.get("lower")), upper=float(lim.get("upper")),
                mimic=None if mim is None else (mim.get("joint"), float(mim.get("multiplier", 1)), float(mim.get("offset", 0))))
        self.children = defaultdict(list)
        for n, j in self.joints.items():
            self.children[j["parent"]].append(n)
        self.root = (set(self.links) - {j["child"] for j in self.joints.values()}).pop()
        self.actuated = [n for n, j in self.joints.items() if j["mimic"] is None]
        self.mass = sum(l["mass"] for l in self.links.values())
        self.sole = {s: self._mesh_point(f"{s}_foot") for s in SIDES}
        self.hand_tip = {s: self._mesh_point(f"{s}_hand") for s in SIDES}
        self._check_structure()

    # ------------------------------------------------------------------ model helpers
    def _mesh_point(self, link):
        """Centre of the link's mesh bounding box in x/y, lowest point in z (sole / finger tip) at zero pose."""
        V = np.vstack([_stl_vertices(os.path.join(self.urdf_dir, f)) for f in self.links[link]["meshes"]])
        lo, hi = V.min(0), V.max(0)
        return np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])

    def _check_structure(self):
        """The analytic solvers assume this topology; fail loudly if the model changes."""
        J = self.joints
        for s in SIDES:
            for n, ax in (("hip_roll", EX), ("hip_pitch", EY), ("knee_upper", EY), ("knee_lower", EY),
                          ("knee_pitch", EY), ("ankle_roll", EX), ("shoulder_roll", EX), ("elbow", EX)):
                assert np.allclose(J[f"{s}_{n}"]["axis"], ax), f"{s}_{n} axis changed"
            assert J[f"{s}_knee_upper"]["mimic"][:2] == (f"{s}_hip_pitch", -1.0)
            assert J[f"{s}_knee_lower"]["mimic"][:2] == (f"{s}_knee_pitch", -1.0)
            assert abs(J[f"{s}_shoulder_pitch"]["axis"][0]) < 1e-6, "shoulder pitch axis must be perpendicular to x"

    def full_q(self, q=None):
        """Complete joint map: missing actuated joints = 0, mimic joints filled from their drivers."""
        q = dict(q or {})
        out = {n: float(q.get(n, 0.0)) for n in self.actuated}
        for n, j in self.joints.items():
            if j["mimic"]:
                drv, mul, off = j["mimic"]
                out[n] = mul * out[drv] + off
        return out

    def home(self):
        return self.full_q(HOME)

    def within_limits(self, q, tol=1e-6):
        return all(self.joints[n]["lower"] - tol <= v <= self.joints[n]["upper"] + tol for n, v in q.items() if n in self.joints)

    # ------------------------------------------------------------------ forward kinematics
    def fk(self, q=None, base_pos=(0, 0, 0), base_rot=np.eye(3)):
        """World transform (4x4) of every link frame."""
        q = self.full_q(q)
        T0 = np.eye(4)
        T0[:3, :3], T0[:3, 3] = base_rot, base_pos
        out, stack = {self.root: T0}, [self.root]
        while stack:
            p = stack.pop()
            for n in self.children[p]:
                j = self.joints[n]
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = rot(j["axis"], q[n]), j["xyz"]
                out[j["child"]] = out[p] @ T
                stack.append(j["child"])
        return out

    def point(self, frames, link, p_local):
        T = frames[link]
        return T[:3, :3] @ np.asarray(p_local, float) + T[:3, 3]

    def sole_pos(self, frames, side):
        return self.point(frames, f"{side}_foot", self.sole[side])

    def hand_pos(self, frames, side):
        return self.point(frames, f"{side}_hand", self.hand_tip[side])

    def com(self, frames):
        return sum(l["mass"] * self.point(frames, n, l["com"]) for n, l in self.links.items()) / self.mass

    # ------------------------------------------------------------------ analytic IK
    def leg_ik(self, side, sole_pos, foot_roll=0.0, hint=None, knee_forward=True):
        """Leg joints that put the sole centre at sole_pos (pelvis frame) with foot roll foot_roll
        (relative to the pelvis). Foot pitch and yaw always equal the pelvis's (see IK_MATH.md 3)."""
        J = self.joints
        r, h1 = J[f"{side}_hip_roll"]["xyz"], J[f"{side}_hip_pitch"]["xyz"]
        u, v = J[f"{side}_knee_upper"]["xyz"], J[f"{side}_knee_lower"]["xyz"]
        w, e = J[f"{side}_knee_pitch"]["xyz"], J[f"{side}_ankle_roll"]["xyz"]
        d = np.asarray(sole_pos, float) - r - rot(EX, foot_roll) @ self.sole[side]
        base = h1 + v + e
        U, W = complex(u[0], u[2]), complex(w[0], w[2])
        sols = []
        for phi in solve_trig(d[1], d[2], base[1] + u[1] + w[1]):
            dl = rot(EX, -phi) @ d - base
            for l1, l2, branch in two_link(complex(dl[0], dl[2]), abs(U), abs(W)):
                q = {f"{side}_hip_roll": phi,
                     f"{side}_hip_pitch": wrap(cmath.phase(U) - cmath.phase(l1)),
                     f"{side}_knee_pitch": wrap(cmath.phase(l2) - cmath.phase(W)),
                     f"{side}_ankle_roll": wrap(foot_roll - phi)}
                sols.append((q, branch == 1))
        return self._pick(sols, hint, knee_forward, f"{side} leg", sole_pos)

    def arm_ik(self, side, hand_pos, hint=None):
        """Arm joints that put the hand tip at hand_pos (chest frame). Position only: 3 DOF."""
        J = self.joints
        r, a = J[f"{side}_shoulder_pitch"]["xyz"], J[f"{side}_shoulder_pitch"]["axis"]
        a = a / np.linalg.norm(a)
        o1, o2, h = J[f"{side}_shoulder_roll"]["xyz"], J[f"{side}_elbow"]["xyz"], self.hand_tip[side]
        d = np.asarray(hand_pos, float) - r
        O2, H = complex(o2[1], o2[2]), complex(h[1], h[2])
        sols = []
        for q1 in solve_trig(d @ EX, d @ np.cross(a, EX), o1[0] + o2[0] + h[0]):
            ds = rot(a, -q1) @ d - o1
            for l1, l2, branch in two_link(complex(ds[1], ds[2]), abs(O2), abs(H)):
                q2 = cmath.phase(l1) - cmath.phase(O2)
                q3 = cmath.phase(l2) - cmath.phase(H) - q2
                sols.append(({f"{side}_shoulder_pitch": q1, f"{side}_shoulder_roll": wrap(q2),
                              f"{side}_elbow": wrap(q3)}, True))
        return self._pick(sols, hint, False, f"{side} arm", hand_pos)

    def _pick(self, sols, hint, prefer_flag, what, target):
        ok = [(q, f) for q, f in sols if self.within_limits(q)]
        if not ok:
            raise IKError(f"{what}: target {np.round(target, 4)} unreachable" + (" within joint limits" if sols else ""))
        hint = self.full_q(hint if hint is not None else HOME)
        cost = lambda qf: (prefer_flag and not qf[1], sum((v - hint[n]) ** 2 for n, v in qf[0].items()))
        return min(ok, key=cost)[0]

    # ------------------------------------------------------------------ whole body
    def whole_body_ik(self, pelvis_pos=(0, 0, 0), pelvis_rpy=(0, 0, 0), feet=None, hands=None, waist_yaw=0.0, hint=None):
        """Joint angles for a pelvis pose (world), foot targets {side: (sole_pos_world, foot_roll_world)}
        and optional hand targets {side: hand_pos_world}. Feet are flat in pitch/yaw relative to the pelvis,
        so with feet flat on the ground keep pelvis pitch = 0 and pelvis yaw = foot yaw."""
        Rp, pp = rpy(*pelvis_rpy), np.asarray(pelvis_pos, float)
        q = {"waist_yaw": waist_yaw}
        hint = self.full_q(hint if hint is not None else HOME)
        for side, (pos, roll) in (feet or {}).items():
            Rrel = Rp.T @ rpy(roll, 0, pelvis_rpy[2])
            if not np.allclose(Rrel, rot(EX, math.atan2(Rrel[2, 1], Rrel[2, 2])), atol=1e-6):
                raise IKError("feet can only roll relative to the pelvis: set pelvis pitch to 0")
            q.update(self.leg_ik(side, Rp.T @ (np.asarray(pos, float) - pp),
                                 math.atan2(Rrel[2, 1], Rrel[2, 2]), hint))
        if hands:
            wj = self.joints["waist_yaw"]
            Rc = Rp @ rot(wj["axis"], waist_yaw)
            pc = pp + Rp @ wj["xyz"]
            for side, pos in hands.items():
                q.update(self.arm_ik(side, Rc.T @ (np.asarray(pos, float) - pc), hint))
        for n in self.actuated:
            q.setdefault(n, hint[n])
        return self.full_q(q)

    def balance_ik(self, com_xy, pelvis_z, feet, pelvis_rpy=(0, 0, 0), hands=None, waist_yaw=0.0, iters=20, tol=1e-5):
        """Whole-body IK that also moves the pelvis horizontally until the CoM projection is at com_xy."""
        pp = np.array([com_xy[0], com_xy[1], pelvis_z], float)
        for _ in range(iters):
            q = self.whole_body_ik(pp, pelvis_rpy, feet, hands, waist_yaw)
            err = np.asarray(com_xy) - self.com(self.fk(q, pp, rpy(*pelvis_rpy)))[:2]
            if np.linalg.norm(err) < tol:
                break
            pp[:2] += err
        return q, pp

    # ------------------------------------------------------------------ numeric fallback
    def numeric_ik(self, targets, q0=None, joints=None, base_pos=(0, 0, 0), base_rot=np.eye(3), iters=200, damping=1e-3, tol=1e-6):
        """Damped least squares on any set of point targets {link: (point_in_link, world_target)}.
        Use for tasks the analytic solvers do not cover (e.g. a point on the forearm)."""
        joints = joints or self.actuated
        q = self.full_q(q0 if q0 is not None else HOME)
        def residual(qq):
            F = self.fk(qq, base_pos, base_rot)
            return np.concatenate([np.asarray(t, float) - self.point(F, l, p) for l, (p, t) in targets.items()])
        for _ in range(iters):
            r = residual(q)
            if np.linalg.norm(r) < tol:
                break
            Jm = np.zeros((len(r), len(joints)))
            for k, n in enumerate(joints):
                qd = dict(q)
                qd[n] += 1e-6
                Jm[:, k] = (r - residual(self.full_q(qd))) / 1e-6
            dq = Jm.T @ np.linalg.solve(Jm @ Jm.T + damping * np.eye(len(r)), r)
            for k, n in enumerate(joints):
                q[n] = float(np.clip(q[n] + dq[k], self.joints[n]["lower"], self.joints[n]["upper"]))
            q = self.full_q(q)
        return q

    # ------------------------------------------------------------------ tables
    def export_tables(self, out_dir=HERE):
        with open(os.path.join(out_dir, "joints.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["joint", "kind", "parent", "child", "origin_x_m", "origin_y_m", "origin_z_m",
                        "axis_x", "axis_y", "axis_z", "lower_rad", "upper_rad", "home_rad", "mimic_of", "mimic_multiplier"])
            home = self.home()
            for n, j in self.joints.items():
                w.writerow([n, "passive" if j["mimic"] else "servo", j["parent"], j["child"],
                            *[f"{x:.6f}" for x in j["xyz"]], *[f"{x:.6f}" for x in j["axis"]],
                            f"{j['lower']:.4f}", f"{j['upper']:.4f}", f"{home[n]:.4f}",
                            j["mimic"][0] if j["mimic"] else "", j["mimic"][1] if j["mimic"] else ""])
        with open(os.path.join(out_dir, "links.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["link", "mass_kg", "com_x_m", "com_y_m", "com_z_m", "ixx", "iyy", "izz", "ixy", "ixz", "iyz"])
            for n, l in self.links.items():
                I = l["inertia"]
                w.writerow([n, f"{l['mass']:.5f}", *[f"{x:.6f}" for x in l["com"]],
                            *[f"{x:.4e}" for x in (I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2])]])


if __name__ == "__main__":
    z = Zang()
    z.export_tables()
    J = z.joints
    u, v, w = (J[f"left_{n}"]["xyz"] for n in ("knee_upper", "knee_lower", "knee_pitch"))
    print(f"links {len(z.links)}, joints {len(z.joints)} ({len(z.actuated)} servos), mass {z.mass*1000:.0f} g")
    print(f"leg: thigh crank {np.linalg.norm(u[[0, 2]])*1000:.2f} mm, knee offset {np.linalg.norm(v)*1000:.2f} mm, "
          f"shin crank {np.linalg.norm(w[[0, 2]])*1000:.2f} mm")
    print("sole point (foot frame, mm):", {s: np.round(z.sole[s] * 1000, 2).tolist() for s in SIDES})
    print("hand tip (hand frame, mm):", {s: np.round(z.hand_tip[s] * 1000, 2).tolist() for s in SIDES})
    for name, q in (("zero", z.full_q()), ("home", z.home())):
        F = z.fk(q)
        print(f"{name}: left sole {np.round(z.sole_pos(F, 'left')*1000, 1)} mm, CoM {np.round(z.com(F)*1000, 1)} mm (pelvis frame)")
    print("wrote joints.csv, links.csv")
