"""Arm self-collision on the real CAD collision meshes (MuJoCo), with a clearance margin.

Only arm-vs-rest pairs are checked (legs are excluded/parallelogram internals are handled by the MJCF).
`project_arm` moves an offending arm pose to the nearest clear pose, preferring 'raise forward / open
outward / unfold elbow', which keeps the gesture's meaning.
"""
import itertools

import mujoco
import numpy as np

from .. import paths
from ..robot import ANAT, SIDE


class ArmCollision:
    def __init__(self, robot, clearance_mm=None):
        self.r = robot
        clearance = (clearance_mm if clearance_mm is not None else robot.motion["collision"]["clearance_mm"]) / 1000.0
        self.clearance = clearance
        self.headroom = robot.motion["collision"].get("projection_headroom_mm", 3.0) / 1000.0
        # private model copy: arm collision geoms get a margin so near-misses are reported as contacts
        self.m = mujoco.MjModel.from_xml_path(paths.SCENE_XML)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.arm_bodies = {s: {m.body(f"{SIDE[s]}_{b}").id for b in ("shoulder", "upper_arm", "hand")} for s in "lr"}
        self.all_arm = self.arm_bodies["l"] | self.arm_bodies["r"]
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in self.all_arm and m.geom_group[g] == 3:
                m.geom_margin[g] = clearance + 0.005
        self.floor = m.geom("floor").id
        # candidate corrections in anatomical degrees (dpitch, droll, delbow), cheapest first
        cands = itertools.product(range(0, 61, 5), range(0, 46, 4), range(0, -61, -15))
        self.cands = sorted(cands, key=lambda c: abs(c[0]) + abs(c[1]) + 0.7 * abs(c[2]))
        self._cache = {}

    def min_dist(self, q, side=None):
        """Smallest arm-to-body distance (m) for model pose q, per arm {'l': d, 'r': d} (or for one side)."""
        self.r.qpos(q, out=self.d.qpos)
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_collision(self.m, self.d)
        out = {"l": np.inf, "r": np.inf}
        m = self.m
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if self.floor in (c.geom1, c.geom2):
                continue
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for s in "lr":
                ab = self.arm_bodies[s]
                if (b1 in ab) != (b2 in ab):  # arm vs something that is not this arm
                    out[s] = min(out[s], c.dist)
        return out if side is None else out[side]

    def ok(self, q, side=None):
        d = self.min_dist(q)
        return all(d[s] >= self.clearance for s in ("lr" if side is None else side))

    def project_arm(self, A, side):
        """Anatomical pose row A (8,) -> pose with this arm clear (other channels untouched). None if impossible."""
        key = (side, tuple(np.round(A, 1)))
        if key in self._cache:
            return self._cache[key]
        cols = [ANAT.index(f"{side}_{j}") for j in ("pitch", "roll", "elbow")]
        lo, hi = self.r.anat_lo[cols], self.r.anat_hi[cols]
        best = None
        target = self.clearance + self.headroom  # headroom: servo lag / overshoot must not reach the body
        for c in self.cands:
            B = A.copy()
            B[cols] = np.clip(A[cols] + np.array(c, float), lo, hi)
            if self.min_dist(self.r.anat_to_q(B)[0], side) >= target:
                best = B
                break
        self._cache[key] = best
        return best

    def frame_dists(self, Q):
        """(N, 2) min distance in mm for the left and right arm."""
        out = np.zeros((len(Q), 2))
        for i, q in enumerate(Q):
            d = self.min_dist(q)
            out[i] = [min(d["l"], 1.0) * 1000, min(d["r"], 1.0) * 1000]
        return out
