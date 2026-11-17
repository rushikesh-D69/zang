"""The safety stage: every motion passes through here before it reaches the twin, an export or the robot.

  1. envelope    anatomical co-speech range, then the model joint limits; legs held to small co-speech limits
  2. collision   arm vs body on the CAD collision meshes with clearance, projected to the nearest clear pose
  3. balance     static CoM inside the feet polygon by a margin, else the motion shrinks toward home
  4. rate        per-joint velocity + acceleration limits with look-ahead (start early, never arrive late)
  5. verify      re-check collision + balance on the final trajectory; anything left is reported, never hidden

The firmware repeats a simpler version of 1 and 4 as the last line of defence (robot_joints.h).
"""
import numpy as np

from ..robot import ANAT
from .collision import ArmCollision


def rate_limit(q, vmax, amax, dt):
    """Track q with |v| <= vmax, |a| <= amax, braking in time (no overshoot)."""
    out = np.empty_like(q)
    out[0] = q[0]
    v = np.zeros(q.shape[1])
    for k in range(1, len(q)):
        err = q[k] - out[k - 1]
        v_des = np.clip(err / dt, -vmax, vmax)
        brake = np.sqrt(2 * amax * np.abs(err))
        v_des = np.clip(v_des, -brake, brake)
        v = np.clip(v_des, v - amax * dt, v + amax * dt)
        out[k] = out[k - 1] + v * dt
    return out


def lookahead_limit(q, vmax, amax, dt, q_start=None):
    """Forward + backward pass averaged (anticipation), then forward again (guarantee).
    q_start pins the first sample (streaming: continue exactly from the previous chunk)."""
    fwd = rate_limit(q, vmax, amax, dt)
    bwd = rate_limit(q[::-1], vmax, amax, dt)[::-1]
    mid = 0.5 * (fwd + bwd)
    mid[0] = q[0] if q_start is None else q_start
    return rate_limit(mid, vmax, amax, dt)


def _hold_and_smooth(delta, fps, hold_s=0.25):
    """Correction offsets: hold the largest correction +-hold_s around where it was needed, then smooth,
    so a projected arm does not twitch in and out of the clear pose."""
    n = len(delta)
    w = max(1, int(hold_s * fps))
    held = delta.copy()
    for k in range(n):
        seg = delta[max(0, k - w):k + w + 1]
        idx = np.argmax(np.abs(seg), axis=0)
        held[k] = seg[idx, np.arange(seg.shape[1])]
    win = np.hanning(max(3, int(0.3 * fps)))
    win /= win.sum()
    pad = len(win)
    sm = np.stack([np.convolve(np.pad(held[:, c], pad, mode="edge"), win, "same")[pad:-pad]
                   for c in range(held.shape[1])], 1)
    big = np.abs(held) > np.abs(sm)  # never smooth a needed correction away
    return np.where(big, held, sm)


class SafetyStage:
    def __init__(self, robot, collision=None):
        self.r = robot
        self.col = collision or ArmCollision(robot)
        mo = robot.motion
        self.min_margin = mo["balance"]["min_margin_mm"]
        lc = mo["leg_cospeech"]
        self.leg = np.array([g == "leg" for g in robot.group])
        self.leg_lim = np.radians(lc["limit_deg"])
        self.vmax = np.where(self.leg, np.minimum(robot.vmax, np.radians(lc["vmax_dps"])), robot.vmax)
        self.amax = np.where(self.leg, np.minimum(robot.amax, np.radians(lc["amax_dps2"])), robot.amax)

    # ---- full trajectory (planner output) -----------------------------------------------------------
    def run(self, A, fps=50, q_start=None):
        """Anatomical trajectory A (N, 8) deg -> safe model trajectory Q (N, 15) rad, report."""
        rep = {}
        dt = 1.0 / fps
        A = np.asarray(A, float)
        # 1. envelope
        Ac = np.clip(A, self.r.anat_lo, self.r.anat_hi)
        rep["envelope_frames"] = int(np.any(np.abs(Ac - A) > 0.5, axis=1).sum())
        # 2. collision projection (both arms, every frame)
        Ac, rep["collision_projected_frames"] = self._project(Ac, fps)
        Q = self._model_limits(self.r.anat_to_q(Ac))
        # 3. balance
        Q, rep["balance_shrunk_frames"] = self._balance(Q, fps)
        # 4. rate limits
        Q2 = self._rate(Q, dt, q_start)
        rep["rate_dev_deg"] = round(float(np.degrees(np.abs(Q2 - Q).max())), 2)
        # 5. verify; repair where the rate limiter cut a corner into the body (up to 3 rounds)
        for rnd in range(3):
            bad = self._collision_frames(Q2)
            if not bad.any():
                break
            A2, _ = self._project(self.r.q_to_anat(Q2), fps, only=bad)
            Q2 = self._rate(self._model_limits(self.r.anat_to_q(A2)), dt, q_start)
        rep["repair_rounds"] = rnd
        rep.update(self.check(Q2, fps))
        return Q2, rep

    def _project(self, A, fps, only=None):
        """Check every frame (cheap), search the nearest clear pose at <= 10 Hz inside each colliding span
        (expensive) and hold that correction across the span; _hold_and_smooth blends it in and out."""
        Q = self.r.anat_to_q(A)
        bad = {s: np.zeros(len(A), bool) for s in "lr"}
        for k in range(len(A)):
            if only is not None and not only[k]:
                continue
            d = self.col.min_dist(Q[k])
            for s in "lr":
                bad[s][k] = d[s] < self.col.clearance
        delta = np.zeros_like(A)
        stride = max(1, fps // 10)
        for s in "lr":
            ks = np.flatnonzero(bad[s])
            if not len(ks):
                continue
            # project at every stride-th colliding frame plus the last one of each span
            span_end = np.append(ks[1:] != ks[:-1] + 1, True)
            probe = sorted(set(ks[::stride]) | set(ks[span_end]))
            fixes = {}
            for k in probe:
                B = self.col.project_arm(A[k], s)
                if B is not None:
                    fixes[k] = B - A[k]
            if not fixes:
                continue
            fk = np.array(sorted(fixes))
            for k in ks:  # each colliding frame takes the larger of its neighbouring probe corrections
                i = np.searchsorted(fk, k)
                cand = [fixes[fk[j]] for j in (i - 1, i) if 0 <= j < len(fk)]
                # each side's fix only touches its own arm columns, so left and right corrections add
                delta[k] += np.where(np.abs(cand[0]) >= np.abs(cand[-1]), cand[0], cand[-1])
        n_fixed = int((bad["l"] | bad["r"]).sum())
        if n_fixed:
            arm_cols = [ANAT.index(f"{s}_{j}") for s in "lr" for j in ("pitch", "roll", "elbow")]
            delta[:, arm_cols] = _hold_and_smooth(delta[:, arm_cols], fps)
        return A + delta, n_fixed

    def _rate(self, Q, dt, q_start):
        # the limiter's final forward pass can overshoot a target that sits on a joint limit by a hair: clamp
        # (clamping only shortens steps, so the speed/acceleration limits still hold)
        return np.clip(lookahead_limit(Q, self.vmax, self.amax, dt, q_start), self.r.lower, self.r.upper)

    def _model_limits(self, Q):
        Q = np.clip(Q, self.r.lower, self.r.upper)
        home = self.r.home
        Q[:, self.leg] = np.clip(Q[:, self.leg], home[self.leg] - self.leg_lim, home[self.leg] + self.leg_lim)
        return Q

    def _balance(self, Q, fps):
        step = max(1, fps // 10)
        idx = np.arange(0, len(Q), step)
        scale = np.ones(len(idx))
        home = self.r.home
        for _ in range(8):
            _, margin = self.r.com_margin(home + (Q[idx] - home) * scale[:, None])
            bad = margin < self.min_margin
            if not bad.any():
                break
            scale[bad] *= 0.85
        s_full = np.interp(np.arange(len(Q)), idx, scale)
        k = np.hanning(max(3, int(0.4 * fps)))
        s_full = np.minimum(s_full, np.convolve(np.pad(s_full, len(k), mode="edge"), k / k.sum(), "same")[len(k):-len(k)])
        return home + (Q - home) * s_full[:, None], int((scale < 0.999).sum() * step)

    def _collision_frames(self, Q, stride=1):
        bad = np.zeros(len(Q), bool)
        for k in range(0, len(Q), stride):
            bad[k] = not self.col.ok(Q[k])
        return bad

    # ---- checks only ------------------------------------------------------------------------------
    def check(self, Q, fps=50):
        """Report for any model trajectory (planner output, manual poses, RL policy rollouts)."""
        Q = np.atleast_2d(Q)
        dist = self.col.frame_dists(Q)
        step = max(1, fps // 10)
        _, margin = self.r.com_margin(Q[::step])
        dt = 1.0 / fps
        v = np.abs(np.diff(Q, axis=0)) / dt if len(Q) > 1 else np.zeros((1, self.r.n))
        within = (Q >= self.r.lower - 1e-6) & (Q <= self.r.upper + 1e-6)
        rep = {
            "frames": len(Q),
            "min_clearance_mm": round(float(dist.min()), 1),
            "collision_frames": int((dist.min(1) < self.col.clearance * 1000 - 0.05).sum()),
            "min_balance_margin_mm": round(float(margin.min()), 1),
            "balance_frames_below": int((margin < self.min_margin).sum() * step),
            "max_speed_ratio": round(float((v / self.vmax).max()), 2),
            "joint_limit_frames": int((~within).any(1).sum()),
        }
        rep["pass"] = (rep["collision_frames"] == 0 and rep["balance_frames_below"] == 0
                       and rep["max_speed_ratio"] <= 1.05 and rep["joint_limit_frames"] == 0)
        return rep

    # ---- single pose (Pose panel, live sliders) -------------------------------------------------------
    def safe_pose(self, q):
        """Model pose -> nearest safe pose (limits + arm collision projection) and a short report."""
        q = self._model_limits(np.atleast_2d(np.asarray(q, float)))[0]
        A = self.r.q_to_anat(q)[0]
        moved = False
        d = self.col.min_dist(q)
        for s in "lr":
            if d[s] < self.col.clearance:
                cols = [ANAT.index(f"{s}_{j}") for j in ("pitch", "roll", "elbow")]
                lo, hi = self.r.anat_lo[cols], self.r.anat_hi[cols]
                A[cols] = np.clip(A[cols], lo, hi)
                B = self.col.project_arm(A, s)
                if B is not None:
                    A, moved = B, True
        if moved:
            q2 = self.r.anat_to_q(A)[0]
            q2[~self._arm_mask()] = q[~self._arm_mask()]
            q = q2
        _, margin = self.r.com_margin(q)
        d = self.col.min_dist(q)
        return q, {"projected": moved, "clearance_mm": round(min(d["l"], d["r"], 1.0) * 1000, 1),
                   "balance_margin_mm": round(float(margin[0]), 1)}

    def _arm_mask(self):
        return np.array([g == "arm" for g in self.r.group])
