"""Draw the Zang kinematic diagrams into figures/ from the live model (URDF + MJCF loop sites).

    ../.venv/bin/python make_figures.py

fig_topology.png       kinematic graph: root body, spanning-tree / active / passive / loop-closure joints,
                       parallelogram submechanisms, and the equivalent serial chain
fig_leg_mechanism.png  side view of the left leg (two parallelograms) at HOME, deep squat and leg-forward
fig_skeleton.png       whole-robot joint skeleton at HOME, front and side views
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Arc

from zang_ik import HOME, SIDES, Zang

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
MJCF = os.path.join(HERE, "..", "zang_description", "mjcf", "scene.xml")
ACTIVE, PASSIVE, LOOP, BODY = "#d62728", "#2ca02c", "#1f77b4", "#444444"


# ---------------------------------------------------------------- 1. topology graph
def fig_topology(z):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(18, 12.5), gridspec_kw={"width_ratios": [3.2, 1]})
    pos = {"pelvis": (0, 0), "chest": (0, 1.6)}
    for s, sx in (("left", 1), ("right", -1)):  # robot left drawn on the figure's right (front view)
        pos.update({f"{s}_shoulder": (1.9 * sx, 2.9), f"{s}_upper_arm": (3.4 * sx, 2.9), f"{s}_hand": (4.9 * sx, 2.9)})
        xc = 2.7 * sx
        pos.update({f"{s}_hip": (xc, -1.3), f"{s}_thigh": (xc - 0.95, -2.65), f"{s}_thigh_shell": (xc + 0.95, -2.65),
                    f"{s}_knee": (xc, -4.0), f"{s}_shin": (xc - 0.95, -5.35), f"{s}_shin_shell": (xc + 0.95, -5.35),
                    f"{s}_ankle": (xc, -6.7), f"{s}_foot": (xc, -8.0)})
    # submechanism outlines
    for s, sx in (("left", 1), ("right", -1)):
        xc = 2.7 * sx
        for yc, lab in ((-2.65, "P1: 4R parallelogram\n(driven by hip_pitch)"), (-5.35, "P2: 4R parallelogram\n(driven by knee_pitch)")):
            ax.add_patch(Ellipse((xc, yc), 3.15, 3.55, fill=False, ls="--", lw=1.4, color="#9467bd"))
            ax.text(xc + 1.65 * sx, yc, lab, color="#9467bd", fontsize=8.5, ha="left" if sx > 0 else "right", va="center")
        ax.add_patch(Ellipse(((pos[f"{s}_hand"][0] + pos[f"{s}_shoulder"][0]) / 2, 2.9), 4.1, 1.25, fill=False, ls="--", lw=1.2, color="#ff7f0e"))
        ax.text((pos[f"{s}_hand"][0] + pos[f"{s}_shoulder"][0]) / 2, 3.65, "serial R-RR arm", color="#ff7f0e", fontsize=8.5, ha="center")

    def edge(a, b, color, ls="-", label=None, rad=0.0):
        pa, pb = np.array(pos[a]), np.array(pos[b])
        ax.add_patch(FancyArrowPatch(pa, pb, arrowstyle="-|>", mutation_scale=14, color=color, lw=2 if ls == "-" else 1.6,
                                     ls=ls, shrinkA=24, shrinkB=24, connectionstyle=f"arc3,rad={rad}"))
        if label:
            m = (pa + pb) / 2 + (np.array([0, 0.3]) if abs(pa[1] - pb[1]) < 0.1 else 0)  # horizontal edges: label above
            ax.text(m[0], m[1], label, fontsize=6.5, color=color, ha="center", va="center",
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))

    for n, j in z.joints.items():
        short = n.replace("left_", "").replace("right_", "")
        edge(j["parent"], j["child"], PASSIVE if j["mimic"] else ACTIVE,
             label=f"{short}\n= {'+' if j['mimic'][1] > 0 else '-'}{j['mimic'][0].split('_', 1)[1]}" if j["mimic"] else short)
    for s in SIDES:
        edge(f"{s}_thigh_shell", f"{s}_knee", LOOP, ls="--", label="loop pin K2")
        edge(f"{s}_shin_shell", f"{s}_knee", LOOP, ls="--", label="loop pin K4")
    for n, (x, y) in pos.items():
        ax.add_patch(plt.Circle((x, y), 0.4, fc="white", ec=BODY, lw=1.6, zorder=3))
        if n == "pelvis":
            ax.add_patch(plt.Circle((x, y), 0.47, fc="none", ec=BODY, lw=1.6, zorder=3))
            ax.add_patch(plt.Circle((x, y + 0.2), 0.06, fc="black", zorder=4))
        label = n.replace("left_", "L\n").replace("right_", "R\n").replace("_", "\n")
        ax.text(x, y - (0.06 if n == "pelvis" else 0), label, fontsize=7, ha="center", va="center", zorder=5, linespacing=0.95)
    ax.set_xlim(-6.6, 6.6)
    ax.set_ylim(-8.7, 4.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Zang kinematic graph: series-parallel hybrid (24 bodies, 15 active, 8 passive, 4 loop pins)", fontsize=12)
    handles = [plt.Line2D([], [], color=ACTIVE, lw=2, label="active joint (servo) u = y"),
               plt.Line2D([], [], color=PASSIVE, lw=2, label="passive spanning-tree pin q = γ(y)"),
               plt.Line2D([], [], color=LOOP, lw=1.6, ls="--", label="loop / cut joint (closure)"),
               plt.Line2D([], [], marker="o", color="w", markeredgecolor=BODY, markersize=11, label="body (link)"),
               plt.Line2D([], [], marker="o", color="w", markerfacecolor="black", markeredgecolor=BODY, markersize=11, label="root body (floating base)"),
               plt.Line2D([], [], color="#9467bd", ls="--", label="parallel submechanism")]
    ax.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.04))

    # equivalent serial chain of one leg (submechanisms collapsed), like a reduced graph
    chain = [("pelvis", "root"), ("hip_roll", "R  (servo)"), ("P1 hip_pitch", "parallelogram\nthigh swings,\nknee stays level"),
             ("P2 knee_pitch", "parallelogram\nshin swings,\nankle stays level"), ("ankle_roll", "R  (servo)"), ("foot", "sole")]
    for k, (name, desc) in enumerate(chain):
        y = 5 - 2.0 * k
        box = k in (2, 3)
        ax2.add_patch(FancyBboxPatch((-0.9, y - 0.45), 1.8, 0.9, boxstyle="round,pad=0.05",
                                     fc="#efe6f7" if box else "white", ec="#9467bd" if box else BODY, lw=1.5))
        ax2.text(0, y, name, ha="center", va="center", fontsize=9, weight="bold")
        ax2.text(1.05, y, desc, ha="left", va="center", fontsize=7.5)
        if k:
            ax2.add_patch(FancyArrowPatch((0, y + 1.55), (0, y + 0.5), arrowstyle="-|>", mutation_scale=12, color=BODY))
    ax2.text(0, -6.4, "per leg: y = (hip_roll, hip_pitch, knee_pitch, ankle_roll)\n"
             "foot orientation = pelvis * Rx(hip_roll + ankle_roll)\n"
             "4 DOF: sole position (3) + foot roll (1)", ha="center", fontsize=8)
    ax2.set_xlim(-1.5, 3.2)
    ax2.set_ylim(-7.2, 6)
    ax2.axis("off")
    ax2.set_title("equivalent serial leg chain", fontsize=11)
    fig.savefig(os.path.join(OUT, "fig_topology.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 2. leg mechanism side view
def leg_points(m, d, side):
    """Pins in the pelvis frame, as (x forward, z up) mm, from the MJCF (loop pins from the closure sites)."""
    Rp, pp = d.xmat[m.body("pelvis").id].reshape(3, 3), d.xpos[m.body("pelvis").id]
    loc = lambda p: (Rp.T @ (p - pp))[[0, 2]] * 1000
    body = lambda n: loc(d.xpos[m.body(f"{side}_{n}").id])
    site = lambda n: loc(d.site(f"{side}_{n}").xpos)
    return {"H1": body("thigh"), "H2": body("thigh_shell"), "K1": body("knee"), "K2": site("upper_loop_b"),
            "K3": body("shin"), "K4": site("lower_loop_b"), "A1": body("ankle"), "A2": body("shin_shell"),
            "R": body("foot"), "HR": body("hip")}


def fig_leg(z):
    m = mujoco.MjModel.from_xml_path(MJCF)
    d = mujoco.MjData(m)
    poses = [("HOME  (hip_pitch = knee_pitch = -0.5)", {}),
             ("deep squat  (-0.9, -0.9)", {"left_hip_pitch": -0.9, "left_knee_pitch": -0.9}),
             ("leg forward  (-0.9, -0.2)", {"left_hip_pitch": -0.9, "left_knee_pitch": -0.2})]
    fig, axes = plt.subplots(1, 3, figsize=(17, 8.5))
    sole = z.sole["left"] * 1000
    for ax, (title, extra) in zip(axes, poses):
        q = z.full_q({**HOME, **extra})
        d.qpos[:] = 0
        d.qpos[3] = 1
        for n, v in q.items():
            d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
        mujoco.mj_kinematics(m, d)
        P = leg_points(m, d, "left")
        seg = lambda a, b, c, lw, ls="-": ax.plot(*np.array([P[a], P[b]]).T, color=c, lw=lw, ls=ls, solid_capstyle="round", zorder=2)
        ax.fill(*np.array([P["H1"], P["H2"], P["HR"]]).T, color="#cccccc", zorder=1)
        seg("H1", "H2", "#555555", 6)
        seg("A1", "A2", "#555555", 6)
        ax.fill(*np.array([P["K1"], P["K2"], P["K4"], P["K3"]]).T, color="#cfe8cf", ec=PASSIVE, lw=2, zorder=1)
        seg("H1", "K1", ACTIVE, 5)
        seg("K3", "A1", ACTIVE, 5)
        seg("H2", "K2", LOOP, 4)
        seg("K4", "A2", LOOP, 4)
        seg("A1", "R", "#555555", 3, ":")
        foot_z = P["R"][1] + sole[2]
        ax.plot([P["R"][0] - 65, P["R"][0] + 65], [foot_z, foot_z], color="#333333", lw=5, solid_capstyle="butt")
        ax.plot([-90, 90], [foot_z - 4, foot_z - 4], color="#999999", lw=1)
        for k, p in P.items():
            if k in ("R", "HR"):
                continue
            servo = k in ("H1", "A1")
            ax.add_patch(plt.Circle(p, 4.2 if servo else 3.0, fc=ACTIVE if servo else "white", ec="black", lw=1.2, zorder=4))
            ax.text(p[0] + (-9 if k in ("H2", "K2", "K4", "A2") else 6), p[1] + 3, k, fontsize=9, zorder=5,
                    ha="right" if k in ("H2", "K2", "K4", "A2") else "left")
        ax.text(P["HR"][0], P["HR"][1] + 6, "hip roll axis (x)", fontsize=8, ha="center")
        ax.add_patch(plt.Circle(P["HR"], 3, fc="#555555", zorder=4))
        ax.text(P["R"][0] + 8, P["R"][1], "ankle roll\naxis (x)", fontsize=8, va="center")
        ax.add_patch(plt.Circle(P["R"], 3, fc="#555555", zorder=4))
        for a, b, lab in (("H1", "K1", "q1 = hip_pitch"), ("A1", "K3", "q2 = knee_pitch")):
            v = P[b] - P[a]
            ang = np.degrees(np.arctan2(v[1], v[0]))
            base = -90 if a == "H1" else 90
            ax.add_patch(Arc(P[a], 34, 34, theta1=min(base, ang), theta2=max(base, ang), color=ACTIVE, lw=1.5, zorder=3))
            ax.text(P[a][0] + 22, P[a][1] + (-14 if a == "H1" else 14), lab, color=ACTIVE, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlim(-110, 120)
        ax.set_ylim(foot_z - 15, 30)
        ax.set_xlabel("x forward [mm, pelvis frame]")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("z up [mm, pelvis frame]")
    handles = [plt.Line2D([], [], color=ACTIVE, lw=5, label="servo-driven crank (thigh H1-K1, shin K3-A1)"),
               plt.Line2D([], [], color=LOOP, lw=4, label="outer shell (H2-K2, K4-A2), pinned both ends"),
               plt.Line2D([], [], color=PASSIVE, lw=2, label="knee coupler K1-K2-K4-K3 (translates, never rotates)"),
               plt.Line2D([], [], color="#555555", lw=6, label="hip / ankle brackets (H1-H2, A1-A2)"),
               plt.Line2D([], [], marker="o", color="w", markerfacecolor=ACTIVE, markeredgecolor="k", markersize=10, label="servo axis"),
               plt.Line2D([], [], marker="o", color="w", markerfacecolor="white", markeredgecolor="k", markersize=9, label="passive pin")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Zang left leg, side view: hip roll - parallelogram P1 - parallelogram P2 - ankle roll\n"
                 "cranks |H1K1| = |K3A1| = 60.4 mm, shells |H2K2| = |K4A2| = 60.0 mm (CAD), knee |K1K3| = 21.3 mm  ->  foot stays parallel to the pelvis",
                 fontsize=12)
    fig.subplots_adjust(bottom=0.14, top=0.86, wspace=0.12)
    fig.savefig(os.path.join(OUT, "fig_leg_mechanism.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 3. skeleton
def fig_skeleton(z):
    F = z.fk(z.home())
    fig, axes = plt.subplots(1, 2, figsize=(13, 9))
    for ax, (ix, xlabel, title) in zip(axes, ((1, "y left [mm]", "front view (looking along -x)"), (0, "x forward [mm]", "side view (from the right)"))):
        for n, j in z.joints.items():
            a, b = F[j["parent"]][:3, 3] * 1000, F[j["child"]][:3, 3] * 1000
            ax.plot([a[ix], b[ix]], [a[2], b[2]], color="#999999" if j["mimic"] else "#333333", lw=1.2 if j["mimic"] else 2, zorder=1)
        for n, j in z.joints.items():
            p = F[j["child"]][:3, 3] * 1000
            if j["mimic"]:
                ax.scatter(p[ix], p[2], s=22, c="white", edgecolors=PASSIVE, zorder=3)
                continue
            ax.scatter(p[ix], p[2], s=45, c=ACTIVE, zorder=3)
            axis = F[j["parent"]][:3, :3] @ j["axis"]
            u = np.array([axis[ix], axis[2]])
            if np.linalg.norm(u) > 0.2:
                ax.annotate("", xy=(p[ix] + 16 * u[0], p[2] + 16 * u[1]), xytext=(p[ix], p[2]),
                            arrowprops=dict(arrowstyle="-|>", color=ACTIVE, lw=1))
            else:
                ax.scatter(p[ix], p[2], s=110, facecolors="none", edgecolors=ACTIVE, zorder=3)
            if ix == 1 and (n.startswith("left") or n == "waist_yaw"):
                ax.text(p[ix] + 4, p[2] + 3, n.replace("left_", "L "), fontsize=7)
            if ix == 0 and (n.startswith("left") or n == "waist_yaw"):
                ax.text(p[ix] + 4, p[2] + 3, n.replace("left_", ""), fontsize=7)
        for s in SIDES:
            for pt in (z.sole_pos(F, s), z.hand_pos(F, s)):
                ax.scatter(pt[ix] * 1000, pt[2] * 1000, marker="s", s=30, c="#ff7f0e", zorder=3)
        c = z.com(F) * 1000
        ax.scatter(c[ix], c[2], marker="X", s=120, c="black", zorder=4)
        ax.text(c[ix] + 5, c[2], "CoM", fontsize=8)
        ax.scatter(0, 0, marker="*", s=180, c="#9467bd", zorder=4)
        ax.text(5, -10, "pelvis (root)", fontsize=8, color="#9467bd")
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
    axes[0].set_ylabel("z up [mm, pelvis frame]")
    handles = [plt.Line2D([], [], marker="o", color="w", markerfacecolor=ACTIVE, markersize=8, label="servo joint (arrow = +axis, ring = axis out of page)"),
               plt.Line2D([], [], marker="o", color="w", markerfacecolor="white", markeredgecolor=PASSIVE, markersize=7, label="passive linkage pin"),
               plt.Line2D([], [], marker="s", color="w", markerfacecolor="#ff7f0e", markersize=7, label="IK end points: sole centre, hand tip"),
               plt.Line2D([], [], marker="X", color="w", markerfacecolor="black", markersize=9, label="centre of mass")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.suptitle("Zang joint skeleton at HOME (frames: x forward, y left, z up)", fontsize=12)
    fig.subplots_adjust(bottom=0.12)
    fig.savefig(os.path.join(OUT, "fig_skeleton.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    z = Zang()
    fig_topology(z)
    fig_leg(z)
    fig_skeleton(z)
    print("wrote", ", ".join(sorted(os.listdir(OUT))))
