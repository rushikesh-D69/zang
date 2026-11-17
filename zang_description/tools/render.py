"""Render test poses (normal colours and one colour per link) for visual checks: render.py OUT_DIR"""
import os, sys, colorsys, numpy as np, mujoco
from PIL import Image

here = os.path.dirname(os.path.abspath(__file__))
m = mujoco.MjModel.from_xml_path(os.path.join(here, "..", "mjcf", "scene.xml"))
d = mujoco.MjData(m)
r = mujoco.Renderer(m, 640, 720)
out = sys.argv[1]
os.makedirs(out, exist_ok=True)

poses = {
    "zero": {},
    "legs": {"left_hip_pitch": -0.6, "left_knee_pitch": -0.6, "right_hip_pitch": 0.4, "right_knee_pitch": 0.0,
             "right_hip_roll": 0.25, "right_ankle_roll": -0.25},
    "arms": {"waist_yaw": 0.5, "left_shoulder_pitch": -1.0, "right_shoulder_pitch": 0.6,
             "left_shoulder_roll": -0.6, "left_elbow": -0.8, "right_elbow": 0.8},
}
views = {"front": (180, -10, 0.2, 0.75), "side": (90, -10, 0.2, 0.75), "iso": (135, -25, 0.2, 0.75),
         "legs_side": (90, -5, 0.1, 0.45), "legs_iso": (140, -20, 0.1, 0.45)}

vis = np.where(m.geom_group == 2)[0]
orig = m.geom_rgba.copy()
link_rgba = orig.copy()
for k, g in enumerate(vis):
    b = m.geom_bodyid[g]
    link_rgba[g, :3] = colorsys.hsv_to_rgb((b * 0.618) % 1.0, 0.65, 0.95)

for colours, tag in ((orig, ""), (link_rgba, "_links")):
    m.geom_rgba[:] = colours
    for pn, pose in poses.items():
        d.qpos[:] = m.qpos0
        for j, v in pose.items():
            d.qpos[m.jnt_qposadr[m.joint(j).id]] = v
        for e in range(m.neq):  # passive linkage pins follow their servo
            d.qpos[m.jnt_qposadr[m.eq_obj1id[e]]] = m.eq_data[e][1] * d.qpos[m.jnt_qposadr[m.eq_obj2id[e]]]
        mujoco.mj_forward(m, d)
        imgs = []
        for az, el, z, dist in views.values():
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [0, 0, z]
            cam.distance, cam.azimuth, cam.elevation = dist, az, el
            r.update_scene(d, cam)
            imgs.append(r.render())
        Image.fromarray(np.hstack(imgs)).save(os.path.join(out, f"{pn}{tag}.png"))
print("ok")
