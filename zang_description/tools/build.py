"""Build URDF + MJCF for Zang from the flattened STEP (build/parts.pkl, build/holes.pkl).

Joints are not in the CAD, so they are defined here: each servo is a revolute joint
between its case (stator) and its horns (rotor). Links are the rigid groups left after
cutting those servo interfaces out of the part-contact graph.

Each half-leg is a parallelogram linkage (servo-driven inner plates + pinned outer shell,
both pinned to the knee), so the knee translates without rotating and the foot stays
parallel to the pelvis. Passive pins are located from coaxial holes and follow their
servo (URDF mimic / MJCF joint equality); the shell-to-knee pins close the loops.
"""
import os, re, pickle, collections, shutil, itertools
import numpy as np
import trimesh
import fast_simplification

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------- joint definitions
# servo instance path in the STEP -> (joint name, child link name)
SERVOS = {
    "DS-R003G:3": ("waist_yaw", "chest"),
    "DS-R003G:2": ("left_hip_roll", "left_hip"),
    "DS-R003G:7": ("left_hip_pitch", "left_thigh"),
    "DS-R003G:8": ("left_knee_pitch", "left_ankle"),
    "DS-R003G:10": ("left_ankle_roll", "left_foot"),
    "DS-R003G:1": ("right_hip_roll", "right_hip"),
    "DS-R003G:6": ("right_hip_pitch", "right_thigh"),
    "DS-R003G:9": ("right_knee_pitch", "right_ankle"),
    "DS-R003G:11": ("right_ankle_roll", "right_foot"),
    "Chest Assembly:1/DS-R003G w/ Metal Horns:2": ("left_shoulder_pitch", "left_shoulder"),
    "DS-R003G:18": ("left_shoulder_roll", "left_upper_arm"),
    "DS-R003G:17": ("left_elbow", "left_hand"),
    "Chest Assembly:1/DS-R003G w/ Metal Horns:1": ("right_shoulder_pitch", "right_shoulder"),
    "DS-R003G:19": ("right_shoulder_roll", "right_upper_arm"),
    "DS-R003G:12": ("right_elbow", "right_hand"),
}
ROOT_PART, ROOT_LINK = "Torso:1", "pelvis"
OUTPUT_HORN = re.compile(r"^(Servo Horn|Metal Teethed Horn|Shoulder Metal Horn)")
IDLER_HORN = re.compile(r"^(Metal Horn|Metal Dummy Horn)")
EXTRA_ROTOR = {"DS-R003G:3": ["Shoulder Metal Horn:1"]}  # waist horn sits outside the servo sub-assembly

# passive leg-linkage pins: (joint, parent-side part, child-side part, child link, driving servo joint, multiplier)
# multipliers assume ideal parallelograms: knee keeps the hip bracket's orientation, ankle keeps the knee's
PINS = [
    ("left_knee_upper", "Leg Link:1", "Knee:1", "left_knee", "left_hip_pitch", -1),
    ("left_knee_lower", "Knee:1", "Leg Link:5", "left_shin", "left_knee_pitch", -1),
    ("left_thigh_shell", "Hip Link 2:1", "Thigh:1", "left_thigh_shell", "left_hip_pitch", 1),
    ("left_shin_shell", "Hip Link 1:2", "Thigh:3", "left_shin_shell", "left_knee_pitch", -1),
    ("right_knee_upper", "Leg Link:3", "Knee:2", "right_knee", "right_hip_pitch", -1),
    ("right_knee_lower", "Knee:2", "Leg Link:7", "right_shin", "right_knee_pitch", -1),
    ("right_thigh_shell", "Hip Link 1:1", "Thigh:2", "right_thigh_shell", "right_hip_pitch", 1),
    ("right_shin_shell", "Hip Link 2:2", "Thigh:4", "right_shin_shell", "right_knee_pitch", -1),
]
LOOP_CLOSURES = [  # (name, part, part): pins left out of the tree; marked with sites in the MJCF
    ("left_upper_loop", "Thigh:1", "Knee:1"),
    ("left_lower_loop", "Thigh:3", "Knee:1"),
    ("right_upper_loop", "Thigh:2", "Knee:2"),
    ("right_lower_loop", "Thigh:4", "Knee:2"),
]
LINKAGE_PARTS = ["Knee:1", "Knee:2", "Thigh:1", "Thigh:2", "Thigh:3", "Thigh:4"]  # pinned only, never fastened
COLLISION_FREE_GROUPS = [  # links of one leg linkage nest inside each other's convex hulls
    [f"{s}_{n}" for n in ("hip", "thigh", "thigh_shell", "knee", "shin", "shin_shell", "ankle")] for s in ("left", "right")
]

# contact-graph corrections (CAD has 0.5 mm print clearances, so touching != fastened everywhere)
DROP = ["Metal Horn:1"]  # loose horn floating above the head
EXTRA_CUTS = [  # waist turntable: pelvis side vs chest side
    ("RotMech Acrylic Piece:1", "RotMech Hip Part:1"),
    ("Torso:1", "Shoulder Metal Horn:1"),
    ("DS-R003G:3/body", "speaker:1"),
]
EXTRA_UNIONS = [  # idler horns ride with the output horn
    ("Chest Assembly:1/DS-R003G w/ Metal Horns:1/Metal Dummy Horn:1",
     "Chest Assembly:1/DS-R003G w/ Metal Horns:1/Metal Teethed Horn:1"),
    ("Chest Assembly:1/DS-R003G w/ Metal Horns:2/Metal Dummy Horn:1",
     "Chest Assembly:1/DS-R003G w/ Metal Horns:2/Metal Teethed Horn:1"),
]

# ---------------------------------------------------------------- physical placeholders (edit me)
SERVO_BODY_MASS = 0.060  # kg per DS-R003G case incl. motor/gears (placeholder - weigh one)
DENSITY = [  # (regex on leaf part name, g/cm^3), first match wins
    (r"Horn", 2.70),                        # aluminium horns
    (r"Screw|Standoff|Zinc|9518", 7.85),    # steel fasteners
    (r"Header", 2.0),
    (r"Acrylic", 1.18),
    (r"[Ss]peaker", 1.5),
    (r"Leg Link", 2.70),                    # 2 mm plates, assumed aluminium
    (r".", 1.00),                           # 3D-printed parts (PLA ~1.24 solid, reduced for infill)
]
JOINT_LIMIT = np.pi / 2   # rad, symmetric placeholder
EFFORT = 2.0              # N*m
VELOCITY = 6.0            # rad/s
KP, DAMPING, ARMATURE = 8.0, 0.05, 0.005
PASSIVE_DAMPING = 0.005   # linkage pins (bolt friction)
# home / rest pose (rad): knees bent forward like a real biped at rest; keep in sync with IK/zang_ik.py HOME
HOME = {f"{s}_{j}": -0.5 for s in ("left", "right") for j in ("hip_pitch", "knee_pitch")}

DEFAULT_RGB = (0.85, 0.85, 0.85)
VISUAL_FACES_PER_MESH = 12000
MIN_COLLISION_VOLUME = 300.0  # mm^3; smaller parts get no collision hull

# CAD frame: -Y forward, +X left, +Z up, mm  ->  robot frame (REP-103): +X forward, +Y left, +Z up, m
R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
to_robot = lambda p_mm: (R @ np.asarray(p_mm, float).T).T * 1e-3


def main():
    data = pickle.load(open(os.path.join(ROOT, "build/parts.pkl"), "rb"))
    P, E = data["parts"], data["edges"]
    name = ["/".join(p["path"][1:]) for p in P]
    idx = {n: i for i, n in enumerate(name)}
    keep = [i for i in range(len(P)) if name[i] not in DROP]

    # servo rotor / stator membership
    servo = {}
    for key in SERVOS:
        rot, sta = set(), set()
        for i in keep:
            if name[i].startswith(key + "/"):
                (rot if IDLER_HORN.match(P[i]["path"][-1]) or OUTPUT_HORN.match(P[i]["path"][-1]) else sta).add(i)
        rot |= {idx[n] for n in EXTRA_ROTOR.get(key, [])}
        servo[key] = (rot, sta)

    cut = {frozenset((idx[a], idx[b])) for a, b in EXTRA_CUTS}
    for rot, sta in servo.values():
        cut |= {frozenset((i, j)) for i, j, *_ in E if (i in rot and j in sta) or (j in rot and i in sta)}
    linkage = {idx[n] for n in LINKAGE_PARTS}
    cut |= {frozenset((i, j)) for i, j, *_ in E if i in linkage or j in linkage}

    parent = {i: i for i in keep}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        parent[find(a)] = find(b)
    for i, j, *_ in E:
        if i in parent and j in parent and frozenset((i, j)) not in cut:
            union(i, j)
    for a, b in EXTRA_UNIONS:
        union(idx[a], idx[b])
    groups = collections.defaultdict(list)
    for i in keep:
        groups[find(i)].append(i)

    # joint geometry in CAD mm: servo axes from horns, pin axes from coaxial holes
    holes = pickle.load(open(os.path.join(ROOT, "build/holes.pkl"), "rb"))
    edges = []
    for key, (rot, sta) in servo.items():
        gr, gs = {find(i) for i in rot}, {find(i) for i in sta}
        assert len(gr) == 1 and len(gs) == 1 and gr != gs, f"{key}: rotor/stator not separable"
        out = [i for i in rot if OUTPUT_HORN.match(P[i]["path"][-1])][0]
        idl = [i for i in rot if IDLER_HORN.match(P[i]["path"][-1])]
        point = P[out]["bbox"].mean(0)
        if idl:
            ax = P[idl[0]]["bbox"].mean(0) - point
        else:
            w, v = np.linalg.eigh(P[out]["inertia"])
            ax = v[:, np.argmax(w)]
        jname, cname = SERVOS[key]
        edges.append(dict(name=jname, a=gr.pop(), b=gs.pop(), child_name=cname, point=point, axis=ax, mimic=None))
    for jname, pa, pb, cname, driver, mult in PINS:
        point, ax = pin_axis(holes, pa, pb, leg_mid_x(P[idx[pa]], P[idx[pb]]))
        edges.append(dict(name=jname, a=find(idx[pa]), b=find(idx[pb]), child_name=cname, point=point, axis=ax,
                          mimic=(driver, mult), parent_side=find(idx[pa])))
    closures = [(n, find(idx[a]), find(idx[b]), pin_axis(holes, a, b, leg_mid_x(P[idx[a]], P[idx[b]]))[0]) for n, a, b in LOOP_CLOSURES]

    # kinematic tree (BFS from the pelvis)
    root = find(idx[ROOT_PART])
    link_of = {root: ROOT_LINK}
    origin = {root: to_robot(P[idx[ROOT_PART]]["centroid"])}
    joints, frontier, pending = [], [root], list(edges)
    while frontier:
        g = frontier.pop(0)
        for e in [e for e in pending if g in (e["a"], e["b"])]:
            child = e["b"] if g == e["a"] else e["a"]
            assert e.get("parent_side", g) == g, f"{e['name']}: pin reached from the wrong side"
            ax = R @ (e["axis"] / np.linalg.norm(e["axis"]))
            ax = ax if ax[np.argmax(np.abs(ax))] > 0 else -ax
            ax[np.abs(ax) < 0.01] = 0.0  # CAD placement noise
            link_of[child] = e["child_name"]
            origin[child] = to_robot(e["point"])
            joints.append(dict(name=e["name"], parent=g, child=child, axis=ax / np.linalg.norm(ax), mimic=e["mimic"]))
            frontier.append(child)
            pending.remove(e)
    unassigned = [g for g in groups if g not in link_of]
    assert not pending and not unassigned, ([e["name"] for e in pending], [[name[i] for i in groups[g]] for g in unassigned])

    # meshes + inertial per link
    mdir = os.path.join(ROOT, "meshes")
    shutil.rmtree(mdir, ignore_errors=True)
    os.makedirs(os.path.join(mdir, "visual"))
    os.makedirs(os.path.join(mdir, "collision"))
    links = {}
    for g, members in groups.items():
        ln = link_of[g]
        o = origin[g]
        M, msum, I = 0.0, np.zeros(3), np.zeros((3, 3))
        parts_mass = []
        for i in members:
            p = P[i]
            leaf = p["path"][-1]
            if name[i].endswith("/body") and "DS-R003G" in name[i]:
                rho = SERVO_BODY_MASS / (p["volume"] * 1e-9)  # kg/m^3
            else:
                rho = next(d for r, d in DENSITY if re.search(r, leaf)) * 1000.0
            m = rho * p["volume"] * 1e-9
            c = to_robot(p["centroid"]) - o
            Ic = R @ (p["inertia"] * rho * 1e-15) @ R.T  # mm^5 * kg/m^3 -> kg m^2
            parts_mass.append((m, c, Ic))
            M += m
            msum += m * c
        com = msum / M
        for m, c, Ic in parts_mass:
            d = c - com
            I += Ic + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

        by_color = collections.defaultdict(list)
        for i in members:
            col = tuple(np.round(P[i]["color"], 3)) if P[i]["color"] else DEFAULT_RGB
            by_color[col].append(i)
        visuals = []
        for k, (col, ids) in enumerate(sorted(by_color.items(), key=lambda kv: -len(kv[1]))):
            V = np.vstack([to_robot(P[i]["V"]) - o for i in ids])
            offs = np.cumsum([0] + [len(P[i]["V"]) for i in ids])
            F = np.vstack([P[i]["F"] + offs[n] for n, i in enumerate(ids)])
            mesh = trimesh.Trimesh(V, F)
            mesh.merge_vertices()
            if len(mesh.faces) > VISUAL_FACES_PER_MESH:
                v2, f2 = fast_simplification.simplify(mesh.vertices, mesh.faces, 1 - VISUAL_FACES_PER_MESH / len(mesh.faces))
                mesh = trimesh.Trimesh(v2, f2)
            fn = f"visual/{ln}_{k}.stl"
            mesh.export(os.path.join(mdir, fn))
            visuals.append((fn, col))
        collisions = []
        for k, i in enumerate(sorted(members, key=lambda i: -P[i]["volume"])):
            if P[i]["volume"] < MIN_COLLISION_VOLUME:
                continue
            hull = trimesh.Trimesh(to_robot(P[i]["V"]) - o, P[i]["F"]).convex_hull
            fn = f"collision/{ln}_{len(collisions)}.stl"
            hull.export(os.path.join(mdir, fn))
            collisions.append(fn)
        links[ln] = dict(mass=M, com=com, inertia=I, visuals=visuals, collisions=collisions,
                         parts=sorted(name[i] for i in members))

    for l in links.values():
        l["sites"] = []
    for n, ga, gb, point in closures:
        for g, tag in ((ga, "a"), (gb, "b")):
            links[link_of[g]]["sites"].append((f"{n}_{tag}", to_robot(point) - origin[g]))

    for j in joints:
        j["parent"], j["child"] = link_of[j["parent"]], link_of[j["child"]]
    org = {link_of[g]: origin[g] for g in origin}
    for j in joints:
        j["xyz"] = org[j["child"]] - org[j["parent"]]

    foot_min_z = min(P[i]["bbox"][0][2] for i in keep) * 1e-3
    base_height = org[ROOT_LINK][2] - foot_min_z

    write_urdf(links, joints)
    excludes = sorted({tuple(sorted(p)) for grp in COLLISION_FREE_GROUPS for p in itertools.combinations(grp, 2)})
    write_mjcf(links, joints, base_height, excludes)
    extra = [p for p in overlapping_links() if p not in excludes]
    excludes = sorted(excludes + extra)
    write_mjcf(links, joints, base_height, excludes)
    write_mjcf(links, joints, base_height, excludes, home_keyframe())
    write_report(links, joints, base_height)
    print("extra collision excludes (convex hulls overlap at zero pose):", extra)


def leg_mid_x(pa, pb):
    return (min(pa["bbox"][0][0], pb["bbox"][0][0]) + max(pa["bbox"][1][0], pb["bbox"][1][0])) / 2


def pin_axis(holes, a, b, x):
    """Common axis of coaxial cylindrical faces of parts a and b; returns (point at CAD x, direction)."""
    pts, d0 = [], None
    for ca in holes[a]:
        for cb in holes[b]:
            if abs(abs(ca["dir"] @ cb["dir"]) - 1) > 1e-3:
                continue
            d = ca["dir"] / np.linalg.norm(ca["dir"])
            off = cb["loc"] - ca["loc"]
            if np.linalg.norm(off - (off @ d) * d) > 0.5:
                continue
            d0 = d if d0 is None else d0
            for c in (ca, cb):
                pts.append(c["loc"] + (x - c["loc"][0]) / d[0] * d)
    assert pts, f"no coaxial holes between {a} and {b}"
    return np.mean(pts, axis=0), d0


f3 = lambda v: " ".join(f"{x:.6g}" for x in v)


def write_urdf(links, joints):
    L = ['<?xml version="1.0"?>', '<robot name="zang">']
    for ln, l in links.items():
        I = l["inertia"]
        L.append(f'  <link name="{ln}">')
        L.append(f'    <inertial><origin xyz="{f3(l["com"])}" rpy="0 0 0"/><mass value="{l["mass"]:.6g}"/>'
                 f'<inertia ixx="{I[0,0]:.6g}" ixy="{I[0,1]:.6g}" ixz="{I[0,2]:.6g}" iyy="{I[1,1]:.6g}" iyz="{I[1,2]:.6g}" izz="{I[2,2]:.6g}"/></inertial>')
        for fn, col in l["visuals"]:
            L.append(f'    <visual><geometry><mesh filename="meshes/{fn}"/></geometry>'
                     f'<material name="{ln}_{os.path.basename(fn)[:-4]}"><color rgba="{f3(col)} 1"/></material></visual>')
        for fn in l["collisions"]:
            L.append(f'    <collision><geometry><mesh filename="meshes/{fn}"/></geometry></collision>')
        L.append("  </link>")
    for j in joints:
        lim = np.pi if j["mimic"] else JOINT_LIMIT
        mimic = f'<mimic joint="{j["mimic"][0]}" multiplier="{j["mimic"][1]}" offset="0"/>' if j["mimic"] else ""
        L.append(f'  <joint name="{j["name"]}" type="revolute"><parent link="{j["parent"]}"/><child link="{j["child"]}"/>'
                 f'<origin xyz="{f3(j["xyz"])}" rpy="0 0 0"/><axis xyz="{f3(j["axis"])}"/>'
                 f'<limit lower="{-lim:.4f}" upper="{lim:.4f}" effort="{EFFORT}" velocity="{VELOCITY}"/>'
                 f'<dynamics damping="{PASSIVE_DAMPING if j["mimic"] else DAMPING}" friction="0.0"/>{mimic}</joint>')
    L.append("</robot>")
    open(os.path.join(ROOT, "zang.urdf"), "w").write("\n".join(L) + "\n")


def write_mjcf(links, joints, base_height, excludes=(), keyframe=None):
    children = collections.defaultdict(list)
    for j in joints:
        children[j["parent"]].append(j)
    assets = []
    for l in links.values():
        for fn, _ in l["visuals"]:
            assets.append(f'    <mesh name="{os.path.basename(fn)[:-4]}" file="../meshes/{fn}"/>')
        for fn in l["collisions"]:
            assets.append(f'    <mesh name="col_{os.path.basename(fn)[:-4]}" file="../meshes/{fn}"/>')

    def body(ln, depth, joint=None, pos=None):
        pad = "  " * depth
        l = links[ln]
        I = l["inertia"]
        out = [f'{pad}<body name="{ln}"' + (f' pos="{f3(pos)}"' if pos is not None else "") + ">"]
        if joint is None:
            out.append(f"{pad}  <freejoint/>")
        else:
            cls = "passive" if joint["mimic"] else "servo"
            out.append(f'{pad}  <joint name="{joint["name"]}" class="{cls}" axis="{f3(joint["axis"])}"/>')
        for sn, sp in l["sites"]:
            out.append(f'{pad}  <site name="{sn}" pos="{f3(sp)}" size="0.002" group="4"/>')
        out.append(f'{pad}  <inertial pos="{f3(l["com"])}" mass="{l["mass"]:.6g}" '
                   f'fullinertia="{I[0,0]:.6g} {I[1,1]:.6g} {I[2,2]:.6g} {I[0,1]:.6g} {I[0,2]:.6g} {I[1,2]:.6g}"/>')
        for fn, col in l["visuals"]:
            out.append(f'{pad}  <geom class="visual" mesh="{os.path.basename(fn)[:-4]}" rgba="{f3(col)} 1"/>')
        for fn in l["collisions"]:
            out.append(f'{pad}  <geom class="collision" mesh="col_{os.path.basename(fn)[:-4]}"/>')
        for j in children[ln]:
            out += body(j["child"], depth + 1, j, j["xyz"])
        out.append(f"{pad}</body>")
        return out

    acts = [f'    <position name="{j["name"]}" joint="{j["name"]}" class="servo"/>' for j in joints if not j["mimic"]]
    excl = "".join(f'\n    <exclude body1="{a}" body2="{b}"/>' for a, b in excludes)
    contact = f"\n  <contact>{excl}\n  </contact>" if excludes else ""
    eqs = "".join(f'\n    <joint joint1="{j["name"]}" joint2="{j["mimic"][0]}" polycoef="0 {j["mimic"][1]} 0 0 0" solref="0.004 1"/>'
                  for j in joints if j["mimic"])
    equality = f"\n  <equality>{eqs}\n  </equality>" if eqs else ""
    xml = f"""<mujoco model="zang">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <default>
    <default class="servo">
      <joint range="{-JOINT_LIMIT:.4f} {JOINT_LIMIT:.4f}" damping="{DAMPING}" armature="{ARMATURE}" frictionloss="0.01"/>
      <position kp="{KP}" ctrlrange="{-JOINT_LIMIT:.4f} {JOINT_LIMIT:.4f}" forcerange="{-EFFORT} {EFFORT}"/>
    </default>
    <default class="passive">
      <joint damping="{PASSIVE_DAMPING}" armature="0.0005"/>
    </default>
    <default class="visual">
      <geom type="mesh" contype="0" conaffinity="0" group="2" density="0"/>
    </default>
    <default class="collision">
      <geom type="mesh" group="3" density="0" rgba="0.2 0.6 0.9 0.3" friction="0.9 0.01 0.001"/>
    </default>
  </default>
  <asset>
{chr(10).join(assets)}
  </asset>
  <worldbody>
{chr(10).join(body(ROOT_LINK, 2, pos=np.array([0, 0, base_height + 0.002])))}
  </worldbody>{contact}{equality}
  <actuator>
{chr(10).join(acts)}
  </actuator>{keyframe or ""}
</mujoco>
"""
    os.makedirs(os.path.join(ROOT, "mjcf"), exist_ok=True)
    open(os.path.join(ROOT, "mjcf/zang.xml"), "w").write(xml)
    open(os.path.join(ROOT, "mjcf/scene.xml"), "w").write("""<mujoco model="zang scene">
  <include file="zang.xml"/>
  <visual><headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/><global azimuth="150" elevation="-20" offwidth="1280" offheight="960"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="grid" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="5 5" texuniform="true" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="grid"/>
  </worldbody>
</mujoco>
""")


def home_keyframe():
    """MJCF <keyframe> for HOME, with the base lowered until the lowest collision vertex touches the floor."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "mjcf/scene.xml"))
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    for jn, v in HOME.items():
        d.qpos[m.jnt_qposadr[m.joint(jn).id]] = v
    for e in range(m.neq):
        d.qpos[m.jnt_qposadr[m.eq_obj1id[e]]] = m.eq_data[e][1] * d.qpos[m.jnt_qposadr[m.eq_obj2id[e]]]
    mujoco.mj_kinematics(m, d)
    zmin = np.inf
    for g in range(m.ngeom):
        if m.geom_group[g] == 3 and m.geom_dataid[g] >= 0:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            zmin = min(zmin, (d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T)[:, 2].min())
    d.qpos[2] -= zmin - 0.0005
    ctrl = [HOME.get(m.actuator(i).name, 0.0) for i in range(m.nu)]
    return (f'\n  <keyframe>\n    <key name="home" qpos="{" ".join(f"{x:.6g}" for x in d.qpos)}"'
            f'\n         ctrl="{" ".join(f"{x:.6g}" for x in ctrl)}"/>\n  </keyframe>')


def overlapping_links():
    import mujoco
    m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "mjcf/scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    pairs = set()
    for c in d.contact[:d.ncon]:
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if b1 and b2 and c.dist < 0:
            pairs.add(tuple(sorted((m.body(b1).name, m.body(b2).name))))
    return sorted(pairs)


def write_report(links, joints, base_height):
    print(f"{'link':16s} {'mass[g]':>8s}  parts")
    for ln, l in links.items():
        short = sorted({re.sub(r"^.*?/?([^/]+?)(/body)?$", r"\1", p) for p in l["parts"] if not re.search(r"Screw|Header|Standoff|Zinc", p)})
        print(f"{ln:16s} {l['mass']*1000:8.1f}  {', '.join(short)}")
    print(f"total mass {sum(l['mass'] for l in links.values())*1000:.0f} g, pelvis height {base_height*1000:.0f} mm")
    for j in joints:
        print(f"{j['name']:22s} {j['parent']:>15s} -> {j['child']:15s} axis={np.round(j['axis'],3)} xyz={np.round(j['xyz']*1000,1)}mm")


if __name__ == "__main__":
    main()
