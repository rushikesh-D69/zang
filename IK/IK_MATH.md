# Zang kinematics and inverse kinematics

Maths behind [zang_ik.py](zang_ik.py). All numbers come from `../zang_description/zang.urdf`, which is generated from the CAD (`Zang Assembly v1.0.2.step`). Units are metres and radians in code and millimetres in this text.

## 1. Conventions

- **Frames** follow REP-103: x forward, y left, z up. The root link is `pelvis`, with its origin at the Torso part's centroid.
- **Link frames** sit on their parent joint's axis and are **unrotated** at q = 0, so every URDF joint has `rpy = 0 0 0`. A joint's transform is `Trans(origin_xyz) * Rot(axis, q)`.
- **Axis direction:** each axis points along +X, +Y or +Z of the robot frame, with positive rotation by the right-hand rule. Left and right joints have the **same** axis direction; they are not mirrored. The two shoulder-pitch axes are tilted 9° in the CAD:
  - left: (0, 0.9877, 0.1564)
  - right: (0, 0.9877, -0.1561)
- **Zero pose (q = 0)** is the CAD pose: legs straight, arms hanging, pelvis 246 mm above the floor.
- **HOME pose** is the rest pose: `hip_pitch = knee_pitch = -0.5` on both legs, all other joints 0. The pelvis is then 231.6 mm high, with both knees bent forward. It is the MJCF keyframe `home` and `zang_ik.HOME`.

### What a positive angle does

Measured from HOME with the pelvis held fixed:

| joint | +0.2 rad moves |
|---|---|
| `waist_yaw` | chest turns left (counter-clockwise seen from above) |
| `*_hip_roll` | leg swings to +y: **left leg outward, right leg inward** |
| `*_hip_pitch` | thigh and foot go **backward**; negative bends the knee forward |
| `*_knee_pitch` | foot goes **forward** relative to the pelvis; with the foot planted, negative pushes the knee forward (deeper squat) |
| `*_ankle_roll` | foot rolls about +x |
| `*_shoulder_pitch` | hand goes **backward**; negative raises the arm forward |
| `*_shoulder_roll` | hand goes to +y: **left arm outward, right arm inward** |
| `*_elbow` | hand goes to +y: **left outward, right inward** |

To map these to real servo commands, apply a per-servo sign and offset in the firmware.

## 2. Kinematic structure

The robot has 15 servo (active) joints, 8 passive pins that follow the servos, and 4 loop closures. That makes 24 links in the spanning tree.

```
pelvis ─ waist_yaw ─ chest ─┬─ L/R shoulder_pitch ─ shoulder ─ shoulder_roll ─ upper_arm ─ elbow ─ hand
                            │
pelvis ─ L/R hip_roll ─ hip ─┬─ hip_pitch (servo) ─ thigh ─ knee_upper (pin) ─ knee ─ knee_lower (pin) ─ shin ─ knee_pitch (servo) ─ ankle ─┬─ ankle_roll ─ foot
                             └─ thigh_shell (pin) ─ thigh_shell ·····(loop: shell ↔ knee pin)·····                                    └─ shin_shell (pin) ─ shin_shell ···(loop: shell ↔ knee pin)···
```

| kind | joints |
|---|---|
| servo (active) | `waist_yaw`, `{L,R}_hip_roll`, `{L,R}_hip_pitch`, `{L,R}_knee_pitch`, `{L,R}_ankle_roll`, `{L,R}_shoulder_pitch`, `{L,R}_shoulder_roll`, `{L,R}_elbow` |
| passive pin (spanning tree, follows a servo) | `knee_upper = -hip_pitch`, `thigh_shell = +hip_pitch`, `knee_lower = -knee_pitch`, `shin_shell = -knee_pitch` |
| loop closure (cut joint) | thigh_shell ↔ knee, shin_shell ↔ knee (sites `*_loop_a/_b` in the MJCF) |

Each passive pin's angle is fixed by its servo (section 3.1). The URDF expresses this as `<mimic>` joints and the MJCF as `<equality><joint polycoef="0 k 0 0 0">`. `joints.csv` lists every joint's origin, axis, limits, home value and mimic relation.

Figures (regenerate with `python make_figures.py`):
- [figures/fig_topology.png](figures/fig_topology.png): the kinematic graph. It shows the root body, active, passive and loop joints, the parallelogram submechanisms, and the equivalent serial leg chain.
- [figures/fig_leg_mechanism.png](figures/fig_leg_mechanism.png): a side view of the left leg with pins H1…A2, at HOME, in a deep squat, and with the leg forward.
- [figures/fig_skeleton.png](figures/fig_skeleton.png): the whole-robot joint skeleton at HOME, with servo axes, IK end points and the centre of mass.

### 2.1 Series-parallel hybrid formulation

The robot follows the standard formulation for series-parallel hybrid robots:

- **Independent coordinates y:** the 15 servo angles. These are also the actuated coordinates u.
- **Spanning-tree coordinates q:** y plus the 8 passive pins. Cutting the 4 loop pins (thigh_shell ↔ knee, shin_shell ↔ knee) turns the closed chains into a tree.
- **Loop closure:** each loop requires that pin K2 or K4 seen from the shell equals the same pin seen from the knee. Solving this gives q = γ(y).
- **Ideal parallelogram:** γ is linear. q_pin = ±y_servo, so the Jacobian G = ∂q/∂y is a constant matrix with entries ±1. The URDF `<mimic>` joints and the MJCF joint equalities encode exactly this.
- **Use in practice:** forward kinematics, IK and dynamics can all run on the spanning tree. Joint-space quantities map back to the servos through G. For example, servo torque τ_y = Gᵀ τ_q.

## 3. Legs: two parallelograms in series

The photos of the real robot and the CAD pin holes give the leg layout.

- **Upper half:** the hip bracket is the fixed frame of the linkage. The *thigh* (servo case of the `hip_pitch` servo plus two side plates) turns about the servo axis H1. The outer *thigh shell* turns about pin H2 on the same bracket. Both connect to the *knee* block, at pins K1 and K2.
- **Lower half:** the same arrangement mirrored. The *shin* (servo case of the `knee_pitch` servo) and the *shin shell* connect the knee (pins K3, K4) to the ankle bracket (servo axis A1, pin A2).
- **Parallelograms:** in both halves, H1K1 ∥ H2K2 and K3A1 ∥ K4A2, and the opposite sides have equal length. The CAD matches an ideal parallelogram to within 0.5 mm.

Pins from the CAD for the left leg (CAD y/z in mm; the pin axis is along x):

| pin | joins | y | z |
|---|---|---|---|
| H1 | hip bracket ↔ thigh (servo axis) | 6.00 | -72.75 |
| H2 | hip bracket ↔ thigh shell | -12.00 | -54.75 |
| K1 | thigh ↔ knee | 5.81 | -133.18 |
| K2 | thigh shell ↔ knee | -12.42 | -114.75 |
| K3 | knee ↔ shin | 5.74 | -154.44 |
| K4 | knee ↔ shin shell | -12.61 | -172.75 |
| A1 | shin ↔ ankle bracket (servo axis) | 5.54 | -214.86 |
| A2 | shin shell ↔ ankle bracket | -12.80 | -232.75 |

Lengths used by the model:

| quantity | value |
|---|---|
| thigh crank \|H1K1\| | 60.44 mm |
| knee offset \|K1K3\| | 21.26 mm |
| shin crank \|K3A1\| | 60.44 mm |
| total length, hip-pitch axis to knee-pitch axis | 142.1 mm (legs straight) |

### 3.1 Consequences

- **The knee never rotates.** In a parallelogram the coupler keeps the orientation of the fixed link, so the knee stays parallel to the hip bracket. This gives `knee_upper = -hip_pitch` and `thigh_shell = +hip_pitch`.
- **The ankle bracket keeps the knee's orientation.** The lower parallelogram does the same thing, so the ankle bracket stays parallel to the knee and therefore to the hip bracket. This gives `knee_lower = -knee_pitch` and `shin_shell = -knee_pitch`.
- **The foot can only roll relative to the pelvis.** Foot pitch and yaw always equal the pelvis's: R_foot = R_pelvis · Rx(hip_roll + ankle_roll).
- **Each leg has 4 DOF:** 3D position of the sole plus foot roll.
  - With both feet flat on the ground, the pelvis pitch must be 0.
  - With no hip yaw, the robot cannot turn in place without sliding a foot. The waist yaw turns only the upper body.

### 3.2 Forward kinematics of one leg (pelvis frame)

Notation: Rx and Ry are rotations about x and y. All vectors are joint origins from `joints.csv`, each in its parent link's frame.

| symbol | joint origin | value (left leg, mm) |
|---|---|---|
| r | `hip_roll` | (-27.3, ±37.5, -17.1) |
| h1 | `hip_pitch` | — |
| u | `knee_upper` | (0, 25, -60.44) |
| v | `knee_lower` | (0, 0, -21.26) |
| w | `knee_pitch` | (0, -25, -60.44) |
| e | `ankle_roll` | — |
| s | sole centre in the foot frame | (25.1, ±20.3, -23.6) |

With hip_roll φ, hip_pitch q_h, knee_pitch q_k and ankle_roll ψ:

```
ankle  a(q_h, q_k) = h1 + Ry(q_h)·u + v + Ry(-q_k)·w          (hip frame; knee and ankle unrotated)
sole   p = r + Rx(φ)·(a + e) + Rx(φ+ψ)·s                       (pelvis frame)
```

### 3.3 Analytic IK (`Zang.leg_ik`)

Inputs: sole target p (pelvis frame) and foot roll ρ relative to the pelvis.

1. Set ankle_roll: ψ = ρ - φ.
2. Move the known terms to one side: d = p - r - Rx(ρ)·s = Rx(φ)·(h1 + v + e + Ry(q_h)u + Ry(-q_k)w).
3. **Hip roll.** Rotations about y leave the y component unchanged, so `[Rx(-φ)·d]_y = h1_y + v_y + e_y + u_y + w_y =: c`. That gives the scalar equation `d_y·cos φ + d_z·sin φ = c`, solved as φ = atan2(d_z, d_y) ± acos(c / √(d_y² + d_z²)).
4. **Pitch plane.** Write x-z vectors as complex numbers ζ = x + i·z. A rotation Ry(θ) is then multiplication by e^{-iθ}. Let
   `T = [Rx(-φ)d - (h1 + v + e)]_xz = e^{-i q_h}·U + e^{+i q_k}·W`, with U = u_xz and W = w_xz.
   This is a planar two-link arm with lengths |U| and |W|:
   - β = acos((|U|² + |T|² - |W|²) / (2|U||T|))
   - link1 = |U|·e^{i(arg T ± β)}
   - q_h = arg U - arg link1
   - q_k = arg(T - link1) - arg W
5. **Branch choice.** The `+β` branch puts the knee **in front of** the hip-ankle line; this is the default (`knee_forward=True`). Among the up to 4 solutions (2 hip-roll × 2 knee), the solver keeps those within joint limits, prefers knee-forward, then picks the one nearest the hint (default HOME).

**Physical knee angle.** The thigh's forward tilt is -q_h, measured from the hip bracket. The shin's forward tilt is -q_k, measured from the foot. The knee flexion is therefore (-q_h) + (-q_k). For a symmetric bend q_h = q_k = -θ, the ankle stays directly under the hip and the leg shortens by 2·60.44·(1 - cos θ). HOME uses θ = 0.5, which lowers the pelvis by about 15 mm.

### 3.4 Is the hybrid model correct? (`verify_linkage.py`)

Two independent checks compare the ideal-parallelogram γ against the true closed loops built from the CAD pin positions:

1. **Exact kinematic closure.** For each servo setting, a least-squares solve finds the passive pin angles that close both four-bar loops exactly; residual below 1 µm.
2. **Closed-chain dynamics.** MuJoCo simulates the robot with the passive pins **free**. The loops are held only by `connect` constraints between the MJCF loop sites, so no mimic relation is assumed.

| check | working range (q in [-0.9, 0.3] rad) | full range (q in [-1.2, 0.6] rad) |
|---|---|---|
| passive pin angle vs ideal γ | ≤ 0.82° | ≤ 1.23° |
| foot position vs ideal model | ≤ 2.6 mm | ≤ 3.8 mm |
| foot pitch (ideal model: exactly 0) | ≤ 1.4° | ≤ 2.2° |
| closed-chain sim: pin vs γ | 0.33° (HOME), 0.65° (deep squat) | 1.22° (q = -1.2) |
| closed-chain sim: loop gap | ≤ 0.12 mm | ≤ 0.12 mm |

**Conclusion:**

- **The structure is right.** The leg really is R → parallelogram → parallelogram → R. The two methods agree with each other, for example 1.22° in simulation against 1.23° from the exact solve. The closed-chain model stands, squats and holds HOME.
- **The remaining deviation is geometric.** In the CAD, the servo cranks are 60.43 mm but the shells' pin spacing is 60.00 mm. The knee's pin offset is 25.92 mm against the hip bracket's 25.46 mm, so the CAD loops are near-parallelograms, not exact ones.
- **Which is correct depends on the real parts.** If the real parts are true parallelograms (likely, since the foot is meant to stay level), the ideal model is exact. If they match the CAD, expect the errors above.
- **How to settle it:** on the robot, measure servo shaft → knee pin on the inner plate, and shell pin → shell pin. If both are 60.0 mm, the ideal model is exact.

## 4. Arms: one pitch joint, then a planar two-link chain

`shoulder_pitch` turns about axis **a** (tilted 9°). `shoulder_roll` and `elbow` both turn about x, so after the pitch rotation they form a planar two-link chain in the y-z plane of the shoulder frame.

| symbol | meaning | value (mm, left arm) |
|---|---|---|
| r | `shoulder_pitch` origin in the chest frame | (0, ±54.5, 71.1) |
| o1 | `shoulder_roll` origin | (-25.0, ±38.1, -8.7) |
| o2 | `elbow` origin; upper-arm length 61.5 | (-0.25, ±9.5, -60.8) |
| h | hand tip in the hand frame (lowest point of the gripper) | (24.6, ±1.35, -85.1) |

Forward kinematics, with shoulder pitch q1, shoulder roll q2 and elbow q3:

```
hand = r + R(a, q1)·( o1 + Rx(q2)·( o2 + Rx(q3)·h ) )
```

IK (`Zang.arm_ik`), position only, 3 DOF. Input: hand target p in the chest frame.

1. Let d = p - r. Rotations about x leave the x component unchanged, so `d · R(a,q1)x̂ = o1_x + o2_x + h_x`. With a ⟂ x̂, this is `(d·x̂)·cos q1 + (d·(a×x̂))·sin q1 = c`, solved as in step 3 of the leg.
2. Compute ds = R(a, -q1)·d - o1 and work in the y-z plane, where Rx(θ) is multiplication by e^{+iθ}:
   - T = ds_yz = e^{i q2}·O2 + e^{i(q2+q3)}·H
   - solve the two-link arm; the elbow branch nearest the hint wins.

## 5. Waist

`waist_yaw` is a single z rotation between pelvis and chest. The chest frame is `T_pelvis · Trans(waist_xyz) · Rz(waist_yaw)`, and arm targets in world coordinates are converted to that frame.

## 6. Whole-body IK (`Zang.whole_body_ik`, `Zang.balance_ik`)

Inputs:
- the world pelvis pose (position, roll/pitch/yaw);
- foot targets `{side: (sole_pos_world, foot_roll_world)}`;
- optional hand targets `{side: hand_pos_world}` and the waist yaw.

Steps:

1. Express each foot target in the pelvis frame and call `leg_ik`. The relative foot orientation must be a pure x rotation (section 3.1); if it isn't, the solver raises `IKError`.
2. Express each hand target in the chest frame and call `arm_ik`.
3. Fill the passive pins from their mimic relations (`full_q`).

`balance_ik` wraps this to place the **centre of mass** (CoM) over a target point, which is how weight shifting works. It solves the whole body, computes the CoM from the URDF inertials (`Zang.com`), and moves the pelvis horizontally by the CoM error. Repeating this converges in a few iterations because most of the mass rides on the pelvis. Use it to keep the CoM over the stance foot before lifting the other one.

`numeric_ik` is a fallback for tasks without a closed form, such as a point on the forearm or many targets at once. It runs damped least squares on the forward kinematics with a finite-difference Jacobian and clamps to joint limits.

## 7. Validation (`test_ik.py`, `demo_mujoco.py`)

| check | result |
|---|---|
| Forward kinematics vs MuJoCo, 20 random full-body poses | max error 1e-15 |
| Leg IK round trip, 500 random poses | 6e-16 m |
| Arm IK round trip, 500 random poses | 2e-16 m |
| Whole-body IK, pelvis rolled and yawed, waist turned | feet exactly flat |
| Balance IK, CoM over the left foot | within 1e-5 m |
| HOME vs MJCF keyframe | equal |
| MuJoCo demo, 17 s: squat, weight shift, 2 s on one leg, arm reach, waist turn | max pelvis tilt 2.7°, no fall |
| Hybrid leg vs true closed loops (`verify_linkage.py`) | see §3.4 |

## 8. Approximations and open items

- **Ideal parallelograms.** The mimic relations assume perfect parallelograms. The CAD dimensions deviate slightly; §3.4 quantifies the effect (≤ 1.4° foot pitch in the working range). Measure the real parts to settle which is right.
- **Placeholder dynamics.** Masses, servo torque (2 N·m), speed and limits (±90°) are guesses; edit them in `../zang_description/tools/build.py`. Weigh the robot and look up the DS-R003G datasheet.
- **CAD asymmetry.** The right lower leg is about 1 mm off mirror in the CAD, so the left and right numbers differ slightly. The IK uses each side's own values.
- **Joint-limit check only.** `leg_ik` and `arm_ik` check joint limits but not self-collision; very deep squats can make the shells meet the brackets.
