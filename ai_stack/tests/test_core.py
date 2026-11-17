"""Core tests: joint contract, ZLink codec (incl. the firmware golden vectors), safety stage, composer, export, link.

    .venv/bin/python -m pytest -q tests/
"""
import json
import os
import socket
import threading
import time

import numpy as np
import pytest

from zang_ai.link import zlink as Z
from zang_ai.paths import WORKSPACE
from zang_ai.robot import ANAT, RobotModel


@pytest.fixture(scope="module")
def robot():
    return RobotModel()


@pytest.fixture(scope="module")
def composer(robot):
    from zang_ai.pipeline import Composer
    return Composer(robot)


# ---- joint contract -----------------------------------------------------------------------------------------
def test_robot_json_matches_mjcf(robot):
    assert robot.n == Z.N_JOINTS == 15
    for i, name in enumerate(robot.names):
        assert robot.m.joint(name).id >= 0 and robot.m.actuator(name).id >= 0
    assert set(robot.anat_sign) == {"l_pitch", "l_roll", "l_elbow", "r_pitch", "r_roll", "r_elbow", "gaze", "sway"}


def test_anatomical_directions(robot):
    """+pitch raises the hand forward, +roll moves it outward, both sides."""
    for side, out in (("l", 1), ("r", -1)):
        A = np.zeros(len(ANAT))
        robot.kinematics(robot.anat_to_q(A)[0])
        p0 = robot._hand_tip(robot.d, "left" if side == "l" else "right").copy()
        A[ANAT.index(f"{side}_pitch")] = 60
        robot.kinematics(robot.anat_to_q(A)[0])
        p1 = robot._hand_tip(robot.d, "left" if side == "l" else "right")
        assert p1[0] - p0[0] > 0.03 and p1[2] > p0[2]
        A[:] = 0
        A[ANAT.index(f"{side}_roll")] = 40
        robot.kinematics(robot.anat_to_q(A)[0])
        p2 = robot._hand_tip(robot.d, "left" if side == "l" else "right")
        assert (p2[1] - p0[1]) * out > 0.02


def test_anat_roundtrip(robot):
    A = np.random.default_rng(0).normal(0, 20, (20, len(ANAT)))
    assert np.abs(robot.q_to_anat(robot.anat_to_q(A)) - A).max() < 1e-9


def test_sway_shifts_com_over_left_foot(robot):
    A = np.zeros((2, len(ANAT)))
    A[1, ANAT.index("sway")] = 1.0
    d = robot.d
    rel = []
    for q in robot.anat_to_q(A):
        robot.kinematics(q)
        mid = 0.5 * (d.xpos[robot.feet["left"]] + d.xpos[robot.feet["right"]])
        rel.append(d.subtree_com[robot.pelvis] - mid)
    assert rel[1][1] > rel[0][1]


def test_home_is_balanced(robot):
    _, margin = robot.com_margin(robot.home)
    assert margin[0] > 30


# ---- ZLink ----------------------------------------------------------------------------------------------------
def test_crc_known_answer():
    assert Z.crc16(b"123456789") == 0x29B1


def test_frame_roundtrip_and_corruption():
    q = np.radians(np.linspace(-80, 80, 15))
    f = Z.frame(Z.JOINT_TARGETS, Z.joint_targets(q, 42, 0x7FFF, 20), 9)
    typ, seq, payload = Z.parse(f)
    assert (typ, seq) == (Z.JOINT_TARGETS, 9)
    bad = bytearray(f)
    bad[12] ^= 0xFF
    with pytest.raises(Z.ZLinkError):
        Z.parse(bytes(bad))


def test_stream_parser_resyncs():
    f = Z.frame(Z.PING, Z.ping(1), 1)
    p = Z.StreamParser()
    out = p.feed(b"\x00ZL\x01garbage" + f + f[:4])
    out += p.feed(f[4:])
    assert [o[0] for o in out] == [Z.PING, Z.PING]


def test_state_pack_unpack():
    q = np.radians(np.arange(15) * 5.0 - 30)
    s = Z.State(1, 0b100001, q, -q, np.radians([1, 2, 3]), np.array([1.5, 0, -2]), np.array([0, 0, 1.0]), 7400, 40, 0)
    s.stream_buffered_ms, s.emotion = 420, 5
    raw = s.pack()
    assert len(raw) == Z.STATE_SIZE_V11 == 96 and Z.STATE_SIZE == 88
    s2 = Z.State.unpack(raw)
    assert np.allclose(s2.q_meas, q, atol=np.radians(0.006)) and s2.flag("torque_on") and s2.flag("calibrated")
    assert (s2.stream_buffered_ms, s2.emotion) == (420, 5)
    old = Z.State.unpack(raw[:88])  # a v1.0 robot's STATE still parses
    assert old.stream_buffered_ms == 0 and np.allclose(old.q_meas, q, atol=np.radians(0.006))


def test_trajectory_chunks_fit_udp():
    t = np.arange(1, 501) * 0.02
    chunks = Z.chunk_trajectory(t, np.zeros((500, 15)), 3)
    assert all(len(c) <= Z.MAX_PAYLOAD for c in chunks) and len(chunks) == 13


def test_golden_vectors_match_python_codec():
    """The vectors the firmware tests against must be exactly what this codec produces."""
    path = os.path.join(WORKSPACE, "firmware", "zang_p4", "components", "zlink", "test", "zlink_vectors.json")
    if not os.path.exists(path):
        pytest.skip("vectors not generated (tools/gen_zlink_vectors.py)")
    doc = json.load(open(path))
    assert doc["crc_check"]["crc16"] == Z.crc16(b"123456789")
    for v in doc["vectors"]:
        typ, seq, payload = Z.parse(bytes.fromhex(v["frame_hex"]))
        assert (typ, seq, payload.hex()) == (v["type"], v["seq"], v["payload_hex"]), v["name"]


# ---- safety ---------------------------------------------------------------------------------------------------
def test_safety_fixes_collisions_and_limits(composer):
    s = composer.safety
    N = 200
    A = np.zeros((N, len(ANAT)))
    tt = np.arange(N) / 50
    A[:, ANAT.index("l_elbow")] = A[:, ANAT.index("r_elbow")] = 95 * np.clip(np.sin(tt), 0, 1)  # elbows into the hips
    A[:, ANAT.index("l_pitch")] = 170 * np.sin(4 * tt) ** 2                                   # too fast, too far
    Q, rep = s.run(A)
    assert rep["collision_projected_frames"] > 0
    assert rep["pass"], rep
    assert rep["min_clearance_mm"] >= s.col.clearance * 1000 - 0.05
    assert (Q >= composer.r.lower - 1e-9).all() and (Q <= composer.r.upper + 1e-9).all()


def test_safe_pose_projects_arm(composer):
    r = composer.r
    A = np.zeros(len(ANAT))
    A[ANAT.index("l_elbow")] = A[ANAT.index("r_elbow")] = 90
    q, rep = composer.safety.safe_pose(r.anat_to_q(A)[0])
    assert rep["projected"] and rep["clearance_mm"] >= 4.0


# ---- composer + export ----------------------------------------------------------------------------------------
def test_compose_text_passes_safety_and_physics(composer):
    res, _ = composer.compose_text("Hello! <wave> I am Zang. Let me think about it. <think>", use_model=False)
    assert res["safety"]["pass"], res["safety"]
    assert res["physics"]["pass"], res["physics"]
    assert res["audio_offset"] > 0  # the wave on the first word needs a wind-up before the audio
    gestures = [e["gesture"] for e in res["events"] if e["type"] == "semantic"]
    assert "wave" in gestures and "think" in gestures


def test_zact_roundtrip(tmp_path, robot):
    from zang_ai.motion.export import read_zact, write_zact
    t = np.arange(300) * 0.02
    Q = np.repeat(robot.home[None], 300, 0)
    Q[:, 9] = np.linspace(0, -1.2, 300)
    p = write_zact(str(tmp_path / "A.zact"), t, Q, "A.mp3")
    z = read_zact(p)
    assert z["audio"] == "A.mp3" and len(z["t"]) == 300
    assert np.abs(z["q"] - Q).max() < np.radians(0.006)


# ---- link -----------------------------------------------------------------------------------------------------
def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_client_emulator_stream_and_estop(robot):
    from zang_ai.link.client import RobotLink
    from zang_ai.link.emulator import Emulator
    port = _free_port()
    emu = Emulator(robot, port=port)
    stop = threading.Event()
    threading.Thread(target=emu.serve, args=(stop,), daemon=True).start()
    link = RobotLink("127.0.0.1", port)
    try:
        assert link.connect()["n_joints"] == 15
        N = 60
        t = np.arange(1, N + 1) * 0.02
        Q = np.repeat(robot.home[None], N, 0)
        Q[:, 9] = np.linspace(0, -0.8, N)
        link.play(t, Q, block=True)
        time.sleep(0.4)
        assert abs(link.state.q_meas[9] + 0.8) < np.radians(1.0)
        link.estop()
        time.sleep(0.15)
        assert link.state.flag("estop") and not link.state.flag("torque_on")
    finally:
        link.close()
        stop.set()


# ---- faces -----------------------------------------------------------------------------------------------------
def test_face_library_renders_distinct_faces():
    from zang_ai.face.render import EMOTIONS, render, resolve
    imgs = {e: render(resolve(e), size=120).astype(int) for e in EMOTIONS}
    assert all(im.shape == (120, 120, 3) for im in imgs.values())
    names = list(imgs)
    for i, a in enumerate(names):  # every emotion must look different from every other one
        for b in names[i + 1:]:
            assert np.abs(imgs[a] - imgs[b]).mean() > 0.5, (a, b)


def test_face_goldens_match_reference_renderer():
    """The firmware compares its C renderer against these files: they must equal what render.py draws today."""
    from zang_ai.face.render import render
    gdir = os.path.join(WORKSPACE, "firmware", "zang_p4", "components", "display", "test", "faces")
    if not os.path.exists(os.path.join(gdir, "goldens.json")):
        pytest.skip("goldens not generated (tools/gen_face_goldens.py)")
    doc = json.load(open(os.path.join(gdir, "goldens.json")))
    for case in doc["cases"][:6]:
        raw = open(os.path.join(gdir, case["file"]), "rb").read()
        img = np.frombuffer(raw.split(b"\n", 3)[3], np.uint8).reshape(240, 240, 3)
        p = dict(case["params"], _wink=case["wink"])
        assert np.abs(render(p, blink=case["blink"]).astype(int) - img).max() <= 1, case["name"]


def test_face_track_follows_gestures(composer):
    res, _ = composer.compose_text("Hi! <wave> I love robots. <heart> Let me think. <think>", use_model=False,
                                   physics=False)
    emos = [k["emotion"] for k in res["face"]]
    assert {"happy", "love", "thinking"} <= set(emos)
    assert all(b["t"] >= a["t"] for a, b in zip(res["face"], res["face"][1:]))
