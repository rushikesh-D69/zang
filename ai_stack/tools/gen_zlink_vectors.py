"""Write ZLink golden vectors for the firmware's host tests.

    .venv/bin/python tools/gen_zlink_vectors.py      (from ai_stack/)

Output: ../firmware/zang_p4/components/zlink/test/zlink_vectors.json. Each vector is a message built by the Python
codec, with its fields and the exact frame bytes (hex). The C test must encode the same fields to the same bytes
and decode the bytes back to the same fields.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zang_ai.link import zlink as Z  # noqa: E402
from zang_ai.paths import WORKSPACE  # noqa: E402

OUT = os.path.join(WORKSPACE, "firmware", "zang_p4", "components", "zlink", "test", "zlink_vectors.json")


def cdeg(q):
    return Z.rad_to_cdeg(q).astype(int).tolist()


def main():
    q = np.radians(np.arange(Z.N_JOINTS) * 7.25 - 50.0)
    q2 = -q[::-1]
    vec = []

    def add(name, typ, seq, payload, fields):
        vec.append({"name": name, "type": typ, "seq": seq, "fields": fields,
                    "payload_hex": payload.hex(), "frame_hex": Z.frame(typ, payload, seq).hex()})

    add("hello", Z.HELLO, 1, Z.hello(123456), {"host_ms": 123456})
    add("ping", Z.PING, 2, Z.ping(0xDEADBEEF), {"host_us": 0xDEADBEEF})
    add("pong", Z.PONG, 3, Z.pong(1000, 2000), {"host_us": 1000, "robot_us": 2000})
    add("hello_ack", Z.HELLO_ACK, 4, Z.hello_ack(15, 0x0100, "zang"),
        {"proto_ver": 1, "n_joints": 15, "fw_ver": 0x0100, "name": "zang"})
    add("joint_targets", Z.JOINT_TARGETS, 5, Z.joint_targets(q, 5000, 0x7FFF, 40),
        {"t_robot_ms": 5000, "mask": 0x7FFF, "move_ms": 40, "q_cdeg": cdeg(q)})
    add("traj_chunk", Z.TRAJ_CHUNK, 6, Z.traj_chunk(9, 1000, [20, 20], np.stack([q, q2]), Z.INTERP_CUBIC),
        {"stream_id": 9, "t0_ms": 1000, "interp": 1, "frames": [{"dt_ms": 20, "q_cdeg": cdeg(q)},
                                                                {"dt_ms": 20, "q_cdeg": cdeg(q2)}]})
    add("stream_start", Z.STREAM_START, 7, Z.stream_start(9, 1, 0), {"stream_id": 9, "clock": 1, "flags": 0})
    add("stream_end", Z.STREAM_END, 8, Z.stream_end(9), {"stream_id": 9})
    add("cmd_estop", Z.CMD, 9, Z.cmd(Z.CMD_ESTOP), {"cmd": Z.CMD_ESTOP, "data_hex": ""})
    add("cmd_go_home", Z.CMD, 10, Z.cmd(Z.CMD_GO_HOME, (1500).to_bytes(2, "little")),
        {"cmd": Z.CMD_GO_HOME, "data_hex": (1500).to_bytes(2, "little").hex()})
    st = Z.State(t_robot_ms=777, flags=0b100101, q_meas=q, q_cmd=q2, rpy=np.radians([1.5, -2.25, 90.0]),
                 gyro_dps=np.array([1.2, -3.4, 5.6]), acc_g=np.array([0.01, -0.02, 0.981]), vbat_mv=7400,
                 temp_max_c=41, fault_servo_id=0)
    add("state", Z.STATE, 11, st.pack()[:Z.STATE_SIZE],
        {"t_robot_ms": 777, "flags": 0b100101, "q_meas_cdeg": cdeg(q), "q_cmd_cdeg": cdeg(q2),
         "rpy_cdeg": [150, -225, 9000], "gyro_ddps": [12, -34, 56], "acc_mg": [10, -20, 981],
         "vbat_mv": 7400, "temp_max_c": 41, "fault_servo_id": 0})
    # ---- v1.1 ----
    st11 = Z.State(t_robot_ms=777, flags=0b100101, q_meas=q, q_cmd=q2, rpy=np.radians([1.5, -2.25, 90.0]),
                   gyro_dps=np.array([1.2, -3.4, 5.6]), acc_g=np.array([0.01, -0.02, 0.981]), vbat_mv=7400,
                   temp_max_c=41, fault_servo_id=0, stream_buffered_ms=450, stream_pos_ms=123456, display_mode=0, emotion=3)
    add("state_v11", Z.STATE, 12, st11.pack(),
        {"t_robot_ms": 777, "flags": 0b100101, "q_meas_cdeg": cdeg(q), "q_cmd_cdeg": cdeg(q2),
         "rpy_cdeg": [150, -225, 9000], "gyro_ddps": [12, -34, 56], "acc_mg": [10, -20, 981],
         "vbat_mv": 7400, "temp_max_c": 41, "fault_servo_id": 0,
         "stream_buffered_ms": 450, "stream_pos_ms": 123456, "display_mode": 0, "emotion": 3})
    add("face_set", Z.FACE_SET, 13, Z.face_set(6, 200, 300, -12, 5, talking=True, blink=False, hold=True),
        {"emotion": 6, "intensity": 200, "blend_ms": 300, "gaze_x": -12, "gaze_y": 5, "flags": 0b101})
    add("face_params", Z.FACE_PARAMS, 14, Z.face_params({0: 0.5, 16: -0.75, 21: 300.0}, 120),
        {"blend_ms": 120, "params": [{"id": 0, "value": 0.5}, {"id": 16, "value": -0.75}, {"id": 21, "value": 300.0}]})
    add("face_track", Z.FACE_TRACK, 15, Z.face_track(9, [(0, 1, 255, 200), (1500, 9, 180, 300)]),
        {"stream_id": 9, "keys": [{"t_ms": 0, "emotion": 1, "intensity": 255, "blend_ms": 200},
                                  {"t_ms": 1500, "emotion": 9, "intensity": 180, "blend_ms": 300}]})
    img = bytes(range(40))
    add("display_image", Z.DISPLAY_IMAGE, 16, Z.display_image_chunks(7, img, Z.IMG_JPEG, show=True, store=True)[0],
        {"image_id": 7, "format": 1, "flags": 3, "total_len": 40, "offset": 0, "data_hex": img.hex()})
    add("display_text", Z.DISPLAY_TEXT, 17, Z.display_text("Hello Zang", 2, 0x07E0, 0x0000, 1500),
        {"size": 2, "fg": 0x07E0, "bg": 0, "duration_ms": 1500, "text": "Hello Zang"})
    add("display_mode", Z.DISPLAY_MODE, 18, Z.display_mode(Z.DISPLAY_IMG, 7), {"mode": 1, "image_id": 7})
    jpg = bytes(range(30))
    add("camera_frame", Z.CAMERA_FRAME, 19, Z.camera_frame_chunks(42, 9000, 320, 240, jpg)[0],
        {"frame_id": 42, "robot_ms": 9000, "w": 320, "h": 240, "format": 1, "total_len": 30, "offset": 0,
         "data_hex": jpg.hex()})
    add("cmd_camera_stream", Z.CMD, 20, Z.cmd(Z.CMD_CAMERA_STREAM, bytes([1, 0, 10, 60])),
        {"cmd": Z.CMD_CAMERA_STREAM, "data_hex": bytes([1, 0, 10, 60]).hex()})
    cal = b"".join(__import__("struct").pack("<hbH", 2048 + i, -1 if i % 2 else 1, 4096) for i in range(15))
    add("calib_report", Z.CALIB_REPORT, 21, cal,
        {"joints": [{"zero_ticks": 2048 + i, "sign": -1 if i % 2 else 1, "ticks_per_rev": 4096} for i in range(15)]})
    doc = {"_doc": "Generated by ai_stack/tools/gen_zlink_vectors.py from the Python ZLink codec. Do not edit.",
           "crc_check": {"input_ascii": "123456789", "crc16": Z.crc16(b"123456789")},
           "vectors": vec}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(doc, fh, indent=1)
    print(f"wrote {len(vec)} vectors -> {OUT}")


if __name__ == "__main__":
    main()
