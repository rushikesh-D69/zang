"""ZLink v1.1 codec (Python side). Spec: firmware/docs/ZLINK.md. The C side is firmware/zang_p4/components/zlink.

Both sides are held together by golden vectors (tools/gen_zlink_vectors.py -> firmware test). Change the spec,
then both codecs, then regenerate the vectors; never one side alone.
"""
import struct
from dataclasses import dataclass, field

import numpy as np

MAGIC = b"ZL"
VERSION = 1
N_JOINTS = 15
MAX_PAYLOAD = 1400
MAX_CHUNK_FRAMES = 40
NO_READING = 0x7FFF

HELLO, HELLO_ACK, PING, PONG = 0x01, 0x02, 0x03, 0x04
JOINT_TARGETS, TRAJ_CHUNK, STREAM_START, STREAM_END = 0x10, 0x11, 0x12, 0x13
AUDIO_CHUNK, MIC_CHUNK, CAMERA_FRAME = 0x20, 0x21, 0x22
STATE, CMD, ACK, CALIB_REPORT, LOG, EVENT = 0x30, 0x40, 0x41, 0x42, 0x50, 0x60
FACE_SET, FACE_PARAMS, FACE_TRACK, DISPLAY_IMAGE, DISPLAY_TEXT, DISPLAY_MODE = 0x70, 0x71, 0x72, 0x73, 0x74, 0x75

(CMD_HOLD, CMD_ESTOP, CMD_TORQUE_ON, CMD_GO_HOME, CMD_PLAY_FILE, CMD_SET_RATE, CMD_CALIB_SET, CMD_CALIB_SAVE,
 CMD_CALIB_GET, CMD_MIC_STREAM, CMD_CAMERA_STREAM, CMD_SET_VOLUME, CMD_SET_BRIGHTNESS, CMD_SNAPSHOT) = range(14)
EV_STATE_CHANGED, EV_STREAM_DONE, EV_FILE_DONE, EV_WATCHDOG, EV_SERVO_FAULT, EV_ESTOP, EV_TARGET_DONE = range(1, 8)
EV_BUTTON, EV_LOW_BATTERY, EV_WAKE_WORD, EV_FACE_SEEN, EV_IMAGE_DONE = range(0x10, 0x15)
DISPLAY_FACE, DISPLAY_IMG, DISPLAY_TXT, DISPLAY_CAMERA, DISPLAY_STATUS = range(5)
IMG_RGB565, IMG_JPEG = 0, 1
FW_V11 = 0x8000          # HELLO_ACK fw_ver bit: v1.1 messages supported
IMAGE_CHUNK = 1360
INTERP_LINEAR, INTERP_CUBIC = 0, 1
CODEC_PCM16_16K, CODEC_PCM16_24K, CODEC_OPUS = 0, 1, 2


class ZLinkError(ValueError):
    pass


def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, xorout 0. crc16(b'123456789') == 0x29B1."""
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def frame(msg_type: int, payload: bytes, seq: int) -> bytes:
    if len(payload) > MAX_PAYLOAD:
        raise ZLinkError(f"payload {len(payload)} > {MAX_PAYLOAD}")
    body = struct.pack("<BBHH", VERSION, msg_type, seq & 0xFFFF, len(payload)) + payload
    return MAGIC + body + struct.pack("<H", crc16(body))


def parse(buf: bytes):
    """One complete frame -> (type, seq, payload). Raises ZLinkError on any defect."""
    if len(buf) < 10 or buf[:2] != MAGIC:
        raise ZLinkError("bad magic / short frame")
    ver, typ, seq, n = struct.unpack_from("<BBHH", buf, 2)
    if ver != VERSION:
        raise ZLinkError(f"version {ver}")
    if len(buf) != 10 + n:
        raise ZLinkError(f"length {len(buf)} != {10 + n}")
    (crc,) = struct.unpack_from("<H", buf, 8 + n)
    if crc != crc16(buf[2:8 + n]):
        raise ZLinkError("crc")
    return typ, seq, buf[8:8 + n]


class StreamParser:
    """Byte stream (TCP / UART) -> frames. Resyncs on magic after garbage or a bad CRC."""

    def __init__(self):
        self.buf = bytearray()

    def feed(self, data: bytes):
        self.buf += data
        out = []
        while True:
            i = self.buf.find(MAGIC)
            if i < 0:
                self.buf = self.buf[-1:]
                return out
            del self.buf[:i]
            if len(self.buf) < 8:
                return out
            n = struct.unpack_from("<H", self.buf, 6)[0]
            if n > MAX_PAYLOAD:
                del self.buf[:2]
                continue
            if len(self.buf) < 10 + n:
                return out
            raw = bytes(self.buf[:10 + n])
            try:
                out.append(parse(raw))
                del self.buf[:10 + n]
            except ZLinkError:
                del self.buf[:2]


# ---- units ----------------------------------------------------------------------------------------------
def rad_to_cdeg(q):
    return np.clip(np.round(np.degrees(np.asarray(q, float)) * 100), -32767, 32766).astype("<i2")


def cdeg_to_rad(c):
    c = np.asarray(c, float)
    return np.where(c == NO_READING, np.nan, np.radians(c / 100.0))


# ---- payloads ---------------------------------------------------------------------------------------------
def hello(host_ms):
    return struct.pack("<I", host_ms & 0xFFFFFFFF)


def hello_ack(n_joints, fw_ver, name):
    return struct.pack("<BBH16s", VERSION, n_joints, fw_ver, name.encode()[:16])


def ping(host_us):
    return struct.pack("<I", host_us & 0xFFFFFFFF)


def pong(host_us, robot_us):
    return struct.pack("<II", host_us & 0xFFFFFFFF, robot_us & 0xFFFFFFFF)


def joint_targets(q_rad, t_robot_ms=0, mask=0x7FFF, move_ms=20):
    return struct.pack("<IHH", t_robot_ms & 0xFFFFFFFF, mask, move_ms) + rad_to_cdeg(q_rad).tobytes()


def traj_chunk(stream_id, t0_ms, dt_ms, Q_rad, interp=INTERP_LINEAR):
    Q_rad = np.atleast_2d(Q_rad)
    if len(Q_rad) > MAX_CHUNK_FRAMES:
        raise ZLinkError(f"{len(Q_rad)} frames > {MAX_CHUNK_FRAMES}")
    out = struct.pack("<IIBB", stream_id, t0_ms, interp, len(Q_rad))
    for dt, q in zip(dt_ms, Q_rad):
        out += struct.pack("<H", int(dt)) + rad_to_cdeg(q).tobytes()
    return out


def stream_start(stream_id, clock=0, flags=0):
    return struct.pack("<IBB", stream_id, clock, flags)


def stream_end(stream_id):
    return struct.pack("<I", stream_id)


def audio_chunk(stream_id, sample_offset, data: bytes, codec=CODEC_PCM16_16K):
    return struct.pack("<IIBB", stream_id, sample_offset, codec, 0) + data


def cmd(code, data: bytes = b""):
    return struct.pack("<B", code) + data


STATE_FMT = "<IH" + "15h" * 2 + "3h3h3h" + "HBB"
STATE_SIZE = struct.calcsize(STATE_FMT)          # v1.0: 88 bytes
STATE_TAIL_FMT = "<HIBB"                         # v1.1 tail: buffered_ms, stream_pos_ms, display_mode, emotion
STATE_SIZE_V11 = STATE_SIZE + struct.calcsize(STATE_TAIL_FMT)  # 96 bytes


@dataclass
class State:
    t_robot_ms: int = 0
    flags: int = 0
    q_meas: np.ndarray = field(default_factory=lambda: np.zeros(N_JOINTS))
    q_cmd: np.ndarray = field(default_factory=lambda: np.zeros(N_JOINTS))
    rpy: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    gyro_dps: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acc_g: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vbat_mv: int = 0
    temp_max_c: int = 0
    fault_servo_id: int = 0
    stream_buffered_ms: int = 0
    stream_pos_ms: int = 0
    display_mode: int = 0
    emotion: int = 0

    FLAGS = ("torque_on", "estop", "streaming", "fault", "fallen", "calibrated")

    def flag(self, name):
        return bool(self.flags >> self.FLAGS.index(name) & 1)

    def pack(self):
        rpy = [NO_READING if np.isnan(x) else int(round(np.degrees(x) * 100)) for x in self.rpy]
        return struct.pack(STATE_FMT, self.t_robot_ms, self.flags, *rad_to_cdeg(np.nan_to_num(self.q_meas)),
                           *rad_to_cdeg(self.q_cmd), *rpy, *np.round(np.asarray(self.gyro_dps) * 10).astype(int),
                           *np.round(np.asarray(self.acc_g) * 1000).astype(int), self.vbat_mv, self.temp_max_c,
                           self.fault_servo_id) + struct.pack(STATE_TAIL_FMT, self.stream_buffered_ms,
                                                              self.stream_pos_ms, self.display_mode, self.emotion)

    @classmethod
    def unpack(cls, payload):
        v = struct.unpack(STATE_FMT, payload[:STATE_SIZE])
        q_meas, q_cmd = np.array(v[2:17]), np.array(v[17:32])
        rpy = np.array(v[32:35], float)
        tail = struct.unpack_from(STATE_TAIL_FMT, payload, STATE_SIZE) if len(payload) >= STATE_SIZE_V11 else (0, 0, 0, 0)
        return cls(v[0], v[1], cdeg_to_rad(q_meas), cdeg_to_rad(q_cmd), cdeg_to_rad(rpy),
                   np.array(v[35:38]) / 10.0, np.array(v[38:41]) / 1000.0, v[41], v[42], v[43], *tail)


# ---- trajectory -> chunks ----------------------------------------------------------------------------------
def chunk_trajectory(t, Q, stream_id, max_frames=MAX_CHUNK_FRAMES):
    """Trajectory (t seconds from stream start, Q rad) -> TRAJ_CHUNK payloads. Frame times are rounded on the
    absolute timeline (no cumulative drift); chunk k's t0_ms is the time its first frame is reached minus dt."""
    t_ms = np.round(np.asarray(t) * 1000).astype(np.int64)
    out = []
    prev = 0
    for a in range(0, len(t_ms), max_frames):
        b = min(a + max_frames, len(t_ms))
        dts = np.diff(np.concatenate([[prev], t_ms[a:b]]))
        out.append(traj_chunk(stream_id, int(prev), dts, Q[a:b]))
        prev = int(t_ms[b - 1])
    return out


# ---- v1.1: display / face / camera / calibration ---------------------------------------------------------------
def face_set(emotion, intensity=255, blend_ms=250, gaze_x=0, gaze_y=0, talking=False, blink=False, hold=False):
    flags = (talking << 0) | (blink << 1) | (hold << 2)
    return struct.pack("<BBHbbB", emotion, intensity, blend_ms, int(np.clip(gaze_x, -128, 127)),
                       int(np.clip(gaze_y, -128, 127)), flags)


def face_params(params, blend_ms=150):
    """params: {param_id: value in the parameter's own units (faces.json)}; sent as float32."""
    items = sorted(params.items())
    return struct.pack("<HB", blend_ms, len(items)) + b"".join(struct.pack("<Bf", pid, float(v)) for pid, v in items)


def face_track(stream_id, keys):
    """keys: [(t_ms, emotion, intensity, blend_ms)]; at most 100 per message."""
    if len(keys) > 100:
        raise ZLinkError("face track: split into <= 100 keys per message")
    return struct.pack("<IB", stream_id, len(keys)) + b"".join(struct.pack("<IBBH", *k) for k in keys)


def display_image_chunks(image_id, data: bytes, fmt=IMG_JPEG, show=True, store=False, chunk=IMAGE_CHUNK):
    flags = (show << 0) | (store << 1)
    return [struct.pack("<HBBII", image_id, fmt, flags, len(data), off) + data[off:off + chunk]
            for off in range(0, len(data), chunk)]


def display_text(text, size=2, fg=0xFFFF, bg=0x0000, duration_ms=0):
    return struct.pack("<BHHH", size, fg, bg, duration_ms) + text.encode("utf-8")[:MAX_PAYLOAD - 7]


def display_mode(mode, image_id=0):
    return struct.pack("<BH", mode, image_id)


def rgb565(r, g, b):
    return ((int(r) & 0xF8) << 8) | ((int(g) & 0xFC) << 3) | (int(b) >> 3)


CAM_HDR = "<IIHHBBII"  # 22 bytes


def camera_frame_chunks(frame_id, robot_ms, w, h, data: bytes, chunk=IMAGE_CHUNK):
    """Robot side (emulator / tests): one JPEG -> CAMERA_FRAME payloads."""
    return [struct.pack(CAM_HDR, frame_id, robot_ms, w, h, IMG_JPEG, 0, len(data), off) + data[off:off + chunk]
            for off in range(0, len(data), chunk)]


class FrameAssembler:
    """CAMERA_FRAME chunks -> complete JPEGs. A frame with a missing chunk is dropped when a newer one starts."""

    def __init__(self):
        self.cur, self.buf, self.got = None, None, 0

    def feed(self, payload):
        fid, ms, w, h, fmt, _, total, off = struct.unpack_from(CAM_HDR, payload)
        data = payload[struct.calcsize(CAM_HDR):]
        if fid != self.cur:
            self.cur, self.buf, self.got = fid, bytearray(total), 0
        self.buf[off:off + len(data)] = data
        self.got += len(data)
        if self.got >= total:
            out = {"frame_id": fid, "robot_ms": ms, "w": w, "h": h, "format": fmt, "data": bytes(self.buf)}
            self.cur = None
            return out
        return None


def calib_report_parse(payload):
    return [dict(zip(("zero_ticks", "sign", "ticks_per_rev"), struct.unpack_from("<hbH", payload, 5 * i)))
            for i in range(len(payload) // 5)]
