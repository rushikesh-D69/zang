"""Action files for the robot's SD card (.zact, spec in firmware/docs/ZLINK.md) + audio side-car."""
import os
import struct

import numpy as np

from ..link.zlink import N_JOINTS, cdeg_to_rad, rad_to_cdeg

ZACT_HDR = "<4sBBHI32s"
HAS_AUDIO = 1


def write_zact(path, t, Q, audio_name=None):
    """t (N,) seconds from start, Q (N, 15) rad -> path. Audio file name (same folder) is optional."""
    Q = np.atleast_2d(Q)
    assert Q.shape[1] == N_JOINTS
    t_ms = np.round(np.asarray(t) * 1000).astype("<u4")
    name = (audio_name or "").encode()[:31]
    with open(path + ".tmp", "wb") as fh:
        fh.write(struct.pack(ZACT_HDR, b"ZACT", 1, N_JOINTS, HAS_AUDIO if audio_name else 0, len(Q), name))
        for tm, q in zip(t_ms, Q):
            fh.write(struct.pack("<I", int(tm)) + rad_to_cdeg(q).tobytes())
    os.replace(path + ".tmp", path)  # atomic: a half-written action never lands on the card
    return path


def read_zact(path):
    raw = open(path, "rb").read()
    magic, ver, nj, flags, n, audio = struct.unpack_from(ZACT_HDR, raw)
    if magic != b"ZACT" or ver != 1:
        raise ValueError(f"{path}: not a ZACT v1 file")
    off = struct.calcsize(ZACT_HDR)
    rec = np.dtype([("t", "<u4"), ("q", "<i2", (nj,))])
    data = np.frombuffer(raw, rec, count=n, offset=off)
    return {"t": data["t"] / 1000.0, "q": cdeg_to_rad(data["q"]),
            "audio": audio.rstrip(b"\0").decode() if flags & HAS_AUDIO else None}
