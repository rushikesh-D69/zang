"""Host side of ZLink over UDP: connect, clock sync, direct targets, streamed trajectories, state, e-stop.

    link = RobotLink("zang.local")        # or the IP; the emulator listens on 127.0.0.1
    link.connect()
    link.play(t, Q)                       # streams TRAJ_CHUNKs ahead of the robot's playback clock
    link.state                            # latest zlink.State (50 Hz)
    link.estop()
"""
import socket
import threading
import time

import numpy as np

from . import zlink as Z

PORT = 4210
TCP_PORT = 4211
LOOKAHEAD_S = 0.6  # how far ahead of playback chunks are sent (covers Wi-Fi jitter; robot buffers it)


class RobotLink:
    def __init__(self, host, port=PORT, on_state=None, on_event=None):
        self.addr = (socket.gethostbyname(host), port)
        self.on_state, self.on_event = on_state, on_event
        self.sock = None
        self.seq = 0
        self.state = None
        self.info = None
        self.rtt_ms = None
        self._acks = {}
        self._stop = threading.Event()
        self._play = None
        self._lock = threading.Lock()
        self._stream_id = int(time.time()) & 0xFFFF
        self.camera = None            # latest complete CAMERA_FRAME {"data": jpeg bytes, "w", "h", "robot_ms", ...}
        self._cam = Z.FrameAssembler()
        self.events = []              # recent EVENTs (kind, data)
        self.calib = None
        self._tcp = None

    # ---- connection -------------------------------------------------------------------------------------
    def connect(self, timeout=1.5):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.2)
        self._stop.clear()
        threading.Thread(target=self._rx, daemon=True, name="zlink-rx").start()
        t_end = time.time() + timeout
        while self.info is None and time.time() < t_end:
            self._send(Z.HELLO, Z.hello(int(time.time() * 1000)))
            time.sleep(0.15)
        if self.info is None:
            self.close()
            raise TimeoutError(f"no HELLO_ACK from {self.addr[0]}:{self.addr[1]}")
        self.sync_clock()
        return self.info

    def close(self):
        self._stop.set()
        if self._tcp:
            try:
                self._tcp.close()
            except OSError:
                pass
            self._tcp = None
        if self._play:
            self._play.set()
        if self.sock:
            self.sock.close()
            self.sock = None

    def sync_clock(self, n=5):
        """RTT from PING/PONG (the robot's clock offset is not needed: streams use relative time)."""
        rtts = []
        for _ in range(n):
            t0 = time.perf_counter()
            self._pong = None
            self._send(Z.PING, Z.ping(int(t0 * 1e6)))
            t_end = time.time() + 0.3
            while self._pong is None and time.time() < t_end:
                time.sleep(0.002)
            if self._pong is not None:
                rtts.append((time.perf_counter() - t0) * 1000)
        self.rtt_ms = round(float(np.median(rtts)), 2) if rtts else None
        return self.rtt_ms

    def _send(self, typ, payload):
        with self._lock:
            self.seq = (self.seq + 1) & 0xFFFF
            if self.sock:
                self.sock.sendto(Z.frame(typ, payload, self.seq), self.addr)
            return self.seq

    def _rx(self):
        while not self._stop.is_set():
            try:
                raw, _ = self.sock.recvfrom(2048)
                typ, seq, payload = Z.parse(raw)
            except (socket.timeout, OSError, Z.ZLinkError):
                continue
            if typ == Z.STATE:
                self.state = Z.State.unpack(payload)
                if self.on_state:
                    self.on_state(self.state)
            elif typ == Z.HELLO_ACK:
                ver, n, fw, name = __import__("struct").unpack("<BBH16s", payload)
                self.info = {"proto": ver, "n_joints": n, "fw_ver": fw, "name": name.rstrip(b"\0").decode()}
            elif typ == Z.PONG:
                self._pong = payload
            elif typ == Z.ACK:
                self._acks[int.from_bytes(payload[:2], "little")] = payload[2]
            elif typ == Z.CAMERA_FRAME:
                f = self._cam.feed(payload)
                if f:
                    self.camera = f
            elif typ == Z.CALIB_REPORT:
                self.calib = Z.calib_report_parse(payload)
            elif typ == Z.EVENT:
                self.events = (self.events + [(payload[0], payload[1:])])[-50:]
                if self.on_event:
                    self.on_event(payload)

    # ---- commands -----------------------------------------------------------------------------------------
    def targets(self, q, move_ms=40):
        return self._send(Z.JOINT_TARGETS, Z.joint_targets(q, 0, 0x7FFF, move_ms))

    def hold(self):
        self.stop_play()
        return self._send(Z.CMD, Z.cmd(Z.CMD_HOLD))

    def estop(self):
        self.stop_play()
        return self._send(Z.CMD, Z.cmd(Z.CMD_ESTOP))

    def torque_on(self):
        return self._send(Z.CMD, Z.cmd(Z.CMD_TORQUE_ON))

    def go_home(self, ms=1500):
        return self._send(Z.CMD, Z.cmd(Z.CMD_GO_HOME, int(ms).to_bytes(2, "little")))

    # ---- streaming ------------------------------------------------------------------------------------------
    def play(self, t, Q, block=False, face=None):
        """Stream a trajectory (t seconds from start, Q rad). Chunks go out LOOKAHEAD_S ahead of playback.
        face: optional face keys [{"t", "id", "intensity", "blend_ms"}] sent as FACE_TRACK on the same clock."""
        self.stop_play()
        self._stream_id = (self._stream_id + 1) & 0xFFFFFFFF
        sid = self._stream_id
        chunks = Z.chunk_trajectory(t, Q, sid)
        face_msgs = []
        if face:
            keys = [(int(round(k["t"] * 1000)), k["id"], int(round(k["intensity"] * 255)), k["blend_ms"]) for k in face]
            face_msgs = [Z.face_track(sid, keys[i:i + 100]) for i in range(0, len(keys), 100)]
        ends = [float(t[min((k + 1) * Z.MAX_CHUNK_FRAMES, len(t)) - 1]) for k in range(len(chunks))]
        stop = self._play = threading.Event()

        def run():
            self._send(Z.STREAM_START, Z.stream_start(sid, 0))
            for m in face_msgs:
                self._send(Z.FACE_TRACK, m)
            t0 = time.perf_counter()
            for k, payload in enumerate(chunks):
                start_of_chunk = 0.0 if k == 0 else ends[k - 1]
                while not stop.is_set() and time.perf_counter() - t0 < start_of_chunk - LOOKAHEAD_S:
                    time.sleep(0.005)
                if stop.is_set():
                    return
                self._send(Z.TRAJ_CHUNK, payload)
            while not stop.is_set() and time.perf_counter() - t0 < ends[-1]:
                time.sleep(0.01)
            self._send(Z.STREAM_END, Z.stream_end(sid))

        th = threading.Thread(target=run, daemon=True, name="zlink-play")
        th.start()
        if block:
            th.join()
        return sid

    def stop_play(self):
        if self._play:
            self._play.set()
            self._play = None

    # ---- v1.1: face, display, camera, audio settings, calibration ----------------------------------------------
    def face(self, emotion_id, intensity=1.0, blend_ms=250, gaze=(0, 0), talking=False, hold=False):
        return self._send(Z.FACE_SET, Z.face_set(emotion_id, int(round(intensity * 255)), blend_ms,
                                                 gaze[0], gaze[1], talking=talking, hold=hold))

    def face_params(self, params_by_id, blend_ms=120):
        return self._send(Z.FACE_PARAMS, Z.face_params(params_by_id, blend_ms))

    def display_text(self, text, size=2, fg=0xFFFF, bg=0x0000, duration_ms=0):
        return self._send(Z.DISPLAY_TEXT, Z.display_text(text, size, fg, bg, duration_ms))

    def display_mode(self, mode, image_id=0):
        return self._send(Z.DISPLAY_MODE, Z.display_mode(mode, image_id))

    def display_image(self, image_id, jpeg: bytes, show=True, store=False, timeout=3.0):
        """Bulk image over TCP 4211 (reliable). Returns when every chunk is written."""
        if self._tcp is None:
            self._tcp = socket.create_connection((self.addr[0], self.addr[1] + 1), timeout=timeout)  # spec: TCP = UDP + 1
        for payload in Z.display_image_chunks(image_id, jpeg, Z.IMG_JPEG, show, store):
            with self._lock:
                self.seq = (self.seq + 1) & 0xFFFF
                self._tcp.sendall(Z.frame(Z.DISPLAY_IMAGE, payload, self.seq))
        return len(jpeg)

    def camera_stream(self, on=True, size=0, fps=10, quality=60):
        return self._send(Z.CMD, Z.cmd(Z.CMD_CAMERA_STREAM, bytes([int(on), size, fps, quality])))

    def snapshot(self):
        return self._send(Z.CMD, Z.cmd(Z.CMD_SNAPSHOT))

    def mic_stream(self, on=True, codec=Z.CODEC_PCM16_16K):
        return self._send(Z.CMD, Z.cmd(Z.CMD_MIC_STREAM, bytes([int(on), codec])))

    def set_volume(self, percent):
        return self._send(Z.CMD, Z.cmd(Z.CMD_SET_VOLUME, bytes([int(percent)])))

    def set_brightness(self, percent):
        return self._send(Z.CMD, Z.cmd(Z.CMD_SET_BRIGHTNESS, bytes([int(percent)])))

    def calib_get(self):
        return self._send(Z.CMD, Z.cmd(Z.CMD_CALIB_GET))
