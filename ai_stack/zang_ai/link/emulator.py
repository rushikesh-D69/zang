"""A software Zang that speaks ZLink over UDP, for developing the studio and host code without hardware.

    .venv/bin/python -m zang_ai.link.emulator [--port 4210] [--physics]

Behaves like the firmware contract (firmware/docs/ZLINK.md v1.1): HELLO/PING, JOINT_TARGETS, TRAJ_CHUNK streams on
a stream clock, CMD (hold, e-stop, torque on, go home, calibration, camera, mic, volume, brightness), last-line joint
limit + rate limit, a 500 ms stream watchdog, STATE at 50 Hz (with the v1.1 tail), FACE_SET/PARAMS/TRACK and
DISPLAY_TEXT/MODE state, DISPLAY_IMAGE over TCP 4211 (saved to out/emulator_display.jpg), and a synthetic camera
(CAMERA_FRAME JPEGs with a moving test pattern). The servos are a first-order lag (kinematic), or the MuJoCo twin with --physics.
"""
import argparse
import socket
import struct
import threading
import time

import numpy as np

from . import zlink as Z

RATE_HZ = 50
WATCHDOG_S = 0.5


class Emulator:
    def __init__(self, robot, port=4210, physics=False, host="127.0.0.1"):
        self.r = robot
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.settimeout(0.005)
        self.peer = None
        self.q_cmd = robot.home.copy()
        self.q = robot.home.copy()
        self.torque, self.estop = True, False
        self.frames = []          # [(t_stream_s, q)] queued for the active stream
        self.stream = None        # (id, wall t0)
        self.last_rx = 0.0
        self.move = None          # (t0, t1, q_from, q_to) for JOINT_TARGETS / GO_HOME
        self.goal = robot.home.copy()  # where the joints stay once a move / stream has finished
        self.seq = 0
        self.t0 = time.perf_counter()
        self.emotion, self.display_mode, self.face_track = 0, Z.DISPLAY_FACE, []
        self.display_text = ""
        self.cam = None            # (fps, size, quality) while streaming
        self._cam_next, self._cam_id = 0.0, 0
        self._img = {}             # image_id -> bytearray being assembled
        self.last_image = None
        self.tcp_port = port + 1
        threading.Thread(target=self._tcp_server, args=(host,), daemon=True).start()
        self.twin = None
        if physics:
            from ..sim.twin import PhysicsTwin
            self.twin = PhysicsTwin(robot)
            self.twin.reset()

    def now_ms(self):
        return int((time.perf_counter() - self.t0) * 1000)

    def send(self, typ, payload):
        if self.peer:
            self.seq = (self.seq + 1) & 0xFFFF
            self.sock.sendto(Z.frame(typ, payload, self.seq), self.peer)

    def handle(self, typ, seq, p):
        now = time.perf_counter()
        if typ == Z.HELLO:
            self.send(Z.HELLO_ACK, Z.hello_ack(self.r.n, 0x0001 | Z.FW_V11, "zang-emu"))
        elif typ == Z.FACE_SET:
            self.emotion = p[0]
            self.display_mode = Z.DISPLAY_FACE
        elif typ == Z.FACE_PARAMS:
            self.display_mode = Z.DISPLAY_FACE
        elif typ == Z.FACE_TRACK:
            sid, n = struct.unpack_from("<IB", p)
            self.face_track = [struct.unpack_from("<IBBH", p, 5 + 8 * k) for k in range(n)]
        elif typ == Z.DISPLAY_TEXT:
            self.display_text = p[7:].decode("utf-8", "replace")
            self.display_mode = Z.DISPLAY_TXT
        elif typ == Z.DISPLAY_MODE:
            self.display_mode = p[0]
        elif typ == Z.PING:
            self.send(Z.PONG, Z.pong(struct.unpack("<I", p)[0], self.now_ms() * 1000))
        elif typ == Z.JOINT_TARGETS:
            _, mask, move_ms = struct.unpack_from("<IHH", p)
            q = Z.cdeg_to_rad(np.frombuffer(p, "<i2", Z.N_JOINTS, 8))
            tgt = self.q_cmd.copy()
            sel = [(mask >> i) & 1 == 1 for i in range(Z.N_JOINTS)]
            tgt[sel] = q[sel]
            self.stream, self.frames = None, []
            self.move = (now, now + max(move_ms, 1) / 1000, self.q_cmd.copy(), tgt)
        elif typ == Z.STREAM_START:
            sid = struct.unpack_from("<I", p)[0]
            self.stream, self.frames, self.move = (sid, now), [], None
        elif typ == Z.TRAJ_CHUNK and self.stream:
            sid, t0_ms, _interp, n = struct.unpack_from("<IIBB", p)
            if sid != self.stream[0]:
                return
            t = t0_ms
            for k in range(n):
                off = 10 + k * 32
                t += struct.unpack_from("<H", p, off)[0]
                self.frames.append((t / 1000.0, Z.cdeg_to_rad(np.frombuffer(p, "<i2", Z.N_JOINTS, off + 2))))
            self.last_rx = now
        elif typ == Z.STREAM_END:
            pass  # playback finishes the queued frames, then holds
        elif typ == Z.CMD:
            code = p[0]
            if code == Z.CMD_ESTOP:
                self.estop, self.torque, self.stream, self.frames = True, False, None, []
                self.goal = self.q_cmd.copy()
            elif code == Z.CMD_TORQUE_ON:
                self.estop, self.torque = False, True
            elif code == Z.CMD_HOLD:
                self.stream, self.frames, self.move = None, [], None
                self.goal = self.q_cmd.copy()
            elif code == Z.CMD_CALIB_GET:
                self.send(Z.CALIB_REPORT, b"".join(struct.pack("<hbH", 2048, 1, 4096) for _ in range(self.r.n)))
            elif code == Z.CMD_CAMERA_STREAM:
                on, size, fps, q = (list(p[1:5]) + [0, 0, 10, 60])[:4]
                self.cam = (max(1, fps), size, q) if on else None
            elif code == Z.CMD_SNAPSHOT:
                self._send_camera(60)
            elif code == Z.CMD_GO_HOME:
                ms = struct.unpack_from("<H", p, 1)[0] if len(p) >= 3 else 1500
                self.stream, self.frames = None, []
                self.move = (now, now + ms / 1000, self.q_cmd.copy(), self.r.home.copy())
            self.send(Z.ACK, struct.pack("<HB", seq, 0))

    def step(self, dt):
        now = time.perf_counter()
        target = self.goal
        if self.estop or not self.torque:
            pass
        elif self.stream and self.frames:
            ts = now - self.stream[1]
            times = np.array([f[0] for f in self.frames])
            if ts >= times[-1]:
                target = self.goal = self.frames[-1][1]
                if now - self.last_rx > WATCHDOG_S:  # stream ended or starved: hold
                    self.stream, self.frames = None, []
            else:
                i = int(np.searchsorted(times, ts))
                if i == 0:
                    target = self.frames[0][1] if ts >= times[0] else self.q_cmd
                else:
                    (ta, qa), (tb, qb) = self.frames[i - 1], self.frames[i]
                    target = qa + (qb - qa) * (ts - ta) / max(tb - ta, 1e-6)
        elif self.move:
            t0, t1, qa, qb = self.move
            u = min(1.0, (now - t0) / (t1 - t0))
            target = qa + (qb - qa) * u
            if u >= 1:
                self.move, self.goal = None, qb.copy()
        # last line of defence: joint limits + rate limit (firmware does the same)
        target = np.clip(target, self.r.lower, self.r.upper)
        step = self.r.vmax * dt
        self.q_cmd = self.q_cmd + np.clip(target - self.q_cmd, -step, step)
        if self.twin is not None:
            d = self.twin.d
            if self.torque and not self.estop:
                d.ctrl[self.r.act] = self.q_cmd
                self.twin.m.actuator_gainprm[self.r.act, 0] = 8.0
            else:
                self.twin.m.actuator_gainprm[self.r.act, 0] = 0.0  # torque off: limp
            import mujoco
            for _ in range(int(dt / self.twin.m.opt.timestep)):
                mujoco.mj_step(self.twin.m, d)
            self.q = d.qpos[self.r.qadr].copy()
        elif self.torque and not self.estop:
            self.q += (self.q_cmd - self.q) * min(1.0, dt / 0.05)

    def state(self):
        flags = (self.torque << 0) | (self.estop << 1) | (bool(self.stream) << 2) | (1 << 5)
        buffered, pos = 0, 0
        if self.stream and self.frames:
            ts = time.perf_counter() - self.stream[1]
            pos = int(ts * 1000)
            buffered = int(max(0.0, self.frames[-1][0] - ts) * 1000)
        # face track on the stream clock drives the reported emotion (the real robot renders it)
        if self.stream and self.face_track:
            for t_ms, emo, _, _ in self.face_track:
                if t_ms <= pos:
                    self.emotion = emo
        rpy = np.full(3, np.nan)  # board v0.0.1 has no IMU
        return Z.State(self.now_ms(), flags, self.q, self.q_cmd, rpy, np.zeros(3), np.zeros(3),
                       7400, 35, 0, min(buffered, 65535), pos, self.display_mode, self.emotion).pack()

    # ---- synthetic camera + bulk image channel -------------------------------------------------------------------
    def _send_camera(self, quality):
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return
        w, h = (320, 240) if not self.cam or self.cam[1] == 0 else (640, 480)
        t = time.perf_counter() - self.t0
        im = Image.new("RGB", (w, h))
        px = im.load()
        for yy in range(0, h, 4):
            for xx in range(0, w, 4):
                c = (int(128 + 127 * np.sin(xx / 30 + t)), int(128 + 127 * np.sin(yy / 25 - t)), 160)
                for dy in range(4):
                    for dx in range(4):
                        px[xx + dx, yy + dy] = c
        d = ImageDraw.Draw(im)
        d.text((10, 10), f"ZANG EMULATOR  t={t:6.2f}s  frame {self._cam_id}", fill=(255, 255, 255))
        import io
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=int(quality))
        self._cam_id += 1
        for c in Z.camera_frame_chunks(self._cam_id, self.now_ms(), w, h, buf.getvalue()):
            self.send(Z.CAMERA_FRAME, c)

    def _tcp_server(self, host):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind((host, self.tcp_port))
        except OSError:
            return
        srv.listen(2)
        while True:
            conn, _ = srv.accept()
            threading.Thread(target=self._tcp_client, args=(conn,), daemon=True).start()

    def _tcp_client(self, conn):
        parser = Z.StreamParser()
        with conn:
            while True:
                data = conn.recv(65536)
                if not data:
                    return
                for typ, seq, p in parser.feed(data):
                    if typ != Z.DISPLAY_IMAGE:
                        continue
                    image_id, fmt, flags, total, off = struct.unpack_from("<HBBII", p)
                    buf = self._img.setdefault(image_id, bytearray(total))
                    buf[off:off + len(p) - 12] = p[12:]
                    if off + len(p) - 12 >= total:
                        self.last_image = bytes(self._img.pop(image_id))
                        try:
                            from ..paths import OUT
                            with open(f"{OUT}/emulator_display.jpg", "wb") as fh:
                                fh.write(self.last_image)
                        except OSError:
                            pass
                        if flags & 1:
                            self.display_mode = Z.DISPLAY_IMG
                        self.send(Z.EVENT, struct.pack("<BHB", Z.EV_IMAGE_DONE, image_id, 0))
                        self.send(Z.ACK, struct.pack("<HB", seq, 0))

    def serve(self, stop=None):
        stop = stop or threading.Event()
        dt = 1.0 / RATE_HZ
        next_t = time.perf_counter()
        while not stop.is_set():
            while True:
                try:
                    raw, addr = self.sock.recvfrom(2048)
                except (socket.timeout, BlockingIOError):
                    break
                except OSError:
                    return
                self.peer = addr
                try:
                    self.handle(*Z.parse(raw))
                except Z.ZLinkError:
                    pass
            if self.cam and time.perf_counter() >= self._cam_next:
                self._cam_next = time.perf_counter() + 1.0 / self.cam[0]
                self._send_camera(self.cam[2])
            if time.perf_counter() >= next_t:
                self.step(dt)
                self.send(Z.STATE, self.state())
                next_t += dt


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=4210)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--physics", action="store_true", help="MuJoCo twin instead of a kinematic servo lag")
    a = ap.parse_args()
    from ..robot import RobotModel
    emu = Emulator(RobotModel(), a.port, a.physics, a.host)
    print(f"zang emulator on udp://{a.host}:{a.port} ({'physics' if a.physics else 'kinematic'})")
    emu.serve()


if __name__ == "__main__":
    main()
