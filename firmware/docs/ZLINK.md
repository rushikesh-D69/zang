# ZLink v1.1: host ↔ robot protocol

ZLink is the **only** contract between `ai_stack/` (PC/cloud brain, studio, RL policy runner) and `firmware/zang_p4/`.
The C implementation is `firmware/zang_p4/components/zlink/`. The Python implementation is `ai_stack/zang_ai/link/zlink.py`.
Both must pass the golden vectors in `firmware/zang_p4/components/zlink/test/zlink_vectors.json`. That file is generated
by `ai_stack/tools/gen_zlink_vectors.py`, and the firmware host test checks it byte for byte.

## Transports

| Transport | Use | Notes |
|---|---|---|
| UDP, robot port **4210** | real-time: targets, trajectory chunks, state, ping | one frame per datagram, payload ≤ 1400 B |
| TCP, robot port **4211** | bulk: file upload/download, audio streams | frames back to back; resync on magic |
| USB-CDC / UART | bench, no Wi-Fi | same framing; resync on magic + CRC |

Wi-Fi itself is provided by the ESP32-C2 co-processor (ESP-Hosted). The P4 application only sees normal lwIP sockets.

## Frame (all fields little-endian)

```
off  size  field
0    2     magic        0x5A 0x4C  ("ZL")
2    1     version      1
3    1     type         message type (below)
4    2     seq          sender's sequence counter (wraps)
6    2     len          payload length N, 0..1400
8    N     payload
8+N  2     crc16        CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflect, xorout 0) over bytes 2 .. 8+N-1
```

## Units

- Joint angle: `int16` **centidegrees**, model joint frame (MJCF/URDF), ordered by `index` in `ai_stack/config/robot.json`
  (15 joints). The firmware converts to servo ticks with its calibration table. The host never sends ticks.
- Time: `uint32` milliseconds. `t_robot_ms` is the robot's monotonic clock. Stream-relative times count from the
  stream's start (see STREAM_START).
- Joint mask: `uint16`, bit i = joint index i is commanded. Unmasked joints hold.

## Messages

| type | name | dir | payload |
|---|---|---|---|
| 0x01 | HELLO | H→R | `u32 host_ms` |
| 0x02 | HELLO_ACK | R→H | `u8 proto_ver, u8 n_joints, u16 fw_ver, char name[16]` |
| 0x03 | PING | H→R | `u32 host_us` |
| 0x04 | PONG | R→H | `u32 host_us (echo), u32 robot_us` |
| 0x10 | JOINT_TARGETS | H→R | `u32 t_robot_ms (0 = now), u16 mask, u16 move_ms, i16 q[15]` |
| 0x11 | TRAJ_CHUNK | H→R | `u32 stream_id, u32 t0_ms, u8 interp, u8 n_frames, n_frames × { u16 dt_ms, i16 q[15] }` |
| 0x12 | STREAM_START | H→R | `u32 stream_id, u8 clock (0 = robot timer, 1 = audio clock), u8 flags` |
| 0x13 | STREAM_END | H→R | `u32 stream_id` |
| 0x20 | AUDIO_CHUNK | H→R | `u32 stream_id, u32 sample_offset, u8 codec, u8 rsv, data…` |
| 0x21 | MIC_CHUNK | R→H | `u32 robot_ms, u8 codec, u8 rsv, data…` |
| 0x30 | STATE | R→H | see below |
| 0x40 | CMD | H→R | `u8 cmd, data…` |
| 0x41 | ACK | R→H | `u16 seq_acked, u8 status` |
| 0x50 | LOG | R→H | `u8 level, char text[]` |
| 0x60 | EVENT | R→H | `u8 kind, data…` (wake word, face seen, touch, fall, button) |

`interp`: 0 = linear (ramp over dt), 1 = cubic Hermite (velocities estimated from neighbours).
TRAJ_CHUNK frames: frame k is reached at `t0_ms + Σ dt_ms[0..k]` on the stream clock. With max 40 frames per chunk,
the payload is 10 + 40 × 32 = 1290 B.
`codec`: 0 = PCM s16le 16 kHz mono, 1 = PCM s16le 24 kHz mono, 2 = Opus.

### STATE (R→H, 50 Hz default)

```
u32 t_robot_ms
u16 flags            bit0 torque_on, bit1 estop, bit2 streaming, bit3 fault, bit4 fallen, bit5 calibrated
i16 q_meas[15]       measured joint angles (centideg), 0x7FFF = not read this cycle
i16 q_cmd[15]        last commanded (after the firmware limiter)
i16 rpy[3]           IMU roll pitch yaw, centideg (0x7FFF if no IMU)
i16 gyro[3]          0.1 deg/s
i16 acc[3]           mg
u16 vbat_mv
u8  temp_max_c
u8  fault_servo_id   0 = none
```

### CMD codes

| code | name | data |
|---|---|---|
| 0 | HOLD | stop streams, hold current pose |
| 1 | ESTOP | torque off immediately (latched until TORQUE_ON) |
| 2 | TORQUE_ON | — |
| 3 | GO_HOME | `u16 move_ms` |
| 4 | PLAY_FILE | `char path[]` (SD card, `.zact`) |
| 5 | SET_RATE | `u8 state_hz` |
| 6 | CALIB_SET | `u8 index, i16 zero_ticks, i8 sign` |
| 7 | CALIB_SAVE | — |

## Safety rule

The host's safety stage (ai_stack `safety/`) is the primary check. The firmware is the **last line**. It clamps every
target to the per-joint limits and rate limits in `robot_joints.h`, drops stale frames (`t` more than 200 ms in the
past), and goes to HOLD if no frame arrives for 500 ms while streaming.

## v1.1 additions (2026-11-17)

These additions are backward compatible: the frame format and version byte are unchanged, receivers ignore unknown
`type`s, and STATE only grows at its tail (receivers accept ≥ 88 bytes). Firmware reports `proto_ver = 1` in HELLO_ACK
and `fw_ver` bit 15 = "v1.1 messages supported".

### Data path through the ICs

```
 PC / studio ──Wi-Fi (UDP 4210 realtime, TCP 4211 bulk)──► ESP32-C2 ──UART 3 Mbaud (ESP-Hosted, lwIP on the P4)──► ESP32-P4
                                                                                                                     │
   servos   ◄── UART half-duplex (GPIO47/48, DS protocol) ── motion player, limiter ◄── TRAJ_CHUNK / JOINT_TARGETS ───┤
   TFT      ◄── SPI 40 MHz (GPIO19-23), face renderer + framebuffer ◄── FACE_SET / FACE_PARAMS / FACE_TRACK / DISPLAY_* ┤
   speaker  ◄── I2S (GPIO30/31/33, NS4168) ◄── AUDIO_CHUNK (stream clock) / PLAY_FILE (SD) ─────────────────────────┤
   mics     ──► PDM2PCM (GPIO45/46) ──► MIC_CHUNK (CMD MIC_STREAM) ──────────────────────────────────────────────────►┤
   camera   ──► DVP 8-bit (GPIO5-16), SCCB I2C (17/18) ──► JPEG ──► CAMERA_FRAME (CMD CAMERA_STREAM) ───────────────►┤
   SD card  ◄─► SDMMC slot 1 (GPIO39-44): .zact actions, audio, stored images
```
Bandwidth budget (UART link ≈ 250–300 kB/s): STATE 50 Hz ≈ 5 kB/s; TRAJ 50 Hz ≈ 1.6 kB/s; speaker PCM 16 kHz
32 kB/s; mic PCM 16 kHz 32 kB/s; a face change is ~10 B. A full 240×240 image is 115 kB as RGB565, so images go as
**JPEG** (≈ 8–15 kB). Camera QVGA JPEG at 10 fps ≈ 100 kB/s, so the camera is the one stream that must be rationed.

### New messages

| type | name | dir | transport | payload |
|---|---|---|---|---|
| 0x22 | CAMERA_FRAME | R→H | UDP | `u32 frame_id, u32 robot_ms, u16 w, u16 h, u8 format (1 = JPEG), u8 rsv, u32 total_len, u32 offset, data…` (chunks ≤ 1360 B; a frame with a missing chunk is dropped) |
| 0x42 | CALIB_REPORT | R→H | UDP | `15 × { i16 zero_ticks, i8 sign, u16 ticks_per_rev }` (answer to CMD CALIB_GET) |
| 0x70 | FACE_SET | H→R | UDP | `u8 emotion, u8 intensity (0–255), u16 blend_ms, i8 gaze_x, i8 gaze_y, u8 flags` (bit0 talking, bit1 blink now, bit2 hold, i.e. no idle animation) |
| 0x71 | FACE_PARAMS | H→R | UDP | `u16 blend_ms, u8 n, n × { u8 param_id, f32 value }` (IEEE-754 little-endian, the parameter's own units; ids from `faces.json`) |
| 0x72 | FACE_TRACK | H→R | UDP | `u32 stream_id, u8 n, n × { u32 t_ms, u8 emotion, u8 intensity, u16 blend_ms }` (face keys on the motion stream's clock) |
| 0x73 | DISPLAY_IMAGE | H→R | TCP | `u16 image_id, u8 format (0 RGB565 LE, 1 JPEG), u8 flags, u32 total_len, u32 offset, data…` (flags: bit0 show when complete, bit1 store on SD as `/img/<id>.jpg`); ACK when complete |
| 0x74 | DISPLAY_TEXT | H→R | UDP | `u8 size (1–4), u16 fg_rgb565, u16 bg_rgb565, u16 duration_ms (0 = until changed), char text[]` (UTF-8, ASCII rendered) |
| 0x75 | DISPLAY_MODE | H→R | UDP | `u8 mode (0 face, 1 image, 2 text, 3 camera preview, 4 status), u16 image_id (mode 1)` |

`faces.json` (in `ai_stack/config/`) is the canonical face library. Emotion ids are list indices; never reorder them,
only append. The P4 renders faces itself with the same SDF algorithm as `ai_stack/zang_ai/face/render.py`, so faces
cost bytes on the link, not pixels. `intensity` blends from neutral (0) to the full emotion (255).
Talking: the host sets flag bit0 while audio plays, or sends a FACE_TRACK; the robot animates the mouth itself.

### New CMD codes

| code | name | data |
|---|---|---|
| 8 | CALIB_GET | — (→ CALIB_REPORT) |
| 9 | MIC_STREAM | `u8 on, u8 codec` |
| 10 | CAMERA_STREAM | `u8 on, u8 size (0 QVGA, 1 VGA), u8 fps, u8 jpeg_quality` |
| 11 | SET_VOLUME | `u8 percent` |
| 12 | SET_BRIGHTNESS | `u8 percent` |
| 13 | SNAPSHOT | — (one CAMERA_FRAME even when streaming is off) |

CALIB_SET (6) gains an optional trailing `u16 ticks_per_rev` (absent = unchanged).

### STATE tail (v1.1)

```
… (88 bytes as above)
u16 stream_buffered_ms   trajectory still queued ahead of the playhead (flow control: the host keeps it at 300–800 ms)
u32 stream_pos_ms        playhead position on the current stream clock (0 if none)
u8  display_mode         as DISPLAY_MODE
u8  emotion              current face emotion id
```
STATE grows to 96 bytes.

### EVENT kinds

| kind | name | data |
|---|---|---|
| 0x01 | STATE_CHANGED | `u8 app_state` |
| 0x02 | STREAM_DONE | — |
| 0x03 | FILE_DONE | — |
| 0x04 | WATCHDOG | — (stream starved → HOLD) |
| 0x05 | SERVO_FAULT | `u8 servo_id` |
| 0x06 | ESTOP | — |
| 0x07 | TARGET_DONE | — |
| 0x10 | BUTTON | `u8 id, u8 pressed` |
| 0x11 | LOW_BATTERY | `u16 mv` |
| 0x12 | WAKE_WORD | `u8 id` (on-device wake word, later) |
| 0x13 | FACE_SEEN | `u8 n, n × { i16 x, i16 y, u16 w, u16 h }` (on-device vision, later) |
| 0x14 | IMAGE_DONE | `u16 image_id, u8 status` |

### Byte order

Everything in ZLink and in `.zact` is **little-endian**. This includes `.zact` header fields and frames.

## Action files on SD (`.zact`)

```
char  magic[4] = "ZACT"
u8    version = 1
u8    n_joints = 15
u16   flags            bit0 has_audio
u32   n_frames         (no 200-frame limit)
char  audio[32]        mp3/wav file name in the same folder, NUL-padded
n_frames × { u32 t_ms (absolute from start), i16 q[15] }
```
