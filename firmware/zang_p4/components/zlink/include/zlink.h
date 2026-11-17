/*
 * zlink.h - ZLink v1.1 codec (pure C, no IDF, host-tested). Spec: firmware/docs/ZLINK.md (canonical).
 *
 * Layers:
 *   1. frame:   zlink_encode()/zlink_decode()   magic, version, type, seq, len, payload, CRC-16/CCITT-FALSE
 *   2. stream:  zlink_stream_*                  resync on magic for TCP / UART byte streams
 *   3. message: zlink_pack_<msg>()/zlink_unpack_<msg>()  explicit little-endian (de)serialisation, no packed
 *               structs, exact length checks.
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ZLINK_MAGIC0         0x5A
#define ZLINK_MAGIC1         0x4C
#define ZLINK_VERSION        1
#define ZLINK_HDR_LEN        8
#define ZLINK_CRC_LEN        2
#define ZLINK_MAX_PAYLOAD    1400
#define ZLINK_MAX_FRAME      (ZLINK_HDR_LEN + ZLINK_MAX_PAYLOAD + ZLINK_CRC_LEN)
#define ZLINK_N_JOINTS       15 /* fixed by protocol v1; firmware static_asserts ROBOT_N_JOINTS == this */
#define ZLINK_TRAJ_MAX_FRAMES 40
#define ZLINK_NAME_LEN       16
#define ZLINK_CDEG_NA        ((int16_t)0x7FFF)
#define ZLINK_UDP_PORT       4210
#define ZLINK_TCP_PORT       4211

typedef enum {
    ZLINK_HELLO = 0x01,
    ZLINK_HELLO_ACK = 0x02,
    ZLINK_PING = 0x03,
    ZLINK_PONG = 0x04,
    ZLINK_JOINT_TARGETS = 0x10,
    ZLINK_TRAJ_CHUNK = 0x11,
    ZLINK_STREAM_START = 0x12,
    ZLINK_STREAM_END = 0x13,
    ZLINK_AUDIO_CHUNK = 0x20,
    ZLINK_MIC_CHUNK = 0x21,
    ZLINK_CAMERA_FRAME = 0x22,   /* v1.1 */
    ZLINK_STATE = 0x30,
    ZLINK_CMD = 0x40,
    ZLINK_ACK = 0x41,
    ZLINK_CALIB_REPORT = 0x42,   /* v1.1 */
    ZLINK_LOG = 0x50,
    ZLINK_EVENT = 0x60,
    ZLINK_FACE_SET = 0x70,       /* v1.1 */
    ZLINK_FACE_PARAMS = 0x71,
    ZLINK_FACE_TRACK = 0x72,
    ZLINK_DISPLAY_IMAGE = 0x73,
    ZLINK_DISPLAY_TEXT = 0x74,
    ZLINK_DISPLAY_MODE = 0x75,
} zlink_type_t;

/* HELLO_ACK.fw_ver bit 15: the firmware supports the v1.1 messages. */
#define ZLINK_FW_V11_BIT 0x8000u

typedef enum {
    ZLINK_CMD_HOLD = 0,
    ZLINK_CMD_ESTOP = 1,
    ZLINK_CMD_TORQUE_ON = 2,
    ZLINK_CMD_GO_HOME = 3,
    ZLINK_CMD_PLAY_FILE = 4,
    ZLINK_CMD_SET_RATE = 5,
    ZLINK_CMD_CALIB_SET = 6,
    ZLINK_CMD_CALIB_SAVE = 7,
    ZLINK_CMD_CALIB_GET = 8,      /* v1.1 */
    ZLINK_CMD_MIC_STREAM = 9,
    ZLINK_CMD_CAMERA_STREAM = 10,
    ZLINK_CMD_SET_VOLUME = 11,
    ZLINK_CMD_SET_BRIGHTNESS = 12,
    ZLINK_CMD_SNAPSHOT = 13,
} zlink_cmd_code_t;

typedef enum {
    ZLINK_DISPLAY_FACE = 0,
    ZLINK_DISPLAY_IMAGE_MODE = 1,
    ZLINK_DISPLAY_TEXT_MODE = 2,
    ZLINK_DISPLAY_CAMERA = 3,
    ZLINK_DISPLAY_STATUS = 4,
} zlink_display_mode_t;

/* FACE_SET.flags */
#define ZLINK_FACE_TALKING (1u << 0)
#define ZLINK_FACE_BLINK   (1u << 1)
#define ZLINK_FACE_HOLD    (1u << 2)
/* DISPLAY_IMAGE.flags */
#define ZLINK_IMAGE_SHOW  (1u << 0)
#define ZLINK_IMAGE_STORE (1u << 1)
/* EVENT kinds (ZLINK.md v1.1) */
typedef enum {
    ZLINK_EV_STATE_CHANGED = 0x01,
    ZLINK_EV_STREAM_DONE = 0x02,
    ZLINK_EV_FILE_DONE = 0x03,
    ZLINK_EV_WATCHDOG = 0x04,
    ZLINK_EV_SERVO_FAULT = 0x05,
    ZLINK_EV_ESTOP = 0x06,
    ZLINK_EV_TARGET_DONE = 0x07,
    ZLINK_EV_BUTTON = 0x10,
    ZLINK_EV_LOW_BATTERY = 0x11,
    ZLINK_EV_WAKE_WORD = 0x12,
    ZLINK_EV_FACE_SEEN = 0x13,
    ZLINK_EV_IMAGE_DONE = 0x14,
} zlink_event_kind_t;

#define ZLINK_CAMERA_CHUNK_MAX 1360
#define ZLINK_FACE_MAX_PARAMS  32
#define ZLINK_FACE_MAX_KEYS    100 /* ai_stack face_track() sends up to 100 keys per message */

typedef enum { ZLINK_INTERP_LINEAR = 0, ZLINK_INTERP_CUBIC = 1 } zlink_interp_t;
typedef enum { ZLINK_CLOCK_ROBOT = 0, ZLINK_CLOCK_AUDIO = 1 } zlink_clock_t;
typedef enum { ZLINK_CODEC_PCM16_16K = 0, ZLINK_CODEC_PCM16_24K = 1, ZLINK_CODEC_OPUS = 2 } zlink_codec_t;

/* STATE.flags bits */
#define ZLINK_STATE_TORQUE_ON  (1u << 0)
#define ZLINK_STATE_ESTOP      (1u << 1)
#define ZLINK_STATE_STREAMING  (1u << 2)
#define ZLINK_STATE_FAULT      (1u << 3)
#define ZLINK_STATE_FALLEN     (1u << 4)
#define ZLINK_STATE_CALIBRATED (1u << 5)

typedef enum {
    ZLINK_OK = 0,
    ZLINK_ERR_ARG = -1,
    ZLINK_ERR_SHORT = -2,   /* need more bytes */
    ZLINK_ERR_MAGIC = -3,
    ZLINK_ERR_VERSION = -4,
    ZLINK_ERR_LEN = -5,     /* len field > ZLINK_MAX_PAYLOAD, or payload size wrong for the message */
    ZLINK_ERR_CRC = -6,
    ZLINK_ERR_NOSPACE = -7, /* output buffer too small */
    ZLINK_ERR_TYPE = -8,
    ZLINK_ERR_VALUE = -9,   /* field out of range (e.g. n_frames > 40) */
} zlink_status_t;

const char *zlink_status_str(zlink_status_t st);

/* ---- CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, xorout 0. check("123456789") = 0x29B1 ---- */
uint16_t zlink_crc16_update(uint16_t crc, const uint8_t *data, size_t n);
static inline uint16_t zlink_crc16(const uint8_t *data, size_t n) { return zlink_crc16_update(0xFFFF, data, n); }

/* ---- frame layer ---- */
typedef struct {
    uint8_t type;
    uint16_t seq;
    uint16_t len;
    const uint8_t *payload; /* points into the decoded buffer */
} zlink_frame_t;

zlink_status_t zlink_encode(uint8_t type, uint16_t seq, const uint8_t *payload, uint16_t len, uint8_t *out,
                            size_t cap, size_t *out_len);
/* Decode one frame at buf[0]. *consumed = frame size on success. ZLINK_ERR_SHORT if incomplete. */
zlink_status_t zlink_decode(const uint8_t *buf, size_t n, zlink_frame_t *frame, size_t *consumed);

/* ---- stream layer (TCP/UART): push bytes, pop frames; resyncs on magic after garbage or CRC errors ---- */
typedef struct {
    uint8_t buf[ZLINK_MAX_FRAME * 2];
    size_t n;
    size_t pending_drop; /* bytes of the last returned frame, dropped on the next call */
    uint32_t crc_errors;
    uint32_t dropped_bytes;
} zlink_stream_t;

void zlink_stream_init(zlink_stream_t *s);
/* Append up to n bytes; returns how many were taken (call zlink_stream_next until false, then push the rest). */
size_t zlink_stream_push(zlink_stream_t *s, const uint8_t *data, size_t n);
/* Next complete valid frame; its payload stays valid until the next push/next call. */
bool zlink_stream_next(zlink_stream_t *s, zlink_frame_t *frame);

/* ---- messages ---- */
typedef struct { uint32_t host_ms; } zlink_hello_t;
typedef struct {
    uint8_t proto_ver;
    uint8_t n_joints;
    uint16_t fw_ver;
    char name[ZLINK_NAME_LEN + 1]; /* NUL-terminated locally; NUL-padded to 16 on the wire */
} zlink_hello_ack_t;
typedef struct { uint32_t host_us; } zlink_ping_t;
typedef struct { uint32_t host_us, robot_us; } zlink_pong_t;
typedef struct {
    uint32_t t_robot_ms; /* 0 = now */
    uint16_t mask;
    uint16_t move_ms;
    int16_t q_cdeg[ZLINK_N_JOINTS];
} zlink_joint_targets_t;
typedef struct {
    uint16_t dt_ms;
    int16_t q_cdeg[ZLINK_N_JOINTS];
} zlink_traj_frame_t;
typedef struct {
    uint32_t stream_id;
    uint32_t t0_ms;
    uint8_t interp;
    uint8_t n_frames;
    zlink_traj_frame_t frames[ZLINK_TRAJ_MAX_FRAMES];
} zlink_traj_chunk_t;
typedef struct { uint32_t stream_id; uint8_t clock; uint8_t flags; } zlink_stream_start_t;
typedef struct { uint32_t stream_id; } zlink_stream_end_t;
typedef struct {
    uint32_t stream_id;
    uint32_t sample_offset;
    uint8_t codec;
    const uint8_t *data; /* not copied */
    uint16_t n_data;
} zlink_audio_chunk_t;
typedef struct {
    uint32_t robot_ms;
    uint8_t codec;
    const uint8_t *data;
    uint16_t n_data;
} zlink_mic_chunk_t;
typedef struct {
    uint32_t t_robot_ms;
    uint16_t flags;
    int16_t q_meas_cdeg[ZLINK_N_JOINTS];
    int16_t q_cmd_cdeg[ZLINK_N_JOINTS];
    int16_t rpy_cdeg[3];
    int16_t gyro_ddps[3];
    int16_t acc_mg[3];
    uint16_t vbat_mv;
    uint8_t temp_max_c;
    uint8_t fault_servo_id;
    /* v1.1 tail */
    uint16_t stream_buffered_ms;
    uint32_t stream_pos_ms;
    uint8_t display_mode;
    uint8_t emotion;
    /* Codec control, not on the wire: pack the 88-byte v1.0 layout (set by unpack when the tail is absent). */
    bool legacy_88;
} zlink_state_t;
#define ZLINK_STATE_PAYLOAD_LEN_V10 88
#define ZLINK_STATE_PAYLOAD_LEN     96
typedef struct {
    uint8_t cmd;
    const uint8_t *data; /* not copied */
    uint16_t n_data;
} zlink_cmd_t;
typedef struct {
    uint8_t index;
    int16_t zero_ticks;
    int8_t sign;
    bool has_ticks_per_rev; /* v1.1 optional trailing u16 */
    uint16_t ticks_per_rev;
} zlink_calib_set_t;
typedef struct { uint8_t on, codec; } zlink_mic_stream_t;
typedef struct { uint8_t on, size, fps, jpeg_quality; } zlink_camera_stream_t;
typedef struct { uint16_t seq_acked; uint8_t status; } zlink_ack_t;
typedef struct { uint8_t level; const char *text; uint16_t n_text; } zlink_log_t;
typedef struct { uint8_t kind; const uint8_t *data; uint16_t n_data; } zlink_event_t;

/* ---- v1.1 messages ---- */
typedef struct {
    uint32_t frame_id, robot_ms;
    uint16_t w, h;
    uint8_t format; /* 1 = JPEG */
    uint32_t total_len, offset;
    const uint8_t *data;
    uint16_t n_data;
} zlink_camera_frame_t;
typedef struct {
    struct {
        int16_t zero_ticks;
        int8_t sign;
        uint16_t ticks_per_rev;
    } joints[ZLINK_N_JOINTS];
} zlink_calib_report_t;
typedef struct {
    uint8_t emotion, intensity;
    uint16_t blend_ms;
    int8_t gaze_x, gaze_y;
    uint8_t flags;
} zlink_face_set_t;
typedef struct {
    uint16_t blend_ms;
    uint8_t n;
    struct {
        uint8_t id;
        float value;
    } params[ZLINK_FACE_MAX_PARAMS];
} zlink_face_params_t;
typedef struct {
    uint32_t stream_id;
    uint8_t n;
    struct {
        uint32_t t_ms;
        uint8_t emotion, intensity;
        uint16_t blend_ms;
    } keys[ZLINK_FACE_MAX_KEYS];
} zlink_face_track_t;
typedef struct {
    uint16_t image_id;
    uint8_t format; /* 0 RGB565 LE, 1 JPEG */
    uint8_t flags;
    uint32_t total_len, offset;
    const uint8_t *data;
    uint16_t n_data;
} zlink_display_image_t;
typedef struct {
    uint8_t size;
    uint16_t fg, bg; /* RGB565 */
    uint16_t duration_ms;
    const char *text; /* not NUL-terminated */
    uint16_t n_text;
} zlink_display_text_t;
typedef struct { uint8_t mode; uint16_t image_id; } zlink_display_mode_msg_t;

/* pack: write payload to p (cap bytes), set *len. unpack: parse exactly len bytes. */
#define ZLINK_DECLARE_MSG(name)                                                                               \
    zlink_status_t zlink_pack_##name(const zlink_##name##_t *m, uint8_t *p, size_t cap, uint16_t *len);       \
    zlink_status_t zlink_unpack_##name(const uint8_t *p, uint16_t len, zlink_##name##_t *m);
ZLINK_DECLARE_MSG(hello)
ZLINK_DECLARE_MSG(hello_ack)
ZLINK_DECLARE_MSG(ping)
ZLINK_DECLARE_MSG(pong)
ZLINK_DECLARE_MSG(joint_targets)
ZLINK_DECLARE_MSG(traj_chunk)
ZLINK_DECLARE_MSG(stream_start)
ZLINK_DECLARE_MSG(stream_end)
ZLINK_DECLARE_MSG(audio_chunk)
ZLINK_DECLARE_MSG(mic_chunk)
ZLINK_DECLARE_MSG(state)
ZLINK_DECLARE_MSG(cmd)
ZLINK_DECLARE_MSG(ack)
ZLINK_DECLARE_MSG(log)
ZLINK_DECLARE_MSG(event)
ZLINK_DECLARE_MSG(camera_frame)
ZLINK_DECLARE_MSG(calib_report)
ZLINK_DECLARE_MSG(face_set)
ZLINK_DECLARE_MSG(face_params)
ZLINK_DECLARE_MSG(face_track)
ZLINK_DECLARE_MSG(display_image)
ZLINK_DECLARE_MSG(display_text)
ZLINK_DECLARE_MSG(display_mode_msg)
#undef ZLINK_DECLARE_MSG

/* CMD data helpers */
zlink_status_t zlink_cmd_go_home_ms(const zlink_cmd_t *c, uint16_t *move_ms);
zlink_status_t zlink_cmd_set_rate_hz(const zlink_cmd_t *c, uint8_t *hz);
zlink_status_t zlink_cmd_calib_set(const zlink_cmd_t *c, zlink_calib_set_t *out);
zlink_status_t zlink_cmd_mic_stream(const zlink_cmd_t *c, zlink_mic_stream_t *out);
zlink_status_t zlink_cmd_camera_stream(const zlink_cmd_t *c, zlink_camera_stream_t *out);
/* SET_VOLUME / SET_BRIGHTNESS: one u8 percent. */
zlink_status_t zlink_cmd_percent(const zlink_cmd_t *c, uint8_t *percent);
/* PLAY_FILE: copies the path into out (NUL-terminated). */
zlink_status_t zlink_cmd_play_file_path(const zlink_cmd_t *c, char *out, size_t cap);

#ifdef __cplusplus
}
#endif
