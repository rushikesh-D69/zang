/*
 * ds_proto.h - DS serial bus servo packet codec (pure C, no IDF; host-tested in test_host/).
 *
 * Protocol family: Feetech-SCS-style half-duplex TTL bus, BIG-endian 16-bit values. Confirmed from Zing's
 * working firmware (zing3.0 serial_servo.cpp) for the same DS motor family:
 *     request  FF FF id len instr params... chk        len = n_params + 2
 *     reply    FF FF id len error params... chk        (Zing also accepted FF F5 as the 2nd header byte)
 *     chk      = ~(id + len + instr/error + params...) & 0xFF
 * Registers marked CONFIRMED were used by Zing on real servos. ASSUMED ones follow the SCS register map and must
 * be verified against a DS-R003G datasheet / bench test before relying on them.
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DS_HEADER       0xFF
#define DS_HEADER_ALT   0xF5 /* some DS servos reply FF F5 (seen in Zing's frame check) */
#define DS_BROADCAST_ID 0xFE
#define DS_MAX_PACKET   160

typedef enum {
    DS_INST_PING = 0x01,       /* ASSUMED */
    DS_INST_READ = 0x02,       /* CONFIRMED */
    DS_INST_WRITE = 0x03,      /* CONFIRMED */
    DS_INST_SYNC_WRITE = 0x83, /* ASSUMED (SCS sync write); disabled by default, see Kconfig */
} ds_instr_t;

typedef enum {
    DS_REG_ID = 0x05,               /* CONFIRMED (write, used by set-ID) */
    DS_REG_DEVIATION = 0x14,        /* CONFIRMED (zero-offset trim, 2 B) */
    DS_REG_TORQUE_ENABLE = 0x28,    /* CONFIRMED (1 B) */
    DS_REG_GOAL_POSITION = 0x2A,    /* CONFIRMED: position(2 B) + time ms(2 B) */
    DS_REG_PRESENT_POSITION = 0x38, /* CONFIRMED (2 B) */
    DS_REG_PRESENT_VOLTAGE = 0x3E,  /* CONFIRMED (1 B, units of 0.1 V per SCS map - verify) */
    DS_REG_PRESENT_TEMP = 0x3F,     /* ASSUMED (1 B, deg C) */
} ds_reg_t;

typedef enum {
    DS_OK = 0,
    DS_ERR_ARG = -1,
    DS_ERR_NOSPACE = -2,
    DS_ERR_SHORT = -3,  /* not enough bytes yet */
    DS_ERR_HEADER = -4, /* no valid header found */
    DS_ERR_CHECKSUM = -5,
    DS_ERR_LENGTH = -6,
} ds_status_t;

typedef struct {
    uint8_t id;
    uint8_t error;          /* servo status byte (0 = OK) */
    const uint8_t *params;  /* points into the parsed buffer */
    uint8_t n_params;
    size_t offset;          /* where the reply started in the buffer */
    size_t consumed;        /* bytes up to the end of this reply */
} ds_reply_t;

/* Checksum over pkt[2 .. len-2] (id .. last param). len = total packet length including checksum byte. */
uint8_t ds_checksum(const uint8_t *pkt, size_t len);

/* Builders return the packet length (> 0) or a negative ds_status_t. */
int ds_build(uint8_t *buf, size_t cap, uint8_t id, uint8_t instr, const uint8_t *params, size_t n_params);
int ds_build_write(uint8_t *buf, size_t cap, uint8_t id, uint8_t reg, const uint8_t *data, size_t n);
int ds_build_read(uint8_t *buf, size_t cap, uint8_t id, uint8_t reg, uint8_t n_bytes);
int ds_build_goal(uint8_t *buf, size_t cap, uint8_t id, uint16_t position, uint16_t time_ms);
int ds_build_torque(uint8_t *buf, size_t cap, uint8_t id, bool enable);
/* Sync write: one packet sets `reg` (n_bytes each) on n servos. data = n * n_bytes, servo-major. */
int ds_build_sync_write(uint8_t *buf, size_t cap, uint8_t reg, uint8_t n_bytes, const uint8_t *ids,
                        const uint8_t *data, size_t n);
/* Sync goal for n servos (position + time each). */
int ds_build_sync_goal(uint8_t *buf, size_t cap, const uint8_t *ids, const uint16_t *pos, uint16_t time_ms,
                       size_t n);

/* True if b[0..1] is a valid header (FF FF or FF F5). Fixes Zing's always-true check. */
bool ds_header_ok(const uint8_t *b);

/*
 * Find and validate the first reply from `expect_id` (or any id if expect_id == DS_BROADCAST_ID) in buf,
 * starting at `skip` (use skip = request length to step over the half-duplex echo).
 * Bad-checksum candidates are skipped, so line noise before the reply is tolerated.
 */
ds_status_t ds_find_reply(const uint8_t *buf, size_t len, size_t skip, uint8_t expect_id, ds_reply_t *out);

static inline uint16_t ds_be16(const uint8_t *p) { return (uint16_t)((p[0] << 8) | p[1]); }
static inline void ds_put_be16(uint8_t *p, uint16_t v)
{
    p[0] = (uint8_t)(v >> 8);
    p[1] = (uint8_t)v;
}

#ifdef __cplusplus
}
#endif
