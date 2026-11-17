/* ds_proto.c - DS serial servo packet codec. Pure C: see ds_proto.h. */
#include "ds_proto.h"

#include <string.h>

uint8_t ds_checksum(const uint8_t *pkt, size_t len)
{
    unsigned sum = 0;
    for (size_t i = 2; i + 1 < len; i++) sum += pkt[i];
    return (uint8_t)(~sum & 0xFF);
}

int ds_build(uint8_t *buf, size_t cap, uint8_t id, uint8_t instr, const uint8_t *params, size_t n_params)
{
    if (!buf || (n_params && !params) || n_params > 250) return DS_ERR_ARG;
    size_t len = 6 + n_params;
    if (cap < len) return DS_ERR_NOSPACE;
    buf[0] = DS_HEADER;
    buf[1] = DS_HEADER;
    buf[2] = id;
    buf[3] = (uint8_t)(n_params + 2);
    buf[4] = instr;
    if (n_params) memcpy(&buf[5], params, n_params);
    buf[len - 1] = ds_checksum(buf, len);
    return (int)len;
}

int ds_build_write(uint8_t *buf, size_t cap, uint8_t id, uint8_t reg, const uint8_t *data, size_t n)
{
    uint8_t p[DS_MAX_PACKET];
    if (n + 1 > sizeof(p) || (n && !data)) return DS_ERR_ARG;
    p[0] = reg;
    if (n) memcpy(&p[1], data, n);
    return ds_build(buf, cap, id, DS_INST_WRITE, p, n + 1);
}

int ds_build_read(uint8_t *buf, size_t cap, uint8_t id, uint8_t reg, uint8_t n_bytes)
{
    const uint8_t p[2] = {reg, n_bytes};
    return ds_build(buf, cap, id, DS_INST_READ, p, 2);
}

int ds_build_goal(uint8_t *buf, size_t cap, uint8_t id, uint16_t position, uint16_t time_ms)
{
    uint8_t d[4];
    ds_put_be16(&d[0], position);
    ds_put_be16(&d[2], time_ms);
    return ds_build_write(buf, cap, id, DS_REG_GOAL_POSITION, d, sizeof(d));
}

int ds_build_torque(uint8_t *buf, size_t cap, uint8_t id, bool enable)
{
    const uint8_t d = enable ? 1 : 0;
    return ds_build_write(buf, cap, id, DS_REG_TORQUE_ENABLE, &d, 1);
}

int ds_build_sync_write(uint8_t *buf, size_t cap, uint8_t reg, uint8_t n_bytes, const uint8_t *ids,
                        const uint8_t *data, size_t n)
{
    if (!ids || !data || n == 0 || n_bytes == 0) return DS_ERR_ARG;
    size_t n_params = 2 + n * (1 + (size_t)n_bytes);
    if (n_params > 250) return DS_ERR_ARG;
    uint8_t p[256];
    p[0] = reg;
    p[1] = n_bytes;
    size_t k = 2;
    for (size_t i = 0; i < n; i++) {
        p[k++] = ids[i];
        memcpy(&p[k], &data[i * n_bytes], n_bytes);
        k += n_bytes;
    }
    return ds_build(buf, cap, DS_BROADCAST_ID, DS_INST_SYNC_WRITE, p, n_params);
}

int ds_build_sync_goal(uint8_t *buf, size_t cap, const uint8_t *ids, const uint16_t *pos, uint16_t time_ms,
                       size_t n)
{
    if (!pos || n == 0 || n > 32) return DS_ERR_ARG;
    uint8_t d[32 * 4];
    for (size_t i = 0; i < n; i++) {
        ds_put_be16(&d[i * 4], pos[i]);
        ds_put_be16(&d[i * 4 + 2], time_ms);
    }
    return ds_build_sync_write(buf, cap, DS_REG_GOAL_POSITION, 4, ids, d, n);
}

bool ds_header_ok(const uint8_t *b)
{
    /* Zing had `(b[0] != 0xff) && ((b[1] != 0xff) || (b[1] != 0xf5))`: the inner OR is always true, so only b[0]
     * was effectively checked. Correct rule: first byte FF AND second byte FF or F5. */
    return b[0] == DS_HEADER && (b[1] == DS_HEADER || b[1] == DS_HEADER_ALT);
}

ds_status_t ds_find_reply(const uint8_t *buf, size_t len, size_t skip, uint8_t expect_id, ds_reply_t *out)
{
    if (!buf || !out) return DS_ERR_ARG;
    ds_status_t last = DS_ERR_HEADER;
    for (size_t i = skip; i + 1 < len; i++) {
        if (!ds_header_ok(&buf[i])) continue;
        if (i + 4 > len) return DS_ERR_SHORT;
        uint8_t id = buf[i + 2];
        uint8_t plen = buf[i + 3]; /* = n_params + 2 (error byte + checksum) */
        if (plen < 2) {
            last = DS_ERR_LENGTH;
            continue;
        }
        size_t total = 4 + (size_t)plen;
        if (i + total > len) {
            last = DS_ERR_SHORT;
            continue; /* a later candidate may still be complete */
        }
        if (ds_checksum(&buf[i], total) != buf[i + total - 1]) {
            last = DS_ERR_CHECKSUM;
            continue;
        }
        if (expect_id != DS_BROADCAST_ID && id != expect_id) continue;
        out->id = id;
        out->error = buf[i + 4];
        out->params = &buf[i + 5];
        out->n_params = (uint8_t)(plen - 2);
        out->offset = i;
        out->consumed = i + total;
        return DS_OK;
    }
    return last;
}
