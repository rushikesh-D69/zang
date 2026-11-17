/* zang_units.h - unit conversions shared by firmware modules (pure C, host-testable). */
#pragma once

#include <stdint.h>
#include <math.h>

#define ZANG_PI_F 3.14159265358979f

/* ZLink joint angles are int16 centidegrees in the model joint frame. */
static inline float zang_cdeg_to_rad(int16_t cdeg) { return (float)cdeg * (ZANG_PI_F / 18000.0f); }

static inline int16_t zang_rad_to_cdeg(float rad)
{
    float c = rad * (18000.0f / ZANG_PI_F);
    if (c > 32766.0f) c = 32766.0f; /* 0x7FFF is reserved for "not available" */
    if (c < -32767.0f) c = -32767.0f;
    return (int16_t)lroundf(c);
}

#define ZANG_CDEG_NA ((int16_t)0x7FFF)
