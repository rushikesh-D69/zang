"""Reference face renderer for Zang's chest TFT (the canonical algorithm).

The studio (web/face.js) and the P4 firmware (display/face_sdf.c) implement EXACTLY this signed-distance algorithm
on the parameters of config/faces.json. tools/gen_face_goldens.py renders every emotion here as the golden images the
firmware host test compares against. Change the algorithm here first, then port it, then regenerate the goldens.

Conventions: 240x240 panel, coordinates relative to the panel centre, +x right, +y DOWN, units px. Every shape has a
signed distance d (negative inside). Coverage = clamp(0.5 - d, 0, 1) gives 1-px anti-aliasing. Colour =
bg*(1-a) + ink*a, with blush under the ink.
"""
import colorsys
import json
import math

import numpy as np

from .. import paths

LIB = json.load(open(paths.config("faces.json"), encoding="utf-8"))
PARAMS = [p["name"] for p in LIB["params"]]
EMOTIONS = [e["name"] for e in LIB["emotions"]]
BASE = LIB["base"]


def resolve(emotion="neutral", overrides=None):
    """Base parameters, then the emotion's, then explicit overrides. -> dict (+ '_wink' flag)."""
    p = dict(BASE)
    e = LIB["emotions"][EMOTIONS.index(emotion)]["p"] if isinstance(emotion, str) else LIB["emotions"][emotion]["p"]
    p.update(e)
    p.update(overrides or {})
    return p


def blend(a, b, u):
    """Linear blend of two parameter sets (u: 0 = a, 1 = b). Symbol and wink switch at the midpoint; hue takes the
    shortest way round the colour wheel."""
    out = {}
    for k in set(a) | set(b):
        va, vb = a.get(k, BASE.get(k, 0)), b.get(k, BASE.get(k, 0))
        if k in ("symbol", "_wink"):
            out[k] = vb if u >= 0.5 else va
        elif k == "hue":
            dh = ((vb - va + 180) % 360) - 180
            out[k] = (va + dh * u) % 360
        else:
            out[k] = va + (vb - va) * u
    return out


def ink_rgb(p):
    r, g, b = colorsys.hsv_to_rgb((p["hue"] % 360) / 360.0, float(np.clip(p["sat"], 0, 1)), 1.0)
    return np.array([r, g, b]) * 255


# ---- signed distance primitives (vectorised over the pixel grid) --------------------------------------------
def sd_round_box(x, y, cx, cy, w, h, r):
    qx = np.abs(x - cx) - (w / 2 - r)
    qy = np.abs(y - cy) - (h / 2 - r)
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r


def sd_segment(x, y, ax, ay, bx, by, r):
    px, py, ex, ey = x - ax, y - ay, bx - ax, by - ay
    t = np.clip((px * ex + py * ey) / max(ex * ex + ey * ey, 1e-9), 0, 1)
    return np.hypot(px - ex * t, py - ey * t) - r


def sd_ellipse(x, y, cx, cy, rx, ry):
    """Approximate ellipse distance (exact on the axes, good enough for 1-px AA)."""
    k = np.hypot((x - cx) / rx, (y - cy) / ry)
    return (k - 1) * min(rx, ry)


def sd_heart(x, y, cx, cy, s):
    """Inigo Quilez heart, point up = tip down; s = height px."""
    px = np.abs(x - cx) / s
    py = -(y - cy) / s + 0.5
    a = np.hypot(px - 0.25, py - 0.75) - math.sqrt(2) / 4
    m = np.maximum(px + py, 0) * 0.5
    b = np.sqrt(np.minimum(px ** 2 + (py - 1) ** 2, (px - m) ** 2 + (py - m) ** 2)) * np.sign(px - py)
    return np.where(py + px > 1, a, b) * s


def cover(d):
    return np.clip(0.5 - d, 0, 1)


# ---- the face ---------------------------------------------------------------------------------------------------
def eye_sd(x, y, p, s, open_scale=1.0):
    """Signed distance of one eye (s = -1 viewer's left, +1 viewer's right)."""
    cx = s * p["eye_dx"] + p["gaze_x"]
    cy = p["eye_y"] + p["gaze_y"]
    w = p["eye_w"]
    h = max(p["eye_h"] * p["eye_open"] * open_scale, 4.0)
    r = p["eye_round"] * min(w, h) / 2
    d = sd_round_box(x, y, cx, cy, w, h, r)
    if p["lid_top"] > 0.001:  # slanted upper lid: keep the part below the lid line
        k = math.tan(math.radians(p["lid_angle"])) * (-s)
        y_line = cy - h / 2 + p["lid_top"] * h
        d = np.maximum(d, -(y - y_line - k * (x - cx)) / math.sqrt(1 + k * k))
    if p["lid_bot"] > 0.001:  # happy squint: an arc pushes up from below, keep outside the circle
        R = w * 0.9
        c2y = cy + h / 2 + R - p["lid_bot"] * h
        d = np.maximum(d, R - np.hypot(x - cx, y - c2y))
    return d


def mouth_sd(x, y, p):
    """Closed: an arc line. Opening (speech, laughs): the line shrinks into a growing oval, so talking animates
    smoothly; a wide-open smile becomes a flat-topped 'D' (laugh)."""
    w0, my, t = p["mouth_w"], p["mouth_y"], p["mouth_thick"]
    if w0 < 1:
        return np.full_like(x, 1e9)
    op = p["mouth_open"]
    sg = p["mouth_curve"] * w0 * 0.35  # sagitta; + = middle lower = smile
    w = w0 * max(0.0, 1 - 1.6 * op)    # the line shrinks as the mouth opens
    d = _arc_sd(x, y, w, my, sg * (w / w0), t) if w >= 1 else np.full_like(x, 1e9)
    if op > 0.02:
        ry = 4 + 22 * op
        rx = w0 * (0.30 + 0.2 * op)
        cy = my + sg * 0.3
        oval = sd_ellipse(x, y, 0, cy, rx, ry)
        if p["mouth_curve"] > 0.3 and op > 0.3:  # laugh: cut the top flat
            oval = np.maximum(oval, (cy - ry * 0.25) - y)
        d = np.minimum(d, oval)
    return d


def _arc_sd(x, y, w, my, sg, t):
    ex = w / 2
    if abs(sg) < 0.5:
        return sd_segment(x, y, -ex, my, ex, my, t / 2)
    R = (ex * ex + sg * sg) / (2 * abs(sg))
    ccy = my + sg / 2 - math.copysign(R, sg)
    e_y = my - sg / 2
    on_arc = (np.abs(x) <= ex) & (np.sign(y - ccy) == math.copysign(1, sg))
    d_arc = np.abs(np.hypot(x, y - ccy) - R) - t / 2
    d_cap = np.minimum(np.hypot(x + ex, y - e_y), np.hypot(x - ex, y - e_y)) - t / 2
    return np.where(on_arc, d_arc, d_cap)


def symbol_sd(x, y, p):
    sym = int(round(p["symbol"]))
    big = np.full_like(x, 1e9)
    cyy = p["eye_y"] + p["gaze_y"]
    if sym == 3:  # zzz: two Z glyphs top right
        d = big
        for zx, zy, zs in ((62, -82, 16), (86, -102, 11)):
            d = np.minimum(d, sd_segment(x, y, zx, zy, zx + zs, zy, 2.5))
            d = np.minimum(d, sd_segment(x, y, zx + zs, zy, zx, zy + zs, 2.5))
            d = np.minimum(d, sd_segment(x, y, zx, zy + zs, zx + zs, zy + zs, 2.5))
        return d
    if sym == 4:  # tear under the viewer's left eye
        tx = -p["eye_dx"] + p["gaze_x"]
        ty = cyy + p["eye_h"] * p["eye_open"] / 2 + 16
        return np.minimum(sd_ellipse(x, y, tx, ty, 6, 7), sd_segment(x, y, tx, ty - 12, tx, ty - 4, 2.5))
    if sym == 5:  # sparkle by the viewer's right eye
        sx = p["eye_dx"] + p["gaze_x"] + p["eye_w"] / 2 + 12
        sy = cyy - p["eye_h"] / 2 - 6
        d = np.minimum(sd_segment(x, y, sx - 11, sy, sx + 11, sy, 2), sd_segment(x, y, sx, sy - 11, sx, sy + 11, 2))
        return np.minimum(d, sd_segment(x, y, sx - 5, sy - 5, sx + 5, sy + 5, 1.5))
    return big


def render(p, size=240, bg=None, blink=1.0):
    """Parameters -> uint8 RGB image (size x size). blink scales eye_open (1 = as given, ~0.05 = mid-blink)."""
    bg = np.array(bg if bg is not None else LIB["panel"]["bg"], float)
    s = size / 240.0
    ii = (np.arange(size) + 0.5) / s - 120
    x, y = np.meshgrid(ii, ii)
    ink = ink_rgb(p)
    a = np.zeros_like(x)
    sym = int(round(p["symbol"]))
    for side in (-1, 1):
        cx, cyy = side * p["eye_dx"] + p["gaze_x"], p["eye_y"] + p["gaze_y"]
        if sym == 1:
            d = sd_heart(x, y, cx, cyy, p["eye_w"] * 1.25)
        elif sym == 2:
            L = p["eye_w"] * 0.45
            d = np.minimum(sd_segment(x, y, cx - L, cyy - L, cx + L, cyy + L, 4),
                           sd_segment(x, y, cx - L, cyy + L, cx + L, cyy - L, 4))
        else:
            wink = 0.07 if (p.get("_wink", 0) >= 0.5 and side == 1) else 1.0
            d = eye_sd(x, y, p, side, blink * wink)
        a = np.maximum(a, cover(d / s if s != 1 else d))
        if p["brow"] > 0.02:
            by = cyy - p["eye_h"] / 2 + p["brow_y"]
            L = p["eye_w"] * 0.55
            dy = math.tan(math.radians(p["brow_angle"])) * L * (-side)
            bd = sd_segment(x, y, cx - L, by - dy, cx + L, by + dy, 4)
            a = np.maximum(a, p["brow"] * cover(bd))
    a = np.maximum(a, cover(mouth_sd(x, y, p)))
    a = np.maximum(a, cover(symbol_sd(x, y, p)))
    img = bg[None, None] * (1 - a[..., None]) + ink[None, None] * a[..., None]
    if p["blush"] > 0.01:
        pink = np.array(colorsys.hsv_to_rgb(340 / 360, 0.55, 1.0)) * 255
        ba = np.zeros_like(x)
        for side in (-1, 1):
            ba = np.maximum(ba, cover(sd_ellipse(x, y, side * (p["eye_dx"] + 8), p["eye_y"] + p["eye_h"] / 2 + 18, 16, 8)))
        ba *= 0.5 * p["blush"] * (1 - a)
        img = img * (1 - ba[..., None]) + pink[None, None] * ba[..., None]
    if LIB["panel"]["round"]:
        img[np.hypot(x, y) > 120] = bg
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def write_ppm(path, img):
    with open(path, "wb") as fh:
        fh.write(f"P6\n{img.shape[1]} {img.shape[0]}\n255\n".encode() + img.tobytes())
