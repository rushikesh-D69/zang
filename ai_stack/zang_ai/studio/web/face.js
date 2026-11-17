// Zang chest-TFT face, JS port of zang_ai/face/render.py (the canonical SDF algorithm; firmware: display/face_sdf.c).
// Keep the three in step: change render.py first, then this file and the C port, then regenerate the goldens.
// Coordinates: panel centre origin, +x right, +y down, px of a 240x240 panel.

export class FaceLib {
  constructor(lib) {
    this.lib = lib;
    this.names = lib.emotions.map((e) => e.name);
    this.params = lib.params.map((p) => p.name);
  }
  resolve(emotion = "neutral", over = {}) {
    const e = typeof emotion === "number" ? this.lib.emotions[emotion] : this.lib.emotions[this.names.indexOf(emotion)];
    return { ...this.lib.base, ...(e ? e.p : {}), ...over };
  }
  // neutral -> emotion by intensity (0..1), exactly like the firmware's FACE_SET intensity
  withIntensity(emotion, intensity) { return blend(this.resolve("neutral"), this.resolve(emotion), intensity); }
}

export function blend(a, b, u) {
  const out = {};
  for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
    const va = a[k] ?? b[k], vb = b[k] ?? a[k];
    if (k === "symbol" || k === "_wink") out[k] = u >= 0.5 ? vb : va;
    else if (k === "hue") { const dh = ((vb - va + 540) % 360) - 180; out[k] = (va + dh * u + 360) % 360; }
    else out[k] = va + (vb - va) * u;
  }
  return out;
}

function hsv(h, s, v) {
  const f = (n) => { const k = (n + h / 60) % 6; return v - v * s * Math.max(0, Math.min(k, 4 - k, 1)); };
  return [f(5) * 255, f(3) * 255, f(1) * 255];
}
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const cover = (d) => clamp(0.5 - d, 0, 1);
function sdRoundBox(x, y, cx, cy, w, h, r) {
  const qx = Math.abs(x - cx) - (w / 2 - r), qy = Math.abs(y - cy) - (h / 2 - r);
  return Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) + Math.min(Math.max(qx, qy), 0) - r;
}
function sdSeg(x, y, ax, ay, bx, by, r) {
  const px = x - ax, py = y - ay, ex = bx - ax, ey = by - ay;
  const t = clamp((px * ex + py * ey) / Math.max(ex * ex + ey * ey, 1e-9), 0, 1);
  return Math.hypot(px - ex * t, py - ey * t) - r;
}
function sdEllipse(x, y, cx, cy, rx, ry) { return (Math.hypot((x - cx) / rx, (y - cy) / ry) - 1) * Math.min(rx, ry); }
function sdHeart(x, y, cx, cy, s) {
  const px = Math.abs(x - cx) / s, py = -(y - cy) / s + 0.5;
  if (py + px > 1) return (Math.hypot(px - 0.25, py - 0.75) - Math.SQRT2 / 4) * s;
  const m = Math.max(px + py, 0) * 0.5;
  return Math.sqrt(Math.min(px * px + (py - 1) ** 2, (px - m) ** 2 + (py - m) ** 2)) * Math.sign(px - py) * s;
}
function eyeSd(x, y, p, s, openScale) {
  const cx = s * p.eye_dx + p.gaze_x, cy = p.eye_y + p.gaze_y, w = p.eye_w;
  const h = Math.max(p.eye_h * p.eye_open * openScale, 4);
  const r = p.eye_round * Math.min(w, h) / 2;
  let d = sdRoundBox(x, y, cx, cy, w, h, r);
  if (p.lid_top > 0.001) {
    const k = Math.tan(p.lid_angle * Math.PI / 180) * (-s), yl = cy - h / 2 + p.lid_top * h;
    d = Math.max(d, -(y - yl - k * (x - cx)) / Math.sqrt(1 + k * k));
  }
  if (p.lid_bot > 0.001) {
    const R = w * 0.9, c2y = cy + h / 2 + R - p.lid_bot * h;
    d = Math.max(d, R - Math.hypot(x - cx, y - c2y));
  }
  return d;
}
function arcSd(x, y, w, my, sg, t) {
  const ex = w / 2;
  if (Math.abs(sg) < 0.5) return sdSeg(x, y, -ex, my, ex, my, t / 2);
  const R = (ex * ex + sg * sg) / (2 * Math.abs(sg)), ccy = my + sg / 2 - Math.sign(sg) * R, ey = my - sg / 2;
  if (Math.abs(x) <= ex && Math.sign(y - ccy) === Math.sign(sg)) return Math.abs(Math.hypot(x, y - ccy) - R) - t / 2;
  return Math.min(Math.hypot(x + ex, y - ey), Math.hypot(x - ex, y - ey)) - t / 2;
}
function mouthSd(x, y, p) {
  const w0 = p.mouth_w, my = p.mouth_y, t = p.mouth_thick, op = p.mouth_open;
  if (w0 < 1) return 1e9;
  const sg = p.mouth_curve * w0 * 0.35, w = w0 * Math.max(0, 1 - 1.6 * op);
  let d = w >= 1 ? arcSd(x, y, w, my, sg * (w / w0), t) : 1e9;
  if (op > 0.02) {
    const ry = 4 + 22 * op, rx = w0 * (0.3 + 0.2 * op), cy = my + sg * 0.3;
    let oval = sdEllipse(x, y, 0, cy, rx, ry);
    if (p.mouth_curve > 0.3 && op > 0.3) oval = Math.max(oval, (cy - ry * 0.25) - y);
    d = Math.min(d, oval);
  }
  return d;
}
function symbolSd(x, y, p) {
  const sym = Math.round(p.symbol), cyy = p.eye_y + p.gaze_y;
  if (sym === 3) {
    let d = 1e9;
    for (const [zx, zy, zs] of [[62, -82, 16], [86, -102, 11]]) {
      d = Math.min(d, sdSeg(x, y, zx, zy, zx + zs, zy, 2.5), sdSeg(x, y, zx + zs, zy, zx, zy + zs, 2.5), sdSeg(x, y, zx, zy + zs, zx + zs, zy + zs, 2.5));
    }
    return d;
  }
  if (sym === 4) {
    const tx = -p.eye_dx + p.gaze_x, ty = cyy + p.eye_h * p.eye_open / 2 + 16;
    return Math.min(sdEllipse(x, y, tx, ty, 6, 7), sdSeg(x, y, tx, ty - 12, tx, ty - 4, 2.5));
  }
  if (sym === 5) {
    const sx = p.eye_dx + p.gaze_x + p.eye_w / 2 + 12, sy = cyy - p.eye_h / 2 - 6;
    return Math.min(sdSeg(x, y, sx - 11, sy, sx + 11, sy, 2), sdSeg(x, y, sx, sy - 11, sx, sy + 11, 2), sdSeg(x, y, sx - 5, sy - 5, sx + 5, sy + 5, 1.5));
  }
  return 1e9;
}

// Draw parameters p into a canvas 2D context at size x size. blink scales eye_open (1 = as given).
// Per-shape bounding boxes (panel px, margin >= 2 px beyond anything the shape can cover): pixels outside a shape's box skip
// that shape's distance function entirely (coverage there is exactly 0), so the output is identical but ~4x cheaper.
function shapeBoxes(p, eyes, sym) {
  const B = { eyes: [], brows: [], mouth: null, symbol: null, blush: null };
  const w = p.eye_w, h = Math.max(p.eye_h, 4), m = 3;
  for (const e of eyes) {
    const rx = sym === 1 ? w * 0.9 : sym === 2 ? w * 0.45 + 6 : w / 2 + m;
    const ry = sym === 1 ? w * 0.9 : sym === 2 ? w * 0.45 + 6 : h / 2 + m;
    B.eyes.push([e.cx - rx, e.cx + rx, e.cy - ry, e.cy + ry]);
    if (p.brow > 0.02) {
      const L = w * 0.55, by = e.cy - p.eye_h / 2 + p.brow_y, dy = Math.abs(Math.tan(p.brow_angle * Math.PI / 180) * L);
      B.brows.push([e.cx - L - 6, e.cx + L + 6, by - dy - 6, by + dy + 6]);
    }
  }
  if (p.mouth_w >= 1) {
    const sg = Math.abs(p.mouth_curve * p.mouth_w * 0.35), rx = p.mouth_w * 0.55 + p.mouth_thick + 4, my = p.mouth_y;
    B.mouth = [-rx, rx, my - sg - 34, my + sg + 34];
  }
  if (Math.round(p.symbol) === 3) B.symbol = [56, 110, -110, -58];
  else if (Math.round(p.symbol) === 4) { const tx = -p.eye_dx + p.gaze_x, ty = p.eye_y + p.gaze_y + p.eye_h * p.eye_open / 2 + 16; B.symbol = [tx - 12, tx + 12, ty - 18, ty + 12]; }
  else if (Math.round(p.symbol) === 5) { const sx = p.eye_dx + p.gaze_x + w / 2 + 12, sy = p.eye_y + p.gaze_y - p.eye_h / 2 - 6; B.symbol = [sx - 16, sx + 16, sy - 16, sy + 16]; }
  if (p.blush > 0.01) { const bx = p.eye_dx + 8, by = p.eye_y + p.eye_h / 2 + 18; B.blush = [-bx - 20, bx + 20, by - 12, by + 12]; }
  return B;
}
const inBox = (b, x, y) => b && x >= b[0] && x <= b[1] && y >= b[2] && y <= b[3];

export function renderFace(ctx, p, size = 240, blink = 1, bg = [4, 6, 10], cull = true) {
  const img = ctx.createImageData(size, size), D = img.data, s = size / 240;
  const ink = hsv(((p.hue % 360) + 360) % 360, clamp(p.sat, 0, 1), 1);
  const pink = hsv(340, 0.55, 1);
  const sym = Math.round(p.symbol);
  const eyes = [-1, 1].map((side) => ({ side, cx: side * p.eye_dx + p.gaze_x, cy: p.eye_y + p.gaze_y,
    wink: (p._wink ?? 0) >= 0.5 && side === 1 ? 0.07 : 1 }));
  const B = cull ? shapeBoxes(p, eyes, sym) : null;
  for (let j = 0; j < size; j++) {
    const y = (j + 0.5) / s - 120;
    for (let i = 0; i < size; i++) {
      const x = (i + 0.5) / s - 120, o = 4 * (j * size + i);
      if (x * x + y * y > 14400) { D[o] = bg[0]; D[o + 1] = bg[1]; D[o + 2] = bg[2]; D[o + 3] = 255; continue; }
      let a = 0;
      for (let k = 0; k < 2; k++) {
        const e = eyes[k];
        if (!B || inBox(B.eyes[k], x, y)) {
          let d;
          if (sym === 1) d = sdHeart(x, y, e.cx, e.cy, p.eye_w * 1.25);
          else if (sym === 2) { const L = p.eye_w * 0.45; d = Math.min(sdSeg(x, y, e.cx - L, e.cy - L, e.cx + L, e.cy + L, 4), sdSeg(x, y, e.cx - L, e.cy + L, e.cx + L, e.cy - L, 4)); }
          else d = eyeSd(x, y, p, e.side, blink * e.wink);
          a = Math.max(a, cover(d));
        }
        if (p.brow > 0.02 && (!B || inBox(B.brows[k], x, y))) {
          const by = e.cy - p.eye_h / 2 + p.brow_y, L = p.eye_w * 0.55, dy = Math.tan(p.brow_angle * Math.PI / 180) * L * (-e.side);
          a = Math.max(a, p.brow * cover(sdSeg(x, y, e.cx - L, by - dy, e.cx + L, by + dy, 4)));
        }
      }
      if (!B || inBox(B.mouth, x, y)) a = Math.max(a, cover(mouthSd(x, y, p)));
      if (!B || inBox(B.symbol, x, y)) a = Math.max(a, cover(symbolSd(x, y, p)));
      let r = bg[0] * (1 - a) + ink[0] * a, g = bg[1] * (1 - a) + ink[1] * a, b = bg[2] * (1 - a) + ink[2] * a;
      if (p.blush > 0.01 && (!B || inBox(B.blush, x, y))) {
        let ba = 0;
        for (const side of [-1, 1]) ba = Math.max(ba, cover(sdEllipse(x, y, side * (p.eye_dx + 8), p.eye_y + p.eye_h / 2 + 18, 16, 8)));
        ba *= 0.5 * p.blush * (1 - a);
        r = r * (1 - ba) + pink[0] * ba; g = g * (1 - ba) + pink[1] * ba; b = b * (1 - ba) + pink[2] * ba;
      }
      D[o] = r; D[o + 1] = g; D[o + 2] = b; D[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

// Live animator: emotion blending, idle blink + gaze drift, talking mouth. Same behaviour as the firmware.
export class FaceAnimator {
  constructor(lib) {
    this.lib = lib;
    this.cur = lib.resolve("neutral");
    this.from = this.cur; this.to = this.cur; this.t0 = 0; this.dur = 1;
    this.override = {};      // designer parameter overrides
    this.talking = false; this.hold = false;
    this.nextBlink = 2000; this.blinkUntil = 0; this.drift = [0, 0]; this.nextDrift = 1500;
  }
  set(emotion, intensity = 1, blendMs = 250, now = performance.now()) {
    this.from = this.current(now);
    this.to = this.lib.withIntensity(emotion, intensity);
    this.t0 = now; this.dur = Math.max(1, blendMs);
  }
  current(now) { return blend(this.from, this.to, clamp((now - this.t0) / this.dur, 0, 1)); }
  frame(now) {
    const p = { ...this.current(now), ...this.override };
    let blink = 1;
    if (!this.hold) {
      if (now > this.nextBlink) { this.blinkUntil = now + 140; this.nextBlink = now + 2500 + Math.random() * 3000; }
      if (now < this.blinkUntil) blink = 0.05;
      if (now > this.nextDrift) { this.drift = [(Math.random() - 0.5) * 6, (Math.random() - 0.5) * 4]; this.nextDrift = now + 1200 + Math.random() * 2000; }
      p.gaze_x += this.drift[0]; p.gaze_y += this.drift[1];
    }
    if (this.talking) p.mouth_open = Math.max(p.mouth_open, 0.15 + 0.5 * Math.abs(Math.sin(now / 90)));
    return { p, blink };
  }
}

// Face track playback (compose result `face` keys on the motion clock) -> parameters at time t
export function trackParams(lib, keys, t) {
  if (!keys || !keys.length) return lib.resolve("neutral");
  let i = 0;
  while (i + 1 < keys.length && keys[i + 1].t <= t) i++;
  const k = keys[i], prev = i ? keys[i - 1] : k;
  const a = lib.withIntensity(prev.emotion, prev.intensity), b = lib.withIntensity(k.emotion, k.intensity);
  return t < k.t ? a : blend(a, b, clamp((t - k.t) / (k.blend_ms / 1000), 0, 1));
}
