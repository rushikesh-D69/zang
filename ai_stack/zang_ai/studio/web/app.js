// Zang Twin Studio client.
// The 3D view never computes kinematics: the server sends MuJoCo body poses, so the picture is exactly the model the
// safety stage and the physics twin used. The chest TFT shows the same canvas as the Display tab (face.js renderer).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { FaceAnimator, FaceLib, renderFace, trackParams } from "./face.js";

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const store = {
  get(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage blocked */ } },
};
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok || j.error) throw new Error(j.error || r.statusText);
  return j;
}
const DEG = 180 / Math.PI;

// ================================================================ theme
function setTheme(t) {
  document.documentElement.dataset.theme = t;
  store.set("zang.theme", t);
  $$("[data-theme-set]").forEach((b) => b.classList.toggle("on", b.dataset.themeSet === t));
  V.applyTheme?.();
  drawTimeline();
}
$$("[data-theme-set]").forEach((b) => (b.onclick = () => setTheme(b.dataset.themeSet)));

// ================================================================ viewer
const V = { bodies: [], showBalance: true, showFace: true };

function initViewer(model, gpu = "") {
  const el = $("#viewport");
  // software WebGL (SwiftShader / llvmpipe) cannot push the CAD meshes at full quality: drop AA and pixel ratio there
  const software = /swiftshader|llvmpipe|softpipe|software|basic render/i.test(gpu);
  V.software = software;
  const renderer = new THREE.WebGLRenderer({ antialias: !software, powerPreference: "high-performance" });
  renderer.setPixelRatio(software ? 1 : Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  el.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(32, 1, 0.01, 20);
  camera.up.set(0, 0, 1);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.addEventListener("change", () => (V.dirty = true));  // render on demand: only when something changed
  const hemi = new THREE.HemisphereLight(0xffffff, 0x7a8696, 1.5);
  const key = new THREE.DirectionalLight(0xffffff, 2.0); key.position.set(1.2, -1, 2);
  const rim = new THREE.DirectionalLight(0xffffff, 0.8); rim.position.set(-1, 1, 0.6);
  scene.add(hemi, key, rim);

  // floor grid (rebuilt per theme)
  let grid = null;
  const makeGrid = () => {
    if (grid) scene.remove(grid);
    grid = new THREE.GridHelper(1.6, 32, new THREE.Color(css("--grid")), new THREE.Color(css("--grid-2")));
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);
  };

  // robot: one group per MuJoCo body, visual meshes inside with their local transforms
  V.bodies = model.bodies.map(() => { const g = new THREE.Group(); scene.add(g); return g; });
  window.__meshes = { total: model.geoms.length, loaded: 0 };  // readiness (tools/screenshot.py)
  const meshCache = {};
  const loadMesh = (id) => (meshCache[id] ??= fetch(`/api/mesh/${id}`).then((r) => r.arrayBuffer()).then((buf) => {
    const [nv, nf] = new Uint32Array(buf, 0, 2);
    // indexed geometry + flat shading: per-face normals come from the GPU (screen-space derivatives), so the CAD's
    // shared vertices across sharp edges shade correctly and no vertex is duplicated (≈6x less work than creased normals)
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(buf, 8, nv * 3), 3));
    g.setIndex(new THREE.BufferAttribute(new Uint32Array(buf, 8 + nv * 12, nf * 3), 1));
    g.computeBoundingSphere();
    return g;
  }));
  for (const geom of model.geoms) {
    loadMesh(geom.mesh).then((g) => {
      const [r, gr, b, a] = geom.rgba;
      const mat = new THREE.MeshStandardMaterial({ color: new THREE.Color(r, gr, b), roughness: 0.62, metalness: 0.05,
        flatShading: true, side: THREE.DoubleSide, transparent: a < 1, opacity: a });
      const mesh = new THREE.Mesh(g, mat);
      mesh.position.fromArray(geom.pos);
      const [w, x, y, z] = geom.quat;
      mesh.quaternion.set(x, y, z, w);
      V.bodies[geom.body].add(mesh);
      window.__meshes.loaded++;
      V.dirty = true;
    });
  }

  // chest display: the CAD has a round window in the chest plate but no TFT part -> virtual screen showing the TFT canvas
  if (model.display) {
    const [w, h] = model.display.size;
    const tex = new THREE.CanvasTexture(TFT.small);  // 96 px canvas: the screen is only ~40 px wide on screen
    tex.colorSpace = THREE.SRGBColorSpace;
    V.tftTex = tex;
    const geo = model.display.shape === "round" ? new THREE.CircleGeometry(w / 2, 64) : new THREE.PlaneGeometry(w, h);
    const screen = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide, toneMapped: false }));
    // panel axes in the chest frame: image right = robot's right (-y), image up = +z, normal along -x (double sided)
    screen.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(
      new THREE.Vector3(0, -1, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(-1, 0, 0)));
    screen.position.fromArray(model.display.center);
    V.bodies[model.display.body].add(screen);
    V.screen = screen;
  }

  // balance overlay: support polygon (feet planted) + centre of pressure from the physics twin
  const sup = model.support.map(([x, y]) => new THREE.Vector3(x, y, 0.0015));
  const supLine = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(sup), new THREE.LineBasicMaterial({ color: 0x16834a }));
  const cop = new THREE.Mesh(new THREE.RingGeometry(0.006, 0.0095, 32), new THREE.MeshBasicMaterial({ color: 0xe0a030, side: THREE.DoubleSide }));
  cop.position.z = 0.002;
  scene.add(supLine, cop);
  V.setCop = (xy) => { cop.visible = V.showBalance && !!xy; if (xy) cop.position.set(xy[0], xy[1], 0.002); V.dirty = true; };
  V.setCop(null);

  V.applyTheme = () => {
    scene.background = new THREE.Color(css("--stage") || "#000");
    makeGrid();
    supLine.material.color.set(css("--curve-5"));
    cop.material.color.set(css("--warn"));
    V.dirty = true;
  };
  V.applyTheme();

  const views = { iso: [0.85, -0.62, 0.42], front: [1.05, 0, 0.24], side: [0, -1.05, 0.24], top: [0.001, 0, 1.25] };
  V.view = (name) => {
    camera.position.set(...(views[name] || views.iso));
    controls.target.set(0, 0, 0.2);
    controls.update();
  };
  V.view(new URLSearchParams(location.hash.slice(1)).get("view") || "iso");

  const resize = () => {
    const w = el.clientWidth, h = el.clientHeight;
    renderer.setSize(w, h);
    camera.aspect = w / Math.max(h, 1);
    camera.updateProjectionMatrix();
    V.dirty = true;
  };
  new ResizeObserver(resize).observe(el);
  resize();

  let frames = 0, t0 = performance.now();
  const label = software ? "SOFTWARE WebGL (slow): fix the GPU driver" : `GPU: ${gpu.replace(/^ANGLE \(|\)$/g, "").slice(0, 48)}`;
  $("#sb-fps").classList.toggle("warn", software);
  $("#sb-fps").title = gpu;
  // per-frame cost of each part, averaged over 1 s: window.__perf and the status-bar tooltip (find the slow part)
  const perf = (window.__perf = { tick: 0, timeline: 0, face: 0, render: 0, n: 0 });
  const loop = (now) => {
    const a0 = performance.now();
    controls.update();
    tick();
    const a1 = performance.now();
    if (TFT.draw(now) && V.tftTex && V.showFace) { V.tftTex.needsUpdate = true; V.dirty = true; }
    const a2 = performance.now();
    if (V.screen && V.screen.visible !== V.showFace) { V.screen.visible = V.showFace; V.dirty = true; }
    const a3 = performance.now();
    if (V.dirty) {  // render on demand: camera, playback, face, theme or resize changed
      V.dirty = false;
      renderer.render(scene, camera);
      frames++;
    }
    const a4 = performance.now();
    perf.tick += a1 - a0 - (P.tlMs || 0); perf.timeline += P.tlMs || 0; P.tlMs = 0;
    perf.face += a2 - a1; perf.render += a4 - a3; perf.n++;
    if (now - t0 > 1000) {
      const n = Math.max(perf.n, 1), f = (v) => (v / n).toFixed(1);
      $("#sb-fps").textContent = `${frames} fps · ${label}`;
      $("#sb-fps").title = `${gpu}\nper loop (ms): tick ${f(perf.tick)} · timeline ${f(perf.timeline)} · face ${f(perf.face)} · render ${f(perf.render)}`;
      perf.last = { fps: frames, tick: +f(perf.tick), timeline: +f(perf.timeline), face: +f(perf.face), render: +f(perf.render) };
      perf.tick = perf.timeline = perf.face = perf.render = perf.n = 0; frames = 0; t0 = now;
    }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

function setBodies(frame) {
  if (!frame || V.bodies.length < frame.length) return;  // 3D view not available: everything else keeps working
  V.dirty = true;
  for (let i = 0; i < frame.length; i++) {
    const p = frame[i];
    V.bodies[i].position.set(p[0], p[1], p[2]);
    V.bodies[i].quaternion.set(p[4], p[5], p[6], p[3]);
  }
}

// ================================================================ chest TFT (shared by the Display tab and the twin)
const TFT = {
  lib: null, anim: null, mode: 0, source: "manual", image: null, text: null, camImg: new Image(), last: 0, lastKey: "",
  small: Object.assign(document.createElement("canvas"), { width: 96, height: 96 }),  // texture for the 3D chest screen
  init(faces) {
    this.lib = new FaceLib(faces);
    this.anim = new FaceAnimator(this.lib);
    this.ctx = $("#tft").getContext("2d");
  },
  // Returns true when the 3D chest-screen texture changed. Cost control: the 240 px canvas (Display tab) is only drawn while
  // that tab is open; the twin's screen uses a 96 px render (the screen is ~40 px on screen); nothing is redrawn when the
  // face parameters have not changed.
  draw(now) {
    const big = $('[data-panel="display"]').classList.contains("on");
    if (!this.anim || now - this.last < (big ? 40 : 55)) return false;
    this.last = now;
    const c = this.ctx, sc = this.small.getContext("2d");
    const toSmall = () => { sc.drawImage($("#tft"), 0, 0, 96, 96); return true; };
    if (this.mode === 1 && this.image) { c.drawImage(this.image, 0, 0, 240, 240); return toSmall(); }
    if (this.mode === 3 && this.camImg.complete && this.camImg.naturalWidth) {
      const im = this.camImg, k = Math.max(240 / im.naturalWidth, 240 / im.naturalHeight);
      c.drawImage(im, (240 - im.naturalWidth * k) / 2, (240 - im.naturalHeight * k) / 2, im.naturalWidth * k, im.naturalHeight * k);
      return toSmall();
    }
    if (this.mode === 2 && this.text) {
      c.fillStyle = this.text.bg; c.fillRect(0, 0, 240, 240);
      c.fillStyle = this.text.fg; c.textAlign = "center"; c.textBaseline = "middle";
      c.font = `700 ${[14, 22, 34, 50][this.text.size - 1]}px ui-monospace, monospace`;
      const lines = this.text.text.split("\n");
      lines.forEach((ln, i) => c.fillText(ln, 120, 120 + (i - (lines.length - 1) / 2) * [18, 28, 42, 60][this.text.size - 1]));
      return toSmall();
    }
    // face: the composed face track while a motion is loaded, else the Display tab's manual face
    let f;
    if (this.source === "track" && P.res?.face?.length) {
      f = { p: trackParams(this.lib, P.res.face, P.t), blink: 1 };
      f.blink = this.anim.frame(now).blink;   // keep idle blink from the animator
      if (P.playing && currentWord()) f.p.mouth_open = Math.max(f.p.mouth_open, 0.15 + 0.5 * Math.abs(Math.sin(now / 90)));
    } else {
      f = this.anim.frame(now);
    }
    const key = (big ? "B" : "S") + f.blink.toFixed(2) + JSON.stringify(f.p, (k, v) => (typeof v === "number" ? Math.round(v * 20) / 20 : v));
    if (key === this.lastKey) return false;   // identical to what is already on screen
    this.lastKey = key;
    if (big) { renderFace(c, f.p, 240, f.blink); return toSmall(); }
    renderFace(sc, f.p, 96, f.blink);
    return true;
  },
};

const D = { emotion: "neutral" };
function initDisplay(faces) {
  TFT.init(faces);
  const lib = TFT.lib;
  // emotion grid with live thumbnails rendered by the same renderer
  $("#emotions").innerHTML = lib.names.map((n) => `<button data-emo="${n}" title="${n}"><canvas width="64" height="64"></canvas>${n}</button>`).join("");
  $$("#emotions button").forEach((b) => renderFace(b.querySelector("canvas").getContext("2d"), lib.resolve(b.dataset.emo), 64));
  const sendFace = () => api("/api/display/face", { emotion: D.emotion, intensity: +$("#f-int").value, blend_ms: +$("#f-blend").value,
    gaze: [+$("#f-gx").value, +$("#f-gy").value], talking: $("#f-talk").checked, hold: $("#f-hold").checked }).then(sentNote).catch(() => {});
  const applyFace = () => {
    TFT.source = "manual"; TFT.mode = 0; setMode(0, false);
    TFT.anim.set(D.emotion, +$("#f-int").value, +$("#f-blend").value);
    TFT.anim.override = { ...TFT.anim.override, gaze_x: +$("#f-gx").value, gaze_y: +$("#f-gy").value };
    TFT.anim.talking = $("#f-talk").checked; TFT.anim.hold = $("#f-hold").checked;
    syncParamSliders();
    sendFace();
  };
  $("#emotions").onclick = (ev) => {
    const b = ev.target.closest("button[data-emo]");
    if (!b) return;
    D.emotion = b.dataset.emo;
    $$("#emotions button").forEach((x) => x.classList.toggle("on", x === b));
    TFT.anim.override = {};
    applyFace();
  };
  for (const id of ["f-int", "f-blend", "f-gx", "f-gy"]) $(`#${id}`).addEventListener("input", (e) => { e.target.nextElementSibling.value = e.target.value; });
  for (const id of ["f-int", "f-gx", "f-gy", "f-talk", "f-hold"]) $(`#${id}`).addEventListener("change", applyFace);
  // designer: every parameter of faces.json, live; sends FACE_PARAMS to the robot
  $("#fparams").innerHTML = faces.params.map((p) => `<label class="j"><span>${p.name}</span><output id="fp-${p.name}"></output>` +
    `<input type="range" data-param="${p.name}" min="${p.min}" max="${p.max}" step="${(p.max - p.min) / 200}" title="${p.doc}"></label>`).join("");
  let pTimer = null;
  $("#fparams").oninput = (ev) => {
    const k = ev.target.dataset.param;
    if (!k) return;
    TFT.source = "manual"; TFT.mode = 0;
    TFT.anim.override = { ...TFT.anim.override, [k]: +ev.target.value };
    $(`#fp-${k}`).value = (+ev.target.value).toFixed(2);
    clearTimeout(pTimer);
    pTimer = setTimeout(() => api("/api/display/params", { params: TFT.anim.override, blend_ms: 80 }).then(sentNote).catch(() => {}), 60);
  };
  $("#f-reset").onclick = () => { TFT.anim.override = {}; syncParamSliders(); sendFace(); };
  $("#f-copy").onclick = () => {
    const base = lib.resolve("neutral"), cur = { ...TFT.anim.current(performance.now()), ...TFT.anim.override }, diff = {};
    for (const k of lib.params) if (Math.abs((cur[k] ?? 0) - (base[k] ?? 0)) > 1e-3) diff[k] = +(+cur[k]).toFixed(3);
    const json = JSON.stringify({ name: "my_emotion", p: diff });
    navigator.clipboard?.writeText(json).catch(() => {});
    $("#dsent").textContent = `Copied → paste into config/faces.json "emotions": ${json}`;
  };
  syncParamSliders();
  // modes
  $$("#dmode button").forEach((b) => (b.onclick = () => setMode(+b.dataset.mode, true)));
  // image
  $("#img-send").onclick = () => {
    const f = $("#img-file").files[0];
    if (!f) return alert("Choose an image first.");
    const rd = new FileReader();
    rd.onload = () => busy($("#img-send"), async () => {
      const r = await api("/api/display/image", { data_url: rd.result, fit: $("#img-fit").value, store: $("#img-store").checked, image_id: 1 });
      const im = new Image(); im.src = r.preview; await im.decode();
      TFT.image = im; setMode(1, false); sentNote(r);
      $("#dsent").textContent += ` · ${(r.bytes / 1024).toFixed(1)} kB JPEG`;
    });
    rd.readAsDataURL(f);
  };
  // text
  $("#txt-dur").addEventListener("input", (e) => (e.target.nextElementSibling.value = +e.target.value ? e.target.value : "∞"));
  $("#txt-send").onclick = () => {
    TFT.text = { text: $("#txt").value, size: +$("#txt-size").value, fg: $("#txt-fg").value, bg: $("#txt-bg").value };
    setMode(2, false);
    api("/api/display/text", { text: $("#txt").value, size: +$("#txt-size").value, fg: $("#txt-fg").value, bg: $("#txt-bg").value,
      duration_ms: +$("#txt-dur").value * 1000 }).then(sentNote).catch((e) => alert(e.message));
  };
  // camera
  $("#cam-on").onclick = () => api("/api/robot/camera", { on: true, fps: +$("#cam-fps").value }).then(() => setMode(3, true)).catch((e) => alert(e.message));
  $("#cam-off").onclick = () => api("/api/robot/camera", { on: false }).catch(() => {});
  $("#cam-snap").onclick = () => api("/api/robot/camera", { snapshot: true }).catch((e) => alert(e.message));
  // brightness
  $("#bright").addEventListener("input", (e) => (e.target.nextElementSibling.value = e.target.value));
  $("#bright").addEventListener("change", (e) => api("/api/display/brightness", { percent: +e.target.value }).catch(() => {}));
}
function syncParamSliders() {
  if (!TFT.lib) return;
  const cur = { ...TFT.lib.withIntensity(D.emotion, +$("#f-int").value), ...TFT.anim.override };
  $$("#fparams input").forEach((el) => { const k = el.dataset.param; el.value = cur[k]; $(`#fp-${k}`).value = (+cur[k]).toFixed(2); });
}
function setMode(mode, send) {
  TFT.mode = mode;
  $$("#dmode button").forEach((x) => x.classList.toggle("on", +x.dataset.mode === mode));
  $$(".dpane").forEach((pn) => (pn.hidden = +pn.dataset.dpane !== mode));
  if (send) api("/api/display/mode", { mode }).then(sentNote).catch(() => {});
}
function sentNote(r) { $("#dsent").textContent = r && r.sent ? "Sent to the robot ✓ (and shown on the twin)" : "Preview only: connect a robot (Device) to send."; }

// ================================================================ playback
const P = { res: null, playing: false, t: 0, t0: 0, tStart: 0, audio: null, loop: false, speed: 1 };
const frameOf = (t) => Math.min(P.res.bodies.length - 1, Math.max(0, Math.round(t * P.res.fps)));
const currentWord = () => (P.res?.words || []).find((w) => P.t >= w.t0 && P.t <= w.t1 + 0.12);

function load(res) {
  if (P.audio) P.audio.pause();
  P.res = res; P.t = 0; P.playing = false;
  P.audio = res.audio ? new Audio(res.audio) : null;
  if (TFT.anim) { TFT.source = "track"; TFT.mode = 0; setMode(0, false); }
  setBodies(res.bodies[0]);
  showReports(res);
  drawTimeline();
  play();
}
function play() {
  if (!P.res) return;
  if (P.t >= P.res.duration - 0.02) P.t = 0;
  P.playing = true; P.tStart = P.t; P.t0 = performance.now();
  if (P.audio) { P.audio.playbackRate = P.speed; P.audio.currentTime = P.t; P.audio.play().catch(() => {}); }
  $("#play").textContent = "❚❚";
}
function pause() { P.playing = false; P.audio?.pause(); $("#play").textContent = "▶"; }
function seek(t) {
  if (!P.res) return;
  P.t = Math.max(0, Math.min(t, P.res.duration));
  if (P.audio) P.audio.currentTime = P.t;
  P.tStart = P.t; P.t0 = performance.now();
  tick(true);
}
function tick(force) {
  const r = P.res;
  if (!r) return;
  if (P.playing) {
    P.t = P.audio && !P.audio.paused ? P.audio.currentTime : P.tStart + (performance.now() - P.t0) / 1000 * P.speed;
    if (P.t >= r.duration) { if (P.loop) { P.t = 0; play(); } else { P.t = r.duration; pause(); } }
  } else if (!force) {
    return;
  }
  const k = frameOf(P.t);
  setBodies(r.bodies[k]);
  V.setCop(r.sim?.cop?.length ? [r.sim.cop[k][0] / 1000, r.sim.cop[k][1] / 1000] : null);
  $("#clock").textContent = `${P.t.toFixed(2).padStart(5, "0")} / ${r.duration.toFixed(2).padStart(5, "0")}`;
  $("#scrub").value = Math.round(1000 * P.t / r.duration);
  const w = currentWord();
  $("#caption").textContent = w ? w.w : "";
  const tilt = r.sim ? r.sim.tilt[k].toFixed(2) : "–", zmp = r.sim?.margin?.length ? r.sim.margin[k].toFixed(1) : "–",
    load = r.sim?.load_pct?.length ? r.sim.load_pct[k].toFixed(0) : "–";
  $("#readout").innerHTML = `<b>${(r.label || "").slice(0, 34)}</b><br>T <b>${P.t.toFixed(2)}</b> s · F <b>${String(k).padStart(4, "0")}</b>/${r.bodies.length - 1} · ${r.fps} Hz<br>` +
    `TILT <b>${tilt}</b>° · ZMP <b>${zmp}</b> mm · LOAD <b>${load}</b>%`;
  drawTimeline();
}

// ================================================================ timeline
const TRACKS = [["SPEECH", 16], ["GESTURE", 16], ["FACE", 14], ["BEAT", 8], ["ARMS", 44], ["GAZE", 18], ["TILT", 18], ["ZMP", 18]];
function drawTimeline() {
  const t0 = performance.now();
  try { drawTimelineInner(); } finally { P.tlMs = (P.tlMs || 0) + performance.now() - t0; }
}
function drawTimelineInner() {
  const c = $("#timeline"), r = P.res;
  const dpr = devicePixelRatio, W = c.clientWidth, H = c.clientHeight;
  if (!W) return;
  if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) { c.width = W * dpr; c.height = H * dpr; }
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  const mono = css("--mono") || "monospace", gutter = 64, x0 = gutter + 6, x1 = W - 10;
  g.font = `600 9.5px ${mono}`; g.textBaseline = "middle";
  // track bands
  let y = 6;
  const rows = {};
  for (const [name, h] of TRACKS) {
    rows[name] = [y, h];
    g.fillStyle = css("--faint"); g.fillText(name, 8, y + h / 2);
    g.strokeStyle = css("--line-2"); g.beginPath(); g.moveTo(x0, y + h + 2.5); g.lineTo(x1, y + h + 2.5); g.stroke();
    y += h + 5;
  }
  if (!r) { g.fillStyle = css("--faint"); g.fillText("no motion loaded", x0 + 8, rows.ARMS[0] + 20); return; }
  const X = (t) => x0 + (x1 - x0) * t / r.duration;
  // speech
  g.font = `11px ${mono}`;
  let xEnd = -1e9;
  for (const w of r.words || []) {
    const xw = X(w.t0);
    g.fillStyle = css("--line"); g.fillRect(xw, rows.SPEECH[0] + 13, Math.max(1, X(w.t1) - xw), 2);
    if (xw < xEnd + 4) continue;
    g.fillStyle = css("--muted"); g.fillText(w.w, xw, rows.SPEECH[0] + 6);
    xEnd = xw + g.measureText(w.w).width;
  }
  // gestures + beats
  for (const e of r.events || []) {
    if (e.type === "semantic") {
      const [yy, hh] = rows.GESTURE;
      g.fillStyle = css("--accent"); g.fillRect(X(e.t) - 1, yy, 2, hh);
      g.fillText(`${e.gesture}${e.variant ? ":" + e.variant : ""}`, X(e.t) + 4, yy + hh / 2);
    } else if (e.type === "push") {  // RL replay: when the pelvis was kicked
      const [yy, hh] = rows.GESTURE;
      g.fillStyle = css("--bad"); g.fillRect(X(e.t) - 1, yy, 2, hh);
      g.fillText("push", X(e.t) + 4, yy + hh / 2);
    } else if (e.type === "beat") {
      const [yy, hh] = rows.BEAT;
      g.fillStyle = css("--muted"); g.fillRect(X(e.t) - 0.5, yy + hh * (1 - Math.min(1, e.prom / 2)), 1.5, hh * Math.min(1, e.prom / 2));
    }
  }
  // face track: one segment per key, labelled with the emotion
  (r.face || []).forEach((k, i) => {
    const [yy, hh] = rows.FACE, xa = X(k.t), xb = X(i + 1 < r.face.length ? r.face[i + 1].t : r.duration);
    g.fillStyle = css("--line"); g.fillRect(xa, yy + 2, Math.max(1, xb - xa - 2), hh - 4);
    g.fillStyle = css("--text");
    if (xb - xa > 34) g.fillText(k.emotion, xa + 3, yy + hh / 2);
  });
  const curve = (row, values, lo, hi, color, width = 1.3) => {
    const [yy, hh] = rows[row];
    g.strokeStyle = color; g.lineWidth = width; g.beginPath();
    values.forEach((v, k) => { const px = X(r.t[k]), py = yy + hh - (Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo) * hh; k ? g.lineTo(px, py) : g.moveTo(px, py); });
    g.stroke();
  };
  // anatomical: 0 l_pitch 1 l_roll 2 l_elbow 3 r_pitch 4 r_roll 5 r_elbow 6 gaze 7 sway
  curve("ARMS", r.anat.map((a) => a[0]), -20, 170, css("--curve-1"));
  curve("ARMS", r.anat.map((a) => a[3]), -20, 170, css("--curve-2"));
  curve("ARMS", r.anat.map((a) => a[2]), -20, 170, css("--curve-3"), 1);
  curve("ARMS", r.anat.map((a) => a[5]), -20, 170, css("--curve-4"), 1);
  curve("GAZE", r.anat.map((a) => a[6]), -15, 15, css("--curve-5"));
  if (r.sim) {
    curve("TILT", r.sim.tilt, 0, r.kind === "policy" ? 60 : 8, css("--warn"));
    if (r.sim.margin.length) curve("ZMP", r.sim.margin, 0, 70, css("--ok"));
  }
  if (r.audio_offset > 0) {
    g.fillStyle = css("--warn"); g.fillRect(X(r.audio_offset), rows.SPEECH[0], 1, H - 10);
    g.fillText("audio", X(r.audio_offset) + 3, H - 8);
  }
  g.fillStyle = css("--bad"); g.fillRect(X(P.t) - 1, 0, 2, H);
}
$("#timeline").addEventListener("pointerdown", (ev) => {
  if (!P.res) return;
  const rect = ev.target.getBoundingClientRect();
  const x0 = 70, x1 = rect.width - 10;
  seek((ev.clientX - rect.left - x0) / (x1 - x0) * P.res.duration);
});
new ResizeObserver(() => drawTimeline()).observe($("#timeline"));

// ================================================================ inspector
const TILE_DEF = [
  ["clearance", (r) => r.safety.min_clearance_mm, "mm", (v) => v < 4],
  ["balance", (r) => r.safety.min_balance_margin_mm, "mm", (v) => v < 8],
  ["zmp margin", (r) => r.physics?.min_zmp_margin_mm, "mm", (v) => v < 3],
  ["max tilt", (r) => r.physics?.max_tilt_deg, "°", (v) => v > 8],
  ["servo load", (r) => r.physics?.max_load_pct, "%", (v) => v > 85],
  ["tracking", (r) => r.physics?.tracking_rms_deg, "° rms", (v) => v > 6],
];
function showPolicyReport(r) {
  const p = r.policy, m = p.metrics, fell = p.fell_at != null;
  $("#verdict").className = `verdict ${fell ? "fail" : "pass"}`;
  $("#verdict").innerHTML = `<div class="big">${fell ? "FELL" : "STOOD"}</div><div class="small">${fell ? `at ${p.fell_at.toFixed(1)} s · ` : ""}${r.duration.toFixed(1)} s replay · push ${p.push} m/s · ${p.config} physics</div>`;
  const tiles = [["max tilt", Math.max(...r.sim.tilt).toFixed(1), "°", Math.max(...r.sim.tilt) > 30],
    ["tracking", m.track_rms_deg?.toFixed(2), "° rms", m.track_rms_deg > 6], ["pushes", r.events.length, "", false],
    ["loop gap", m.loop_gap_max_mm?.toFixed(2), "mm", m.loop_gap_max_mm > 3]];
  $("#tiles").innerHTML = tiles.map(([k, v, unit, bad]) =>
    `<div class="tile ${bad ? "bad" : ""}"><div class="k">${k}</div><div class="v">${v ?? "–"}<small>${v == null ? "" : unit}</small></div></div>`).join("");
  $("#safety").innerHTML = "<div><span>safety stage</span><span>not applied</span></div><div><span>note</span><span>raw policy output</span></div>";
  const rows = [["policy", p.run === "home" ? "hold home pose" : p.run.split("/").pop()], ["task", p.task], ["IMU", p.imu ? "yes" : "no"],
    ["physics", `${p.config} (${m.dt_ms} ms ${m.integrator})`], ["collisions", m.collisions], ["eq residual", `${m.eq_resid_max_deg?.toFixed(2)}°`],
    ...(p.note ? [["export", p.note]] : [])];
  $("#physics-rep").innerHTML = rows.map(([k, v]) => `<div><span>${k}</span><span>${esc(v)}</span></div>`).join("");
  chip("#chip-safety", null);
  chip("#chip-physics", !fell);
}
function showReports(r) {
  if (r.kind === "policy") return showPolicyReport(r);
  const pass = r.safety.pass && (!r.physics || r.physics.pass);
  $("#verdict").className = `verdict ${pass ? "pass" : "fail"}`;
  $("#verdict").innerHTML = `<div class="big">${pass ? "SAFE" : "BLOCKED"}</div><div class="small">${r.duration.toFixed(1)} s · ${r.bodies.length} frames${r.physics ? ` · physics attempt ${r.physics.attempt}` : " · physics not run"}</div>`;
  $("#tiles").innerHTML = TILE_DEF.map(([k, f, unit, bad]) => {
    const v = f(r);
    return `<div class="tile ${v != null && bad(v) ? "bad" : ""}"><div class="k">${k}</div><div class="v">${v ?? "–"}<small>${v == null ? "" : unit}</small></div></div>`;
  }).join("");
  const rows = (obj, checks) => [
    ...Object.entries(checks || {}).map(([k, v]) => `<div><span>${k.replaceAll("_", " ")}</span><span class="${v ? "pass" : "fail"}">${v ? "PASS" : "FAIL"}</span></div>`),
    ...Object.entries(obj).filter(([k, v]) => typeof v !== "object" && k !== "pass").map(([k, v]) => `<div><span>${k.replaceAll("_", " ")}</span><span>${v}</span></div>`),
  ].join("");
  $("#safety").innerHTML = rows(r.safety, { "all checks": r.safety.pass });
  $("#physics-rep").innerHTML = r.physics ? rows(r.physics, r.physics.checks) : "<div><span>not run</span><span></span></div>";
  chip("#chip-safety", r.safety.pass);
  chip("#chip-physics", r.physics ? r.physics.pass : null);
}
function chip(id, ok) { $(id).className = `chip ${ok == null ? "" : ok ? "ok" : "bad"}`; }

// ================================================================ compose / gestures
async function busy(btn, fn) {
  btn.disabled = true;
  try { return await fn(); } catch (e) { alert(e.message); } finally { btn.disabled = false; }
}
for (const id of ["v", "a", "d"]) $(`#${id}`).addEventListener("input", (e) => (e.target.nextElementSibling.value = (+e.target.value).toFixed(2)));
const compose = () => busy($("#compose"), async () => {
  $("#compose-log").textContent = "planning → safety stage" + ($("#physics").checked ? " → physics twin" : "") + " …";
  const t = performance.now();
  const r = await api("/api/compose", { text: $("#text").value, vad: [+$("#v").value, +$("#a").value, +$("#d").value],
    voice: $("#voice").value || null, physics: $("#physics").checked, use_model: $("#hf").checked });
  $("#compose-log").textContent = `${r.duration.toFixed(1)} s of motion in ${((performance.now() - t) / 1000).toFixed(1)} s\n` +
    (r.events || []).filter((e) => e.type === "semantic").map((e) => `${e.t.toFixed(2).padStart(6)}s  ${(e.gesture + ":" + e.variant).padEnd(16)} "${e.word}"`).join("\n");
  load(r);
});
$("#compose").onclick = compose;
$("#text").addEventListener("keydown", (e) => { if (e.key === "Enter" && e.ctrlKey) { e.preventDefault(); compose(); } });
$("#ask-go").onclick = () => busy($("#ask-go"), async () => {
  const r = await api("/api/llm", { prompt: $("#ask").value });
  $("#text").value = r.reply;
  ["v", "a", "d"].forEach((id, i) => { $(`#${id}`).value = r.vad[i]; $(`#${id}`).nextElementSibling.value = (+r.vad[i]).toFixed(2); });
});
async function initGestures() {
  const g = await api("/api/gestures");
  const render = (f) => ($("#gesture-list").innerHTML = Object.entries(g).filter(([n, v]) => !f || n.includes(f) || v.means.toLowerCase().includes(f)).map(([name, v]) =>
    `<div class="g"><b>${name}</b><div class="m">${v.means}</div>${(v.variants.length ? v.variants : [null]).map((id) =>
      `<button data-g="${name}" data-v="${id ?? ""}">${id ?? name}</button>`).join("")}</div>`).join(""));
  render("");
  $("#gesture-filter").oninput = (e) => render(e.target.value.toLowerCase().trim());
  $("#gesture-list").onclick = (ev) => {
    const b = ev.target.closest("button[data-g]");
    if (b) busy(b, async () => load(await api("/api/gesture", { name: b.dataset.g, variant: b.dataset.v || null })));
  };
}

// ================================================================ pose
let model = null, pose = [], poseTimer = null;
function initPose() {
  pose = model.joints.map((j) => j.home);
  let html = "", grp = "";
  model.joints.forEach((j, i) => {
    if (j.group !== grp) { grp = j.group; html += `<div class="grp">${grp}</div>`; }
    html += `<label class="j"><span>${j.name}</span><output id="jv${i}">${(j.home * DEG).toFixed(1)}°</output>` +
      `<input type="range" data-i="${i}" min="${(j.lower * DEG).toFixed(0)}" max="${(j.upper * DEG).toFixed(0)}" step="0.5" value="${(j.home * DEG).toFixed(1)}"></label>`;
  });
  $("#sliders").innerHTML = html;
  $("#sliders").oninput = (ev) => {
    const i = +ev.target.dataset.i;
    pose[i] = ev.target.value / DEG;
    $(`#jv${i}`).value = `${(+ev.target.value).toFixed(1)}°`;
    sendPose();
  };
  $("#pose-home").onclick = () => { pose = model.joints.map((j) => j.home); syncSliders(); sendPose(); };
  $("#pose-zero").onclick = () => { model.joints.forEach((j, i) => { if (j.group === "arm") pose[i] = 0; }); syncSliders(); sendPose(); };
}
function syncSliders() {
  $$("#sliders input").forEach((el) => { const i = +el.dataset.i; el.value = pose[i] * DEG; $(`#jv${i}`).value = `${(pose[i] * DEG).toFixed(1)}°`; });
}
let poseBusy = false, posePending = false;
async function sendPose() {  // one request in flight; slider moves that arrive meanwhile collapse into one newest request
  if (poseBusy) { posePending = true; return; }
  poseBusy = true;
  try { await sendPoseNow(); } finally {
    poseBusy = false;
    if (posePending) { posePending = false; sendPose(); }
  }
}
async function sendPoseNow() {
  const r = await api("/api/fk", { q: pose, safe: $("#pose-safe").checked });
  P.res = null; pause(); V.setCop(null);
  setBodies(r.bodies);
  if (r.report) $("#pose-report").textContent = `clearance ${r.report.clearance_mm} mm · balance ${r.report.balance_margin_mm} mm` + (r.report.projected ? "\n↳ arm projected clear of the body" : "");
  if ($("#pose-live").checked) api("/api/robot/pose", { q: r.q, move_ms: 200 }).catch(() => {});
  drawTimeline();
}

// ================================================================ RL policies (rl/runs_index.py via /api/policies)
const POL = { data: null, open: null, loaded: false };
const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function polVerdict(run, base) {
  // push-recovery at 0.45 m/s in the sim2sim physics, against holding the home pose
  const t = run.tests?.["0.45"]?.sim2sim, b = base?.["0.45"]?.sim2sim;
  if (!t) return { cls: "", text: "not tested" };
  const delta = b ? t.fall - b.fall : null;
  const cls = delta == null ? "mid" : delta <= 0 ? "good" : delta < 0.3 ? "mid" : "bad";
  return { cls, text: `falls ${(t.fall * 100).toFixed(0)}%${b ? ` (hold-home ${(b.fall * 100).toFixed(0)}%)` : ""}` };
}
function sparkline(canvas, pts, color) {
  const dpr = devicePixelRatio, W = canvas.clientWidth, H = canvas.clientHeight;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const g = canvas.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (!pts || pts.length < 2) { g.fillStyle = css("--faint"); g.font = `10px ${css("--mono")}`; g.fillText("no data", 6, H / 2); return; }
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys), pad = 4;
  const X = (x) => pad + (W - 2 * pad) * (x - x0) / Math.max(x1 - x0, 1e-9), Y = (y) => H - pad - (H - 2 * pad) * (y - y0) / Math.max(y1 - y0, 1e-9);
  g.strokeStyle = color; g.lineWidth = 1.4; g.beginPath();
  pts.forEach((p, i) => (i ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1]))));
  g.stroke();
  g.fillStyle = css("--muted"); g.font = `9.5px ${css("--mono")}`;
  g.fillText(y1.toFixed(2), 4, 10); g.fillText(y0.toFixed(2), 4, H - 3);
  g.textAlign = "right"; g.fillText(`${ys[ys.length - 1].toFixed(2)} ▸ ${x1}`, W - 4, 10); g.textAlign = "left";
}
function renderPolicies() {
  const d = POL.data;
  if (!d) return;
  const task = $("#pol-task").value, hide = $("#pol-hide-tiny").checked;
  const tiny = (r) => (r.progress_unit === "steps" ? r.iterations < 20000 : r.iterations < 20);
  const runs = d.runs.filter((r) => (!task || r.task === task) && !(hide && tiny(r)));
  const b = d.baseline?.["0.45"]?.sim2sim;
  $("#pol-note").textContent = `${runs.length} of ${d.runs.length} runs shown · scanned ${d.generated}`;
  $("#pol-baseline").innerHTML = b ? `<b>Baseline: hold the home pose</b> (zero action)<br>push 0.45 m/s: falls ${(b.fall * 100).toFixed(0)}% · tracking ${b.track_rms_deg}° · tilt max ${b.tilt_max_deg}°<br>A learned policy must beat this to be worth running.<br><button data-replay="home">▶ Replay baseline</button>` : '<button data-replay="home">▶ Replay baseline (hold home pose)</button>';
  $("#pol-list").innerHTML = runs.map((r) => {
    const v = polVerdict(r, d.baseline), open = POL.open === r.id;
    return `<div class="pol ${open ? "open" : ""}" data-id="${esc(r.id)}">
      <div class="t"><b title="${esc(r.id)}">${esc(r.name)}</b><span class="badge ${v.cls}">${esc(v.text)}</span></div>
      <div class="m"><span class="badge">${esc(r.trainer)}</span><span class="badge">${esc(r.task)}</span><span class="badge ${r.imu ? "imu" : ""}">${r.imu ? "IMU" : "no IMU"}</span>
        ${r.iterations} ${r.progress_unit} · reward ${r.last.reward?.toFixed(1) ?? "–"} · ${esc(r.created)}</div>
      ${open ? polDetail(r, d.baseline) : ""}</div>`;
  }).join("") || '<p class="hint">No runs yet. Train one: <code>make rl-smoke</code> or <code>make rl-train-stand</code>.</p>';
  if (POL.open) {
    const r = d.runs.find((x) => x.id === POL.open);
    const el = document.querySelector(".pol.open");
    if (r && el) el.querySelectorAll("canvas[data-curve]").forEach((c, i) => sparkline(c, r.curves[c.dataset.curve], [css("--curve-1"), css("--curve-2"), css("--curve-5")][i % 3]));
  }
}
function polDetail(r, base) {
  const rows = ["0.45", "0.6", "0.75"].filter((k) => r.tests?.[k]).map((k) => {
    const t = r.tests[k].sim2sim ?? r.tests[k].train, bb = base?.[k]?.sim2sim;
    return `<tr><td>${k} m/s</td><td>${(t.fall * 100).toFixed(0)}%</td><td>${bb ? (bb.fall * 100).toFixed(0) + "%" : "–"}</td><td>${t.track_rms_deg}°</td><td>${t.tilt_max_deg}°</td></tr>`;
  }).join("");
  const ex = r.exported;
  return `<div class="detail">
    <button class="primary" data-replay="${esc(r.id)}">▶ Replay in 3D viewer</button>
    ${[["reward", "mean reward"], ["episode_s", "episode length"], ["fall_rate", "fall rate (training)"]].filter(([k]) => r.curves[k]).map(([k, l]) =>
      `<div class="cl">${l}</div><canvas data-curve="${k}"></canvas>`).join("")}
    ${rows ? `<div class="cl">push recovery (sim2sim)</div><table><tr><th>push</th><th>falls</th><th>hold-home</th><th>track rms</th><th>tilt max</th></tr>${rows}</table>` : '<p class="hint">No push test recorded. Run <code>make rl-sim2sim</code> for this run.</p>'}
    <div class="m">export: ${ex.onnx ? "ONNX ✓" : "ONNX –"} · ${ex.p4_header ? "P4 header ✓" : "P4 header –"} · ${ex.meta ? "meta ✓" : "not exported"}${r.mlp_macs ? ` · ${(r.mlp_macs / 1000).toFixed(0)}k MACs/step` : ""}</div>
    <div class="m">checkpoints: ${r.checkpoints}${r.final_model ? " + final" : ""} · obs ${r.obs_dim ?? "?"} → actions ${r.act_dim ?? "?"}</div>
    <div class="m">${esc(r.id)}</div></div>`;
}
async function loadPolicies(refresh = false) {
  $("#pol-note").textContent = "Scanning rl/logs …";
  try {
    POL.data = await api("/api/policies" + (refresh ? "?refresh=1" : ""));
    const tasks = [...new Set(POL.data.runs.map((r) => r.task))];
    const cur = $("#pol-task").value;
    $("#pol-task").innerHTML = '<option value="">All tasks</option>' + tasks.map((t) => `<option ${t === cur ? "selected" : ""}>${esc(t)}</option>`).join("");
    POL.loaded = true;
    const hp = new URLSearchParams(location.hash.slice(1));  // demo hook: #tab=policies&polopen=3 opens the 4th run
    if (hp.has("polopen") && POL.open == null) POL.open = POL.data.runs[+hp.get("polopen")]?.id ?? null;
    renderPolicies();
  } catch (e) { $("#pol-note").textContent = "Could not read policies: " + e.message; }
}
$("#pol-refresh").onclick = () => loadPolicies(true);
$("#pol-task").onchange = renderPolicies;
$("#pol-hide-tiny").onchange = renderPolicies;
for (const id of ["rp-push", "rp-sec", "rp-vx"]) $(`#${id}`).addEventListener("input", (e) => (e.target.nextElementSibling.value = (+e.target.value).toFixed(id === "rp-sec" ? 0 : 2)));
async function replayPolicy(btn) {
  const label = btn.textContent;
  btn.textContent = "Running the policy …";
  try {
    const r = await api("/api/policies/replay", { run: btn.dataset.replay, push: +$("#rp-push").value, seconds: +$("#rp-sec").value,
      config: $("#rp-cfg").value, seed: +$("#rp-seed").value, vx: +$("#rp-vx").value });
    load(r);
  } finally { btn.textContent = label; }
}
const polClick = (ev) => {
  const btn = ev.target.closest("button[data-replay]");
  if (btn) { ev.stopPropagation(); busy(btn, () => replayPolicy(btn)); return true; }
  return false;
};
$("#pol-baseline").onclick = polClick;
$("#pol-list").onclick = (ev) => {
  if (polClick(ev)) return;
  const card = ev.target.closest(".pol");
  if (!card) return;
  POL.open = POL.open === card.dataset.id ? null : card.dataset.id;
  renderPolicies();
};

// ================================================================ device
$("#connect").onclick = () => busy($("#connect"), async () => {
  const r = await api("/api/robot/connect", { host: $("#host").value, port: +$("#port").value });
  $("#robot-state").textContent = `connected: ${r.info.name} fw ${r.info.fw_ver} · rtt ${r.rtt_ms} ms`;
});
for (const [id, act] of [["#torque", "torque"], ["#home", "home"], ["#hold", "hold"], ["#estop", "estop"]])
  $(id).onclick = () => api(`/api/robot/${act}`, {}).catch((e) => alert(e.message));
$("#robot-play").onclick = () => (P.res ? api("/api/robot/play", { id: P.res.id }).then(play).catch((e) => alert(e.message)) : alert("Generate a motion first."));
setInterval(async () => {
  try {
    const s = await api("/api/robot/state");
    $("#sb-server").className = "ok";
    if (!s.connected) { chip("#chip-robot", null); $("#sb-link").textContent = "link offline"; return; }
    const f = s.state?.flags || {};
    chip("#chip-robot", !f.estop);
    if (s.camera_frame != null && s.camera_frame !== TFT.camFrame) {
      TFT.camFrame = s.camera_frame;
      const url = `/api/robot/camera.jpg?f=${s.camera_frame}`;
      TFT.camImg.src = url; $("#cam-img").src = url;
    }
    $("#sb-link").textContent = `link ${s.info?.name ?? ""} · rtt ${s.rtt_ms ?? "?"} ms${f.estop ? " · E-STOP" : f.streaming ? " · streaming" : ""}`;
    if (s.state && $('[data-panel="device"]').classList.contains("on"))
      $("#robot-state").textContent = "joint                    meas     cmd\n" + model.joints.map((j, i) =>
        `${j.name.padEnd(22)} ${s.state.q_meas_deg[i].toFixed(1).padStart(7)} ${s.state.q_cmd_deg[i].toFixed(1).padStart(7)}`).join("\n") +
        `\n\nvbat ${s.state.vbat_mv} mV · temp ${s.state.temp_max_c} °C\nflags ${Object.entries(f).filter(([, v]) => v).map(([k]) => k).join(" ")}`;
  } catch { $("#sb-server").className = ""; }
}, 400);

// ================================================================ export
$("#export").onclick = () => (P.res ? busy($("#export"), async () => {
  const r = await api("/api/export", { id: P.res.id, name: $("#export-name").value });
  $("#export-log").textContent = `${r.zact}  (${r.frames} frames)` + (r.audio ? `\n${r.audio}` : "");
}) : alert("Generate a motion first."));

// ================================================================ shell
$$("#tabs button").forEach((b) => (b.onclick = () => {
  $$("#tabs button").forEach((x) => x.classList.toggle("on", x === b));
  $$(".panel").forEach((p) => p.classList.toggle("on", p.dataset.panel === b.dataset.tab));
  if (b.dataset.tab === "policies" && !POL.loaded) loadPolicies();
}));
$("#play").onclick = () => (P.playing ? pause() : play());
$("#to-start").onclick = () => seek(0);
$("#step-back").onclick = () => { pause(); seek(P.t - 0.02); };
$("#step-fwd").onclick = () => { pause(); seek(P.t + 0.02); };
$("#loop").onclick = (e) => { P.loop = !P.loop; e.currentTarget.classList.toggle("on", P.loop); };
$("#scrub").oninput = (e) => { if (P.res) { pause(); seek(e.target.value / 1000 * P.res.duration); } };
$$("#speed button").forEach((b) => (b.onclick = () => {
  P.speed = +b.dataset.speed;
  $$("#speed button").forEach((x) => x.classList.toggle("on", x === b));
  if (P.playing) { pause(); play(); }
}));
$$("[data-view]").forEach((b) => (b.onclick = () => V.view(b.dataset.view)));
$("#ov-balance").onclick = (e) => { V.showBalance = !V.showBalance; e.currentTarget.classList.toggle("on", V.showBalance); tick(true); };
$("#ov-face").onclick = (e) => { V.showFace = !V.showFace; e.currentTarget.classList.toggle("on", V.showFace); };
document.addEventListener("keydown", (ev) => {
  if (ev.target.matches("input, textarea, select")) return;
  if (ev.code === "Space") { ev.preventDefault(); P.playing ? pause() : play(); }
  if (ev.code === "ArrowRight") { pause(); seek(P.t + (ev.shiftKey ? 1 : 0.02)); }
  if (ev.code === "ArrowLeft") { pause(); seek(P.t - (ev.shiftKey ? 1 : 0.02)); }
  if (ev.code === "Home") seek(0);
});

// ---- errors: shown in the viewport and printed in the `make studio` terminal (POST /api/clientlog) ----
function reportError(where, err) {
  const msg = `${where}: ${err?.message || err}`;
  console.error(msg, err);
  fetch("/api/clientlog", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ where, message: String(err?.message || err), stack: String(err?.stack || ""), agent: navigator.userAgent }) }).catch(() => {});
  let b = $("#viewer-error");
  if (!b) { b = document.createElement("div"); b.id = "viewer-error"; b.className = "viewer-error"; $("#viewport").appendChild(b); }
  b.innerHTML += `<div>${msg.replace(/</g, "&lt;")}</div>`;
}
window.addEventListener("error", (e) => reportError("script", e.error || e.message));
window.addEventListener("unhandledrejection", (e) => reportError("promise", e.reason));

function webglStatus() {
  const c = document.createElement("canvas");
  const gl = c.getContext("webgl2") || c.getContext("webgl");
  if (!gl) return null;
  const d = gl.getExtension("WEBGL_debug_renderer_info");
  return gl.getParameter(d ? d.UNMASKED_RENDERER_WEBGL : gl.RENDERER);
}

(async () => {
  model = await api("/api/model");
  $$("[data-theme-set]").forEach((b) => b.classList.toggle("on", b.dataset.themeSet === document.documentElement.dataset.theme));
  // a broken 3D view must not take the rest of the studio down with it
  const gpu = webglStatus();
  if (!gpu) {
    const brave = !!navigator.brave;
    reportError("3D view", new Error(brave
      ? "Brave is blocking WebGL on this page. Fix: click the Brave lion icon in the address bar → switch Shields DOWN for " +
        "localhost (or set 'Block fingerprinting' to Disabled), then reload. If it still fails, open brave://gpu: WebGL must " +
        "not say 'Disabled' (brave://flags → 'Override software rendering list' → Enabled → relaunch). Or run `make app`."
      : "this browser has no WebGL. Fix: Settings → System → turn on 'Use graphics acceleration when available' and relaunch; " +
        "check chrome://gpu (WebGL must say 'Hardware accelerated'); or run `make app` (software fallback) or use Firefox."));
  } else {
    try {
      initViewer(model, gpu);
      setBodies(model.home_bodies);
    } catch (e) {
      reportError("3D view", e);
    }
  }
  for (const [name, fn] of [["gestures", initGestures], ["pose", initPose]]) {
    try { fn(); } catch (e) { reportError(name, e); }
  }
  try { initDisplay(await api("/api/faces")); } catch (e) { reportError("display", e); }
  drawTimeline();
  $("#sb-model").textContent = `${model.joints.length} joints · ${model.bodies.length - 1} bodies · chest TFT ${model.display ? model.display.opening_mm[0] + " mm " + model.display.shape : "none"}`;
  const st = await api("/api/status");
  for (const v of st.voices) $("#voice").insertAdjacentHTML("beforeend", `<option value="${v}">${v}</option>`);
  // URL hooks for demos / screenshots: #theme=dark&compose&seek=1.2&view=front
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.get("theme")) setTheme(h.get("theme"));
  if (h.get("tab")) $(`#tabs [data-tab="${h.get("tab")}"]`)?.click();
  if (h.get("emotion")) $(`#emotions [data-emo="${h.get("emotion")}"]`)?.click();
  if (h.has("compose")) {
    await compose();
    if (h.has("seek")) { pause(); seek(+h.get("seek")); }
  }
  if (h.has("replay")) {  // demo hook: #tab=policies&replay=3&push=0.6&seek=6.5  (replay=home for the baseline)
    await loadPolicies();
    const run = h.get("replay") === "home" ? "home" : POL.data.runs[+h.get("replay")].id;
    load(await api("/api/policies/replay", { run, push: +(h.get("push") ?? 0.45), seconds: 8, seed: 1, config: "train", vx: 0.1 }));
    if (h.has("seek")) { pause(); seek(+h.get("seek")); }
  }
  window.__ready = true;
})();
