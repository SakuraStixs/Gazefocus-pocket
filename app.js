// GazeFocus Pocket: the GazeFocus pipeline in a phone browser.
// Front camera -> MediaPipe Face Landmarker -> eye/head features (as in gazefocus/features.py)
// -> eye-shape profile + blink guard -> per-user ridge regression (as in gazefocus/model.py)
// -> fixation filter + spring glide (as in gazefocus/smoothing.py) -> dwell to focus, undo,
// and learning from taps (as in gazefocus/learning.py, drift part).
import { FaceLandmarker, FilesetResolver } from "./vendor/vision_bundle.js";

const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => Math.min(Math.max(v, a), b);
const median = (a) => { if (!a.length) return NaN; const s = [...a].sort((p, q) => p - q); return s[s.length >> 1]; };
const pct = (a, p) => { const s = [...a].sort((x, y) => x - y); return s[clamp(Math.round((p / 100) * (s.length - 1)), 0, s.length - 1)]; };
const now = () => performance.now() / 1000;

// ------------------------------------------------------------------ features
const EYE_L = { a: 33, b: 133, up: 159, lo: 145, iris: [468, 469, 470, 471, 472] }; // image-left eye (your right)
const EYE_R = { a: 362, b: 263, up: 386, lo: 374, iris: [473, 474, 475, 476, 477] };
const FLOOR = [0.01, 0.01, 0.03, 0.03, 2, 2, 1.5, 1.5, 2]; // ex ey bx by yaw pitch tx ty tz
const QUAD = [[0, 0], [1, 1], [0, 1]];
const LAMBDAS = [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1, 3, 10];

function eyeGeom(P, e) {
  const A = P[e.a], B = P[e.b], dx = B[0] - A[0], dy = B[1] - A[1], w = Math.hypot(dx, dy);
  if (w < 1e-6) return null;
  const ux = dx / w, uy = dy / w, nx = -uy, ny = ux, cx = (A[0] + B[0]) / 2, cy = (A[1] + B[1]) / 2;
  let ix = 0, iy = 0;
  for (const k of e.iris) { ix += P[k][0]; iy += P[k][1]; }
  ix /= e.iris.length; iy /= e.iris.length;
  const open = Math.hypot(P[e.lo][0] - P[e.up][0], P[e.lo][1] - P[e.up][1]) / w;
  return { ex: ((ix - cx) * ux + (iy - cy) * uy) / w, ey: ((ix - cx) * nx + (iy - cy) * ny) / w, open, w };
}
function eyeWeights(wl, wr) { // the nearer (wider-looking) eye counts more when you turn
  const big = Math.max(wl, wr, 1e-9), ratio = Math.min(wl, wr) / big;
  const t = clamp((0.92 - ratio) / (0.92 - 0.55), 0, 1), far = 0.5 * (1 - t * t * (3 - 2 * t));
  return wl >= wr ? [1 - far, far] : [far, 1 - far];
}
let matrixLayout = null; // "col" (translation at 12..14) or "row" (at 3, 7, 11)
function headPose(m) {
  const d = m.data;
  if (!matrixLayout) matrixLayout = Math.abs(d[3]) + Math.abs(d[7]) + Math.abs(d[11]) < 1e-6 ? "col" : "row";
  const at = matrixLayout === "col" ? (r, c) => d[c * 4 + r] : (r, c) => d[r * 4 + c];
  const n = Math.hypot(at(0, 2), at(1, 2), at(2, 2)) || 1;
  const f = [at(0, 2) / n, at(1, 2) / n, at(2, 2) / n];
  const yaw = Math.atan2(f[0], f[2]) * 180 / Math.PI, pitch = Math.atan2(-f[1], Math.hypot(f[0], f[2])) * 180 / Math.PI;
  return [yaw, pitch, at(0, 3), at(1, 3), at(2, 3)];
}
function features(res, W, H) {
  if (!res.faceLandmarks || !res.faceLandmarks.length) return null;
  const P = res.faceLandmarks[0].map((p) => [p.x * W, p.y * H]);
  if (P.length < 478) return null;
  const l = eyeGeom(P, EYE_L), r = eyeGeom(P, EYE_R);
  if (!l || !r) return null;
  const [wl, wr] = eyeWeights(l.w, r.w);
  const bs = {};
  if (res.faceBlendshapes && res.faceBlendshapes.length) for (const c of res.faceBlendshapes[0].categories) bs[c.categoryName] = c.score;
  const g = (k) => bs[k] || 0;
  // image-left eye is MediaPipe's "Right"
  const hl = g("eyeLookInRight") - g("eyeLookOutRight"), vl = g("eyeLookUpRight") - g("eyeLookDownRight");
  const hr = g("eyeLookOutLeft") - g("eyeLookInLeft"), vr = g("eyeLookUpLeft") - g("eyeLookDownLeft");
  const head = res.facialTransformationMatrixes && res.facialTransformationMatrixes.length
    ? headPose(res.facialTransformationMatrixes[0]) : [0, 0, 0, 0, 0];
  return {
    x: [wl * l.ex + wr * r.ex, wl * l.ey + wr * r.ey, wl * hl + wr * hr, wl * vl + wr * vr, ...head],
    open: wl * l.open + wr * r.open, blink: wl * g("eyeBlinkRight") + wr * g("eyeBlinkLeft"), P,
  };
}

// ------------------------------------------------------------------ eye shape profile
class EyeProfile {
  constructor(o = {}) { Object.assign(this, { closedB: 0.5, partialB: 0.35, closedO: 0.08, partialO: 0.13, fitted: false, openMed: null }, o); }
  state(f) {
    if (f.blink >= this.closedB || f.open <= this.closedO) return "closed";
    if (f.blink >= this.partialB || f.open <= this.partialO) return "partial";
    return "open";
  }
  static fit(stats) {
    if (stats.length < 60) return new EyeProfile();
    const medO = median(stats.map((s) => s[1]));
    const keep = stats.filter((s) => s[0] < 0.7 && s[1] > 0.5 * medO);
    if (keep.length < 40) return new EyeProfile();
    const b97 = pct(keep.map((s) => s[0]), 97), o03 = pct(keep.map((s) => s[1]), 3);
    const closedB = clamp(b97 + 0.25, 0.4, 0.9), partialB = clamp(b97 + 0.1, 0.25, closedB - 0.05);
    const closedO = clamp(0.45 * o03, 0.04, 0.2), partialO = clamp(0.72 * o03, closedO + 0.01, 0.3);
    return new EyeProfile({ closedB, partialB, closedO, partialO, fitted: true, openMed: medO });
  }
}

// ------------------------------------------------------------------ ridge regression (per-user model)
function solve(A, b) { // Gaussian elimination, A n x n, b n x m
  const n = A.length, m = b[0].length, M = A.map((r, i) => [...r, ...b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const d = M[c][c] || 1e-12;
    for (let r = 0; r < n; r++) if (r !== c) {
      const k = M[r][c] / d;
      if (k) for (let j = c; j < n + m; j++) M[r][j] -= k * M[c][j];
    }
  }
  return M.map((r, i) => r.slice(n).map((v) => v / (M[i][i] || 1e-12)));
}
function design(x, mu, sd) {
  const z = x.map((v, i) => clamp((v - mu[i]) / sd[i], -5, 5));
  return [...z, ...QUAD.map(([i, j]) => z[i] * z[j])];
}
function ridge(Z, Y, w, lam) {
  const n = Z[0].length, sw = w.reduce((a, b) => a + b, 0);
  const zm = Array(n).fill(0), ym = [0, 0];
  Z.forEach((z, k) => { for (let i = 0; i < n; i++) zm[i] += w[k] * z[i] / sw; ym[0] += w[k] * Y[k][0] / sw; ym[1] += w[k] * Y[k][1] / sw; });
  const A = Array.from({ length: n }, () => Array(n).fill(0)), B = Array.from({ length: n }, () => [0, 0]);
  Z.forEach((z, k) => {
    const zc = z.map((v, i) => v - zm[i]), yc = [Y[k][0] - ym[0], Y[k][1] - ym[1]];
    for (let i = 0; i < n; i++) {
      const wi = w[k] * zc[i] / sw;
      for (let j = 0; j < n; j++) A[i][j] += wi * zc[j];
      B[i][0] += wi * yc[0]; B[i][1] += wi * yc[1];
    }
  });
  for (let i = 0; i < n; i++) A[i][i] += lam;
  const Wt = solve(A, B);
  const b0 = [ym[0] - zm.reduce((s, v, i) => s + v * Wt[i][0], 0), ym[1] - zm.reduce((s, v, i) => s + v * Wt[i][1], 0)];
  return { Wt, b0 };
}
function fitModel(samples, size) { // samples: [{x, u, v, g}]
  const X = samples.map((s) => s.x), dim = X[0].length;
  const groups = [...new Set(samples.map((s) => s.g))];
  const cnt = {}; samples.forEach((s) => { cnt[s.g] = (cnt[s.g] || 0) + 1; });
  const w = samples.map((s) => 1 / cnt[s.g]);
  const mu = Array(dim).fill(0), sd = Array(dim).fill(0), sw = w.reduce((a, b) => a + b, 0);
  X.forEach((x, k) => x.forEach((v, i) => { mu[i] += w[k] * v / sw; }));
  X.forEach((x, k) => x.forEach((v, i) => { sd[i] += w[k] * (v - mu[i]) ** 2 / sw; }));
  for (let i = 0; i < dim; i++) sd[i] = Math.max(Math.sqrt(sd[i]), FLOOR[i]);
  const Z = X.map((x) => design(x, mu, sd)), Y = samples.map((s) => [s.u, s.v]);
  let best = null;
  for (const lam of LAMBDAS) { // leave-one-point-out picks the regularisation
    const errs = [];
    for (const g of groups) {
      const tr = samples.map((s, k) => k).filter((k) => samples[k].g !== g);
      const te = samples.map((s, k) => k).filter((k) => samples[k].g === g);
      const m = ridge(tr.map((k) => Z[k]), tr.map((k) => Y[k]), tr.map((k) => w[k]), lam);
      errs.push(median(te.map((k) => {
        const p = predictZ(m, Z[k]); return Math.hypot((p[0] - Y[k][0]) * size[0], (p[1] - Y[k][1]) * size[1]);
      })));
    }
    const e = errs.reduce((a, b) => a + b, 0) / errs.length;
    if (!best || e < best.e) best = { e, lam };
  }
  const m = ridge(Z, Y, w, best.lam);
  return { ...m, mu, sd, cv: best.e };
}
function predictZ(m, z) { return [m.b0[0] + z.reduce((s, v, i) => s + v * m.Wt[i][0], 0), m.b0[1] + z.reduce((s, v, i) => s + v * m.Wt[i][1], 0)]; }
const predict = (m, x) => predictZ(m, design(x, m.mu, m.sd));

// ------------------------------------------------------------------ fixation filter + spring (as on the desktop, phone-sized radius)
class Fix {
  constructor() { this.ms = 20 * 20; this.reset(); }
  reset() { this.anchor = null; this.out = null; this.n = 0; this.cand = []; this.t = null; }
  get radius() { return clamp(2.2 * Math.sqrt(this.ms), 28, 110); }
  get deadband() { return clamp(0.15 * this.radius, 3, 10); }
  interrupt() { this.cand = []; }
  update(x, y, t) {
    const dt = this.t == null ? 1 / 30 : clamp(t - this.t, 1e-3, 0.2); this.t = t;
    if (!this.anchor) { this.n = 1; this.anchor = this.out = [x, y]; return this.out; }
    const [ax, ay] = this.anchor, d = Math.hypot(x - ax, y - ay);
    if (d <= this.radius) {
      this.n++; this.ms += (d * d - this.ms) * (1 - Math.exp(-dt / 1.1));
      const a = Math.max(1 / this.n, 1 - Math.exp(-dt / 0.8));
      this.anchor = [ax + (x - ax) * a, ay + (y - ay) * a]; this.cand = [];
      if (Math.hypot(this.anchor[0] - this.out[0], this.anchor[1] - this.out[1]) > this.deadband) this.out = this.anchor;
      return this.out;
    }
    if (this.cand.length) {
      const cx = this.cand.reduce((s, p) => s + p[0], 0) / this.cand.length, cy = this.cand.reduce((s, p) => s + p[1], 0) / this.cand.length;
      if (Math.hypot(x - cx, y - cy) > this.radius) this.cand = [];
    }
    this.cand.push([x, y, t]);
    if (this.cand.length >= 2 && this.cand[this.cand.length - 1][2] - this.cand[0][2] >= 0.05) {
      const mx = this.cand.reduce((s, p) => s + p[0], 0) / this.cand.length, my = this.cand.reduce((s, p) => s + p[1], 0) / this.cand.length;
      this.n = this.cand.length; this.cand = []; this.anchor = this.out = [mx, my];
    }
    return this.out;
  }
}

// ------------------------------------------------------------------ state
const st = {
  lm: null, video: $("video"), fps: 0, lastFrameT: null, face: false, feat: null, profile: new EyeProfile(), eyeStats: [],
  guardUntil: 0, model: null, offset: [0, 0], fix: new Fix(), gaze: null, cursor: null, vel: [0, 0], hist: [],
  focus: 0, cand: null, candSince: 0, lastFocus: 0, history: [], blocked: null, light: null, mode: "intro",
  cal: null, ripples: [], toastUntil: 0, learned: 0,
};
const tiles = [...document.querySelectorAll(".tile")];
window.__gazefocus = st; // for debugging from the browser console

function show(id) { for (const s of document.querySelectorAll(".screen")) s.classList.toggle("show", s.id === id); }
function toast(text, col = "var(--accent)", secs = 1.6) {
  const el = $("toast"); el.textContent = text; el.style.setProperty("--dot", col); el.style.display = "flex"; st.toastUntil = now() + secs;
}

// ------------------------------------------------------------------ camera + tracker
async function start() {
  $("startBtn").disabled = true; $("err").textContent = "";
  try {
    $("startBtn").textContent = "Starting camera…";
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } }, audio: false });
    st.video.srcObject = stream; await st.video.play();
  } catch (e) {
    $("err").textContent = e && e.name === "NotAllowedError"
      ? "Camera access was blocked. Allow the camera for this site in your browser settings, then reload."
      : `Couldn't start the camera (${e && e.message ? e.message : e}). This page needs the front camera and an https address.`;
    $("startBtn").disabled = false; $("startBtn").textContent = "Start camera"; return;
  }
  try {
    $("startBtn").textContent = "Loading the face tracker…";
    const files = await FilesetResolver.forVisionTasks("./vendor/wasm");
    const opts = (delegate) => ({ baseOptions: { modelAssetPath: "./models/face_landmarker.task", delegate }, runningMode: "VIDEO",
      numFaces: 1, outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true });
    try { st.lm = await FaceLandmarker.createFromOptions(files, opts("GPU")); }
    catch (e) { st.lm = await FaceLandmarker.createFromOptions(files, opts("CPU")); }
  } catch (e) {
    $("err").textContent = `Couldn't load the face tracker (${e && e.message ? e.message : e}).`;
    $("startBtn").disabled = false; $("startBtn").textContent = "Start camera"; return;
  }
  show("live"); st.mode = "live"; layoutFx();
  requestAnimationFrame(frameLoop);
  setTimeout(startCal, 600);
}
$("startBtn").onclick = start;

let lastVideoTime = -1;
function frameLoop() {
  const v = st.video;
  if (st.lm && v.readyState >= 2 && v.currentTime !== lastVideoTime) {
    lastVideoTime = v.currentTime;
    const t = now();
    let res = null;
    try { res = st.lm.detectForVideo(v, performance.now()); } catch (e) { res = null; }
    if (res) onFrame(res, t);
  }
  drawFx();
  requestAnimationFrame(frameLoop);
}

function onFrame(res, t) {
  if (st.lastFrameT != null) { const inst = 1 / Math.max(t - st.lastFrameT, 1e-3); st.fps = st.fps ? 0.9 * st.fps + 0.1 * inst : inst; }
  st.lastFrameT = t;
  const f = features(res, st.video.videoWidth, st.video.videoHeight);
  st.face = !!f; st.feat = f;
  drawCam(f);
  if (!f) return;
  const prof = st.cal && st.cal.live ? st.cal.live : st.profile;
  const state = prof.state(f);
  f.state = state;
  if (state !== "open") { st.guardUntil = t + 0.1; st.fix.interrupt(); }
  const usable = state === "open" && t >= st.guardUntil;
  if (st.cal) { calFrame(f, t, usable); return; }
  if (!st.model || !usable) return;
  // gaze estimate -> smoothing -> cursor / dwell
  const W = innerWidth, H = innerHeight;
  const p = predict(st.model, f.x);
  const x = clamp(p[0] * W + st.offset[0], -0.05 * W, 1.05 * W), y = clamp(p[1] * H + st.offset[1], -0.05 * H, 1.05 * H);
  st.hist.push([t, x, y]); while (st.hist.length && st.hist[0][0] < t - 1.5) st.hist.shift();
  const g = st.fix.update(x, y, t);
  st.gaze = { x: clamp(g[0], 0, W - 1), y: clamp(g[1], 0, H - 1) };
  if (!st.cursor) st.cursor = { ...st.gaze };
  dwell(t);
}

// ------------------------------------------------------------------ dwell / focus / undo
function tileAt(x, y) {
  for (const [i, el] of tiles.entries()) { const r = el.getBoundingClientRect(); if (x >= r.left && x <= r.right && y >= r.top && y <= r.bottom) return i; }
  return null;
}
function setFocus(i) { st.focus = i; tiles.forEach((el, k) => el.classList.toggle("focus", k === i)); }
function dwell(t) {
  const i = tileAt(st.gaze.x, st.gaze.y);
  if (st.blocked) {
    const [bi, left, since] = st.blocked;
    if (i === bi && t - since < 15) { st.blocked[1] = null; st.cand = null; return; }
    if (left == null) st.blocked[1] = t; else if (t - left > 0.5 || t - since >= 15) st.blocked = null;
  }
  if (i == null || i === st.focus) { st.cand = null; return; }
  if (i !== st.cand) { st.cand = i; st.candSince = t; return; }
  if (t - st.candSince < 0.5 || t - st.lastFocus < 0.45) return;
  st.history.push(st.focus); setFocus(i); st.lastFocus = t; st.cand = null;
}
$("undoBtn").onclick = () => {
  if (!st.history.length) { toast("Nothing to undo", "var(--muted)"); return; }
  const prev = st.history.pop(), wrong = st.focus;
  setFocus(prev); st.blocked = [wrong, null, now()]; st.cand = null; st.lastFocus = now();
  toast(`Back to ${tiles[prev].querySelector("h2").textContent}`);
};

// ------------------------------------------------------------------ learning from taps
$("tiles").addEventListener("pointerdown", (ev) => {
  if (!st.model || st.cal) return;
  const t = now(), x = ev.clientX, y = ev.clientY, i = tileAt(x, y);
  if (i != null && i !== st.focus) { st.history.push(st.focus); setFocus(i); st.lastFocus = t; }
  setTimeout(() => learnTap(t, x, y), 200);
});
function learnTap(t, x, y) {
  const fr = st.hist.filter((h) => h[0] >= t - 0.35 && h[0] <= t + 0.15);
  const rip = { t: now(), x, y, label: "", col: "#8494A6" };
  st.ripples.push(rip);
  if (fr.length < 3) { rip.label = "skipped: eyes moving or closed"; rip.col = "#FFBE50"; return; }
  const mx = median(fr.map((h) => h[1])), my = median(fr.map((h) => h[2]));
  const rx = x - mx, ry = y - my, err = Math.hypot(rx, ry), lim = 0.45 * Math.min(innerWidth, innerHeight);
  if (err > lim) { rip.label = "skipped: you weren't looking there"; rip.col = "#FFBE50"; return; }
  const cap = 0.06 * innerWidth;
  st.offset[0] += clamp(0.25 * rx, -cap, cap); st.offset[1] += clamp(0.25 * ry, -cap, cap);
  st.learned++; rip.label = `learned, was ${err.toFixed(0)} px off`; rip.col = "#6EE696";
}

// ------------------------------------------------------------------ calibration (gaze dots + eye shape)
function calPoints() {
  const g = [];
  for (const v of [0.1, 0.5, 0.9]) for (const u of [0.1, 0.5, 0.9]) g.push([u, v]);
  const inner = [[0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7]];
  const pts = [[0.5, 0.5], ...g.filter(([u, v]) => !(u === 0.5 && v === 0.5)), ...inner];
  for (let i = pts.length - 1; i > 1; i--) { const j = 1 + Math.floor(Math.random() * i); [pts[i], pts[j]] = [pts[j], pts[i]]; }
  return pts;
}
function startCal() {
  // slower phones get longer per dot, so every dot still collects ~8 frames
  const collect = clamp(8 / Math.max(st.fps || 30, 1), 0.85, 3.0);
  st.cal = { pts: calPoints(), i: -1, t0: now() + 1.6, samples: [], frames: [], live: new EyeProfile(), stats: [],
             travel: 0.65, collect };
  st.fix.reset(); st.gaze = null; st.cursor = null;
  show("cal"); $("calText").style.display = "block"; $("calDot").style.display = "none";
  $("calCount").textContent = "";
}
function calFrame(f, t, usable) {
  const c = st.cal, TRAVEL = c.travel, COLLECT = c.collect;
  c.stats.push([f.blink, f.open]);
  if (c.stats.length % 15 === 0 && c.stats.length >= 60) c.live = EyeProfile.fit(c.stats);
  if (t < c.t0) return;
  const k = Math.floor((t - c.t0) / (TRAVEL + COLLECT)), ph = (t - c.t0) - k * (TRAVEL + COLLECT);
  if (k !== c.i) { // previous point done
    if (c.i >= 0) keepPoint(c);
    c.i = k; c.frames = [];
    if (k >= c.pts.length) { finishCal(); return; }
    $("calText").style.display = "none";
    const [u, v] = c.pts[k], dot = $("calDot");
    dot.style.display = "block"; dot.style.left = (u * innerWidth) + "px"; dot.style.top = (v * innerHeight) + "px";
    $("calCount").textContent = `${k + 1} / ${c.pts.length}`;
  }
  const ring = $("calRing"), s = ph < TRAVEL ? 1 - 0.55 * (ph / TRAVEL) : 0.45;
  ring.style.transform = `scale(${s})`; ring.style.opacity = ph < TRAVEL ? "1" : "0.5";
  if (ph >= TRAVEL && usable) c.frames.push(f.x.slice());
}
function keepPoint(c) {
  const F = c.frames; if (F.length < 3) return;
  const key = (x) => [x[0], x[1], x[2], x[3]];
  const med = [0, 1, 2, 3].map((i) => median(F.map((x) => key(x)[i])));
  const mad = [0, 1, 2, 3].map((i) => median(F.map((x) => Math.abs(key(x)[i] - med[i]))) * 1.4826 + 1e-6);
  let kept = F.filter((x) => key(x).every((v, i) => Math.abs(v - med[i]) / mad[i] < 3));
  if (kept.length < Math.max(3, F.length / 3)) kept = F;
  const [u, v] = c.pts[c.i];
  for (const x of kept) c.samples.push({ x, u, v, g: c.i });
}
function finishCal() {
  const c = st.cal;
  $("calDot").style.display = "none";
  const groups = new Set(c.samples.map((s) => s.g));
  if (groups.size < 8) {
    st.cal = null; show("results");
    $("resTitle").textContent = "Couldn't calibrate";
    $("resLine").textContent = `Only ${groups.size} of ${c.pts.length} dots had a steady, open-eyed view of your face. Try more light from the front and keep your face in view.`;
    $("resEyes").textContent = ""; $("goBtn").style.display = "none"; return;
  }
  st.model = fitModel(c.samples, [innerWidth, innerHeight]);
  st.profile = EyeProfile.fit(c.stats);
  st.offset = [0, 0]; st.learned = 0; st.cal = null; st.fix.reset();
  const e = st.model.cv, pctW = 100 * e / innerWidth;
  $("resTitle").textContent = e < 0.18 * innerWidth ? "Calibrated" : "Calibrated, but rough";
  $("resLine").textContent = `Typical error ≈ ${e.toFixed(0)} px, about ${pctW.toFixed(0)}% of the screen width. ${groups.size} of ${c.pts.length} dots used.`;
  $("resEyes").textContent = st.profile.fitted
    ? `Eye shape learned: your eyes are open at about ${st.profile.openMed.toFixed(2)}, and frames below ${st.profile.partialO.toFixed(2)} are skipped as half-closed.`
    : "Not enough frames to learn your eye shape, so default eye thresholds are used.";
  $("goBtn").style.display = ""; show("results");
}
$("goBtn").onclick = () => { show("live"); toast("Gaze focus on"); };
$("redoBtn").onclick = () => startCal();
$("calBtn").onclick = () => startCal();

// ------------------------------------------------------------------ drawing
const fx = $("fx"), fctx = fx.getContext("2d");
function layoutFx() {
  const dpr = Math.min(devicePixelRatio || 1, 2.5);
  fx.style.width = innerWidth + "px"; fx.style.height = innerHeight + "px";
  fx.width = Math.round(innerWidth * dpr); fx.height = Math.round(innerHeight * dpr);
  fctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
addEventListener("resize", () => { layoutFx(); });
let lastDraw = now(), lastChips = 0;
function drawFx() {
  const t = now(), dt = Math.min(0.1, t - lastDraw); lastDraw = t;
  fctx.clearRect(0, 0, innerWidth, innerHeight);
  if (st.gaze && st.cursor && !st.cal) { // spring glide, ~0.15 s
    const om = 5.8 / 0.15, e = Math.exp(-om * dt);
    for (const [i, k] of [[0, "x"], [1, "y"]]) {
      const tgt = st.gaze[k], x0 = st.cursor[k] - tgt, v0 = st.vel[i], c = v0 + om * x0;
      st.cursor[k] = tgt + (x0 + c * dt) * e; st.vel[i] = (v0 - om * c * dt) * e;
    }
    const { x, y } = st.cursor;
    fctx.lineWidth = 4; fctx.strokeStyle = "rgba(0,0,0,.55)"; fctx.beginPath(); fctx.arc(x, y, 16, 0, 7); fctx.stroke();
    fctx.lineWidth = 2.5; fctx.strokeStyle = "#50D2FF"; fctx.stroke();
    fctx.fillStyle = "#50D2FF"; fctx.beginPath(); fctx.arc(x, y, 3, 0, 7); fctx.fill();
  }
  for (const p of st.ripples) {
    const age = t - p.t; if (age > 1.8) continue;
    fctx.globalAlpha = 1 - age / 1.8; fctx.strokeStyle = p.col; fctx.lineWidth = 2.5;
    fctx.beginPath(); fctx.arc(p.x, p.y, 10 + age * 40, 0, 7); fctx.stroke();
    if (p.label) {
      fctx.font = "700 13px Figtree, system-ui, sans-serif"; const w = fctx.measureText(p.label).width;
      const lx = clamp(p.x - w / 2, 8, innerWidth - w - 8), ly = p.y - 34;
      fctx.fillStyle = "rgba(10,14,19,.9)"; fctx.beginPath(); fctx.roundRect(lx - 8, ly - 12, w + 16, 24, 12); fctx.fill();
      fctx.fillStyle = p.col; fctx.textBaseline = "middle"; fctx.fillText(p.label, lx, ly);
    }
    fctx.globalAlpha = 1;
  }
  // dwell bars
  tiles.forEach((el, i) => {
    const bar = el.querySelector(".dwell");
    bar.style.width = st.cand === i && st.lastFrameT ? `${clamp((t - st.candSince) / 0.5, 0, 1) * 100}%` : "0";
  });
  if (st.toastUntil && t > st.toastUntil) { $("toast").style.display = "none"; st.toastUntil = 0; }
  if (t - lastChips > 0.25) { lastChips = t; chips(); }
}
function chip(id, text, cls) { const el = $(id); el.textContent = text; el.className = "chip" + (cls ? " " + cls : ""); }
function chips() {
  chip("cFace", st.face ? "face found" : "no face", st.face ? "on" : "bad");
  const s = st.feat && st.feat.state;
  chip("cEyes", !st.feat ? "eyes –" : s === "open" ? (now() < st.guardUntil ? "eyes reopening" : "eyes open") : s === "partial" ? "half-closed: skipped" : "blink: skipped",
       !st.feat ? "" : s === "open" && now() >= st.guardUntil ? "on" : "w");
  chip("cLight", st.light == null ? "light –" : st.light < 45 ? "too dark" : "light ok", st.light == null ? "" : st.light < 45 ? "w" : "on");
  chip("cFps", `${st.fps ? st.fps.toFixed(0) : "–"} fps${st.learned ? ` · ${st.learned} taps learned` : ""}`, "");
}
const camfx = $("camfx"), cctx = camfx.getContext("2d"), probe = document.createElement("canvas"), pctx = probe.getContext("2d", { willReadFrequently: true });
probe.width = 32; probe.height = 24;
let lastProbe = 0;
function drawCam(f) {
  const v = st.video, W = v.videoWidth, H = v.videoHeight;
  if (!W) return;
  if (camfx.width !== W) { camfx.width = W; camfx.height = H; }
  cctx.clearRect(0, 0, W, H);
  if (f) {
    cctx.fillStyle = "#50D2FF";
    for (const e of [EYE_L, EYE_R]) for (const k of [e.a, e.b, e.up, e.lo, e.iris[0]]) { cctx.beginPath(); cctx.arc(f.P[k][0], f.P[k][1], W / 120, 0, 7); cctx.fill(); }
  }
  const t = now();
  if (t - lastProbe > 0.5) { // how lit is your face? (centre of the frame)
    lastProbe = t;
    pctx.drawImage(v, W * 0.3, H * 0.2, W * 0.4, H * 0.5, 0, 0, 32, 24);
    const d = pctx.getImageData(0, 0, 32, 24).data; let s = 0;
    for (let i = 0; i < d.length; i += 4) s += 0.3 * d[i] + 0.59 * d[i + 1] + 0.11 * d[i + 2];
    st.light = s / (d.length / 4);
    if (st.light < 45 && st.model && !st.cal && (!st.darkWarned || t - st.darkWarned > 30)) { st.darkWarned = t; toast("It's dark: light your face from the front", "#FFBE50", 3); }
  }
}
$("cam").onclick = () => $("cam").classList.toggle("hide");
