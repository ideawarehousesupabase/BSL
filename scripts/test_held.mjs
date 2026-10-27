// Held hand shapes through the live SignSession (no camera needed).
//
// 1. Held-shape words (sign_templates.json "static"): for each dictionary clip with a still moment,
//    the hand is raised into that still position and held for 1.4 s (as a user would be told to do),
//    with live-like jitter and the clip's own example removed from the index (a different signer).
// 2. Demo hand shapes (BSLDict "two", "v", "four", "five", "three", "stop"): a single frame raised and
//    held, to check that the held-shape words do not take over the client's demo gestures.
//
// Usage: node scripts/test_held.mjs <landmark_dir> [camera_fps, default 25] [google_labels.json]
// google_labels.json (optional): Google Gesture Recognizer label per clip frame,
// {clip name: [[category, score] | null, ...]}, so the test includes Google's demo-gesture votes.
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { SignSession } from '../demo/static/demo/sign_session.js';
import { staticVector, smoothHands, frameCost, configure } from '../demo/static/demo/sign_matcher.js';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const lmDir = process.argv[2];
const FPS = Number(process.argv[3] || 25);
const google = process.argv[4] ? JSON.parse(fs.readFileSync(process.argv[4], 'utf8')) : {};
const index = JSON.parse(fs.readFileSync(path.join(root, 'demo/static/demo/data/sign_templates.json'), 'utf8'));
const clips = JSON.parse(fs.readFileSync(path.join(root, 'demo/static/demo/data/sign_clips.json'), 'utf8'));
configure(index.params);
const DEMO = ['Hello', 'Stop', 'Four', 'Three', 'Peace', 'Help', 'Good job', 'No', 'Love'];

let seed = 1;
const rand = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
const gauss = s => s * Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
const empty = aspect => ({ frame: { aspect, hands: [], body: null } });
// a frame moved down by dy (the raise) with landmark jitter
const shifted = (fr, dy, s) => ({
  aspect: fr.aspect, body: fr.body,
  hands: fr.hands.map(h => ({ label: h.label, lm: h.lm.map(p => [p[0] + gauss(s), p[1] + dy + gauss(s), p[2] + gauss(s)]) })),
});
const gestureOf = (name, i) => {
  const g = google[name]?.[i];
  return g ? { categoryName: g[0], score: g[1] } : undefined;
};

// Same rule as held_window in build_index.py: longest run where the smoothed wrist stays still.
function heldWindow(frames) {
  let best = null, start = null, last = null;
  smoothHands(frames).forEach((fr, i) => {
    let still = false;
    if (fr.hands.length) {
      const lm = fr.hands[0].lm, a = fr.aspect;
      const w = [lm[0][0] * a, lm[0][1]];
      const palm = Math.hypot(w[0] - lm[9][0] * a, w[1] - lm[9][1]) || 1e-6;
      still = last !== null && Math.hypot(w[0] - last[0], w[1] - last[1]) / palm < 0.06;
      last = w;
    } else last = null;
    if (still) {
      if (start === null) start = i;
      if (best === null || i - start > best[1] - best[0]) best = [start, i];
    } else start = null;
  });
  return best && best[1] - best[0] + 1 >= 5 ? best : null;
}

// Streams 25 fps source frames as a camera at FPS would deliver them; returns the decisions.
function run(source, idx) {
  const atFps = [];
  for (let k = 0; k * 25 / FPS < source.length; k++) atFps.push(source[Math.floor(k * 25 / FPS)]);
  let t = 0;
  const got = [];
  const session = new SignSession({ index: idx, now: () => t, onAccept: label => got.push(label), onUnsure: () => got.push(null) });
  for (const { frame, gesture } of atFps) {
    t += 1000 / FPS;
    const raw = frame.hands[0] ? frame.hands[0].lm.map(p => ({ x: p[0], y: p[1], z: p[2] })) : undefined;
    session.push(frame, raw, frame.hands[0] ? gesture : undefined);
  }
  return got;
}

// raise (8 frames from 0.3 below), hold for `holdFrames`, hands down
function raiseAndHold(held, aspect, gestures, jit) {
  const pingpong = i => { const n = held.length, k = i % (2 * n - 2 || 1); return k < n ? k : 2 * n - 2 - k; };
  return [
    ...Array(8).fill(empty(aspect)),
    ...Array.from({ length: 8 }, (_, i) => ({ frame: shifted(held[0], 0.3 * (8 - i) / 8, jit), gesture: gestures[0] })),
    ...Array.from({ length: 35 }, (_, i) => ({ frame: shifted(held[pingpong(i)], 0, jit), gesture: gestures[pingpong(i)] })),
    ...Array(8).fill(empty(aspect)),
  ];
}

const pct = (a, b) => `${a}/${b} (${Math.round(100 * a / Math.max(1, b))}%)`;
const noStatic = { ...index, static: undefined };

// 1. held-shape words
let right = 0, total = 0, unsure = 0, before = 0;
const wrong = [];
for (const sign of index.static?.signs || []) {
  for (const url of clips.words[sign.word].clips) {
    const name = path.basename(url).replace(/\.mp4$/, '');
    const file = path.join(lmDir, name + '.json');
    if (!fs.existsSync(file)) continue;
    const d = JSON.parse(fs.readFileSync(file, 'utf8'));
    const win = heldWindow(d.frames);
    if (!win) continue;
    const held = d.frames.slice(win[0], win[1] + 1);
    // held out: remove this clip's own example (the template closest to its clean held shape)
    const own = [staticVector(held, 'Right'), staticVector(held, 'Left')].filter(Boolean);
    const dists = sign.templates.map(tpl => Math.min(...own.map(v => frameCost(v, tpl))));
    const drop = dists.indexOf(Math.min(...dists));
    if (sign.templates.length < 2) continue;
    const heldOut = { ...index, static: { ...index.static, signs: index.static.signs.map(s => s.word === sign.word
      ? { ...s, templates: s.templates.filter((_, i) => i !== drop) } : s) } };
    const gestures = held.map((_, i) => gestureOf(name, win[0] + i));
    seed = 7 + total;
    const got = run(raiseAndHold(held, d.frames[0].aspect, gestures, 0.004), heldOut);
    seed = 7 + total;   // same jitter for the "before" run
    const old = run(raiseAndHold(held, d.frames[0].aspect, gestures, 0.004), { ...heldOut, static: undefined });
    total++;
    const ok = l => l && l.toLowerCase() === sign.label.toLowerCase();
    if (ok(got[0])) right++;
    else if (got[0] == null) unsure++;
    else wrong.push(`${sign.label}->${got[0]}`);
    if (ok(old[0])) before++;
  }
}
console.log(`camera ${FPS} fps${process.argv[4] ? ' · with Google gesture labels' : ''}`);
console.log(`held-shape words: right ${pct(right, total)} | wrong ${pct(wrong.length, total)} | "not sure" ${pct(unsure, total)}` +
  ` | before held-shape words were added: right ${pct(before, total)}`);
console.log('  wrong:', wrong.join(', ') || 'none');

// 2. demo hand shapes
const changed = [];
let demo = 0, n = 0;
for (const word of ['two', 'v', 'four', 'five', 'three', 'stop']) {
  for (const url of clips.words[word]?.clips || []) {
    const name = path.basename(url).replace(/\.mp4$/, '');
    const file = path.join(lmDir, name + '.json');
    if (!fs.existsSync(file)) continue;
    const d = JSON.parse(fs.readFileSync(file, 'utf8'));
    for (const q of [0.4, 0.5, 0.6]) {
      const i = Math.round(d.core[0] + q * (d.core[1] - d.core[0]));
      if (!d.frames[i].hands.length) continue;
      const one = { ...d.frames[i], hands: d.frames[i].hands.slice(0, 1) };
      // held out: remove the demo-shape example closest to this frame (it may come from this clip)
      const own = [staticVector([one], 'Right'), staticVector([one], 'Left')];
      const demos = index.static?.demo || [];
      let drop = null, best = Infinity;
      demos.forEach((g, gi) => g.templates.forEach((tpl, ti) => {
        const c = Math.min(...own.map(v => frameCost(v, tpl)));
        if (c < best) { best = c; drop = [gi, ti]; }
      }));
      const idx = drop ? { ...index, static: { ...index.static, demo: demos.map((g, gi) => gi !== drop[0] ? g
        : { ...g, templates: g.templates.filter((_, ti) => ti !== drop[1]) }) } } : index;
      seed = 100 + n;
      const now = run(raiseAndHold([one], one.aspect, [gestureOf(name, i)], 0.003), idx)[0];
      seed = 100 + n;
      const old = run(raiseAndHold([one], one.aspect, [gestureOf(name, i)], 0.003), noStatic)[0];
      n++;
      if (DEMO.includes(now)) demo++;
      if (now !== old) changed.push(`${word}: ${old ?? 'unsure'} -> ${now ?? 'unsure'}`);
    }
  }
}
console.log(`demo hand shapes (one hand held): ${pct(demo, n)} came out as a demo gesture | changed by the held-shape words: ${changed.length}`);
if (changed.length) console.log('  ' + changed.join(', '));
