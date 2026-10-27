// End-to-end check of personal calibration, without a camera.
// For each sign, one BSLDict clip plays the "user": it is re-performed with natural variation
// (speed +-20%, tilt +-6 deg, size +-8%, shift +-3%, landmark jitter) three times through the live
// SignSession in calibration (capture) mode, then once more in normal mode. That clip itself is
// removed from the dictionary examples, so only the calibration can make it match.
//
// Usage: node scripts/test_calibration.mjs <landmark_dir> [all|none] [hard]
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { SignSession } from '../demo/static/demo/sign_session.js';
import { queryVariants, dtw, configure } from '../demo/static/demo/sign_matcher.js';
import { SAMPLES, addExample, mergeIndex } from '../demo/static/demo/calibration.js';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const lmDir = process.argv[2];
const calibrate = (process.argv[3] || 'all') === 'all';
const index = JSON.parse(fs.readFileSync(path.join(root, 'demo/static/demo/data/sign_templates.json'), 'utf8'));
const clips = JSON.parse(fs.readFileSync(path.join(root, 'demo/static/demo/data/sign_clips.json'), 'utf8'));
configure(index.params);

let seed = 7;
const rand = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
const uni = (a, b) => a + (b - a) * rand();
const gauss = s => s * Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());

// "hard" (3rd argument) roughly doubles the variation, for a more conservative estimate.
const hard = process.argv[4] === 'hard';
const FPS = Number(process.env.FPS || 25);   // camera frame rate to simulate (clips are 25 fps)
const V = hard ? { speed: [0.65, 1.35], tilt: 12, size: 0.15, shift: 0.05, jitter: 0.008 }
               : { speed: [0.8, 1.25], tilt: 6, size: 0.08, shift: 0.03, jitter: 0.004 };
function perform(frames) {
  const speed = uni(...V.speed), ang = uni(-V.tilt, V.tilt) * Math.PI / 180, sc = uni(1 - V.size, 1 + V.size);
  const dx = uni(-V.shift, V.shift), dy = uni(-V.shift, V.shift);
  const n = frames.length, m = Math.max(4, Math.round(n / speed));
  const a = frames[0].aspect, cx = 0.5 * a, cy = 0.5;
  const tr = (x, y) => {
    let X = x * a - cx, Y = y - cy;
    [X, Y] = [(X * Math.cos(ang) - Y * Math.sin(ang)) * sc, (X * Math.sin(ang) + Y * Math.cos(ang)) * sc];
    return [(X + cx) / a + dx + gauss(V.jitter), Y + cy + dy + gauss(V.jitter)];
  };
  const out = [];
  for (let i = 0; i < m; i++) {
    const fr = frames[Math.min(n - 1, Math.round(i * (n - 1) / (m - 1)))];
    out.push({
      aspect: a,
      hands: fr.hands.map(h => ({ label: h.label, lm: h.lm.map(p => [...tr(p[0], p[1]), p[2] * sc + gauss(V.jitter)]) })),
      body: fr.body ? Object.fromEntries(Object.entries(fr.body).map(([k, v]) => [k, tr(v[0], v[1])])) : null,
    });
  }
  return out;
}

const empty = aspect => ({ aspect, hands: [], body: null });
function play(session, frames, clock) {
  const aspect = frames[0].aspect;
  const atFps = []; for (let k = 0; k * 25 / FPS < frames.length; k++) atFps.push(frames[Math.floor(k * 25 / FPS)]);
  const pad = Math.ceil(0.32 * FPS);
  for (const fr of [...Array(pad).fill(empty(aspect)), ...atFps, ...Array(pad).fill(empty(aspect))]) {
    clock.t += 1000 / FPS;
    session.push(fr, fr.hands[0] ? fr.hands[0].lm.map(p => ({ x: p[0], y: p[1], z: p[2] })) : undefined);
  }
}

let right = 0, wrong = 0, unsure = 0;
const mistakes = [];
for (const sign of index.signs) {
  const url = clips.words[sign.word].clips[0];
  const file = path.join(lmDir, path.basename(url).replace(/\.mp4$/, '.json'));
  if (!fs.existsSync(file)) continue;
  const d = JSON.parse(fs.readFileSync(file, 'utf8'));
  const user = d.frames.slice(d.core[0], d.core[1] + 1);
  // remove the user's own clip from the dictionary examples
  const own = queryVariants(user);
  const dists = sign.templates.map(t => Math.min(...own.map(v => dtw(v, t))));
  const drop = dists.indexOf(Math.min(...dists));
  const dict = { ...index, signs: index.signs.map(s => s.word === sign.word ? { ...s, templates: s.templates.filter((_, i) => i !== drop) } : s) };

  const clock = { t: 0 };
  const personal = {};
  if (calibrate) {
    const cap = new SignSession({ index: dict, now: () => clock.t, onCapture: (frames, held) => addExample(personal, sign.label, frames, held) });
    for (let k = 0; k < SAMPLES; k++) play(cap, perform(user), clock);
  }
  const got = [];
  const live = new SignSession({ index: mergeIndex(dict, personal), now: () => clock.t,
    onAccept: label => got.push(label), onUnsure: () => got.push(null) });
  play(live, perform(user), clock);
  if (got[0] === sign.label) right++;
  else if (got[0] == null) unsure++;
  else { wrong++; mistakes.push(`${sign.label}->${got[0]}`); }
}
const n = right + wrong + unsure;
const pct = x => `${x}/${n} (${Math.round(100 * x / n)}%)`;
console.log(`camera ${FPS} fps, ${calibrate ? 'with' : 'without'} calibration${hard ? ' (hard variation)' : ''}: right ${pct(right)} | wrong ${pct(wrong)} | not sure ${pct(unsure)}`);
if (mistakes.length) console.log('wrong:', mistakes.join(', '));
