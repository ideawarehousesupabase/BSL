// Streams recorded BSLDict landmarks through the live SignSession (as the camera would),
// with each clip's own example removed from the index and small landmark jitter added.
// Checks that every clip produces exactly one decision and reports how often it is right.
//
// Usage: node scripts/test_session.mjs <landmark_dir> [camera_fps, default 25]
// (the BSLDict clips are 25 fps; a lower fps drops frames like a slow webcam does)
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { SignSession } from '../demo/static/demo/sign_session.js';
import { queryVariants, dtw, configure } from '../demo/static/demo/sign_matcher.js';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const lmDir = process.argv[2];
const FPS = Number(process.argv[3] || 25);
// frames a webcam running at FPS would deliver from a 25 fps recording
const atFps = frames => { const out = []; for (let k = 0; k * 25 / FPS < frames.length; k++) out.push(frames[Math.floor(k * 25 / FPS)]); return out; };
const pad = Math.ceil(0.32 * FPS);  // ~0.3 s without hands before and after
const index = JSON.parse(fs.readFileSync(path.join(root, 'demo/static/demo/data/sign_templates.json'), 'utf8'));
const clips = JSON.parse(fs.readFileSync(path.join(root, 'demo/static/demo/data/sign_clips.json'), 'utf8'));
configure(index.params);

let seed = 1;
const rand = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
const gauss = s => s * Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
const jitter = fr => ({
  aspect: fr.aspect,
  hands: fr.hands.map(h => ({ label: h.label, lm: h.lm.map(p => [p[0] + gauss(0.004), p[1] + gauss(0.004), p[2] + gauss(0.004)]) })),
  body: fr.body,
});
const empty = aspect => ({ aspect, hands: [], body: null });

let decisions = 0, correct = 0, multi = 0, none = 0;
const wrong = [];
for (const sign of index.signs) {
  for (const url of clips.words[sign.word].clips) {
    const file = path.join(lmDir, path.basename(url).replace(/\.mp4$/, '.json'));
    if (!fs.existsSync(file)) continue;
    const d = JSON.parse(fs.readFileSync(file, 'utf8'));
    // held out: remove this clip's own example (the template closest to its clean landmarks)
    const own = queryVariants(d.frames.slice(d.core[0], d.core[1] + 1));
    if (!own.length) continue;
    const dists = sign.templates.map(t => Math.min(...own.map(v => dtw(v, t))));
    const drop = dists.indexOf(Math.min(...dists));
    const heldOut = { ...index, signs: index.signs.map(s => s.word === sign.word
      ? { ...s, templates: s.templates.filter((_, i) => i !== drop) } : s) };
    if (!heldOut.signs.find(s => s.word === sign.word).templates.length) continue;

    let t = 0;
    const got = [];
    const session = new SignSession({ index: heldOut, now: () => t,
      onAccept: label => got.push(label), onUnsure: () => got.push(null) });
    const aspect = d.frames[0].aspect;
    const stream = [...Array(pad).fill(empty(aspect)), ...atFps(d.frames).map(jitter), ...Array(pad).fill(empty(aspect))];
    for (const fr of stream) {
      t += 1000 / FPS;
      const raw = fr.hands[0] ? fr.hands[0].lm.map(p => ({ x: p[0], y: p[1], z: p[2] })) : undefined;
      session.push(fr, raw);
    }
    if (!got.length) { none++; continue; }
    if (got.length > 1) multi++;
    decisions++;
    if (got[0] === sign.label) correct++; else wrong.push(`${sign.label}->${got[0] ?? 'unsure'}`);
  }
}
const unsure = wrong.filter(w => w.endsWith('->unsure')).length;
const spoken = decisions - unsure;
const pct = (a, b) => `${a}/${b} (${Math.round(100 * a / Math.max(1, b))}%)`;
console.log(`camera ${FPS} fps | clips: ${decisions + none} | one decision each: ${decisions - multi} | more than one: ${multi} | none: ${none}`);
console.log(`right word spoken: ${pct(correct, decisions)} | wrong word spoken: ${pct(spoken - correct, decisions)} | "not sure": ${pct(unsure, decisions)}`);
console.log(`of the words spoken, correct: ${pct(correct, spoken)}`);
const DEMO = ['Hello', 'Stop', 'Four', 'Three', 'Peace', 'Help', 'Good job', 'No', 'Love'];
const byDemo = wrong.filter(w => DEMO.includes(w.split('->')[1]));
console.log(`wrong words that were demo gestures: ${byDemo.length} (${byDemo.join(', ')})`);
console.log('examples of wrong words:', wrong.filter(w => !w.endsWith('->unsure')).slice(0, 12).join(', '));
