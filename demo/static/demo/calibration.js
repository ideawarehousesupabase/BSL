// Personal calibration: the user signs each word a few times and those recordings become extra
// examples for that user. They are kept in this browser only (localStorage), never uploaded.
import { sequenceVectors, staticVector, dominantLabel } from './sign_matcher.js';

const KEY = 'gestureflow.calibration.v1';
export const SAMPLES = 3;   // recordings per sign

export const keyFor = label => label.toLowerCase().trim().replace(/\s+/g, '-');

export function loadPersonal() {
  try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; }
}

export function savePersonal(personal) {
  try { localStorage.setItem(KEY, JSON.stringify(personal)); return true; } catch { return false; }
}

const round4 = x => Math.round(x * 1e4) / 1e4;

// Adds one recorded sign; returns false if the recording had too few frames with hands.
// held: when the sign was a held hand shape, the frames of the hold. It is then saved as a held-shape
// example (statics) instead of a movement, since the only movement was raising the hand.
export function addExample(personal, label, frames, held) {
  const k = keyFor(label);
  if (held) {
    const withHands = frames.filter(f => f.hands.length);
    const v = staticVector(held, dominantLabel(withHands.length ? withHands : held));
    if (!v) return false;
    if (!personal[k]) personal[k] = { label, templates: [] };
    (personal[k].statics ||= []).push(v.map(round4));
    return true;
  }
  const seq = sequenceVectors(frames);
  if (!seq) return false;
  if (!personal[k]) personal[k] = { label, templates: [] };
  personal[k].templates.push(seq.map(f => f.map(round4)));
  return true;
}

// Recordings saved for one sign (movements + held shapes).
export const exampleCount = p => (p?.templates?.length || 0) + (p?.statics?.length || 0);

export function forget(personal, label) { delete personal[keyFor(label)]; }

// Dictionary examples + this user's examples, in the format sign_matcher.matchSequence and
// matchStatic expect.
export function mergeIndex(index, personal) {
  const signs = index.signs.map(s => ({ ...s }));
  // ignore recordings saved by an older version (different feature layout), which would break matching
  const valid = t => Array.isArray(t) && t.length === index.seqLen && t.every(f => Array.isArray(f) && f.length === index.frameDim);
  for (const [word, raw] of Object.entries(personal)) {
    const p = { ...raw, templates: (raw.templates || []).filter(valid) };  // movement recordings
    if (!p.templates.length) continue;
    const s = signs.find(x => x.word === word);
    if (s) s.templates = [...p.templates, ...s.templates];
    else signs.push({ word, label: p.label, templates: p.templates });
  }
  if (!index.static) return { ...index, signs };
  // held-shape recordings
  const slot = (index.frameDim - 1) / 2;   // where the second hand's slot starts
  const statics = index.static.signs.map(s => ({ ...s }));
  for (const [word, raw] of Object.entries(personal)) {
    const vs = (raw.statics || []).filter(v => Array.isArray(v) && v.length === index.frameDim);
    if (!vs.length) continue;
    const s = statics.find(x => x.word === word);
    if (s) s.templates = [...vs, ...s.templates];
    else statics.push({ word, label: raw.label, templates: vs, twoHanded: 2 * vs.filter(v => v[slot] >= 0.5).length > vs.length });
  }
  return { ...index, signs, static: { ...index.static, signs: statics } };
}
