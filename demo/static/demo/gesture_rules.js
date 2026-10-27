// The original GestureFlow handshape rules, kept as a fallback for the demo gestures.
// Changes from the first version:
//  - "Stop" (still open hand) and "Hello" (waving open hand) are told apart by movement;
//    before, "Stop" could never be reached.
//  - Thumb checks are measured relative to palm size, so they work at any camera distance.
//  - Confidence is computed from how clearly each finger is straight or bent, not a fixed number.

function sub(a, b) { return { x: a.x - b.x, y: a.y - b.y, z: (a.z || 0) - (b.z || 0) }; }
function mag(v) { return Math.hypot(v.x, v.y, v.z || 0); }
function angle(a, b) {
  const m = mag(a) * mag(b);
  if (!m) return Math.PI / 2;
  return Math.acos(Math.max(-1, Math.min(1, (a.x * b.x + a.y * b.y + (a.z || 0) * (b.z || 0)) / m)));
}

const BENT = Math.PI * 0.55;

// Returns how straight a finger is: >0 straight, <0 bent, magnitude = how clear.
function fingerScore(lm, tip, pip, mcp) {
  return (BENT - angle(sub(lm[tip], lm[pip]), sub(lm[pip], lm[mcp]))) / BENT;
}

function thumbScore(lm) {
  const palm = mag(sub(lm[9], lm[0])) || 1e-6;
  const bend = (Math.PI * 0.65 - angle(sub(lm[4], lm[3]), sub(lm[3], lm[2]))) / (Math.PI * 0.65);
  const spread = (mag(sub(lm[4], lm[5])) / palm) - 0.45;
  return Math.min(bend, spread * 2);
}

function thumbDir(lm) {
  const palm = mag(sub(lm[9], lm[0])) || 1e-6;
  const dy = (lm[2].y - lm[4].y) / palm;
  if (dy > 0.35) return 'up';
  if (dy < -0.35) return 'down';
  return 'side';
}

// lm: 21 points {x,y,z}; waving: true if the open hand is moving side to side.
export function classifyHand(lm, waving) {
  const s = {
    thumb: thumbScore(lm),
    index: fingerScore(lm, 8, 6, 5),
    middle: fingerScore(lm, 12, 10, 9),
    ring: fingerScore(lm, 16, 14, 13),
    pinky: fingerScore(lm, 20, 18, 17),
  };
  const f = Object.fromEntries(Object.entries(s).map(([k, v]) => [k, v > 0]));
  const clarity = Object.values(s).reduce((t, v) => t + Math.min(1, Math.abs(v) * 4), 0) / 5;
  const conf = Math.round(50 + 49 * clarity);
  const four = f.index && f.middle && f.ring && f.pinky;

  if (four && f.thumb) return { g: waving ? 'Hello' : 'Stop', c: conf };
  if (four && !f.thumb) return { g: 'Four', c: conf };
  if (!f.thumb && f.index && f.middle && f.ring && !f.pinky) return { g: 'Three', c: conf };
  if (f.index && f.middle && !f.ring && !f.pinky) return { g: 'Peace', c: conf };
  if (f.index && !f.middle && !f.ring && !f.pinky && !f.thumb) return { g: 'Help', c: conf };
  if (f.thumb && !f.index && !f.middle && !f.ring && !f.pinky) {
    const dir = thumbDir(lm);
    if (dir === 'down') return { g: 'No', c: conf };
    if (dir === 'up') return { g: 'Good job', c: conf };
  }
  if (f.index && !f.middle && !f.ring && f.pinky && f.thumb) return { g: 'Love', c: conf };
  return { g: null, c: 0 };
}

export const RULE_GESTURES = ['Hello', 'Stop', 'Four', 'Three', 'Peace', 'Help', 'Good job', 'No', 'Love'];
