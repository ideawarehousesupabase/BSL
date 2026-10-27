// Sign matching by template comparison (no training, no AI model).
// Mirrors scripts/sign_features.py — keep the two files in step.
//
// Frame: { aspect, hands: [{ label: 'Left'|'Right', lm: [[x,y,z] x21] }], body: {nose, ls, rs} | null }
// Per hand: finger-joint angles + finger spread (handshape), pointing/palm direction, wrist position
// relative to the shoulders.

const FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];
export const SEQ_LEN = 12;
const SMOOTH = 5;   // hand points are averaged over this many neighbouring frames (reduces jitter)
const N_SHAPE = 18;
const N_ORIENT = 5;
const SLOT_DIM = 1 + N_SHAPE + N_ORIENT + 2;

// Matching weights. Defaults match sign_features.py; sign_templates.json carries the tuned values.
const P = { band: 3, shapeWeight: 1.5, orientWeight: 0.5, locWeight: 0.5, missingHandCost: 1.0, nondominantWeight: 0.6,
  dtwScale: 1, embScale: 1, embWeight: 0 };
export function configure(params) { if (params) Object.assign(P, params); }

const pt = (p, aspect) => [p[0] * aspect, p[1]];
const dist2 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

// Angle at b between b->a and b->c, as a fraction of pi.
function angle(a, b, c) {
  const v1 = sub(a, b), v2 = sub(c, b);
  const n = Math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2) * Math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2);
  if (n < 1e-9) return 0;
  const dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
  return Math.acos(Math.max(-1, Math.min(1, dot / n))) / Math.PI;
}

export function dominantLabel(frames) {
  const travel = { Left: 0, Right: 0 }, seen = { Left: 0, Right: 0 }, last = {};
  for (const fr of frames) {
    for (const h of fr.hands) {
      const w = pt(h.lm[0], fr.aspect);
      seen[h.label] += 1;
      if (last[h.label]) travel[h.label] += dist2(w, last[h.label]);
      last[h.label] = w;
    }
  }
  if (!seen.Left && !seen.Right) return 'Right';
  if (seen.Left && seen.Right) return travel.Left > travel.Right ? 'Left' : 'Right';
  return seen.Left ? 'Left' : 'Right';
}

function handSlot(lm, aspect, mirror, origin, scale) {
  const Q = lm.map(p => [mirror * p[0] * aspect, p[1], p[2] * aspect]);
  const out = [1];
  for (const [a, b, c, d] of FINGERS) out.push(angle(Q[0], Q[a], Q[b]), angle(Q[a], Q[b], Q[c]), angle(Q[b], Q[c], Q[d]));
  for (let i = 1; i < 4; i++) {
    const base = Q[FINGERS[i][0]];
    const u = sub(Q[FINGERS[i][1]], base);
    const w = sub(Q[FINGERS[i + 1][1]], Q[FINGERS[i + 1][0]]);
    out.push(angle([base[0] + u[0], base[1] + u[1], base[2] + u[2]], base, [base[0] + w[0], base[1] + w[1], base[2] + w[2]]));
  }
  const d = sub(Q[9], Q[0]);
  const dn = Math.hypot(d[0], d[1]) + 1e-9;
  const e1 = sub(Q[5], Q[0]), e2 = sub(Q[17], Q[0]);
  const nrm = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
  const nn = Math.sqrt(nrm[0] ** 2 + nrm[1] ** 2 + nrm[2] ** 2) + 1e-9;
  out.push(d[0] / dn, d[1] / dn, nrm[0] / nn, nrm[1] / nn, nrm[2] / nn);
  if (origin) out.push((Q[0][0] - mirror * origin[0]) / scale, (Q[0][1] - origin[1]) / scale);
  else out.push(0, 0);
  return out;
}

export function frameVector(fr, dom) {
  const mirror = dom === 'Left' ? -1 : 1;
  let origin = null, scale = 0;
  if (fr.body) {
    const ls = pt(fr.body.ls, fr.aspect), rs = pt(fr.body.rs, fr.aspect);
    scale = dist2(ls, rs);
    if (scale > 1e-6) origin = [(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2];
  }
  const byLabel = {};
  for (const h of fr.hands) byLabel[h.label] = h;
  const vec = [];
  for (const lab of [dom, dom === 'Left' ? 'Right' : 'Left']) {
    const h = byLabel[lab];
    if (h) vec.push(...handSlot(h.lm, fr.aspect, mirror, origin, scale));
    else for (let i = 0; i < SLOT_DIM; i++) vec.push(0);
  }
  vec.push(origin ? 1 : 0);
  return vec;
}

// Average each hand's points over SMOOTH neighbouring frames (same hand label only).
export function smoothHands(frames) {
  const half = Math.floor(SMOOTH / 2);
  return frames.map((fr, i) => ({
    aspect: fr.aspect,
    body: fr.body,
    hands: fr.hands.map(h => {
      const acc = Array.from({ length: 21 }, () => [0, 0, 0]);
      let n = 0;
      for (let j = Math.max(0, i - half); j < Math.min(frames.length, i + half + 1); j++) {
        const m = frames[j].hands.find(g => g.label === h.label);
        if (!m) continue;
        n++;
        for (let k = 0; k < 21; k++) { acc[k][0] += m.lm[k][0]; acc[k][1] += m.lm[k][1]; acc[k][2] += m.lm[k][2]; }
      }
      return { label: h.label, lm: acc.map(c => [c[0] / n, c[1] / n, c[2] / n]) };
    }),
  }));
}

export function sequenceVectors(frames, domOverride) {
  const withHands = smoothHands(frames).filter(f => f.hands.length);
  if (withHands.length < 3) return null;
  const dom = domOverride || dominantLabel(withHands);
  const vecs = withHands.map(f => frameVector(f, dom));
  const n = vecs.length, out = [];
  for (let i = 0; i < SEQ_LEN; i++) out.push(vecs[Math.floor(i * (n - 1) / (SEQ_LEN - 1) + 0.5)]);
  return out;
}

function slotCost(a, b, off, bothBody) {
  const pa = a[off] >= 0.5, pb = b[off] >= 0.5;
  if (!pa && !pb) return 0;
  if (pa !== pb) return P.missingHandCost;
  let s = 0;
  for (let i = 0; i < N_SHAPE; i++) s += Math.abs(a[off + 1 + i] - b[off + 1 + i]);
  s = s / N_SHAPE * 3;
  const o0 = off + 1 + N_SHAPE;
  let o = 0;
  for (let i = 0; i < N_ORIENT; i++) o += (a[o0 + i] - b[o0 + i]) ** 2;
  let cost = P.shapeWeight * s + P.orientWeight * Math.sqrt(o);
  if (bothBody) {
    const j = o0 + N_ORIENT;
    cost += P.locWeight * Math.hypot(a[j] - b[j], a[j + 1] - b[j + 1]);
  }
  return cost;
}

export function frameCost(a, b) {
  const bothBody = a[a.length - 1] >= 0.5 && b[b.length - 1] >= 0.5;
  return slotCost(a, b, 0, bothBody) + P.nondominantWeight * slotCost(a, b, SLOT_DIM, bothBody);
}

export function dtw(a, b) {
  const n = a.length;
  const d = Array.from({ length: n + 1 }, () => new Array(n + 1).fill(Infinity));
  const steps = Array.from({ length: n + 1 }, () => new Array(n + 1).fill(0));
  d[0][0] = 0;
  for (let i = 1; i <= n; i++) {
    for (let j = Math.max(1, i - P.band); j <= Math.min(n, i + P.band); j++) {
      const c = frameCost(a[i - 1], b[j - 1]);
      let best = d[i - 1][j - 1], k = steps[i - 1][j - 1];
      if (d[i - 1][j] < best) { best = d[i - 1][j]; k = steps[i - 1][j]; }
      if (d[i][j - 1] < best) { best = d[i][j - 1]; k = steps[i][j - 1]; }
      d[i][j] = best + c;
      steps[i][j] = k + 1;
    }
  }
  return d[n][n] / Math.max(steps[n][n], 1);
}

// Best matching word for a sequence of frames, with a measured confidence.
// The sequence read with each hand as the dominant one. For two-handed signs either hand may
// count as dominant, so matching keeps the closer reading (mirrors query_variants in Python).
export function queryVariants(frames) {
  return [sequenceVectors(frames, 'Right'), sequenceVectors(frames, 'Left')].filter(Boolean);
}

// Nearest sign by `distance(sign)`, with a measured confidence (closeness to the threshold, and the
// gap to the runner-up).
function rank(signs, threshold, distance) {
  const ranked = signs.map(sign => ({ word: sign.word, label: sign.label, twoHanded: !!sign.twoHanded, demo: !!sign.demo,
    distance: distance(sign) }))
    .sort((x, y) => x.distance - y.distance);
  if (!ranked.length) return null;
  const [top, second] = ranked;
  const closeness = Math.max(0, 1 - top.distance / (1.5 * threshold));
  const margin = second ? Math.min(1, 3 * (second.distance - top.distance) / Math.max(second.distance, 1e-6)) : 1;
  const confidence = Math.round(100 * (0.6 * closeness + 0.4 * margin));
  return { ...top, confidence, accepted: top.distance <= threshold, runnerUp: second ? second.label : null };
}

// Sign embedding (scripts/train_embedding.py): a small network trained on ~1,000 other dictionary
// words turns a sequence into 64 numbers; the same sign by different signers lands close together.
// Mirrors embed() in sign_features.py.
export function embed(seq, model) {
  let h = seq.flat();
  model.layers.forEach((layer, k) => {
    h = layer.w.map((row, i) => { let s = layer.b[i]; for (let j = 0; j < row.length; j++) s += row[j] * h[j]; return s; });
    if (k < model.layers.length - 1) h = h.map(x => Math.max(0, x));
  });
  const n = Math.sqrt(h.reduce((s, x) => s + x * x, 0)) || 1e-12;
  return h.map(x => x / n);
}

// sign_embedding.json stores each network's weights as 8-bit numbers (base64) times a scale.
const unpacked = new WeakMap();
export function unpackEmbedding(emb) {
  if (!unpacked.has(emb)) {
    unpacked.set(emb, emb.models.map(m => ({
      layers: m.layers.map(l => {
        const bytes = Uint8Array.from(atob(l.q), c => c.charCodeAt(0));
        const q = new Int8Array(bytes.buffer);
        const [rows, cols] = l.shape;
        const w = [];
        for (let r = 0; r < rows; r++) w.push(Array.from(q.subarray(r * cols, (r + 1) * cols), v => v * l.scale));
        return { w, b: l.b };
      }),
    })));
  }
  return unpacked.get(emb);
}

const embCache = new WeakMap();   // template -> its embedding by each network (templates never change once loaded)
function templateEmbeddings(tpl, nets) {
  if (!embCache.has(tpl)) embCache.set(tpl, nets.map(n => embed(tpl, n)));
  return embCache.get(tpl);
}
const cosDistance = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return 1 - s; };

export function matchSequence(frames, index) {
  configure(index.params);
  const variants = queryVariants(frames);
  if (!variants.length) return null;
  const nets = P.embWeight > 0 && index.embedding ? unpackEmbedding(index.embedding) : null;
  const qEmb = nets ? nets.map(n => variants.map(v => embed(v, n))) : null;   // [network][reading]
  return rank(index.signs, index.threshold, sign => {
    let best = Infinity;
    for (const tpl of sign.templates) {
      let d = Infinity;
      for (const v of variants) d = Math.min(d, dtw(v, tpl));
      d /= P.dtwScale;
      if (nets) {   // average over the networks of the closest reading (as build_index.py)
        const te = templateEmbeddings(tpl, nets);
        let e = 0;
        te.forEach((t, m) => { e += Math.min(...qEmb[m].map(q => cosDistance(q, t))); });
        d += P.embWeight * (e / nets.length) / P.embScale;
      }
      best = Math.min(best, d);
    }
    return best;
  });
}

// A held hand shape (no movement): the average frame vector over the smoothed frames with hands.
export function staticVector(frames, dom) {
  const withHands = smoothHands(frames).filter(f => f.hands.length);
  if (!withHands.length) return null;
  const vecs = withHands.map(f => frameVector(f, dom));
  return vecs[0].map((_, i) => { let s = 0; for (const v of vecs) s += v[i]; return s / vecs.length; });
}

// Held-shape words (index.static, built from the still part of the dictionary clips).
// Read with either hand as dominant, like queryVariants. The demo hand shapes (index.static.demo)
// take part as competitors: when one of them is nearest, the result has demo: true (not a word).
export function matchStatic(frames, index) {
  configure(index.params);
  const st = index.static;
  if (!st || !st.signs.length) return null;
  const variants = [staticVector(frames, 'Right'), staticVector(frames, 'Left')].filter(Boolean);
  if (!variants.length) return null;
  const candidates = [...st.signs, ...(st.demo || []).map(d => ({ ...d, demo: true }))];
  const m = rank(candidates, st.threshold, sign => {
    let best = Infinity;
    for (const tpl of sign.templates) for (const v of variants) best = Math.min(best, frameCost(v, tpl));
    return best;
  });
  return { ...m, liveTwoHanded: variants[0][SLOT_DIM] >= 0.5 };
}
