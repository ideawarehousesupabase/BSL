// Splits the live landmark stream into individual signs and decides what each one was.
// Kept free of DOM code so it can be tested with recorded landmarks
// (see scripts/test_session.mjs and scripts/test_calibration.mjs).
//
// Two kinds of input are handled:
//  - moving signs (the BSL dictionary vocabulary): a sign ends when the hand pauses briefly or leaves
//    the camera, and is matched against the dictionary/calibration examples;
//  - held hand shapes (the demo gestures Peace, Four, Stop...): a short raise, then the hand held
//    still for about a second; decided by the hand-shape rules, never by the moving dictionary signs
//    (that mix-up produced "sad sad sad" for every held hand).
//
// Everything is measured in time, not frames, so a slow webcam (e.g. 7 fps) behaves like a fast one,
// and each finished sign is rebuilt at 25 fps (the rate of the dictionary clips) before matching.
import { matchSequence, matchStatic } from './sign_matcher.js';
import { classifyHand } from './gesture_rules.js';

const REF_MS = 40;          // 25 fps: the dictionary clips' frame interval
const MIN_SIGN_MS = 200;    // shortest sign
const MAX_SIGN_MS = 3600;   // longest sign kept
const ABSENT_END_MS = 150;  // hands gone this long = sign finished
const STILL = 0.04;         // wrist movement per 40 ms, in palm widths
const MOVE_REARM = 0.12;    // movement (per 40 ms) that starts the next sign after one was accepted
const EMA_TAU = 57.7;       // wrist smoothing time constant (half-way per 40 ms frame)

// Google MediaPipe Gesture Recognizer labels -> the client's demo gesture names.
// Open_Palm is "Stop" when held still and "Hello" when waving. Three and Four are not in Google's
// model, so they still come from gesture_rules.js.
const GOOGLE_GESTURES = { Victory: 'Peace', Thumb_Up: 'Good job', Thumb_Down: 'No', Pointing_Up: 'Help', ILoveYou: 'Love' };
const GOOGLE_MIN_SCORE = 0.6;

// Linear blend of two frames' hand points (same hand label) and body points.
function lerpFrame(A, B, u) {
  const mix = (p, q) => p.map((v, i) => v + (q[i] - v) * u);
  const near = u < 0.5 ? A : B, far = near === A ? B : A;
  const hands = near.hands.map(h => {
    const o = far.hands.find(x => x.label === h.label);
    if (!o) return h;
    const [a, b] = near === A ? [h, o] : [o, h];
    return { label: h.label, lm: a.lm.map((p, k) => mix(p, b.lm[k])) };
  });
  let body = near.body;
  if (A.body && B.body) body = { nose: mix(A.body.nose, B.body.nose), ls: mix(A.body.ls, B.body.ls), rs: mix(A.body.rs, B.body.rs) };
  return { aspect: A.aspect, hands, body };
}

// Rebuilds a sign recorded at any frame rate as 25 fps frames (filling in between real frames).
export function resampleTo25fps(entries) {
  if (entries.length < 2) return entries.map(e => e.frame);
  const out = [];
  const end = entries[entries.length - 1].t;
  let j = 0;
  for (let t = entries[0].t; t <= end + 1e-6; t += REF_MS) {
    while (j < entries.length - 2 && entries[j + 1].t < t) j++;
    const a = entries[j], b = entries[j + 1];
    const u = b.t > a.t ? Math.min(1, Math.max(0, (t - a.t) / (b.t - a.t))) : 0;
    out.push(u === 0 ? a.frame : u === 1 ? b.frame : lerpFrame(a.frame, b.frame, u));
  }
  return out;
}

export class SignSession {
  // options: { index, holdFrames, cooldownMs, confThresh, shapeMoveMs, shapeHoldMs,
  //            onAccept(label, conf, how), onUnsure(text, conf), onCapture(frames, held), onDebug(info), now() }
  // While `onCapture` is set (calibration), each finished sign is handed over instead of recognised
  // (held: the frames of the hold when it was a held hand shape, else null).
  constructor(options) {
    this.o = {
      holdFrames: 10,     // pause that ends a moving sign, in 25 fps frames (10 = 0.4 s)
      cooldownMs: 600,
      confThresh: 40,     // in testing (45 words, other signers) 78% of signs are spoken and 77% of those are right
      shapeMoveMs: 600,   // a held hand shape: at most this much movement first (the raise)...
      shapeHoldMs: 1000,  // ...then held still this long (or shapeHoldEndMs, if the hands then drop)
      shapeHoldEndMs: 400,
      // accept the best match up to 1.25 x the distance threshold (at 7 fps this lifted calibrated
      // signs from 86% to 97% right with no extra wrong words; see scripts/test_calibration.mjs)
      distSlack: 1.25,
      now: () => performance.now(),
      ...options,
    };
    this.lastLabel = null;      // last accepted sign
    this.handsLeft = true;      // hands left the camera since the last accepted sign
    this.reset();
  }

  set(key, value) { this.o[key] = value; }

  // Forget the last accepted sign (used by "Reset all"), so it may be signed again straight away.
  clearHistory() { this.lastLabel = null; this.handsLeft = true; }

  reset() {
    this.segment = []; this.decided = false;
    this.wristHistory = []; this.lastWrist = null; this.recentWrist = [];
    this.lastHandT = null; this.lastMovingT = null; this.lastUnsteadyT = null;
    this.cooldownUntil = 0;
  }

  holdMs() { return this.o.holdFrames * REF_MS; }

  longEnough() {
    const s = this.segment;
    return s.length >= 3 && s[s.length - 1].t - s[0].t >= MIN_SIGN_MS;
  }

  // frame: { aspect, hands: [{label, lm:[[x,y,z]...]}], body }; rawHand: first hand as [{x,y,z}...] for the rules;
  // gesture (optional): Google Gesture Recognizer result for that hand, { categoryName, score }
  push(frame, rawHand, gesture) {
    const t = this.o.now();
    if (!frame.hands.length) {
      this.lastWrist = null;
      this.recentWrist = [];
      if (this.lastHandT != null && t - this.lastHandT >= ABSENT_END_MS) {
        this.handsLeft = true;
        if (!this.decided && this.longEnough()) this.decide(t, true);
        const cooldown = this.cooldownUntil;
        this.reset();
        this.cooldownUntil = cooldown;
      }
      return;
    }
    const dt = this.lastHandT == null ? REF_MS : Math.max(1, t - this.lastHandT);
    this.lastHandT = t;
    const { fast, steady } = this.wristMotion(frame, t, dt);
    if (this.decided) {
      // wait for the hand to move away from the accepted sign before starting the next one
      if (fast > MOVE_REARM && t > this.cooldownUntil) {
        const cooldown = this.cooldownUntil;
        this.reset();
        this.lastHandT = t;
        this.cooldownUntil = cooldown;
      } else return;
    }
    this.segment.push({ frame, raw: rawHand, gesture, motion: fast, steady, t, dt });
    while (this.segment.length > 2 && t - this.segment[0].t > MAX_SIGN_MS) this.segment.shift();
    if (fast >= STILL || this.lastMovingT == null) this.lastMovingT = t;
    if (steady >= STILL || this.lastUnsteadyT == null) this.lastUnsteadyT = t;
    if (!this.longEnough()) return;

    const mightBeShape = this.movingMs() <= this.o.shapeMoveMs;
    if (mightBeShape && this.steadyMs() >= this.o.shapeHoldMs) this.decide(t);           // held hand shape
    else if (this.stillMs() >= this.holdMs() && !mightBeShape) this.decide(t);           // end of a moving sign
  }

  // fast: movement of the lightly smoothed wrist (to spot the end of a moving sign);
  // steady: movement over ~120 ms, so landmark jitter on a held hand averages out.
  // Both are per 40 ms, so they mean the same at any frame rate.
  wristMotion(frame, t, dt) {
    const lm = frame.hands[0].lm;
    const raw = [lm[0][0] * frame.aspect, lm[0][1]];
    const palm = Math.hypot((lm[9][0] - lm[0][0]) * frame.aspect, lm[9][1] - lm[0][1]) || 1e-6;
    const prev = this.lastWrist;
    const k = 1 - Math.exp(-dt / EMA_TAU);
    const w = prev ? [prev[0] + k * (raw[0] - prev[0]), prev[1] + k * (raw[1] - prev[1])] : raw;
    this.lastWrist = w;
    this.recentWrist.push([w, t]);
    while (this.recentWrist.length > 2 && t - this.recentWrist[1][1] >= 120) this.recentWrist.shift();
    const [old, oldT] = this.recentWrist[0];
    const fast = prev ? Math.hypot(w[0] - prev[0], w[1] - prev[1]) / palm * (REF_MS / dt) : 0;
    const steady = t > oldT ? Math.hypot(w[0] - old[0], w[1] - old[1]) / palm * (REF_MS / (t - oldT)) : 0;
    this.wristHistory.push([w[0], palm, t]);
    while (this.wristHistory.length && t - this.wristHistory[0][2] > 1200) this.wristHistory.shift();
    return { fast, steady };
  }

  stillMs() { return this.lastMovingT == null ? 0 : this.o.now() - this.lastMovingT; }
  steadyMs() { return this.lastUnsteadyT == null ? 0 : this.o.now() - this.lastUnsteadyT; }

  // How long the hand was held still at the end of the segment (until its last frame).
  heldAtEndMs() {
    const last = this.segment[this.segment.length - 1];
    return this.lastUnsteadyT == null || !last ? 0 : last.t - this.lastUnsteadyT;
  }

  // Time the hand was clearly moving in this segment (jitter-proof measure).
  movingMs() {
    let ms = 0;
    for (let i = 1; i < this.segment.length; i++) {
      if (this.segment[i].steady > STILL * 1.5) ms += this.segment[i].t - this.segment[i - 1].t;
    }
    return ms;
  }

  // Passes an accepted sign on, but not the same sign twice in a row unless the hands left the
  // camera in between (otherwise small hand adjustments repeated it).
  emit(label, conf, how) {
    if (label === this.lastLabel && !this.handsLeft) return;
    this.lastLabel = label;
    this.handsLeft = false;
    this.o.onAccept?.(label, conf, how);
  }

  // Waving = the wrist swings left/right, reversing at least twice, each swing at least a third of a palm.
  isWaving() {
    if (this.wristHistory.length < 3) return false;
    const palms = this.wristHistory.map(h => h[1]).sort((a, b) => a - b);
    const palm = palms[palms.length >> 1];      // steady palm size (median), so size jitter is not motion
    const xs = this.wristHistory.map(h => h[0] / palm);
    let turns = 0, dir = 0, extreme = xs[0];
    for (const x of xs.slice(1)) {
      const d = x - extreme;
      if (dir === 0) { if (Math.abs(d) >= 0.33) { dir = Math.sign(d); extreme = x; } continue; }
      if (Math.sign(d) === dir) extreme = x;                       // still moving the same way
      else if (Math.abs(d) >= 0.33) { turns++; dir = -dir; extreme = x; }   // a real reversal
    }
    return turns >= 2;
  }

  // Demo gesture for one frame: Google's trained recogniser when it is confident, else our rules.
  frameGesture(s, waving) {
    const g = s.gesture;
    if (g && g.score >= GOOGLE_MIN_SCORE) {
      const c = Math.round(100 * g.score);
      if (g.categoryName === 'Open_Palm') return { g: waving ? 'Hello' : 'Stop', c, google: true };
      if (GOOGLE_GESTURES[g.categoryName]) return { g: GOOGLE_GESTURES[g.categoryName], c, google: true };
    }
    return s.raw ? classifyHand(s.raw, waving) : { g: null, c: 0 };
  }

  // Majority vote of the demo gestures over the last still moment (holdMs, at least 3 frames).
  // google: the winning gesture came from Google's recogniser in at least half of those frames.
  ruleVote() {
    const waving = this.isWaving();
    const end = this.segment[this.segment.length - 1]?.t ?? 0;
    let tail = this.segment.filter(s => end - s.t < this.holdMs());
    if (tail.length < 3) tail = this.segment.slice(-3);
    tail = tail.filter(s => s.raw || s.gesture).map(s => this.frameGesture(s, waving));
    const counts = {};
    tail.forEach(r => { if (r.g) counts[r.g] = (counts[r.g] || 0) + 1; });
    const [g, n] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0] || [null, 0];
    const conf = g ? Math.round(tail.filter(r => r.g === g).reduce((t, r) => t + r.c, 0) / n) : 0;
    const google = !!g && tail.filter(r => r.g === g && r.google).length >= Math.ceil(tail.length / 2);
    return { g, n, total: tail.length, conf, counts, waving, google };
  }

  // The frames of the final still moment (since the hand last moved), rebuilt at 25 fps.
  heldFrames() {
    const seg = this.segment;
    let part = seg.filter(s => s.t >= this.lastUnsteadyT);
    if (part.length < 3) part = seg.slice(-3);
    return resampleTo25fps(part);
  }

  // The moving part of the segment (like the dictionary clips, cut from first to last real movement),
  // rebuilt at 25 fps.
  signFrames(heldShape) {
    const seg = this.segment;
    let lastMove = heldShape ? seg.findLastIndex(s => s.steady > STILL) : seg.findLastIndex(s => s.motion >= STILL);
    if (lastMove < 0) lastMove = seg.length - 1;
    const endT = seg[lastMove].t + 2 * REF_MS;
    let first = seg.findIndex(s => s.motion > STILL * 1.5);
    const startT = first < 0 ? seg[0].t : seg[first].t - 2 * REF_MS;
    let part = seg.filter(s => s.t >= startT && s.t <= endT);
    if (part.length < 3 || part[part.length - 1].t - part[0].t < MIN_SIGN_MS) part = seg;
    return resampleTo25fps(part);
  }

  decide(t0, handsDropped = false) {
    const { index, confThresh } = this.o;
    this.decided = true;
    this.cooldownUntil = this.o.now() + this.o.cooldownMs;
    // held hand shape: short raise, then still for shapeHoldMs (or shapeHoldEndMs before the hands dropped)
    const heldShape = this.movingMs() <= this.o.shapeMoveMs &&
      (handsDropped ? this.heldAtEndMs() >= this.o.shapeHoldEndMs : this.steadyMs() >= this.o.shapeHoldMs);
    const vote = this.ruleVote();

    if (heldShape && !this.o.onCapture) {
      // Two candidates: the demo gestures (Google's recogniser / our rules) and the dictionary words
      // that can be recognised from a held hand shape (index.static).
      const st = index?.static ? matchStatic(this.heldFrames(), index) : null;
      const stOk = st && !st.demo && st.distance <= this.o.distSlack * index.static.threshold && st.confidence >= confThresh;
      const voteOk = vote.g && vote.n >= Math.ceil(vote.total / 2);
      const last = this.segment[this.segment.length - 1]?.gesture;
      this.o.onDebug?.({ path: 'held shape', movingMs: Math.round(this.movingMs()), vote: vote.counts,
        voteTotal: vote.total, voteFrom: vote.google ? 'google' : 'rules',
        google: last ? `${last.categoryName} ${Math.round(100 * last.score)}%` : 'none',
        heldWord: st && { label: st.demo ? `demo shape ${st.label}` : st.label, confidence: st.confidence, accepted: stOk, next: st.runnerUp,
          distance: Math.round(100 * st.distance / index.static.threshold) / 100, twoHanded: st.twoHanded && st.liveTwoHanded } });
      const how = `held hand shape · ${Math.round(this.o.now() - t0)} ms`;
      if (stOk && st.twoHanded && st.liveTwoHanded) {
        this.emit(st.label, st.confidence, how);       // both hands held: a two-handed sign (the demo gestures use one hand)
      } else if (voteOk && vote.google) {
        this.emit(vote.g, vote.conf, how);             // Google's recogniser is sure: a demo gesture
      } else if (stOk) {
        this.emit(st.label, st.confidence, how);
      } else if (voteOk) {
        this.emit(vote.g, vote.conf, how);             // Three / Four, or a demo gesture Google was unsure of, from our rules
      } else {
        this.o.onUnsure?.(st ? `Not sure — closest held sign: ${st.label} (${st.confidence}%)`
          : 'Not sure — hold the sign still at its place, or make the sign with its movement', st ? st.confidence : 0);
      }
      return;
    }

    const frames = this.signFrames(heldShape);
    if (this.o.onCapture) { this.o.onCapture(frames, heldShape ? this.heldFrames() : null); return; }

    const match = index && index.signs.length ? matchSequence(frames, index) : null;
    // during a moving sign only Google's recogniser may call a demo gesture (our rules read the open hand
    // in e.g. "father" or "cake" as Stop)
    const ruleOk = vote.g && vote.google && vote.n >= Math.ceil(vote.total * 0.8) && vote.conf >= confThresh;
    const latency = Math.round(this.o.now() - t0);
    this.o.onDebug?.({ path: 'moving sign', movingMs: Math.round(this.movingMs()), frames: this.segment.length,
      rule: vote.counts, dictionary: match && { label: match.label, confidence: match.confidence, accepted: match.accepted,
        next: match.runnerUp } });
    const close = match && match.distance <= this.o.distSlack * index.threshold;
    if (close && match.confidence >= confThresh) {
      this.emit(match.label, match.confidence, `sign match${match.runnerUp ? ` (next: ${match.runnerUp})` : ''} · ${latency} ms`);
    } else if (ruleOk && (!match || match.distance > 1.25 * index.threshold)) {
      // demo gestures only when the movement is clearly unlike every dictionary sign
      // (otherwise e.g. an open moving hand in "father" or "food" was read as "Hello")
      this.emit(vote.g, vote.conf, `demo gesture · ${latency} ms`);
    } else {
      this.o.onUnsure?.(match ? `Not sure — closest: ${match.label} (${match.confidence}%)` : '— not recognised —', match ? match.confidence : 0);
    }
  }
}
