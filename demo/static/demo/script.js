// Hand -> Speech: live BSL sign recognition with MediaPipe hand/pose landmarks.
// Signs are recognised by comparing the movement with recorded BSLDict examples
// (sign_matcher.js, via sign_session.js); the original demo gestures still work through gesture_rules.js.
import { FilesetResolver, GestureRecognizer, PoseLandmarker }
  from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1/vision_bundle.mjs';
import { SignSession } from './sign_session.js';
import { RULE_GESTURES } from './gesture_rules.js';
import { SAMPLES, keyFor, loadPersonal, savePersonal, addExample, exampleCount, forget, mergeIndex } from './calibration.js';

const WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1/wasm';
const MODELS = '/static/demo/models/';
const TEMPLATES = '/static/demo/data/sign_templates.json';

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const startBtn = document.getElementById('startCamera');
const stopBtn = document.getElementById('stopCamera');
const resetLastBtn = document.getElementById('resetLast');
const resetAllBtn = document.getElementById('resetAll');
const result = document.getElementById('result');
const confBar = document.getElementById('confidence-fill');
const sentenceBox = document.getElementById('sentenceBox');
const statusLine = document.getElementById('engineStatus');
const knownList = document.getElementById('knownSigns');
const calibSign = document.getElementById('calibSign');
const calibRecordBtn = document.getElementById('calibRecord');
const calibClearBtn = document.getElementById('calibClear');
const calibStatus = document.getElementById('calibStatus');
const calibList = document.getElementById('calibList');
const calibDemo = document.getElementById('calibDemo');
const calibHint = document.getElementById('calibHint');
let clipIndex = { words: {} };                    // dictionary clips, to show how each sign is made

let hands = null, pose = null, stream = null, running = false, starting = false;
let fileset = null, delegateUsed = '';
let loopId = 0;          // only the newest camera loop may run (two loops sent duplicate timestamps)
let lastTs = 0;          // MediaPipe needs strictly increasing timestamps
let rebuilding = false;
let index = { signs: [], threshold: 1 };       // dictionary examples
let personal = loadPersonal();                   // this user's calibration examples
let lastPose = null, frameNo = 0;
let words = [];

// Splits the camera stream into signs; tuned by the Advanced tuning sliders.
const session = new SignSession({
  holdFrames: 10,     // frames a hand must stay still to finish a sign
  cooldownMs: 600,    // pause after a sign is accepted
  confThresh: 40,     // minimum match score to accept
  onAccept: (label, conf, how) => accept(label, conf, how),
  onUnsure: (text, conf) => { result.textContent = text; setBar(conf); },
});
document.getElementById('stability')?.addEventListener('input', e => session.set('holdFrames', +e.target.value));
document.getElementById('cooldown')?.addEventListener('input', e => session.set('cooldownMs', +e.target.value));
document.getElementById('confth')?.addEventListener('input', e => session.set('confThresh', +e.target.value));
// Add ?debug=1 to the address to see how each sign was decided.
if (new URLSearchParams(location.search).has('debug')) {
  const box = document.getElementById('debugInfo');
  if (box) {
    box.hidden = false;
    session.set('onDebug', info => { box.textContent = JSON.stringify(info, null, 1); });
  }
}
let fpsCount = 0, fpsStart = performance.now(), fps = 0;

// ---------- speech
let voice = null;
function pickVoice() {
  const voices = speechSynthesis.getVoices();
  voice = voices.find(v => v.lang === 'en-GB') || voices.find(v => v.lang.startsWith('en')) || null;
}
pickVoice();
speechSynthesis.onvoiceschanged = pickVoice;
function speak(text) {
  try {
    const u = new SpeechSynthesisUtterance(text);
    u.lang = 'en-GB';
    if (voice) u.voice = voice;
    if (text.endsWith('?')) u.pitch = 1.1;
    speechSynthesis.cancel();
    speechSynthesis.speak(u);
  } catch (e) { /* speech not available */ }
}

// ---------- setup
// The sign list is loaded straight away so users can browse it before starting the camera.
const indexReady = fetch(TEMPLATES)
  .then(r => (r.ok ? r.json() : { signs: [], threshold: 1 }))
  .catch(() => ({ signs: [], threshold: 1 }))
  .then(data => { index = data; applyIndex(); showKnownSigns(); fillCalibration(); return data; });

// recordings of signs no longer offered (e.g. the removed "Stop" demo gesture) are ignored
function applyIndex() {
  const known = new Set(allLabels().map(keyFor));
  session.set('index', mergeIndex(index, Object.fromEntries(Object.entries(personal).filter(([k]) => known.has(k)))));
}

// Words that can be signed by holding the hand shape still (no movement needed).
const heldWords = () => new Set((index.static?.signs || []).map(s => s.word));
// All BSL dictionary signs: moving signs plus the held-shape words that are not also moving signs.
function bslSigns() {
  const moving = new Set(index.signs.map(s => s.word));
  return [...index.signs, ...(index.static?.signs || []).filter(s => !moving.has(s.word))];
}

async function loadEngine() {
  status('Loading sign recognition…');
  [fileset] = await Promise.all([FilesetResolver.forVisionTasks(WASM), indexReady]);
  await createModels();
  status(`Ready (${delegateUsed}) — ${bslSigns().length} BSL signs from the dictionary + ${RULE_GESTURES.length} demo gestures` +
    (calibratedCount() ? ` · ${calibratedCount()} calibrated by you.` : '.'));
}

async function createModels() {
  const make = async (Task, file, extra, delegate) => Task.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODELS + file, delegate }, runningMode: 'VIDEO', ...extra,
  });
  const used = [];
  const create = async (Task, file, extra) => {
    try { const t = await make(Task, file, extra, 'GPU'); used.push('GPU'); return t; }
    catch { used.push('CPU'); return make(Task, file, extra, 'CPU'); }
  };
  [hands, pose] = await Promise.all([
    // Google's Gesture Recognizer: the same 21 hand points as before, plus trained recognition of
    // Open_Palm, Victory, Thumb_Up/Down, Pointing_Up and ILoveYou (used for the demo gestures)
    create(GestureRecognizer, 'gesture_recognizer.task', { numHands: 2, minHandDetectionConfidence: 0.5 }),
    create(PoseLandmarker, 'pose_landmarker_lite.task', { numPoses: 1 }),
  ]);
  delegateUsed = used.includes('CPU') ? 'CPU' : 'GPU';
}

// If MediaPipe reports a broken graph, close the models and load them again; the camera keeps running.
async function rebuildModels() {
  if (rebuilding) return;
  rebuilding = true;
  try {
    try { hands?.close(); pose?.close(); } catch (_) { /* already broken */ }
    await createModels();
    lastTs = 0;
    session.reset();
    lastError = '';
    status('Recovered — sign recognition restarted.');
  } catch (e) {
    console.error('rebuild failed', e);
  } finally {
    rebuilding = false;
  }
}

// Hides the sign chips that do not contain the search text.
function filterKnownSigns() {
  const q = (document.getElementById('signSearch')?.value || '').trim().toLowerCase();
  let shown = 0;
  for (const c of knownList.querySelectorAll('.chip')) {
    c.hidden = !c.textContent.toLowerCase().includes(q);
    if (!c.hidden) shown++;
  }
  const empty = knownList.querySelector('.sign-empty');
  if (empty) empty.hidden = shown > 0;
}
document.getElementById('signSearch')?.addEventListener('input', filterKnownSigns);

// How to make each demo gesture (they are not in BSLDict, so there is no dictionary clip).
const DEMO_HOWTO = {
  'Hello': 'open hand, palm to the camera, fingers and thumb spread, wave side to side.',
  'Four': 'four fingers straight, thumb folded across the palm, palm to the camera, hold still.',
  'Three': 'index, middle and ring fingers straight, thumb and little finger folded, hold still.',
  'Peace': 'index and middle fingers up in a V, other fingers and thumb folded, hold still.',
  'Help': 'index finger pointing up, other fingers and thumb folded, hold still.',
  'Good job': 'fist with the thumb pointing up, hold still.',
  'No': 'fist with the thumb pointing down, hold still.',
  'Love': 'thumb, index and little finger straight, middle and ring fingers folded, palm to the camera, hold still.',
};
const DEMO_EMOJI = { 'Hello': '👋', 'Four': '4️⃣', 'Three': '3️⃣', 'Peace': '✌️', 'Help': '☝️', 'Good job': '👍', 'No': '👎', 'Love': '🤟' };
const withEmoji = g => `${DEMO_EMOJI[g] || ''} ${g}`.trim();

// Sentences made from demo gestures (see unified_patterns.js); the video shows the sentence in BSL,
// played from the dictionary clips of these words.
const DEMO_SENTENCES = [
  { sentence: 'Do you understand me?', gestures: ['Three', 'Peace'], bsl: ['you', 'understand', 'me'] },
  { sentence: 'Thank you for helping me.', gestures: ['Help', 'Good job'], bsl: ['help', 'me', 'thank-you'] },
];

// Plays one clip on repeat, or several clips one after another, starting again after the last.
function playClips(video, urls) {
  if (!video || !urls.length) return;
  let i = 0;
  video.onerror = null;
  video.loop = urls.length === 1;
  video.onended = urls.length === 1 ? null : () => { i = (i + 1) % urls.length; video.src = urls[i]; video.play().catch(() => {}); };
  video.hidden = false;
  video.src = urls[0];
  video.play().catch(() => {});
}

// Lists the recognisable signs; clicking a dictionary sign plays its clip so the user can copy it.
async function showKnownSigns() {
  if (!knownList) return;
  const clips = await fetch('/static/demo/data/sign_clips.json').then(r => r.json()).catch(() => ({ words: {} }));
  clipIndex = clips;
  if (calibSign?.value) showCalibDemo(calibSign.value);
  const demo = document.getElementById('signDemo');
  knownList.innerHTML = '';
  const held = heldWords();
  for (const s of bslSigns().sort((a, b) => a.label.localeCompare(b.label))) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = held.has(s.word) ? 'chip chip-held' : 'chip';
    b.textContent = s.label;
    if (held.has(s.word)) b.title = 'Can also be signed by holding its hand shape still';
    const url = clips.words[s.word]?.clips[0];
    if (url && demo) b.addEventListener('click', () => playClips(demo, [url]));
    knownList.appendChild(b);
  }
  const hint = document.getElementById('signHint');
  for (const s of knownList.querySelectorAll('.chip')) s.addEventListener('click', () => { if (hint) hint.hidden = true; });
  for (const g of RULE_GESTURES) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'chip chip-demo';
    b.textContent = withEmoji(g);
    b.title = 'Demo gesture (not a BSL sign)';
    b.addEventListener('click', () => {
      if (hint) { hint.hidden = false; hint.textContent = `${withEmoji(g)}: ${DEMO_HOWTO[g]}`; }
      if (!demo) return;
      // optional own recording: media/signs/demo/<gesture>.mp4 (e.g. good-job.mp4)
      playClips(demo, [`/media/signs/demo/${g.toLowerCase().replace(/ /g, '-')}.mp4`]);
      demo.onerror = () => { demo.hidden = true; };
    });
    knownList.appendChild(b);
  }
  for (const s of DEMO_SENTENCES) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'chip chip-demo';
    b.textContent = s.sentence;
    b.title = `Sign ${s.gestures.join(' then ')}`;
    const urls = s.bsl.map(w => clips.words[w]?.clips[0]).filter(Boolean);
    b.addEventListener('click', () => {
      if (hint) {
        hint.hidden = false;
        hint.textContent = `"${s.sentence}" — to make it here, sign ${s.gestures.map(withEmoji).join(' then ')}. ` +
          `The video shows the sentence in BSL (${s.bsl.map(w => w.toUpperCase()).join(' ')}).`;
      }
      playClips(demo, urls);
    });
    knownList.appendChild(b);
  }
  const empty = document.createElement('p');
  empty.className = 'muted small sign-empty';
  empty.textContent = 'No matching sign.';
  empty.hidden = true;
  knownList.appendChild(empty);
  filterKnownSigns();
  const note = document.createElement('p');
  note.className = 'muted small';
  note.textContent = (held.size ? 'Signs with a dashed border can also be signed without movement: make the hand shape at ' +
    'its place and hold it still for about a second. ' : '') + `Demo gestures (not BSL): ${RULE_GESTURES.map(withEmoji).join(', ')}. Click any sign or sentence above to see how to sign it.`;
  knownList.appendChild(note);
}

function status(t) { if (statusLine) statusLine.textContent = t; }

// ---------- calibration (record each sign SAMPLES times; examples stay in this browser)
const calibratedCount = () => Object.values(personal).filter(p => exampleCount(p) >= SAMPLES).length;
function allLabels() {
  const seen = new Map();
  for (const l of [...bslSigns().map(s => s.label), ...RULE_GESTURES]) if (!seen.has(keyFor(l))) seen.set(keyFor(l), l);
  return [...seen.values()].sort((a, b) => a.localeCompare(b));
}

const isCalibrated = l => exampleCount(personal[keyFor(l)]) >= SAMPLES;

// Plays the dictionary clip of the sign being calibrated, so the user copies that version of the sign.
function showCalibDemo(label) {
  if (!calibDemo) return;
  const url = clipIndex.words[keyFor(label)]?.clips[0];
  if (url && !RULE_GESTURES.includes(label)) {
    calibDemo.hidden = false;
    if (!calibDemo.src.endsWith(url)) calibDemo.src = url;
    calibDemo.play().catch(() => {});
    if (calibHint) calibHint.textContent = heldWords().has(keyFor(label))
      ? `How to sign "${label}" (dictionary clip). Copy it, or just hold its hand shape still at its place for a second, then press Record.`
      : `How to sign "${label}" (dictionary clip). Copy it, then press Record.`;
  } else {
    calibDemo.hidden = true;
    calibDemo.removeAttribute('src');
    if (calibHint) calibHint.textContent = `"${label}" is a demo hand shape: raise your hand, hold the shape still for a second, then lower it.`;
  }
}

function nextUncalibrated(after) {
  const labels = allLabels();
  const start = Math.max(0, labels.indexOf(after) + 1);
  return [...labels.slice(start), ...labels.slice(0, start)].find(l => !isCalibrated(l)) || null;
}

function fillCalibration() {
  if (!calibSign) return;
  const current = calibSign.value || nextUncalibrated(null);
  calibSign.innerHTML = '';
  for (const l of allLabels()) {
    const o = document.createElement('option');
    const n = exampleCount(personal[keyFor(l)]);
    o.value = l;
    o.textContent = n >= SAMPLES ? `${l} ✓` : l;
    calibSign.appendChild(o);
  }
  if (current) calibSign.value = current;
  const done = Object.values(personal).filter(p => exampleCount(p) >= SAMPLES);
  calibList.textContent = done.length
    ? `Calibrated by you (${done.length} of ${allLabels().length}): ` + done.map(p => p.label).sort().join(', ')
    : 'Nothing calibrated yet.';
  showCalibDemo(calibSign.value);
}
calibSign?.addEventListener('change', () => showCalibDemo(calibSign.value));

let capturing = null;   // { label, got }
async function startCalibration() {
  if (capturing) { stopCalibration('Calibration cancelled.'); return; }
  const label = calibSign.value;
  if (!label) return;
  if (!running) await startCam();
  if (!running) return;
  forget(personal, label);              // re-recording replaces the old examples
  capturing = { label, got: 0 };
  calibRecordBtn.textContent = 'Cancel';
  calibStatus.textContent = `Sign "${label}" now (1 of ${SAMPLES}). Hold still or lower your hands when you finish.`;
  session.reset();
  session.set('onCapture', (frames, held) => {
    if (!addExample(personal, label, frames, held)) {
      calibStatus.textContent = `Hands not seen clearly — sign "${label}" again (${capturing.got + 1} of ${SAMPLES}).`;
      return;
    }
    capturing.got++;
    if (capturing.got >= SAMPLES) {
      savePersonal(personal);
      applyIndex();
      const next = nextUncalibrated(label);
      stopCalibration(`✓ "${label}" calibrated.` + (next ? ` Next: "${next}" — watch the clip, then press Record.` : ' All signs calibrated!'));
      if (next) { calibSign.value = next; showCalibDemo(next); }
    } else {
      calibStatus.textContent = `Got it (${capturing.got} of ${SAMPLES}). Sign "${label}" again.`;
    }
  });
}

function stopCalibration(message) {
  if (capturing && capturing.got < SAMPLES) personal = loadPersonal();  // drop a half-finished recording
  capturing = null;
  session.set('onCapture', null);
  session.reset();
  calibRecordBtn.textContent = `Record ${SAMPLES} times`;
  calibStatus.textContent = message;
  fillCalibration();
}

calibRecordBtn?.addEventListener('click', startCalibration);
calibClearBtn?.addEventListener('click', () => {
  if (!confirm('Remove all your calibration recordings from this browser?')) return;
  personal = {};
  savePersonal(personal);
  applyIndex();
  stopCalibration('Your calibration was removed.');
});

// ---------- camera
async function startCam() {
  if (running || starting) return;    // a second click while loading used to start a second loop
  starting = true;
  try {
    if (!hands) await loadEngine();
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: 640, height: 480 }, audio: false });
    video.srcObject = stream;
    await video.play();
    running = true;
    session.reset();
    result.textContent = '— camera started —';
    const id = ++loopId;
    video.requestVideoFrameCallback(now => onFrame(now, id));
  } catch (e) {
    result.textContent = hands ? 'Camera permission denied or not available.' : 'Could not load sign recognition: ' + e.message;
  } finally {
    starting = false;
  }
}

function stopCam() {
  running = false;
  loopId++;              // any pending frame callback of the old loop is ignored
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  video.srcObject = null;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  result.textContent = '— waiting —';
  setBar(0);
  session.reset();
}

function resetLast() { words.pop(); renderSentence(); result.textContent = '— last removed —'; }
function resetAll() { words = []; session.reset(); session.clearHistory(); renderSentence(); result.textContent = '— reset —'; setBar(0); }

startBtn?.addEventListener('click', startCam);
stopBtn?.addEventListener('click', stopCam);
resetLastBtn?.addEventListener('click', resetLast);
resetAllBtn?.addEventListener('click', resetAll);

// ---------- per-frame processing
// Errors are caught so a single bad frame can never stop the camera loop (before, one exception
// silently froze detection while the video kept playing).
let lastError = '';
function reportError(where, e) {
  const msg = `${where}: ${e && e.message ? e.message : e}`;
  if (msg !== lastError) { lastError = msg; console.error(where, e); status('Problem — ' + msg + ' (still running)'); }
}

function onFrame(now, id) {
  if (!running || id !== loopId) return;
  if (rebuilding) { video.requestVideoFrameCallback(n => onFrame(n, id)); return; }
  const ts = Math.max(now, lastTs + 1);   // strictly increasing, even if the browser repeats a time
  lastTs = ts;
  try {
    const hr = hands.recognizeForVideo(video, ts);
    if (frameNo++ % 3 === 0) {             // body position changes slowly; every 3rd frame is enough
      const pr = pose.detectForVideo(video, ts);
      const p = pr.landmarks[0];
      // only use the body when both shoulders are really visible (sitting close to the camera, MediaPipe
      // guesses shoulder positions off-screen, which would distort where the hand is)
      const seen = q => q && (q.visibility ?? 1) > 0.5 && q.y < 1.02;
      lastPose = p && seen(p[11]) && seen(p[12]) ? { nose: [p[0].x, p[0].y], ls: [p[11].x, p[11].y], rs: [p[12].x, p[12].y] } : null;
    }
    const aspect = video.videoWidth / video.videoHeight;
    const frame = {
      aspect,
      hands: hr.landmarks.map((lm, i) => ({ label: hr.handedness[i][0].categoryName, lm: lm.map(p => [p.x, p.y, p.z]) })),
      body: lastPose,
    };
    draw(hr.landmarks);
    try {
      session.push(frame, hr.landmarks[0], hr.gestures[0]?.[0]);
    } catch (e) {
      reportError('sign matching', e);
      session.reset();
    }
    tickFps();
  } catch (e) {
    reportError('hand detection', e);
    rebuildModels();      // a MediaPipe graph error is permanent until the models are recreated
  }
  if (running && id === loopId) video.requestVideoFrameCallback(n => onFrame(n, id));
}

function accept(label, conf, how) {
  words.push(label);
  setBar(conf);
  result.textContent = `${label} • Confidence: ${conf}% — ${how}`;
  const joined = words.slice(-3).join(' ').toLowerCase();
  for (const [pat, obj] of Object.entries(window.UNIFIED_PATTERNS || {})) {
    if (joined.endsWith(pat)) {
      words = [];
      sentenceBox.textContent = obj.sentence;
      speak(obj.sentence);
      return;
    }
  }
  renderSentence();
  speak(label);
}

function renderSentence() {
  const s = words.join(' ');
  sentenceBox.textContent = s ? s.charAt(0).toUpperCase() + s.slice(1) : '(start signing...)';
}

function setBar(c) { if (confBar) confBar.style.width = Math.max(0, Math.min(100, c)) + '%'; }

function tickFps() {
  fpsCount++;
  const now = performance.now();
  if (now - fpsStart >= 1000) {
    fps = Math.round((fpsCount * 1000) / (now - fpsStart));
    fpsCount = 0; fpsStart = now;
    if (!lastError) status(`Running (${delegateUsed}) · ${fps} fps · ${bslSigns().length} BSL signs + ${RULE_GESTURES.length} demo gestures` +
      (calibratedCount() ? ` · ${calibratedCount()} calibrated` : ''));
  }
}

// ---------- drawing (accounts for the video being cropped with object-fit: cover)
const CONNS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20]];
function draw(handList) {
  overlay.width = overlay.clientWidth;
  overlay.height = overlay.clientHeight;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw) return;
  const scale = Math.max(overlay.width / vw, overlay.height / vh);
  const ox = (overlay.width - vw * scale) / 2, oy = (overlay.height - vh * scale) / 2;
  const X = p => ox + p.x * vw * scale, Y = p => oy + p.y * vh * scale;
  ctx.lineWidth = 2;
  ctx.strokeStyle = 'rgba(255,255,255,.85)';
  ctx.fillStyle = 'rgba(58,166,255,.95)';
  for (const lm of handList) {
    for (const [a, b] of CONNS) {
      ctx.beginPath(); ctx.moveTo(X(lm[a]), Y(lm[a])); ctx.lineTo(X(lm[b]), Y(lm[b])); ctx.stroke();
    }
    for (const p of lm) { ctx.beginPath(); ctx.arc(X(p), Y(p), 3, 0, Math.PI * 2); ctx.fill(); }
  }
}
