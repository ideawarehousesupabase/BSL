// Speech -> Sign: listens in British English and plays the matching BSL video clips in order.
// Words without a clip are fingerspelled letter by letter.
import { createLookup } from './speech_lookup.js';

const heard = document.getElementById('heard');
const player = document.getElementById('signPlayer');
const caption = document.getElementById('signCaption');
const chips = document.getElementById('signChips');
const startBtn = document.getElementById('startRec');
const stopBtn = document.getElementById('stopRec');
const clearBtn = document.getElementById('clearRec');
const replayBtn = document.getElementById('replaySigns');
const speedSel = document.getElementById('signSpeed');
const statusLine = document.getElementById('speechStatus');

let translate = null;
let rec = null, listening = false;
let queue = [], pos = 0, playing = false;

const ready = Promise.all([
  fetch('/static/demo/data/sign_clips.json').then(r => r.json()),
  fetch('/static/demo/data/synonyms.json').then(r => (r.ok ? r.json() : {})).catch(() => ({})),
]).then(([clips, syn]) => {
  translate = createLookup(clips, syn);
  const n = Object.keys(clips.words).filter(w => w.length > 1).length;
  status(`${n} BSL signs available, plus fingerspelling for any other word.`);
}).catch(() => status('Could not load the sign dictionary.'));

function status(t) { if (statusLine) statusLine.textContent = t; }
function show(t, interim = false) { heard.textContent = 'Heard: ' + t; heard.classList.toggle('interim', interim); }

// ---------- playback
function speed(item) {
  const base = parseFloat(speedSel?.value || '1');
  return item.kind === 'letter' ? base * 1.5 : base;
}

function renderChips() {
  chips.innerHTML = '';
  queue.forEach((item, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'chip' + (item.kind === 'letter' ? ' letter' : '') + (i === pos && playing ? ' current' : '') + (i < pos ? ' done' : '');
    b.textContent = item.label;
    b.title = item.kind === 'letter' ? `fingerspelling "${item.word}"` : `heard "${item.heard}" (${item.via})`;
    b.addEventListener('click', () => playFrom(i));
    chips.appendChild(b);
  });
}

function playFrom(i) {
  if (i >= queue.length) { playing = false; renderChips(); return; }
  pos = i;
  playing = true;
  const item = queue[i];
  caption.textContent = item.kind === 'letter' ? `Fingerspelling: ${item.word.toUpperCase()} — ${item.label}` : item.label;
  renderChips();
  if (!item.url) {  // no clip for this letter: show it as text briefly
    player.removeAttribute('src'); player.load();
    setTimeout(() => { if (queue[pos] === item) playFrom(pos + 1); }, 800 / speed(item));
    return;
  }
  player.src = item.url;
  player.playbackRate = speed(item);
  player.play().catch(() => {});
}

player?.addEventListener('ended', () => playFrom(pos + 1));
player?.addEventListener('error', () => playFrom(pos + 1));
player?.addEventListener('loadedmetadata', () => { player.playbackRate = speed(queue[pos] || {}); });

function enqueue(text) {
  const items = translate(text);
  if (!items.length) return;
  const startAt = queue.length;
  queue.push(...items);
  if (!playing) playFrom(startAt); else renderChips();
}

// ---------- listening
function start() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { show('speech recognition is not supported in this browser — please use Chrome or Edge.'); return; }
  if (listening) return;
  ready.then(() => {
    if (!translate) return;
    rec = new SR();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = 'en-GB';
    rec.onresult = e => {
      let interim = '';
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const r = e.results[i];
        if (r.isFinal) { show(r[0].transcript.trim()); enqueue(r[0].transcript); } else interim += r[0].transcript;
      }
      if (interim) show(interim.trim() + ' …', true);
    };
    rec.onerror = e => {
      if (e.error === 'not-allowed' || e.error === 'service-not-allowed') { listening = false; show('microphone permission denied.'); }
    };
    // Chrome stops recognition after a pause; restart while the user still wants to listen.
    rec.onend = () => { if (listening) setTimeout(() => { try { rec.start(); } catch (_) { /* already started */ } }, 250); };
    listening = true;
    rec.start();
    show('Listening…');
  });
}

function stop() { listening = false; if (rec) { rec.stop(); rec = null; } show('Stopped listening.'); }

function clearAll() {
  queue = []; pos = 0; playing = false;
  player.removeAttribute('src'); player.load();
  caption.textContent = '—';
  heard.textContent = 'Heard: —';
  renderChips();
}

startBtn?.addEventListener('click', start);
stopBtn?.addEventListener('click', stop);
clearBtn?.addEventListener('click', clearAll);
replayBtn?.addEventListener('click', () => playFrom(0));
