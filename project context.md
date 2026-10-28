# Project context — Real-Time BSL / GestureFlow BSL Translator

Hand-over document so work can continue in a new chat or another Claude account.
It covers everything done up to 28 Oct 2026: what was built, every method tried and its measured results,
the current status, and how to run, rebuild and test. If you're an assistant reading this, read it completely
before changing anything.

---

## 1. Who and what

- **Developer (the user):** an MVP developer at "Innovator One".
  - A software developer, not an ML/NLP engineer.
  - Wants honest, measured results, short time estimates, and is under job pressure.
- **Client:** Ankur Shah (CEO) and Pritesh Thakkar (CTO), Real-Time BSL Ltd (UK). Their Innovator Founder visa business plan is
  `C:\Users\devan\OneDrive\Desktop\Innovator One\Business Plan Real-Time BSL (2).pdf`.
  **Never modify that PDF.**
- **Product idea (business plan):** real-time British Sign Language (BSL) ↔ English. Webcam signing becomes text and voice, and
  speech becomes signs. Sold B2B (NHS, councils, schools, retail).
- **The code:** the client's demo, `C:\Users\devan\OneDrive\Desktop\Innovator One\ankur and pritesh\GestureFlow-BSL-Translator-main`.
  - Django 5.2.7 + SQLite; all translation logic is browser JavaScript.
  - A git repo exists (first commit "first commit"); the user pushes to GitHub themselves.
- **Agreed scope:** improve only the client's two features, using open data:
  - **Hand → Speech** (sign → spoken English)
  - **Speech → Sign** (spoken English → BSL video)

### User rules and preferences (keep following these)
- **Output:** no emoji in any output; Speech → Sign must show real BSL clips plus fingerspelling.
- **Scope:** no new admin, CSV-upload or other features; don't add Firebase (stay in the client's Django stack).
- **ML:** first "no AI training"; from 27 Oct ML is allowed. The user said: "you have full freedom … make 40 words
  in a working condition for hand to speech, it can be ANY 40 words".
- **Hand → Speech constraints:** must work without calibration by the client, with no own recordings available,
  and run in the browser.
- **Honest numbers:** quote measured numbers only, never estimates. Small-data selection is heavily flattering, so always give
  honest (locked-away / held-out) figures.
- **Ports:** the user runs their own server on **port 8000**; never touch it. Previews run on **port 8010**
  (`ankur and pritesh/.claude/launch.json`, `manage.py runserver 127.0.0.1:8010 --noreload`; restart it after
  template or URL changes because of `--noreload`).
- **Long background jobs:** the laptop overheats. The user may ask to pause; suspend processes with `psutil`
  (`p.suspend()` / `p.resume()`). They have an RTX 4050 laptop GPU (6 GB).

---

## 2. What the client's code originally did (analysis, 21 Oct)

- **Hand → Speech:**
  - Legacy `@mediapipe/hands` (unpinned CDN) plus hard-coded if/else rules on finger states.
  - 9 one-handed generic gestures: Hello, Four, Three, Peace, Help, Good job, No, Stop, Love.
  - Problems:
    - fake fixed confidence numbers;
    - "Stop" impossible (rule order);
    - "Yes" and "Thank you" advertised but impossible;
    - single signs never spoken (only 2–3-gesture combos);
    - not scale or rotation safe;
    - fake "calibrate";
    - no voice language set.
- **Speech → Sign:**
  - Web Speech API, but output was **emoji**.
  - It matched speech words against gesture *names*, so "hello" became "Hello, how are you?" 👋✌️.
  - A stop-word list deleted real words.
  - The listener stopped after a pause, and unknown speech showed a "didn't understand" message.
- **Other (outside scope, noted only):**
  - `translate_api` returned a random fake translation (now removed).
  - Hard-coded `SECRET_KEY`, `DEBUG=True`, auto-created `admin/admin123` superuser.
  - Subscription prices (£5/£10) don't match the business plan (£199/£499/~£1,000).
- **Business-plan red flags given to the user for the meeting:**
  - copy-pasted sections (IllumiShield lighting, "Zero-Knowledge Proof");
  - inconsistent accuracy and latency figures;
  - TRL-7 claims;
  - the demo signs aren't real BSL.

A long briefing (Parts A–G: business plan explained, MVP features, dataset research) is in
`ankur and pritesh/Real-Time-BSL_MVP_Briefing.md`. The plan history is in `C:\Users\devan\.claude\plans\c-users-devan-onedrive-desktop-innovato-kind-planet.md`.

---

## 3. Datasets researched and chosen

| Dataset | Verdict |
|---|---|
| **BSLDict** (Oxford VGG, github.com/gulvarol/bsldict): 14,122 clips, 9,283 words, sign trim points (`temp_interval`) | **Used.** CC BY 4.0 for the dataset, but **video copyright stays with the original owners**. Fine for the MVP demo; get permission or own recordings before charging customers |
| BOBSL / BSL-1K (BBC + Oxford) | Non-commercial research only; not used |
| BSL Corpus (UCL, CC BY-SA, lexical elicitation, many signers) | Would be great data, but videos are long sessions that need cutting; not used yet |
| BSL SignBank (UCL) | Reference only (permission needed) |
| WordNet (via NLTK) | Used for Speech → Sign synonyms and irregular forms |
| wordfreq (pip, offline only) | Used to choose common everyday words |

**Pre-trained models checked (27 Oct):**
- **Oxford BSL-1K I3D:** no licence in the repo; needs a Python/GPU server and sends video off the device. Rejected.
- **Google MediaPipe Gesture Recognizer:** used for the demo gestures (7 classes).
- **MediaPipe Model Maker:** deprecated and only handles static images.
- **Kaggle/Google sign models:** ASL, not BSL.

---

## 4. Timeline of work, methods tried and results

All accuracy figures are from automated tests on BSLDict clips by signers other than the matched examples
(no webcam), unless stated. "Session test" means clips are streamed through the page's real live logic
(`sign_session.js`), with the clip's own example removed and landmark jitter added.

### Phase 1 (21 Oct): first data-backed version, no ML
- **Pipeline:** download a BSLDict subset → trim with ffmpeg → MediaPipe hand + pose landmarks offline → build JSON for the browser.
  - First subset: 244 words / 853 clips (`media/signs`).
- **Speech → Sign:** rewritten (`speech_lookup.js`, `speech_app.js`).
  - Lookup order: phrase → word → alternative word → irregular form → word stem → WordNet synonym → fingerspelling a–z.
  - A video player plays the clips in order (speed and replay controls, auto-restarting microphone). No emoji.
  - 209 words had clips at this point.
- **Hand → Speech:** template matching with no training.
  - **Per-frame features:** 53 numbers.
    - Per hand: 18 finger-joint angles and spreads, 5 orientation values, 2 for wrist location relative to the shoulders.
    - Two hand slots (dominant / other) plus a body flag.
    - Hand points are smoothed over 5 frames.
  - **Matching:** each sign is resampled to 12 frames and compared by DTW (band 3). The live sign is read with either hand as dominant.
  - **Exact parity:** Python `scripts/sign_features.py` and JS `demo/static/demo/sign_matcher.js` agree to about 1e-15.
- **Vocabulary:** 36 words chosen by greedy removal, plus the 9 demo gestures (44 total). The user fixed it at 44 at the time.
- **Personal calibration:** 3 recordings per sign, kept in browser localStorage.
- **Results:**
  - Session test at 25 fps: 54% right / 13% wrong / 33% "not sure".
  - Calibrated (simulated same signer): 100%.
- **Tried, gave ≤1 point:** weight tuning (`tune_params.py`, since removed) and face-relative location.

### Phase 2 (22 Oct): fixes from the user's webcam tests
- **"Everything comes out as sad":** a held hand was matched to the nearest *moving* sign.
  - Fix: route held hand shapes (short raise, then still for 1 s) to the hand-shape logic. Added jitter-proof motion measures, a waving detector, and no repeating the same word unless the hands leave the camera.
- **Detection stopped, MediaPipe "Packet timestamp mismatch":**
  - Fix: a single frame loop (`loopId`), strictly increasing timestamps, try/catch around each frame, and a model rebuild after a graph error.
- **The user's webcam runs at ~7 fps, and moving signs weren't detected:**
  - Fix: the session works in milliseconds, not frames, and each finished sign is rebuilt at 25 fps (`resampleTo25fps`); distance slack 1.25.
  - At 7 fps: 44% right / 17% wrong uncalibrated; 97% calibrated.
- **Demo gestures:**
  - Switched the page to **Google MediaPipe GestureRecognizer** (it also gives the 21 hand points). Mapping: Open_Palm → Stop or Hello (if waving), Victory → Peace, Thumb_Up → Good job, Thumb_Down → No, Pointing_Up → Help, ILoveYou → Love.
  - Our rules (`gesture_rules.js`) handle Three/Four.

### Phase 3 (26 Oct): held-shape words ("words that don't need movement")
- Each clip's still moment is averaged into one 53-number vector, and words are matched by nearest neighbour.
- **First version:** 23 words at 92% held-out, but they stole the demo shapes: Peace became "2", Four became "when"/"like".
- **Fixes:**
  - Added the demo shapes as **competitor templates** (BSLDict "two", "v", "four", "five", "three", "good", "one").
  - Left out words whose held shape *is* a demo gesture (2, what, understand, like, when, here).
  - Precedence: a two-handed word with two hands held, then Google's confident demo gesture, then the held word, then the rules (Three/Four), then "not sure".
  - Calibration of a held sign saves a held-shape example (`statics`).
- **Result:** held words 87% right at 25 fps (before: 24%).

### Phase 4 (27 Oct): ML allowed; target 40+ working words for strangers
1. **Larger pool:**
   - `make_candidates.py` gives 1,071 everyday BSLDict words with ≥3 clips (wordfreq Zipf ≥ 3.5, minus specialist glossaries and religious terms).
   - 3,908 videos downloaded (255 dead links); landmarks for 4,077 clips across 1,162 words.
   - Speech → Sign now has **1,162 words**.
2. **Sign embedding (the ML part):** `scripts/train_embedding.py`.
   - **Network:** PyTorch MLP, input 12×53 → hidden 256 → 128 → 64-dim unit vector.
   - **Training:** cosine-softmax classification over *other* words, with label smoothing.
   - **Augmentation:** jitter, time crop, in-plane rotation, location shift, handshape noise.
   - **Evaluation:** 5-fold **by word**, so a word's clips are always embedded by a network that never saw that word (honest few-shot test).
   - **Matching:** combined distance = `dtw/dtwScale + 2 × (1 − cosine)/embScale` (scales = medians).
   - **Browser:** plain JS (`embed`, `unpackEmbedding` in `sign_matcher.js`); weights stored as 8-bit numbers in base64 (~0.3 MB per network). Python/JS parity is about 1e-16.
3. **Results on random 40-word sets (no word choosing; an honest measure of the method):**

   | Method | Right |
   |---|---|
   | DTW only (old) | 24–26% |
   | Single network | 45% |
   | DTW + single network | 48–52% |
   | Network sizes: (256,128) vs (512,256) | 45.2% vs 41.2% (smaller is better) |
   | Variants: base / reg (dropout 0.4, wd 0.01) / 150 epochs / big+reg | 45.2 / **47.7** / 43.1 / 45.9% |
   | **Ensemble of those 4 different types** | **52.4%** |
   | 3 networks with 3,203 extra training-only clips of 1,607 more words | 45.8% each, 49.2% as an ensemble (extra words did **not** help) |
   | All 7 networks | 52.9% (not worth double the size) |
4. **Choosing the 40+ words, and the selection-bias lesson:**
   - Greedy "keep the words matched right" on 978 words claimed **99.7%**, but the honest locked-clip test gave **50%**.
     With only 3–6 clips per word, the selection mostly picks words that were lucky on those exact clips.
   - Honest test (`select_words.py --holdout`): in each of 4 rounds, lock one clip per word away, choose words without it, and test only the locked clips.
   - **Strategies compared** (weight 2, single network):

     | Strategy | Locked-clip accuracy |
     |---|---|
     | accuracy-greedy | 69.6% |
     | **mean margin, words with 4+ clips** | **71.7%** (most reliable: all 45 words tested) |
     | min margin | lower |
     | DTW only (same test) | 37.5% |

   - **With the 4-network ensemble: locked-clip 78.6%.** That gave the final 45 words.
5. **Live-page tuning:**
   - Our rules mistook open hands inside moving signs ("father", "cake") for "Stop".
     Now **only Google's recogniser may call a demo gesture during a moving sign**.
   - Default confidence raised 35 → 40.
   - Held words that looked like Four/Stop/Three when held (cake, daughter, orange) were removed from the held list (they still work as moving signs).
6. **GPU:** a separate venv `C:\Users\devan\bsl-data\.venv-gpu` with PyTorch 2.14.1+cu130 (RTX 4050).
   - `train_embedding.py` uses CUDA automatically: 11 min per 5-fold network vs ~40 min on the CPU; the final 4 networks take 5.5 min.

### Phase 5 (27–28 Oct): cleanup
- **Removed files:**
  - `demo/static/demo/models/hand_landmarker.task` (unused; GestureRecognizer includes hand tracking)
  - `scripts/wordlist_mvp.txt`
  - `scripts/tune_params.py`
  - all `__pycache__`
- **Removed code:**
  - the fake `/translate/` API
  - `best_distance()`
  - the automatic word-choice mode in `build_index.py` (`--vocab` is now required)
- **Docs:** README.txt and README_FIRST.txt updated.
- **Verification:**
  - Rebuilt data identical to the live data.
  - `manage.py check` clean; all pages load.
  - All tests unchanged.
- The user had pushed to GitHub *before* the cleanup. The cleanup itself had not been committed at the time of writing.

---

## 5. Current status (28 Oct 2026)

### Hand → Speech: 45 BSL words + 9 demo gestures, no calibration needed
**Words:** agree, baby, black, boat, brother, cake, chocolate, choose, dark, date, daughter, dinner, donkey, door, England, father,
fog, frog, fruit, goodbye, helicopter, how, how much, I understand, internet, interview, life, many, orange, own, partner,
picture, plant, rabbit, rat, sad, stop, Sunday, tea, temperature, tool, weekend, what, when, yesterday.

**Also signable by holding the hand shape still (9):** boat, England, father, frog, helicopter, I understand, rabbit, Sunday, temperature.

**Demo gestures (9):** Hello (wave), Stop, Peace, Help, Good job, No, Love (Google recogniser); Three, Four (rules).

**Measured accuracy (final build):**

| Test | Old method (DTW only) | **Current** |
|---|---|---|
| Honest locked-clip test (45 words chosen without the tested clips) | 37.5% | **78.6%** |
| Build report, held-out top-1 within the 45 words | 61% | **70.5%** |
| Session test, 25 fps, confidence 40 (`test_session.mjs`) | 45% right / 25% wrong (same words, conf 35) | **59% right / 18% wrong / 23% not sure** (76% of spoken words right) |
| Session test, 7 fps | – | 53% right / 17% wrong / 30% not sure |
| Held-shape words (`test_held.mjs`, with Google labels) | – | **94%** at 25 fps, 81% at 7 fps; demo shapes unaffected |
| Same person after calibration (`test_calibration.mjs all`) | – | **100%** (45/45) at 25 and 7 fps |

**Not yet done:**
- No real-webcam test of the final version.
- No test with real Deaf signers.
- Real users vary more than dictionary signers, so expect lower live numbers.

**Main limit:**
- BSLDict has only 4–6 clips per word, often regional variants.
- Advice for demos: copy the clip shown when clicking a word in "Signs I can recognise", sign clearly, then pause or lower the hands.
- The demo person can also use "Calibrate my signs" (3 recordings per word).

### Speech → Sign
- 1,162 words with real BSL clips (163 MB in `media/signs`), plus phrases, stems, WordNet synonyms (194) and irregular forms (293).
- Fingerspelling for everything else; no emoji.
- Chrome or Edge needed (Web Speech API; it sends audio to Google, so tell the client because of the GDPR claims).

---

## 6. Project files (what each does)

```
GestureFlow-BSL-Translator-main/
  manage.py, bsl_full_v1/ (settings: MEDIA_URL/MEDIA_ROOT added; urls serve /media/), demo/ (Django app)
  demo/views.py, urls.py          pages: / , /speech-to-sign/, login/signup/logout, /subscription/, /contact/, /admin/
  templates/base.html             footer credits (BSLDict, MediaPipe, WordNet)
  demo/templates/demo/index.html  Hand→Speech page (status, ?debug=1 box, calibration panel, "Signs I can recognise", Confidence slider default 40)
  demo/templates/demo/speech.html Speech→Sign page (video player)
  demo/static/demo/
    script.js          Hand→Speech page controller: camera, GestureRecognizer + PoseLandmarker (CDN tasks-vision 1.1.1, self-hosted models),
                       frame loop (crash-proof), calibration UI, sign list, speech output (en-GB), sentence combos
    sign_session.js    splits the live stream into signs; held-shape vs moving routing; precedence rules; time-based (any fps)
    sign_matcher.js    features, DTW, sign embedding (embed/unpackEmbedding), matchSequence, matchStatic — mirrors sign_features.py
    calibration.js     personal recordings in localStorage (key gestureflow.calibration.v1): templates (moving) + statics (held)
    gesture_rules.js   rule-based Three/Four (+ fallbacks); RULE_GESTURES list
    unified_patterns.js sentence combos for demo gestures (emoji removed)
    speech_lookup.js, speech_app.js   Speech→Sign lookup and player
    models/            gesture_recognizer.task, pose_landmarker_lite.task (self-hosted)
    data/sign_templates.json  45 words' templates + held shapes (+ demo competitors) + the 4 embedding networks + params/thresholds (~2.4 MB)
    data/sign_clips.json      1,162 words -> clip URLs; data/synonyms.json; data/eval_report.json (measured accuracy)
  media/signs/<word>/*.mp4   trimmed BSLDict clips (own recordings would be own_*.mp4 and listed first)
  scripts/ (offline, run with C:\Users\devan\bsl-data\.venv python)
    make_candidates.py   -> wordlist_candidates.txt (1,071 everyday words, >=3 clips)
    download_subset.py   BSLDict videos (HTTPS / yt-dlp)
    process_clips.py     ffmpeg trim to 25 fps/360p + MediaPipe landmarks -> media/signs + landmark JSON
    pkl_to_json.py       SAFE loader for bsldict_v1.pkl (never plain pickle.load)
    sign_features.py     Python twin of sign_matcher.js (features, DTW, static vectors, embedding, fuse)
    train_embedding.py   sign-embedding networks (k-fold out-of-fold eval, or --folds 0 final; GPU if available; --train-extra, --ensemble)
    merge_embeddings.py  merge final networks -> scripts/sign_embedding.json (the 4-network ensemble, 1.4 MB)
    select_words.py      choose the words (margin method, 4+ clips, --holdout honest test, --skip select_skip.txt, distance cache)
    build_index.py       builds sign_clips.json, sign_templates.json, eval_report.json (--vocab required, --static-vocab, --embedding, --emb-weight 2)
    build_synonyms.py    WordNet synonyms/irregulars for Speech→Sign
    add_recordings.py    add the client's own recordings (own_*.mp4) for both features
    vocabulary.txt, static_vocabulary.txt, select_skip.txt, wordlist_candidates.txt
    test_session.mjs <landmarks> [fps]                 live-session test (moving words)
    test_held.mjs <landmarks> [fps] [google_labels]    held-shape words + demo-shape protection
    test_calibration.mjs <landmarks> all|none [hard]   (env FPS=7) same-signer after calibration
  README_DATA.md   full method, measured accuracy, rebuild commands, licences (keep it up to date)
  README.txt, README_FIRST.txt   how to run
  project context.md  (this file)
```

---

## 7. Outside the project: data and tools (`C:\Users\devan\bsl-data`, not in git)

- **Environments:**
  - `.venv`: Python 3.10 with mediapipe 1.1.1, opencv, yt-dlp, imageio-ffmpeg, nltk, numpy, wordfreq, psutil, torch (CPU), scikit-learn.
  - `.venv-gpu`: numpy + torch 2.14.1+cu130 (GPU training).
- **Inputs:**
  - `meta/bsldict_meta.json`: BSLDict metadata converted safely.
  - `models/`: hand / pose / gesture `.task` files for the offline scripts.
  - `nltk_data/`
- **Downloads and landmarks:**
  - `raw_videos/`: downloaded originals (~7,100), with `failed.txt` for dead links.
  - `landmarks/`: the first 853 clips.
  - `pool/landmarks/`: 4,077 clips (**use this for builds and tests**).
  - `pool/clips/`: trimmed clips, already copied into `media/signs`.
  - `train_extra/`: 3,203 training-only clips of 1,607 more words (didn't help).
- **Network outputs:**
  - `pool/emb_v_base|emb_v_reg|emb_v_ep150|emb_v_bigreg/oof_embeddings.json`: out-of-fold networks used for word selection.
  - `pool/final_*`: final networks (merged into `scripts/sign_embedding.json`).
  - `pool/embedding_features_c6.npz`: cached training features.
- **Selection files:**
  - `pool/sel/`: selection outputs and `raw_lists_cache.json` (distance cache; delete it if the networks change).
  - `pool/raw_all.json`: distance cache used by `experiments/sel_strategies.py`.
- **Test data:** `experiments/google_labels.json`, Google gesture labels per frame for the `test_held.mjs` clips.
- **Other:** `experiments/` (one-off analysis scripts), `backup_data/`, `pool/dtw_only/` (DTW-only comparison build).

---

## 8. How to run, rebuild and test

**Run the web app (user):** double-click `RUN_WINDOWS.bat`, or:
```
venv\Scripts\activate
python manage.py migrate
python manage.py runserver
```
- Open http://127.0.0.1:8000/ in Chrome or Edge and press Ctrl+F5 after data changes.
- `/?debug=1` shows why each sign was or wasn't recognised.
- Demo admin account: admin / admin123.

**Rebuild the data (with the current word lists and networks):**
```
cd C:\Users\devan\bsl-data
set P=..\OneDrive\Desktop\Innovator One\ankur and pritesh\GestureFlow-BSL-Translator-main
.venv\Scripts\python -I "%P%\scripts\build_index.py" --meta meta\bsldict_meta.json --clips "%P%\media\signs" --landmarks pool\landmarks --out "%P%\demo\static\demo\data" --vocab "%P%\scripts\vocabulary.txt" --static-vocab "%P%\scripts\static_vocabulary.txt" --embedding "%P%\scripts\sign_embedding.json" --emb-weight 2.0
.venv\Scripts\python -I "%P%\scripts\build_synonyms.py" --clips-index "%P%\demo\static\demo\data\sign_clips.json" --out "%P%\demo\static\demo\data\synonyms.json" --nltk-data nltk_data
```
The full pipeline (candidates → download → landmarks → 4 k-fold networks → select_words → final networks → merge → build)
is written out in `README_DATA.md` §3.

**Tests (no camera needed; run from the project folder):**
```
node scripts/test_session.mjs C:/Users/devan/bsl-data/pool/landmarks 25        (and 7)
node scripts/test_held.mjs C:/Users/devan/bsl-data/pool/landmarks 25 C:/Users/devan/bsl-data/experiments/google_labels.json
node scripts/test_calibration.mjs C:/Users/devan/bsl-data/pool/landmarks all   (FPS=7 for 7 fps)
venv\Scripts\python manage.py check
```
**Expected:** 59/18/23 (25 fps), 53/17/30 (7 fps), held 94%, calibration 100%.

### Gotchas learned
- Keep `sign_features.py` and `sign_matcher.js` in exact parity. After any matcher change, check with a parity script; examples are in `bsl-data/experiments/parity_*.mjs`.
- Parameters travel in `sign_templates.json` → `params` (`embWeight` 2, `dtwScale`, `embScale`, and the DTW weights).
- With 3–6 clips per word, any "choose the best words" figure is flattered. Report the locked-clip / held-out numbers.
- Windows Python can't read `/c/...` paths; use relative or `C:/` paths. Python heredocs with Windows paths need raw strings.
  `python -I` ignores env vars, including PYTHONDONTWRITEBYTECODE.
- The preview server uses `--noreload`, so restart it after template or URL changes.
- When killing processes by command line, exclude your own process: a filter on "train_embedding.py" once killed the shell running it.

---

## 9. Ideas not done yet (possible next steps)

- **Biggest win:** the client's own recordings.
  - 5–10 takes per word from 2–5 signers, with consent.
  - Add them with `scripts/add_recordings.py`, then rebuild.
  - This also replaces third-party clips (licensing).
- **More data:**
  - BSL Corpus lexical-elicitation videos (many signers per concept; CC BY-SA; need cutting by ELAN annotations).
  - Ask UCL/Oxford about commercial use.
- **Real-webcam evaluation** with the user and, ideally, Deaf signers. Use the `?debug=1` output to tune the confidence (now 40) and the timing.
- **Model ideas not tried:**
  - temporal-convolution or transformer networks;
  - metric-learning losses (triplet / prototypical);
  - test-time augmentation.
- **Security** before any deployment: hard-coded SECRET_KEY, DEBUG=True, auto admin/admin123, empty ALLOWED_HOSTS.
- **Subscription prices** in the app don't match the business plan.
