# Sign data pipeline (BSLDict → GestureFlow)

The two translation pages use real British Sign Language video from **BSLDict**
(University of Oxford VGG). Signs are recognised by comparing the live hand movement with the
recorded dictionary clips, helped by a small neural network (the "sign embedding") that was trained on
other dictionary words and runs in the browser. No video leaves the device.

| Page | Uses | Files |
|---|---|---|
| Speech → Sign | Trimmed BSL clips for **1,162 words**, plus fingerspelling (a–z) for anything else | `media/signs/<word>/*.mp4`, `demo/static/demo/data/sign_clips.json`, `synonyms.json` |
| Hand → Speech | Landmark examples of the **45 words** in `scripts/vocabulary.txt` (9 of them also as held hand shapes), the sign-embedding networks, and each user's own calibration recordings (stored in their browser) | `demo/static/demo/data/sign_templates.json` |
| (report) | Leave-one-out accuracy, measured, not estimated | `demo/static/demo/data/eval_report.json` |

## How Hand → Speech recognises signs

1. MediaPipe finds 21 points per hand and the shoulders, in every camera frame.
2. Each hand point is smoothed over 5 frames. Each frame then becomes: finger-joint bend angles and
   finger spread (handshape), pointing and palm direction (orientation), and wrist position relative
   to the shoulders (location). Angles don't change with hand size, rotation or camera distance.
3. A finished sign (12 frames × 53 numbers) is compared with every dictionary example in two ways:
   - **dynamic time warping** (DTW) over the frames;
   - the **sign embedding**: 4 small neural networks (`scripts/train_embedding.py`, PyTorch, offline)
     each turn the sign into 64 numbers so that the same sign by different signers lands close
     together. They were trained on ~930 *other* dictionary words (never on the 45 words they
     recognise), so they learned what differs between signers and what differs between signs.
     The browser runs them as plain JavaScript (`embed` in `sign_matcher.js`, about 1.4 MB of
     8-bit weights, `scripts/sign_embedding.json` → `sign_templates.json`).
   The two distances are added (`dtw/dtwScale + 2 × embedding/embScale`). The live sign is read both
   ways round (either hand as dominant). The closest word wins if it is close and clear enough.
4. **Vocabulary (45 words)**: agree, baby, black, boat, brother, cake, chocolate, choose, dark, date,
   daughter, dinner, donkey, door, England, father, fog, frog, fruit, goodbye, helicopter, how, how much,
   I understand, internet, interview, life, many, orange, own, partner, picture, plant, rabbit, rat, sad,
   stop, Sunday, tea, temperature, tool, weekend, what, when, yesterday — plus the 9 demo gestures
   (Hello, Stop, Four, Three, Peace, Help, Good job, No, Love).
   Chosen by `scripts/select_words.py` from the 429 everyday BSLDict words with at least 4 clips
   (several signers): the words whose clips are most clearly closer to each other than to any other
   word. `scripts/select_skip.txt` lists technical/odd words never chosen.
5. **Any webcam speed.** All timings are in milliseconds, not frames, and each finished sign is rebuilt
   at 25 fps before matching, so a slow 7 fps webcam behaves like the 25 fps dictionary clips.
6. **Held hand shapes vs moving signs** (`sign_session.js`). A short raise followed by a still hold
   (1 s, or 0.4 s before the hands drop) is a held hand shape. Demo gestures are decided by **Google's
   MediaPipe Gesture Recognizer** (open palm → Stop/Hello, V → Peace, thumbs up → Good job, thumbs down
   → No, pointing up → Help, I-love-you → Love); our rules in `gesture_rules.js` add Three and Four.
   During a moving sign only Google's recogniser may call a demo gesture (our rules mistook the open
   hand in "father" or "cake" for Stop). The same word is not added twice in a row unless the hands
   leave the camera in between. Open `/?debug=1` to see how each sign was decided.
7. **Held-shape words** (`scripts/static_vocabulary.txt`): boat, England, father, frog, helicopter,
   I understand, rabbit, Sunday, temperature can also be signed by holding the hand shape still at its
   place for about a second (dashed border in the sign list). The demo hand shapes compete in the same
   matcher, so a held Peace/Four/Stop is not read as a word. Cake, daughter and orange are left out:
   held still they look like Four/Stop/Three.
8. **Personal calibration** ("Calibrate my signs" on the page): 3 recordings per sign, kept in the
   browser (localStorage), never uploaded. Not needed, but it raises accuracy further for that person.

### Measured accuracy (27 Oct 2026)

All tests use dictionary clips by signers other than the examples being matched (no camera needed).

| Test | Old method (DTW only) | **New method (DTW + sign embedding)** |
|---|---|---|
| Honest word choice test: one clip of every word locked away *before* choosing 45 words, only those clips tested (`select_words.py --holdout`, 4 rounds, 360 tests) | 37.5% right | **78.6% right** |
| Random 40-word sets, no choosing at all (60 sets) | 24.5% right | **52.5% right** |
| `node scripts/test_session.mjs <landmarks> 25`: every clip of the 45 words streamed through the page's own sign splitter, its own example removed, live-like jitter, default confidence 40 | 45% right, 25% wrong (same 45 words) | **59% right, 18% wrong, 23% "not sure"** (76% of spoken words right) |
| Same at a **7 fps** webcam | — | **53% right, 17% wrong**, 30% "not sure" |
| `node scripts/test_held.mjs <landmarks> 25 google_labels.json`: held-shape words raised and held | — | **94% right** (34/36) at 25 fps, 81% at 7 fps; demo hand shapes unchanged |
| `node scripts/test_calibration.mjs <landmarks> all`: the same person signs again after calibrating | — | **100% (45/45)** at 25 and 7 fps |

The live-session figure is lower than the "honest" 78.6% because the page must also find where each
sign starts and ends, and refuses to speak when unsure (the "Confidence %" slider: 25 → 64% right /
24% wrong, 40 → 60% / 18%, 45 → 58% / 14%). The limit is the data: BSLDict has only 4–6 clips per
word, often regional variants, so a signer using another variant is not recognised. Real users vary
more than dictionary signers, so expect live webcam results somewhat below these figures. Copying the
dictionary clip shown for each sign ("Signs I can recognise") gives the best results. Confirm with real
Deaf signers before quoting numbers to pilot partners.

**Shared improvement for everyone:** record each sign 5–10 times with 2–5 of your own signers and add
them with `scripts/add_recordings.py` (section 4). That raises accuracy for users who have not
calibrated, and replaces third-party clips on Speech → Sign.

The web app itself still only needs Django. The steps below are only needed to rebuild
or extend the data.

## 1. One-off setup (outside the project folder, e.g. `C:\bsl-data`)

```bat
py -m venv .venv
.venv\Scripts\python -m pip install mediapipe opencv-python yt-dlp imageio-ffmpeg nltk numpy wordfreq psutil
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
curl -L -o models\hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
curl -L -o models\pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
curl -L -o meta\bsldict_v1.pkl https://www.robots.ox.ac.uk/~vgg/research/bsldict/data/bsldict_v1.pkl
.venv\Scripts\python -c "import nltk; nltk.download('wordnet', download_dir='nltk_data')"
```

With an NVIDIA GPU, a second environment with the CUDA build of PyTorch trains the networks about 4×
faster (`py -m venv .venv-gpu`, then `pip install numpy torch --index-url https://download.pytorch.org/whl/cu130`);
`train_embedding.py` uses the GPU automatically when it finds one.

`bsldict_v1.pkl` is a Python pickle. Convert it with the safe loader (`scripts/pkl_to_json.py`), which
refuses anything except numpy arrays, and never load it with plain `pickle.load`:

```bat
.venv\Scripts\python <project>\scripts\pkl_to_json.py meta\bsldict_v1.pkl meta\bsldict_meta.json
```

## 2. Choose words

- `scripts/wordlist_candidates.txt`: the 1,071 everyday BSLDict words with at least 3 clips
  (made by `scripts/make_candidates.py`; specialist glossaries and religious terms left out). These are
  downloaded; Speech → Sign plays all of them.
- `scripts/vocabulary.txt` / `scripts/static_vocabulary.txt`: the words recognised on **Hand → Speech**
  (moving / also held), chosen by `scripts/select_words.py`.

## 3. Build

```bat
set P=<path to GestureFlow-BSL-Translator-main>
.venv\Scripts\python %P%\scripts\make_candidates.py --meta meta\bsldict_meta.json --out %P%\scripts\wordlist_candidates.txt
.venv\Scripts\python %P%\scripts\download_subset.py --meta meta\bsldict_meta.json --words %P%\scripts\wordlist_candidates.txt --out raw_videos
.venv\Scripts\python %P%\scripts\process_clips.py --meta meta\bsldict_meta.json --raw raw_videos --clips %P%\media\signs --landmarks landmarks --models models
rem sign embedding: k-fold networks for honest word choice (4 network types; ~10 min each on a GPU)
.venv\Scripts\python %P%\scripts\train_embedding.py --landmarks landmarks --words %P%\scripts\wordlist_candidates.txt --folds 5 --out-dir emb_base
.venv\Scripts\python %P%\scripts\train_embedding.py --landmarks landmarks --words %P%\scripts\wordlist_candidates.txt --folds 5 --out-dir emb_reg --dropout 0.4 --weight-decay 0.01
.venv\Scripts\python %P%\scripts\train_embedding.py --landmarks landmarks --words %P%\scripts\wordlist_candidates.txt --folds 5 --out-dir emb_ep150 --epochs 150
.venv\Scripts\python %P%\scripts\train_embedding.py --landmarks landmarks --words %P%\scripts\wordlist_candidates.txt --folds 5 --out-dir emb_bigreg --hidden 512,256 --dropout 0.4 --weight-decay 0.01
.venv\Scripts\python %P%\scripts\select_words.py --landmarks landmarks --words %P%\scripts\wordlist_candidates.txt --skip %P%\scripts\select_skip.txt --oof emb_base\oof_embeddings.json,emb_reg\oof_embeddings.json,emb_ep150\oof_embeddings.json,emb_bigreg\oof_embeddings.json --target 45 --holdout --out-moving sel\vocabulary.txt --out-static sel\static_vocabulary.txt
rem copy sel\vocabulary.txt to %P%\scripts\vocabulary.txt; the held words are checked by hand (see static_vocabulary.txt)
rem final networks: the same 4 types with --folds 0 --exclude %P%\scripts\vocabulary.txt --out-dir final_<type>, then
.venv\Scripts\python %P%\scripts\merge_embeddings.py final_base\sign_embedding.json final_reg\sign_embedding.json final_ep150\sign_embedding.json final_bigreg\sign_embedding.json --out %P%\scripts\sign_embedding.json
.venv\Scripts\python %P%\scripts\build_index.py --meta meta\bsldict_meta.json --clips %P%\media\signs --landmarks landmarks --out %P%\demo\static\demo\data --vocab %P%\scripts\vocabulary.txt --static-vocab %P%\scripts\static_vocabulary.txt --embedding %P%\scripts\sign_embedding.json --emb-weight 2.0
.venv\Scripts\python %P%\scripts\build_synonyms.py --clips-index %P%\demo\static\demo\data\sign_clips.json --out %P%\demo\static\demo\data\synonyms.json --nltk-data nltk_data
```

Every step skips work that is already done, so re-running after adding words is quick.
Clip file names keep BSLDict's original names. Some borrow a related dictionary page's name
(e.g. `media/signs/hello/h_005_058_000_his.mp4`), but each clip is the video BSLDict lists for that
word (checked against its download link `.../hello.mp4`).
Some YouTube-hosted BSLDict videos have been deleted; those are listed in `raw_videos/failed.txt`.

Check the live behaviour after any change (no camera needed):
`node scripts/test_session.mjs landmarks`, `node scripts/test_calibration.mjs landmarks all` and `node scripts/test_held.mjs landmarks`.

## 4. Add your own recordings (recommended)

Put videos in one folder per word (BSLDict spelling), e.g. `recordings\doctor\take1.mp4`.
Use a plain background and good light, film waist-up, start and end with hands down, one sign per
video, 5–10 takes per sign from 2–5 signers. Then run:

```bat
.venv\Scripts\python %P%\scripts\add_recordings.py --recordings recordings --clips %P%\media\signs --landmarks landmarks --models models
```

Then re-run `build_index.py` (with `--vocab`) and `build_synonyms.py` from step 3. The script finds
where each sign starts and stops, saves the trimmed clip as `media/signs/<word>/own_*.mp4` (played
first on Speech → Sign) and adds the examples for Hand → Speech. Get written consent from everyone
recorded.

Optional: `scripts/tune_params.py --landmarks landmarks` re-tunes the matching weights.
Copy the printed values into `scripts/sign_features.py`, then re-run `build_index.py`.

## Keeping Python and JavaScript in step

`scripts/sign_features.py` (offline) and `demo/static/demo/sign_matcher.js` (browser) implement
the same features and matching. If you change one, change the other. The matching weights travel
inside `sign_templates.json` (`params`), so tuning them needs no JavaScript change.

## Licences and attribution

- **BSLDict**: CC BY 4.0 for the dataset. **Copyright in each video stays with its original
  owner** (the source site is recorded per video in the metadata). The clips are fine for the MVP
  demo. Get the owners' permission or replace them with your own recordings before charging customers.
- **WordNet**: Princeton WordNet licence. **MediaPipe** models (hand, pose, gesture recogniser) and library: Apache 2.0.
- **PyTorch** (BSD) and **wordfreq** (MIT code; word data CC BY-SA) are used only by the offline scripts. The sign-embedding
  networks are our own, trained on BSLDict landmarks.
- The site footer carries the attribution line.
