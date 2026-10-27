
BSL Translator — Full Stack (Django + Frontend, Hand→Speech & Speech→Sign) — 2025-11-08

## Features
- Hand→Speech: MediaPipe hand + body landmarks and Google's gesture recogniser (tasks-vision 1.1.1). Signs are
  matched against real BSL dictionary clips (BSLDict) by comparing hand movement, helped by a small neural
  network (sign embedding) that runs in the browser; no video leaves the device. Recognises 45 BSL words
  (scripts/vocabulary.txt, listed on the page under "Signs I can recognise") + 9 demo gestures. Every
  recognised sign is spoken in a British English voice; sign sequences can form full sentences
  (unified_patterns.js). Confidence is measured, not fixed. Optional "Calibrate my signs": sign each word
  3 times; the recordings stay in the user's browser.
- Speech→Sign: British English speech recognition -> BSL video clips (1,162 words) played in order, word by
  word, with phrases, word roots and synonyms. Words without a clip are fingerspelled (a–z). No emoji.
- Auth: login, signup, logout (Django).
- Subscription + Contact (DB-backed). Migrations included.

## Setup
1) Create venv & install:
   python -m venv venv
   venv\Scripts\activate   (Windows)
   pip install -r requirements.txt   (Django is enough to run)

2) Init DB:
   python manage.py migrate

   An admin account is auto-created on first migrate by signals:
   - username: admin
   - password: admin123

3) Run:
   python manage.py runserver
   Open http://127.0.0.1:8000/

   Tabs:
   - Hand→Speech at "/"
   - Speech→Sign at "/speech-to-sign/"
   - Auth: /login, /signup
   - Subscription: /subscription (requires login)
   - Contact: /contact

## Notes
- Hand→Speech: demo/static/demo/script.js (page), sign_session.js (splits signs), sign_matcher.js (matching),
  gesture_rules.js (demo gestures Three/Four), calibration.js (personal calibration)
- Speech→Sign: demo/static/demo/speech_app.js (page/player), speech_lookup.js (word lookup)
- Sign data (clips, templates, synonyms) and how to rebuild/extend it: README_DATA.md
- Use Chrome or Edge (speech recognition is not available in Firefox).

Enjoy!
