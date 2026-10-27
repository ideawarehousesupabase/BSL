
BSL Translator — Full Stack (Django + Frontend, Hand→Speech & Speech→Sign) — 2025-11-08

## Features
- Hand→Speech: MediaPipe hand + body landmarks (tasks-vision 1.1.1). Signs are matched against real BSL
  dictionary clips (BSLDict) by comparing hand movement. No AI model is trained. Every recognised sign is spoken
  in a British English voice; sign sequences can form full sentences (unified_patterns.js). The original
  demo gestures still work as a fallback. Confidence is measured, not fixed.
  Recognises a fixed set of 44 signs (scripts/vocabulary.txt + 9 demo gestures). For best accuracy use
  "Calibrate my signs": sign each word 3 times; the recordings stay in the user's browser.
- Speech→Sign: British English speech recognition -> BSL video clips played in order, word by word, with
  phrases, word roots and synonyms. Words without a clip are fingerspelled (a–z). No emoji.
- Auth: login, signup, logout (Django).
- Subscription + Contact (DB-backed). Migrations included.
- Optional ML training path (TFJS) if you want to replace rule-based logic later.

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
- Hand→Speech: demo/static/demo/script.js (page), sign_matcher.js (matching), gesture_rules.js (demo gestures)
- Speech→Sign: demo/static/demo/speech_app.js (page/player), speech_lookup.js (word lookup)
- Sign data (clips, templates, synonyms) and how to rebuild/extend it: README_DATA.md
- Use Chrome or Edge (speech recognition is not available in Firefox).

## Optional: ML Training (local)
- Use a separate data capture tool (not included in Django app here) or repurpose front-end landmarks capture.
- Convert to TFJS and drop files into demo/static/demo/model_tfjs/

Enjoy!
