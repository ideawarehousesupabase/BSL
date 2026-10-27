GESTUREFLOW BSL TRANSLATOR — FINAL DEMO

INCLUDED
- Hand -> Speech camera detection and sentence builder
- Speech -> Sign on a separate tab with BSL video clips and fingerspelling
- Login, Signup, Logout
- SQLite database authentication
- Subscription page and history
- Contact Us form saved to database
- Django admin
- Reset Last / Reset All
- BSL signs from the BSLDict dictionary (list shown under "Signs I can recognise") plus demo gestures:
  Hello (wave), Stop, Peace, Help, Good job, No, Love, Three, Four

EASIEST WAY TO RUN ON WINDOWS
1. Extract GestureFlow_BSL_Final.zip.
2. Double-click RUN_WINDOWS.bat.
3. Allow installation and migrations to finish.
4. The browser should open http://127.0.0.1:8000/
5. In Chrome/Edge, allow camera and microphone permissions.

MANUAL METHOD
Open Command Prompt inside the extracted folder and run:

py -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver

Then open http://127.0.0.1:8000/

PAGES
Hand -> Speech: / 
Speech -> Sign: /speech-to-sign/
Signup: /signup/
Login: /login/
Subscription: /subscription/
Contact: /contact/
Admin: /admin/

DEMO ADMIN
Username: admin
Password: admin123

STOP SERVER
Press Ctrl+C in the Command Prompt window.

IMPORTANT
TensorFlow is NOT in requirements.txt. It is not required to run this version and was the reason the previous Windows installation failed. This demo uses MediaPipe hand/body landmarks and matches them against recorded BSL dictionary clips (no AI training). Recognition covers a limited vocabulary; see demo/static/demo/data/eval_report.json for measured accuracy and README_DATA.md to rebuild or extend the sign data.
