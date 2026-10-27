@echo off
title GestureFlow BSL Translator
if not exist venv py -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python manage.py migrate
start http://127.0.0.1:8000/
python manage.py runserver
pause
