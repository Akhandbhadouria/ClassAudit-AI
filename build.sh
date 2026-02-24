#!/usr/bin/env bash
# exit on error
set -o errexit

# ── System dependencies for dlib / face_recognition ──
apt-get update && apt-get install -y cmake build-essential libopenblas-dev liblapack-dev libx11-dev

# ── Install Python dependencies ──
pip install --upgrade pip
pip install -r requirements.txt

# ── Collect static files ──
python manage.py collectstatic --no-input

# ── Run database migrations ──
python manage.py migrate
