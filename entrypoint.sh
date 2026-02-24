#!/usr/bin/env bash
set -o errexit

echo "Collecting static files..."
python manage.py collectstatic --no-input

echo "Running database migrations..."
python manage.py migrate --no-input

echo "Starting Gunicorn server..."
exec gunicorn facere.wsgi:application \
    --bind 0.0.0.0:${PORT:-10000} \
    --workers 3 \
    --timeout 120
