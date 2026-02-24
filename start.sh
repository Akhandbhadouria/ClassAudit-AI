#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset

# Collect static files
python manage.py collectstatic --noinput

# Run database migrations (optional: only if you want Render to auto-migrate)
python manage.py migrate --noinput

# Start Gunicorn server
exec gunicorn facere.wsgi:application --bind 0.0.0.0:8000
