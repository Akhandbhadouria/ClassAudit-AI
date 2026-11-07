"""
accounts/routing.py
───────────────────
WebSocket URL patterns for the accounts app.

URL: ws://<host>/ws/attendance/
"""

from django.urls import re_path
from accounts import consumers

websocket_urlpatterns = [
    re_path(r"^ws/attendance/$", consumers.AttendanceConsumer.as_asgi()),
]

