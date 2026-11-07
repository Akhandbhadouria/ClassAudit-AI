"""
ASGI config for facere project.

Upgraded to Django Channels for WebSocket support.
Handles both HTTP (via Django's standard ASGI app) and
WebSocket connections (via Channels routing).

Run with Daphne (dev):
    daphne facere.asgi:application

Or with Uvicorn (alternative):
    uvicorn facere.asgi:application --host 0.0.0.0 --port 8000
"""

import os

from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "facere.settings")

# Import WebSocket URL patterns AFTER setting the env var
from accounts.routing import websocket_urlpatterns  # noqa: E402

application = ProtocolTypeRouter(
    {
        # Standard Django HTTP requests
        "http": get_asgi_application(),

        # WebSocket requests — wrapped with AuthMiddlewareStack so
        # scope["user"] is populated from the session cookie automatically.
        "websocket": AuthMiddlewareStack(
            URLRouter(websocket_urlpatterns)
        ),
    }
)
