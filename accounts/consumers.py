"""
accounts/consumers.py
─────────────────────
Django Channels WebSocket consumer for live class attendance monitoring.

Flow:
  Browser  ──ws:// connect──►  AttendanceConsumer.connect()
  Browser  ──{face_image}──►   AttendanceConsumer.receive()   (runs face verify)
  Browser  ◄──{status/duration}──  AttendanceConsumer.send()

One persistent TCP connection per teacher replaces the previous 5-second HTTP
polling loop.  Zero new TCP handshakes, zero CSRF overhead after handshake.
"""

import json
import base64
import logging
import datetime
import os

import cv2
import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.conf import settings

from accounts.utils.face_embedding import get_embedding, find_target_face_in_frame
from accounts.utils.liveness_detector import check_single_frame_liveness

logger = logging.getLogger("accounts")


# ──────────────────────────────────────────────────────────────────────────────
# Database helpers (must be wrapped for async context)
# ──────────────────────────────────────────────────────────────────────────────

@database_sync_to_async
def _get_ongoing_session(teacher):
    """Return the teacher's currently ongoing ClassSession, or None."""
    from accounts.models import ClassSession
    return ClassSession.objects.filter(teacher=teacher, status="Ongoing").first()


@database_sync_to_async
def _get_teacher(user):
    """Return the Teacher instance linked to this user."""
    return user.teacher


@database_sync_to_async
def _get_user_image(user):
    """Return the first UserImages record for this user, or None."""
    from accounts.models import UserImages
    return UserImages.objects.filter(user=user).first()


@database_sync_to_async
def _add_active_duration(session, seconds: int):
    """Atomically add `seconds` to the session's total_active_duration and save."""
    from accounts.models import ClassSession
    # Re-fetch to avoid stale state across async iterations
    session = ClassSession.objects.get(pk=session.pk)
    session.total_active_duration += datetime.timedelta(seconds=seconds)
    session.save(update_fields=["total_active_duration"])
    return session


# ──────────────────────────────────────────────────────────────────────────────
# Consumer
# ──────────────────────────────────────────────────────────────────────────────

class AttendanceConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer that handles live attendance verification for a teacher.

    Replaces the old HTTP POST to /teacher/update-live-attendance/.
    """

    # How many seconds to credit per verified frame.
    # Browser sends a frame every SEND_INTERVAL_MS ms (defined in the template).
    CREDIT_SECONDS = 5

    async def connect(self):
        """
        Authenticate the user on connect.
        Only allow authenticated teachers with an ongoing session.
        """
        user = self.scope.get("user")

        if user is None or not user.is_authenticated:
            logger.warning("WS attendance: unauthenticated connect rejected")
            await self.close(code=4001)
            return

        try:
            self.teacher = await _get_teacher(user)
        except Exception:
            logger.warning("WS attendance: user %s has no Teacher profile", user)
            await self.close(code=4002)
            return

        self.user = user
        self._load_stored_embedding()   # cache embedding on connect
        await self.accept()
        logger.info("WS attendance: teacher %s connected", user.username)

        # Send an initial acknowledgement so the browser knows the socket is ready
        await self.send(text_data=json.dumps({
            "status": "connected",
            "message": "WebSocket connected. Monitoring started.",
        }))

    def _load_stored_embedding(self):
        """
        Load the teacher's stored 512-D ONNX face embedding from disk into memory.
        Called once on connect so disk I/O doesn't happen on every frame.
        """
        self._stored_emb = None
        user_dir = os.path.join(settings.BASE_DIR, "data", "users", self.user.username)
        embedding_path = os.path.join(user_dir, "embeddings.npy")
        if os.path.exists(embedding_path):
            self._stored_emb = np.load(embedding_path)
            logger.debug("WS attendance: loaded embedding for %s", self.user.username)
        else:
            logger.warning("WS attendance: no embedding file for %s", self.user.username)

    async def disconnect(self, close_code):
        logger.info(
            "WS attendance: teacher %s disconnected (code=%s)",
            getattr(self, "user", "?"),
            close_code,
        )

    async def receive(self, text_data=None, bytes_data=None):
        """
        Process one frame from the browser.

        Expected JSON payload:
            { "face_image": "data:image/jpeg;base64,<...>" }

        Responds with:
            { "status": "success"|"warning"|"error",
              "message": "...",
              "duration": "5m 30s"   (only on success)
            }
        """
        try:
            payload = json.loads(text_data or "{}")
            face_image_data = payload.get("face_image", "")

            if not face_image_data:
                await self._send_response("error", "No image data received.")
                return

            # ── Decode base64 frame ──
            try:
                face_data_str = face_image_data.split(",")[1]
                image_data = base64.b64decode(face_data_str)
            except Exception:
                await self._send_response("error", "Invalid image encoding.")
                return

            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                await self._send_response("error", "Could not decode image.")
                return

            # ── Liveness check ──
            is_live, live_detail = check_single_frame_liveness(frame)
            if not is_live:
                await self._send_response("warning", f"Active presence check: {live_detail}")
                return

            # ── Embedding check ──
            if self._stored_emb is None:
                await self._send_response(
                    "error", "Registration data missing. Please re-register."
                )
                return

            # ── Multi-face presence resolution ──
            is_matched, _box, total_faces = find_target_face_in_frame(frame, self._stored_emb)

            if is_matched:
                session = await _get_ongoing_session(self.teacher)
                if session is None:
                    await self._send_response("error", "No ongoing session.")
                    return

                session = await _add_active_duration(session, self.CREDIT_SECONDS)

                total_secs = int(session.total_active_duration.total_seconds())
                duration_str = f"{total_secs // 60}m {total_secs % 60}s"

                message = "Authorized"
                if total_faces > 1:
                    message = f"Authorized (Teacher identified among {total_faces} people)"

                await self._send_response("success", message, duration=duration_str,
                                          total_faces=total_faces)

            else:
                if total_faces > 0:
                    await self._send_response(
                        "warning",
                        f"{total_faces} face(s) visible, but teacher not identified.",
                    )
                else:
                    await self._send_response("warning", "No face detected in camera.")

        except Exception as exc:
            logger.error("WS attendance receive error: %s", exc, exc_info=True)
            await self._send_response("error", str(exc))

    # ── Helper ──────────────────────────────────────────────────────────────

    async def _send_response(self, status: str, message: str, **extra):
        payload = {"status": status, "message": message, **extra}
        await self.send(text_data=json.dumps(payload))
