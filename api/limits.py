"""A cap on the size of a whole request, checked before the body is read or parsed.

The per-file checks in api/main.py run after the framework has already received and parsed the
whole body, so a huge request would still be read in full first. This sits in front of everything:
a request that announces a body over its limit is refused from the header alone, and one that does
not announce a size (chunked) is refused as soon as it has sent more than the limit.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]

MULTIPART_OVERHEAD = 1_048_576  # boundaries and field headers on top of the files themselves
SMALL_BODY = 65_536  # any route that takes no upload (health, records) or a short JSON body
JSON_OVERHEAD = 65_536  # the JSON around a base64 image in /query
UPLOAD_PATHS = ("/analyze", "/analyze/trace")

log = logging.getLogger("api")


def body_limit(path: str, api_cfg: dict[str, Any]) -> int:
    """The most bytes a request to this path may carry, from the upload cap and the batch size."""
    upload = int(api_cfg["max_upload_bytes"])
    if path in UPLOAD_PATHS:
        return upload + MULTIPART_OVERHEAD
    if path == "/index":
        return int(api_cfg["max_batch_files"]) * upload + MULTIPART_OVERHEAD
    if path == "/query":
        base64_image = 4 * ((upload + 2) // 3)  # an image of `upload` bytes, base64-encoded
        return base64_image + JSON_OVERHEAD
    return SMALL_BODY


class BodyLimitMiddleware:
    """Refuse requests whose body is over the limit for their path with a 413."""

    def __init__(
        self, app: Callable[[Scope, Receive, Send], Awaitable[None]], api_cfg: dict[str, Any]
    ):
        self.app = app
        self.api_cfg = api_cfg

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = body_limit(scope["path"], self.api_cfg)
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and not _within(declared, limit):
            await self._refuse(scope, send)
            return

        seen = 0
        refused = False

        async def counted() -> Message:
            # A body with no declared size is counted as it arrives. Past the limit we answer 413
            # at once and tell the app the client is gone, so it stops reading. (Raising instead
            # would be turned into a 400 by the framework.)
            nonlocal seen, refused
            message = await receive()
            if message["type"] == "http.request" and not refused:
                seen += len(message.get("body", b""))
                if seen > limit:
                    refused = True
                    await self._refuse(scope, send)
                    return {"type": "http.disconnect"}
            return message

        async def watched(message: Message) -> None:
            if not refused:  # the app may still try to answer a request we already refused
                await send(message)

        await self.app(scope, counted, watched)

    async def _refuse(self, scope: Scope, send: Send) -> None:
        log.warning("request body over its limit", extra={"route": scope["path"], "status": 413})
        body = json.dumps({"detail": "request body too large"}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"connection", b"close"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def _within(declared: bytes, limit: int) -> bool:
    """True when the Content-Length header is a valid size no larger than the limit."""
    try:
        return 0 <= int(declared) <= limit
    except ValueError:
        return False
