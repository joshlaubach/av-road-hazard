"""The visitor's session: an unguessable id the server hands out in a cookie.

The id is never taken from the request body or a header, only from the cookie the server set, and a
cookie that is not a well-formed server-issued id is replaced. The cookie is HttpOnly (page scripts
cannot read it) and SameSite=Lax. The web page and the API share one origin through the Next.js
rewrite, so no cross-site cookie is needed.
"""

from __future__ import annotations

from fastapi import Request, Response

from rag.index import SESSION_ID, new_session_id

COOKIE = "rh_session"
MAX_AGE_S = 7 * 24 * 3600


def session_for(request: Request, response: Response) -> str:
    """The caller's session id; a new one is issued (and set on the response) if there is none."""
    current = request.cookies.get(COOKIE, "")
    if SESSION_ID.fullmatch(current):
        return current
    fresh = new_session_id()
    response.set_cookie(
        COOKIE,
        fresh,
        max_age=MAX_AGE_S,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
    )
    return fresh


def client_address(request: Request) -> str:
    """The address of the direct peer. Forwarding headers are not trusted: anyone can send them."""
    return request.client.host if request.client else "unknown"
