from __future__ import annotations

from fastapi import Request

from api.session import client_address


def request(client: tuple[str, int] | None, forwarded: str | None = None) -> Request:
    headers = [(b"x-forwarded-for", forwarded.encode())] if forwarded else []
    return Request({"type": "http", "headers": headers, "client": client})


def test_the_address_is_the_peer_and_the_code_never_reads_the_forwarding_header() -> None:
    assert client_address(request(("10.0.0.5", 4000), forwarded="203.0.113.9")) == "10.0.0.5"


def test_no_peer_is_one_shared_unknown_address() -> None:
    assert client_address(request(None, forwarded="203.0.113.9")) == "unknown"
