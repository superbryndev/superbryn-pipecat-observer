"""Local stand-in for the SuperBryn API: catches the call report, the presign
request, the WAV upload and anything else the package sends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aiohttp import web


@dataclass
class Caught:
    events: list[dict[str, Any]] = field(default_factory=list)
    event_headers: list[dict[str, str]] = field(default_factory=list)
    presign_requests: list[dict[str, Any]] = field(default_factory=list)
    uploads: dict[str, bytes] = field(default_factory=dict)
    other: list[str] = field(default_factory=list)


class Catcher:
    """aiohttp app on 127.0.0.1 that answers like the real API."""

    REPORT_PATH = "/public-api/v1/calls/agent-data"

    def __init__(self, port: int = 0) -> None:
        self.caught = Caught()
        self._port = port
        self._runner: web.AppRunner | None = None
        self.base_url = ""

    async def start(self) -> None:
        app = web.Application(client_max_size=50 * 1024 * 1024)
        app.router.add_post(self.REPORT_PATH, self._report)
        app.router.add_post("/api/recording-upload-url", self._presign)
        app.router.add_put("/upload/{name}", self._upload)
        app.router.add_route("*", "/{tail:.*}", self._other)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", self._port)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        self.base_url = f"http://127.0.0.1:{port}"

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

    async def _report(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.caught.events.append(body)
        self.caught.event_headers.append(dict(request.headers))
        if body.get("surface") == "testing":
            return web.json_response({"surface": "testing", "status": "linked", "is_test": True})
        return web.json_response({"surface": body.get("surface"), "ok": True})

    async def _presign(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.caught.presign_requests.append(body)
        name = f"{body.get('session_id') or 'call'}.wav"
        url = f"{self.base_url}/upload/{name}"
        return web.json_response(
            {"upload_url": url, "object_key": f"recordings/{name}", "public_url": url}
        )

    async def _upload(self, request: web.Request) -> web.Response:
        self.caught.uploads[request.match_info["name"]] = await request.read()
        return web.Response(status=200)

    async def _other(self, request: web.Request) -> web.Response:
        self.caught.other.append(f"{request.method} {request.path}")
        return web.json_response({})
