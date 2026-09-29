"""Optional phone notifications through ntfy.sh.

Uses ntfy's JSON publish API, so titles may contain emoji (the previous driver
sent them as HTTP headers, which cannot carry non-latin-1 text, and its final
"Race Complete" message never arrived). Pass the topic on the command line or
in the NTFY_TOPIC environment variable; never commit it, since anyone who knows a
topic can read and post to it.
"""

from __future__ import annotations

import json
import urllib.request


class Notifier:
    def __init__(self, topic: str | None, server: str = "https://ntfy.sh"):
        self.topic = topic
        self.server = server.rstrip("/")

    def __call__(
        self, message: str, title: str | None = None, priority: int = 3, tags: list[str] | None = None
    ) -> None:
        if not self.topic:
            return
        body = {"topic": self.topic, "message": message, "priority": priority}
        if title:
            body["title"] = title
        if tags:
            body["tags"] = tags
        req = urllib.request.Request(
            self.server, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            urllib.request.urlopen(req, timeout=5).close()
        except Exception as e:  # a notification failure must never kill a run
            print(f"  (ntfy failed: {e})")
