"""Event log: who ate, when, for how long, and who got turned away.

One JSON object per line, one file per day. Useful for spotting a cat who has
stopped eating, and for tuning thresholds after the fact.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class Event:
    kind: str                 # opened | closed | denied | error | startup | shutdown
    bowl: str
    cat: str = ""
    timestamp: float = field(default_factory=time.time)
    detail: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> str:
        payload = asdict(self)
        payload["time"] = datetime.fromtimestamp(self.timestamp).isoformat(timespec="seconds")
        return json.dumps(payload, sort_keys=True)


class EventLog:
    def __init__(self, directory: str | Path | None):
        self.directory = Path(directory) if directory else None
        self._lock = threading.Lock()
        self.recent: list[Event] = []
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)

    def __call__(self, event: Event) -> None:
        self.write(event)

    def write(self, event: Event) -> None:
        with self._lock:
            self.recent.append(event)
            del self.recent[:-200]
            if self.directory:
                path = self.directory / f"events-{datetime.fromtimestamp(event.timestamp):%Y-%m-%d}.jsonl"
                try:
                    with path.open("a") as fh:
                        fh.write(event.as_json() + "\n")
                except OSError:
                    log.exception("could not append to %s", path)
        detail = " ".join(f"{k}={v}" for k, v in event.detail.items())
        log.info("[%s] %s %s %s", event.bowl, event.kind, event.cat, detail)
