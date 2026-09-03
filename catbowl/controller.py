"""Per-bowl state machine.

Deliberately free of cameras, models and hardware: it takes an observation and
a timestamp and returns nothing but side effects on an injected actuator. That
makes every timing rule in here testable in milliseconds instead of minutes.

    CLOSED --(owner confirmed)--> OPEN --(empty / intruder / timeout)--> COOLDOWN
      ^                                                                    |
      +--------------------(cooldown elapsed)------------------------------+
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from typing import Callable

from .actuators import Actuator
from .config import BowlConfig
from .events import Event
from .recognizer import VoteTracker

log = logging.getLogger(__name__)

# How long the bowl must look empty before a stale vote window is discarded.
VOTE_DECAY_S = 1.5
# Do not spam the log with 'denied' for a cat that just sits there.
DENY_REPEAT_S = 30.0


class BowlState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    COOLDOWN = "cooldown"


class BowlController:
    def __init__(
        self,
        cfg: BowlConfig,
        actuator: Actuator,
        vote_window: int = 6,
        votes_required: int = 4,
        clock: Callable[[], float] = time.monotonic,
        on_event: Callable[[Event], None] | None = None,
    ):
        self.cfg = cfg
        self.actuator = actuator
        self.clock = clock
        self.on_event = on_event or (lambda event: None)
        self.votes = VoteTracker(vote_window, votes_required)

        self.state = BowlState.CLOSED
        self.last_seen = 0.0
        self.last_decision: str | None = None
        self.last_confidence = 0.0
        self._owner_since: float | None = None
        self._intruder_since: float | None = None
        self._opened_at = 0.0
        self._cooldown_until = 0.0
        self._denied_at: dict[str, float] = {}
        # Set when a sitting ended on the clock rather than because the cat
        # left. It blocks the next open until the bowl has been empty for
        # policy.rearm_absent_s, so the cat has to step away and come back.
        self._needs_arrival = False
        self._manual: str | None = None
        # observe() runs on the bowl's worker thread; set_manual() is called
        # from the status server's. Both mutate the same state machine.
        self._lock = threading.RLock()
        self.stats = {"opens": 0, "denials": 0, "seconds_open": 0.0}

    # -- public API --------------------------------------------------------- #

    @property
    def cat(self) -> str:
        return self.cfg.cat

    def observe(self, present: bool, label: str | None = None, confidence: float = 0.0) -> None:
        """Feed one frame's worth of evidence into the machine."""
        with self._lock:
            self._observe(present, label, confidence)

    def _observe(self, present: bool, label: str | None, confidence: float) -> None:
        now = self.clock()
        if present:
            self.last_seen = now
            if label:
                self.votes.update(label)
                self.last_confidence = confidence
            # The confirmation timer runs alongside the vote window rather than
            # after it, so the two delays overlap instead of stacking.
            if label == self.cat and self._owner_since is None:
                self._owner_since = now
        elif now - self.last_seen > VOTE_DECAY_S:
            self.votes.clear()
            self._owner_since = None

        self.last_decision = self.votes.decision()

        if (
            self._needs_arrival
            and not present
            and now - self.last_seen >= self.cfg.policy.rearm_absent_s
        ):
            self._needs_arrival = False
            log.info("%s: bowl clear, ready to open for %s again", self.cfg.id, self.cat)
            self._emit("rearmed", cat=self.cat)

        # A manual hold outranks the state machine. Evidence above is still
        # collected so the status page keeps showing what the camera sees, but
        # no transition fires: the lid stays where a human put it.
        if self._manual is not None:
            return

        if self.state is BowlState.COOLDOWN:
            self._tick_cooldown(now)
        elif self.state is BowlState.CLOSED:
            self._tick_closed(now, present)
        else:
            self._tick_open(now, present)

    def set_manual(self, mode: str | None) -> None:
        """Pin the lid open or closed by hand, or hand control back.

        ``mode`` is "open", "closed", or None to resume automatic control.
        Resuming drops into the normal cooldown rather than straight to CLOSED,
        so a cat still standing at the bowl cannot re-open it instantly.
        """
        if mode not in (None, "open", "closed"):
            raise ValueError(f"manual mode must be open/closed/None, got {mode!r}")
        with self._lock:
            self._set_manual(mode)

    def _set_manual(self, mode: str | None) -> None:
        now = self.clock()
        self._manual = mode
        self._intruder_since = None
        # A human moving the lid outranks a pending time-limit lockout too.
        self._needs_arrival = False

        if mode == "open":
            if self.state is not BowlState.OPEN:
                self._opened_at = now
                self.stats["opens"] += 1
            self.state = BowlState.OPEN
            self.actuator.open()
        elif mode == "closed":
            if self.state is BowlState.OPEN:
                duration = round(now - self._opened_at, 1)
                self.stats["seconds_open"] = round(self.stats["seconds_open"] + duration, 1)
            self.state = BowlState.CLOSED
            self.actuator.close()
        else:
            if self.state is BowlState.OPEN:
                duration = round(now - self._opened_at, 1)
                self.stats["seconds_open"] = round(self.stats["seconds_open"] + duration, 1)
                self.actuator.close()
            self.state = BowlState.COOLDOWN
            self._cooldown_until = now + self.cfg.policy.cooldown_s
            self.votes.clear()
            self._owner_since = None

        self._emit("manual", cat=self.cfg.cat, detail={"lid": mode or "auto"})

    @property
    def manual(self) -> str | None:
        return self._manual

    def force_close(self, reason: str = "shutdown") -> None:
        if self.state is BowlState.OPEN:
            self._close(self.clock(), reason)
        else:
            self.actuator.close()

    def status(self) -> dict:
        now = self.clock()
        return {
            "bowl": self.cfg.id,
            "cat": self.cfg.cat,
            "state": self.state.value,
            "manual": self._manual,
            "waiting_for_rearm": self._needs_arrival,
            "lid": round(self.actuator.position, 2),
            "seen": self.last_decision or "-",
            "confidence": round(self.last_confidence, 3),
            "votes": self.votes.tally(),
            "open_for_s": round(now - self._opened_at, 1) if self.state is BowlState.OPEN else 0.0,
            "since_seen_s": round(now - self.last_seen, 1) if self.last_seen else None,
            **self.stats,
        }

    # -- states ------------------------------------------------------------- #

    def _tick_cooldown(self, now: float) -> None:
        if now >= self._cooldown_until:
            self.state = BowlState.CLOSED
            self.votes.clear()
            self._owner_since = None

    def _tick_closed(self, now: float, present: bool) -> None:
        # Still waiting for the cat to step back from the last sitting. Evidence
        # keeps flowing so the status page stays honest, but nothing opens.
        if self._needs_arrival:
            return

        winner = self.last_decision
        if winner == self.cat and present and self._owner_since is not None:
            if now - self._owner_since >= self.cfg.policy.open_confirm_s:
                self._open(now)
            return

        if winner and winner != self.cat and present:
            last = self._denied_at.get(winner, 0.0)
            if now - last >= DENY_REPEAT_S:
                self._denied_at[winner] = now
                self.stats["denials"] += 1
                self._emit("denied", cat=winner, detail={"owner": self.cat})

    def _tick_open(self, now: float, present: bool) -> None:
        winner = self.last_decision

        if self.cfg.policy.close_on_intruder and present and winner and winner != self.cat:
            if self._intruder_since is None:
                self._intruder_since = now
            if now - self._intruder_since >= self.cfg.policy.intruder_grace_s:
                self._close(now, "intruder", extra={"intruder": winner})
                return
        else:
            self._intruder_since = None

        if not present and now - self.last_seen >= self.cfg.policy.close_delay_s:
            self._close(now, "left")
            return

        if self.cfg.policy.max_open_s and now - self._opened_at >= self.cfg.policy.max_open_s:
            # The cat is very likely still at the bowl, so this close has to
            # latch: otherwise the cooldown expires under its nose and it eats
            # straight through the limit.
            self._needs_arrival = True
            self._close(now, "max_open_s")

    # -- transitions -------------------------------------------------------- #

    def _open(self, now: float) -> None:
        self.state = BowlState.OPEN
        self._opened_at = now
        self._intruder_since = None
        self._owner_since = None
        self.stats["opens"] += 1
        self._emit("opened", cat=self.cat, detail={"confidence": round(self.last_confidence, 3)})
        self.actuator.open()

    def _close(self, now: float, reason: str, extra: dict | None = None) -> None:
        duration = round(now - self._opened_at, 1)
        self.stats["seconds_open"] = round(self.stats["seconds_open"] + duration, 1)
        self.state = BowlState.COOLDOWN
        self._cooldown_until = now + self.cfg.policy.cooldown_s
        self._intruder_since = None
        self.actuator.close()
        self._emit("closed", cat=self.cat, detail={"reason": reason, "duration_s": duration, **(extra or {})})

    def _emit(self, kind: str, cat: str = "", detail: dict | None = None) -> None:
        try:
            self.on_event(Event(kind=kind, bowl=self.cfg.id, cat=cat, detail=detail or {}))
        except Exception:  # pragma: no cover - a broken sink must not stop the lid
            log.exception("event sink raised for %s/%s", self.cfg.id, kind)
