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
from .rations import Ledger
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

        self.ledger = Ledger(cfg.rations)

        self.state = BowlState.CLOSED
        self.last_seen = 0.0
        self.last_decision: str | None = None
        self.last_confidence = 0.0
        # When each cat this bowl feeds was first seen in the current approach.
        self._seen_since: dict[str, float] = {}
        # The cat the lid is currently open for, and when its meal was last
        # charged to its allowance.
        self._feeding: str | None = None
        self._charged_at = 0.0
        self._intruder_since: float | None = None
        self._opened_at = 0.0
        self._cooldown_until = 0.0
        self._denied_at: dict[str, float] = {}
        self._manual: str | None = None
        # When "closed" was last sent to the lid. The app closes every lid
        # before building its controller, so construction counts as one.
        self._closed_sent_at = clock()
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
            # after it, so the two delays overlap instead of stacking. One
            # timer per cat the bowl feeds: which of them is at the bowl is not
            # settled yet, and whichever it turns out to be should not have to
            # start its clock again.
            if label in self.cfg.cats and label not in self._seen_since:
                self._seen_since[label] = now
        elif now - self.last_seen > VOTE_DECAY_S:
            self.votes.clear()
            self._seen_since.clear()

        self.last_decision = self.votes.decision()

        # A manual hold outranks the state machine. Evidence above is still
        # collected so the status page keeps showing what the camera sees, but
        # no transition fires: the lid stays where a human put it.
        if self._manual is not None:
            if self._manual == "closed":
                self._keep_closed(now)
            return

        if self.state is BowlState.COOLDOWN:
            self._tick_cooldown(now)
        elif self.state is BowlState.CLOSED:
            self._tick_closed(now, present)
        else:
            self._tick_open(now, present)

        if self.state is not BowlState.OPEN:
            self._keep_closed(now)

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
            self._send_close(now)
        else:
            if self.state is BowlState.OPEN:
                duration = round(now - self._opened_at, 1)
                self.stats["seconds_open"] = round(self.stats["seconds_open"] + duration, 1)
                self._send_close(now)
            self.state = BowlState.COOLDOWN
            self._cooldown_until = now + self.cfg.policy.cooldown_s
            self.votes.clear()
            self._seen_since.clear()

        self._emit("manual", cat=self._feeding or self.cfg.cat, detail={"lid": mode or "auto"})

    @property
    def manual(self) -> str | None:
        return self._manual

    def force_close(self, reason: str = "shutdown") -> None:
        if self.state is BowlState.OPEN:
            self._close(self.clock(), reason)
        else:
            self._send_close(self.clock())

    def status(self) -> dict:
        now = self.clock()
        return {
            "bowl": self.cfg.id,
            # `cat` is who the bowl is open for when it is open, and the list
            # of who it serves otherwise - the status page shows one name per
            # bowl, and while a lid is up that name should be the cat eating.
            "cat": self._feeding or ", ".join(self.cfg.cats),
            "cats": list(self.cfg.cats),
            "feeding": self._feeding,
            "rations": self.ledger.status(self.cfg.cats, now),
            "state": self.state.value,
            "manual": self._manual,
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
            self._seen_since.clear()

    def _candidate(self, now: float) -> str | None:
        """The cat this bowl would open for right now, if any.

        Opening asks less than every other transition: policy.open_votes
        sightings, not a consensus. See PolicyConfig.open_votes for why. A cat
        must still be the best represented in the window - one frame of J does
        not open the bowl while K is standing in front of it - and it must have
        allowance left.
        """
        seen = [cat for cat in self.cfg.cats if cat in self._seen_since]
        if not seen:
            return None
        # Most-voted first, so the cat actually at the bowl is considered
        # before one glimpsed behind it.
        seen.sort(key=lambda cat: (self.votes.count(cat), -self._seen_since[cat]), reverse=True)
        for cat in seen:
            if (self.votes.count(cat) >= self.cfg.policy.open_votes
                    and self.votes.leader(cat)
                    and now - self._seen_since[cat] >= self.cfg.policy.open_confirm_s):
                return cat
        return None

    def _tick_closed(self, now: float, present: bool) -> None:
        winner = self.last_decision
        if present:
            candidate = self._candidate(now)
            if candidate is not None and self.ledger.allows(candidate, now):
                self._open(now, candidate)
                return
            if candidate is not None:
                # Recognised, allowed here, but out of allowance for now.
                self._deny(now, candidate, "out of ration")
                return

        if winner and winner not in self.cfg.cats and present:
            self._deny(now, winner, "not this bowl's cat")

    def _deny(self, now: float, cat: str, reason: str) -> None:
        """Log a refusal, at most once every DENY_REPEAT_S per cat.

        A cat that settles down in front of a bowl it cannot open would
        otherwise fill the log with one line per frame.
        """
        last = self._denied_at.get(cat, 0.0)
        if now - last < DENY_REPEAT_S:
            return
        self._denied_at[cat] = now
        self.stats["denials"] += 1
        detail = {"reason": reason}
        refill = self.ledger.refills_at(cat, now)
        if refill is not None:
            detail["retry_in_s"] = round(refill - now, 1)
        self._emit("denied", cat=cat, detail=detail)

    def _tick_open(self, now: float, present: bool) -> None:
        winner = self.last_decision
        self._charge(now)

        if self.cfg.policy.close_on_intruder and present and winner and winner != self._feeding:
            # Any cat but the one being fed, including another this bowl
            # serves: they get their own turn, with their own allowance.
            if self._intruder_since is None:
                self._intruder_since = now
            if now - self._intruder_since >= self.cfg.policy.intruder_grace_s:
                self._close(now, "intruder", extra={"intruder": winner})
                return
        else:
            self._intruder_since = None

        if self._feeding and not self.ledger.allows(self._feeding, now):
            self._close(now, "ration")
            return

        if not present and now - self.last_seen >= self.cfg.policy.close_delay_s:
            self._close(now, "left")
            return

        if (self.cfg.policy.max_open_s and self._feeding not in self.cfg.uncapped
                and now - self._opened_at >= self.cfg.policy.max_open_s):
            self._close(now, "max_open_s")

    def _charge(self, now: float) -> None:
        """Bill the open lid to whoever it is open for, up to this moment.

        Charged as the meal happens rather than at the end: a cat that never
        leaves would otherwise never be billed, and an allowance that expires
        in small pieces comes back smoothly an hour later.
        """
        if self._feeding is None:
            return
        self.ledger.spend(self._feeding, now, max(0.0, now - self._charged_at))
        self._charged_at = now

    # -- transitions -------------------------------------------------------- #

    def _open(self, now: float, cat: str) -> None:
        self.state = BowlState.OPEN
        self._opened_at = now
        self._intruder_since = None
        self._seen_since.clear()
        self._feeding = cat
        self._charged_at = now
        self.stats["opens"] += 1
        detail = {"confidence": round(self.last_confidence, 3)}
        remaining = self.ledger.remaining(cat, now)
        if self.ledger.limited(cat):
            detail["ration_left_s"] = round(remaining, 1)
        self._emit("opened", cat=cat, detail=detail)
        self.actuator.open()

    def _close(self, now: float, reason: str, extra: dict | None = None) -> None:
        self._charge(now)
        fed = self._feeding
        duration = round(now - self._opened_at, 1)
        self.stats["seconds_open"] = round(self.stats["seconds_open"] + duration, 1)
        self._feeding = None
        self.state = BowlState.COOLDOWN
        self._cooldown_until = now + self.cfg.policy.cooldown_s
        self._intruder_since = None
        self._send_close(now)
        self._emit("closed", cat=fed or self.cfg.cat,
                   detail={"reason": reason, "duration_s": duration, **(extra or {})})

    def _send_close(self, now: float) -> None:
        self._closed_sent_at = now
        self.actuator.close()

    def _keep_closed(self, now: float) -> None:
        """Re-send "closed" every reassert_closed_s while the lid should be down.

        A servo that goes limp between moves can be pawed open with the bowl
        shut, and nothing else would notice until the next meal. The servo
        knows its absolute angle, so re-sending the closed angle puts the lid
        back wherever it was pushed to. A servo that is still holding is
        already there and the actuator skips the write.

        The return is not slewed: the software never learns where the lid was
        pushed to, so it cannot ramp from there, and the servo returns at its
        own full speed.
        """
        every = self.cfg.policy.reassert_closed_s
        if every and now - self._closed_sent_at >= every:
            self._send_close(now)

    def _emit(self, kind: str, cat: str = "", detail: dict | None = None) -> None:
        try:
            self.on_event(Event(kind=kind, bowl=self.cfg.id, cat=cat, detail=detail or {}))
        except Exception:  # pragma: no cover - a broken sink must not stop the lid
            log.exception("event sink raised for %s/%s", self.cfg.id, kind)
