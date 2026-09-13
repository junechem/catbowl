"""How much open lid each cat has left.

One bowl shared by cats with different rules needs somewhere to keep "J has had
ninety of its hundred and twenty seconds this hour". That is all this is: a
rolling sum of open-lid time, per cat.

Time is charged in the small increments the control loop runs at rather than in
one lump when the lid shuts, for two reasons. A meal that is still in progress
has already been eaten, so it should already count - otherwise a cat that never
leaves is never billed. And an allowance that expires in small pieces comes back
smoothly: an hour after a two-minute meal the budget refills over two minutes,
not all at once at the end.

Nothing here is written to disk. A restart hands every cat a fresh allowance,
which errs towards feeding a cat twice rather than starving one that has eaten
nothing - the right way round for a machine that stands between an animal and
its dinner.
"""

from __future__ import annotations

from collections import deque

from .config import RationConfig

# A cat with no ration configured gets this much: all of it.
UNLIMITED = float("inf")


class Ledger:
    """Per-cat rolling budgets of open-lid seconds."""

    def __init__(self, rations: dict[str, RationConfig] | None = None):
        self.rations = dict(rations or {})
        # cat -> (when it was spent, how many seconds), oldest first.
        self._spent: dict[str, deque[tuple[float, float]]] = {}

    def limited(self, cat: str) -> bool:
        return cat in self.rations

    def spend(self, cat: str, now: float, seconds: float) -> None:
        """Charge *seconds* of open lid to *cat*."""
        if seconds <= 0 or cat not in self.rations:
            return
        self._spent.setdefault(cat, deque()).append((now, seconds))

    def spent(self, cat: str, now: float) -> float:
        """Seconds charged to *cat* inside its window, ending now."""
        ration = self.rations.get(cat)
        if ration is None:
            return 0.0
        entries = self._spent.get(cat)
        if not entries:
            return 0.0
        cutoff = now - ration.per_s
        while entries and entries[0][0] <= cutoff:
            entries.popleft()
        return sum(seconds for _, seconds in entries)

    def remaining(self, cat: str, now: float) -> float:
        """Open-lid seconds *cat* may still have. Infinite without a ration."""
        ration = self.rations.get(cat)
        if ration is None:
            return UNLIMITED
        return max(0.0, ration.seconds - self.spent(cat, now))

    def allows(self, cat: str, now: float) -> bool:
        return self.remaining(cat, now) > 0.0

    def refills_at(self, cat: str, now: float) -> float | None:
        """When *cat* will next have any allowance at all, or None if it has some.

        The oldest charge in the window is the first to expire, so that is the
        moment an exhausted cat gets a sliver back. Used only to tell a human
        when the bowl will open again.
        """
        if self.allows(cat, now):
            return None
        entries = self._spent.get(cat)
        ration = self.rations.get(cat)
        if not entries or ration is None:
            return None
        return entries[0][0] + ration.per_s

    def status(self, cats: list[str], now: float) -> dict[str, dict]:
        """What each cat has left, for the status page."""
        out = {}
        for cat in cats:
            ration = self.rations.get(cat)
            if ration is None:
                out[cat] = {"limited": False}
                continue
            remaining = self.remaining(cat, now)
            entry = {"limited": True,
                     "remaining_s": round(remaining, 1),
                     "allowance_s": ration.seconds,
                     "per_s": ration.per_s}
            refill = self.refills_at(cat, now)
            if refill is not None:
                entry["refills_in_s"] = round(refill - now, 1)
            out[cat] = entry
        return out
