"""Reading a pile of captures as visits rather than as loose photos.

A frame is not an independent sample. The rig fires every two seconds at a cat
that stays for a minute, so thirty photos in a row are one animal, and the
answer to "which cat is this" is the same for all of them. Judging each one
alone throws that away: a blurred frame mid-visit becomes `unsure` even though
the twenty around it were called J at 0.99.

So the photos are grouped by the gap between them, and each visit gets one
verdict from the average of its frames. That is worth more than a majority vote
on labels, because it keeps the model's uncertainty: fifteen frames at 0.6 for J
and one at 0.9 for F average out to J, which is almost certainly right.

The prior is refused when the visit does not look like one cat. Two cats sharing
a bowl, or one leaving as another arrives, produce a visit with confident frames
for both, and smoothing that would file the lot under whichever cat happened to
be photographed more. Those fall back to per-photo answers, which is where they
belong: in front of a human.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Captures arrive every capture.interval_s (2 s by default), and a cat that
# settles in to eat is photographed continuously. A gap this long means the
# bowl was empty in between, so whatever comes next is a fresh arrival and
# nothing about the last cat carries over.
VISIT_GAP_S = 20.0
# How much of a visit's confident opinion has to dissent before the visit is
# treated as more than one cat, and left alone.
MIXED_FRACTION = 0.25


@dataclass
class Shot:
    """One photo, its timestamp, and what the classifier made of it."""

    name: str
    taken: float | None
    probabilities: dict[str, float]
    label: str                  # the per-photo answer, after the confidence floor
    confidence: float
    visit: int = -1
    # Filled in once the visit has spoken: where the photo is actually filed,
    # and whether it got there on its neighbours' evidence rather than its own.
    verdict: str = ""
    rescued: bool = False


def taken_at(name: str) -> float | None:
    """Seconds since the epoch-ish, parsed from `<bowl>-<date>-<time>-<ms>.jpg`.

    Only differences between these matter, so a fixed-length decoding of the
    digits is enough and no calendar is consulted. Anything not shaped like a
    capture name returns None and is never grouped with anything.
    """
    from datetime import datetime

    parts = name.rsplit(".", 1)[0].split("-")
    if len(parts) < 3:
        return None
    _, date, clock = parts[0], parts[1], parts[2]
    milliseconds = parts[3] if len(parts) > 3 else "0"
    try:
        stamp = datetime.strptime(date + clock, "%Y%m%d%H%M%S")
        return stamp.timestamp() + int(milliseconds) / 1000.0
    except ValueError:
        return None


def group_visits(shots: list[Shot], gap_s: float = VISIT_GAP_S) -> int:
    """Number each shot's visit, in place. Returns how many visits there were.

    *shots* must already be in time order. Photos with no readable timestamp
    are each their own visit: an unknown time cannot be said to be near
    anything else's.
    """
    visit = -1
    previous: float | None = None
    for shot in shots:
        if shot.taken is None:
            visit += 1
            previous = None
        elif previous is None or shot.taken - previous > gap_s:
            visit += 1
            previous = shot.taken
        else:
            previous = shot.taken
        shot.visit = visit
    return visit + 1


@dataclass
class Verdict:
    """What a whole visit is taken to be."""

    label: str | None           # None: no single answer, judge the photos alone
    confidence: float = 0.0
    mixed: bool = False         # confident frames disagreed: probably two cats
    reason: str = ""


def visit_verdict(shots: list[Shot], threshold: float,
                  mixed_fraction: float = MIXED_FRACTION) -> Verdict:
    """One answer for a run of photos, or None to fall back to per-photo.

    A visit is named when frames that *were* sure agree with each other, and
    the unsure frames in between inherit that name. The alternative - averaging
    every frame and demanding the average clear the threshold - punishes a cat
    for holding still through a blurry stretch, which is most of a meal.
    """
    if not shots:
        return Verdict(None, reason="empty")

    confident = [s for s in shots if s.confidence >= threshold and s.probabilities]
    # A handful of sure frames can speak for a long visit, but not a single one
    # for a hundred: one false confident frame would then mislabel the lot.
    # This is the same idea as votes_required at runtime, scaled to the visit.
    needed = max(1, round(0.1 * len(shots)))
    if len(confident) < needed:
        return Verdict(None,
                       reason=f"only {len(confident)} of {len(shots)} frames were sure")

    tally: dict[str, int] = {}
    for shot in confident:
        best = max(shot.probabilities, key=lambda k: shot.probabilities[k])
        tally[best] = tally.get(best, 0) + 1
    winner = max(tally, key=lambda k: tally[k])
    dissent = (len(confident) - tally[winner]) / len(confident)
    if dissent >= mixed_fraction:
        # Two cats were recognised in here with confidence. One stray frame is
        # a misfire, a quarter of them is a second cat, and only the second is
        # a reason to distrust the whole visit.
        return Verdict(None, mixed=True,
                       reason=f"{len(confident) - tally[winner]} of {len(confident)} "
                              f"confident frames were not {winner}")

    mean = sum(s.probabilities.get(winner, 0.0) for s in shots) / len(shots)
    return Verdict(winner, mean)


@dataclass
class Outcome:
    """What presorting a pile did, for the summary it prints."""

    counts: dict[str, int] = field(default_factory=dict)
    visits: int = 0
    smoothed: int = 0           # visits decided as a whole
    mixed: int = 0              # visits refused for looking like two cats
    rescued: int = 0            # photos named only because their neighbours were


def decide(shots: list[Shot], threshold: float, gap_s: float = VISIT_GAP_S,
           unsure: str = "unsure", use_visits: bool = True) -> Outcome:
    """Fill in every shot's verdict, in place, and report what happened."""
    shots.sort(key=lambda s: (s.taken is None, s.taken or 0.0, s.name))
    outcome = Outcome(visits=group_visits(shots, gap_s) if use_visits else len(shots))

    index = 0
    while index < len(shots):
        end = index + 1
        if use_visits:
            while end < len(shots) and shots[end].visit == shots[index].visit:
                end += 1
        visit = shots[index:end]

        verdict = visit_verdict(visit, threshold) if use_visits else Verdict(None)
        if verdict.label is not None:
            outcome.smoothed += 1
            for shot in visit:
                shot.verdict = verdict.label
                # Named by its neighbours: on its own this frame would have
                # been filed as unsure, or under the wrong cat.
                shot.rescued = shot.confidence < threshold or shot.label != verdict.label
        else:
            outcome.mixed += 1 if verdict.mixed else 0
            for shot in visit:
                shot.verdict = shot.label if shot.confidence >= threshold else unsure

        for shot in visit:
            outcome.counts[shot.verdict] = outcome.counts.get(shot.verdict, 0) + 1
            outcome.rescued += 1 if shot.rescued else 0
        index = end
    return outcome
