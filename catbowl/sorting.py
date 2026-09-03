"""Filing captured photos into labelled folders.

The rig banks every detection in ``<capture.dir>/unsorted``; this is the other
half, the one a human drives from /sort. Labelling is a rename and nothing else:
no image is opened, decoded, re-encoded or copied, so a Pi that is busy watching
for cats spends microseconds per decision rather than a frame's worth of CPU.

Nothing here deletes. Even the discard pile is a folder, because "obviously
useless" is a judgement made in half a second on a phone, and a future model
trained to recognise an empty bowl would want exactly those frames back.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Sequence

log = logging.getLogger(__name__)

# Where a photo goes when it shows nothing worth keeping. Not a delete: see the
# module docstring.
DISCARD = "discard"
# Cap on one directory listing. The queue is served in batches, so there is no
# reason to walk 5000 entries to answer a request that shows one image.
MAX_LISTED = 200
# How long a listing stays good. The capture thread adds files underneath us;
# a few seconds of staleness costs nothing and saves a syscall storm.
LIST_TTL_S = 5.0

# Capture filenames are ours (bowl-date-time.jpg), but the name arrives back
# from a browser, so it is treated as hostile until it matches this.
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}\.jpg$")


class SortError(ValueError):
    """A request that names a file or label the sorter will not act on."""


class Sorter:
    """Moves captured photos from `unsorted/` into one folder per label."""

    def __init__(self, root: str | Path, labels: Sequence[str], clock=time.monotonic):
        self.root = Path(root)
        self.unsorted = self.root / "unsorted"
        self.labels = list(labels)
        self.buckets = [*self.labels, DISCARD]
        self._clock = clock
        self._listing: list[str] = []
        self._listed_at = 0.0
        self._remaining = 0
        # One step of undo: the last move, as (destination, original name).
        self._last_move: tuple[Path, str] | None = None

    # -- queue -------------------------------------------------------------- #

    def pending(self, limit: int = 24, refresh: bool = False) -> list[str]:
        """The next few unsorted filenames, oldest first."""
        now = self._clock()
        if refresh or not self._listing or now - self._listed_at > LIST_TTL_S:
            self._refresh()
        return self._listing[:limit]

    def _refresh(self) -> None:
        self._listed_at = self._clock()
        if not self.unsorted.is_dir():
            self._listing, self._remaining = [], 0
            return
        names = []
        with os.scandir(self.unsorted) as entries:
            for entry in entries:
                if entry.name.endswith(".jpg"):
                    names.append(entry.name)
                    if len(names) >= MAX_LISTED:
                        break
        # The names start with the bowl id and then an ISO-ish timestamp, so a
        # plain sort is chronological per bowl - close enough to file them in
        # the order they were taken without stat()-ing every one.
        self._listing = sorted(names)
        self._remaining = len(names)

    def counts(self) -> dict[str, int]:
        """How many photos sit in each bucket, plus what is left to sort."""
        out = {bucket: _count_jpgs(self.root / bucket) for bucket in self.buckets}
        out["unsorted"] = self._remaining_estimate()
        return out

    def _remaining_estimate(self) -> int:
        if self._clock() - self._listed_at > LIST_TTL_S or not self._listed_at:
            self._refresh()
        # MAX_LISTED is a ceiling on the walk, not on the folder.
        return self._remaining

    # -- actions ------------------------------------------------------------ #

    def path_for(self, name: str) -> Path:
        """The file behind a name from the browser, or raise SortError."""
        if not SAFE_NAME.match(name):
            raise SortError(f"bad filename: {name!r}")
        path = self.unsorted / name
        # Belt and braces: the regex already forbids separators and "..", but a
        # path that has escaped the folder must never be served or moved.
        if path.parent.resolve() != self.unsorted.resolve():
            raise SortError(f"bad filename: {name!r}")
        return path

    def assign(self, name: str, label: str) -> str:
        """File one photo under *label*. Returns the name it was filed as."""
        if label not in self.buckets:
            raise SortError(f"unknown label {label!r}; expected one of {', '.join(self.buckets)}")
        source = self.path_for(name)
        if not source.is_file():
            raise SortError(f"no such photo: {name}")

        target_dir = self.root / label
        target_dir.mkdir(parents=True, exist_ok=True)
        target = _free_path(target_dir / name)
        os.replace(source, target)          # same filesystem: an atomic rename
        self._last_move = (target, name)
        self._forget(name)
        log.info("sorted %s -> %s", name, label)
        return target.name

    def undo(self) -> str | None:
        """Put the last sorted photo back. One step; that is all a thumb needs."""
        if self._last_move is None:
            return None
        target, name = self._last_move
        self._last_move = None
        if not target.is_file():
            return None
        self.unsorted.mkdir(parents=True, exist_ok=True)
        os.replace(target, _free_path(self.unsorted / name))
        self._listing.insert(0, name)
        self._remaining += 1
        log.info("un-sorted %s", name)
        return name

    def _forget(self, name: str) -> None:
        if name in self._listing:
            self._listing.remove(name)
            self._remaining = max(0, self._remaining - 1)


def _count_jpgs(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    with os.scandir(directory) as entries:
        return sum(1 for entry in entries if entry.name.endswith(".jpg"))


def _free_path(path: Path) -> Path:
    """*path*, or the first `name-2.jpg`, `name-3.jpg`... that does not exist.

    Two photos can share a name after an undo, or when a folder is refilled from
    a backup. Renaming the newcomer is always better than silently replacing
    something already labelled by hand.
    """
    if not path.exists():
        return path
    for suffix in range(2, 1000):
        candidate = path.with_name(f"{path.stem}-{suffix}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise SortError(f"cannot find a free name for {path.name}")
