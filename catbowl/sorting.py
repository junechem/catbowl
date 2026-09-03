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

# The queue and the browser both need "unsorted" to be addressable as a bucket:
# re-filing a mistake means moving a photo back into it.
UNSORTED = "unsorted"
# One page of the browse grid. Thumbnails are the full captured crops - a few kB
# each, and no server-side decode - so a page is cheap but not free.
PAGE_SIZE = 40

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
        # Everything a photo can be filed into or browsed from.
        self.all_buckets = [UNSORTED, *self.buckets]
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

    def dir_for(self, bucket: str) -> Path:
        """The folder behind a bucket name from the browser."""
        if bucket == UNSORTED:
            return self.unsorted
        if bucket not in self.buckets:
            raise SortError(
                f"unknown bucket {bucket!r}; expected one of {', '.join(self.all_buckets)}"
            )
        return self.root / bucket

    def path_for(self, name: str, bucket: str = UNSORTED) -> Path:
        """The file behind a name from the browser, or raise SortError."""
        if not SAFE_NAME.match(name):
            raise SortError(f"bad filename: {name!r}")
        directory = self.dir_for(bucket)
        path = directory / name
        # Belt and braces: the regex already forbids separators and "..", but a
        # path that has escaped the folder must never be served or moved.
        if path.parent.resolve() != directory.resolve():
            raise SortError(f"bad filename: {name!r}")
        return path

    def listing(self, bucket: str, offset: int = 0, limit: int = PAGE_SIZE) -> tuple[list[str], int]:
        """One page of a bucket, newest first, with the bucket's total.

        Newest first because a mistake is nearly always one just made, and the
        order comes from the timestamp in the filename rather than from stat():
        it is the same order, across every bowl, without a syscall per photo.
        """
        directory = self.dir_for(bucket)
        if not directory.is_dir():
            return [], 0
        with os.scandir(directory) as entries:
            names = [entry.name for entry in entries if entry.name.endswith(".jpg")]
        names.sort(key=_taken_at, reverse=True)
        offset = max(0, offset)
        return names[offset:offset + max(1, limit)], len(names)

    def assign(self, name: str, label: str) -> str:
        """File one photo out of the queue. Returns the name it was filed as."""
        if label == UNSORTED:
            raise SortError("a photo in the queue is already unsorted")
        filed = self.move(name, UNSORTED, label)
        self._last_move = (self.dir_for(label) / filed, name)
        self._forget(name)
        return filed

    def move(self, name: str, source_bucket: str, target_bucket: str) -> str:
        """Move one photo between buckets. Returns the name it landed under.

        This is what fixes a mis-sort: a photo can go straight to another label,
        or back to `unsorted` to be judged again from the phone.
        """
        if source_bucket == target_bucket:
            return name
        source = self.path_for(name, source_bucket)
        target_dir = self.dir_for(target_bucket)
        if not source.is_file():
            raise SortError(f"no such photo: {name}")

        target_dir.mkdir(parents=True, exist_ok=True)
        target = _free_path(target_dir / name)
        os.replace(source, target)          # same filesystem: an atomic rename
        if UNSORTED in (source_bucket, target_bucket):
            self._listed_at = 0.0           # the queue's cached listing is stale
        log.info("moved %s: %s -> %s", name, source_bucket, target_bucket)
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


def _taken_at(name: str) -> str:
    """A sort key that orders photos by when they were taken, across bowls.

    Capture names are `<bowl>-<date>-<time>-<ms>.jpg`, so dropping the bowl id
    leaves a key that sorts chronologically. Anything not shaped like that (a
    file dropped in by hand) sorts last under the epoch, rather than raising.
    """
    parts = name.split("-", 1)
    return parts[1] if len(parts) == 2 else "\x00" + name


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
