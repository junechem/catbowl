"""The cat gate: motion is cheap but blind, ssdlite is selective but slow.

These tests never load torch. The expensive detector is injected as a stub that
records how often it was asked, because how often it runs is the whole point of
the hybrid.
"""

import numpy as np
import pytest

from catbowl.config import ConfigError, DetectorConfig
from catbowl.detector import Detection, Detector, HybridCatDetector, build_detector

FRAME = np.zeros((80, 120, 3), dtype=np.uint8)


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class Scripted(Detector):
    """Returns whatever it is told to, and counts the calls."""

    def __init__(self, result):
        self.result = result
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return self.result


MOVED = Detection((10, 10, 40, 40), 0.3, "motion")
IS_CAT = Detection((12, 12, 30, 30), 0.9, "ssdlite")


def build(motion, confirm, clock=None, **cfg_kwargs):
    cfg = DetectorConfig(type="hybrid", **cfg_kwargs)
    return HybridCatDetector(cfg, motion=motion, confirm=confirm, clock=clock or FakeClock())


def test_motion_alone_is_not_enough_to_report_a_cat():
    """The failure this whole class exists to prevent: a dog opening a lid."""
    confirm = Scripted(None)          # ssdlite: whatever moved, it is not a cat
    gate = build(Scripted(MOVED), confirm)
    assert gate.detect(FRAME) is None
    assert confirm.calls == 1, "motion must be checked, not trusted"


def test_a_confirmed_cat_is_reported_with_the_tight_box():
    gate = build(Scripted(MOVED), Scripted(IS_CAT))
    detection = gate.detect(FRAME)
    assert detection is IS_CAT, "ssdlite's box crops better than motion's"


def test_stillness_short_circuits_before_the_expensive_detector():
    confirm = Scripted(IS_CAT)
    gate = build(Scripted(None), confirm)
    assert gate.detect(FRAME) is None
    assert confirm.calls == 0, "an empty room must cost nothing"


def test_a_confirmation_is_cached_for_the_rest_of_the_visit():
    clock = FakeClock()
    confirm = Scripted(IS_CAT)
    gate = build(Scripted(MOVED), confirm, clock=clock, confirm_every_s=2.0)

    assert gate.detect(FRAME) is IS_CAT
    for _ in range(10):               # cat settles in and keeps eating
        clock.advance(0.1)
        assert gate.detect(FRAME) is MOVED
    assert confirm.calls == 1, "one confirmation should cover the whole visit"


def test_the_confirmation_expires_and_is_rechecked():
    clock = FakeClock()
    confirm = Scripted(IS_CAT)
    gate = build(Scripted(MOVED), confirm, clock=clock, confirm_every_s=2.0)

    gate.detect(FRAME)
    clock.advance(2.5)
    gate.detect(FRAME)
    assert confirm.calls == 2, "a long visit must be re-checked, not trusted forever"


def test_a_rejection_backs_off_instead_of_rechecking_every_frame():
    """A swaying curtain must not pin the CPU running ssdlite at full rate."""
    clock = FakeClock()
    confirm = Scripted(None)
    gate = build(Scripted(MOVED), confirm, clock=clock, reject_backoff_s=1.0)

    for _ in range(10):
        clock.advance(0.05)
        assert gate.detect(FRAME) is None
    assert confirm.calls == 1

    clock.advance(1.5)
    gate.detect(FRAME)
    assert confirm.calls == 2, "the backoff expires, it does not give up for good"


def test_the_cat_leaving_drops_the_confirmation():
    clock = FakeClock()
    motion = Scripted(MOVED)
    confirm = Scripted(IS_CAT)
    gate = build(motion, confirm, clock=clock, confirm_every_s=60.0, visit_gap_s=2.0)

    assert gate.detect(FRAME) is IS_CAT
    motion.result = None              # cat wanders off
    assert gate.detect(FRAME) is None
    clock.advance(2.0)                # long enough to count as gone
    assert gate.detect(FRAME) is None
    motion.result = MOVED             # something else arrives, well inside 60 s
    gate.detect(FRAME)
    assert confirm.calls == 2, "the next visitor gets checked on its own merits"


def test_a_brief_still_moment_does_not_end_the_visit():
    """A settled cat stops registering as motion long before it has left."""
    clock = FakeClock()
    motion = Scripted(MOVED)
    confirm = Scripted(IS_CAT)
    gate = build(motion, confirm, clock=clock, confirm_every_s=60.0, visit_gap_s=2.0)

    assert gate.detect(FRAME) is IS_CAT
    motion.result = None
    clock.advance(1.0)                # still, but not for long enough
    assert gate.detect(FRAME) is None
    motion.result = MOVED
    gate.detect(FRAME)
    assert confirm.calls == 1, "the visit survived the pause; no re-check needed"


def test_a_head_down_cat_keeps_the_lid_open():
    """The failure this gate was rebuilt for.

    ssdlite stops recognising a cat the moment it puts its face in the bowl.
    Treating that as 'the cat left' shut the lid on the animal mid-meal.
    """
    clock = FakeClock()
    motion = Scripted(MOVED)
    confirm = Scripted(IS_CAT)
    gate = build(motion, confirm, clock=clock,
                 confirm_every_s=2.0, confirm_grace_s=25.0)

    assert gate.detect(FRAME) is IS_CAT
    confirm.result = None             # head goes down; ssdlite sees no cat

    for _ in range(60):               # 24 s of eating
        clock.advance(0.4)
        assert gate.detect(FRAME) is MOVED, "the cat is still there"

    assert confirm.calls > 1, "it kept checking, in case the cat was swapped"


def test_a_sustained_refusal_finally_ends_the_visit():
    clock = FakeClock()
    motion = Scripted(MOVED)
    confirm = Scripted(IS_CAT)
    gate = build(motion, confirm, clock=clock,
                 confirm_every_s=2.0, confirm_grace_s=10.0)

    assert gate.detect(FRAME) is IS_CAT
    confirm.result = None
    clock.advance(11.0)
    assert gate.detect(FRAME) is None, "grace ran out with no cat ever seen again"


def test_reset_clears_the_gate_state():
    clock = FakeClock()
    confirm = Scripted(IS_CAT)
    gate = build(Scripted(MOVED), confirm, clock=clock, confirm_every_s=60.0)
    gate.detect(FRAME)
    gate.reset()
    gate.detect(FRAME)
    assert confirm.calls == 2


def test_hybrid_is_the_default_and_ssdlite_is_built_lazily():
    """Constructing the default detector must not pull torch into memory."""
    cfg = DetectorConfig()
    assert cfg.type == "hybrid"
    gate = build_detector(cfg)
    assert isinstance(gate, HybridCatDetector)
    assert gate._confirm is None, "ssdlite loads on first use, not at startup"


def test_an_unknown_detector_type_is_rejected():
    with pytest.raises(ConfigError, match="hybrid/motion/ssdlite/none"):
        DetectorConfig(type="magic")


def test_negative_gate_timings_are_rejected():
    with pytest.raises(ConfigError, match="confirm_every_s"):
        DetectorConfig(confirm_every_s=-1)
