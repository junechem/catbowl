"""The cat gate: motion is cheap but blind, ssdlite is selective but slow.

These tests never load torch. The expensive detector is injected as a stub that
records how often it was asked, because how often it runs is the whole point of
the hybrid.
"""

from pathlib import Path

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
        assert gate.detect(FRAME).bbox == MOVED.bbox
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
        detection = gate.detect(FRAME)
        assert detection is not None, "the cat is still there"
        # Not `is MOVED`: the box reported between confirmations is the motion
        # widened to the cat ssdlite last saw. Same box here, new object.
        assert detection.bbox == MOVED.bbox

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


# --------------------------------------------------------------------------- #
# what gets cropped between confirmations
# --------------------------------------------------------------------------- #

# A cat settled at a bowl barely moves. Background subtraction reports only the
# part that did: one ear, or the end of a tail.
TUFT = Detection((40, 20, 6, 5), 0.1, "motion")
WHOLE_CAT = Detection((10, 10, 50, 40), 0.9, "ssdlite")


def in_a_visit(motion, confirm, clock, **cfg_kwargs):
    """A gate that has already seen and confirmed a cat."""
    gate = build(motion, confirm, clock, **cfg_kwargs)
    assert gate.detect(FRAME) is WHOLE_CAT
    return gate


def test_a_tuft_of_fur_is_widened_to_the_cat_it_belongs_to():
    """The bug that made the collected dataset useless.

    Between confirmations the gate used to hand back the raw motion box, so
    most captured crops were a few pixels of moving fur - unlabellable by a
    human, never mind a classifier.
    """
    clock = FakeClock()
    motion, confirm = Scripted(WHOLE_CAT), Scripted(WHOLE_CAT)
    gate = in_a_visit(motion, confirm, clock)

    motion.result = TUFT
    clock.advance(0.5)                      # still inside confirm_every_s
    detection = gate.detect(FRAME)

    assert confirm.calls == 1, "widening must not cost another inference"
    assert detection.bbox == (10, 10, 50, 40), "the whole animal, not the ear"


def test_the_box_follows_a_cat_that_shifts_along_the_bowl():
    """Union, not replacement: the remembered box is the size, motion is the place."""
    clock = FakeClock()
    motion = Scripted(WHOLE_CAT)
    gate = in_a_visit(motion, Scripted(WHOLE_CAT), clock)

    motion.result = Detection((70, 15, 10, 10), 0.2, "motion")   # moved right
    clock.advance(0.5)
    assert gate.detect(FRAME).bbox == (10, 10, 70, 40)


def test_a_head_down_cat_is_still_cropped_whole():
    """ssdlite refusing is the normal answer for a cat with its face in the bowl,
    and confirm_grace_s keeps the visit alive through it. The crop has to survive
    it too, or every frame of a cat actually eating is a tuft."""
    clock = FakeClock()
    motion, confirm = Scripted(WHOLE_CAT), Scripted(WHOLE_CAT)
    gate = in_a_visit(motion, confirm, clock, confirm_every_s=2.0, confirm_grace_s=25.0)

    motion.result, confirm.result = TUFT, None
    clock.advance(3.0)                      # past the re-check: ssdlite says no cat
    detection = gate.detect(FRAME)

    assert confirm.calls == 2, "the re-check must still have happened"
    assert detection is not None, "confirm_grace_s keeps the visit alive"
    assert detection.bbox == (10, 10, 50, 40)


def test_the_next_visit_does_not_inherit_the_last_cat_s_outline():
    clock = FakeClock()
    motion, confirm = Scripted(WHOLE_CAT), Scripted(WHOLE_CAT)
    gate = in_a_visit(motion, confirm, clock, visit_gap_s=2.0)

    motion.result = None                    # the cat leaves
    clock.advance(5.0)
    assert gate.detect(FRAME) is None

    # A new visit, confirmed with a box of its own, must not be unioned with
    # the cat that was here before - that would crop in half a metre of floor.
    motion.result = TUFT
    confirm.result = Detection((90, 50, 20, 20), 0.9, "ssdlite")
    clock.advance(5.0)
    assert gate.detect(FRAME).bbox == (90, 50, 20, 20)


def test_reset_forgets_the_remembered_box():
    clock = FakeClock()
    gate = in_a_visit(Scripted(WHOLE_CAT), Scripted(WHOLE_CAT), clock)
    gate.reset()
    assert gate._last_cat is None


# --------------------------------------------------------------------------- #
# counting heads
#
# The classifier answers "which cat is this crop", which is the wrong question
# when two cats are at one bowl: whoever the crop shows, the other one is
# standing beside it and would eat through an open lid. Only the detector can
# see both, so the count travels with the detection.
# --------------------------------------------------------------------------- #

TWO_CATS = Detection((12, 12, 30, 30), 0.9, "ssdlite", crowd=2)


def test_one_cat_is_the_default_count():
    assert Detection((0, 0, 4, 4), 0.5, "motion").crowd == 1


def test_a_second_cat_is_reported_on_the_confirming_frame():
    gate = build(Scripted(MOVED), Scripted(TWO_CATS))
    assert gate.detect(FRAME).crowd == 2


def test_the_count_survives_the_frames_between_confirmations():
    """Motion cannot count, so between checks the last real count stands."""
    clock = FakeClock()
    gate = build(Scripted(MOVED), Scripted(TWO_CATS), clock=clock, confirm_every_s=2.0)
    gate.detect(FRAME)
    clock.advance(0.5)
    between = gate.detect(FRAME)
    assert between.source == "hybrid"
    assert between.crowd == 2, "the second cat did not leave just because motion cannot see it"


def test_the_count_drops_back_when_the_second_cat_leaves():
    clock = FakeClock()
    confirm = Scripted(TWO_CATS)
    gate = build(Scripted(MOVED), confirm, clock=clock, confirm_every_s=2.0)
    gate.detect(FRAME)
    confirm.result = IS_CAT           # one of them wandered off
    clock.advance(2.5)
    assert gate.detect(FRAME).crowd == 1


def test_detector_model_must_be_one_we_can_build():
    with pytest.raises(ConfigError, match="ssdlite/yolo"):
        DetectorConfig(model="yolov99")


YOLO_320 = Path(__file__).resolve().parents[1] / "models" / "yolo11n-320.onnx"


@pytest.mark.skipif(not YOLO_320.exists(), reason="YOLO11n ONNX export not present")
def test_yolo_finds_nothing_in_a_blank_frame_and_keeps_its_box_in_the_frame():
    from catbowl.detector import YoloCatDetector

    det = YoloCatDetector(DetectorConfig(model="yolo", yolo_path=str(YOLO_320)))
    assert det.size == 320
    assert det.detect(np.full((720, 1280, 3), 128, np.uint8)) is None
