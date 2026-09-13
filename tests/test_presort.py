"""Judging a pile of captures a visit at a time.

No model here: the classifier's output is written by hand, because what is
being tested is what happens to a run of frames once the probabilities exist.
"""

import pytest

from catbowl.presort import Shot, decide, group_visits, taken_at, visit_verdict

THRESHOLD = 0.86


def shot(second, probabilities, minute=0):
    """One capture at 12:{minute}:{second}, with the given probabilities."""
    name = f"bowl1-20260912-12{minute:02d}{second:02d}-000.jpg"
    best = max(probabilities, key=lambda k: probabilities[k])
    return Shot(name=name, taken=taken_at(name), probabilities=dict(probabilities),
                label=best, confidence=probabilities[best])


def sure(second, cat, minute=0, confidence=0.97):
    rest = (1 - confidence) / 2
    probabilities = {"J": rest, "K": rest, "F": rest}
    probabilities[cat] = confidence
    return shot(second, probabilities, minute)


def unsure_shot(second, leader="J", minute=0):
    """A blurred frame: the right cat is still ahead, but nowhere near sure."""
    probabilities = {"J": 0.30, "K": 0.34, "F": 0.36}
    probabilities[leader] = 0.55
    total = sum(probabilities.values())
    return shot(second, {k: v / total for k, v in probabilities.items()}, minute)


# --------------------------------------------------------------------------- #
# reading the clock off a filename
# --------------------------------------------------------------------------- #

def test_captures_two_seconds_apart_are_two_seconds_apart():
    first = taken_at("bowl1-20260912-195223-383.jpg")
    second = taken_at("bowl1-20260912-195225-383.jpg")
    assert second - first == pytest.approx(2.0)


def test_a_name_that_is_not_a_capture_has_no_time():
    assert taken_at("holiday.jpg") is None
    assert taken_at("bowl1-notadate-nope-000.jpg") is None


# --------------------------------------------------------------------------- #
# grouping
# --------------------------------------------------------------------------- #

def test_a_steady_stream_of_frames_is_one_visit():
    shots = [sure(second, "J") for second in range(0, 20, 2)]
    assert group_visits(shots) == 1


def test_a_gap_starts_a_new_visit():
    shots = [sure(0, "J"), sure(2, "J"), sure(40, "K"), sure(42, "K")]
    assert group_visits(shots, gap_s=20) == 2
    assert [s.visit for s in shots] == [0, 0, 1, 1]


def test_a_long_visit_is_not_split_by_its_own_length():
    """Thirty seconds of eating is one cat, however far it is from the start."""
    shots = [sure(second, "J") for second in range(0, 58, 2)]
    assert group_visits(shots, gap_s=20) == 1


def test_a_photo_with_no_timestamp_stands_alone():
    odd = Shot(name="holiday.jpg", taken=None, probabilities={"J": 1.0}, label="J", confidence=1.0)
    shots = [sure(0, "J"), odd, sure(2, "J")]
    assert group_visits(shots, gap_s=20) == 3


# --------------------------------------------------------------------------- #
# one verdict per visit
# --------------------------------------------------------------------------- #

def test_a_confident_visit_gets_its_cat():
    verdict = visit_verdict([sure(s, "J") for s in range(0, 10, 2)], THRESHOLD)
    assert verdict.label == "J"


def test_one_blurred_frame_is_carried_by_its_neighbours():
    """The whole point: a bad frame mid-visit is not a different cat."""
    shots = [sure(0, "J"), sure(2, "J"), unsure_shot(4), sure(6, "J"), sure(8, "J")]
    outcome = decide(shots, THRESHOLD)

    assert [s.verdict for s in shots] == ["J"] * 5
    assert outcome.rescued == 1
    assert outcome.counts == {"J": 5}


def test_a_visit_nobody_is_sure_about_stays_unsure():
    """Smoothing must not invent confidence that no frame ever had."""
    shots = [unsure_shot(s) for s in range(0, 10, 2)]
    outcome = decide(shots, THRESHOLD)
    assert [s.verdict for s in shots] == ["unsure"] * 5
    assert outcome.smoothed == 0


def test_two_cats_in_one_visit_are_left_to_a_human():
    """A bowl both cats visited must not be filed under whoever posed longer."""
    shots = [sure(0, "J"), sure(2, "J"), sure(4, "J"),
             sure(6, "K"), sure(8, "K")]
    outcome = decide(shots, THRESHOLD)

    assert outcome.mixed == 1
    assert outcome.smoothed == 0
    assert [s.verdict for s in shots] == ["J", "J", "J", "K", "K"], \
        "each frame keeps its own confident answer"


def test_a_single_stray_frame_does_not_veto_a_visit():
    """One misfire in twenty is a misfire, not a second cat."""
    shots = [sure(s, "J") for s in range(0, 36, 2)] + [sure(36, "F")]
    outcome = decide(shots, THRESHOLD)
    assert outcome.mixed == 0
    assert {s.verdict for s in shots} == {"J"}
    assert outcome.rescued == 1, "the stray frame was overruled by the visit"


def test_separate_visits_are_judged_separately():
    shots = [sure(0, "J"), sure(2, "J"), sure(30, "K"), sure(32, "K")]
    decide(shots, THRESHOLD, gap_s=20)
    assert [s.verdict for s in shots] == ["J", "J", "K", "K"]


def test_the_visit_prior_can_be_turned_off():
    shots = [sure(0, "J"), sure(2, "J"), unsure_shot(4), sure(6, "J")]
    outcome = decide(shots, THRESHOLD, use_visits=False)
    assert [s.verdict for s in shots] == ["J", "J", "unsure", "J"]
    assert outcome.rescued == 0


def test_photos_are_judged_in_time_order_whatever_order_they_arrive():
    shots = [sure(6, "J"), unsure_shot(4), sure(0, "J"), sure(2, "J")]
    decide(shots, THRESHOLD)
    assert [s.name for s in shots] == sorted(s.name for s in shots)
    assert {s.verdict for s in shots} == {"J"}


def test_one_sure_frame_cannot_speak_for_a_very_long_visit():
    """A single false-confident frame must not relabel a hundred photos."""
    shots = [unsure_shot(s % 60, minute=s // 60) for s in range(0, 60, 2)]
    shots.append(sure(0, "F", minute=1))
    outcome = decide(shots, THRESHOLD)
    assert outcome.smoothed == 0
    assert outcome.counts["unsure"] == 30


def test_a_few_sure_frames_can_speak_for_a_long_visit():
    shots = [unsure_shot(s, minute=0) for s in range(0, 50, 2)]
    shots += [sure(s, "J", minute=0) for s in range(50, 60, 2)]
    outcome = decide(shots, THRESHOLD)
    assert outcome.smoothed == 1
    assert set(outcome.counts) == {"J"}
