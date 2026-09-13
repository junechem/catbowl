"""Calibration has to reach angles outside the current guess, or it cannot fix it."""

import pytest

from catbowl.cli import calibration_range
from catbowl.config import ServoConfig


def test_a_single_servo_gets_the_full_travel():
    """The bug: it stopped at the configured 170 and would not go to 180."""
    wide, lo, hi = calibration_range([ServoConfig(channel=1, closed_deg=170, open_deg=85)])
    assert (lo, hi) == (0.0, 180.0)
    assert (wide[0].closed_deg, wide[0].open_deg) == (0.0, 180.0)
    assert wide[0].channel == 1


def test_a_mirrored_partner_still_mirrors_across_the_full_travel():
    wide, lo, hi = calibration_range([
        ServoConfig(channel=0, closed_deg=10, open_deg=95),
        ServoConfig(channel=1, closed_deg=170, open_deg=85),
    ])
    assert (lo, hi) == (0.0, 180.0)
    assert (wide[1].closed_deg, wide[1].open_deg) == (180.0, 0.0)


def test_the_range_narrows_so_no_partner_is_driven_past_its_end():
    # The partner sits 40 degrees further on, so the first servo cannot
    # go above 140 without pushing it past 180.
    wide, lo, hi = calibration_range([
        ServoConfig(channel=0, closed_deg=10, open_deg=95),
        ServoConfig(channel=1, closed_deg=50, open_deg=135),
    ])
    assert (lo, hi) == (0.0, pytest.approx(140.0))
    assert all(0.0 <= s.closed_deg <= 180.0 and 0.0 <= s.open_deg <= 180.0 for s in wide)
