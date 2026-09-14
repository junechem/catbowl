"""Per-cat allowances on a shared bowl.

The rig has one bowl and three cats. K eats whenever she likes; J and F get two
minutes an hour each. These tests are the only place that rule is written down
in a way that fails when it stops being true.
"""

import pytest

from catbowl.actuators import MockActuator
from catbowl.config import (ActuatorConfig, BowlConfig, ConfigError, PolicyConfig,
                            RationConfig, ServoConfig)
from catbowl.controller import BowlController, BowlState
from catbowl.events import Event
from catbowl.rations import UNLIMITED, Ledger

HOUR = 3600.0
TWO_MINUTES = 120.0


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


# --------------------------------------------------------------------------- #
# the ledger on its own
# --------------------------------------------------------------------------- #

@pytest.fixture
def ledger():
    return Ledger({"J": RationConfig(seconds=TWO_MINUTES, per_s=HOUR)})


def test_a_cat_without_a_ration_is_never_refused(ledger):
    assert ledger.remaining("K", 0.0) == UNLIMITED
    ledger.spend("K", 0.0, 10_000)
    assert ledger.allows("K", 0.0)
    assert not ledger.limited("K")


def test_time_spent_comes_off_the_allowance(ledger):
    ledger.spend("J", 0.0, 30)
    assert ledger.remaining("J", 0.0) == 90
    assert ledger.allows("J", 0.0)


def test_an_exhausted_allowance_refuses(ledger):
    ledger.spend("J", 0.0, TWO_MINUTES)
    assert ledger.remaining("J", 0.0) == 0
    assert not ledger.allows("J", 0.0)


def test_the_allowance_comes_back_as_the_window_rolls_past_it(ledger):
    ledger.spend("J", 0.0, 60)
    ledger.spend("J", 600, 60)              # ten minutes later, now empty
    assert not ledger.allows("J", 600)

    assert not ledger.allows("J", HOUR - 1), "the first meal is still inside the hour"
    assert ledger.remaining("J", HOUR + 1) == 60, "the first meal has aged out"
    assert ledger.remaining("J", HOUR + 601) == TWO_MINUTES, "and now the second"


def test_it_says_when_an_empty_allowance_comes_back(ledger):
    ledger.spend("J", 100, TWO_MINUTES)
    assert ledger.refills_at("J", 200) == 100 + HOUR
    assert ledger.refills_at("J", HOUR + 200) is None, "it already has some"


# --------------------------------------------------------------------------- #
# the bowl, shared
# --------------------------------------------------------------------------- #

@pytest.fixture
def shared():
    """One bowl for J, K and F: two minutes an hour each for J and F, K free."""
    clock = FakeClock()
    events: list[Event] = []
    bowl = BowlConfig(
        id="bowl1",
        cats=["J", "K", "F"],
        rations={"J": RationConfig(seconds=TWO_MINUTES, per_s=HOUR),
                 "F": RationConfig(seconds=TWO_MINUTES, per_s=HOUR)},
        servos=[ServoConfig(channel=0)],
        policy=PolicyConfig(open_confirm_s=0.0, close_delay_s=5.0, max_open_s=0,
                            close_on_intruder=True, intruder_grace_s=2.0,
                            cooldown_s=1.0),
    )
    actuator = MockActuator("bowl1", bowl.servos, ActuatorConfig(driver="mock"))
    controller = BowlController(bowl, actuator, vote_window=6, votes_required=4,
                                clock=clock, on_event=events.append)
    return controller, actuator, clock, events


def eat(controller, clock, cat, seconds, dt=0.2):
    """*cat* stands at the bowl for *seconds*."""
    for _ in range(int(seconds / dt)):
        controller.observe(True, cat, 0.95)
        clock.advance(dt)
    controller.observe(True, cat, 0.95)


def leave(controller, clock, seconds=8.0, dt=0.5):
    for _ in range(int(seconds / dt)):
        controller.observe(False)
        clock.advance(dt)


def test_the_bowl_opens_for_every_cat_it_serves(shared):
    controller, actuator, clock, _ = shared
    for cat in ("J", "K", "F"):
        eat(controller, clock, cat, 1.0)
        assert actuator.is_open, f"{cat} should have been fed"
        assert controller.status()["feeding"] == cat
        leave(controller, clock)
        clock.advance(2.0)


def test_a_cat_this_bowl_does_not_serve_is_refused(shared):
    controller, actuator, clock, events = shared
    eat(controller, clock, "stranger", 3.0)
    assert not actuator.is_open
    denied = [e for e in events if e.kind == "denied"]
    assert denied and denied[0].detail["reason"] == "not this bowl's cat"


def test_a_rationed_cat_is_cut_off_when_its_time_runs_out(shared):
    controller, actuator, clock, events = shared
    eat(controller, clock, "J", TWO_MINUTES + 5)

    assert not actuator.is_open, "J has had its two minutes"
    closed = [e for e in events if e.kind == "closed"]
    assert closed[-1].detail["reason"] == "ration"
    assert closed[-1].cat == "J"
    assert controller.ledger.remaining("J", clock()) == 0


def test_an_exhausted_cat_is_not_let_back_in(shared):
    controller, actuator, clock, events = shared
    eat(controller, clock, "J", TWO_MINUTES + 5)
    leave(controller, clock)
    clock.advance(60)

    eat(controller, clock, "J", 5.0)
    assert not actuator.is_open
    refusals = [e for e in events if e.kind == "denied" and e.cat == "J"]
    assert refusals, "J should have been told no"
    assert refusals[-1].detail["reason"] == "out of ration"
    assert refusals[-1].detail["retry_in_s"] > 0


def test_an_hour_later_a_rationed_cat_eats_again(shared):
    controller, actuator, clock, _ = shared
    eat(controller, clock, "J", TWO_MINUTES + 5)
    leave(controller, clock)

    clock.advance(HOUR + 10)
    eat(controller, clock, "J", 1.0)
    assert actuator.is_open


def test_an_unrationed_cat_eats_as_long_as_it_likes(shared):
    controller, actuator, clock, _ = shared
    eat(controller, clock, "K", TWO_MINUTES * 3)
    assert actuator.is_open, "K has no allowance to run out of"
    assert controller.state is BowlState.OPEN


def test_one_cat_running_out_does_not_touch_another(shared):
    controller, actuator, clock, _ = shared
    eat(controller, clock, "J", TWO_MINUTES + 5)
    leave(controller, clock)
    clock.advance(5)

    eat(controller, clock, "F", 1.0)
    assert actuator.is_open, "F has its own allowance"
    assert controller.status()["feeding"] == "F"


def test_a_short_meal_leaves_the_rest_of_the_allowance(shared):
    """The reason this is a budget and not one meal an hour."""
    controller, actuator, clock, _ = shared
    eat(controller, clock, "J", 30.0)
    leave(controller, clock)                       # startled away
    clock.advance(600)

    eat(controller, clock, "J", 1.0)
    assert actuator.is_open, "J kept the time it did not eat"
    # 30s of meal, the 5s close_delay the lid stayed up behind it, and 1s of
    # this visit. The close delay counts: the lid was open for that too.
    assert 82 < controller.ledger.remaining("J", clock()) < 90


def test_another_cat_arriving_ends_the_meal(shared):
    """Even one the bowl serves: K gets her own turn, not J's open lid."""
    controller, actuator, clock, events = shared
    eat(controller, clock, "J", 2.0)
    assert actuator.is_open

    eat(controller, clock, "K", 4.0)
    closed = [e for e in events if e.kind == "closed"]
    assert closed and closed[-1].detail["reason"] == "intruder"


def test_only_the_cat_being_fed_is_charged(shared):
    controller, _, clock, _ = shared
    eat(controller, clock, "K", 20.0)
    assert controller.ledger.remaining("J", clock()) == TWO_MINUTES


def test_a_ration_for_a_cat_the_bowl_does_not_feed_is_a_config_error():
    with pytest.raises(ConfigError, match="does not feed"):
        BowlConfig(id="bowl1", cats=["J"], rations={"K": RationConfig(seconds=10)})


# --------------------------------------------------------------------------- #
# the sitting cap, and the cat it does not apply to
# --------------------------------------------------------------------------- #

@pytest.fixture
def capped():
    """max_open_s of 30, with K exempt from it."""
    clock = FakeClock()
    events: list[Event] = []
    bowl = BowlConfig(
        id="bowl1",
        cats=["J", "K", "F"],
        uncapped=["K"],
        servos=[ServoConfig(channel=0)],
        policy=PolicyConfig(open_confirm_s=0.0, close_delay_s=5.0, max_open_s=30.0,
                            close_on_intruder=True, intruder_grace_s=2.0,
                            cooldown_s=1.0),
    )
    actuator = MockActuator("bowl1", bowl.servos, ActuatorConfig(driver="mock"))
    controller = BowlController(bowl, actuator, vote_window=6, votes_required=4,
                                clock=clock, on_event=events.append)
    return controller, actuator, clock, events


def test_an_uncapped_cat_is_not_cut_off_by_max_open_s(capped):
    controller, actuator, clock, events = capped
    eat(controller, clock, "K", 300.0)
    assert actuator.is_open, "K stays fed for as long as she stays"
    assert not [e for e in events if e.kind == "closed"]


def test_a_capped_cat_still_is(capped):
    controller, actuator, clock, events = capped
    eat(controller, clock, "J", 40.0)
    closed = [e for e in events if e.kind == "closed"]
    assert closed and closed[0].detail["reason"] == "max_open_s"


def test_an_uncapped_lid_still_drops_when_she_leaves(capped):
    controller, actuator, clock, events = capped
    eat(controller, clock, "K", 60.0)
    leave(controller, clock)
    assert not actuator.is_open
    assert [e for e in events if e.kind == "closed"][-1].detail["reason"] == "left"


def test_an_uncapped_lid_still_drops_for_an_intruder(capped):
    controller, actuator, clock, events = capped
    eat(controller, clock, "K", 60.0)
    eat(controller, clock, "J", 4.0)
    assert [e for e in events if e.kind == "closed"][-1].detail["reason"] == "intruder"


@pytest.mark.parametrize("uncapped, rations, message", [
    (["M"], {}, "does not feed"),
    (["J"], {"J": RationConfig(seconds=60)}, "both uncapped and on a ration"),
])
def test_uncapped_is_checked(uncapped, rations, message):
    with pytest.raises(ConfigError, match=message):
        BowlConfig(id="bowl1", cats=["J", "K"], uncapped=uncapped, rations=rations)
