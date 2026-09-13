"""State machine tests. A fake clock lets minutes of behaviour run in microseconds."""

import pytest

from catbowl.actuators import MockActuator
from catbowl.config import ActuatorConfig, BowlConfig, PolicyConfig, ServoConfig
from catbowl.controller import BowlController, BowlState
from catbowl.events import Event

OWNER = "mochi"
INTRUDER = "pepper"


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def rig_factory():
    """Builds a controller, so a test can vary one policy knob."""
    def build(**policy):
        clock = FakeClock()
        events: list[Event] = []
        bowl = BowlConfig(
            id="bowl1",
            cat=OWNER,
            servos=[ServoConfig(channel=0)],
            policy=PolicyConfig(
                **{"open_confirm_s": 1.0, "close_delay_s": 5.0, "max_open_s": 60.0,
                   "close_on_intruder": True, "intruder_grace_s": 2.0, "cooldown_s": 3.0,
                   **policy},
            ),
        )
        actuator = MockActuator("bowl1", bowl.servos, ActuatorConfig(driver="mock"))
        controller = BowlController(bowl, actuator, vote_window=4, votes_required=3,
                                    clock=clock, on_event=events.append)
        return controller, actuator, clock, events
    return build


@pytest.fixture
def rig(rig_factory):
    return rig_factory()


def feed(controller, clock, label, frames=4, dt=0.2, present=True, confidence=0.9):
    """Simulate *frames* consecutive observations at *dt* apart."""
    for _ in range(frames):
        controller.observe(present, label, confidence)
        clock.advance(dt)


def test_starts_closed(rig):
    controller, actuator, _, _ = rig
    assert controller.state is BowlState.CLOSED
    assert not actuator.is_open


def test_opens_for_its_own_cat_after_confirmation(rig):
    controller, actuator, clock, events = rig

    feed(controller, clock, OWNER, frames=3, dt=0.2)   # 0.6s: votes met, timer not
    assert controller.state is BowlState.CLOSED, "should not open before open_confirm_s"

    feed(controller, clock, OWNER, frames=4, dt=0.2)   # past 1.0s
    assert controller.state is BowlState.OPEN
    assert actuator.is_open
    assert [e.kind for e in events] == ["opened"]
    assert events[0].cat == OWNER


def test_one_confident_sighting_is_enough_to_open(rig):
    """A cat walking up has no history, and waiting for a consensus costs it
    seconds at the bowl. One frame the classifier was sure about opens the lid;
    being wrong is caught by the intruder rule a moment later."""
    controller, actuator, clock, _ = rig
    controller.observe(True, OWNER, 0.9)         # one good frame ...
    for _ in range(2):                           # ... buried in noise
        clock.advance(0.2)
        controller.observe(True, "unknown", 0.2)
    clock.advance(0.7)
    controller.observe(True, "unknown", 0.2)     # open_confirm_s has now passed
    assert actuator.is_open


def test_a_bowl_can_be_told_to_wait_for_a_consensus(rig_factory):
    """policy.open_votes restores the cautious behaviour for anyone who wants it."""
    controller, actuator, clock, _ = rig_factory(open_votes=3)
    controller.observe(True, OWNER, 0.9)
    for _ in range(3):
        clock.advance(0.2)
        controller.observe(True, "unknown", 0.2)
    clock.advance(1.0)
    controller.observe(True, "unknown", 0.2)
    assert not actuator.is_open, "one sighting is not three"

    feed(controller, clock, OWNER, frames=3)
    assert actuator.is_open


def test_the_owner_cannot_open_its_lid_while_another_cat_is_in_front_of_it(rig):
    """One frame of J is enough, but not while K is what most frames show."""
    controller, actuator, clock, _ = rig
    feed(controller, clock, INTRUDER, frames=3)
    controller.observe(True, OWNER, 0.9)         # a glimpse of the owner behind
    clock.advance(1.2)
    controller.observe(True, INTRUDER, 0.9)
    assert not actuator.is_open


def test_wrong_cat_is_denied_and_logged_once(rig):
    controller, actuator, clock, events = rig
    feed(controller, clock, INTRUDER, frames=20, dt=0.2)
    assert controller.state is BowlState.CLOSED
    assert not actuator.is_open
    denials = [e for e in events if e.kind == "denied"]
    assert len(denials) == 1, "a lingering cat should not spam the log"
    assert denials[0].cat == INTRUDER
    assert denials[0].detail["reason"] == "not this bowl's cat"


def test_closes_after_the_cat_leaves(rig):
    controller, actuator, clock, events = rig
    feed(controller, clock, OWNER, frames=8, dt=0.2)
    assert actuator.is_open

    feed(controller, clock, None, frames=10, dt=0.2, present=False)   # 2s away
    assert actuator.is_open, "a brief look away must not drop the lid"

    feed(controller, clock, None, frames=20, dt=0.2, present=False)   # past 5s
    assert not actuator.is_open
    closed = [e for e in events if e.kind == "closed"][0]
    assert closed.detail["reason"] == "left"
    assert closed.detail["duration_s"] > 0


def test_intruder_at_an_open_bowl_closes_it(rig):
    controller, actuator, clock, events = rig
    feed(controller, clock, OWNER, frames=8, dt=0.2)
    assert actuator.is_open

    feed(controller, clock, INTRUDER, frames=3, dt=0.2)     # 0.6s, inside the grace period
    assert actuator.is_open

    feed(controller, clock, INTRUDER, frames=12, dt=0.2)    # past intruder_grace_s
    assert not actuator.is_open
    closed = [e for e in events if e.kind == "closed"][0]
    assert closed.detail["reason"] == "intruder"
    assert closed.detail["intruder"] == INTRUDER


def test_intruder_tolerated_when_the_policy_says_so(rig):
    controller, actuator, clock, _ = rig
    controller.cfg.policy.close_on_intruder = False
    feed(controller, clock, OWNER, frames=8, dt=0.2)
    feed(controller, clock, INTRUDER, frames=30, dt=0.2)
    assert actuator.is_open


def test_max_open_is_a_hard_ceiling(rig):
    controller, actuator, clock, events = rig
    feed(controller, clock, OWNER, frames=8, dt=0.2)
    for _ in range(400):                       # owner never leaves
        controller.observe(True, OWNER, 0.9)
        clock.advance(0.2)
        if not actuator.is_open:
            break
    assert not actuator.is_open
    assert [e for e in events if e.kind == "closed"][0].detail["reason"] == "max_open_s"


def test_cooldown_blocks_an_immediate_reopen(rig):
    controller, actuator, clock, _ = rig
    feed(controller, clock, OWNER, frames=8, dt=0.2)
    feed(controller, clock, None, frames=30, dt=0.2, present=False)
    assert controller.state is BowlState.COOLDOWN

    feed(controller, clock, OWNER, frames=6, dt=0.2)        # 1.2s of cooldown left
    assert not actuator.is_open, "lid must not chatter open again during cooldown"

    clock.advance(3.0)
    feed(controller, clock, OWNER, frames=8, dt=0.2)
    assert actuator.is_open


def test_stale_votes_decay_when_the_bowl_is_empty(rig):
    controller, actuator, clock, _ = rig
    feed(controller, clock, OWNER, frames=2, dt=0.2)        # partial evidence
    feed(controller, clock, None, frames=10, dt=0.5, present=False)
    assert controller.votes.tally() == {}

    controller.observe(True, OWNER, 0.9)                    # one fresh frame
    assert controller.votes.tally() == {OWNER: 1}, "the stale votes are gone"
    clock.advance(2.0)
    controller.observe(True, OWNER, 0.9)
    assert actuator.is_open, "a fresh sighting opens on its own merits"


def test_force_close_parks_the_lid(rig):
    controller, actuator, clock, events = rig
    feed(controller, clock, OWNER, frames=8, dt=0.2)
    controller.force_close("shutdown")
    assert not actuator.is_open
    assert [e for e in events if e.kind == "closed"][0].detail["reason"] == "shutdown"


def test_status_is_json_friendly(rig):
    import json

    controller, _, clock, _ = rig
    feed(controller, clock, OWNER, frames=8, dt=0.2)
    payload = json.loads(json.dumps(controller.status()))
    assert payload["state"] == "open"
    assert payload["cat"] == OWNER
    assert payload["opens"] == 1


def test_a_broken_event_sink_cannot_stop_the_lid(rig):
    controller, actuator, clock, _ = rig

    def explode(_event):
        raise RuntimeError("log disk full")

    controller.on_event = explode
    feed(controller, clock, OWNER, frames=8, dt=0.2)
    assert actuator.is_open


# --- manual override ------------------------------------------------------- #

def test_a_manual_open_holds_against_an_empty_bowl(rig):
    controller, actuator, clock, _ = rig
    controller.set_manual("open")
    assert actuator.position == 1.0

    for _ in range(50):
        clock.advance(1.0)
        controller.observe(present=False)
    assert actuator.position == 1.0, "the state machine must not override a hand"
    assert controller.manual == "open"


def test_a_manual_close_holds_against_the_right_cat(rig):
    controller, actuator, clock, _ = rig
    controller.set_manual("closed")
    feed(controller, clock, OWNER, frames=50)
    assert actuator.position == 0.0, "held shut, even for the owner"


def test_resuming_auto_goes_through_cooldown(rig):
    controller, actuator, clock, _ = rig
    controller.set_manual("open")
    controller.set_manual(None)

    assert controller.manual is None
    assert controller.state is BowlState.COOLDOWN
    assert actuator.position == 0.0

    feed(controller, clock, OWNER, frames=10)
    assert actuator.position == 0.0, "a cat standing there cannot reopen instantly"


def test_the_lid_returns_to_automatic_control_after_the_cooldown(rig):
    controller, actuator, clock, _ = rig
    controller.set_manual("open")
    controller.set_manual(None)
    clock.advance(4.0)                      # cooldown_s is 3.0
    feed(controller, clock, OWNER, frames=10)
    assert actuator.position == 1.0, "the state machine has the lid back"


def test_an_unknown_manual_mode_is_rejected(rig):
    controller, _, _, _ = rig
    with pytest.raises(ValueError):
        controller.set_manual("ajar")


def test_a_capped_sitting_reopens_once_the_cat_is_detected_again(rig):
    """The portioning rule: 30 seconds, the lid drops, then normal service.

    The cat does not have to go anywhere. Once the cooldown lapses the machine
    asks the same question it always asks - is this cat here and confirmed - and
    opens again when the answer is yes.
    """
    controller, actuator, clock, events = rig
    controller.cfg.policy.max_open_s = 30.0

    feed(controller, clock, OWNER, frames=8, dt=0.2)
    assert actuator.is_open

    feed(controller, clock, OWNER, frames=200, dt=0.2)      # 40s: the cap fires
    closed = [e for e in events if e.kind == "closed"]
    assert closed[0].detail["reason"] == "max_open_s"

    # The cooldown (3s) plus a fresh confirmation, with the cat never moving.
    feed(controller, clock, OWNER, frames=40, dt=0.2)
    assert actuator.is_open
    assert controller.stats["opens"] == 2


def test_a_manual_open_still_works_after_the_cap(rig):
    controller, actuator, clock, _ = rig
    controller.cfg.policy.max_open_s = 30.0

    feed(controller, clock, OWNER, frames=8, dt=0.2)
    feed(controller, clock, OWNER, frames=200, dt=0.2)

    controller.set_manual("open")
    assert actuator.is_open


# --------------------------------------------------------------------------- #
# keeping a limp lid shut
# --------------------------------------------------------------------------- #

def writes(actuator):
    return len(actuator.angles)


def test_a_shut_bowl_resends_closed_so_a_pawed_lid_goes_back(rig):
    """A limp servo can be pushed open with the bowl shut. Nothing else would
    notice until the next meal, so "closed" is re-sent on a timer."""
    controller, actuator, clock, _ = rig
    before = writes(actuator)

    feed(controller, clock, None, frames=24, dt=0.5, present=False)   # 12 s, empty

    assert writes(actuator) - before == 2, "once at 5 s and once at 10 s"
    assert actuator.angles[-1] == controller.cfg.servo.closed_deg


def test_an_open_lid_is_never_pulled_shut_by_the_reassert(rig):
    controller, actuator, clock, _ = rig
    feed(controller, clock, OWNER, frames=8, dt=0.2)
    assert actuator.is_open
    feed(controller, clock, OWNER, frames=40, dt=0.2)                 # 8 s of eating
    assert actuator.is_open


def test_a_manual_open_is_left_alone_and_a_manual_close_is_kept(rig):
    controller, actuator, clock, _ = rig
    controller.set_manual("open")
    feed(controller, clock, None, frames=24, dt=0.5, present=False)
    assert actuator.is_open

    controller.set_manual("closed")
    before = writes(actuator)
    feed(controller, clock, None, frames=12, dt=0.5, present=False)
    assert writes(actuator) - before == 1


def test_a_holding_servo_is_not_rewritten(rig):
    """It is already pushing back towards closed; the actuator skips the write."""
    controller, actuator, clock, _ = rig
    actuator.servos[0].detach_when_idle = False
    controller.force_close()                  # attach and hold
    before = writes(actuator)
    feed(controller, clock, None, frames=24, dt=0.5, present=False)
    assert writes(actuator) == before


def test_the_reassert_can_be_turned_off(rig):
    controller, actuator, clock, _ = rig
    controller.cfg.policy.reassert_closed_s = 0
    before = writes(actuator)
    feed(controller, clock, None, frames=24, dt=0.5, present=False)
    assert writes(actuator) == before


# --------------------------------------------------------------------------- #
# two cats
#
# The worker reports CROWD instead of a name when the detector counts more than
# one cat. The state machine needs no special case for it: an unfamiliar label
# cannot open the lid, and it closes an open one like any other intruder.
# --------------------------------------------------------------------------- #

def test_a_crowd_never_opens_the_lid(rig):
    from catbowl import CROWD

    controller, actuator, clock, _ = rig
    feed(controller, clock, CROWD, frames=20)
    assert controller.state is BowlState.CLOSED
    assert not actuator.is_open


def test_a_second_cat_arriving_closes_an_open_lid(rig):
    from catbowl import CROWD

    controller, actuator, clock, events = rig
    feed(controller, clock, OWNER, frames=10)
    assert actuator.is_open, "the owner alone should have been fed"

    feed(controller, clock, CROWD, frames=20)      # a housemate joins
    assert not actuator.is_open
    reasons = [e.detail.get("reason") for e in events if e.kind == "closed"]
    assert "intruder" in reasons


def test_a_sighting_that_has_scrolled_out_of_the_window_does_not_open(rig):
    """"One sighting" means one the camera can still see, not one from a minute ago."""
    controller, actuator, clock, _ = rig
    controller.observe(True, OWNER, 0.9)
    for _ in range(4):                           # vote_window is 4: the owner falls out
        clock.advance(0.3)
        controller.observe(True, "unknown", 0.2)
    assert controller.votes.count(OWNER) == 0
    assert not actuator.is_open
