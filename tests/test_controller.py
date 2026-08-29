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
def rig():
    clock = FakeClock()
    events: list[Event] = []
    bowl = BowlConfig(
        id="bowl1",
        cat=OWNER,
        servo=ServoConfig(channel=0),
        policy=PolicyConfig(
            open_confirm_s=1.0, close_delay_s=5.0, max_open_s=60.0,
            close_on_intruder=True, intruder_grace_s=2.0, cooldown_s=3.0,
        ),
    )
    actuator = MockActuator("bowl1", bowl.servo, ActuatorConfig(driver="mock"))
    controller = BowlController(bowl, actuator, vote_window=4, votes_required=3,
                                clock=clock, on_event=events.append)
    return controller, actuator, clock, events


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


def test_single_stray_frame_never_opens_a_lid(rig):
    controller, actuator, clock, _ = rig
    for _ in range(10):
        controller.observe(True, OWNER, 0.9)     # one good frame ...
        clock.advance(0.2)
        for _ in range(3):                       # ... buried in noise
            controller.observe(True, "unknown", 0.2)
            clock.advance(0.2)
    assert controller.state is BowlState.CLOSED
    assert not actuator.is_open


def test_wrong_cat_is_denied_and_logged_once(rig):
    controller, actuator, clock, events = rig
    feed(controller, clock, INTRUDER, frames=20, dt=0.2)
    assert controller.state is BowlState.CLOSED
    assert not actuator.is_open
    denials = [e for e in events if e.kind == "denied"]
    assert len(denials) == 1, "a lingering cat should not spam the log"
    assert denials[0].cat == INTRUDER
    assert denials[0].detail["owner"] == OWNER


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
    clock.advance(2.0)
    controller.observe(True, OWNER, 0.9)
    assert not actuator.is_open, "old votes must not combine with new ones to open a lid"


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
