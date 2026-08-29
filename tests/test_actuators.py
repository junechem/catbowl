import pytest

from catbowl.actuators import ActuatorFactory, MockActuator
from catbowl.config import ActuatorConfig, ServoConfig


def make(**servo_kwargs):
    servo = ServoConfig(channel=0, closed_deg=10, open_deg=100, **servo_kwargs)
    cfg = ActuatorConfig(driver="mock", move_speed_deg_s=90, step_deg=3)
    return MockActuator("bowl1", servo, cfg), servo


def test_angle_mapping_spans_the_configured_range():
    actuator, servo = make()
    assert actuator.angle_for(0.0) == servo.closed_deg
    assert actuator.angle_for(1.0) == servo.open_deg
    assert actuator.angle_for(0.5) == pytest.approx(55.0)


def test_out_of_range_fractions_are_clamped():
    actuator, servo = make()
    actuator.move_to(5.0)
    assert actuator.position == 1.0
    assert max(actuator.angles) <= servo.open_deg
    actuator.move_to(-3.0)
    assert actuator.position == 0.0
    assert min(actuator.angles) >= servo.closed_deg


def test_the_lid_is_slewed_rather_than_snapped():
    """A single 90-degree jump at full servo speed is what catches an ear."""
    actuator, _ = make()
    actuator.open()
    assert len(actuator.angles) == 30, "90 degrees at 3 degrees per step"
    assert actuator.angles == sorted(actuator.angles), "must move monotonically"
    steps = [b - a for a, b in zip(actuator.angles, actuator.angles[1:])]
    assert max(steps) <= 3.001


def test_detach_releases_the_servo_when_idle():
    actuator, _ = make(detach_when_idle=True)
    actuator.open()
    assert actuator.detaches == 1
    actuator.close()
    assert actuator.detaches == 2


def test_holding_servos_are_not_detached():
    actuator, _ = make(detach_when_idle=False)
    actuator.open()
    assert actuator.detaches == 0


def test_is_open_reflects_position():
    actuator, _ = make()
    assert not actuator.is_open
    actuator.open()
    assert actuator.is_open
    actuator.close()
    assert not actuator.is_open


def test_shutdown_closes_then_releases():
    actuator, _ = make(detach_when_idle=False)
    actuator.open()
    actuator.shutdown()
    assert actuator.position == 0.0
    assert actuator.detaches == 1


def test_factory_parks_every_lid_it_made():
    factory = ActuatorFactory(ActuatorConfig(driver="mock"))
    lids = [factory.create(f"bowl{i}", ServoConfig(channel=i)) for i in range(3)]
    for lid in lids:
        lid.open()
    factory.shutdown()
    assert all(lid.position == 0.0 for lid in lids)
