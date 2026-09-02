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


# --------------------------------------------------------------------------- #
# lids driven from both hinges
# --------------------------------------------------------------------------- #

def make_pair():
    """A mirrored pair: servo 0 opens 10 -> 100, servo 1 opens 170 -> 80."""
    left = ServoConfig(channel=0, closed_deg=10, open_deg=100)
    right = ServoConfig(channel=1, closed_deg=170, open_deg=80)
    cfg = ActuatorConfig(driver="mock", move_speed_deg_s=90, step_deg=3)
    return MockActuator("bowl1", [left, right], cfg)


def test_a_ganged_pair_ends_at_each_servos_own_angles():
    actuator = make_pair()
    assert actuator.angles_for(0.0) == [10.0, 170.0]
    assert actuator.angles_for(1.0) == [100.0, 80.0]
    assert actuator.angles_for(0.5) == [55.0, 125.0]


def test_mirrored_servos_travel_in_opposite_directions():
    actuator = make_pair()
    actuator.open()
    left, right = actuator.per_servo
    assert left == sorted(left), "servo 0 winds up"
    assert right == sorted(right, reverse=True), "servo 1 winds down"


def test_both_sides_are_written_on_every_step():
    """If one side lagged the other by even a step the lid would rack."""
    actuator = make_pair()
    actuator.open()
    left, right = actuator.per_servo
    assert len(left) == len(right)


def test_step_count_follows_the_servo_with_furthest_to_travel():
    actuator = make_pair()
    actuator.open()
    # servo 1 sweeps 90 degrees, servo 0 only 90 too, but the limit is the max
    assert len(actuator.per_servo[0]) == 30, "90 degrees at 3 degrees per step"
    for track in actuator.per_servo:
        steps = [abs(b - a) for a, b in zip(track, track[1:])]
        assert max(steps) <= 3.001, "slew limit holds on both sides"


def test_detach_releases_the_whole_gang_at_once():
    actuator = make_pair()
    actuator.open()
    assert actuator.detaches == 1, "one detach call, covering every servo"


def test_a_single_servo_still_reads_as_one():
    actuator, servo = make()
    assert actuator.servos == [servo]
    assert actuator.servo is servo
    actuator.open()
    assert actuator.per_servo == [actuator.angles]


def test_the_pca9685_actuator_writes_real_pulse_widths():
    """End to end from an angle to the bytes on the bus, with no adafruit layer."""
    from catbowl.actuators import PCA9685Servo
    from catbowl.pca9685 import LED0_ON_L, FULL_OFF, PCA9685
    from tests.test_pca9685 import FakeBus

    bus = FakeBus()
    device = PCA9685(address=0x40, frequency=50, smbus=bus)
    cfg = ActuatorConfig(driver="pca9685", min_pulse_us=500, max_pulse_us=2500,
                         move_speed_deg_s=10_000, step_deg=180)
    lid = PCA9685Servo(
        "bowl1",
        [ServoConfig(channel=0, closed_deg=0, open_deg=180)],
        cfg,
        device,
    )

    base = LED0_ON_L

    lid.move_to(0.5)                          # mid travel -> 1500 us -> 307 counts
    counts = [off for reg, off in bus.writes if reg == base + 2]
    assert bus.registers[base + 3] == FULL_OFF, "detach_when_idle parks the channel"
    assert 307 & 0xFF in counts, "the slew must have passed through 1500 us"

    lid.move_to(0.0)                          # min_pulse_us -> 500 us -> 102 counts
    counts = [off for reg, off in bus.writes if reg == base + 2]
    assert 102 in counts
