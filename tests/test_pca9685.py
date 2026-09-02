"""The register writes, against a fake bus.

The values here are not derived from the driver - they are what we set by hand
with i2cset when bringing the board up, and what actually made a servo move.
"""

import pytest

from catbowl.pca9685 import (
    ALL_LED_OFF_H,
    FULL_OFF,
    LED0_ON_L,
    MODE1,
    PRESCALE,
    PCA9685,
    PCA9685Error,
)


class FakeBus:
    """Records every write and answers reads from the same store."""

    def __init__(self):
        self.registers: dict[int, int] = {}
        self.writes: list[tuple[int, int]] = []
        self.closed = False

    def write_byte_data(self, address, register, value):
        self.registers[register] = value
        self.writes.append((register, value))

    def read_byte_data(self, address, register):
        return self.registers.get(register, 0)

    def close(self):
        self.closed = True


@pytest.fixture
def chip():
    bus = FakeBus()
    return PCA9685(address=0x40, frequency=50, smbus=bus), bus


def test_fifty_hertz_is_prescale_0x79():
    """The value we set by hand the night the servo first moved."""
    assert PCA9685._prescale_for(50) == 0x79


def test_prescale_is_written_while_the_oscillator_is_stopped():
    """PRESCALE is read-only unless SLEEP is set - get this wrong and the
    frequency silently stays at the power-on 200 Hz, and servos do not move."""
    bus = FakeBus()
    PCA9685(address=0x40, frequency=50, smbus=bus)

    order = [r for r, _ in bus.writes]
    prescale_at = order.index(PRESCALE)
    sleeping = [v for r, v in bus.writes[:prescale_at] if r == MODE1]
    assert sleeping and sleeping[-1] & 0x10, "MODE1 must have SLEEP set before PRESCALE"


def test_the_chip_is_woken_and_restarted():
    bus = FakeBus()
    PCA9685(address=0x40, frequency=50, smbus=bus)
    final = bus.registers[MODE1]
    assert not final & 0x10, "SLEEP must be clear or nothing is output at all"
    assert final & 0x80, "RESTART brings the outputs back after the wake"


def test_a_centred_servo_pulse_is_307_counts(chip):
    """1.5 ms of a 20 ms frame, in 4096ths: the 0x0133 we wrote by hand."""
    device, bus = chip
    device.set_pulse_us(0, 1500)
    base = LED0_ON_L
    assert bus.registers[base + 0] == 0x00, "ON_L"
    assert bus.registers[base + 1] == 0x00, "ON_H"
    off = bus.registers[base + 2] | (bus.registers[base + 3] << 8)
    assert off == 307


@pytest.mark.parametrize("pulse_us, counts", [(1000, 205), (1500, 307), (2000, 410)])
def test_the_pulse_table_we_verified_on_the_bench(chip, pulse_us, counts):
    device, bus = chip
    device.set_pulse_us(3, pulse_us)
    base = LED0_ON_L + 4 * 3
    assert (bus.registers[base + 2] | (bus.registers[base + 3] << 8)) == counts


def test_each_channel_has_its_own_four_registers(chip):
    device, bus = chip
    device.set_pulse_us(15, 1500)
    assert bus.registers[LED0_ON_L + 4 * 15 + 2] == 307 & 0xFF
    assert LED0_ON_L + 4 * 0 + 2 not in bus.registers, "channel 0 must be untouched"


def test_release_sets_full_off_rather_than_a_zero_pulse(chip):
    """A zero-width pulse is still a pulse. FULL_OFF stops them, so the servo
    goes limp instead of buzzing against the lid."""
    device, bus = chip
    device.set_pulse_us(2, 1500)
    device.release(2)
    assert bus.registers[LED0_ON_L + 4 * 2 + 3] == FULL_OFF


def test_sleep_parks_every_channel_then_stops_the_oscillator(chip):
    device, bus = chip
    device.sleep()
    assert bus.registers[ALL_LED_OFF_H] == FULL_OFF
    assert bus.registers[MODE1] & 0x10, "SLEEP set"


def test_close_parks_the_chip_and_releases_the_bus(chip):
    device, bus = chip
    device.close()
    assert bus.registers[MODE1] & 0x10
    assert bus.closed


def test_pulses_are_clamped_to_the_frame(chip):
    device, bus = chip
    device.set_pulse_us(0, 999_999)
    off = bus.registers[LED0_ON_L + 2] | (bus.registers[LED0_ON_L + 3] << 8)
    assert off == 4095, "must never exceed 12 bits and wrap to a short pulse"


def test_channels_outside_the_chip_are_rejected(chip):
    device, _ = chip
    with pytest.raises(PCA9685Error, match="0..15"):
        device.set_pulse_us(16, 1500)


def test_an_impossible_frequency_is_rejected():
    with pytest.raises(PCA9685Error, match="24-1526"):
        PCA9685(address=0x40, frequency=10, smbus=FakeBus())
