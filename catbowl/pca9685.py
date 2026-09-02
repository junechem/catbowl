"""Direct I2C driver for the PCA9685 16-channel PWM chip.

This replaces adafruit-circuitpython-servokit. Not because that library is bad,
but because of what it drags behind it: servokit depends on Blinka, Blinka
depends on lgpio, and lgpio is published only as a source distribution that
links against a C library Debian does not package. On Raspberry Pi OS the build
reaches the linker and dies on ``cannot find -llgpio``, and no amount of build
tooling fixes a library that is not in the archive.

The portability layer was never earning its keep here. Blinka exists to make
CircuitPython drivers run anywhere; this project talks to exactly one chip on
one bus. That is a page of register writes, and they are the same ones we
already verified by hand with ``i2cset`` when bringing the board up.

Datasheet: NXP PCA9685, "16-channel, 12-bit PWM Fm+ I2C-bus LED controller".
"""

from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)

# --- registers ------------------------------------------------------------ #
MODE1 = 0x00
MODE2 = 0x01
LED0_ON_L = 0x06          # each channel occupies 4 bytes from here
ALL_LED_ON_L = 0xFA
ALL_LED_OFF_H = 0xFD
PRESCALE = 0xFE

# --- MODE1 bits ----------------------------------------------------------- #
RESTART = 0x80
AI = 0x20                 # auto-increment, so a channel can be written in one go
SLEEP = 0x10
ALLCALL = 0x01

# --- MODE2 bits ----------------------------------------------------------- #
OUTDRV = 0x04             # totem-pole outputs, which is what a servo input wants

FULL_OFF = 0x10           # in the OFF_H byte: hold the output low, no pulses

OSC_HZ = 25_000_000.0     # internal oscillator
STEPS = 4096              # 12-bit resolution


class PCA9685Error(RuntimeError):
    pass


class PCA9685:
    """One PCA9685 on an I2C bus.

    The chip powers up asleep (MODE1 = 0x11, oscillator stopped) and outputs
    nothing at all until woken, regardless of what is written to the channel
    registers. Every loss of VCC resets it to that state, so ``configure`` is
    called from __init__ rather than left to the caller to remember.
    """

    def __init__(self, bus: int | str = 1, address: int = 0x40, frequency: int = 50, smbus=None):
        if smbus is None:
            try:
                from smbus2 import SMBus
            except ImportError as exc:   # pragma: no cover - import guard
                raise PCA9685Error(
                    "smbus2 is not installed - pip install smbus2"
                ) from exc
            smbus = SMBus(bus)
        self._bus = smbus
        self.address = address
        self.frequency = frequency
        self._own_bus = True
        self.configure(frequency)

    # -- register access --------------------------------------------------- #

    def _read(self, register: int) -> int:
        return self._bus.read_byte_data(self.address, register)

    def _write(self, register: int, value: int) -> None:
        self._bus.write_byte_data(self.address, register, value & 0xFF)

    # -- setup ------------------------------------------------------------- #

    def configure(self, frequency: int | None = None) -> None:
        """Wake the chip and set the PWM frame rate.

        PRESCALE is writable only while the oscillator is stopped, so the order
        matters: sleep, set the divider, wake, then pulse RESTART to bring the
        outputs back with their previous duty cycles.
        """
        if frequency is not None:
            self.frequency = frequency
        prescale = self._prescale_for(self.frequency)

        self._write(MODE2, OUTDRV)
        self._write(MODE1, SLEEP | AI | ALLCALL)   # stop the oscillator
        self._write(PRESCALE, prescale)
        self._write(MODE1, AI | ALLCALL)           # wake
        time.sleep(0.005)                          # oscillator needs ~500 us
        self._write(MODE1, RESTART | AI | ALLCALL)
        log.debug("pca9685 at 0x%02x: %d Hz (prescale %d)",
                  self.address, self.frequency, prescale)

    @staticmethod
    def _prescale_for(frequency: int) -> int:
        if not 24 <= frequency <= 1526:
            raise PCA9685Error(f"frequency {frequency} Hz is outside the chip's 24-1526 Hz range")
        # Datasheet 7.3.5. 50 Hz gives 121 (0x79), the value we set by hand.
        value = round(OSC_HZ / (STEPS * frequency)) - 1
        return max(3, min(255, int(value)))

    # -- output ------------------------------------------------------------ #

    def set_pulse_us(self, channel: int, microseconds: float) -> None:
        """Drive one channel with a pulse of the given width."""
        period_us = 1_000_000.0 / self.frequency
        counts = int(round(microseconds / period_us * STEPS))
        counts = max(0, min(STEPS - 1, counts))
        self._set_raw(channel, 0, counts)

    def release(self, channel: int) -> None:
        """Stop pulsing a channel. The servo goes limp instead of holding."""
        self._check_channel(channel)
        base = LED0_ON_L + 4 * channel
        self._write(base + 0, 0x00)
        self._write(base + 1, 0x00)
        self._write(base + 2, 0x00)
        self._write(base + 3, FULL_OFF)

    def release_all(self) -> None:
        self._write(ALL_LED_ON_L + 0, 0x00)
        self._write(ALL_LED_ON_L + 1, 0x00)
        self._write(ALL_LED_ON_L + 2, 0x00)
        self._write(ALL_LED_OFF_H, FULL_OFF)

    def sleep(self) -> None:
        """Park the chip: outputs off, oscillator stopped."""
        self.release_all()
        self._write(MODE1, SLEEP | AI | ALLCALL)

    def _set_raw(self, channel: int, on: int, off: int) -> None:
        self._check_channel(channel)
        base = LED0_ON_L + 4 * channel
        self._write(base + 0, on & 0xFF)
        self._write(base + 1, on >> 8)
        self._write(base + 2, off & 0xFF)
        self._write(base + 3, off >> 8)

    @staticmethod
    def _check_channel(channel: int) -> None:
        if not 0 <= channel <= 15:
            raise PCA9685Error(f"channel must be 0..15, got {channel}")

    def close(self) -> None:
        try:
            self.sleep()
        finally:
            if self._own_bus and hasattr(self._bus, "close"):
                self._bus.close()
