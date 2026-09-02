"""Lid actuation.

Every driver moves the lid through a slew-rate limit rather than commanding the
end position directly. That matters for more than elegance: a servo told to jump
90 degrees will do it at full torque in ~0.15 s, which is how you catch a cat's
ear. Stepping there over half a second at low speed makes the lid something a
cat can push back against.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from .config import ActuatorConfig, ServoConfig

log = logging.getLogger(__name__)


class ActuatorError(RuntimeError):
    pass


class Actuator:
    """A lid that can be somewhere between closed (0.0) and open (1.0)."""

    def __init__(self, name: str, sleep: Callable[[float], None] = time.sleep):
        self.name = name
        self._sleep = sleep
        self._position = 0.0
        self._lock = threading.Lock()

    @property
    def position(self) -> float:
        return self._position

    @property
    def is_open(self) -> bool:
        return self._position > 0.5

    def open(self) -> None:
        self.move_to(1.0)

    def close(self) -> None:
        self.move_to(0.0)

    def move_to(self, fraction: float) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def shutdown(self) -> None:
        """Leave the hardware in a safe state (lid closed, drive released)."""
        try:
            self.close()
        finally:
            self.release()

    def release(self) -> None:
        """Stop actively driving the actuator."""


class ServoActuator(Actuator):
    """Shared slew logic for anything that takes an angle in degrees.

    One lid, one or more servos. A wide lid usually has a servo on each hinge,
    facing opposite ways, so the pair is *not* driven to the same angle: each
    servo carries its own closed_deg/open_deg and the slew walks all of them
    from their own closed angle to their own open angle together. Every step of
    the ramp writes every servo before sleeping, so the two sides stay in
    lockstep and the lid does not rack.
    """

    def __init__(
        self,
        name: str,
        servo: ServoConfig | list[ServoConfig],
        actuator: ActuatorConfig,
        sleep: Callable[[float], None] = time.sleep,
    ):
        super().__init__(name, sleep)
        # Callers may hand over a single ServoConfig or a list of them.
        self.servos: list[ServoConfig] = list(servo) if isinstance(servo, (list, tuple)) else [servo]
        if not self.servos:
            raise ActuatorError(f"{name}: needs at least one servo")
        self.cfg = actuator
        self._attached = False

    @property
    def servo(self) -> ServoConfig:
        """The first servo. Kept so single-servo call sites read unchanged."""
        return self.servos[0]

    def angles_for(self, fraction: float) -> list[float]:
        """Where each servo should sit for a lid that is *fraction* open."""
        fraction = min(1.0, max(0.0, fraction))
        return [
            s.closed_deg + fraction * (s.open_deg - s.closed_deg)
            for s in self.servos
        ]

    def angle_for(self, fraction: float) -> float:
        """The first servo's angle. For display and for single-servo lids."""
        return self.angles_for(fraction)[0]

    def move_to(self, fraction: float) -> None:
        fraction = min(1.0, max(0.0, fraction))
        with self._lock:
            start = self._position
            if abs(fraction - start) < 1e-3 and self._attached:
                return
            # Step count comes from whichever servo has furthest to travel, so
            # the slew rate limit holds for both sides of a mirrored pair.
            span = max(
                abs(end - begin)
                for begin, end in zip(self.angles_for(start), self.angles_for(fraction))
            )
            steps = max(1, int(round(span / self.cfg.step_deg)))
            delay = (span / steps) / self.cfg.move_speed_deg_s if span else 0.0

            for i in range(1, steps + 1):
                intermediate = start + (fraction - start) * (i / steps)
                self._write_angles(self.angles_for(intermediate))
                self._attached = True
                self._position = intermediate
                if delay:
                    self._sleep(delay)

            self._position = fraction
            log.info("%s lid -> %s", self.name, "open" if fraction > 0.5 else "closed")
            if self.servo.detach_when_idle:
                self._sleep(0.35)     # let the horn arrive before cutting drive
                self.release()

    def release(self) -> None:
        if self._attached:
            self._detach()
            self._attached = False

    def _write_angles(self, degrees: list[float]) -> None:
        """Write one angle per servo, in self.servos order."""
        for index, value in enumerate(degrees):
            self._write_angle(index, value)

    def _write_angle(self, index: int, degrees: float) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def _detach(self) -> None:
        """Stop sending pulses on every servo. Default is to keep holding."""


class MockActuator(ServoActuator):
    """Records everything, moves instantly. For tests and dry runs."""

    def __init__(self, name: str, servo, actuator: ActuatorConfig, **kwargs):
        super().__init__(name, servo, actuator, sleep=lambda _: None)
        # angles is the first servo's track, so single-servo assertions read the
        # same as they always did; per_servo holds one track per servo.
        self.angles: list[float] = []
        self.per_servo: list[list[float]] = [[] for _ in self.servos]
        self.detaches = 0

    def _write_angle(self, index: int, degrees: float) -> None:
        value = round(degrees, 2)
        self.per_servo[index].append(value)
        if index == 0:
            self.angles.append(value)

    def _detach(self) -> None:
        self.detaches += 1


class PCA9685Servo(ServoActuator):
    """One or more channels of a shared 16-channel I2C PWM board."""

    def __init__(self, name: str, servo, actuator: ActuatorConfig, device):
        super().__init__(name, servo, actuator)
        self._device = device
        self._channels = []
        for cfg in self.servos:
            if cfg.channel is None:
                raise ActuatorError(f"{name}: servo.channel is required for the pca9685 driver")
            self._channels.append(cfg.channel)

    def _pulse_us(self, degrees: float) -> float:
        lo, hi = self.cfg.min_pulse_us, self.cfg.max_pulse_us
        return lo + (degrees / 180.0) * (hi - lo)

    def _write_angle(self, index: int, degrees: float) -> None:
        self._device.set_pulse_us(self._channels[index], self._pulse_us(degrees))

    def _detach(self) -> None:
        for channel in self._channels:
            self._device.release(channel)


class GpioServo(ServoActuator):
    """Servo wired straight to a Pi header pin, driven by pigpio."""

    def __init__(self, name: str, servo, actuator: ActuatorConfig, pi):
        super().__init__(name, servo, actuator)
        self._pi = pi
        self._gpios = []
        for cfg in self.servos:
            if cfg.gpio is None:
                raise ActuatorError(f"{name}: servo.gpio is required for the gpio driver")
            self._gpios.append(cfg.gpio)

    def _pulse_us(self, degrees: float) -> int:
        lo, hi = self.cfg.min_pulse_us, self.cfg.max_pulse_us
        return int(round(lo + (degrees / 180.0) * (hi - lo)))

    def _write_angle(self, index: int, degrees: float) -> None:
        self._pi.set_servo_pulsewidth(self._gpios[index], self._pulse_us(degrees))

    def _detach(self) -> None:
        for gpio in self._gpios:
            self._pi.set_servo_pulsewidth(gpio, 0)


class ActuatorFactory:
    """Owns the hardware handles shared between bowls (the I2C board, pigpio)."""

    def __init__(self, cfg: ActuatorConfig, device=None):
        self.cfg = cfg
        self._device = device
        self._pi = None
        self._made: list[Actuator] = []

    def _pca9685(self):
        if self._device is None:
            from .pca9685 import PCA9685

            log.info("opening PCA9685 at 0x%02x", self.cfg.i2c_address)
            self._device = PCA9685(
                bus=self.cfg.i2c_bus,
                address=self.cfg.i2c_address,
                frequency=self.cfg.frequency,
            )
        return self._device

    def _pigpio(self):
        if self._pi is None:
            import pigpio

            pi = pigpio.pi()
            if not pi.connected:
                raise ActuatorError(
                    "cannot reach pigpiod - start it with 'sudo systemctl enable --now pigpiod'"
                )
            self._pi = pi
        return self._pi

    def create(self, name: str, servo: ServoConfig | list[ServoConfig]) -> Actuator:
        driver = self.cfg.driver
        if driver == "mock":
            actuator = MockActuator(name, servo, self.cfg)
        elif driver == "gpio":
            actuator = GpioServo(name, servo, self.cfg, self._pigpio())
        else:
            actuator = PCA9685Servo(name, servo, self.cfg, self._pca9685())
        self._made.append(actuator)
        return actuator

    def shutdown(self) -> None:
        for actuator in self._made:
            try:
                actuator.shutdown()
            except Exception:  # pragma: no cover - best effort
                log.exception("failed to park %s", actuator.name)
        if self._device is not None:
            try:
                self._device.close()      # outputs off, oscillator stopped
            except Exception:  # pragma: no cover - best effort
                log.exception("failed to park the PCA9685")
            self._device = None
        if self._pi is not None:
            self._pi.stop()
            self._pi = None
