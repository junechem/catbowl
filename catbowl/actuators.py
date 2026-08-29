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
    """Shared slew logic for anything that takes an angle in degrees."""

    def __init__(
        self,
        name: str,
        servo: ServoConfig,
        actuator: ActuatorConfig,
        sleep: Callable[[float], None] = time.sleep,
    ):
        super().__init__(name, sleep)
        self.servo = servo
        self.cfg = actuator
        self._attached = False

    def angle_for(self, fraction: float) -> float:
        fraction = min(1.0, max(0.0, fraction))
        return self.servo.closed_deg + fraction * (self.servo.open_deg - self.servo.closed_deg)

    def move_to(self, fraction: float) -> None:
        fraction = min(1.0, max(0.0, fraction))
        with self._lock:
            start = self._position
            if abs(fraction - start) < 1e-3 and self._attached:
                return
            span = abs(self.angle_for(fraction) - self.angle_for(start))
            steps = max(1, int(round(span / self.cfg.step_deg)))
            delay = (span / steps) / self.cfg.move_speed_deg_s if span else 0.0

            for i in range(1, steps + 1):
                intermediate = start + (fraction - start) * (i / steps)
                self._write_angle(self.angle_for(intermediate))
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

    def _write_angle(self, degrees: float) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def _detach(self) -> None:
        """Stop sending pulses. Default is to keep holding."""


class MockActuator(ServoActuator):
    """Records everything, moves instantly. For tests and dry runs."""

    def __init__(self, name: str, servo: ServoConfig, actuator: ActuatorConfig, **kwargs):
        super().__init__(name, servo, actuator, sleep=lambda _: None)
        self.angles: list[float] = []
        self.detaches = 0

    def _write_angle(self, degrees: float) -> None:
        self.angles.append(round(degrees, 2))

    def _detach(self) -> None:
        self.detaches += 1


class PCA9685Servo(ServoActuator):
    """One channel of a shared 16-channel I2C PWM board."""

    def __init__(self, name: str, servo: ServoConfig, actuator: ActuatorConfig, kit):
        super().__init__(name, servo, actuator)
        if servo.channel is None:
            raise ActuatorError(f"{name}: servo.channel is required for the pca9685 driver")
        self._servo = kit.servo[servo.channel]
        self._servo.set_pulse_width_range(actuator.min_pulse_us, actuator.max_pulse_us)
        self._servo.actuation_range = 180

    def _write_angle(self, degrees: float) -> None:
        self._servo.angle = degrees

    def _detach(self) -> None:
        self._servo.angle = None   # adafruit's way of saying "stop the pulses"


class GpioServo(ServoActuator):
    """Servo wired straight to a Pi header pin, driven by pigpio."""

    def __init__(self, name: str, servo: ServoConfig, actuator: ActuatorConfig, pi):
        super().__init__(name, servo, actuator)
        if servo.gpio is None:
            raise ActuatorError(f"{name}: servo.gpio is required for the gpio driver")
        self._pi = pi
        self._gpio = servo.gpio

    def _pulse_us(self, degrees: float) -> int:
        lo, hi = self.cfg.min_pulse_us, self.cfg.max_pulse_us
        return int(round(lo + (degrees / 180.0) * (hi - lo)))

    def _write_angle(self, degrees: float) -> None:
        self._pi.set_servo_pulsewidth(self._gpio, self._pulse_us(degrees))

    def _detach(self) -> None:
        self._pi.set_servo_pulsewidth(self._gpio, 0)


class ActuatorFactory:
    """Owns the hardware handles shared between bowls (the I2C board, pigpio)."""

    def __init__(self, cfg: ActuatorConfig):
        self.cfg = cfg
        self._kit = None
        self._pi = None
        self._made: list[Actuator] = []

    def _servokit(self):
        if self._kit is None:
            from adafruit_servokit import ServoKit

            log.info("opening PCA9685 at 0x%02x", self.cfg.i2c_address)
            self._kit = ServoKit(channels=16, address=self.cfg.i2c_address, frequency=self.cfg.frequency)
        return self._kit

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

    def create(self, name: str, servo: ServoConfig) -> Actuator:
        driver = self.cfg.driver
        if driver == "mock":
            actuator = MockActuator(name, servo, self.cfg)
        elif driver == "gpio":
            actuator = GpioServo(name, servo, self.cfg, self._pigpio())
        else:
            actuator = PCA9685Servo(name, servo, self.cfg, self._servokit())
        self._made.append(actuator)
        return actuator

    def shutdown(self) -> None:
        for actuator in self._made:
            try:
                actuator.shutdown()
            except Exception:  # pragma: no cover - best effort
                log.exception("failed to park %s", actuator.name)
        if self._pi is not None:
            self._pi.stop()
            self._pi = None
