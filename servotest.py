"""Bench test: one servo wired straight to the Pi header.

    brown/black  -> pin 6   (GND)
    red          -> pin 4   (5V)
    orange/yellow-> pin 12  (GPIO18)

No lid attached. Ctrl-C to stop.
"""

from gpiozero import AngularServo
from time import sleep

servo = AngularServo(
    18,
    min_angle=0,
    max_angle=180,
    min_pulse_width=0.0005,   # 500 us  - matches actuator.min_pulse_us
    max_pulse_width=0.0025,   # 2500 us - matches actuator.max_pulse_us
)

try:
    while True:
        servo.angle = 10      # closed_deg
        sleep(1)
        servo.angle = 95      # open_deg
        sleep(1)
except KeyboardInterrupt:
    servo.detach()            # stop the pulses so it does not buzz
