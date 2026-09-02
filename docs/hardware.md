# Hardware

## Bill of materials

| Part | Suggested | Qty | Why this one |
|---|---|---|---|
| Raspberry Pi 4 (4 GB) | — | 1 | 4 GB is enough; 8 GB gains nothing here |
| microSD | 32 GB A2 | 1 | A1/A2 rating matters more than size |
| Pi power supply | Official 5.1 V 3 A USB-C | 1 | Undervoltage causes camera dropouts that look like model bugs |
| USB webcam | Logitech C270 or any UVC 720p | 3 | Must support **MJPEG** — see the USB bandwidth note |
| Servo | MG996R (metal gear) or MG90S (small lids) | 3 | Metal gear survives a cat batting the lid |
| PWM driver | PCA9685 16-channel I2C board | 1 | Keeps servo timing off the Pi's CPU and out of GPIO |
| Servo power supply | 5 V 5–6 A barrel supply | 1 | **Not** the Pi's 5 V rail — see below |
| Wire, screw terminals, dupont jumpers | — | — | |
| Lid, hinge, servo horn linkage | 3 mm acrylic or thin ply | 3 | Keep it light — mass is the safety risk |

Roughly £120–160 / $150–200 all in, assuming you have the Pi.

## Why three cameras

You asked and left the choice open, so: **one camera per bowl**, mounted about
25–35 cm above and slightly in front of each bowl, looking down at maybe 30°.

- Identification is unambiguous. Whoever camera 2 sees is who is at bowl 2, with
  no logic mapping a face position to a bowl and no ties to break when two cats
  crowd together.
- Each camera gets a large, close, consistently framed view of one cat's face
  and shoulders, which is worth far more accuracy than any model change.
- Each bowl becomes an independent worker. One dead camera stops one bowl.

The single wide-angle alternative is fully supported — the commented block at the
bottom of `config/bowls.yaml` sets it up with one device and three ROIs. It costs
less and wires more simply, but crowding gets ambiguous and each cat occupies far
fewer pixels. Use it if the bowls must sit close together anyway.

### USB bandwidth (the one real gotcha)

The Pi 4's four USB ports share one USB 2.0 controller (~35 MB/s in practice).
Three cameras streaming *raw* YUYV at 640×480×10 fps is ~55 MB/s and will fail,
usually as "cannot allocate bandwidth" or one camera silently going black.
`catbowl` requests **MJPEG** on every camera (`cameras.py:OpenCVCapture`), which
brings each stream down to ~1–2 MB/s. Check that your cameras support it:

```
v4l2-ctl -d /dev/video0 --list-formats
```

If a camera is YUYV-only, drop it to 320×240 or use fewer cameras. Prefer the
USB 3.0 ports (blue) — they are on the same controller but the ports themselves
negotiate better.

Also note that most webcams claim **two** `/dev/videoN` nodes (the second is a
metadata node). That is why the shipped config uses devices 0, 2 and 4. Run
`python -m catbowl cameras` to see which ones actually deliver frames.

## Why servos on a PCA9685

Hobby servos are the right actuator for lifting a light lid a few centimetres:
three wires, built-in position control, no limit switches, no homing, £4 each.
A linear actuator gives a cleaner straight lift but needs an H-bridge per bowl
and costs five times as much; a stepper needs a driver board and homing on every
boot. Neither buys you anything for a 100 g lid.

The PCA9685 rather than direct GPIO because:

- The Pi has no spare hardware PWM channels for three servos, and software PWM
  jitters, which makes servos buzz and creep.
- One I2C board drives all three from one external supply, with 13 channels spare.
- `actuator.driver: gpio` (via `pigpiod`) is implemented as a fallback if you
  would rather not buy the board.

## Wiring

```
                 +-------------------------------+
   5V 5A PSU --->| V+  (6-pin header)        SDA |<--- Pi pin 3  (GPIO2 / SDA)
        GND  --->| GND (6-pin header)        SCL |<--- Pi pin 5  (GPIO3 / SCL)
                 |                           VCC |<--- Pi pin 1  (3.3 V, logic only)
                 |         PCA9685           GND |<--- Pi pin 6  (GND)
                 | ch0   ch1   ch2   ch3         |
                 +--|-----|-----|-----|----------+
                    |     |     |     |
                    +--+--+     |     |
                       |        |     |
                   bowl1 lid  bowl2 bowl3
                  (two servos, one per hinge)

   Pi USB ports:  webcam0, webcam1, webcam2
```

**Feed `V+` at the 6-pin header, not the green screw terminal.** On the cheap
clones (HW-170, HiLetgo, hiBCTR) the terminal block often does not reach the
servo rail — you measure 5 V at the screws and 0 V at the servo V+ pins, and
nothing moves. Both boards in our 2-pack failed this way. The `V+` pin on the
6-pin header is the same net on the far side of the fault, so power it there and
leave the screw terminal empty. Whatever protection the terminal block offered
is bypassed too, so check polarity with a meter *before* connecting: red probe
on your + wire must read **+5 V**, not −5 V.

Rules that matter:

1. **Servos never draw from the Pi.** An MG996R stalls at ~2.5 A. Three of them
   will brown out the Pi instantly, which shows up as random reboots and SD card
   corruption. `V+` comes from the separate 5 V supply. The Pi's own 5 V pin is
   fine for bench-testing one unloaded servo and nothing more.
2. **Common ground.** The PSU ground and the Pi ground must be tied together, or
   the servo signal has no reference and the servos will twitch.
3. `VCC` on the PCA9685 is the *logic* supply — 3.3 V from the Pi. It is not the
   servo power rail.
4. Put a 470–1000 µF electrolytic capacitor across V+/GND at the board to absorb
   the current spike when three servos start at once.
5. An inline fuse (3–5 A) on the PSU's positive lead is cheap insurance next to
   a water bowl.

Enable I2C once: `sudo raspi-config` → Interface Options → I2C → Yes, then
`i2cdetect -y 1` should show the board at `0x40`.

The chip is driven directly over `/dev/i2c-1` by `catbowl/pca9685.py`, through
`smbus2` — pure Python, no compiled extension. There is deliberately no
adafruit-circuitpython-servokit here: it depends on Blinka, which depends on
lgpio, which PyPI ships only as a source distribution linking against a C
library Debian does not package, so it cannot be built on Raspberry Pi OS.

## The lid

Design goal: the plate should **lift up and clear**, not slide across like a
guillotine. A plate hinged at the back of the bowl and lifted by a short pushrod
from the servo horn does this naturally — as it closes it descends onto the rim,
and a cat's head is pushed out rather than trapped.

- Keep the plate light (3 mm acrylic, ~50–100 g). Mass, not servo torque, is what
  hurts.
- Aim for 60–90° of travel. `python -m catbowl calibrate --bowl bowl1` walks the
  servo to angles you type so you can find the two end positions, then put them
  in the config as `servo.closed_deg` / `servo.open_deg`.

### Two servos on one lid

A wide or heavy lid is steadier driven from both hinges. List them under
`servos:` instead of `servo:` and give each its own channel:

```yaml
  - id: bowl1
    cat: mochi
    servos:
      - {channel: 0, closed_deg: 10,  open_deg: 95}
      - {channel: 1, closed_deg: 170, open_deg: 85}
```

The pair is **ganged**: every step of the slew writes both servos before
sleeping, so the two sides cannot drift apart and rack the lid.

They face opposite ways, so they do not share one pair of angles with a sign
flip — each carries its own `closed_deg`/`open_deg`. Mirroring about 180° is
only a starting guess; horn splines land where they land, so calibrate. In
`calibrate` the angle you type is **servo 0's** angle and the others follow
proportionally through their own range, and the prompt prints all of them.

Get this wrong and the two servos fight each other: they will buzz, draw stall
current, and twist the lid. Move to the closed position first, by hand at low
speed, and confirm both horns sit where you expect before opening.
- Leave a 5 mm gap at the closed position rather than clamping down hard. The
  point is to stop a cat *eating*, not to seal the bowl.
- Mount the servo so a cat cannot reach the linkage, and so nothing metal sits
  where food or water can splash.

See [safety.md](safety.md) before you let the cats near it.
