# catbowl

Three cats, three bowls, three diets. A Raspberry Pi watches each bowl, works out
which cat is standing at it, and lifts the lid only for that bowl's owner.

```
camera ─▶ motion/cat detector ─▶ crop ─▶ MobileNet embedding ─▶ classifier
                                                                    │
                                            sliding vote window ◀───┘
                                                    │
                                          per-bowl state machine ─▶ servo ─▶ lid
```

Each bowl runs its own worker thread. Cameras are opened once and shared, so one
wide-angle camera with three ROIs works as well as three separate webcams.

## Status

Working and tested end to end without hardware:

```
python -m catbowl selftest      # synthetic cameras, real pipeline, simulated lids
python -m pytest -q             # 85 tests
```

The hardware paths (PCA9685, pigpio, picamera2) are written but have not been run
against real silicon — that part is yours to verify with `catbowl doctor` and
`catbowl calibrate`.

## Quick start

On a workstation, to try the pipeline:

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m catbowl selftest
```

On the Pi:

```
scripts/install_pi.sh           # packages, venv, I2C check, camera probe
```

Then:

1. **Wire it up** — [docs/hardware.md](docs/hardware.md). Bill of materials,
   wiring diagram, and the two mistakes that will bite you (servo power off the
   Pi's rail, and USB bandwidth with three cameras).
2. **Edit `config/bowls.yaml`** — cat names, camera devices, servo channels.
3. **Find the lid angles** — `python -m catbowl calibrate --bowl bowl1`.
4. **Build a dataset and train** — [docs/training.md](docs/training.md).
   Import your phone photos, then capture more from the mounted cameras.
5. **Watch before you move anything** — `python -m catbowl run --dry-run` logs
   every decision without touching a servo. Read [docs/safety.md](docs/safety.md).
6. **Install the service** — `systemd/catbowl.service`.

While it runs, `http://<pi>:8080/` shows each bowl's state, what its camera is
looking at right now, and the last twenty events.

## Commands

| Command | What it does |
|---|---|
| `catbowl run` | Run the feeder. `--dry-run` simulates lids, `--collect` banks retraining data |
| `catbowl doctor` | Check packages, config, model and hardware, and say what is missing |
| `catbowl cameras` | Probe every `/dev/video*` and report which ones deliver frames |
| `catbowl calibrate --bowl bowl1` | Walk a servo to angles you type, to find the end positions |
| `catbowl import --src DIR --label mochi` | Crop cats out of existing photos into the dataset |
| `catbowl capture --bowl bowl1` | Record labelled crops from the mounted camera |
| `catbowl train` | Train the classifier, print a confusion matrix and a suggested threshold |
| `catbowl eval` | Score a trained classifier and list what it got wrong |
| `catbowl selftest` | Full pipeline on synthetic cameras and simulated lids |

## How the decisions are made

Three gates sit between a camera frame and a moving lid:

- **Confidence floor** (`recognition.min_confidence`). Any frame the classifier
  is not sure about becomes `unknown`.
- **Vote window** (`votes_required` of `vote_window`). `unknown` can never win,
  so a single odd frame cannot move anything.
- **Confirmation time** (`policy.open_confirm_s`). The right cat must be there
  continuously, not just passing through. This runs alongside the vote window
  rather than after it, so the delays overlap instead of stacking.

Closing is separately conservative: `close_delay_s` of a continuously empty bowl,
or `intruder_grace_s` of a wrong cat at an open bowl, or `max_open_s` as a
backstop. Everything unknown, stale or broken resolves to *closed*.

## Layout

```
catbowl/
  config.py       validated YAML schema - a typo fails at startup, not at 3am
  cameras.py      capture backends, one grabber thread per device, per-bowl ROIs
  detector.py     motion (cheap) or SSDLite COCO cat detection (selective)
  embedder.py     frozen backbone: torch, tflite, or a dependency-free fallback
  recognizer.py   classifier bundle, confidence floor, sliding vote window
  controller.py   per-bowl state machine (no hardware, no cameras, fully tested)
  actuators.py    slew-limited servo drivers: PCA9685, pigpio GPIO, mock
  app.py          worker threads, wiring, graceful shutdown
  status.py       the phone-friendly status page
  training.py     dataset building, training, threshold sweep
config/bowls.yaml every knob, commented
docs/             hardware, training, safety
tests/            85 tests, no hardware needed
```

## Known limits

- Two similar-looking cats are the hard case for any vision approach.
  [docs/training.md](docs/training.md#if-two-of-your-cats-look-nearly-identical)
  covers what to do, up to and including admitting that an RFID collar is what
  commercial feeders use.
- The Pi 4 runs about 5 inferences per second per bowl with `mobilenet_v3_small`
  and motion gating. `detector.type: ssdlite` is much more selective but costs
  roughly 10× more CPU; drop `loop_fps` to 2–3 if you switch.
- Nothing here stops a determined cat from physically shoving a closed lid aside.
  That is deliberate — see [docs/safety.md](docs/safety.md).
