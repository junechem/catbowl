"""Configuration schema for the feeder.

Everything the rig needs to know lives in one YAML file (see config/bowls.yaml).
Loading it produces validated dataclasses so that a typo in the config fails at
start-up with a clear message rather than halfway through the night.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when the YAML file is structurally valid but semantically wrong."""


@dataclass
class CameraConfig:
    device: Any = 0            # 0/1/2 (V4L2 index), "csi:0", "file:clip.mp4", "synthetic"
    width: int = 640
    height: int = 480
    fps: int = 10
    roi: list[float] | None = None   # [x, y, w, h] as fractions of the frame, or None
    rotate: int = 0                  # 0/90/180/270, applied before the ROI crop
    flip: bool = False               # mirror horizontally

    def __post_init__(self) -> None:
        if self.rotate not in (0, 90, 180, 270):
            raise ConfigError(f"camera.rotate must be 0/90/180/270, got {self.rotate!r}")
        if self.roi is not None:
            if len(self.roi) != 4:
                raise ConfigError("camera.roi must be [x, y, w, h] as fractions of the frame")
            x, y, w, h = self.roi
            if not all(0.0 <= v <= 1.0 for v in self.roi) or w <= 0 or h <= 0:
                raise ConfigError(f"camera.roi values must be in 0..1 with w,h > 0, got {self.roi}")
            if x + w > 1.0001 or y + h > 1.0001:
                raise ConfigError(f"camera.roi extends past the frame edge: {self.roi}")

    @property
    def key(self) -> str:
        """Identity of the underlying capture device, shared between bowls."""
        return f"{self.device}@{self.width}x{self.height}"


@dataclass
class ServoConfig:
    channel: int | None = None   # PCA9685 channel
    gpio: int | None = None      # BCM pin, when driving the servo straight off the Pi
    closed_deg: float = 10.0
    open_deg: float = 95.0
    detach_when_idle: bool = True   # stop the PWM once the lid has settled

    def __post_init__(self) -> None:
        for name in ("closed_deg", "open_deg"):
            value = getattr(self, name)
            if not 0.0 <= value <= 180.0:
                raise ConfigError(f"servo.{name} must be 0..180, got {value}")
        if abs(self.open_deg - self.closed_deg) < 5.0:
            raise ConfigError("servo.open_deg and servo.closed_deg are less than 5 degrees apart")


@dataclass
class PolicyConfig:
    open_confirm_s: float = 0.8    # how long the right cat must be seen before the lid lifts
    close_delay_s: float = 10.0    # how long the bowl must be empty before the lid drops
    max_open_s: float = 900.0      # hard ceiling on a single sitting
    # After max_open_s ends a sitting, the bowl must look empty for this long
    # before it will open again. Without it a cat that never steps back simply
    # gets the lid returned a cooldown later, which defeats the ceiling: the
    # point of the limit is to break the meal into portions, and that only
    # works if the cat has to walk away and come back for the next one.
    rearm_absent_s: float = 5.0
    close_on_intruder: bool = True
    intruder_grace_s: float = 2.0  # a wrong cat must linger this long before we close
    cooldown_s: float = 3.0        # dead time after closing, stops the lid oscillating

    def __post_init__(self) -> None:
        for name in ("open_confirm_s", "close_delay_s", "max_open_s",
                     "rearm_absent_s", "intruder_grace_s", "cooldown_s"):
            if getattr(self, name) < 0:
                raise ConfigError(f"policy.{name} must not be negative")
        if self.max_open_s and self.max_open_s < self.close_delay_s:
            raise ConfigError("policy.max_open_s must be larger than policy.close_delay_s")


@dataclass
class BowlConfig:
    id: str
    cat: str
    camera: CameraConfig = field(default_factory=CameraConfig)
    # A lid may be driven by more than one servo - a heavy or wide lid usually
    # wants one on each hinge. They are ganged: every servo in this list is
    # slewed in lockstep. Mirrored servos face opposite ways, so each carries
    # its own closed_deg/open_deg rather than sharing one pair with a sign flip.
    servos: list[ServoConfig] = field(default_factory=lambda: [ServoConfig()])
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    enabled: bool = True

    @property
    def servo(self) -> ServoConfig:
        """The first servo. Kept so single-servo call sites read unchanged."""
        return self.servos[0]


@dataclass
class RecognitionConfig:
    backend: str = "torch"                 # torch | tflite | mock
    model: str = "mobilenet_v3_small"
    tflite_model_path: str | None = None
    classifier: str = "models/classifier.joblib"
    input_size: int = 224
    min_confidence: float = 0.75
    vote_window: int = 6                   # frames kept in the sliding window
    votes_required: int = 4                # agreeing frames needed for a decision

    def __post_init__(self) -> None:
        if self.backend not in ("torch", "tflite", "mock"):
            raise ConfigError(f"recognition.backend must be torch/tflite/mock, got {self.backend!r}")
        if not 0.0 < self.min_confidence <= 1.0:
            raise ConfigError("recognition.min_confidence must be in (0, 1]")
        if self.votes_required > self.vote_window:
            raise ConfigError("recognition.votes_required cannot exceed recognition.vote_window")
        if self.votes_required < 1:
            raise ConfigError("recognition.votes_required must be at least 1")


@dataclass
class DetectorConfig:
    type: str = "hybrid"          # hybrid | motion | ssdlite | none
    min_area_frac: float = 0.02   # ignore blobs smaller than this share of the frame
    score_threshold: float = 0.5  # ssdlite and hybrid only
    pad_frac: float = 0.15        # grow the box before cropping, to catch ears/whiskers
    warmup_frames: int = 30       # motion only: frames spent learning the empty scene
    # hybrid only: how long a "yes, that is a cat" answer stays good before the
    # expensive detector is asked again, and how long a "no" suppresses it.
    confirm_every_s: float = 2.0
    reject_backoff_s: float = 1.0
    # Once a visit has started, how long ssdlite may keep failing before the
    # visit is declared over. An eating cat is head-down in a bowl and stops
    # looking like a cat to a COCO detector for long stretches; without this
    # the gate revokes it mid-meal and the lid shuts on the animal.
    confirm_grace_s: float = 25.0
    # How long motion must be absent before a visit ends. Covers the moment a
    # settled cat stops moving enough for background subtraction to notice.
    visit_gap_s: float = 2.0

    def __post_init__(self) -> None:
        if self.type not in ("hybrid", "motion", "ssdlite", "none"):
            raise ConfigError(
                f"detector.type must be hybrid/motion/ssdlite/none, got {self.type!r}"
            )
        for name in ("confirm_every_s", "reject_backoff_s",
                     "confirm_grace_s", "visit_gap_s"):
            if getattr(self, name) < 0:
                raise ConfigError(f"detector.{name} must not be negative")


@dataclass
class CaptureConfig:
    """Bank photographs of whatever the detector finds, for later training.

    These land unsorted in one folder: the rig has no reliable idea which cat it
    is looking at until a classifier exists, and guessing would poison the very
    dataset being collected. Sort them into per-cat folders by hand, then point
    `catbowl train` at that.
    """

    dir: str | None = None          # None disables capture entirely
    interval_s: float = 2.0         # seconds between saved images, per bowl
    max_images: int = 5000          # stop once the folder holds this many; a Pi's SD card is small
    save_frame: bool = False        # also save the whole frame beside the crop
    # The buckets offered by the sorting page at /sort. Each becomes a folder
    # under `dir`, so they have to be usable as directory names. "M" is the
    # more-than-one-cat pile: not useful for training a single-cat classifier,
    # but kept, because a frame with two cats in it is exactly what a future
    # model will need to learn to refuse.
    labels: list[str] = field(default_factory=lambda: ["J", "K", "F", "M"])

    def __post_init__(self) -> None:
        if self.interval_s < 0:
            raise ConfigError("capture.interval_s must not be negative")
        if self.max_images < 0:
            raise ConfigError("capture.max_images must not be negative")
        if not self.labels:
            raise ConfigError("capture.labels must not be empty")
        self.labels = [str(label) for label in self.labels]
        for label in self.labels:
            if not label or not all(c.isalnum() or c in "_-" for c in label):
                raise ConfigError(
                    f"capture.labels entries must be letters, digits, - or _ "
                    f"(they become folder names), got {label!r}"
                )
        if len(set(self.labels)) != len(self.labels):
            raise ConfigError("capture.labels contains a duplicate")


@dataclass
class ActuatorConfig:
    driver: str = "pca9685"       # pca9685 | gpio | mock
    i2c_bus: int = 1              # /dev/i2c-1, the Pi's header bus
    i2c_address: int = 0x40
    frequency: int = 50           # Hz, standard analogue-servo frame rate
    move_speed_deg_s: float = 120.0   # slew limit; keeps the lid from slamming
    step_deg: float = 3.0
    min_pulse_us: int = 500           # pulse width at 0 deg (check your servo's datasheet)
    max_pulse_us: int = 2500          # pulse width at 180 deg

    def __post_init__(self) -> None:
        if self.driver not in ("pca9685", "gpio", "mock"):
            raise ConfigError(f"actuator.driver must be pca9685/gpio/mock, got {self.driver!r}")
        if self.move_speed_deg_s <= 0 or self.step_deg <= 0:
            raise ConfigError("actuator.move_speed_deg_s and actuator.step_deg must be positive")
        if not 200 <= self.min_pulse_us < self.max_pulse_us <= 3000:
            raise ConfigError("actuator pulse widths must satisfy 200 <= min < max <= 3000 (microseconds)")


@dataclass
class AppConfig:
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    actuator: ActuatorConfig = field(default_factory=ActuatorConfig)
    bowls: list[BowlConfig] = field(default_factory=list)
    loop_fps: float = 5.0
    log_dir: str = "logs"
    status_port: int | None = 8080
    snapshot_dir: str | None = None   # if set, save the crop behind every decision
    capture: CaptureConfig = field(default_factory=CaptureConfig)

    def bowl(self, bowl_id: str) -> BowlConfig:
        for bowl in self.bowls:
            if bowl.id == bowl_id:
                return bowl
        raise KeyError(bowl_id)

    @property
    def cats(self) -> list[str]:
        return [bowl.cat for bowl in self.bowls]


def _build(cls, data: dict | None, context: str):
    data = dict(data or {})
    known = {f.name for f in cls.__dataclass_fields__.values()}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"unknown key(s) in {context}: {', '.join(sorted(unknown))}")
    try:
        return cls(**data)
    except TypeError as exc:
        raise ConfigError(f"{context}: {exc}") from exc


def load_config(path: str | Path) -> AppConfig:
    """Read and validate the YAML config at *path*."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")
    return build_config(raw)


def build_config(raw: dict) -> AppConfig:
    raw = copy.deepcopy(raw)
    defaults = raw.pop("bowl_defaults", {}) or {}

    bowls_raw = raw.pop("bowls", []) or []
    if not bowls_raw:
        raise ConfigError("config must define at least one bowl")

    bowls: list[BowlConfig] = []
    for index, entry in enumerate(bowls_raw):
        if not isinstance(entry, dict):
            raise ConfigError(f"bowls[{index}] must be a mapping")
        entry = _merge(defaults, entry)
        context = f"bowls[{index}]"
        for required in ("id", "cat"):
            if not entry.get(required):
                raise ConfigError(f"{context} is missing required key '{required}'")
        bowls.append(
            BowlConfig(
                id=str(entry["id"]),
                cat=str(entry["cat"]),
                enabled=bool(entry.get("enabled", True)),
                camera=_build(CameraConfig, entry.get("camera"), f"{context}.camera"),
                servos=_build_servos(entry, context),
                policy=_build(PolicyConfig, entry.get("policy"), f"{context}.policy"),
            )
        )

    app = AppConfig(
        recognition=_build(RecognitionConfig, raw.pop("recognition", None), "recognition"),
        detector=_build(DetectorConfig, raw.pop("detector", None), "detector"),
        actuator=_build(ActuatorConfig, raw.pop("actuator", None), "actuator"),
        bowls=bowls,
        loop_fps=float(raw.pop("loop_fps", 5.0)),
        log_dir=str(raw.pop("log_dir", "logs")),
        status_port=raw.pop("status_port", 8080),
        snapshot_dir=raw.pop("snapshot_dir", None),
        capture=_build(CaptureConfig, raw.pop("capture", None), "capture"),
    )
    if raw:
        raise ConfigError(f"unknown top-level key(s): {', '.join(sorted(raw))}")
    _validate(app)
    return app


def _build_servos(entry: dict, context: str) -> list[ServoConfig]:
    """Build a bowl's servo list from either `servo:` or `servos:`.

    `servo:` is one mapping and stays the way single-servo bowls are written.
    `servos:` is a list of them, for a lid driven from both hinges. Each entry
    inherits bowl_defaults.servo, so a mirrored pair only has to spell out what
    actually differs - usually just the channel and the two angles.
    """
    if entry.get("servos") is not None:
        raw_list = entry["servos"]
        if not isinstance(raw_list, list) or not raw_list:
            raise ConfigError(f"{context}.servos must be a non-empty list")
        base = entry.get("servo") or {}
        servos = []
        for i, item in enumerate(raw_list):
            if not isinstance(item, dict):
                raise ConfigError(f"{context}.servos[{i}] must be a mapping")
            servos.append(_build(ServoConfig, {**base, **item}, f"{context}.servos[{i}]"))
        return servos
    return [_build(ServoConfig, entry.get("servo"), f"{context}.servo")]


def _merge(base: dict, override: dict) -> dict:
    """Shallow-merge per-bowl overrides onto bowl_defaults, one level deep."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **value}
        else:
            out[key] = value
    return out


def _validate(app: AppConfig) -> None:
    if app.loop_fps <= 0:
        raise ConfigError("loop_fps must be positive")

    seen_ids: set[str] = set()
    seen_cats: set[str] = set()
    seen_channels: dict[int, str] = {}
    seen_gpios: dict[int, str] = {}
    for bowl in app.bowls:
        if bowl.id in seen_ids:
            raise ConfigError(f"duplicate bowl id {bowl.id!r}")
        seen_ids.add(bowl.id)
        if bowl.cat in seen_cats:
            raise ConfigError(f"cat {bowl.cat!r} is assigned to more than one bowl")
        seen_cats.add(bowl.cat)

        # Every servo on the lid gets checked, not just the first, so a typo in
        # the second hinge fails at start-up instead of at 3am.
        for servo in bowl.servos:
            if app.actuator.driver == "pca9685":
                if servo.channel is None:
                    raise ConfigError(f"bowl {bowl.id!r}: servo.channel is required for the pca9685 driver")
                if not 0 <= servo.channel <= 15:
                    raise ConfigError(f"bowl {bowl.id!r}: servo.channel must be 0..15")
                if servo.channel in seen_channels:
                    raise ConfigError(
                        f"bowl {bowl.id!r} reuses servo channel {servo.channel} "
                        f"(already used by {seen_channels[servo.channel]!r})"
                    )
                seen_channels[servo.channel] = bowl.id
            elif app.actuator.driver == "gpio":
                if servo.gpio is None:
                    raise ConfigError(f"bowl {bowl.id!r}: servo.gpio is required for the gpio driver")
                if servo.gpio in seen_gpios:
                    raise ConfigError(
                        f"bowl {bowl.id!r} reuses GPIO {servo.gpio} "
                        f"(already used by {seen_gpios[servo.gpio]!r})"
                    )
                seen_gpios[servo.gpio] = bowl.id

    # Two bowls may share a camera, but only if each carves out its own ROI.
    by_camera: dict[str, list[BowlConfig]] = {}
    for bowl in app.bowls:
        by_camera.setdefault(bowl.camera.key, []).append(bowl)
    for key, group in by_camera.items():
        if len(group) > 1 and any(b.camera.roi is None for b in group):
            names = ", ".join(b.id for b in group)
            raise ConfigError(
                f"bowls {names} share camera {key} so each one needs its own camera.roi"
            )
