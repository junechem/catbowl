"""Command line entry point: ``python -m catbowl <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

log = logging.getLogger("catbowl")

DEFAULT_CONFIG = "config/bowls.yaml"


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-18s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #

def cmd_run(args) -> int:
    from .app import FeederApp
    from .config import load_config

    cfg = load_config(args.config)
    if args.dry_run:
        cfg.actuator.driver = "mock"
        log.info("dry run: lids are simulated, no PWM will be sent")
    if args.no_status:
        cfg.status_port = None
    if args.collect and not cfg.snapshot_dir:
        cfg.snapshot_dir = "data/collected"

    app = FeederApp(cfg, dry_run=args.no_model)

    def handle_signal(signum, _frame):
        log.info("caught %s", signal.Signals(signum).name)
        for worker in app.workers:
            worker.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handle_signal)

    app.run_forever()
    return 0


# --------------------------------------------------------------------------- #
# dataset commands
# --------------------------------------------------------------------------- #

def cmd_import(args) -> int:
    from .training import build_import_detector, import_photos

    detector = build_import_detector(args.detector)
    stats = import_photos(
        source=args.src,
        label=args.label,
        out_root=args.out,
        detector=detector,
        keep_uncropped=args.keep_uncropped,
    )
    print(f"\n{args.label}: {stats['saved']} crops written to {Path(args.out) / args.label}")
    for key in ("no_detection", "unreadable", "too_small"):
        if stats[key]:
            print(f"  {key.replace('_', ' ')}: {stats[key]}")
    if stats["no_detection"] > stats["total"] * 0.4:
        print("  Tip: many photos had no detectable cat. Try --keep-uncropped if the "
              "photos are already close-ups, or --detector none.")
    return 0


def cmd_capture(args) -> int:
    """Grab labelled crops from a live camera - the on-rig half of the dataset."""
    import cv2

    from .cameras import CameraHub
    from .config import CameraConfig, DetectorConfig, load_config
    from .detector import build_detector

    if args.bowl:
        cfg = load_config(args.config)
        bowl = cfg.bowl(args.bowl)
        camera_cfg, detector_cfg = bowl.camera, cfg.detector
        label = args.label or bowl.cat
    else:
        camera_cfg = CameraConfig(device=args.device)
        detector_cfg = DetectorConfig(type=args.detector)
        label = args.label
    if not label:
        print("error: --label is required when --bowl is not given", file=sys.stderr)
        return 2

    out_dir = Path(args.out) / label
    out_dir.mkdir(parents=True, exist_ok=True)
    hub = CameraHub()
    view = hub.view(camera_cfg)
    detector = build_detector(detector_cfg)

    print(f"Capturing '{label}' for {args.seconds:.0f}s into {out_dir} (Ctrl-C to stop early).")
    if detector_cfg.type == "motion":
        print("Hold still for a few seconds first - the motion detector is learning the empty scene.")

    saved = 0
    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline and saved < args.max_frames:
            frame = view.read(only_new=True)
            if frame is None:
                time.sleep(0.02)
                continue
            detection = detector.detect(frame.image)
            if detection is None:
                continue
            crop = detection.crop(frame.image, pad_frac=0.15)
            if min(crop.shape[:2]) < 64:
                continue
            name = f"{label}-{datetime.now():%Y%m%d-%H%M%S-%f}.jpg"
            cv2.imwrite(str(out_dir / name), crop)
            saved += 1
            print(f"\r  {saved} crops  ({int(deadline - time.monotonic())}s left) ", end="", flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        hub.close()
    print(f"\nsaved {saved} crops to {out_dir}")
    return 0


def cmd_train(args) -> int:
    from .config import RecognitionConfig, load_config
    from .training import format_report, train

    if Path(args.config).exists():
        recognition = load_config(args.config).recognition
    else:
        recognition = RecognitionConfig()
    if args.backend:
        recognition.backend = args.backend
    out_path = args.out or recognition.classifier

    bundle, metrics = train(
        data_root=args.data,
        recognition=recognition,
        out_path=out_path,
        test_size=args.test_size,
        augment=not args.no_augment,
        target_precision=args.target_precision,
    )
    print()
    print(format_report(metrics))
    print(f"\nsaved to {out_path}")
    if metrics["raw_accuracy"] < 0.9:
        print("\nAccuracy under 90%. Usually this means too few photos, or crops that are "
              "mostly background. Check a few files in your crops directory.")
    return 0


def cmd_eval(args) -> int:
    from .config import load_config
    from .recognizer import ClassifierBundle, Recognizer
    from .embedder import build_embedder
    from .training import load_dataset

    cfg = load_config(args.config)
    bundle = ClassifierBundle.load(args.model or cfg.recognition.classifier)
    recognizer = Recognizer(build_embedder(cfg.recognition), bundle, args.threshold or bundle.min_confidence)

    import cv2
    dataset = load_dataset(args.data)
    correct = rejected = wrong = 0
    mistakes: list[str] = []
    for path, truth in zip(dataset.paths, dataset.labels):
        image = cv2.imread(str(path))
        if image is None:
            continue
        prediction = recognizer.predict(image)
        if not prediction.is_known:
            rejected += 1
        elif prediction.label == truth:
            correct += 1
        else:
            wrong += 1
            if len(mistakes) < 15:
                mistakes.append(f"  {path.name}: {truth} -> {prediction.label} ({prediction.confidence:.2f})")

    total = correct + rejected + wrong
    print(f"threshold {recognizer.min_confidence:.2f} over {total} images")
    print(f"  correct : {correct:5d}  ({correct / total:.1%})")
    print(f"  wrong   : {wrong:5d}  ({wrong / total:.1%})   <- these open the wrong lid")
    print(f"  rejected: {rejected:5d}  ({rejected / total:.1%})   <- bowl stays shut, cat waits")
    if mistakes:
        print("\nmisclassified:")
        print("\n".join(mistakes))
    return 0


# --------------------------------------------------------------------------- #
# hardware commands
# --------------------------------------------------------------------------- #

def cmd_calibrate(args) -> int:
    """Find the two angles that mean 'lid down' and 'lid clear of the bowl'."""
    from .actuators import ActuatorFactory
    from .config import load_config

    cfg = load_config(args.config)
    bowl = cfg.bowl(args.bowl)
    factory = ActuatorFactory(cfg.actuator)
    actuator = factory.create(bowl.id, bowl.servo)

    print(f"Calibrating {bowl.id} ({bowl.cat}).")
    print(f"  current: closed={bowl.servo.closed_deg}  open={bowl.servo.open_deg}")
    print("Type an angle in degrees to move there, 'o'/'c' to test the configured")
    print("positions, or 'q' to quit. Keep fingers and cats clear.\n")
    try:
        while True:
            try:
                raw = input("angle> ").strip().lower()
            except EOFError:
                break
            if raw in ("q", "quit", "exit"):
                break
            if raw == "o":
                actuator.open()
                continue
            if raw == "c":
                actuator.close()
                continue
            try:
                degrees = float(raw)
            except ValueError:
                print("  enter a number, or o/c/q")
                continue
            span = bowl.servo.open_deg - bowl.servo.closed_deg
            fraction = (degrees - bowl.servo.closed_deg) / span if span else 0.0
            actuator.move_to(min(1.0, max(0.0, fraction)))
            print(f"  moved to {actuator.angle_for(actuator.position):.1f} deg "
                  f"(fraction {actuator.position:.2f})")
    finally:
        factory.shutdown()
    print("\nPut the angles you settled on into config/bowls.yaml as "
          "servo.closed_deg / servo.open_deg.")
    return 0


def cmd_cameras(args) -> int:
    """Probe every /dev/video* and report which ones actually deliver frames."""
    import cv2

    devices = sorted(Path("/dev").glob("video*"))
    if not devices:
        print("no /dev/video* devices found")
        return 1
    print(f"{'device':<14}{'opens':<8}{'resolution':<14}fps")
    for device in devices:
        index = int(str(device).rsplit("video", 1)[1])
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"{str(device):<14}{'no':<8}")
            cap.release()
            continue
        ok, frame = cap.read()
        shape = f"{frame.shape[1]}x{frame.shape[0]}" if ok and frame is not None else "-"
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"{str(device):<14}{'yes':<8}{shape:<14}{fps:.0f}")
        cap.release()
    print("\nUse the number from /dev/videoN as camera.device in the config.")
    return 0


def cmd_doctor(args) -> int:
    """Check that everything this rig needs is actually present."""
    ok = True

    def check(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        print(f"  [{'ok' if passed else 'XX'}] {name}{'  - ' + detail if detail else ''}")

    print("python packages")
    for module, why in (("cv2", "camera and image handling"),
                        ("numpy", "arrays"),
                        ("sklearn", "training"),
                        ("yaml", "config")):
        try:
            __import__(module)
            check(module, True)
        except ImportError as exc:
            check(module, False, str(exc))

    backends = []
    for module in ("torch", "ai_edge_litert", "tflite_runtime"):
        try:
            __import__(module)
            backends.append(module)
        except ImportError:
            pass
    check("an inference backend", bool(backends), ", ".join(backends) or "none of torch/litert found")

    print("\nconfig")
    cfg = None
    try:
        from .config import load_config

        cfg = load_config(args.config)
        check(f"{args.config} parses", True, f"{len(cfg.bowls)} bowls: " +
              ", ".join(f"{b.id}={b.cat}" for b in cfg.bowls))
    except Exception as exc:
        check(f"{args.config} parses", False, str(exc))

    print("\nmodel")
    if cfg:
        model_path = Path(cfg.recognition.classifier)
        if model_path.exists():
            try:
                from .recognizer import ClassifierBundle

                bundle = ClassifierBundle.load(model_path)
                check("classifier loads", True,
                      f"labels: {', '.join(bundle.labels)}, threshold {bundle.min_confidence:.2f}")
                missing = [b.cat for b in cfg.bowls if b.cat not in bundle.labels]
                check("every bowl's cat is a known label", not missing,
                      f"missing from the model: {', '.join(missing)}" if missing else "")
                check("backbone matches config", bundle.embedder.backend == cfg.recognition.backend,
                      f"model={bundle.embedder.backend} config={cfg.recognition.backend}")
            except Exception as exc:
                check("classifier loads", False, str(exc))
        else:
            check(f"{model_path} exists", False, "run 'catbowl train' first")

    print("\nhardware")
    video = sorted(Path("/dev").glob("video*"))
    check("camera device present", bool(video), ", ".join(p.name for p in video) or "no /dev/video*")
    if cfg and cfg.actuator.driver == "pca9685":
        i2c = sorted(Path("/dev").glob("i2c-*"))
        check("i2c enabled", bool(i2c), ", ".join(p.name for p in i2c) or
              "enable it with raspi-config -> Interface Options -> I2C")
        if shutil.which("i2cdetect"):
            print("      (run 'i2cdetect -y 1' - the PCA9685 should show at "
                  f"0x{cfg.actuator.i2c_address:02x})")
        try:
            import adafruit_servokit  # noqa: F401
            check("adafruit-circuitpython-servokit installed", True)
        except ImportError:
            check("adafruit-circuitpython-servokit installed", False, "pip install it on the Pi")
    elif cfg and cfg.actuator.driver == "gpio":
        try:
            import pigpio

            pi = pigpio.pi()
            check("pigpiod reachable", pi.connected,
                  "" if pi.connected else "sudo systemctl enable --now pigpiod")
            if pi.connected:
                pi.stop()
        except ImportError:
            check("pigpio installed", False, "pip install pigpio")

    print("\n" + ("all checks passed" if ok else "some checks failed - see above"))
    return 0 if ok else 1


def cmd_selftest(args) -> int:
    """End-to-end run on synthetic cameras and simulated lids. No hardware."""
    import tempfile

    import cv2
    import numpy as np

    from .app import FeederApp
    from .cameras import SyntheticCapture
    from .config import DetectorConfig, build_config
    from .detector import build_detector
    from .training import train

    workdir = Path(tempfile.mkdtemp(prefix="catbowl-selftest-"))
    crops = workdir / "crops"
    cats = ["alpha", "bravo", "charlie"]
    print(f"selftest workspace: {workdir}")

    # 1. Fabricate a dataset: one distinctly coloured blob per "cat".
    for variant, cat in enumerate(cats):
        (crops / cat).mkdir(parents=True)
        camera = SyntheticCapture(320, 240, fps=1000, variant=variant)
        # Crop through the same detector the runtime uses, so training and
        # inference see the same kind of image.
        detector = build_detector(DetectorConfig(warmup_frames=3, min_area_frac=0.01))
        saved = 0
        for _ in range(4000):
            if saved >= 30:
                break
            frame = camera.read()
            detection = detector.detect(frame)
            if detection is None:
                continue
            crop = detection.crop(frame, pad_frac=0.15)
            if min(crop.shape[:2]) < 16:
                continue
            noisy = np.clip(crop.astype(np.int16) +
                            np.random.default_rng(saved).integers(-12, 12, crop.shape), 0, 255)
            cv2.imwrite(str(crops / cat / f"{saved:03d}.jpg"), noisy.astype(np.uint8))
            saved += 1
    print(f"  built {len(cats) * 30} synthetic crops")

    # 2. Train the mock backbone on it.
    from .config import RecognitionConfig

    recognition = RecognitionConfig(backend="mock", classifier=str(workdir / "classifier.joblib"))
    _, metrics = train(crops, recognition, out_path=recognition.classifier, augment=False)
    print(f"  trained: accuracy {metrics['raw_accuracy']:.0%}, "
          f"threshold {metrics['suggested_threshold']:.2f}")
    if metrics["raw_accuracy"] < 0.95:
        print("  FAIL: the synthetic classes should be trivially separable")
        return 1

    # 3. Run the full app against synthetic cameras and mock lids.
    cfg = build_config({
        "recognition": {"backend": "mock", "classifier": recognition.classifier,
                        "min_confidence": 0.6, "vote_window": 4, "votes_required": 3},
        "detector": {"type": "motion", "warmup_frames": 5, "min_area_frac": 0.01},
        "actuator": {"driver": "mock"},
        "loop_fps": 10,
        "status_port": None,
        "log_dir": str(workdir / "logs"),
        "bowls": [
            {"id": f"bowl{i + 1}", "cat": cat,
             "camera": {"device": f"synthetic:{i}", "width": 320, "height": 240, "fps": 20},
             "servo": {"channel": i},
             "policy": {"open_confirm_s": 0.3, "close_delay_s": 1.0, "cooldown_s": 0.5,
                        "max_open_s": 30}}
            for i, cat in enumerate(cats)
        ],
    })

    app = FeederApp(cfg)
    app.build()
    app.start()
    print(f"  running for {args.seconds}s ...")
    try:
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            time.sleep(0.5)
    finally:
        app.stop()

    opened = {w.cfg.id: w.controller.stats["opens"] for w in app.workers}
    print("\n  opens per bowl:", json.dumps(opened))
    for worker in app.workers:
        print(f"    {worker.cfg.id}: {worker.frames} frames, "
              f"{worker.inferences} inferences, error={worker.last_error}")

    if not all(count > 0 for count in opened.values()):
        print("\nFAIL: every bowl should have opened for its own cat at least once")
        return 1
    if any(worker.last_error for worker in app.workers):
        print("\nFAIL: a worker reported an error")
        return 1
    print("\nPASS: detection, recognition, voting, state machine and lid control all ran")
    if not args.keep:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="catbowl", description=__doc__)
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG, help="path to bowls.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run the feeder")
    p.add_argument("--dry-run", action="store_true", help="simulate the lids, touch no hardware")
    p.add_argument("--no-model", action="store_true", help="detection only, skip recognition")
    p.add_argument("--no-status", action="store_true", help="disable the status web page")
    p.add_argument("--collect", action="store_true", help="save crops of every decision for retraining")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("import", help="import phone photos into the crops directory")
    p.add_argument("--src", required=True, help="folder of photos of one cat")
    p.add_argument("--label", required=True, help="that cat's name")
    p.add_argument("--out", default="data/crops")
    p.add_argument("--detector", default="ssdlite", choices=["ssdlite", "none"])
    p.add_argument("--keep-uncropped", action="store_true",
                   help="keep photos where no cat was detected")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("capture", help="capture labelled crops from a live camera")
    p.add_argument("--bowl", help="take the camera settings from this bowl")
    p.add_argument("--device", default=0, help="camera device, if --bowl is not used")
    p.add_argument("--label", help="cat name (defaults to the bowl's cat)")
    p.add_argument("--out", default="data/crops")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--interval", type=float, default=0.4, help="seconds between saved crops")
    p.add_argument("--max-frames", type=int, default=400)
    p.add_argument("--detector", default="motion", choices=["motion", "ssdlite", "none"])
    p.set_defaults(func=cmd_capture)

    p = sub.add_parser("train", help="train the classifier on data/crops")
    p.add_argument("--data", default="data/crops")
    p.add_argument("--out", help="where to write the classifier (default: from config)")
    p.add_argument("--backend", choices=["torch", "tflite", "mock"])
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--no-augment", action="store_true", help="skip mirrored copies")
    p.add_argument("--target-precision", type=float, default=0.99,
                   help="precision the suggested threshold should hit")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval", help="score a trained classifier against a folder of crops")
    p.add_argument("--data", default="data/crops")
    p.add_argument("--model")
    p.add_argument("--threshold", type=float)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("calibrate", help="interactively find the servo end positions")
    p.add_argument("--bowl", required=True)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("cameras", help="list and probe attached cameras")
    p.set_defaults(func=cmd_cameras)

    p = sub.add_parser("doctor", help="check dependencies, config, model and hardware")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("selftest", help="end-to-end run with no hardware at all")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--keep", action="store_true", help="keep the temporary workspace")
    p.set_defaults(func=cmd_selftest)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        if args.verbose:
            raise
        log.error("%s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
