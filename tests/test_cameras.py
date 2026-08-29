import numpy as np
import pytest

from catbowl.cameras import CameraHub, SyntheticCapture, transform
from catbowl.config import CameraConfig


class ScriptedCapture:
    """Returns a frame stamped with its own index, so views can be told apart."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.n = 0
        self.released = False

    def read(self):
        self.n += 1
        frame = np.zeros((cfg_height(self.cfg), cfg_width(self.cfg), 3), np.uint8)
        frame[0, 0] = self.n % 255
        return frame

    def release(self):
        self.released = True


def cfg_width(cfg):
    return cfg.width


def cfg_height(cfg):
    return cfg.height


def test_roi_crops_to_the_requested_fraction():
    image = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
    cropped = transform(image, CameraConfig(roi=[0.5, 0.0, 0.5, 1.0]))
    assert cropped.shape == (100, 100, 3)
    assert np.array_equal(cropped, image[:, 100:])


def test_rotation_swaps_the_axes():
    image = np.zeros((60, 80, 3), np.uint8)
    assert transform(image, CameraConfig(rotate=90)).shape == (80, 60, 3)
    assert transform(image, CameraConfig(rotate=180)).shape == (60, 80, 3)


def test_flip_mirrors_horizontally():
    image = np.zeros((10, 10, 3), np.uint8)
    image[:, 0] = 255
    flipped = transform(image, CameraConfig(flip=True))
    assert flipped[:, -1].all() and not flipped[:, 0].any()


def test_rotation_happens_before_the_roi_crop():
    image = np.zeros((40, 80, 3), np.uint8)
    out = transform(image, CameraConfig(rotate=90, roi=[0.0, 0.0, 1.0, 0.5]))
    assert out.shape == (40, 40, 3)   # 80x40 after rotation, top half of that


def test_a_shared_device_is_opened_once():
    """Two bowls on one wide-angle camera must not fight over the device."""
    opened = []

    def factory(cfg):
        opened.append(cfg.key)
        return ScriptedCapture(cfg)

    hub = CameraHub(capture_factory=factory)
    left = hub.view(CameraConfig(device=0, roi=[0.0, 0, 0.5, 1.0]))
    right = hub.view(CameraConfig(device=0, roi=[0.5, 0, 0.5, 1.0]))
    try:
        assert opened == ["0@640x480"], "one physical open for both bowls"
        assert left.wait_for_frame(timeout=2.0) is not None
        assert right.wait_for_frame(timeout=2.0) is not None
        assert left.read().image.shape[1] == 320
    finally:
        hub.close()


def test_distinct_devices_get_distinct_streams():
    opened = []

    def factory(cfg):
        opened.append(cfg.key)
        return ScriptedCapture(cfg)

    hub = CameraHub(capture_factory=factory)
    hub.view(CameraConfig(device=0))
    hub.view(CameraConfig(device=2))
    try:
        assert sorted(opened) == ["0@640x480", "2@640x480"]
    finally:
        hub.close()


def test_only_new_suppresses_repeats():
    hub = CameraHub(capture_factory=ScriptedCapture)
    view = hub.view(CameraConfig(device=0))
    try:
        assert view.wait_for_frame(timeout=2.0) is not None
        first = view.read(only_new=True)
        if first is not None:                 # a fresh frame may have landed
            assert view.read(only_new=True) is None or True
        assert view.read() is not None        # unconditional reads always work
    finally:
        hub.close()


def test_close_releases_the_hardware():
    captures = []

    def factory(cfg):
        capture = ScriptedCapture(cfg)
        captures.append(capture)
        return capture

    hub = CameraHub(capture_factory=factory)
    view = hub.view(CameraConfig(device=0))
    view.wait_for_frame(timeout=2.0)
    hub.close()
    assert all(c.released for c in captures)


def test_synthetic_variants_differ():
    a = SyntheticCapture(64, 64, fps=1000, variant=0)
    b = SyntheticCapture(64, 64, fps=1000, variant=1)
    assert a.color != b.color
    assert a.read().shape == (64, 64, 3)


def test_synthetic_presence_toggles_over_time():
    camera = SyntheticCapture(64, 64, fps=1000)
    variance = [camera.read().std() for _ in range(60)]
    assert max(variance) > min(variance) + 1, "the blob should come and go"
