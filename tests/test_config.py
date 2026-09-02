import pytest

from catbowl.config import ConfigError, build_config, load_config

BASE = {
    "actuator": {"driver": "pca9685"},
    "bowls": [
        {"id": "bowl1", "cat": "mochi", "camera": {"device": 0}, "servo": {"channel": 0}},
        {"id": "bowl2", "cat": "pepper", "camera": {"device": 2}, "servo": {"channel": 1}},
    ],
}


def cfg(**overrides):
    import copy

    data = copy.deepcopy(BASE)
    data.update(copy.deepcopy(overrides))
    return build_config(data)


def test_minimal_config_builds():
    app = cfg()
    assert [b.id for b in app.bowls] == ["bowl1", "bowl2"]
    assert app.cats == ["mochi", "pepper"]
    assert app.bowl("bowl2").servo.channel == 1


def test_bowl_defaults_are_merged_and_overridden():
    app = cfg(
        bowl_defaults={"servo": {"closed_deg": 5, "open_deg": 100}, "policy": {"close_delay_s": 4}},
        bowls=[
            {"id": "a", "cat": "x", "camera": {"device": 0}, "servo": {"channel": 0}},
            {"id": "b", "cat": "y", "camera": {"device": 2}, "servo": {"channel": 1, "open_deg": 120}},
        ],
    )
    assert app.bowl("a").servo.open_deg == 100
    assert app.bowl("b").servo.open_deg == 120     # per-bowl wins
    assert app.bowl("b").servo.closed_deg == 5     # default survives a partial override
    assert app.bowl("b").policy.close_delay_s == 4


def test_shipped_config_is_valid():
    app = load_config("config/bowls.yaml")
    assert len(app.bowls) == 3
    assert len(set(app.cats)) == 3


@pytest.mark.parametrize(
    "override, message",
    [
        ({"bowls": [{"id": "a", "cat": "x", "servo": {"channel": 0}},
                    {"id": "a", "cat": "y", "servo": {"channel": 1}}]}, "duplicate bowl id"),
        ({"bowls": [{"id": "a", "cat": "x", "servo": {"channel": 0}},
                    {"id": "b", "cat": "x", "servo": {"channel": 1}}]}, "more than one bowl"),
        ({"bowls": [{"id": "a", "cat": "x", "servo": {"channel": 0}},
                    {"id": "b", "cat": "y", "servo": {"channel": 0}}]}, "reuses servo channel"),
        ({"bowls": [{"id": "a", "cat": "x"}]}, "servo.channel is required"),
        ({"bowls": []}, "at least one bowl"),
        ({"bowls": [{"cat": "x", "servo": {"channel": 0}}]}, "missing required key 'id'"),
        ({"recognition": {"votes_required": 9, "vote_window": 4}}, "cannot exceed"),
        ({"recognition": {"backend": "magic"}}, "recognition.backend"),
        ({"recognition": {"min_confidence": 1.5}}, "min_confidence"),
        ({"actuator": {"driver": "pca9685", "min_pulse_us": 2600, "max_pulse_us": 2500}}, "pulse widths"),
        ({"bowls": [{"id": "a", "cat": "x", "servo": {"channel": 0, "closed_deg": 90, "open_deg": 92}}]},
         "5 degrees apart"),
        ({"bowls": [{"id": "a", "cat": "x", "servo": {"channel": 0},
                     "camera": {"roi": [0.5, 0, 0.8, 1.0]}}]}, "past the frame edge"),
        ({"bowls": [{"id": "a", "cat": "x", "servo": {"channel": 0},
                     "camera": {"rotate": 45}}]}, "rotate"),
        ({"nonsense": 1}, "unknown top-level key"),
        ({"recognition": {"nonsense": 1}}, "unknown key"),
    ],
)
def test_invalid_configs_are_rejected(override, message):
    with pytest.raises(ConfigError, match=message):
        cfg(**override)


def test_shared_camera_requires_per_bowl_roi():
    with pytest.raises(ConfigError, match="needs its own camera.roi"):
        cfg(bowls=[
            {"id": "a", "cat": "x", "camera": {"device": 0}, "servo": {"channel": 0}},
            {"id": "b", "cat": "y", "camera": {"device": 0}, "servo": {"channel": 1}},
        ])


def test_shared_camera_with_rois_is_allowed():
    app = cfg(bowls=[
        {"id": "a", "cat": "x", "camera": {"device": 0, "roi": [0.0, 0, 0.5, 1.0]}, "servo": {"channel": 0}},
        {"id": "b", "cat": "y", "camera": {"device": 0, "roi": [0.5, 0, 0.5, 1.0]}, "servo": {"channel": 1}},
    ])
    assert app.bowls[0].camera.key == app.bowls[1].camera.key   # one physical device


def test_gpio_driver_requires_pins():
    with pytest.raises(ConfigError, match="servo.gpio is required"):
        cfg(actuator={"driver": "gpio"})
    app = cfg(actuator={"driver": "gpio"},
              bowls=[{"id": "a", "cat": "x", "camera": {"device": 0}, "servo": {"gpio": 17}},
                     {"id": "b", "cat": "y", "camera": {"device": 2}, "servo": {"gpio": 27}}])
    assert app.bowl("b").servo.gpio == 27


def test_a_bowl_may_drive_its_lid_with_several_servos():
    app = build_config({
        "bowl_defaults": {"servo": {"detach_when_idle": False}},
        "bowls": [{
            "id": "a", "cat": "mochi",
            "servos": [
                {"channel": 0, "closed_deg": 10, "open_deg": 95},
                {"channel": 1, "closed_deg": 170, "open_deg": 85},
            ],
        }],
    })
    servos = app.bowl("a").servos
    assert [s.channel for s in servos] == [0, 1]
    assert [s.open_deg for s in servos] == [95, 85]
    assert all(s.detach_when_idle is False for s in servos), "defaults reach every servo"
    assert app.bowl("a").servo is servos[0], "bowl.servo still means the first one"


def test_two_servos_on_one_lid_cannot_share_a_channel():
    with pytest.raises(ConfigError, match="reuses servo channel"):
        build_config({"bowls": [{
            "id": "a", "cat": "mochi",
            "servos": [{"channel": 0}, {"channel": 0}],
        }]})


def test_a_second_bowl_cannot_steal_a_ganged_channel():
    with pytest.raises(ConfigError, match="reuses servo channel"):
        build_config({"bowls": [
            {"id": "a", "cat": "mochi", "servos": [{"channel": 0}, {"channel": 1}]},
            {"id": "b", "cat": "pepper", "servo": {"channel": 1}},
        ]})


def test_an_empty_servos_list_is_rejected():
    with pytest.raises(ConfigError, match="non-empty list"):
        build_config({"bowls": [{"id": "a", "cat": "mochi", "servos": []}]})
