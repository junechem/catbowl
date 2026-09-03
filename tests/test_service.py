"""The shipped systemd unit.

The unit is the one piece of the rig that is never exercised until it is
running on the Pi at boot, where a mistake shows up as a restart loop and a
lid that stays shut. `-c/--config` is a top-level flag, so it has to come
before the subcommand; putting it after `run` is an `unrecognized arguments`
exit 2, over and over, five seconds apart. That has now happened twice, so it
is a test.
"""

import shlex
from pathlib import Path

import pytest

from catbowl.cli import build_parser

UNIT = Path(__file__).resolve().parent.parent / "systemd" / "catbowl.service"


def directives(name: str) -> list[str]:
    return [line.split("=", 1)[1].strip()
            for line in UNIT.read_text().splitlines()
            if line.startswith(f"{name}=")]


def test_the_unit_exists_and_starts_one_thing():
    assert UNIT.is_file()
    assert len(directives("ExecStart")) == 1


def test_the_unit_command_line_actually_parses():
    argv = shlex.split(directives("ExecStart")[0])
    assert argv[1:3] == ["-m", "catbowl"], f"expected `python -m catbowl ...`, got {argv}"

    # Everything after `-m catbowl` is exactly what argparse sees on the Pi.
    args = build_parser().parse_args(argv[3:])
    assert args.command == "run"
    assert args.config == "config/bowls.yaml"


def test_the_unit_waits_for_a_camera_instead_of_looping():
    assert directives("ConditionPathExistsGlob") == ["/dev/video*"]
    assert directives("Restart") == ["always"]


def test_the_torch_cache_is_writable():
    """ProtectHome=read-only made the first weight download a restart loop.

    torchvision writes the ssdlite weights to ~/.cache/torch the first time the
    detector is built, which under a read-only home is an OSError at startup,
    every five seconds, for ever.
    """
    paths = directives("ReadWritePaths")[0].split()
    assert "__DIR__" in paths
    assert any(p.lstrip("-").endswith("/.cache/torch") for p in paths), paths


@pytest.mark.parametrize("placeholder", ["__USER__", "__DIR__", "__HOME__"])
def test_the_placeholders_are_still_there_for_the_installer(placeholder):
    """install_service.sh substitutes these; a hard-coded path would ship my Pi."""
    assert placeholder in UNIT.read_text()
