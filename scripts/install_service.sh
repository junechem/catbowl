#!/usr/bin/env bash
# Install catbowl as a systemd service that starts itself at boot, and whenever
# a camera is plugged in afterwards.
#
# The shipped unit carries __USER__/__DIR__ placeholders rather than hard-coded
# paths, because the checkout is rarely at /home/pi/catbowl. This script fills
# them in from wherever it is being run.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

RUN_AS="${SUDO_USER:-$USER}"
UNIT=/etc/systemd/system/catbowl.service
RULE=/etc/udev/rules.d/99-catbowl.rules

if [[ ! -x .venv/bin/python ]]; then
    echo "no .venv here - run scripts/install_pi.sh first" >&2
    exit 1
fi

# Only ask for groups this system actually has: systemd refuses to start a unit
# that names a missing group, and a desktop Linux box has no gpio or i2c group.
GROUPS_WANTED=()
for group in video i2c gpio; do
    if getent group "$group" >/dev/null; then
        GROUPS_WANTED+=("$group")
        sudo usermod -aG "$group" "$RUN_AS"
    fi
done

echo "==> Installing $UNIT (user $RUN_AS, dir $HERE)"
sed -e "s|__USER__|$RUN_AS|g" \
    -e "s|__DIR__|$HERE|g" \
    -e "s|^SupplementaryGroups=.*|SupplementaryGroups=${GROUPS_WANTED[*]}|" \
    systemd/catbowl.service | sudo tee "$UNIT" >/dev/null

echo "==> Installing $RULE (starts catbowl when a camera appears)"
sudo cp systemd/99-catbowl.rules "$RULE"
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=video4linux --action=add

sudo systemctl daemon-reload
sudo systemctl enable catbowl.service
sudo systemctl restart catbowl.service || true

echo
if ls /dev/video* >/dev/null 2>&1; then
    systemctl --no-pager --lines=0 status catbowl.service || true
else
    echo "No camera is plugged in, so the service is idle by design."
    echo "Plug one in and udev will start it."
fi
cat <<'NEXT'

  journalctl -u catbowl -f          watch it run
  sudo systemctl stop catbowl       stop it now (it comes back at boot)
  sudo systemctl disable catbowl    stop it starting at boot

Note that the group changes above only take effect in a new login session -
the service itself picks them up immediately, but your own shell will not.
NEXT
