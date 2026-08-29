#!/usr/bin/env bash
# Set up catbowl on a fresh Raspberry Pi OS (Bookworm) install.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

echo "==> System packages"
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev libatlas-base-dev i2c-tools v4l-utils

echo "==> Python environment"
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements-pi.txt

echo "==> I2C"
if ! ls /dev/i2c-* >/dev/null 2>&1; then
  echo "    I2C is not enabled. Run: sudo raspi-config -> Interface Options -> I2C"
else
  echo "    devices: $(ls /dev/i2c-*)"
  echo "    the PCA9685 should appear at 0x40 below:"
  sudo i2cdetect -y 1 || true
fi

echo "==> Cameras"
./.venv/bin/python -m catbowl cameras || true

echo "==> Checks"
./.venv/bin/python -m catbowl doctor || true

cat <<'NEXT'

Next steps:
  1. Edit config/bowls.yaml   - camera devices, servo channels, cat names
  2. .venv/bin/python -m catbowl calibrate --bowl bowl1   (repeat per bowl)
  3. Build a dataset and train - see docs/training.md
  4. .venv/bin/python -m catbowl run --dry-run            (watch, do not move lids)
  5. Install the service:
       sudo cp systemd/catbowl.service /etc/systemd/system/
       sudo systemctl daemon-reload && sudo systemctl enable --now catbowl
       journalctl -u catbowl -f
NEXT
