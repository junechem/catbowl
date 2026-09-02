#!/usr/bin/env bash
# Set up catbowl on a fresh Raspberry Pi OS install (Bookworm or Trixie).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

echo "==> System packages"
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev i2c-tools v4l-utils

# A tuned BLAS for numpy. Trixie (Raspberry Pi OS 13) dropped ATLAS entirely,
# so prefer OpenBLAS and keep the ATLAS line only for older images. Neither is
# fatal: numpy's own ARM wheels ship a bundled OpenBLAS, so this is a speed-up
# rather than a dependency.
if ! sudo apt-get install -y libopenblas-dev 2>/dev/null; then
    sudo apt-get install -y libatlas-base-dev 2>/dev/null || \
        echo "    no system BLAS installed - numpy will use its bundled one"
fi

echo "==> Python environment"
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip

# torch first, and explicitly from PyTorch's CPU index.
#
# PyPI's aarch64 torch wheel is now built for NVIDIA's ARM machines (GH200 and
# friends), so plain `pip install torch` on a Pi drags in ~2 GB of CUDA runtime
# - a single cuDNN wheel is 445 MB - for a board with no NVIDIA GPU at all.
# Downloads that size regularly fail on a Pi's network, which is what the
# "incomplete-download" error is. The +cpu build has no nvidia dependencies.
#
# It has to happen before requirements-pi.txt: torch is already satisfied by
# then, so the `torch>=2.2` line there will not go back to PyPI for it.
echo "    torch (CPU build - PyPI's aarch64 wheel would pull ~2 GB of CUDA)"
./.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

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
