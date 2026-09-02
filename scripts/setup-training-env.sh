#!/usr/bin/env bash
# Build a virtual environment for dataset work and training.
#
#   ./scripts/setup-training-env.sh            -> .venv-train
#   ./scripts/setup-training-env.sh myenv      -> myenv
#
# A "virtual environment" is a private folder of Python packages belonging to
# this project only, so installing torch here cannot break anything else on the
# machine. Nothing is installed system-wide.
#
# Works the same on a workstation and on the Pi. It installs no hardware
# packages - see requirements-pi.txt for those.

set -euo pipefail

cd "$(dirname "$0")/.."
ENV_DIR="${1:-.venv-train}"

if [ -d "$ENV_DIR" ]; then
    echo "$ENV_DIR already exists. Delete it first, or pass a different name."
    exit 1
fi

echo "creating $ENV_DIR"
python3 -m venv "$ENV_DIR"

# A current pip; the torch wheels need one new enough to understand their tags.
"$ENV_DIR/bin/pip" install --upgrade pip

echo "installing training dependencies (torch is ~800 MB, this takes a while)"
"$ENV_DIR/bin/pip" install -r requirements-train.txt

# Install catbowl itself in editable mode, so `python -m catbowl` picks up your
# edits without reinstalling.
"$ENV_DIR/bin/pip" install -e .

echo
"$ENV_DIR/bin/python" - <<'PY'
import sklearn, torch, torchvision, cv2, numpy, yaml, joblib
print("torch       ", torch.__version__)
print("torchvision ", torchvision.__version__)
print("scikit-learn", sklearn.__version__)
print("opencv      ", cv2.__version__)
print("numpy       ", numpy.__version__)
PY

echo
echo "done. Use it with:"
echo "  $ENV_DIR/bin/python -m catbowl train"
echo
echo "or activate it for the session:"
echo "  source $ENV_DIR/bin/activate      # bash/zsh"
echo "  source $ENV_DIR/bin/activate.fish # fish"
