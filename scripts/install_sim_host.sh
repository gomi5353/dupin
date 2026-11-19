#!/usr/bin/env bash
# Install the simulator stack (Isaac Sim 5.1.0, Isaac Lab, LeIsaac, LeRobot,
# packages/simulator) into a separate virtual environment on the host, so that
# recording and rollout can run with the Isaac Sim GUI without Docker.
#
# The environment is .venv-sim, not .venv: Isaac Sim pins numpy 1.26 and
# torch 2.7, while the uv workspace (.venv, training and verification) uses
# newer versions. Keeping two environments avoids the conflict.
#
# Usage: make install-sim      (or: bash scripts/install_sim_host.sh)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${SIM_VENV:-${ROOT}/.venv-sim}"
PY="${VENV}/bin/python"
ISAACLAB="${ROOT}/dependencies/IsaacLab"

export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }

glibc="$(getconf GNU_LIBC_VERSION | awk '{print $2}')"
if [ "$(printf '%s\n' 2.35 "${glibc}" | sort -V | head -n1)" != "2.35" ]; then
    echo "Isaac Sim pip wheels need glibc >= 2.35 (Ubuntu 22.04 or newer); found ${glibc}." >&2
    exit 1
fi

echo "== Git submodules =="
git -C "${ROOT}" submodule update --init --recursive

if [ ! -x "${PY}" ]; then
    echo "== Creating ${VENV} (Python 3.11) =="
    uv venv "${VENV}" --python 3.11 --seed
fi

echo "== Installing pinned packages (requirements-sim.txt) =="
uv pip install --python "${PY}" --no-deps \
    --index-strategy unsafe-best-match \
    --extra-index-url https://pypi.nvidia.com \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r "${ROOT}/requirements-sim.txt"

echo "== Installing Isaac Lab and packages/simulator (editable) =="
uv pip install --python "${PY}" --no-deps --no-build-isolation \
    -e "${ISAACLAB}/source/isaaclab" \
    -e "${ISAACLAB}/source/isaaclab_assets" \
    -e "${ISAACLAB}/source/isaaclab_contrib" \
    -e "${ISAACLAB}/source/isaaclab_mimic" \
    -e "${ISAACLAB}/source/isaaclab_rl" \
    -e "${ISAACLAB}/source/isaaclab_tasks" \
    -e "${ROOT}/packages/simulator"

echo "== Import check =="
"${PY}" - <<'EOF'
import importlib.metadata as md
for name in ("isaacsim", "isaaclab", "leisaac", "lerobot", "simulator", "torch", "numpy"):
    print(f"{name:10s} {md.version(name)}")
import torch
print("torch CUDA:", torch.cuda.is_available())
EOF

echo
echo "Done. Start a simulator shell with:  make launch-sim"
echo "(or: source scripts/sim_env.sh)"
