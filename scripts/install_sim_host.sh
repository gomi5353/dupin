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
# egl_probe 1.0.2 (built from source) declares cmake_minimum_required(2.8.12);
# CMake >= 4 rejects that unless a policy minimum is given. Harmless on CMake 3.
export CMAKE_POLICY_VERSION_MINIMUM=3.5

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }

glibc="$(getconf GNU_LIBC_VERSION | awk '{print $2}')"
if [ "$(printf '%s\n' 2.35 "${glibc}" | sort -V | head -n1)" != "2.35" ]; then
    echo "Isaac Sim pip wheels need glibc >= 2.35 (Ubuntu 22.04 or newer); found ${glibc}." >&2
    exit 1
fi

echo "== Host graphics libraries =="
# Isaac Sim needs these on the host even when headless (RTX renderer uses Vulkan).
missing=""
ldcache="$(ldconfig -p)"   # not piped into grep -q: SIGPIPE + pipefail gives false "missing"
for lib in libvulkan.so.1 libGLU.so.1 libXt.so.6 libX11.so.6; do
    grep -q "${lib}" <<<"${ldcache}" || missing="${missing} ${lib}"
done
if [ -n "${missing}" ]; then
    echo "Fix everything at once (on the VM host): bash scripts/setup_brev_host.sh" >&2
    echo "Missing host libraries:${missing}" >&2
    echo "Install with: sudo apt-get install -y libvulkan1 libglu1-mesa libxt6 libx11-6" >&2
    exit 1
fi
for tool in cmake gcc g++ make; do
    command -v "${tool}" >/dev/null || {
        echo "Missing build tool: ${tool} (some packages are built from source)." >&2
        echo "Fix: bash scripts/setup_brev_host.sh   (or: apt-get install -y cmake build-essential, with sudo only if you are not root)" >&2
        exit 1
    }
done
echo "cmake $(cmake --version | head -n1 | awk '{print $3}')"
# The NVIDIA Vulkan ICD initialises through EGL; without the NVIDIA GLVND vendor
# file only Mesa EGL is found and Isaac Sim fails with ERROR_INCOMPATIBLE_DRIVER.
if ls /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0 >/dev/null 2>&1 \
    && [ ! -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ] \
    && [ ! -f /etc/glvnd/egl_vendor.d/10_nvidia.json ]; then
    echo "Missing /usr/share/glvnd/egl_vendor.d/10_nvidia.json (Vulkan will fail with ERROR_INCOMPATIBLE_DRIVER)." >&2
    echo "Create it with:" >&2
    echo "  sudo mkdir -p /usr/share/glvnd/egl_vendor.d" >&2
    echo "  echo '{\"file_format_version\":\"1.0.0\",\"ICD\":{\"library_path\":\"libEGL_nvidia.so.0\"}}' | sudo tee /usr/share/glvnd/egl_vendor.d/10_nvidia.json" >&2
    exit 1
fi

# Kit/PhysX dlopen libcuda.so; without it "omni.physx handle on CUDA lib is (nil)",
# PhysX falls back to the CPU solver and the simulation hangs (no error shown).
if [ -f /usr/lib/x86_64-linux-gnu/libcuda.so.1 ] && [ ! -e /usr/lib/x86_64-linux-gnu/libcuda.so ]; then
    echo "Missing /usr/lib/x86_64-linux-gnu/libcuda.so symlink (PhysX would silently run on CPU and hang)." >&2
    echo "Create it with: sudo ln -s libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so && sudo ldconfig" >&2
    exit 1
fi

echo "== Git submodules =="
git -C "${ROOT}" submodule update --init --recursive

if [ ! -x "${PY}" ]; then
    echo "== Creating ${VENV} (Python 3.11) =="
    uv venv "${VENV}" --python 3.11 --seed --prompt simulator
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
