# Source this file to use the host simulator environment (.venv-sim):
#
#     source scripts/sim_env.sh
#
# It activates .venv-sim and sets the same variables the Isaac Lab container
# sets, so recording and rollout commands from the guides work unchanged.

# `make launch-sim` uses this file as the bash rcfile; keep the user's own setup.
if [ -n "${SIM_RCFILE:-}" ] && [ -f "${HOME}/.bashrc" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/.bashrc"
fi

_sim_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
_sim_venv="${SIM_VENV:-${_sim_root}/.venv-sim}"

if [ ! -f "${_sim_venv}/bin/activate" ]; then
    echo "No simulator environment at ${_sim_venv}. Run 'make install-sim' first." >&2
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "${_sim_venv}/bin/activate"

# Accept the Isaac Sim EULA non-interactively (as in the container).
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y

# The container mounts datasets/ as LeRobot's dataset folder; do the same here,
# so recorded datasets land in datasets/<hf_user>/<name> on both paths.
export HF_LEROBOT_HOME="${_sim_root}/datasets"

# Use the NVIDIA Vulkan driver if several ICDs are installed.
if [ -z "${VK_ICD_FILENAMES:-}" ]; then
    for _icd in /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json; do
        if [ -f "${_icd}" ]; then export VK_ICD_FILENAMES="${_icd}"; break; fi
    done
fi

cd "${_sim_root}"
echo "Simulator environment active: $(python --version), $(command -v python)"
unset _sim_root _sim_venv _icd
