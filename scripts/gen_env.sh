# Used by `make launch-gen` as the bash rcfile: activates the general
# environment (.venv) in a new shell; `exit` leaves it.
# To activate in your current shell instead: source .venv/bin/activate

# Keep the user's own shell setup.
if [ -f "${HOME}/.bashrc" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/.bashrc"
fi

_gen_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

if [ ! -f "${_gen_root}/.venv/bin/activate" ]; then
    echo "No general environment at ${_gen_root}/.venv. Run 'make install-gen' first." >&2
    unset _gen_root
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "${_gen_root}/.venv/bin/activate"
cd "${_gen_root}"
echo "General environment active: $(python --version), $(command -v python)"
unset _gen_root
