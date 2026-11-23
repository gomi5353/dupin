#!/usr/bin/env bash
# One-time setup for a fresh Brev "Isaac Lab + Isaac Sim" VM, so that the
# simulator environment (make install-sim / make launch-sim) can start.
# Idempotent: it checks each item and changes only what is missing.
#
# It works in both places a terminal can be opened:
#   - the VM's SSH shell (prompt ubuntu@brev-...), or
#   - a terminal on the noVNC desktop (localhost:6080), which runs inside the
#     Docker container (prompt root@<hash>, or ubuntu@<hash> after `su ubuntu`).
# The two are different file systems: files created in one are not visible in
# the other. Pick one and keep the repository and `make launch-sim` there.
#
#     bash scripts/setup_brev_host.sh
#
# Checked / fixed (each one fails silently or confusingly in Isaac Sim):
#   1. libvulkan1, libGLU, X libs missing     -> Vulkan / libGLU.so.1 errors
#   2. glvnd egl_vendor.d/10_nvidia.json missing -> ERROR_INCOMPATIBLE_DRIVER
#   3. libcuda.so link missing                -> PhysX runs on the CPU and hangs
#   1c. ffmpeg missing                        -> video decoding fails (lerobot-train, lerobot-dataset-viz)
#   4. uv, cmake, gcc/g++/make missing        -> make install-sim fails while
#      building a package from source
#   5. (SSH shell only) noVNC display :0 lives inside the container: relay its
#      X socket to the host so a GUI run (DISPLAY=:0) shows on localhost:6080.
#
# Fixes need root. The script decides itself: as root it runs commands directly
# (root needs no `sudo`), as a user it uses `sudo` only if
# that works without a password (true for `ubuntu` in the SSH shell, false for
# `ubuntu` in the container), and otherwise it reports what is missing. So do
# not add `sudo` by hand: in the container run the script as root (`su -`)
# when it asks for root. The NVIDIA toolkit normally provides the driver
# files in the container, so usually only cmake/uv need installing there.
set -uo pipefail

# Whatever happens below, leave the terminal usable (echo/line editing on).
trap '[ -t 0 ] && stty sane 2>/dev/null' EXIT

LIBDIR=/usr/lib/x86_64-linux-gnu
IN_CONTAINER=0
[ -f /.dockerenv ] && IN_CONTAINER=1
echo "Running in: $([ "${IN_CONTAINER}" = 1 ] && echo 'Docker container (noVNC terminal)' || echo 'VM host (SSH shell)') as $(id -un)"

# SUDO: "" when root, "sudo" when passwordless sudo works, "none" otherwise.
if [ "$(id -u)" = 0 ]; then SUDO=""
elif sudo -n true 2>/dev/null; then SUDO="sudo"
else SUDO="none"; fi

FAILED=0
# run_root <cmd...>: run with root rights, or report that it cannot.
run_root() {
    if [ "${SUDO}" = "none" ]; then
        echo "  needs root: $*" >&2
        FAILED=1
        return 1
    fi
    ${SUDO} "$@"
}

# apt_install <pkgs...>: wait for the apt/dpkg lock (a fresh VM runs its own
# first-boot apt jobs for several minutes) instead of failing, and record failure.
apt_install() {
    local lock="-o DPkg::Lock::Timeout=900"
    run_root apt-get ${lock} update -qq || true
    run_root env DEBIAN_FRONTEND=noninteractive apt-get ${lock} install -y -qq "$@" \
        || { echo "  apt-get install failed: $*" >&2; FAILED=1; }
}

echo "== 1. Libraries =="
LDCACHE="$(ldconfig -p)"
MISSING_LIBS=""
for lib in libvulkan.so.1 libGLU.so.1 libXt.so.6 libX11.so.6; do
    grep -q "${lib}" <<<"${LDCACHE}" || MISSING_LIBS="${MISSING_LIBS} ${lib}"
done
if [ -z "${MISSING_LIBS}" ]; then
    echo "present"
else
    echo "missing:${MISSING_LIBS}"
    apt_install libvulkan1 vulkan-tools libglu1-mesa libxt6 libx11-6 libxrandr2 libxinerama1 \
        libxcursor1 libxi6 libegl1 libsm6 x11-utils x11-xserver-utils
    LDCACHE="$(ldconfig -p)"
    for lib in ${MISSING_LIBS}; do
        grep -q "${lib}" <<<"${LDCACHE}" || { echo "  still missing after install: ${lib}" >&2; FAILED=1; }
    done
fi
# Tools for the X relay (host only).
if [ "${IN_CONTAINER}" = 0 ] && ! command -v socat >/dev/null; then
    apt_install socat x11-utils x11-xserver-utils
fi

echo "== 1b. Build tools (cmake, gcc, g++, make) =="
MISSING_TOOLS=""
for tool in cmake gcc g++ make; do
    command -v "${tool}" >/dev/null || MISSING_TOOLS="${MISSING_TOOLS} ${tool}"
done
if [ -z "${MISSING_TOOLS}" ]; then
    echo "present"
else
    echo "missing:${MISSING_TOOLS}"
    apt_install cmake build-essential
fi

echo "== 1c. FFmpeg (video decoding of lerobot-train and lerobot-dataset-viz) =="
if command -v ffmpeg >/dev/null; then
    echo "present"
else
    echo "missing"
    apt_install ffmpeg
fi

echo "== 2. NVIDIA EGL vendor file (needed by the Vulkan driver) =="
if [ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ] || [ -f /etc/glvnd/egl_vendor.d/10_nvidia.json ]; then
    echo "present"
else
    echo "missing"
    run_root mkdir -p /usr/share/glvnd/egl_vendor.d && \
    printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_nvidia.so.0"\n    }\n}\n' \
        | run_root tee /usr/share/glvnd/egl_vendor.d/10_nvidia.json >/dev/null && echo "created"
fi

echo "== 3. libcuda.so link (PhysX GPU) =="
if [ -e "${LIBDIR}/libcuda.so" ]; then
    echo "present"
elif [ ! -e "${LIBDIR}/libcuda.so.1" ]; then
    echo "libcuda.so.1 not found: NVIDIA driver not visible here" >&2
    FAILED=1
else
    echo "missing"
    run_root ln -s libcuda.so.1 "${LIBDIR}/libcuda.so" && run_root ldconfig && echo "created"
fi

echo "== 4. uv =="
if command -v uv >/dev/null || [ -x "${HOME}/.local/bin/uv" ]; then
    echo "present"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh || FAILED=1
fi
echo 'If "uv" is not found afterwards: export PATH="$HOME/.local/bin:$PATH"'

echo "== 5. noVNC display :0 relay =="
XPID="$(pgrep -x Xtigervnc | head -1 || true)"
if [ "${IN_CONTAINER}" = 1 ]; then
    echo "inside the container: DISPLAY=:0 is local, use it directly; skipped"
elif [ -z "${XPID}" ]; then
    echo "no Xtigervnc process (no noVNC desktop on this VM); skipped"
else
    # Container side: allow local (non-network) X clients.
    CID="$(docker ps -q | head -1 || true)"
    if [ -n "${CID}" ]; then docker exec -e DISPLAY=:0 "${CID}" xhost +local: || true; fi
    # Host side: /tmp/.X11-unix/X0 -> container's X socket.
    if ! DISPLAY=:0 xdpyinfo >/dev/null 2>&1; then
        run_root pkill -f "socat UNIX-LISTEN:/tmp/.X11-unix/X0" 2>/dev/null || true
        run_root mkdir -p /tmp/.X11-unix
        run_root rm -f /tmp/.X11-unix/X0
        # setsid -f: own session, no controlling terminal. A plain `sudo ... &`
        # shares the terminal and can leave it without echo (typing invisible).
        run_root setsid -f socat UNIX-LISTEN:/tmp/.X11-unix/X0,fork,mode=0777 \
            "UNIX-CONNECT:/proc/${XPID}/root/tmp/.X11-unix/X0" \
            >/tmp/xrelay.log 2>&1 </dev/null
        sleep 2
    fi
    if DISPLAY=:0 xdpyinfo >/dev/null 2>&1; then
        echo "DISPLAY=:0 reachable (GUI runs show on localhost:6080)"
    else
        echo "WARNING: DISPLAY=:0 still unreachable, see /tmp/xrelay.log" >&2
        FAILED=1
    fi
fi

echo "== NVIDIA driver (kernel module vs user-space libraries) =="
if command -v nvidia-smi >/dev/null; then
    if SMI="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1)"; then
        echo "${SMI}"
        case "${SMI}" in
            *,\ 580.*) ;;
            *) echo "WARNING: Isaac Sim 5.1 is validated with the 580.x driver; other branches (e.g. 595.x) may segfault in librtx.scenedb.plugin.so. Prefer a 580.x instance." >&2 ;;
        esac
    else
        echo "${SMI}" >&2
        echo "ERROR: NVIDIA driver is broken here (kernel module and libraries differ). Reboot once; if it persists, do not use this instance." >&2
        FAILED=1
    fi
fi

echo "== Vulkan check =="
ICD=""
for f in /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json; do
    [ -f "${f}" ] && ICD="${f}" && break
done
if [ -z "${ICD}" ]; then
    echo "no nvidia_icd.json found" >&2; FAILED=1
else
    VK_ICD_FILENAMES="${ICD}" vulkaninfo --summary 2>&1 | grep -E "deviceName|ERROR" | head -3 || true
fi
echo
if [ "${FAILED}" = 0 ]; then
    echo "OK. Next: make install-sim && make launch-sim"
else
    echo "Some items could not be fixed (see messages above). Re-run this script as root." >&2
    exit 1
fi
echo "The X relay (step 5) is lost on reboot; re-run this script to restore it."
