#!/usr/bin/env bash
# One-time host setup for a fresh Brev "Isaac Lab + Isaac Sim" VM, so that the
# host simulator (make install-sim / make launch-sim) can start. Idempotent.
#
# Run on the VM HOST (prompt ubuntu@brev-...), NOT inside the Docker container
# (prompt root@<hash>):
#
#     bash scripts/setup_brev_host.sh
#
# What it fixes (each one gives a silent failure in Isaac Sim otherwise):
#   1. libvulkan1 / libGLU / X libs missing    -> Vulkan / libGLU.so.1 errors
#   2. glvnd egl_vendor.d/10_nvidia.json missing -> ERROR_INCOMPATIBLE_DRIVER
#   3. libcuda.so symlink missing               -> PhysX falls back to CPU, hangs
#   4. noVNC (localhost:6080) display :0 lives inside the container: relay its X
#      socket to the host so the GUI run (DISPLAY=:0) shows up on noVNC.
set -euo pipefail

LIBDIR=/usr/lib/x86_64-linux-gnu

if [ -f /.dockerenv ]; then
    echo "This is the Docker container. Run this on the VM host (exit the container first)." >&2
    exit 1
fi

echo "== 1. Host libraries =="
sudo apt-get update -qq || true
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    libvulkan1 vulkan-tools libglu1-mesa libxt6 libx11-6 libxrandr2 libxinerama1 \
    libxcursor1 libxi6 libegl1 libsm6 x11-utils x11-xserver-utils socat

echo "== 2. NVIDIA EGL vendor file (needed by the Vulkan driver) =="
if [ ! -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
    sudo mkdir -p /usr/share/glvnd/egl_vendor.d
    printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_nvidia.so.0"\n    }\n}\n' \
        | sudo tee /usr/share/glvnd/egl_vendor.d/10_nvidia.json >/dev/null
    echo "created 10_nvidia.json"
else
    echo "present"
fi

echo "== 3. libcuda.so symlink (PhysX GPU) =="
if [ ! -e "${LIBDIR}/libcuda.so" ]; then
    sudo ln -s libcuda.so.1 "${LIBDIR}/libcuda.so"
    echo "created libcuda.so"
else
    echo "present"
fi
sudo ldconfig

echo "== 4. noVNC display :0 relay =="
XPID="$(pgrep -x Xtigervnc | head -1 || true)"
if [ -z "${XPID}" ]; then
    echo "no Xtigervnc process (no noVNC desktop on this VM); skipped"
else
    # Container side: allow local (non-network) X clients.
    CID="$(docker ps -q | head -1 || true)"
    [ -n "${CID}" ] && docker exec -e DISPLAY=:0 "${CID}" xhost +local: || true
    # Host side: /tmp/.X11-unix/X0 -> container's X socket.
    if ! DISPLAY=:0 xdpyinfo >/dev/null 2>&1; then
        sudo pkill -f "socat UNIX-LISTEN:/tmp/.X11-unix/X0" 2>/dev/null || true
        sudo mkdir -p /tmp/.X11-unix
        sudo rm -f /tmp/.X11-unix/X0
        sudo nohup socat UNIX-LISTEN:/tmp/.X11-unix/X0,fork,mode=0777 \
            "UNIX-CONNECT:/proc/${XPID}/root/tmp/.X11-unix/X0" \
            >/tmp/xrelay.log 2>&1 </dev/null &
        sleep 2
    fi
    if DISPLAY=:0 xdpyinfo >/dev/null 2>&1; then
        echo "DISPLAY=:0 reachable (GUI runs show on localhost:6080)"
    else
        echo "WARNING: DISPLAY=:0 still unreachable, see /tmp/xrelay.log" >&2
    fi
fi

echo "== Vulkan check =="
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json vulkaninfo --summary 2>&1 \
    | grep -E "deviceName|ERROR" | head -3 || true
echo
echo "Done. Next: make install-sim && make launch-sim"
echo "The X relay (step 4) is lost on reboot; re-run this script to restore it."
