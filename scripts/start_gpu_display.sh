#!/usr/bin/env bash
# Start (or restore) a GPU-backed X display for the Isaac Sim GUI on a VM whose
# noVNC desktop is a software X server (Xtigervnc). Isaac Sim's Vulkan renderer
# cannot present to that server; this script runs a second X server on the GPU
# with the NVIDIA Xorg driver and publishes it through x11vnc and noVNC:
#
#     Xorg :2 (NVIDIA)  ->  x11vnc :5902 (localhost)  ->  noVNC on port 6080
#
# Port 6080 is the one Brev publishes as the VM's HTTPS "novnc-..." link, so
# students keep using that single URL. This script takes 6080 over from the
# VM's own noVNC (which shows the software desktop :1) and points it at the GPU
# display, where it also starts an XFCE desktop (panel, terminal). The browser
# connection drops once: reload the page and you see the GPU desktop.
#
# Called by scripts/setup_brev_host.sh (which installs the packages). Run it
# again after a reboot: none of these processes survive one. Idempotent.
#
#     bash scripts/start_gpu_display.sh
#
# Override with GPU_DISPLAY_NUM (2), GPU_VNC_PORT (5902), GPU_NOVNC_PORT (6080),
# GPU_DISPLAY_VT (vt8). x11vnc has no password and listens on localhost only;
# do not expose the noVNC port publicly.
set -uo pipefail

NUM="${GPU_DISPLAY_NUM:-2}"
RFB="${GPU_VNC_PORT:-5902}"
WEB="${GPU_NOVNC_PORT:-6080}"
VT="${GPU_DISPLAY_VT:-vt8}"
CONF=/etc/X11/xorg-nvidia-gpu.conf
ENVFILE=/tmp/gpu_display.env   # sourced by scripts/sim_env.sh

if [ "$(id -u)" = 0 ]; then SUDO=""
elif sudo -n true 2>/dev/null; then SUDO="sudo"
else echo "needs root (passwordless sudo): run as root or fix sudo" >&2; exit 1; fi

for cmd in Xorg x11vnc websockify nvidia-smi; do
    command -v "${cmd}" >/dev/null || { echo "missing ${cmd}: run bash scripts/setup_brev_host.sh" >&2; exit 1; }
done
[ -d /usr/share/novnc ] || { echo "missing /usr/share/novnc: run bash scripts/setup_brev_host.sh" >&2; exit 1; }

# PCI bus id of the GPU in X syntax (decimal): nvidia-smi prints 00000000:00:03.0 (hex).
RAW="$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1)"
IFS=':.' read -r _dom BUS DEV FN <<<"${RAW}"
BUSID="PCI:$((16#${BUS})):$((16#${DEV})):$((16#${FN}))"
echo "GPU at ${RAW} -> BusID ${BUSID}"

CONF_TEXT="Section \"ServerLayout\"
    Identifier \"L\"
    Screen 0 \"S\"
EndSection
Section \"Device\"
    Identifier \"D\"
    Driver \"nvidia\"
    BusID \"${BUSID}\"
    Option \"AllowEmptyInitialConfiguration\" \"True\"
    Option \"HardDPMS\" \"False\"
EndSection
Section \"Screen\"
    Identifier \"S\"
    Device \"D\"
    DefaultDepth 24
    SubSection \"Display\"
        Virtual 1920 1080
        Depth 24
    EndSubSection
EndSection"
if [ "$(cat "${CONF}" 2>/dev/null)" != "${CONF_TEXT}" ]; then
    printf '%s\n' "${CONF_TEXT}" | ${SUDO} tee "${CONF}" >/dev/null
    echo "wrote ${CONF}"
fi

# --- X server on the GPU ---
if pgrep -f "Xorg :${NUM}( |\$)" >/dev/null; then
    echo "Xorg :${NUM} already running"
else
    ${SUDO} rm -f "/tmp/.X11-unix/X${NUM}" "/tmp/.X${NUM}-lock"
    ${SUDO} setsid -f Xorg ":${NUM}" -config "${CONF}" -noreset \
        +extension GLX +extension RANDR +extension RENDER -nolisten tcp "${VT}" \
        >/tmp/xorg-gpu.log 2>&1 </dev/null
    for _ in $(seq 1 20); do [ -S "/tmp/.X11-unix/X${NUM}" ] && break; sleep 1; done
    if [ ! -S "/tmp/.X11-unix/X${NUM}" ]; then
        echo "Xorg :${NUM} did not start. Log (/var/log/Xorg.${NUM}.log):" >&2
        grep -E "\(EE\)" "/var/log/Xorg.${NUM}.log" 2>/dev/null | tail -10 >&2
        tail -5 /tmp/xorg-gpu.log >&2
        echo "If it complains about no screens, add  Option \"ConnectedMonitor\" \"DFP-0\"  to the Device section of ${CONF}." >&2
        exit 1
    fi
    echo "Xorg :${NUM} started"
fi

# --- desktop, VNC, noVNC ---
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-$(id -u)}"
mkdir -p "${XDG_RUNTIME_DIR}" && chmod 700 "${XDG_RUNTIME_DIR}"
# Desktop on the GPU display: XFCE (panel + terminal) if installed, else openbox.
# openbox is only the fallback (an earlier run, or an XFCE-less image): replace
# it with XFCE (panel, terminal) once XFCE is available.
if command -v startxfce4 >/dev/null && pgrep -u "$(id -u)" -x openbox >/dev/null; then
    pkill -u "$(id -u)" -x openbox || true
    sleep 1
fi
# (the root property outlives a dead window manager: check the window exists)
WM_WIN="$(DISPLAY=":${NUM}" xprop -root _NET_SUPPORTING_WM_CHECK 2>/dev/null | grep -o '0x[0-9a-f]*' | head -1)"
if [ -z "${WM_WIN}" ] || ! DISPLAY=":${NUM}" xprop -id "${WM_WIN}" >/dev/null 2>&1; then
    if command -v startxfce4 >/dev/null && command -v dbus-launch >/dev/null; then
        DISPLAY=":${NUM}" setsid -f dbus-launch --exit-with-session startxfce4 \
            >/tmp/desktop-gpu.log 2>&1 </dev/null
    elif command -v openbox >/dev/null; then
        DISPLAY=":${NUM}" setsid -f openbox >/tmp/desktop-gpu.log 2>&1 </dev/null
    else
        echo "no desktop (startxfce4/openbox) installed: the display has no window manager" >&2
    fi
fi
if ! pgrep -f "x11vnc -display :${NUM} " >/dev/null; then
    setsid -f x11vnc -display ":${NUM}" -rfbport "${RFB}" -localhost -nopw -forever -shared \
        >/tmp/x11vnc-gpu.log 2>&1 </dev/null
fi
# noVNC on ${WEB}. If the VM's own noVNC holds that port and points at the
# software desktop (VNC port 5901), replace it: reuse its web directory, which
# Brev customises. (Its parent is the lifecycle script, nothing restarts it.)
WEBDIR=/usr/share/novnc
OLD_PAT="websockify .*0\\.0\\.0\\.0:${WEB} 127\\.0\\.0\\.1:[0-9]+\$"
OLD_PID="$(pgrep -f "${OLD_PAT}" | head -1 || true)"
if [ -n "${OLD_PID}" ] && ! tr '\0' ' ' <"/proc/${OLD_PID}/cmdline" | grep -q "127.0.0.1:${RFB}"; then
    D="$(tr '\0' ' ' <"/proc/${OLD_PID}/cmdline" | grep -o -- '--web=[^ ]*' | cut -d= -f2)"
    [ -n "${D}" ] && [ -d "${D}" ] && WEBDIR="${D}"
    echo "replacing the noVNC on port ${WEB} (was the software desktop :1)"
    pkill -f "${OLD_PAT}" || true
    for _ in $(seq 1 10); do ss -ltn 2>/dev/null | grep -q ":${WEB} " || break; sleep 1; done
fi
if ! ss -ltn 2>/dev/null | grep -q ":${WEB} "; then
    setsid -f websockify --web="${WEBDIR}" "${WEB}" "127.0.0.1:${RFB}" \
        >/tmp/websockify-gpu.log 2>&1 </dev/null
fi
sleep 3

# --- verify: the NVIDIA GPU must be a Vulkan device on this display ---
XDG="${XDG_RUNTIME_DIR:-/tmp}"
if DISPLAY=":${NUM}" XDG_RUNTIME_DIR="${XDG}" vulkaninfo --summary 2>&1 | grep -q "deviceName.*NVIDIA"; then
    echo "Vulkan on DISPLAY=:${NUM}: NVIDIA GPU found"
else
    echo "Vulkan on DISPLAY=:${NUM} does not list the NVIDIA GPU (see /var/log/Xorg.${NUM}.log)" >&2
    exit 1
fi
ss -ltn 2>/dev/null | grep -q ":${WEB} " || { echo "noVNC did not start on ${WEB} (see /tmp/websockify-gpu.log)" >&2; exit 1; }

echo "GPU_DISPLAY=:${NUM}" >"${ENVFILE}"
echo "GPU display ready: the noVNC page on port ${WEB} (the Brev novnc-... link) now shows it."
echo "Reload that page; open a terminal on the new desktop and continue there."
echo "Run the GUI with DISPLAY=:${NUM} (make launch-sim sets it for you)."
