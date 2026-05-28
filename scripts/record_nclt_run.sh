#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

log() {
  echo "[record_nclt_run] $*" >&2
}

log "starting with bash: ${BASH_VERSION}"

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_SETUP="/opt/ros/noetic/setup.bash"
WS_SETUP="${WORKSPACE_DIR}/devel/setup.bash"

BAG_PATH="${BAG_PATH:-}"
START_SEC="${START_SEC:-2800}"
END_SEC="${END_SEC:-3400}"
PLAY_RATE="${PLAY_RATE:-2}"
DISPLAY_NAME="${DISPLAY_NAME:-:0.0}"
RECORD_FPS="${RECORD_FPS:-15}"
RECORD_DIR="${RECORD_DIR:-${WORKSPACE_DIR}/recordings}"
RECORD_NAME="${RECORD_NAME:-nclt_fastlio_$(date +%Y%m%d_%H%M%S).mp4}"
RECORD_PATH="${RECORD_DIR}/${RECORD_NAME}"
LOG_DIR="${LOG_DIR:-${WORKSPACE_DIR}/recordings/logs}"
LAUNCH_LOG="${LOG_DIR}/roslaunch_$(date +%Y%m%d_%H%M%S).log"
FFMPEG_LOG="${LOG_DIR}/ffmpeg_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${RECORD_DIR}"
mkdir -p "${LOG_DIR}"

if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "Missing ROS setup: ${ROS_SETUP}" >&2
  exit 1
fi

if [[ ! -f "${WS_SETUP}" ]]; then
  echo "Missing workspace setup: ${WS_SETUP}" >&2
  exit 1
fi

if [[ -z "${BAG_PATH}" ]]; then
  echo "BAG_PATH is not set. Export BAG_PATH=/path/to/your.bag before running." >&2
  exit 1
fi

if [[ ! -f "${BAG_PATH}" ]]; then
  echo "Missing bag file: ${BAG_PATH}" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required but not found in PATH." >&2
  exit 1
fi

log "sourcing ROS env: ${ROS_SETUP}"
source "${ROS_SETUP}"
log "sourcing workspace env: ${WS_SETUP}"
source "${WS_SETUP}"
log "environment ready"

DURATION_SEC=$((END_SEC - START_SEC))
if (( DURATION_SEC <= 0 )); then
  echo "END_SEC must be greater than START_SEC." >&2
  exit 1
fi

SCREEN_SIZE="$(
  xdpyinfo -display "${DISPLAY_NAME}" 2>/dev/null \
    | awk '/dimensions:/ {print $2; found=1} END {if (!found) exit 1}'
)"
SCREEN_SIZE="${SCREEN_SIZE:-1920x1080}"
log "detected screen size: ${SCREEN_SIZE}"

LAUNCH_PID=""
RECORD_PID=""

cleanup() {
  set +e
  if [[ -n "${RECORD_PID}" ]] && kill -0 "${RECORD_PID}" 2>/dev/null; then
    kill -INT "${RECORD_PID}" 2>/dev/null
    wait "${RECORD_PID}" 2>/dev/null
  fi
  if [[ -n "${LAUNCH_PID}" ]] && kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    kill -INT "${LAUNCH_PID}" 2>/dev/null
    wait "${LAUNCH_PID}" 2>/dev/null
  fi
}

trap cleanup EXIT INT TERM

log "workspace: ${WORKSPACE_DIR}"
log "bag: ${BAG_PATH}"
log "playback: start=${START_SEC}s end=${END_SEC}s duration=${DURATION_SEC}s rate=${PLAY_RATE}x"
log "recording: ${RECORD_PATH}"
log "display: ${DISPLAY_NAME} (${SCREEN_SIZE})"
log "launch log: ${LAUNCH_LOG}"
log "ffmpeg log: ${FFMPEG_LOG}"

log "starting roslaunch..."
roslaunch fast_lio_sam_sc_qn run.launch lidar:=nclt >"${LAUNCH_LOG}" 2>&1 &
LAUNCH_PID=$!
log "roslaunch started with PID ${LAUNCH_PID}"

log "waiting 10 seconds for ROS graph and RViz to come up..."
sleep 10

log "starting ffmpeg capture..."
ffmpeg -y \
  -video_size "${SCREEN_SIZE}" \
  -framerate "${RECORD_FPS}" \
  -f x11grab \
  -i "${DISPLAY_NAME}" \
  -c:v libx264 \
  -preset veryfast \
  -crf 23 \
  -pix_fmt yuv420p \
  "${RECORD_PATH}" >"${FFMPEG_LOG}" 2>&1 &
RECORD_PID=$!

log "screen recording started with PID ${RECORD_PID}"
log "press Enter to start rosbag playback in paused mode"
read -r

log "starting rosbag play..."
rosbag play "${BAG_PATH}" \
  --pause \
  --clock \
  -s "${START_SEC}" \
  -u "${DURATION_SEC}" \
  -r "${PLAY_RATE}"

log "rosbag play finished, stopping recording"
