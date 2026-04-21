#!/usr/bin/env bash
# ============================================================
# Bimanual SO-101 Teleoperation — Phi9
# ============================================================
# Mapping:
#   LEFT  arm pair → black leader  + black follower
#   RIGHT arm pair → red leader    + red follower
#
# Calibration files live in ./calibration/
#   teleoperators/bimanual_left.json   (black_leader_arm)
#   teleoperators/bimanual_right.json  (red_leader_arm)
#   robots/bimanual_left.json          (black_follower_arm)
#   robots/bimanual_right.json         (red_follower_arm)
#
# Usage:
#   ./run_teleop.sh            # plain teleoperation
#   ./run_teleop.sh --display  # teleoperation + live display
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALIB_DIR="$SCRIPT_DIR/calibration"

# ── Ports (from Arms_Calibration/ports.json) ──────────────
BLACK_LEADER_PORT="/dev/tty.usbmodem5B3D0460711"
RED_LEADER_PORT="/dev/tty.usbmodem5AE60557611"
BLACK_FOLLOWER_PORT="/dev/tty.usbmodem5B141157181"
RED_FOLLOWER_PORT="/dev/tty.usbmodem5AE60536461"

# ── Optional flags ─────────────────────────────────────────
DISPLAY_FLAG=""
if [[ "${1:-}" == "--display" ]]; then
    DISPLAY_FLAG="--display_data=true"
fi

# ── Activate conda env & run ───────────────────────────────
PYTHON="/Users/rudraksh/miniconda3/envs/lerobot/bin/python"
TELEOP_BIN="/Users/rudraksh/miniconda3/envs/lerobot/bin/lerobot-teleoperate"

echo "Starting bimanual SO-101 teleoperation..."
echo "  Left  leader : $BLACK_LEADER_PORT"
echo "  Right leader : $RED_LEADER_PORT"
echo "  Left  follower: $BLACK_FOLLOWER_PORT"
echo "  Right follower: $RED_FOLLOWER_PORT"
echo ""

"$TELEOP_BIN" \
    --robot.type=bi_so_follower \
    --robot.id=bimanual \
    --robot.calibration_dir="$CALIB_DIR/robots" \
    --robot.left_arm_config.port="$BLACK_FOLLOWER_PORT" \
    --robot.right_arm_config.port="$RED_FOLLOWER_PORT" \
    --teleop.type=bi_so_leader \
    --teleop.id=bimanual \
    --teleop.calibration_dir="$CALIB_DIR/teleoperators" \
    --teleop.left_arm_config.port="$BLACK_LEADER_PORT" \
    --teleop.right_arm_config.port="$RED_LEADER_PORT" \
    $DISPLAY_FLAG
