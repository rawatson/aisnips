#!/bin/bash
# Self-play test for Microchess (no external dependencies needed).
# Plays the engine against itself with different time controls to verify it works.
#
# Usage: ./scripts/selfplay.sh [num_games]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MICROCHESS="$PROJECT_DIR/target/release/microchess"

NUM_GAMES="${1:-5}"

if [ ! -f "$MICROCHESS" ]; then
    echo "Building microchess..."
    (cd "$PROJECT_DIR" && cargo build --release)
fi

echo "=== Microchess Self-Play Test ==="
echo "Games: $NUM_GAMES"
echo ""

# Simple self-play: feed positions back and forth
python3 "$SCRIPT_DIR/selfplay.py" "$MICROCHESS" "$NUM_GAMES"
