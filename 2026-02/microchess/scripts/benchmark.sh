#!/bin/bash
# Benchmark Microchess against Stockfish at reduced strength using cutechess-cli.
#
# Prerequisites:
#   sudo apt-get install cutechess stockfish
#
# Usage:
#   ./scripts/benchmark.sh [num_games] [target_elo]
#
# Example:
#   ./scripts/benchmark.sh 100 1200

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

NUM_GAMES="${1:-20}"
TARGET_ELO="${2:-1200}"

MICROCHESS="$PROJECT_DIR/target/release/microchess"

# Build if needed
if [ ! -f "$MICROCHESS" ]; then
    echo "Building microchess..."
    (cd "$PROJECT_DIR" && cargo build --release)
fi

# Check for cutechess-cli
if ! command -v cutechess-cli &>/dev/null; then
    echo "cutechess-cli not found. Install it with: sudo apt-get install cutechess"
    echo ""
    echo "Falling back to self-play test..."
    echo ""
    "$SCRIPT_DIR/selfplay.sh" "$NUM_GAMES"
    exit 0
fi

# Check for stockfish
if ! command -v stockfish &>/dev/null; then
    echo "stockfish not found. Install it with: sudo apt-get install stockfish"
    exit 1
fi

echo "=== Microchess Benchmark ==="
echo "Games: $NUM_GAMES"
echo "Target opponent ELO: $TARGET_ELO"
echo ""

# Run tournament
cutechess-cli \
    -engine name=Microchess cmd="$MICROCHESS" proto=uci \
    -engine name="Stockfish-$TARGET_ELO" cmd=stockfish proto=uci \
        option.UCI_LimitStrength=true \
        "option.UCI_Elo=$TARGET_ELO" \
    -each tc=40/60+0.5 \
    -games "$NUM_GAMES" \
    -rounds "$NUM_GAMES" \
    -pgnout "$PROJECT_DIR/results/benchmark.pgn" \
    -recover \
    -repeat \
    -openings file=/dev/null \
    2>&1 | tee "$PROJECT_DIR/results/benchmark.log"

echo ""
echo "Results saved to results/benchmark.pgn and results/benchmark.log"
