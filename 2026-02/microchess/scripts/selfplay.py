#!/usr/bin/env python3
"""Self-play test for Microchess. Plays the engine against itself."""

import subprocess
import sys
import os


def start_engine(path):
    proc = subprocess.Popen(
        [path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    return proc


def send(proc, cmd):
    proc.stdin.write(cmd + "\n")
    proc.stdin.flush()


def read_until(proc, prefix):
    while True:
        line = proc.stdout.readline().strip()
        if line.startswith(prefix):
            return line


def play_game(engine_path, game_num, movetime=200):
    """Play a single game, return result string."""
    engine = start_engine(engine_path)
    send(engine, "uci")
    read_until(engine, "uciok")
    send(engine, "isready")
    read_until(engine, "readyok")

    moves = []
    result = None

    for ply in range(500):  # Max 250 moves per side
        move_str = " ".join(moves)
        if moves:
            send(engine, f"position startpos moves {move_str}")
        else:
            send(engine, "position startpos")

        send(engine, f"go movetime {movetime}")
        line = read_until(engine, "bestmove")
        best = line.split()[1]

        if best == "(none)" or best == "0000":
            # Game over
            side = "White" if ply % 2 == 0 else "Black"
            result = f"{'Black' if ply % 2 == 0 else 'White'} wins (no legal moves for {side})"
            break

        moves.append(best)

        # Simple repetition detection: check if the last 4 moves repeat
        if len(moves) >= 8:
            last4 = moves[-4:]
            prev4 = moves[-8:-4]
            if last4 == prev4:
                result = "Draw (repetition)"
                break

        if len(moves) > 200:
            result = "Draw (too many moves)"
            break

    if result is None:
        result = "Draw (max ply reached)"

    send(engine, "quit")
    engine.wait(timeout=5)

    return result, moves


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <engine_path> [num_games]")
        sys.exit(1)

    engine_path = sys.argv[1]
    num_games = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    if not os.path.exists(engine_path):
        print(f"Engine not found: {engine_path}")
        sys.exit(1)

    print(f"Engine: {engine_path}")
    print(f"Games: {num_games}")
    print()

    for i in range(1, num_games + 1):
        print(f"Game {i}/{num_games}...", end=" ", flush=True)
        result, moves = play_game(engine_path, i)
        print(f"{result} ({len(moves)} plies)")

    print("\nSelf-play complete!")


if __name__ == "__main__":
    main()
