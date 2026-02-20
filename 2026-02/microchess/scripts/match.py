#!/usr/bin/env python3
"""
Match runner: pits Microchess against Stockfish at configurable ELO.
Uses python-chess for full rule enforcement (draw by repetition, 50-move, etc).

Usage:
    python3 scripts/match.py [num_games] [target_elo] [movetime_ms]
    python3 scripts/match.py 20 1200 500
"""

import chess
import chess.engine
import sys
import os
import math
import time

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MICROCHESS = os.path.join(PROJECT_DIR, "target", "release", "microchess")
STOCKFISH = "/usr/games/stockfish"


def play_game(microchess_path, stockfish_path, target_elo, movetime_ms, microchess_white):
    """Play a single game. Returns result from Microchess perspective: 1.0, 0.5, 0.0"""
    board = chess.Board()

    mc_engine = chess.engine.SimpleEngine.popen_uci(microchess_path)
    sf_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    # Configure Stockfish strength.
    # Stockfish 16 minimum UCI_Elo is 1320. For lower targets, use Skill Level.
    if target_elo >= 1320:
        sf_engine.configure({
            "UCI_LimitStrength": True,
            "UCI_Elo": target_elo,
        })
    else:
        # Map target ELO to Skill Level 0-20.
        # Rough mapping: Skill 0 ~ 800 ELO, Skill 20 ~ 3190 ELO
        skill = max(0, min(20, (target_elo - 800) * 20 // 2390))
        sf_engine.configure({
            "Skill Level": skill,
        })

    limit = chess.engine.Limit(time=movetime_ms / 1000.0)

    try:
        while not board.is_game_over(claim_draw=True):
            is_white_turn = board.turn == chess.WHITE
            is_microchess_turn = (is_white_turn == microchess_white)

            if is_microchess_turn:
                result = mc_engine.play(board, limit)
            else:
                result = sf_engine.play(board, limit)

            board.push(result.move)

        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return 0.5, board, "draw"
        elif (outcome.winner == chess.WHITE) == microchess_white:
            return 1.0, board, "win"
        else:
            return 0.0, board, "loss"
    finally:
        mc_engine.quit()
        sf_engine.quit()


def estimate_elo(wins, draws, losses, opponent_elo):
    """Estimate ELO from match results using the standard formula."""
    total = wins + draws + losses
    if total == 0:
        return opponent_elo
    score = (wins + 0.5 * draws) / total

    # Avoid division by zero at extremes
    if score <= 0.0:
        return opponent_elo - 400
    if score >= 1.0:
        return opponent_elo + 400

    # ELO difference from expected score
    elo_diff = -400.0 * math.log10(1.0 / score - 1.0)
    return opponent_elo + elo_diff


def main():
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    target_elo = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
    movetime_ms = int(sys.argv[3]) if len(sys.argv) > 3 else 500

    if not os.path.exists(MICROCHESS):
        print(f"Engine not found at {MICROCHESS}")
        print("Build with: cargo build --release")
        sys.exit(1)

    if not os.path.exists(STOCKFISH):
        print(f"Stockfish not found at {STOCKFISH}")
        sys.exit(1)

    print("=" * 60)
    print("  Microchess ELO Benchmark")
    print("=" * 60)
    print(f"  Games:       {num_games}")
    print(f"  Opponent:    Stockfish @ {target_elo} ELO")
    print(f"  Move time:   {movetime_ms}ms per move")
    print(f"  Alternating: White/Black each game")
    print("=" * 60)
    print()

    wins = 0
    draws = 0
    losses = 0

    start = time.time()

    for i in range(1, num_games + 1):
        microchess_white = (i % 2 == 1)  # Alternate colors
        color = "White" if microchess_white else "Black"

        print(f"Game {i:3d}/{num_games} (Microchess as {color})... ", end="", flush=True)

        try:
            score, board, result_str = play_game(
                MICROCHESS, STOCKFISH, target_elo, movetime_ms, microchess_white
            )
        except Exception as e:
            print(f"ERROR: {e}")
            losses += 1
            continue

        if score == 1.0:
            wins += 1
            tag = "WIN "
        elif score == 0.5:
            draws += 1
            tag = "DRAW"
        else:
            losses += 1
            tag = "LOSS"

        plies = len(list(board.move_stack))
        est_elo = estimate_elo(wins, draws, losses, target_elo)
        print(f"{tag}  ({plies:3d} plies)  | W:{wins} D:{draws} L:{losses}  Est. ELO: {est_elo:.0f}")

    elapsed = time.time() - start
    total = wins + draws + losses
    score_pct = (wins + 0.5 * draws) / total * 100 if total > 0 else 0
    est_elo = estimate_elo(wins, draws, losses, target_elo)

    print()
    print("=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(f"  Wins:   {wins:4d}  ({wins/total*100:5.1f}%)")
    print(f"  Draws:  {draws:4d}  ({draws/total*100:5.1f}%)")
    print(f"  Losses: {losses:4d}  ({losses/total*100:5.1f}%)")
    print(f"  Score:  {score_pct:.1f}%")
    print(f"  Estimated ELO: {est_elo:.0f}  (vs Stockfish @ {target_elo})")
    print(f"  Time:   {elapsed:.1f}s ({elapsed/total:.1f}s/game)")
    print("=" * 60)

    # Confidence note
    if total < 20:
        print("  NOTE: <20 games, ELO estimate has high variance.")
        print("  Run more games for a reliable estimate.")
    print()


if __name__ == "__main__":
    main()
