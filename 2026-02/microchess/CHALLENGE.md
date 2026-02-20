# Microchess: Beat a 1200 ELO Engine

## The Challenge

In late 2024/early 2025, articles surfaced showing ChatGPT losing to an Atari-era
chess engine rated ~1200 ELO. The goal of this project is to write a chess engine
**from scratch in Rust** that can consistently outperform a 1200 ELO engine.

### What is 1200 ELO?

- A casual club player who knows basic tactics
- Understands opening principles but doesn't memorize lines
- Can spot 1-2 move tactics (forks, pins, skewers)
- Often blunders pieces in the middlegame
- Weak endgame technique

### Success Criteria

1. The engine plays **legal chess** (all moves valid, handles all special rules)
2. The engine achieves an **estimated ELO > 1200** when benchmarked against
   calibrated opponents (e.g., Stockfish at reduced strength, or known-ELO engines)
3. The engine wins **>60%** of games against a 1200-rated opponent over 100+ games

## Approach

We're building a classical chess engine (not ML-based) with:

- **Board representation**: Bitboard-based for fast move generation
- **Move generation**: Full legal move generation including castling, en passant, promotions
- **Search**: Alpha-beta pruning with iterative deepening
- **Evaluation**: Material counting + piece-square tables + basic positional heuristics
- **UCI protocol**: Standard interface so we can pit it against other engines

## Benchmarking

We use `cutechess-cli` to run automated matches against calibrated opponents.
The typical benchmark opponent is Stockfish with `UCI_LimitStrength` set to 1200,
or other engines with known ELO ratings.

## Iterative Improvement

The engine is designed to be improved incrementally:

1. **Phase 1**: Legal move generation + random moves (baseline)
2. **Phase 2**: Material-only evaluation + minimax (should reach ~800 ELO)
3. **Phase 3**: Alpha-beta + piece-square tables (should reach ~1200 ELO)
4. **Phase 4**: Quiescence search + move ordering (should reach ~1400+ ELO)
5. **Phase 5**: Transposition tables, null-move pruning, etc. (1600+ ELO)

We start at Phase 3 to have a solid baseline, then iterate.
