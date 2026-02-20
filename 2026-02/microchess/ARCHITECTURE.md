# Microchess Architecture

## Module Overview

```
src/
  main.rs          -- Entry point, UCI loop
  board.rs         -- Board state, bitboard representation
  movegen.rs       -- Legal move generation
  search.rs        -- Alpha-beta search with iterative deepening
  eval.rs          -- Position evaluation (material + positional)
  uci.rs           -- UCI protocol handler
  types.rs         -- Core types: Piece, Color, Square, Move, etc.
```

## Board Representation

We use a **mailbox + bitboard hybrid**:

- `pieces: [Bitboard; 6]` — one bitboard per piece type (Pawn..King)
- `colors: [Bitboard; 2]` — one bitboard per color (White, Black)
- `squares: [Option<Piece>; 64]` — mailbox for quick piece-at-square lookup

This gives us fast bitboard operations for move generation AND fast square
lookups for evaluation.

## Move Generation

Moves are generated as pseudo-legal first (ignoring pins/checks), then
filtered for legality by making the move and checking if the king is in check.
This is simpler to implement correctly, and the performance cost is small at
our target depth.

Move types handled:
- Normal moves (quiet + captures)
- Pawn double push
- En passant
- Castling (kingside + queenside)
- Promotions (queen, rook, bishop, knight)

## Search

- **Iterative deepening**: Search depth 1, then 2, then 3... until time runs out
- **Alpha-beta pruning**: Standard negamax with alpha-beta window
- **Quiescence search**: Extend captures at leaf nodes to avoid horizon effect
- **Move ordering**: Captures first (MVV-LVA), then killer moves, then quiet moves

## Evaluation

Centipawn-based scoring:
- **Material**: P=100, N=320, B=330, R=500, Q=900
- **Piece-square tables**: Positional bonuses per piece per square
- **Basic terms**: Doubled pawns, isolated pawns, open files for rooks, king safety

## UCI Protocol

Standard UCI protocol for engine communication:
- `uci` / `isready` / `ucinewgame` handshake
- `position startpos moves ...` to set up position
- `go wtime btime ...` to start searching
- `bestmove e2e4` to report results
- `quit` to exit

This lets us plug into cutechess-cli, Arena, or any UCI-compatible GUI.
