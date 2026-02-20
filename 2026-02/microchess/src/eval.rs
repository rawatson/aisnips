use crate::board::*;
use crate::types::*;

/// Evaluation score in centipawns, from the perspective of the side to move.
pub type Score = i32;

pub const MATE_SCORE: Score = 30000;
pub const DRAW_SCORE: Score = 0;

// Material values in centipawns
const PAWN_VALUE: Score = 100;
const KNIGHT_VALUE: Score = 320;
const BISHOP_VALUE: Score = 330;
const ROOK_VALUE: Score = 500;
const QUEEN_VALUE: Score = 900;

pub fn piece_value(kind: PieceKind) -> Score {
    match kind {
        PieceKind::Pawn => PAWN_VALUE,
        PieceKind::Knight => KNIGHT_VALUE,
        PieceKind::Bishop => BISHOP_VALUE,
        PieceKind::Rook => ROOK_VALUE,
        PieceKind::Queen => QUEEN_VALUE,
        PieceKind::King => 0,
    }
}

/// Piece-square tables (from White's perspective, rank 0 = row 0 of array).
/// Values are centipawns bonus/penalty for placing a piece on that square.
/// These are simplified versions of well-known PSTs.

#[rustfmt::skip]
const PAWN_PST: [Score; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
];

#[rustfmt::skip]
const KNIGHT_PST: [Score; 64] = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
];

#[rustfmt::skip]
const BISHOP_PST: [Score; 64] = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
];

#[rustfmt::skip]
const ROOK_PST: [Score; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
];

#[rustfmt::skip]
const QUEEN_PST: [Score; 64] = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
];

#[rustfmt::skip]
const KING_MIDDLEGAME_PST: [Score; 64] = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
];

#[rustfmt::skip]
const KING_ENDGAME_PST: [Score; 64] = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
];

fn pst_index(sq: Square, color: Color) -> usize {
    match color {
        Color::White => sq as usize,
        // Mirror vertically for black
        Color::Black => (sq ^ 56) as usize,
    }
}

/// Evaluate the position from the side to move's perspective.
pub fn evaluate(board: &Board) -> Score {
    let mut score: Score = 0;

    // Determine game phase for king PST interpolation
    let total_material = count_non_pawn_material(board);
    let endgame_weight = if total_material < 1300 {
        256
    } else if total_material > 5000 {
        0
    } else {
        256 * (5000 - total_material) / 3700
    };
    let middlegame_weight = 256 - endgame_weight;

    for sq in 0..64u8 {
        if let Some(piece) = board.piece_at(sq) {
            let idx = pst_index(sq, piece.color);
            let material = piece_value(piece.kind);
            let positional = match piece.kind {
                PieceKind::Pawn => PAWN_PST[idx],
                PieceKind::Knight => KNIGHT_PST[idx],
                PieceKind::Bishop => BISHOP_PST[idx],
                PieceKind::Rook => ROOK_PST[idx],
                PieceKind::Queen => QUEEN_PST[idx],
                PieceKind::King => {
                    (KING_MIDDLEGAME_PST[idx] * middlegame_weight
                        + KING_ENDGAME_PST[idx] * endgame_weight)
                        / 256
                }
            };

            let value = material + positional;
            match piece.color {
                Color::White => score += value,
                Color::Black => score -= value,
            }
        }
    }

    // Bishop pair bonus
    if popcount(board.piece_bb(PieceKind::Bishop, Color::White)) >= 2 {
        score += 30;
    }
    if popcount(board.piece_bb(PieceKind::Bishop, Color::Black)) >= 2 {
        score -= 30;
    }

    // Return from side-to-move perspective
    match board.side {
        Color::White => score,
        Color::Black => -score,
    }
}

fn count_non_pawn_material(board: &Board) -> i32 {
    let mut mat = 0;
    for color in [Color::White, Color::Black] {
        mat += popcount(board.piece_bb(PieceKind::Knight, color)) as i32 * KNIGHT_VALUE;
        mat += popcount(board.piece_bb(PieceKind::Bishop, color)) as i32 * BISHOP_VALUE;
        mat += popcount(board.piece_bb(PieceKind::Rook, color)) as i32 * ROOK_VALUE;
        mat += popcount(board.piece_bb(PieceKind::Queen, color)) as i32 * QUEEN_VALUE;
    }
    mat
}

/// MVV-LVA (Most Valuable Victim - Least Valuable Attacker) score for move ordering.
pub fn mvv_lva_score(board: &Board, m: Move) -> Score {
    if m.is_en_passant() {
        return PAWN_VALUE * 10 - PAWN_VALUE; // capturing a pawn with a pawn
    }
    if let Some(victim) = board.piece_at(m.to_sq()) {
        let attacker = board.piece_at(m.from_sq()).unwrap();
        return piece_value(victim.kind) * 10 - piece_value(attacker.kind);
    }
    0
}
