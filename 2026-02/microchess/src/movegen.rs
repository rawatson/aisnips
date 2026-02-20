use crate::board::*;
use crate::types::*;

/// Generate all legal moves for the current position.
pub fn generate_moves(board: &Board) -> Vec<Move> {
    let mut moves = Vec::with_capacity(64);
    generate_pseudo_legal(board, &mut moves);

    // Filter to only legal moves (king not in check after the move)
    let mut legal = Vec::with_capacity(moves.len());
    let mut board_copy = board.clone();
    for m in moves {
        let undo = board_copy.make_move(m);
        // After making the move, side has flipped. Check if the *previous* side's king is in check.
        let king_bb = board_copy.piece_bb(PieceKind::King, board.side);
        if king_bb != 0 {
            let king_sq = king_bb.trailing_zeros() as Square;
            if !board_copy.is_attacked_by(king_sq, board_copy.side) {
                legal.push(m);
            }
        }
        board_copy.unmake_move(m, undo);
    }
    legal
}

fn generate_pseudo_legal(board: &Board, moves: &mut Vec<Move>) {
    let us = board.side;
    let them = us.flip();
    let our_pieces = board.colors[us as usize];
    let their_pieces = board.colors[them as usize];
    let occ = board.occupancy();

    // Pawns
    generate_pawn_moves(board, us, our_pieces, their_pieces, occ, moves);

    // Knights
    for from in bb_iter(board.piece_bb(PieceKind::Knight, us)) {
        let attacks = KNIGHT_ATTACKS[from as usize] & !our_pieces;
        for to in bb_iter(attacks) {
            moves.push(Move::new(from, to));
        }
    }

    // Bishops
    for from in bb_iter(board.piece_bb(PieceKind::Bishop, us)) {
        let attacks = bishop_attacks(from, occ) & !our_pieces;
        for to in bb_iter(attacks) {
            moves.push(Move::new(from, to));
        }
    }

    // Rooks
    for from in bb_iter(board.piece_bb(PieceKind::Rook, us)) {
        let attacks = rook_attacks(from, occ) & !our_pieces;
        for to in bb_iter(attacks) {
            moves.push(Move::new(from, to));
        }
    }

    // Queens
    for from in bb_iter(board.piece_bb(PieceKind::Queen, us)) {
        let attacks = (bishop_attacks(from, occ) | rook_attacks(from, occ)) & !our_pieces;
        for to in bb_iter(attacks) {
            moves.push(Move::new(from, to));
        }
    }

    // King
    let king_sq = board.piece_bb(PieceKind::King, us).trailing_zeros() as Square;
    let king_attacks = KING_ATTACKS[king_sq as usize] & !our_pieces;
    for to in bb_iter(king_attacks) {
        moves.push(Move::new(king_sq, to));
    }

    // Castling
    generate_castling(board, us, occ, king_sq, moves);
}

fn generate_pawn_moves(
    board: &Board,
    us: Color,
    our_pieces: Bitboard,
    their_pieces: Bitboard,
    occ: Bitboard,
    moves: &mut Vec<Move>,
) {
    let pawns = board.piece_bb(PieceKind::Pawn, us);
    let _ = our_pieces; // we use occ instead

    match us {
        Color::White => {
            // Single push
            let single = (pawns << 8) & !occ;
            for to in bb_iter(single & !0xFF00000000000000) {
                moves.push(Move::new(to - 8, to));
            }
            // Promotions from single push
            for to in bb_iter(single & 0xFF00000000000000) {
                add_promotions(moves, to - 8, to);
            }
            // Double push
            let double = ((single & 0x0000000000FF0000) << 8) & !occ;
            for to in bb_iter(double) {
                moves.push(Move::new(to - 16, to));
            }
            // Captures left (file decreases)
            let cap_left = (pawns & !0x0101010101010101) << 7;
            for to in bb_iter(cap_left & their_pieces & !0xFF00000000000000) {
                moves.push(Move::new(to - 7, to));
            }
            for to in bb_iter(cap_left & their_pieces & 0xFF00000000000000) {
                add_promotions(moves, to - 7, to);
            }
            // Captures right (file increases)
            let cap_right = (pawns & !0x8080808080808080) << 9;
            for to in bb_iter(cap_right & their_pieces & !0xFF00000000000000) {
                moves.push(Move::new(to - 9, to));
            }
            for to in bb_iter(cap_right & their_pieces & 0xFF00000000000000) {
                add_promotions(moves, to - 9, to);
            }
            // En passant
            if let Some(ep) = board.en_passant {
                let ep_bb = bb(ep);
                if cap_left & ep_bb != 0 {
                    // The pawn capturing is on (ep - 7) which means from = ep - 7 for white
                    // Actually: cap_left shifts pawns << 7, so the from sq is to - 7
                    moves.push(Move::new_en_passant(ep - 7, ep));
                }
                if cap_right & ep_bb != 0 {
                    moves.push(Move::new_en_passant(ep - 9, ep));
                }
            }
        }
        Color::Black => {
            // Single push
            let single = (pawns >> 8) & !occ;
            for to in bb_iter(single & !0x00000000000000FF) {
                moves.push(Move::new(to + 8, to));
            }
            // Promotions
            for to in bb_iter(single & 0x00000000000000FF) {
                add_promotions(moves, to + 8, to);
            }
            // Double push
            let double = ((single & 0x0000FF0000000000) >> 8) & !occ;
            for to in bb_iter(double) {
                moves.push(Move::new(to + 16, to));
            }
            // Captures right (from black's perspective, file increases)
            let cap_right = (pawns & !0x8080808080808080) >> 7;
            for to in bb_iter(cap_right & their_pieces & !0x00000000000000FF) {
                moves.push(Move::new(to + 7, to));
            }
            for to in bb_iter(cap_right & their_pieces & 0x00000000000000FF) {
                add_promotions(moves, to + 7, to);
            }
            // Captures left
            let cap_left = (pawns & !0x0101010101010101) >> 9;
            for to in bb_iter(cap_left & their_pieces & !0x00000000000000FF) {
                moves.push(Move::new(to + 9, to));
            }
            for to in bb_iter(cap_left & their_pieces & 0x00000000000000FF) {
                add_promotions(moves, to + 9, to);
            }
            // En passant
            if let Some(ep) = board.en_passant {
                let ep_bb = bb(ep);
                if cap_right & ep_bb != 0 {
                    moves.push(Move::new_en_passant(ep + 7, ep));
                }
                if cap_left & ep_bb != 0 {
                    moves.push(Move::new_en_passant(ep + 9, ep));
                }
            }
        }
    }
}

fn add_promotions(moves: &mut Vec<Move>, from: Square, to: Square) {
    moves.push(Move::new_promotion(from, to, PieceKind::Queen));
    moves.push(Move::new_promotion(from, to, PieceKind::Rook));
    moves.push(Move::new_promotion(from, to, PieceKind::Bishop));
    moves.push(Move::new_promotion(from, to, PieceKind::Knight));
}

fn generate_castling(board: &Board, us: Color, occ: Bitboard, king_sq: Square, moves: &mut Vec<Move>) {
    match us {
        Color::White => {
            // Kingside: e1-g1, need f1,g1 empty, e1,f1,g1 not attacked
            if board.castling & WK_CASTLE != 0
                && occ & (bb(5) | bb(6)) == 0
                && !board.is_attacked_by(4, Color::Black)
                && !board.is_attacked_by(5, Color::Black)
                && !board.is_attacked_by(6, Color::Black)
            {
                moves.push(Move::new_castling(king_sq, 6));
            }
            // Queenside: e1-c1, need b1,c1,d1 empty, e1,d1,c1 not attacked
            if board.castling & WQ_CASTLE != 0
                && occ & (bb(1) | bb(2) | bb(3)) == 0
                && !board.is_attacked_by(4, Color::Black)
                && !board.is_attacked_by(3, Color::Black)
                && !board.is_attacked_by(2, Color::Black)
            {
                moves.push(Move::new_castling(king_sq, 2));
            }
        }
        Color::Black => {
            // Kingside: e8-g8
            if board.castling & BK_CASTLE != 0
                && occ & (bb(61) | bb(62)) == 0
                && !board.is_attacked_by(60, Color::White)
                && !board.is_attacked_by(61, Color::White)
                && !board.is_attacked_by(62, Color::White)
            {
                moves.push(Move::new_castling(king_sq, 62));
            }
            // Queenside: e8-c8
            if board.castling & BQ_CASTLE != 0
                && occ & (bb(57) | bb(58) | bb(59)) == 0
                && !board.is_attacked_by(60, Color::White)
                && !board.is_attacked_by(59, Color::White)
                && !board.is_attacked_by(58, Color::White)
            {
                moves.push(Move::new_castling(king_sq, 58));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_startpos_moves() {
        let board = Board::startpos();
        let moves = generate_moves(&board);
        // Starting position has exactly 20 legal moves
        assert_eq!(moves.len(), 20, "Expected 20 moves, got {}: {:?}", moves.len(), moves);
    }

    #[test]
    fn test_perft_startpos_depth1() {
        let board = Board::startpos();
        assert_eq!(perft(&board, 1), 20);
    }

    #[test]
    fn test_perft_startpos_depth2() {
        let board = Board::startpos();
        assert_eq!(perft(&board, 2), 400);
    }

    #[test]
    fn test_perft_startpos_depth3() {
        let board = Board::startpos();
        assert_eq!(perft(&board, 3), 8902);
    }

    #[test]
    fn test_perft_startpos_depth4() {
        let board = Board::startpos();
        assert_eq!(perft(&board, 4), 197281);
    }

    /// Kiwipete position — a well-known perft test position.
    #[test]
    fn test_perft_kiwipete_depth1() {
        let board = Board::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .unwrap();
        assert_eq!(perft(&board, 1), 48);
    }

    #[test]
    fn test_perft_kiwipete_depth2() {
        let board = Board::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .unwrap();
        assert_eq!(perft(&board, 2), 2039);
    }

    #[test]
    fn test_perft_kiwipete_depth3() {
        let board = Board::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .unwrap();
        assert_eq!(perft(&board, 3), 97862);
    }

    /// Position 3 from the Chess Programming Wiki perft results page.
    #[test]
    fn test_perft_position3_depth4() {
        let board = Board::from_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1").unwrap();
        assert_eq!(perft(&board, 1), 14);
        assert_eq!(perft(&board, 2), 191);
        assert_eq!(perft(&board, 3), 2812);
        assert_eq!(perft(&board, 4), 43238);
    }

    /// Perft function for testing move generation correctness.
    fn perft(board: &Board, depth: u32) -> u64 {
        if depth == 0 {
            return 1;
        }
        let moves = generate_moves(board);
        if depth == 1 {
            return moves.len() as u64;
        }
        let mut count = 0;
        let mut board = board.clone();
        for m in moves {
            let undo = board.make_move(m);
            count += perft(&board, depth - 1);
            board.unmake_move(m, undo);
        }
        count
    }
}
