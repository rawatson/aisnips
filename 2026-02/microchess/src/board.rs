use crate::types::*;

/// Castling rights packed into a u8.
/// Bit 0: White kingside
/// Bit 1: White queenside
/// Bit 2: Black kingside
/// Bit 3: Black queenside
pub type CastlingRights = u8;

pub const WK_CASTLE: u8 = 1;
pub const WQ_CASTLE: u8 = 2;
pub const BK_CASTLE: u8 = 4;
pub const BQ_CASTLE: u8 = 8;

/// Undo information for a move.
#[derive(Clone, Copy)]
pub struct UndoInfo {
    pub captured: Option<Piece>,
    pub castling: CastlingRights,
    pub en_passant: Option<Square>,
    pub halfmove: u16,
    pub moved_piece: Piece,
}

#[derive(Clone)]
pub struct Board {
    /// Bitboards per piece type
    pub pieces: [Bitboard; 6],
    /// Bitboards per color
    pub colors: [Bitboard; 2],
    /// Mailbox
    pub squares: [Option<Piece>; 64],
    /// Side to move
    pub side: Color,
    /// Castling rights
    pub castling: CastlingRights,
    /// En passant target square (the square the capturing pawn lands on)
    pub en_passant: Option<Square>,
    /// Halfmove clock (for 50-move rule)
    pub halfmove: u16,
    /// Fullmove counter
    pub fullmove: u16,
}

impl Board {
    pub fn new() -> Board {
        Board {
            pieces: [0; 6],
            colors: [0; 2],
            squares: [None; 64],
            side: Color::White,
            castling: 0,
            en_passant: None,
            halfmove: 0,
            fullmove: 1,
        }
    }

    pub fn startpos() -> Board {
        Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap()
    }

    pub fn piece_at(&self, sq: Square) -> Option<Piece> {
        self.squares[sq as usize]
    }

    pub fn put_piece(&mut self, sq: Square, piece: Piece) {
        let b = bb(sq);
        self.pieces[piece.kind as usize] |= b;
        self.colors[piece.color as usize] |= b;
        self.squares[sq as usize] = Some(piece);
    }

    pub fn remove_piece(&mut self, sq: Square) {
        if let Some(piece) = self.squares[sq as usize] {
            let b = bb(sq);
            self.pieces[piece.kind as usize] &= !b;
            self.colors[piece.color as usize] &= !b;
            self.squares[sq as usize] = None;
        }
    }

    pub fn occupancy(&self) -> Bitboard {
        self.colors[0] | self.colors[1]
    }

    pub fn piece_bb(&self, kind: PieceKind, color: Color) -> Bitboard {
        self.pieces[kind as usize] & self.colors[color as usize]
    }

    /// Check if a square is attacked by the given color.
    pub fn is_attacked_by(&self, sq: Square, by_color: Color) -> bool {
        let occ = self.occupancy();

        // Pawn attacks
        let pawn_bb = self.piece_bb(PieceKind::Pawn, by_color);
        if pawn_attacks(by_color.flip(), sq) & pawn_bb != 0 {
            return true;
        }

        // Knight attacks
        if KNIGHT_ATTACKS[sq as usize] & self.piece_bb(PieceKind::Knight, by_color) != 0 {
            return true;
        }

        // King attacks
        if KING_ATTACKS[sq as usize] & self.piece_bb(PieceKind::King, by_color) != 0 {
            return true;
        }

        // Bishop/Queen (diagonal)
        let diag_attackers =
            self.piece_bb(PieceKind::Bishop, by_color) | self.piece_bb(PieceKind::Queen, by_color);
        if bishop_attacks(sq, occ) & diag_attackers != 0 {
            return true;
        }

        // Rook/Queen (straight)
        let straight_attackers =
            self.piece_bb(PieceKind::Rook, by_color) | self.piece_bb(PieceKind::Queen, by_color);
        if rook_attacks(sq, occ) & straight_attackers != 0 {
            return true;
        }

        false
    }

    pub fn in_check(&self) -> bool {
        let king_sq = (self.piece_bb(PieceKind::King, self.side)).trailing_zeros() as Square;
        self.is_attacked_by(king_sq, self.side.flip())
    }

    /// Make a move, returning undo information.
    pub fn make_move(&mut self, m: Move) -> UndoInfo {
        let from = m.from_sq();
        let to = m.to_sq();
        let piece = self.squares[from as usize].unwrap();
        let captured = if m.is_en_passant() {
            // Captured pawn is on a different square
            let cap_sq = match self.side {
                Color::White => to - 8,
                Color::Black => to + 8,
            };
            let cap = self.squares[cap_sq as usize];
            self.remove_piece(cap_sq);
            cap
        } else {
            let cap = self.squares[to as usize];
            if cap.is_some() {
                self.remove_piece(to);
            }
            cap
        };

        let undo = UndoInfo {
            captured,
            castling: self.castling,
            en_passant: self.en_passant,
            halfmove: self.halfmove,
            moved_piece: piece,
        };

        // Move the piece
        self.remove_piece(from);

        if m.is_promotion() {
            self.put_piece(
                to,
                Piece {
                    kind: m.promotion_piece(),
                    color: piece.color,
                },
            );
        } else {
            self.put_piece(to, piece);
        }

        // Handle castling move (move the rook)
        if m.is_castling() {
            let (rook_from, rook_to) = match to {
                6 => (7, 5),   // White kingside: h1->f1
                2 => (0, 3),   // White queenside: a1->d1
                62 => (63, 61), // Black kingside: h8->f8
                58 => (56, 59), // Black queenside: a8->d8
                _ => unreachable!(),
            };
            let rook = self.squares[rook_from as usize].unwrap();
            self.remove_piece(rook_from);
            self.put_piece(rook_to, rook);
        }

        // Update castling rights
        // If king moves, lose both castling rights
        if piece.kind == PieceKind::King {
            match piece.color {
                Color::White => self.castling &= !(WK_CASTLE | WQ_CASTLE),
                Color::Black => self.castling &= !(BK_CASTLE | BQ_CASTLE),
            }
        }
        // If rook moves from or is captured on its starting square
        self.update_castling_for_square(from);
        self.update_castling_for_square(to);

        // Update en passant
        self.en_passant = if piece.kind == PieceKind::Pawn && from.abs_diff(to) == 16 {
            Some((from + to) / 2)
        } else {
            None
        };

        // Update halfmove clock
        if piece.kind == PieceKind::Pawn || captured.is_some() {
            self.halfmove = 0;
        } else {
            self.halfmove += 1;
        }

        if self.side == Color::Black {
            self.fullmove += 1;
        }

        self.side = self.side.flip();
        undo
    }

    pub fn unmake_move(&mut self, m: Move, undo: UndoInfo) {
        self.side = self.side.flip();

        if self.side == Color::Black {
            self.fullmove -= 1;
        }

        let from = m.from_sq();
        let to = m.to_sq();

        // Remove piece from destination
        self.remove_piece(to);

        // Restore moving piece to source
        self.put_piece(from, undo.moved_piece);

        // Restore captured piece
        if m.is_en_passant() {
            if let Some(cap) = undo.captured {
                let cap_sq = match self.side {
                    Color::White => to - 8,
                    Color::Black => to + 8,
                };
                self.put_piece(cap_sq, cap);
            }
        } else if let Some(cap) = undo.captured {
            self.put_piece(to, cap);
        }

        // Undo castling rook move
        if m.is_castling() {
            let (rook_from, rook_to) = match to {
                6 => (7, 5),
                2 => (0, 3),
                62 => (63, 61),
                58 => (56, 59),
                _ => unreachable!(),
            };
            let rook = self.squares[rook_to as usize].unwrap();
            self.remove_piece(rook_to);
            self.put_piece(rook_from, rook);
        }

        self.castling = undo.castling;
        self.en_passant = undo.en_passant;
        self.halfmove = undo.halfmove;
    }

    fn update_castling_for_square(&mut self, sq: Square) {
        match sq {
            0 => self.castling &= !WQ_CASTLE,
            7 => self.castling &= !WK_CASTLE,
            56 => self.castling &= !BQ_CASTLE,
            63 => self.castling &= !BK_CASTLE,
            _ => {}
        }
    }

    /// Parse FEN string.
    pub fn from_fen(fen: &str) -> Option<Board> {
        let mut board = Board::new();
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 4 {
            return None;
        }

        // Piece placement
        let mut rank = 7u8;
        let mut file = 0u8;
        for ch in parts[0].chars() {
            match ch {
                '/' => {
                    rank = rank.wrapping_sub(1);
                    file = 0;
                }
                '1'..='8' => {
                    file += (ch as u8) - b'0';
                }
                _ => {
                    let color = if ch.is_uppercase() {
                        Color::White
                    } else {
                        Color::Black
                    };
                    let kind = match ch.to_ascii_lowercase() {
                        'p' => PieceKind::Pawn,
                        'n' => PieceKind::Knight,
                        'b' => PieceKind::Bishop,
                        'r' => PieceKind::Rook,
                        'q' => PieceKind::Queen,
                        'k' => PieceKind::King,
                        _ => return None,
                    };
                    board.put_piece(sq(file, rank), Piece { kind, color });
                    file += 1;
                }
            }
        }

        // Side to move
        board.side = match parts[1] {
            "w" => Color::White,
            "b" => Color::Black,
            _ => return None,
        };

        // Castling
        for ch in parts[2].chars() {
            match ch {
                'K' => board.castling |= WK_CASTLE,
                'Q' => board.castling |= WQ_CASTLE,
                'k' => board.castling |= BK_CASTLE,
                'q' => board.castling |= BQ_CASTLE,
                '-' => {}
                _ => return None,
            }
        }

        // En passant
        if parts[3] != "-" {
            let ep_bytes = parts[3].as_bytes();
            if ep_bytes.len() == 2 {
                let f = ep_bytes[0] - b'a';
                let r = ep_bytes[1] - b'1';
                board.en_passant = Some(sq(f, r));
            }
        }

        // Halfmove and fullmove
        if parts.len() > 4 {
            board.halfmove = parts[4].parse().unwrap_or(0);
        }
        if parts.len() > 5 {
            board.fullmove = parts[5].parse().unwrap_or(1);
        }

        Some(board)
    }

    pub fn to_fen(&self) -> String {
        let mut fen = String::new();

        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                match self.squares[sq(file, rank) as usize] {
                    None => empty += 1,
                    Some(piece) => {
                        if empty > 0 {
                            fen.push(char::from_digit(empty, 10).unwrap());
                            empty = 0;
                        }
                        let c = match piece.kind {
                            PieceKind::Pawn => 'p',
                            PieceKind::Knight => 'n',
                            PieceKind::Bishop => 'b',
                            PieceKind::Rook => 'r',
                            PieceKind::Queen => 'q',
                            PieceKind::King => 'k',
                        };
                        if piece.color == Color::White {
                            fen.push(c.to_ascii_uppercase());
                        } else {
                            fen.push(c);
                        }
                    }
                }
            }
            if empty > 0 {
                fen.push(char::from_digit(empty, 10).unwrap());
            }
            if rank > 0 {
                fen.push('/');
            }
        }

        fen.push(' ');
        fen.push(match self.side {
            Color::White => 'w',
            Color::Black => 'b',
        });

        fen.push(' ');
        if self.castling == 0 {
            fen.push('-');
        } else {
            if self.castling & WK_CASTLE != 0 {
                fen.push('K');
            }
            if self.castling & WQ_CASTLE != 0 {
                fen.push('Q');
            }
            if self.castling & BK_CASTLE != 0 {
                fen.push('k');
            }
            if self.castling & BQ_CASTLE != 0 {
                fen.push('q');
            }
        }

        fen.push(' ');
        match self.en_passant {
            None => fen.push('-'),
            Some(ep) => fen.push_str(&sq_name(ep)),
        }

        fen.push_str(&format!(" {} {}", self.halfmove, self.fullmove));
        fen
    }

    /// Simple text display for debugging.
    pub fn display(&self) -> String {
        let mut s = String::new();
        for rank in (0..8).rev() {
            s.push_str(&format!("{} ", rank + 1));
            for file in 0..8 {
                match self.squares[sq(file, rank) as usize] {
                    None => s.push('.'),
                    Some(piece) => {
                        let c = match piece.kind {
                            PieceKind::Pawn => 'p',
                            PieceKind::Knight => 'n',
                            PieceKind::Bishop => 'b',
                            PieceKind::Rook => 'r',
                            PieceKind::Queen => 'q',
                            PieceKind::King => 'k',
                        };
                        if piece.color == Color::White {
                            s.push(c.to_ascii_uppercase());
                        } else {
                            s.push(c);
                        }
                    }
                }
                s.push(' ');
            }
            s.push('\n');
        }
        s.push_str("  a b c d e f g h\n");
        s
    }
}

// ---- Attack tables (computed at startup) ----

pub static KNIGHT_ATTACKS: [Bitboard; 64] = {
    let mut table = [0u64; 64];
    let offsets: [(i8, i8); 8] = [
        (-2, -1),
        (-2, 1),
        (-1, -2),
        (-1, 2),
        (1, -2),
        (1, 2),
        (2, -1),
        (2, 1),
    ];
    let mut sq = 0;
    while sq < 64 {
        let r = (sq / 8) as i8;
        let f = (sq % 8) as i8;
        let mut i = 0;
        while i < 8 {
            let nr = r + offsets[i].0;
            let nf = f + offsets[i].1;
            if nr >= 0 && nr < 8 && nf >= 0 && nf < 8 {
                table[sq] |= 1u64 << (nr * 8 + nf);
            }
            i += 1;
        }
        sq += 1;
    }
    table
};

pub static KING_ATTACKS: [Bitboard; 64] = {
    let mut table = [0u64; 64];
    let offsets: [(i8, i8); 8] = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ];
    let mut sq = 0;
    while sq < 64 {
        let r = (sq / 8) as i8;
        let f = (sq % 8) as i8;
        let mut i = 0;
        while i < 8 {
            let nr = r + offsets[i].0;
            let nf = f + offsets[i].1;
            if nr >= 0 && nr < 8 && nf >= 0 && nf < 8 {
                table[sq] |= 1u64 << (nr * 8 + nf);
            }
            i += 1;
        }
        sq += 1;
    }
    table
};

/// Pawn attacks from a single square (used for attack detection).
pub fn pawn_attacks(color: Color, sq: Square) -> Bitboard {
    let b = bb(sq);
    match color {
        Color::White => ((b & !0x0101010101010101) << 7) | ((b & !0x8080808080808080) << 9),
        Color::Black => ((b & !0x8080808080808080) >> 7) | ((b & !0x0101010101010101) >> 9),
    }
}

/// Sliding piece attacks using classical approach (no magic bitboards for simplicity).
/// This is adequate for our target strength.
pub fn bishop_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    let mut attacks = 0u64;
    let directions: [(i8, i8); 4] = [(-1, -1), (-1, 1), (1, -1), (1, 1)];
    for &(dr, df) in &directions {
        let mut r = rank_of(sq) as i8 + dr;
        let mut f = file_of(sq) as i8 + df;
        while r >= 0 && r < 8 && f >= 0 && f < 8 {
            let s = (r * 8 + f) as u8;
            attacks |= bb(s);
            if occ & bb(s) != 0 {
                break;
            }
            r += dr;
            f += df;
        }
    }
    attacks
}

pub fn rook_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    let mut attacks = 0u64;
    let directions: [(i8, i8); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    for &(dr, df) in &directions {
        let mut r = rank_of(sq) as i8 + dr;
        let mut f = file_of(sq) as i8 + df;
        while r >= 0 && r < 8 && f >= 0 && f < 8 {
            let s = (r * 8 + f) as u8;
            attacks |= bb(s);
            if occ & bb(s) != 0 {
                break;
            }
            r += dr;
            f += df;
        }
    }
    attacks
}
