/// Core types for the chess engine.

pub type Bitboard = u64;
pub type Square = u8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    pub fn flip(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PieceKind {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Piece {
    pub kind: PieceKind,
    pub color: Color,
}

/// A chess move, packed into a u32.
/// Bits 0-5: from square
/// Bits 6-11: to square
/// Bits 12-13: promotion piece (0=Knight, 1=Bishop, 2=Rook, 3=Queen)
/// Bit 14: is promotion
/// Bit 15: is en passant
/// Bit 16: is castling
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Move(pub u32);

impl Move {
    pub const NULL: Move = Move(0);

    pub fn new(from: Square, to: Square) -> Move {
        Move((from as u32) | ((to as u32) << 6))
    }

    pub fn new_promotion(from: Square, to: Square, promo: PieceKind) -> Move {
        let promo_bits = match promo {
            PieceKind::Knight => 0,
            PieceKind::Bishop => 1,
            PieceKind::Rook => 2,
            PieceKind::Queen => 3,
            _ => 3,
        };
        Move((from as u32) | ((to as u32) << 6) | (promo_bits << 12) | (1 << 14))
    }

    pub fn new_en_passant(from: Square, to: Square) -> Move {
        Move((from as u32) | ((to as u32) << 6) | (1 << 15))
    }

    pub fn new_castling(from: Square, to: Square) -> Move {
        Move((from as u32) | ((to as u32) << 6) | (1 << 16))
    }

    pub fn from_sq(self) -> Square {
        (self.0 & 0x3F) as Square
    }

    pub fn to_sq(self) -> Square {
        ((self.0 >> 6) & 0x3F) as Square
    }

    pub fn is_promotion(self) -> bool {
        (self.0 & (1 << 14)) != 0
    }

    pub fn promotion_piece(self) -> PieceKind {
        match (self.0 >> 12) & 3 {
            0 => PieceKind::Knight,
            1 => PieceKind::Bishop,
            2 => PieceKind::Rook,
            _ => PieceKind::Queen,
        }
    }

    pub fn is_en_passant(self) -> bool {
        (self.0 & (1 << 15)) != 0
    }

    pub fn is_castling(self) -> bool {
        (self.0 & (1 << 16)) != 0
    }

    /// Convert to UCI string like "e2e4" or "e7e8q"
    pub fn to_uci(self) -> String {
        let from = self.from_sq();
        let to = self.to_sq();
        let from_file = (b'a' + (from % 8)) as char;
        let from_rank = (b'1' + (from / 8)) as char;
        let to_file = (b'a' + (to % 8)) as char;
        let to_rank = (b'1' + (to / 8)) as char;

        let mut s = format!("{}{}{}{}", from_file, from_rank, to_file, to_rank);
        if self.is_promotion() {
            let c = match self.promotion_piece() {
                PieceKind::Knight => 'n',
                PieceKind::Bishop => 'b',
                PieceKind::Rook => 'r',
                PieceKind::Queen => 'q',
                _ => 'q',
            };
            s.push(c);
        }
        s
    }

    /// Parse from UCI string like "e2e4" or "e7e8q".
    /// Needs board context to detect en passant and castling.
    pub fn from_uci(s: &str, board: &crate::board::Board) -> Option<Move> {
        let bytes = s.as_bytes();
        if bytes.len() < 4 {
            return None;
        }
        let from_file = bytes[0].wrapping_sub(b'a');
        let from_rank = bytes[1].wrapping_sub(b'1');
        let to_file = bytes[2].wrapping_sub(b'a');
        let to_rank = bytes[3].wrapping_sub(b'1');

        if from_file > 7 || from_rank > 7 || to_file > 7 || to_rank > 7 {
            return None;
        }

        let from = from_rank * 8 + from_file;
        let to = to_rank * 8 + to_file;

        // Check for promotion
        if bytes.len() >= 5 {
            let promo = match bytes[4] {
                b'n' => PieceKind::Knight,
                b'b' => PieceKind::Bishop,
                b'r' => PieceKind::Rook,
                b'q' => PieceKind::Queen,
                _ => return None,
            };
            return Some(Move::new_promotion(from, to, promo));
        }

        // Check for castling (king moves 2 squares)
        if let Some(piece) = board.piece_at(from) {
            if piece.kind == PieceKind::King && from_file.abs_diff(to_file) == 2 {
                return Some(Move::new_castling(from, to));
            }
            // Check for en passant
            if piece.kind == PieceKind::Pawn && to_file != from_file {
                if board.piece_at(to).is_none() {
                    return Some(Move::new_en_passant(from, to));
                }
            }
        }

        Some(Move::new(from, to))
    }
}

impl std::fmt::Debug for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_uci())
    }
}

impl std::fmt::Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_uci())
    }
}

// Square helpers
pub const fn sq(file: u8, rank: u8) -> Square {
    rank * 8 + file
}

pub const fn file_of(sq: Square) -> u8 {
    sq % 8
}

pub const fn rank_of(sq: Square) -> u8 {
    sq / 8
}

pub fn sq_name(s: Square) -> String {
    let f = (b'a' + file_of(s)) as char;
    let r = (b'1' + rank_of(s)) as char;
    format!("{}{}", f, r)
}

// Bitboard helpers
pub const fn bb(sq: Square) -> Bitboard {
    1u64 << sq
}

pub fn bb_iter(mut b: Bitboard) -> impl Iterator<Item = Square> {
    std::iter::from_fn(move || {
        if b == 0 {
            None
        } else {
            let sq = b.trailing_zeros() as Square;
            b &= b - 1;
            Some(sq)
        }
    })
}

pub fn popcount(b: Bitboard) -> u32 {
    b.count_ones()
}
