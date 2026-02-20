use std::time::Instant;

use crate::board::Board;
use crate::eval::{self, Score, DRAW_SCORE, MATE_SCORE};
use crate::movegen::generate_moves;
use crate::types::*;

pub struct SearchInfo {
    pub nodes: u64,
    pub start_time: Instant,
    pub time_limit_ms: u64,
    pub stopped: bool,
    pub root_best_move: Move,
    pub root_depth: i32,
}

impl SearchInfo {
    pub fn new(time_limit_ms: u64) -> SearchInfo {
        SearchInfo {
            nodes: 0,
            start_time: Instant::now(),
            time_limit_ms,
            stopped: false,
            root_best_move: Move::NULL,
            root_depth: 0,
        }
    }

    fn check_time(&mut self) {
        if self.nodes % 2048 == 0 {
            let elapsed = self.start_time.elapsed().as_millis() as u64;
            if elapsed >= self.time_limit_ms {
                self.stopped = true;
            }
        }
    }
}

/// Iterative deepening search. Returns the best move found.
pub fn search(board: &Board, time_limit_ms: u64) -> Move {
    let mut info = SearchInfo::new(time_limit_ms);
    let mut best_move = Move::NULL;
    let mut prev_best = Move::NULL;

    for depth in 1..=64i32 {
        info.stopped = false;
        info.root_depth = depth;
        info.root_best_move = Move::NULL;

        let score = alpha_beta(board, depth, -MATE_SCORE - 1, MATE_SCORE + 1, prev_best, true, &mut info);

        if info.stopped {
            break;
        }

        best_move = info.root_best_move;
        prev_best = best_move;

        // Print UCI info
        let elapsed = info.start_time.elapsed().as_millis().max(1) as u64;
        let nps = info.nodes * 1000 / elapsed;
        eprintln!(
            "info depth {} score cp {} nodes {} nps {} time {} pv {}",
            depth, score, info.nodes, nps, elapsed, best_move
        );

        // If we found a mate, stop searching
        if score.abs() > MATE_SCORE - 100 {
            break;
        }

        // Don't start a new iteration if we've used > 50% of time
        if elapsed > time_limit_ms / 2 {
            break;
        }
    }

    best_move
}

fn alpha_beta(
    board: &Board,
    depth: i32,
    mut alpha: Score,
    beta: Score,
    hint_move: Move,
    is_root: bool,
    info: &mut SearchInfo,
) -> Score {
    if info.stopped {
        return 0;
    }
    info.check_time();
    if info.stopped {
        return 0;
    }

    // Quiescence at depth 0
    if depth <= 0 {
        return quiescence(board, alpha, beta, info);
    }

    info.nodes += 1;

    let moves = generate_moves(board);

    // Checkmate or stalemate
    if moves.is_empty() {
        if board.in_check() {
            return -MATE_SCORE + (info.root_depth - depth) as Score;
        } else {
            return DRAW_SCORE;
        }
    }

    // 50-move rule
    if board.halfmove >= 100 {
        return DRAW_SCORE;
    }

    // Order moves: hint move first, then captures (MVV-LVA), then quiet moves
    let mut scored_moves: Vec<(Score, Move)> = moves
        .into_iter()
        .map(|m| {
            let score = if m == hint_move {
                100000 // Search the previous best move first
            } else {
                order_score(board, m)
            };
            (score, m)
        })
        .collect();
    scored_moves.sort_unstable_by(|a, b| b.0.cmp(&a.0));

    let mut best_move = scored_moves[0].1;

    for (_score, m) in scored_moves {
        let mut board = board.clone();
        let undo = board.make_move(m);

        let score = -alpha_beta(&board, depth - 1, -beta, -alpha, Move::NULL, false, info);

        board.unmake_move(m, undo);

        if info.stopped {
            return 0;
        }

        if score > alpha {
            alpha = score;
            best_move = m;

            if alpha >= beta {
                break; // Beta cutoff
            }
        }
    }

    if is_root {
        info.root_best_move = best_move;
    }

    alpha
}

fn quiescence(board: &Board, mut alpha: Score, beta: Score, info: &mut SearchInfo) -> Score {
    if info.stopped {
        return 0;
    }
    info.check_time();
    if info.stopped {
        return 0;
    }

    info.nodes += 1;

    let stand_pat = eval::evaluate(board);

    if stand_pat >= beta {
        return beta;
    }
    if stand_pat > alpha {
        alpha = stand_pat;
    }

    // Delta pruning: if we're so far behind that even capturing a queen won't help, bail
    if stand_pat + 1000 < alpha {
        return alpha;
    }

    // Generate only capture moves
    let moves = generate_moves(board);
    let mut captures: Vec<(Score, Move)> = moves
        .into_iter()
        .filter(|m| is_capture(board, *m))
        .map(|m| (eval::mvv_lva_score(board, m), m))
        .collect();
    captures.sort_unstable_by(|a, b| b.0.cmp(&a.0));

    for (_score, m) in captures {
        let mut board = board.clone();
        let undo = board.make_move(m);

        let score = -quiescence(&board, -beta, -alpha, info);

        board.unmake_move(m, undo);

        if info.stopped {
            return 0;
        }

        if score > alpha {
            alpha = score;
            if alpha >= beta {
                break;
            }
        }
    }

    alpha
}

fn is_capture(board: &Board, m: Move) -> bool {
    m.is_en_passant() || board.piece_at(m.to_sq()).is_some()
}

fn order_score(board: &Board, m: Move) -> Score {
    if is_capture(board, m) {
        10000 + eval::mvv_lva_score(board, m)
    } else if m.is_promotion() {
        9000
    } else {
        0
    }
}
