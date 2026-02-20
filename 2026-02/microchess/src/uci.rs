use std::io::{self, BufRead};

use crate::board::Board;
use crate::search;
use crate::types::Move;

pub fn uci_loop() {
    let stdin = io::stdin();
    let mut board = Board::startpos();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens[0] {
            "uci" => {
                println!("id name Microchess");
                println!("id author Claude");
                println!("uciok");
            }
            "isready" => {
                println!("readyok");
            }
            "ucinewgame" => {
                board = Board::startpos();
            }
            "position" => {
                parse_position(&tokens, &mut board);
            }
            "go" => {
                let time_ms = parse_go(&tokens, &board);
                let best = search::search(&board, time_ms);
                println!("bestmove {}", best);
            }
            "quit" => {
                break;
            }
            "d" => {
                // Debug: display board
                eprint!("{}", board.display());
                eprintln!("FEN: {}", board.to_fen());
            }
            _ => {}
        }
    }
}

fn parse_position(tokens: &[&str], board: &mut Board) {
    let mut idx = 1;
    if idx >= tokens.len() {
        return;
    }

    if tokens[idx] == "startpos" {
        *board = Board::startpos();
        idx += 1;
    } else if tokens[idx] == "fen" {
        idx += 1;
        let mut fen_parts = Vec::new();
        while idx < tokens.len() && tokens[idx] != "moves" {
            fen_parts.push(tokens[idx]);
            idx += 1;
        }
        let fen = fen_parts.join(" ");
        if let Some(b) = Board::from_fen(&fen) {
            *board = b;
        }
    }

    // Parse moves
    if idx < tokens.len() && tokens[idx] == "moves" {
        idx += 1;
        while idx < tokens.len() {
            if let Some(m) = Move::from_uci(tokens[idx], board) {
                board.make_move(m);
            }
            idx += 1;
        }
    }
}

fn parse_go(tokens: &[&str], board: &Board) -> u64 {
    let mut wtime: u64 = 0;
    let mut btime: u64 = 0;
    let mut winc: u64 = 0;
    let mut binc: u64 = 0;
    let mut movetime: u64 = 0;
    let mut _depth: u32 = 0;

    let mut i = 1;
    while i < tokens.len() {
        match tokens[i] {
            "wtime" => {
                i += 1;
                wtime = tokens.get(i).and_then(|s| s.parse().ok()).unwrap_or(0);
            }
            "btime" => {
                i += 1;
                btime = tokens.get(i).and_then(|s| s.parse().ok()).unwrap_or(0);
            }
            "winc" => {
                i += 1;
                winc = tokens.get(i).and_then(|s| s.parse().ok()).unwrap_or(0);
            }
            "binc" => {
                i += 1;
                binc = tokens.get(i).and_then(|s| s.parse().ok()).unwrap_or(0);
            }
            "movetime" => {
                i += 1;
                movetime = tokens.get(i).and_then(|s| s.parse().ok()).unwrap_or(0);
            }
            "depth" => {
                i += 1;
                _depth = tokens.get(i).and_then(|s| s.parse().ok()).unwrap_or(0);
            }
            "infinite" => {
                return 1_000_000; // 1000 seconds
            }
            _ => {}
        }
        i += 1;
    }

    if movetime > 0 {
        return movetime;
    }

    // Time management: use ~1/30 of remaining time + increment
    let (time, inc) = match board.side {
        crate::types::Color::White => (wtime, winc),
        crate::types::Color::Black => (btime, binc),
    };

    if time > 0 {
        let think_time = time / 30 + inc * 3 / 4;
        // Don't use more than 1/3 of remaining time
        let max_time = time / 3;
        think_time.min(max_time).max(50) // At least 50ms
    } else {
        5000 // Default 5 seconds
    }
}
