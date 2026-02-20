mod board;
mod eval;
mod movegen;
mod search;
mod types;
mod uci;

fn main() {
    uci::uci_loop();
}
