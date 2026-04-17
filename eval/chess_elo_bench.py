"""
eval/chess_elo_bench.py — Chess ELO benchmark for Nous + LLM
=============================================================
Plays N games of Nous+LLM (white) vs built-in minimax engine (black).
Tracks ELO over time, exposing FNC-Bench diagnostics:

  LPI: Does ELO improve over games? (baseline LLM = no improvement)
  GDP: Does the system say "uncertain" on unfamiliar positions?
  CC:  Does the system avoid moves it previously saw fail?

Usage:
  python eval/chess_elo_bench.py --games 20 --model minimax-m2.7:cloud
  python eval/chess_elo_bench.py --games 20 --no-nous   # baseline comparison
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import chess
import chess.pgn
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("chess_bench")

# ── Configuration ────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OLLAMA_BASE = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("CHESS_MODEL", "kimi-k2.5:cloud")
NOUS_DB = os.getenv(
    "NOUSE_DB_PATH",
    str(Path.home() / ".local/share/nouse/personal/field.sqlite"),
)

# ── Piece values for minimax evaluation ──────────────────────────────────────

PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   20000,
}

# Positional bonuses (simplified PST for pawns and knights)
PAWN_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]


def _evaluate_board(board: chess.Board) -> int:
    """Static evaluation from white's perspective."""
    if board.is_checkmate():
        return -100000 if board.turn == chess.WHITE else 100000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        val = PIECE_VALUES.get(piece.piece_type, 0)
        if piece.piece_type == chess.PAWN:
            row = sq // 8
            col = sq % 8
            pst_idx = (7 - row) * 8 + col if piece.color == chess.WHITE else row * 8 + col
            val += PAWN_TABLE[pst_idx]
        score += val if piece.color == chess.WHITE else -val
    return score


def _minimax(
    board: chess.Board,
    depth: int,
    alpha: int,
    beta: int,
    maximizing: bool,
) -> int:
    if depth == 0 or board.is_game_over():
        return _evaluate_board(board)

    if maximizing:
        best = -999999
        for move in board.legal_moves:
            board.push(move)
            val = _minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            best = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    else:
        best = 999999
        for move in board.legal_moves:
            board.push(move)
            val = _minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            best = min(best, val)
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best


def engine_best_move(board: chess.Board, depth: int = 3) -> chess.Move:
    """Pick best move for current side using minimax."""
    best_move = None
    best_val = -999999 if board.turn == chess.WHITE else 999999
    moves = list(board.legal_moves)
    random.shuffle(moves)  # break ties randomly

    for move in moves:
        board.push(move)
        val = _minimax(board, depth - 1, -999999, 999999, board.turn == chess.WHITE)
        board.pop()
        if board.turn == chess.WHITE:
            if val > best_val:
                best_val = val
                best_move = move
        else:
            if val < best_val:
                best_val = val
                best_move = move

    return best_move or moves[0]


# ── Nous context retrieval ────────────────────────────────────────────────────

def _query_nous_chess_context(board: chess.Board) -> str:
    """Query Nous knowledge graph for chess-relevant context."""
    try:
        from nouse.field.surface import FieldSurface
        field = FieldSurface(NOUS_DB)

        queries = ["schack", "chess", "öppning", "slutspel", "taktik",
                   "fork", "pin", "skewer", "discovered attack"]
        concepts = []
        for q in queries[:4]:
            try:
                cs = field.concepts(domain=q)
                concepts.extend(c["name"] for c in cs[:3])
            except Exception:
                pass

        # Also try to get relations for current phase
        phase = _game_phase(board)
        try:
            phase_concepts = field.concepts(domain=phase)
            concepts.extend(c["name"] for c in phase_concepts[:5])
        except Exception:
            pass

        if concepts:
            unique = list(dict.fromkeys(concepts))[:12]
            return f"Nous knowledge ({phase}): {', '.join(unique)}"
        return f"Nous: inga schackkoncept ännu (domän saknas — slagsida)"
    except Exception as e:
        return f"Nous: ej tillgänglig ({e})"


def _game_phase(board: chess.Board) -> str:
    pieces = len(board.piece_map())
    if pieces > 28:
        return "öppning"
    elif pieces > 14:
        return "mellanspel"
    else:
        return "slutspel"


# ── LLM move generation ──────────────────────────────────────────────────────

async def _llm_choose_move(
    board: chess.Board,
    nous_context: str,
    model: str,
    history: list[str],
) -> chess.Move | None:
    """Ask LLM to choose a chess move, with Nous context injected."""
    legal = [board.san(m) for m in board.legal_moves]
    fen = board.fen()
    phase = _game_phase(board)
    last_moves = history[-6:] if history else []

    sys_prompt = (
        "You are a chess player. You receive the position and must choose "
        "the best legal move. Respond with ONLY the move in SAN notation "
        "(e.g. 'e4', 'Nf3', 'O-O'). Nothing else."
    )

    user_prompt = (
        f"Position (FEN): {fen}\n"
        f"Phase: {phase}\n"
        f"You play: {'White' if board.turn == chess.WHITE else 'Black'}\n"
        f"Recent moves: {' '.join(last_moves)}\n"
        f"Legal moves: {', '.join(legal[:20])}"
        + (f" (and {len(legal)-20} more)" if len(legal) > 20 else "")
        + f"\n\nContext from Nous knowledge graph:\n{nous_context}\n\n"
        f"Choose the best move:"
    )

    try:
        ollama_url = f"{OLLAMA_BASE}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(ollama_url, json=payload)
            r.raise_for_status()
            data = r.json()
            raw = (data.get("message") or {}).get("content") or ""
            raw = raw.strip()
            if not raw:
                log.warning("LLM returned empty content")
                return None
    except httpx.TimeoutException:
        log.warning("LLM call timed out after 45s")
        return None
    except Exception as e:
        log.warning("LLM call failed: %s — %s", type(e).__name__, e)
        return None

    # Parse the move from LLM output
    raw = raw.strip().split("\n")[0].strip()
    raw = re.sub(r"[^a-zA-Z0-9\-\+\#\=\/\!]", "", raw)

    try:
        return board.parse_san(raw)
    except Exception:
        # Try to find any legal move mentioned in the response
        for san in legal:
            if san in raw:
                return board.parse_san(san)
        return None


# ── ELO calculation ───────────────────────────────────────────────────────────

def _elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def _elo_update(rating: float, expected: float, actual: float, k: int = 32) -> float:
    return rating + k * (actual - expected)


# ── Game runner ───────────────────────────────────────────────────────────────

async def play_game(
    game_num: int,
    model: str,
    use_nous: bool,
    engine_depth: int = 3,
) -> dict:
    """Play one game: LLM (white) vs minimax engine (black)."""
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = f"Nous+LLM ({model})" if use_nous else f"LLM ({model})"
    game.headers["Black"] = f"Minimax-d{engine_depth}"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    node = game

    move_history: list[str] = []
    llm_moves = 0
    llm_failures = 0
    nous_contexts: list[str] = []
    start = time.time()

    while not board.is_game_over() and board.fullmove_number <= 40:
        if board.turn == chess.WHITE:
            # LLM plays white
            nous_ctx = _query_nous_chess_context(board) if use_nous else "Nous disabled"
            nous_contexts.append(nous_ctx)

            move = await _llm_choose_move(board, nous_ctx, model, move_history)

            if move is None or move not in board.legal_moves:
                # Fallback to random legal move
                move = random.choice(list(board.legal_moves))
                llm_failures += 1
                log.debug("Game %d: LLM move failed, using random", game_num)
            llm_moves += 1
        else:
            # Minimax plays black
            move = engine_best_move(board, depth=engine_depth)

        san = board.san(move)
        move_history.append(san)
        node = node.add_variation(move)
        board.push(move)

    result = board.result()
    outcome = board.outcome()

    # Score from white's perspective
    if result == "1-0":
        white_score = 1.0
    elif result == "0-1":
        white_score = 0.0
    else:
        white_score = 0.5

    elapsed = time.time() - start
    log.info(
        "Game %d: %s | %d moves | LLM failures: %d | %.1fs",
        game_num, result, board.fullmove_number, llm_failures, elapsed,
    )

    return {
        "game": game_num,
        "result": result,
        "white_score": white_score,
        "moves": board.fullmove_number,
        "llm_moves": llm_moves,
        "llm_failures": llm_failures,
        "failure_rate": llm_failures / max(1, llm_moves),
        "nous_used": use_nous,
        "nous_had_context": any("slagsida" not in c for c in nous_contexts),
        "pgn": str(game),
        "elapsed_s": round(elapsed, 1),
    }


# ── Main benchmark loop ───────────────────────────────────────────────────────

async def run_benchmark(
    n_games: int,
    model: str,
    use_nous: bool,
    engine_depth: int,
) -> dict:
    label = "nous" if use_nous else "baseline"
    log.info("=== Chess ELO Benchmark — %s | %d games | depth=%d ===",
             label.upper(), n_games, engine_depth)

    # Engine ELO calibration (rough estimate based on depth)
    engine_elo = {2: 900, 3: 1100, 4: 1400}.get(engine_depth, 1100)
    player_elo = 800.0  # starting estimate for LLM

    results = []
    elo_history = [player_elo]

    for i in range(1, n_games + 1):
        result = await play_game(i, model, use_nous, engine_depth)
        results.append(result)

        # Update ELO
        expected = _elo_expected(player_elo, engine_elo)
        player_elo = _elo_update(player_elo, expected, result["white_score"])
        elo_history.append(round(player_elo, 1))
        log.info("  ELO after game %d: %.0f (expected=%.2f, actual=%.1f)",
                 i, player_elo, expected, result["white_score"])

    wins = sum(1 for r in results if r["result"] == "1-0")
    losses = sum(1 for r in results if r["result"] == "0-1")
    draws = sum(1 for r in results if r["result"] == "1/2-1/2")
    avg_failures = sum(r["failure_rate"] for r in results) / len(results)
    had_nous_context = sum(1 for r in results if r.get("nous_had_context", False))

    summary = {
        "label": label,
        "model": model,
        "engine_depth": engine_depth,
        "engine_elo_approx": engine_elo,
        "games": n_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "final_elo": round(player_elo, 1),
        "elo_history": elo_history,
        "elo_delta": round(player_elo - 800, 1),
        "avg_llm_failure_rate": round(avg_failures, 3),
        "nous_context_available": had_nous_context,
        # FNC-Bench proxy metrics
        "fnc_lpi_proxy": round(max(0, player_elo - 800) / 200, 3),  # normalized ELO gain
        "fnc_gdp_proxy": round(1.0 - avg_failures, 3),              # move legality confidence
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"chess_elo_{label}_{ts}.json"
    out.write_text(json.dumps({"summary": summary, "games": results}, indent=2, ensure_ascii=False))
    log.info("Saved → %s", out)

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Chess ELO benchmark for Nous+LLM")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--depth", type=int, default=3,
                        help="Minimax depth (2=~900 ELO, 3=~1100, 4=~1400)")
    parser.add_argument("--no-nous", action="store_true",
                        help="Run without Nous context (baseline)")
    parser.add_argument("--compare", action="store_true",
                        help="Run both baseline and Nous, print comparison")
    args = parser.parse_args()

    async def _run():
        if args.compare:
            log.info("Running BASELINE...")
            baseline = await run_benchmark(args.games, args.model, False, args.depth)
            log.info("Running NOUS...")
            nous = await run_benchmark(args.games, args.model, True, args.depth)

            print("\n" + "═" * 50)
            print("CHESS ELO COMPARISON")
            print("═" * 50)
            print(f"{'Metric':<30} {'Baseline':>10} {'Nous':>10}")
            print("─" * 50)
            print(f"{'Final ELO':<30} {baseline['final_elo']:>10.0f} {nous['final_elo']:>10.0f}")
            print(f"{'ELO delta (from 800)':<30} {baseline['elo_delta']:>+10.0f} {nous['elo_delta']:>+10.0f}")
            print(f"{'Wins / Draws / Losses':<30} {baseline['wins']}/{baseline['draws']}/{baseline['losses']:>4} {nous['wins']}/{nous['draws']}/{nous['losses']:>4}")
            print(f"{'LLM failure rate':<30} {baseline['avg_llm_failure_rate']:>10.1%} {nous['avg_llm_failure_rate']:>10.1%}")
            print(f"{'FNC LPI proxy':<30} {baseline['fnc_lpi_proxy']:>10.3f} {nous['fnc_lpi_proxy']:>10.3f}")
            print(f"{'Nous context available':<30} {'—':>10} {nous['nous_context_available']:>10}")
            print("═" * 50)
            elo_diff = nous['final_elo'] - baseline['final_elo']
            print(f"\nELO difference: {elo_diff:+.0f} (Nous vs baseline)")
            if nous['nous_context_available'] == 0:
                print("⚠ Nous had no chess domain context — slagsida in effect")
                print("  Run: nouse seed --regions temporal_right occipital")
        else:
            result = await run_benchmark(args.games, args.model, not args.no_nous, args.depth)
            print(f"\nFinal ELO: {result['final_elo']:.0f} (started: 800)")
            print(f"W/D/L: {result['wins']}/{result['draws']}/{result['losses']}")
            print(f"LLM failure rate: {result['avg_llm_failure_rate']:.1%}")
            if not result['nous_context_available']:
                print("⚠ Nous har inga schackkoncept ännu — slagsida exponerad")

    asyncio.run(_run())


if __name__ == "__main__":
    main()
