"""Benchmark the snake-solving algorithm over many headless games.

Runs SnakeGame.play_one_game() repeatedly (no pygame window, no frame delay)
and aggregates: score (apples eaten), percent of the board filled, ticks
(moves taken), and wall-clock time per game -- so you can see how the
algorithm's survivability (does it fill the board / how far does it get
before getting boxed in) and efficiency (how fast, how many moves) trend
across runs.

Usage:
    python3 Benchmark.py --games 50 --width 20 --height 20
    python3 Benchmark.py --games 100 --width 10 --height 10 --plot results.png
"""

import argparse
import json
import os
import statistics
import time
import signal

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from tqdm import tqdm
from random import random

from Main import SnakeGame

console = Console()


def run_single_game(width, height, seed, complete_max_free=25, timeout_seconds=30):
    game = SnakeGame(width, height, block_size=20, headless=True, verbose=False, first_game_seed=seed)
    game.search.complete_max_free = complete_max_free

    start = time.perf_counter()

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Game seed {seed} exceeded {timeout_seconds}s timeout")

    # Set up signal handler for timeout (Unix-like systems only)
    old_handler = None
    try:
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        filled_board = game.play_one_game()
        signal.alarm(0)  # Cancel alarm
    except TimeoutError as e:
        console = Console()
        console.print(f"[red]{e}[/red]")
        console.print(f"[yellow]Snake length at timeout: {len(game.snake)}, Ticks: {game.ticks}[/yellow]")
        elapsed = time.perf_counter() - start
        return {
            "seed": seed,
            "width": width,
            "height": height,
            "score": game.score(),
            "start_len": game.start_len,
            "length": len(game.snake),
            "cells": len(game.grid),
            "percent_filled": 100 * len(game.snake) / len(game.grid),
            "ticks": game.ticks,
            "point_tick_history": game.point_tick_history,
            "seconds": elapsed,
            "filled_board": False,
            "timed_out": True,
        }
    finally:
        if old_handler is not None:
            signal.signal(signal.SIGALRM, old_handler)

    elapsed = time.perf_counter() - start

    cells = len(game.grid)
    length = len(game.snake)
    return {
        "seed": seed,
        "width": width,
        "height": height,
        "score": game.score(),
        "start_len": game.start_len,
        "length": length,
        "cells": cells,
        "percent_filled": 100 * length / cells,
        "ticks": game.ticks,
        "point_ticks": game.avg_point_ticks,
        "point_tick_history": game.point_tick_history,
        "seconds": elapsed,
        "filled_board": filled_board,
        "timed_out": False,
    }


def run_benchmark(width, height, games, base_seed, complete_max_free, timeout_seconds=30):
    if base_seed == 0:
        base_seed = random()
    results = []
    for i in tqdm(range(games), desc=f"Benchmarking {width}x{height}, base_seed={base_seed}", unit="game"):
        results.append(run_single_game(width, height, base_seed + i, complete_max_free, timeout_seconds))
    return results


def save_results(results, path):
    """Append this run's per-game records to the JSON data file at path."""
    existing = load_results(path)
    existing.extend(results)
    with open(path, "w") as f:
        json.dump(existing, f)
    console.print(f"[bold]Saved {len(results)} game record(s) to {path} ({len(existing)} total)[/bold]")


def load_results(path):
    """Load previously saved per-game records from the JSON data file at path."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def fmt(values, digits=2):
    return f"{statistics.mean(values):.{digits}f}"


def summarize(results, width, height):
    n = len(results)
    scores = [r["score"] for r in results]
    percents = [r["percent_filled"] for r in results]
    point_ticks = [r["point_ticks"] for r in results]
    seconds = [r["seconds"] for r in results]
    wins = sum(1 for r in results if r["filled_board"])

    def stats_row(label, values, digits=2, suffix=""):
        return (
            label,
            f"{statistics.mean(values):.{digits}f}{suffix}",
            f"{statistics.median(values):.{digits}f}{suffix}",
            f"{min(values):.{digits}f}{suffix}",
            f"{max(values):.{digits}f}{suffix}",
            f"{statistics.pstdev(values):.{digits}f}{suffix}" if n > 1 else "-",
        )

    table = Table(title=f"Snake Algorithm Benchmark  ({width}x{height} board, {n} games)",
                  show_lines=False)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Mean", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Std Dev", justify="right")

    table.add_row(*stats_row("Score (apples eaten)", scores, digits=1))
    table.add_row(*stats_row("Board filled", percents, digits=2, suffix="%"))
    table.add_row(*stats_row("Average ticks (moves) per point", point_ticks, digits=1))
    table.add_row(*stats_row("Time per game", seconds, digits=4, suffix="s"))

    console.print(table)

    win_rate = 100 * wins / n
    summary_text = (
        f"[bold]{wins}/{n}[/bold] games filled the entire board "
        f"([bold]{win_rate:.1f}%[/bold] win rate)\n"
        f"Total benchmark time: [bold]{sum(seconds):.2f}s[/bold]  "
        f"({n / sum(seconds):.1f} games/sec)"
    )
    console.print(Panel(summary_text, title="Survivability", border_style="green" if win_rate > 50 else "yellow"))


def plot(results, width, height, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    percents = [r["percent_filled"] for r in results]
    point_ticks = [r["point_ticks"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(f"Snake Benchmark: {width}x{height} board, {len(results)} games")

    axes[0].hist(percents, bins=100, color="#2ea043", edgecolor="black")
    axes[0].set_title("Board Filled (%)")
    axes[0].set_xlabel("% filled")
    axes[0].set_ylabel("games")

    axes[1].hist(point_ticks, bins=100, color="#3c78e6", edgecolor="black")
    axes[1].set_title("Average moves per point")
    axes[1].set_xlabel("moves")
    axes[0].set_ylabel("games")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=120)
    console.print(f"[bold]Saved plot to {path}[/bold]")


def plot_progress(records, path):
    """Plot mean (+/- 1 std dev) moves-to-reach-food against percent of board
    filled, pooling every saved game record regardless of board size -- percent
    filled is the size-agnostic x-axis that lets different board dimensions sit
    on the same chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buckets = [[] for _ in range(101)]  # index = percent filled, rounded
    for r in records:
        history = r.get("point_tick_history")
        cells = r.get("cells")
        start_len = r.get("start_len")
        if not history or not cells or start_len is None:
            continue
        for i, interval in enumerate(history):
            percent = 100 * (start_len + i + 1) / cells
            bucket = min(100, round(percent))
            buckets[bucket].append(interval)

    xs, means, lows, highs = [], [], [], []
    for percent, values in enumerate(buckets):
        if not values:
            continue
        mean = statistics.mean(values)
        stdev = statistics.pstdev(values) if len(values) > 1 else 0
        xs.append(percent)
        means.append(mean)
        lows.append(max(0, mean - stdev))
        highs.append(mean + stdev)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.suptitle(f"Snake Benchmark: moves to reach each food ({len(records)} games)")

    ax.fill_between(xs, lows, highs, color="#2a78d6", alpha=0.2, linewidth=0)
    ax.plot(xs, means, color="#2a78d6", linewidth=2)
    ax.set_title("Moves per food vs. board progress")
    ax.set_xlabel("% of board filled")
    ax.set_ylabel("moves to reach food (mean ± 1 std dev)")
    ax.set_xlim(0, 100)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=120)
    console.print(f"[bold]Saved progress plot to {path}[/bold]")


def main():
    parser = argparse.ArgumentParser(description="Benchmark the snake algorithm over many headless games.")
    parser.add_argument("--width", type=int, default=10, help="grid width (default: 10)")
    parser.add_argument("--height", type=int, default=10, help="grid height (default: 10)")
    parser.add_argument("--games", type=int, default=50, help="number of games to run (default: 50)")
    parser.add_argument("--seed", type=int, default=0, help="base RNG seed; game i uses seed+i (default: 0)")
    parser.add_argument("--complete-max-free", type=int, default=25,
                        help="SearchContext.complete_max_free tuning knob (default: 25)")
    parser.add_argument("--plot", type=str, default=None,
                        help="if given, save a histogram + progress figure to this path (e.g. results.png)")
    parser.add_argument("--data-file", type=str, default="benchmark_data.json",
                        help="JSON file that accumulates per-game records across runs (default: benchmark_data.json)")
    parser.add_argument("--replot-only", action="store_true",
                        help="skip running games; just (re)generate plots from --data-file")
    args = parser.parse_args()

    if args.replot_only:
        all_records = load_results(args.data_file)
        if not all_records:
            console.print(f"[red]No saved records found in {args.data_file}[/red]")
            return
        if args.plot:
            plot_progress(all_records, args.plot)
        return

    results = run_benchmark(args.width, args.height, args.games, args.seed, args.complete_max_free)
    summarize(results, args.width, args.height)
    save_results(results, args.data_file)

    if args.plot:
        plot(results, args.width, args.height, args.plot)
        all_records = load_results(args.data_file)
        progress_path = args.plot.rsplit(".", 1)
        progress_path = f"{progress_path[0]}_progress.{progress_path[1]}" if len(progress_path) == 2 else f"{args.plot}_progress"
        plot_progress(all_records, progress_path)


if __name__ == "__main__":
    main()
