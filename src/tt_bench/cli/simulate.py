"""Interactive Turing Tumble simulator CLI."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from tt_bench.simulator.board import Board, build_gear_connections
from tt_bench.simulator.components import Component, ComponentType, Side
from tt_bench.simulator.renderer import BoardRenderer

# CLI Interface
# =============================================================================


def run_cli():
    """Interactive CLI for the Turing Tumble simulator."""
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Turing Tumble Simulator")
    parser.add_argument(
        "--load",
        type=str,
        help="Load a challenge JSON file",
    )
    parser.add_argument(
        "--run",
        type=str,
        help="Run a sequence of marbles (e.g., 'blue,red,blue' or 'b,r,b')",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify solution after loading",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset board before running",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Maximum steps per marble (default: 500)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Render simulation in real time in the terminal",
    )
    parser.add_argument(
        "--tick",
        type=float,
        default=0.2,
        help="Seconds between real-time frames (default: 0.2)",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not clear terminal between real-time frames",
    )

    args = parser.parse_args()

    board = None

    if args.load:
        board, task = load_challenge(args.load)
        board.max_steps = args.steps
        print(f"Loaded: {task.get('task_id', 'unknown')}")
        print(f"Objective: {task.get('objective', 'none')}")
        print()
        print(board.render())

        if args.verify:
            print("\nVerifying solution...")
            is_valid = verify_solution(args.load)
            print(f"Solution valid: {is_valid}")

    if board:
        if args.reset:
            board.reset()
            print("\nBoard reset.")

        if args.run:
            # Parse sequence
            seq = [s.strip().lower() for s in args.run.split(",")]
            seq = [
                "blue" if s in ("b", "blue") else "red" if s in ("r", "red") else s
                for s in seq
            ]

            def infer_side(result: MarbleResult) -> str:
                if not result.path:
                    return "unknown"
                start_x = result.path[0][0]
                if start_x == board.blue_hopper_x:
                    return "blue"
                if start_x == board.red_hopper_x:
                    return "red"
                return "unknown"

            print(f"\nRunning sequence: {seq}")
            if args.realtime:
                print(f"Real-time mode enabled (tick={max(args.tick, 0.0):.3f}s)")

                remaining = list(seq)
                results = []

                marble_idx = 0
                while remaining:
                    side_str = remaining.pop(0)
                    if side_str not in ("blue", "red"):
                        print(f"Skipping unknown marble side: {side_str}")
                        continue

                    current_side = Side(side_str)

                    def show_step(
                        step_board: Board,
                        position: tuple[int, int],
                        step_number: int,
                    ) -> None:
                        if not args.no_clear:
                            print("\033[2J\033[H", end="")
                        marker = None
                        if 0 <= position[0] < step_board.cols and 0 <= position[1] < step_board.rows:
                            marker = position
                        print(
                            f"Real-time simulation | Marble {marble_idx + 1} "
                            f"({current_side.value}) | Step {step_number}"
                        )
                        print(step_board.render(show_marble_path=marker))
                        if args.tick > 0:
                            time.sleep(args.tick)

                    result = board.release_marble(current_side, step_callback=show_step)
                    results.append(result)

                    trigger_releases: list[str] = []
                    while board._pending_trigger_releases:
                        paired_side = board._pending_trigger_releases.pop(0)
                        if paired_side == Side.BLUE and board.blue_balls_remaining > 0:
                            trigger_releases.append("blue")
                        elif paired_side == Side.RED and board.red_balls_remaining > 0:
                            trigger_releases.append("red")
                    remaining = trigger_releases + remaining

                    if result.terminated and result.termination_reason in (
                        "infinite_loop",
                        "max_steps_exceeded",
                        "fell_off_side",
                    ):
                        break

                    marble_idx += 1
            else:
                results = board.run(seq)

            for i, result in enumerate(results):
                side_label = infer_side(result)
                print(f"\nMarble {i + 1} ({side_label}):")
                print(f"  Path length: {len(result.path)} steps")
                print(f"  Caught by: {result.caught_by}")
                print(
                    f"  Terminated: {result.terminated} ({result.termination_reason})"
                )

            print("\n" + board.render())

    if not args.load and not args.run:
        # Interactive mode
        print("Turing Tumble Simulator - Interactive Mode")
        print("Commands: load <file>, run <seq>, reset, render, quit")

        board = None
        while True:
            try:
                cmd = input("\n> ").strip().split(maxsplit=1)
                if not cmd:
                    continue

                if cmd[0] == "quit" or cmd[0] == "q":
                    break
                elif cmd[0] == "load" and len(cmd) > 1:
                    board, task = load_challenge(cmd[1])
                    print(f"Loaded: {task.get('task_id', 'unknown')}")
                    print(board.render())
                elif cmd[0] == "run" and len(cmd) > 1:
                    if board:
                        seq = [s.strip() for s in cmd[1].split(",")]
                        results = board.run(seq)
                        print(f"Released {len(results)} marbles")
                        print(board.render())
                    else:
                        print("No board loaded")
                elif cmd[0] == "reset" and board:
                    board.reset()
                    print("Board reset")
                elif cmd[0] == "render" and board:
                    print(board.render())
                elif cmd[0] == "help":
                    print("Commands: load <file>, run <seq>, reset, render, quit")
                else:
                    print("Unknown command. Type 'help' for available commands.")
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    run_cli()
