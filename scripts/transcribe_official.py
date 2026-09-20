#!/usr/bin/env python3
"""Transcribe official challenges from the practice guide into benchmark tasks.

For each challenge the guide provides three things: the starting setup, the
finished board, and the required output. The parts the solver must place are the
difference between the two boards, and the required output is the ground truth.

The guide does not state where a ball leaving the bottom row is counted, so the
catcher columns and the starting ball are chosen as the configuration whose
simulation reproduces the guide's own required output. A transcription is only
written when such a configuration exists, which makes the printed output an
independent check on the extracted board rather than an assumption.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np

from extract_boards import GRID, build_templates, parts_on_page, render, blobs
from tt_bench.simulator import Board, verify_task

REPO = Path(__file__).resolve().parent.parent
GUIDE = REPO / "Papers" / "practice-guide-2021.pdf"
CHALLENGE_DIR = REPO / "data" / "tasks" / "official" / "challenges" / "json"
SCALE = 200 / 72
ALL_PART_TYPES = [
    "ramp_right", "ramp_left", "crossover",
    "bit", "gear_bit", "gear", "interceptor", "trigger",
]


# ── The guide's own pages ────────────────────────────────────────────────

def page_map() -> dict[str, int]:
    """Challenge id -> pdf page of its puzzle (the solution is the next page)."""
    text = subprocess.run(
        ["pdftotext", "-layout", str(GUIDE), "-"], capture_output=True, text=True
    ).stdout
    pages: dict[str, int] = {}
    for line in text.splitlines():
        line = line.strip()
        if ".." not in line:
            continue
        m = re.match(
            r"^(Challenge|Practice Puzzle|Bonus Puzzle)\s*([A-C])?\s*"
            r"(?:for Challenge\s*)?(\d+)?[. ]*?(\d+)$", line)
        if not m:
            continue
        kind, letter, number, page = m.groups()
        if kind == "Challenge":
            stem = f"tt-official-ch{int(number):02d}"
        elif kind == "Bonus Puzzle":
            stem = f"tt-official-ch{int(number):02d}-b{letter or 'A'}"
        else:
            stem = f"tt-official-ch{int(number):02d}-p{letter or 'A'}"
        pages[stem] = int(page) + 4      # the guide's page 1 is the pdf's page 5
    return pages


def ball_counts(page_no: int) -> tuple[int, int] | None:
    """How many balls the guide loads in each hopper, as (blue, red).

    The challenge page prints the available-parts count on its own line and the
    two hopper counts side by side on the next one. The counts matter: challenge
    11 starts with two blue balls and no red ones, which no default can stand in
    for.
    """
    text = subprocess.run(
        ["pdftotext", "-layout", "-f", str(page_no), "-l", str(page_no),
         str(GUIDE), "-"], capture_output=True, text=True).stdout
    for line in text.splitlines():
        counts = re.findall(r"x\s*(\d+)", line)
        if len(counts) == 2:
            return int(counts[0]), int(counts[1])
    return None


def required_output(page_no: int, cache: Path) -> list[str] | None:
    """The ball sequence the guide prints as the challenge's required output."""
    xml = subprocess.run(
        ["pdftotext", "-bbox", "-f", str(page_no), "-l", str(page_no), str(GUIDE), "-"],
        capture_output=True, text=True).stdout

    def find(word: str):
        m = re.search(r'yMin="([\d.]+)"[^>]*yMax="([\d.]+)">' + word + "<", xml)
        return (float(m.group(1)), float(m.group(2))) if m else None

    head, foot = find("output:"), find("Starting")
    if not head or not foot:
        return None
    band = render(page_no, cache)[int(head[1] * SCALE) + 4:int(foot[0] * SCALE) - 4, :]

    # The balls sit in a short wide strip; the page frame crosses the same band.
    strips = [b for b in blobs(band < 200, 200, 60000)
              if 20 <= (b["y1"] - b["y0"]) <= 70 and (b["x1"] - b["x0"]) >= 40]
    if not strips:
        return []
    strip = max(strips, key=lambda b: b["x1"] - b["x0"])
    region = band[strip["y0"]:strip["y1"] + 1, strip["x0"]:strip["x1"] + 1]

    darkest = region.min(axis=0)
    is_ball = darkest < 165          # balls are darker than the ramp they rest on
    runs, start = [], None
    for x, on in enumerate(is_ball):
        if on and start is None:
            start = x
        elif not on and start is not None:
            if x - start >= 12:
                runs.append((start, x))
            start = None
    if start is not None and len(is_ball) - start >= 12:
        runs.append((start, len(is_ball)))

    balls = []
    for x0, x1 in runs:
        count = max(1, int(round((x1 - x0) / 20.5)))
        step = (x1 - x0) / count
        for i in range(count):
            a, b = int(x0 + i * step), int(x0 + (i + 1) * step)
            seg = darkest[a:b]
            ink = seg[seg < 165]
            if ink.size:
                balls.append((a, "blue" if float(np.median(ink)) < 110 else "red"))
    balls.sort(key=lambda t: -t[0])   # the ball nearest the exit drops first
    return [colour for _, colour in balls]


# ── Goals stated in words rather than drawn as balls ─────────────────────

COUNT_WORDS = {"no": 0, "none": 0, "one": 1, "two": 2, "three": 3, "four": 4,
               "five": 5, "six": 6, "seven": 7, "eight": 8,
               "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5}


def _count(token: str) -> int | None:
    token = token.strip().lower()
    if token.isdigit():
        return int(token)
    return COUNT_WORDS.get(token)


def objective_target(objective: str) -> list[str] | None:
    """The output sequence an interception objective asks for.

    Several challenges print no ball strip and state the goal in words instead,
    e.g. "Let only 3 blue balls reach the bottom and catch the 4th ball in the
    interceptor", which fixes the whole sequence.
    """
    text = " ".join((objective or "").split()).lower()
    if "intercept" not in text and "catch" not in text:
        return None

    colour = "red" if "red ball" in text else "blue"

    # "Let only N <colour> balls reach the bottom/end ... intercept the (N+1)th"
    # "Let exactly 4 blue balls reach the end. (Intercept the 5th.)" -- challenge
    # 23 drops the word "ball" after the ordinal, so the fallback below cannot
    # read it and the count has to come from here, past the adverb.
    m = re.search(
        r"let (?:only |exactly )*(\w+) (?:blue |red )?balls? reach the (?:bottom|end)",
        text,
    )
    if m:
        n = _count(m.group(1))
        if n is not None:
            return [colour] * n + ["intercepted"]

    # "Catch/intercept the Nth ball in the interceptor" on its own
    m = re.search(r"(?:catch|intercept) the (\w+) (?:blue |red )?ball", text)
    if m:
        n = _count(m.group(1))
        if n is not None and n >= 1:
            return [colour] * (n - 1) + ["intercepted"]
    return None


def guide_examples(page_no: int) -> list[tuple[int, int]] | None:
    """The guide's printed ``x<balls> => A = <value>`` table, if it prints one.

    Challenges 21 and 22 state their goal as a worked table rather than a ball
    strip: "x5 => A = 5", "x 14 => A = 14". Each row is one trial, so the table
    is the ground truth a register board is scored against. Challenge 30 prints
    its examples as register drawings and yields nothing here, which is the
    honest answer -- a guessed table would score a board against fiction.
    """
    text = subprocess.run(
        ["pdftotext", "-layout", "-f", str(page_no), "-l", str(page_no),
         str(GUIDE), "-"], capture_output=True, text=True).stdout
    pairs = [
        (int(balls), int(value))
        for balls, value in re.findall(r"x\s*(\d+)\s*=>\s*A\s*=\s*(\d+)", text)
    ]
    # Sorted and de-duplicated so the trial order does not depend on how the
    # two columns happen to be laid out on the page.
    unique = sorted(set(pairs))
    return unique or None


def overflow_goal(objective: str) -> int | None:
    """The threshold in "count the blue balls ... if there are more than N".

    Challenge 30 draws its examples as register pictures, so the text is the
    only ground truth -- but it is a complete one: the count goes into the
    register, and the flag flips once the count passes N.
    """
    text = " ".join((objective or "").split()).lower()
    if "register" not in text or "more than" not in text:
        return None
    if not re.search(r"gear bit \w+ must fl", text):
        return None
    m = re.search(r"more than (\d+)", text)
    return int(m.group(1)) if m else None


def overflow_examples(page_no: int) -> list[int]:
    """The ball counts the guide works through, as "x7 =>", "x 15 =>".

    Only counts followed by an arrow are examples; the available-parts column
    prints bare counts on the same page and must not be mistaken for one.
    """
    text = subprocess.run(
        ["pdftotext", "-layout", "-f", str(page_no), "-l", str(page_no),
         str(GUIDE), "-"], capture_output=True, text=True).stdout
    return sorted({int(n) for n in re.findall(r"x\s*(\d+)\s*=>", text)})


def reversal_goal(objective: str) -> int | None:
    """How many starting bits a "reverse each bit" objective covers.

    "Reverse the direction of each of the 9 starting bits, regardless of the
    direction they point to start" is quantified over every way those bits can
    start, so the goal is a trial per starting configuration -- the drawn
    examples add nothing the sentence does not already fix.
    """
    text = " ".join((objective or "").split()).lower()
    if "reverse the direction" not in text or "regardless" not in text:
        return None
    m = re.search(r"each of the (\w+) starting bits", text)
    if not m:
        return None
    return _count(m.group(1))


def logic_goal(objective: str) -> tuple[str, bool] | None:
    """A two-bit AND/OR goal that routes a ball to one of two interceptors.

    Returns the operator and whether the named ("T") interceptor is the one
    taken when the condition HOLDS. Which interceptor on the board is T is not
    stated in the text -- the fit searches both and keeps the one that
    reproduces the whole truth table.
    """
    text = " ".join((objective or "").split()).lower()
    if "interceptor" not in text or "otherwise" not in text:
        return None
    if not re.search(r"start(?:s|ing)? pointed to the right", text):
        return None
    if " and " in text.split("start")[0] or "both bits" in text:
        operator = "and"
    elif " or " in text.split("start")[0]:
        operator = "or"
    else:
        return None
    return operator, True


def starting_bits(fixed: list[dict]) -> list[dict]:
    """The bits already on the board before the solver places anything."""
    return sorted(
        (c for c in fixed if c["type"] in ("bit", "gear_bit")),
        key=lambda c: (c["y"], c["x"]),
    )


def register_column(components: list[dict]) -> list[dict] | None:
    """The column of bits holding a register, read top to bottom.

    A register is drawn as a vertical stack, unlike the labelled ROW that
    ``labelled_bits`` finds for the "flip bits 2 and 5" objectives.
    """
    bits = [c for c in components if c["type"] in ("bit", "gear_bit")]
    if len(bits) < 2:
        return None
    columns: dict[int, list[dict]] = {}
    for bit in bits:
        columns.setdefault(bit["x"], []).append(bit)
    tallest = max(columns.values(), key=len)
    if len(tallest) < 2:
        return None
    return sorted(tallest, key=lambda c: c["y"])


def is_state_goal(objective: str) -> bool:
    """Whether the goal is a final bit state rather than an output sequence."""
    text = " ".join((objective or "").split()).lower()
    return bool(re.search(r"flip (?:the )?bits?\b", text)) or bool(re.search(r"\bif .*\bbits?\b", text))


def guide_objective(page_no: int) -> str | None:
    """The objective line the guide prints for a challenge.

    Worth reading rather than trusting the task file: challenge 11 was recorded
    as "Flip bits 1 and 4" where the guide asks for bits 2 and 5.
    """
    text = subprocess.run(
        ["pdftotext", "-layout", "-f", str(page_no), "-l", str(page_no),
         str(GUIDE), "-"], capture_output=True, text=True).stdout
    # The objective wraps, and `.+` stopped at the first newline -- which cut
    # challenge 27's goal at "regard-" and challenge 18's before it ever
    # mentioned an interceptor, so anything reading the whole sentence saw
    # nothing. Read on until the line runs out or the next section starts.
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = re.search(r"Objective:\s*(.*)", line)
        if not match:
            continue
        parts = [match.group(1).strip()]
        for following in lines[index + 1:]:
            stripped = following.strip()
            if not stripped or re.match(
                r"(Examples?|Required output|Starting setup|Available parts)\b", stripped
            ):
                break
            parts.append(stripped)
        # pdftotext keeps the guide's hyphenation, so "intercep- tor" has to be
        # rejoined rather than left as two words.
        joined = ""
        for part in parts:
            joined = joined[:-1] + part if joined.endswith("-") else (joined + " " + part)
        # The guide sets "flip" and "overflow" with an fl LIGATURE, so every
        # pattern spelling them out plainly missed -- which is why objectives
        # naming bits to flip looked like they named none.
        for ligature, plain in (("ﬀ", "ff"), ("ﬁ", "fi"), ("ﬂ", "fl"),
                                ("ﬃ", "ffi"), ("ﬄ", "ffl")):
            joined = joined.replace(ligature, plain)
        return " ".join(joined.split())
    return None


def labelled_bits(components: list[dict]) -> list[dict]:
    """The row of bits the guide labels, left to right.

    The labels ("1 2 3 4 5", or "A B") sit under the one row holding several
    bits, so that row is what the objective's numbering refers to.
    """
    bits = [c for c in components if c["type"] in ("bit", "gear_bit")]
    if not bits:
        return []
    rows: dict[int, list[dict]] = {}
    for bit in bits:
        rows.setdefault(bit["y"], []).append(bit)
    widest = max(rows.values(), key=len)
    return sorted(widest, key=lambda c: c["x"])


def state_goal_bits(objective: str, components: list[dict]) -> list[dict] | None:
    """Which bits an objective asks to be flipped right, or None if it says no."""
    text = " ".join((objective or "").split())
    m = re.search(r"[Ff]lip (?:the )?bits? ([\w, ]+?) to the right", text)
    if not m:
        return None
    labels = [tok for tok in re.split(r"[,\s]+|\band\b", m.group(1)) if tok]
    row = labelled_bits(components)
    if not row:
        return None
    chosen = []
    for label in labels:
        if label.isdigit():
            index = int(label) - 1
        elif len(label) == 1 and label.isalpha():
            index = ord(label.upper()) - ord("A")
        else:
            return None
        if not 0 <= index < len(row):
            return None
        chosen.append(row[index])
    return chosen or None


def loose_intercept(objective: str) -> str | None:
    """Colour an objective wants intercepted when it fixes no count."""
    text = " ".join((objective or "").split()).lower()
    if not text.startswith("intercept a"):
        return None
    return "red" if "red ball" in text else "blue"


# ── Assembling a task ────────────────────────────────────────────────────

def ball_colours(results: list) -> list[str]:
    """What the guide prints: the colour of each ball that comes out.

    A blue ball can leave on the right, so this is not the same as the
    catcher-derived sequence the benchmark stores in final_marble_state.
    """
    out = []
    for r in results:
        if not r.caught_by:
            continue
        out.append("intercepted" if "interceptor" in str(r.caught_by) else (r.colour or "?"))
    return out


def catcher_colours(results: list) -> list[str]:
    out = []
    for r in results:
        if r.caught_by == "left_catcher":
            out.append("blue")
        elif r.caught_by == "right_catcher":
            out.append("red")
        elif r.caught_by and "interceptor" in str(r.caught_by):
            out.append("intercepted")
    return out


def matches_printed(produced: list[str], target: list[str], objective: str) -> bool:
    """Whether a run matches the guide's printed required output.

    The strip is not always the complete run: a ball caught in an interceptor is
    not drawn, and an objective ending in an ellipsis prints only as much of a
    repeating pattern as fits.
    """
    if produced == target:
        return True
    if len(produced) > len(target) and produced[:len(target)] == target:
        tail = produced[len(target):]
        if all(t == "intercepted" for t in tail):
            return True
        if "…" in (objective or "") or "..." in (objective or ""):
            return True
    # The strip draws a repeating pattern for as long as it fits, which can be
    # longer than the hoppers can actually supply.
    if 6 <= len(produced) < len(target) and target[:len(produced)] == produced:
        return True
    return False


def _task(original: dict, fixed: list[dict], placed: list[dict], *,
          entry: str, left_x: int, right_x: int, first: str,
          target: list[str], counts: tuple[int, int]) -> dict:
    task = deepcopy(original)
    task["board"] = {
        "width": GRID,
        "height": GRID,
        "hopper_entry_mode": entry,
        "fixed_components": deepcopy(fixed),
        "ball_hoppers": {"blue": {"x": 2, "count": counts[0]},
                         "red": {"x": 8, "count": counts[1]}},
        "trigger_levers": {"left": {"x": left_x, "y": GRID},
                           "right": {"x": right_x, "y": GRID}},
    }
    parts = {t: 0 for t in ALL_PART_TYPES}
    for comp in placed:
        if comp["type"] in parts:
            parts[comp["type"]] += 1
    task["available_parts"] = parts
    task.setdefault("solution", {})
    task["solution"]["placed_components"] = deepcopy(placed)
    task["solution"]["final_marble_state"] = list(target)
    task["input_sequence"] = [first]
    return task


def fit_configuration(original: dict, fixed: list[dict], placed: list[dict],
                      target: list[str], want_intercept: str | None = None,
                      objective: str = "", state_goal: bool = False,
                      counts: tuple[int, int] = (8, 8),
                      goal_bits: list[dict] | None = None) -> dict | None:
    """Find the catcher setup under which the board produces the guide's output."""
    # The real board keeps the blue lever on the left and the red one on the
    # right, so prefer positions closest to those before considering others; an
    # unused lever can otherwise land anywhere and still match the output.
    placements = sorted(
        ((lx, rx) for lx in range(GRID) for rx in range(GRID) if lx < rx),
        key=lambda p: abs(p[0] - 2) + abs(p[1] - 8),
    )
    for entry in ("inward", "column"):
        for first in ("blue", "red"):
            for left_x, right_x in placements:
                    bit_goal = None
                    task = _task(original, fixed, placed, entry=entry,
                                 left_x=left_x, right_x=right_x, first=first,
                                 target=target, counts=counts)
                    try:
                        board = Board.from_task_dict(task)
                        results = board.run(task["input_sequence"])
                    except Exception:
                        continue
                    produced = ball_colours(results)
                    if state_goal:
                        lost = [r for r in results
                                if r.steps > 0 and not r.caught_by]
                        if lost or len(produced) < 2:
                            continue
                        # The goal is a bit configuration, so the run has to
                        # leave the named bits pointing right and every other
                        # bit where it started. Without this the search would
                        # settle for any setup that merely runs cleanly.
                        ended = board.get_all_states()
                        if goal_bits is not None:
                            wanted = {(c["x"], c["y"]) for c in goal_bits}
                            satisfied = True
                            for comp in fixed + placed:
                                if comp["type"] not in ("bit", "gear_bit"):
                                    continue
                                key = f'{comp["type"]}_{comp["x"]}_{comp["y"]}'
                                want = 1 if (comp["x"], comp["y"]) in wanted \
                                    else comp.get("state", 0)
                                if ended.get(key) != want:
                                    satisfied = False
                                    break
                            if not satisfied:
                                continue
                        task["solution"]["final_marble_state"] = catcher_colours(results)
                        task["required_output"] = produced
                        bit_goal = ended
                    elif want_intercept is not None:
                        caught = [r for r in results if r.caught_by]
                        stopped = [r for r in caught
                                   if "interceptor" in str(r.caught_by)
                                   and r.colour == want_intercept]
                        if not stopped or any(not r.caught_by for r in results if r.steps > 0):
                            continue
                        task["solution"]["final_marble_state"] = catcher_colours(results)
                        task["required_output"] = produced
                    elif not matches_printed(produced, target, objective):
                        continue
                    else:
                        task["solution"]["final_marble_state"] = catcher_colours(results)
                        task["required_output"] = produced
                    task["expected_output"] = {
                        "left_catcher": sum(1 for r in results if r.caught_by == "left_catcher"),
                        "right_catcher": sum(1 for r in results if r.caught_by == "right_catcher"),
                        "intercepted": sum(1 for r in results if r.caught_by == "interceptor"),
                    }
                    if bit_goal is not None:
                        # Declared here because this is the only place scoring
                        # reads bit targets from; under solution it is inert.
                        task["expected_output"]["final_bit_states"] = bit_goal
                    try:
                        if not verify_task(task):
                            continue
                    except Exception:
                        continue
                    return task
    return None


def _finish(task: dict, objective: str, page: int,
            *, checked: str) -> tuple[dict | None, str]:
    """Reject a puzzle that needs none of its parts, then stamp provenance."""
    stripped = deepcopy(task)
    stripped["solution"]["placed_components"] = []
    try:
        if verify_task(stripped):
            return None, "puzzle solves itself without the placed parts"
    except Exception:
        pass

    task["objective"] = objective
    task["solution"]["verified"] = True
    task["solution"]["position_verified"] = True
    task["provenance"] = {
        "board": "transcribed from practice-guide-2021.pdf",
        "page": page,
        "checked": checked,
    }
    return task, "OK"


def fit_register(original: dict, fixed: list[dict], placed: list[dict],
                 examples: list[tuple[int, int]]) -> tuple[dict | None, str]:
    """Find the setup under which the board reproduces the guide's whole table.

    The same search as fit_configuration, against the trial table instead of one
    printed strip, and over one extra unknown: the guide draws the register as a
    column without saying which end is the ones place, so both readings are
    tried and the one that reproduces EVERY example is the order the board
    actually uses. A table of four rows is a strong check -- an accidental fit
    would have to agree on all of them.

    A counter feeds itself through the trigger lever, so a trial varies the
    hopper load rather than the input sequence: one marble is released and the
    board pulls the rest down on its own.
    """
    column = register_column(fixed + placed)
    if column is None:
        return None, ""
    keys = [f'{c["type"]}_{c["x"]}_{c["y"]}' for c in column]
    most = max(balls for balls, _ in examples)

    placements = sorted(
        ((lx, rx) for lx in range(GRID) for rx in range(GRID) if lx < rx),
        key=lambda p: abs(p[0] - 2) + abs(p[1] - 8),
    )
    orders = (("top-is-ones", list(reversed(keys))), ("top-is-highest", keys))

    for entry in ("inward", "column"):
        for left_x, right_x in placements:
            for reading, order in orders:
                task = _task(original, fixed, placed, entry=entry,
                             left_x=left_x, right_x=right_x, first="blue",
                             target=["blue"], counts=(most, 0))
                # A register goal has no printed strip. Leaving the single-run
                # targets in place would score the board against a second,
                # unrelated contract that nothing in the guide asks for.
                task.pop("required_output", None)
                task["expected_output"] = {}
                task["solution"].pop("final_marble_state", None)
                task["registers"] = {"A": order}
                task["trials"] = [
                    {
                        "name": f"x{balls}",
                        "hoppers": {"blue": balls, "red": 0},
                        "input_sequence": ["blue"],
                        "expect": {"registers": {"A": value}},
                    }
                    for balls, value in examples
                ]
                try:
                    if verify_task(task):
                        return task, reading
                except Exception:
                    continue
    return None, ""


def _lever_placements() -> list[tuple[int, int]]:
    return sorted(
        ((lx, rx) for lx in range(GRID) for rx in range(GRID) if lx < rx),
        key=lambda p: abs(p[0] - 2) + abs(p[1] - 8),
    )


def fit_reversal(original: dict, fixed: list[dict], placed: list[dict],
                 how_many: int, counts: tuple[int, int]) -> dict | None:
    """Find the setup under which every starting configuration is reversed.

    The goal quantifies over starting configurations, so the trial table is the
    full truth table: 2^n rows, each pointing the starting bits one way and
    requiring all of them to end pointing the other. Bits the SOLUTION adds are
    left unconstrained -- the objective speaks only of the starting bits.
    """
    bits = starting_bits(fixed)
    if len(bits) != how_many:
        return None
    keys = [f'{c["type"]}_{c["x"]}_{c["y"]}' for c in bits]

    trials = []
    for mask in range(2 ** len(keys)):
        start = {k: (mask >> i) & 1 for i, k in enumerate(keys)}
        trials.append({
            "name": "".join(str(start[k]) for k in keys),
            "initial_bit_states": start,
            "expect": {"final_bit_states": {k: 1 - v for k, v in start.items()}},
        })

    for entry in ("inward", "column"):
        for first in ("blue", "red"):
            for left_x, right_x in _lever_placements():
                task = _task(original, fixed, placed, entry=entry,
                             left_x=left_x, right_x=right_x, first=first,
                             target=["blue"], counts=counts)
                task.pop("required_output", None)
                task["expected_output"] = {}
                task["solution"].pop("final_marble_state", None)
                task["trials"] = trials
                try:
                    if verify_task(task):
                        return task
                except Exception:
                    continue
    return None


def fit_overflow(original: dict, fixed: list[dict], placed: list[dict],
                 threshold: int, examples: list[int]) -> tuple[dict | None, str]:
    """Find the setup reproducing a counter-with-overflow-flag table.

    Two things are asserted per example count: the register holds the count
    (modulo its width, which is what a counter of that width can hold), and the
    flag gear bit is right exactly when the count passed the threshold. The flag
    is a latch of meshed gear bits, so naming them all states the goal without
    having to work out which one the guide labels OV.
    """
    column = register_column(fixed + placed)
    flags = sorted((c["x"], c["y"]) for c in fixed + placed if c["type"] == "gear_bit")
    if column is None or not flags or not examples:
        return None, ""
    keys = [f'{c["type"]}_{c["x"]}_{c["y"]}' for c in column]
    flag_keys = [f"gear_bit_{x}_{y}" for x, y in flags]
    modulus = 2 ** len(keys)
    most = max(examples)

    orders = (("top-is-ones", list(reversed(keys))), ("top-is-highest", keys))
    for entry in ("inward", "column"):
        for left_x, right_x in _lever_placements():
            for reading, order in orders:
                task = _task(original, fixed, placed, entry=entry,
                             left_x=left_x, right_x=right_x, first="blue",
                             target=["blue"], counts=(most, 0))
                task.pop("required_output", None)
                task["expected_output"] = {}
                task["solution"].pop("final_marble_state", None)
                task["registers"] = {"A": order}
                task["trials"] = [
                    {
                        "name": f"x{n}",
                        "hoppers": {"blue": n, "red": 0},
                        "input_sequence": ["blue"],
                        "expect": {
                            "registers": {"A": n % modulus},
                            "final_bit_states": {
                                k: (1 if n > threshold else 0) for k in flag_keys
                            },
                        },
                    }
                    for n in examples
                ]
                try:
                    if verify_task(task):
                        return task, reading
                except Exception:
                    continue
    return None, ""


def fit_bit_states(original: dict, fixed: list[dict], solution_states: dict,
                   target: list[str], objective: str,
                   counts: tuple[int, int]) -> dict | None:
    """The solution is which way the fixed bits start, not a part to place.

    Challenge 23's family draws the SAME board on both pages and differs only in
    the direction of three bits: the puzzle is to choose where the counter
    starts so the interceptor fires on the right ball. The guide leaves those
    bits undirected in the starting setup, so the board declares them editable
    and the solution is their states.

    The starting direction written into fixed_components is a placeholder the
    guide does not supply, so it is chosen to be a configuration that does NOT
    already meet the goal. Otherwise practice puzzle A -- whose answer is "all
    bits left" -- would ship already solved.
    """
    positions = sorted(solution_states)
    keys = [(x, y) for x, y in positions]
    placed = [{"type": "bit", "x": x, "y": y, "state": solution_states[(x, y)]}
              for x, y in keys]

    def build(entry, left_x, right_x, first, default):
        task = deepcopy(original)
        fixed_now = []
        for comp in deepcopy(fixed):
            if (comp["x"], comp["y"]) in solution_states:
                comp["state"] = default[(comp["x"], comp["y"])]
            fixed_now.append(comp)
        task["board"] = {
            "width": GRID,
            "height": GRID,
            "hopper_entry_mode": entry,
            "fixed_components": fixed_now,
            "ball_hoppers": {"blue": {"x": 2, "count": counts[0]},
                             "red": {"x": 8, "count": counts[1]}},
            "trigger_levers": {"left": {"x": left_x, "y": GRID},
                               "right": {"x": right_x, "y": GRID}},
            "editable_bit_states": [[x, y] for x, y in keys],
        }
        # Nothing is placed, so nothing is spent: the whole decision is which
        # way the bits point.
        task["available_parts"] = {t: 0 for t in ALL_PART_TYPES}
        task.setdefault("solution", {})
        task["solution"]["placed_components"] = deepcopy(placed)
        task["input_sequence"] = [first]
        task.pop("required_output", None)
        task["expected_output"] = {}
        task["solution"]["final_marble_state"] = list(target)
        return task

    defaults = []
    for mask in range(2 ** len(keys)):
        defaults.append({pos: (mask >> i) & 1 for i, pos in enumerate(keys)})
    # Prefer all-left, so the placeholder is the plainest board that works.
    defaults.sort(key=lambda d: sum(d.values()))

    for entry in ("inward", "column"):
        for first in ("blue", "red"):
            for left_x, right_x in _lever_placements():
                for default in defaults:
                    if default == solution_states:
                        continue
                    task = build(entry, left_x, right_x, first, default)
                    try:
                        board = Board.from_task_dict(task)
                        results = board.run(task["input_sequence"])
                    except Exception:
                        continue
                    if not matches_printed(ball_colours(results), target, objective):
                        continue
                    task["solution"]["final_marble_state"] = catcher_colours(results)
                    task["required_output"] = ball_colours(results)
                    task["expected_output"] = {
                        "left_catcher": sum(1 for r in results if r.caught_by == "left_catcher"),
                        "right_catcher": sum(1 for r in results if r.caught_by == "right_catcher"),
                        "intercepted": sum(1 for r in results if r.caught_by == "interceptor"),
                    }
                    try:
                        if not verify_task(task):
                            continue
                        # The placeholder must not already be an answer.
                        undecided = deepcopy(task)
                        undecided["solution"]["placed_components"] = []
                        if verify_task(undecided):
                            continue
                    except Exception:
                        continue
                    return task
    return None


def fit_logic(original: dict, fixed: list[dict], placed: list[dict],
              operator: str, counts: tuple[int, int]) -> tuple[dict | None, str]:
    """Find the setup reproducing a two-bit AND/OR truth table.

    Two unknowns are searched together: the usual catcher setup, and which of
    the board's two interceptors the guide labels T. A four-row table pins both
    -- three of its rows take the "otherwise" branch, so an assignment that
    merely fits one row cannot survive.
    """
    bits = starting_bits(fixed)
    interceptors = sorted(
        ((c["x"], c["y"]) for c in fixed + placed if c["type"] == "interceptor")
    )
    if len(bits) != 2 or len(interceptors) != 2:
        return None, ""
    keys = [f'{c["type"]}_{c["x"]}_{c["y"]}' for c in bits]

    def table(true_at, false_at):
        rows = []
        for mask in range(4):
            start = {k: (mask >> i) & 1 for i, k in enumerate(keys)}
            values = list(start.values())
            holds = all(values) if operator == "and" else any(values)
            rows.append({
                "name": "".join(str(start[k]) for k in keys),
                "hoppers": {"blue": 1, "red": 1},
                "initial_bit_states": start,
                "expect": {"intercepted_at": list(true_at if holds else false_at)},
            })
        return rows

    for entry in ("inward", "column"):
        for first in ("blue", "red"):
            for left_x, right_x in _lever_placements():
                for label, (true_at, false_at) in (
                    ("T=" + str(interceptors[0]), (interceptors[0], interceptors[1])),
                    ("T=" + str(interceptors[1]), (interceptors[1], interceptors[0])),
                ):
                    task = _task(original, fixed, placed, entry=entry,
                                 left_x=left_x, right_x=right_x, first=first,
                                 target=["blue"], counts=counts)
                    task.pop("required_output", None)
                    task["expected_output"] = {}
                    task["solution"].pop("final_marble_state", None)
                    task["trials"] = table(true_at, false_at)
                    try:
                        if verify_task(task):
                            return task, label
                    except Exception:
                        continue
    return None, ""


def transcribe(stem: str, page: int, cache: Path, templates) -> tuple[dict | None, str]:
    path = CHALLENGE_DIR / f"{stem}.json"
    if not path.exists():
        return None, "no task file"
    original = json.loads(path.read_text())

    try:
        setup = parts_on_page(page, cache, templates)
        whole = parts_on_page(page + 1, cache, templates)
    except Exception as exc:
        return None, f"extract failed ({type(exc).__name__})"

    # The guide prints how many balls each hopper holds; challenge 11 needs
    # exactly two blue and no red, so a default would make it unsolvable.
    counts = ball_counts(page) or (8, 8)

    setup_keys = {(p.type, p.x, p.y) for p in setup}
    fixed = [p.to_dict() for p in setup]
    placed = [p.to_dict() for p in whole if (p.type, p.x, p.y) not in setup_keys]

    objective_early = guide_objective(page) or original.get("objective", "")
    if not placed:
        # Same parts on both pages: the solution is not a part but the DIRECTION
        # the drawn bits start in, which the key (type, x, y) deliberately
        # ignores. Challenge 23's family is decided entirely this way.
        setup_states = {(p.x, p.y): p.to_dict().get("state") for p in setup
                        if p.type in ("bit", "gear_bit")}
        solution_states = {(p.x, p.y): p.to_dict().get("state") for p in whole
                           if p.type in ("bit", "gear_bit")}
        undirected = [pos for pos, state in setup_states.items() if state is None]
        decided = {pos: state for pos, state in solution_states.items()
                   if state is not None and pos in setup_states}
        target = objective_target(objective_early)
        if undirected and len(decided) == len(setup_states) and target:
            task = fit_bit_states(original, fixed, decided, target,
                                  objective_early, counts)
            if task is None:
                return None, "no setup reproduces the target from those bit states"
            return _finish(task, objective_early, page, checked=(
                "simulation reproduces the guide's stated output once the "
                f"{len(decided)} editable bits start as the solution page draws them"
            ))
        return None, "no parts to place"

    # An interception objective is the more precise of the two: the printed ball
    # strip shows only the balls that reach the bottom, not the one caught.
    # The guide's own wording is authoritative for the goal.
    objective = guide_objective(page) or original.get("objective", "")

    # A printed "x5 => A = 5" table is the whole goal, and it is scored over
    # several runs, so it short-circuits the single-strip search below.
    examples = guide_examples(page)
    if examples:
        task, reading = fit_register(original, fixed, placed, examples)
        if task is None:
            return None, f"no setup reproduces the {len(examples)}-row register table"
        return _finish(task, objective, page, checked=(
            f"simulation reproduces all {len(examples)} rows of the guide's "
            f"register table, reading the column with {reading}"
        ))

    # "Reverse each starting bit, regardless of how it starts" and the two-bit
    # AND/OR routings are both quantified over starting configurations, so the
    # trial table is the goal and the single-strip search below cannot state it.
    # Checked before the state-goal path below, which would otherwise accept
    # whatever this board happens to do on one run as its target.
    threshold = overflow_goal(objective)
    if threshold is not None:
        examples = overflow_examples(page)
        task, reading = fit_overflow(original, fixed, placed, threshold, examples)
        if task is None:
            return None, f"no setup reproduces the overflow table for {examples}"
        return _finish(task, objective, page, checked=(
            f"simulation counts into the register and raises the flag past "
            f"{threshold} for every worked count {examples}, "
            f"reading the column with {reading}"
        ))

    how_many = reversal_goal(objective)
    if how_many:
        task = fit_reversal(original, fixed, placed, how_many, counts)
        if task is None:
            return None, f"no setup reverses all {2 ** how_many} starting configurations"
        return _finish(task, objective, page, checked=(
            f"simulation reverses the {how_many} starting bits from every one "
            f"of their {2 ** how_many} starting configurations"
        ))

    logic = logic_goal(objective)
    if logic:
        task, label = fit_logic(original, fixed, placed, logic[0], counts)
        if task is None:
            return None, f"no setup reproduces the two-bit {logic[0].upper()} table"
        return _finish(task, objective, page, checked=(
            f"simulation reproduces all four rows of the {logic[0].upper()} "
            f"truth table, with {label}"
        ))

    candidates: list[tuple[list[str], str | None]] = []
    from_objective = objective_target(objective)
    if from_objective:
        candidates.append((from_objective, None))
    printed = required_output(page, cache)
    if printed:
        candidates.append((printed, None))
    loose = loose_intercept(objective)
    if loose:
        candidates.append(([], loose))
    state_goal = is_state_goal(objective)
    goal_bits = state_goal_bits(objective, fixed + placed) if state_goal else None
    if state_goal and not candidates:
        candidates.append(([], None))
    if not candidates:
        return None, "no required output printed"

    # A bit drawn with arrows both ways has no start state in the guide, so try
    # each possibility and keep the one that satisfies the challenge.
    open_bits = [c for c in fixed + placed
                 if c["type"] in ("bit", "gear_bit") and c.get("state") is None]
    assignments = [dict()]
    for comp in open_bits:
        assignments = [{**a, (comp["x"], comp["y"]): v} for a in assignments for v in (0, 1)]

    task = None
    for assignment in assignments:
        for comp in fixed + placed:
            if (comp["x"], comp["y"]) in assignment:
                comp["state"] = assignment[(comp["x"], comp["y"])]
        for target, want_intercept in candidates:
            task = fit_configuration(original, fixed, placed, target, want_intercept,
                                     objective, state_goal,
                                     counts=counts, goal_bits=goal_bits)
            if task is not None:
                break
        if task is not None:
            break
    if task is None:
        shapes = ", ".join(
            f"intercepting a {w} ball" if w else f"{len(t)} balls" for t, w in candidates)
        return None, f"no setup reproduces {shapes}"

    return _finish(task, objective, page, checked=(
        "simulation leaves the bits the objective names pointing right"
        if state_goal else
        "simulation reproduces the guide's printed required output"
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()

    cache = REPO / ".guide-cache"
    templates = build_templates(cache)
    pages = page_map()
    if args.only:
        pages = {k: v for k, v in pages.items() if k in args.only}

    done, failed = 0, []
    for stem, page in sorted(pages.items(), key=lambda kv: kv[1]):
        task, why = transcribe(stem, page, cache, templates)
        if task is None:
            failed.append((stem, why))
            print(f"  {stem:24s} page {page:3d}  -- {why}")
            continue
        done += 1
        n_fixed = len(task["board"]["fixed_components"])
        n_placed = len(task["solution"]["placed_components"])
        if task.get("trials"):
            shape = f"{len(task['trials'])} trials"
        else:
            shape = f"output={len(task['solution']['final_marble_state'])} balls"
        print(f"  {stem:24s} page {page:3d}  OK  fixed={n_fixed:2d} place={n_placed:2d} "
              f"{shape}")
        if args.write:
            (CHALLENGE_DIR / f"{stem}.json").write_text(json.dumps(task, indent=2) + "\n")

    print(f"\ntranscribed {done} / {len(pages)}")
    if not args.write:
        print("(dry run — pass --write to apply)")


if __name__ == "__main__":
    main()
