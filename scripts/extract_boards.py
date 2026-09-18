#!/usr/bin/env python3
"""Read Turing Tumble board diagrams out of the official practice guide.

The guide draws each puzzle twice — the starting setup on the challenge page and
the finished board on the following solution page — so the parts the solver is
meant to place are the difference between the two.

Board parts are printed black on a light grey peg lattice, so parts are found as
dark blobs and snapped to the lattice fitted from the grey pegs.
"""

from __future__ import annotations

import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
GUIDE = REPO / "Papers" / "practice-guide-2021.pdf"
DPI = 200
GRID = 11


@dataclass(frozen=True)
class Part:
    type: str
    x: int
    y: int
    state: int | None = None

    def to_dict(self) -> dict:
        d = {"type": self.type, "x": self.x, "y": self.y}
        if self.state is not None:
            d["state"] = self.state
        return d


def render(page: int, cache: Path) -> np.ndarray:
    cache.mkdir(parents=True, exist_ok=True)
    stem = cache / f"page{page:03d}"
    png = Path(f"{stem}-{page:03d}.png")
    if not png.exists():
        subprocess.run(
            ["pdftoppm", "-f", str(page), "-l", str(page), "-r", str(DPI),
             "-png", str(GUIDE), str(stem)],
            check=True, capture_output=True,
        )
    return np.array(Image.open(png).convert("L"))


def blobs(mask: np.ndarray, min_px: int, max_px: int) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    found = []
    for sy, sx in np.argwhere(mask):
        if seen[sy, sx]:
            continue
        queue = deque([(sy, sx)])
        seen[sy, sx] = True
        pts = []
        while queue:
            y, x = queue.popleft()
            pts.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and mask[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        if min_px <= len(pts) <= max_px:
            ys = [p[0] for p in pts]
            xs = [p[1] for p in pts]
            found.append({
                "n": len(pts), "cy": float(np.mean(ys)), "cx": float(np.mean(xs)),
                "y0": min(ys), "y1": max(ys), "x0": min(xs), "x1": max(xs),
                "pts": pts,
            })
    return found


def _cluster(values: list[float], tol: float) -> list[float]:
    values = sorted(values)
    groups = [[values[0]]]
    for v in values[1:]:
        if v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [float(np.mean(g)) for g in groups]


def fit_lattice(page: np.ndarray, row_tol: float = 24) -> tuple[float, float, float]:
    """Origin and pitch of the 11x11 peg lattice, in pixels."""
    grey = (page > 140) & (page < 215)
    pegs = blobs(grey, 40, 400)
    if len(pegs) < 30:
        raise ValueError("too few pegs to fit a lattice")

    # A peg row/column holds many pegs; page artwork such as the hoppers forms
    # sparse lines that would otherwise fit the lattice just as well.
    def dense(values: list[float], least: int, tol: float) -> list[float]:
        values = sorted(values)
        groups = [[values[0]]]
        for v in values[1:]:
            if v - groups[-1][-1] <= tol:
                groups[-1].append(v)
            else:
                groups.append([v])
        return [float(np.mean(g)) for g in groups if len(g) >= least]

    cols = dense([b["cx"] for b in pegs], 4, 14)
    # A board row draws two features at slightly different heights, so rows need
    # a looser tolerance than columns or each one splits in two.
    rows = dense([b["cy"] for b in pegs], 4, row_tol)
    if len(cols) < 4 or len(rows) < 4:
        raise ValueError("peg lattice too sparse")
    # Columns are evenly spaced and unambiguous, so they set the pitch.
    col_gaps = [g for g in (b - a for a, b in zip(cols, cols[1:])) if 40 <= g <= 80]
    if not col_gaps:
        raise ValueError("no usable column spacing")
    unit = float(np.median(col_gaps))

    def align(values: list[float]) -> float:
        """Origin of the regular lattice, ignoring stray page artwork."""
        best, best_score = None, -1
        for origin in values:
            hits = [v for v in values if abs((v - origin) / unit - round((v - origin) / unit)) < 0.12]
            spread = [round((v - origin) / unit) for v in hits]
            inside = [s for s in spread if 0 <= s <= GRID - 1]
            score = len(inside)
            if score > best_score:
                best, best_score = origin + min(inside) * unit, score
        if best_score < 8:
            raise ValueError("no regular lattice found")
        return best

    return align(cols), align(rows), unit


# Clean instances of each part, as (type, pdf page, column, row). Every part is
# drawn from the same artwork at the same scale, so one sample each is enough.
TEMPLATE_SOURCES = [
    ("ramp_right", 6, 3, 0),
    ("ramp_left", 6, 4, 1),
    ("crossover", 31, 3, 2),
    ("bit_left", 48, 5, 2),
    ("bit_right", 64, 3, 0),
    # A bit drawn with arrows both ways: the puzzle leaves its start state open.
    ("bit_either", 52, 5, 2),
    ("interceptor", 48, 5, 8),
]
WINDOW = 1.55  # crop size in lattice units
RISE = 0.28    # glyphs are drawn above the peg they sit on


def _crop(page: np.ndarray, x0: float, y0: float, unit: float,
          col: float, row: float) -> np.ndarray:
    half = unit * WINDOW / 2
    cx = x0 + col * unit
    cy = y0 + row * unit - unit * RISE
    box = page[int(cy - half):int(cy + half), int(cx - half):int(cx + half)]
    side = int(unit * WINDOW)
    out = np.zeros((side, side), dtype=bool)
    sub = (box < 110)
    out[:sub.shape[0], :sub.shape[1]] = sub[:side, :side]
    return out


def _isolate(window: np.ndarray) -> np.ndarray:
    """Drop ink belonging to neighbouring parts that reaches into the crop."""
    side = window.shape[0]
    keep = np.zeros_like(window)
    for blob in blobs(window, 30, side * side):
        if (abs(blob["cx"] - side / 2) < side * 0.3
                and abs(blob["cy"] - side / 2) < side * 0.4):
            for y, x in blob["pts"]:
                keep[y, x] = True
    return keep


def build_templates(cache: Path) -> dict[str, np.ndarray]:
    templates: dict[str, np.ndarray] = {}
    for kind, page_no, col, row in TEMPLATE_SOURCES:
        page = render(page_no, cache)
        x0, y0, unit = fit_lattice(page)
        templates[kind] = _isolate(_crop(page, x0, y0, unit, col, row))
    return templates


def best_at(page: np.ndarray, x0: float, y0: float, unit: float,
            col: int, row: int, templates: dict[str, np.ndarray],
            reach: int = 9, step: int = 3) -> tuple[str, float]:
    """Best-matching part at one peg, allowing for small lattice-fit drift.

    The fitted origin wanders by a fraction of a cell between pages, which is
    enough to ruin a fixed comparison, so each part is sought over a small
    neighbourhood instead.
    """
    side = int(unit * WINDOW)
    half = side // 2
    cx = int(x0 + col * unit)
    cy = int(y0 + row * unit - unit * RISE)
    pad = reach + 2
    y1, y2 = cy - half - pad, cy + half + pad
    x1, x2 = cx - half - pad, cx + half + pad
    if y1 < 0 or x1 < 0 or y2 > page.shape[0] or x2 > page.shape[1]:
        return "", 0.0
    patch = page[y1:y2, x1:x2] < 110

    def scan(offsets) -> tuple[str, float, int, int]:
        best, score, by, bx = "", 0.0, 0, 0
        for dy, dx in offsets:
            oy, ox = pad + dy, pad + dx
            window = patch[oy:oy + side, ox:ox + side]
            if window.shape != (side, side):
                continue
            for kind, tpl in templates.items():
                t = tpl[:side, :side]
                total = np.count_nonzero(t)
                if not total:
                    continue
                value = np.count_nonzero(window[:t.shape[0], :t.shape[1]] & t) / total
                if value > score:
                    best, score, by, bx = kind, value, dy, dx
        return best, score, by, bx

    coarse = [(dy, dx)
              for dy in range(-reach, reach + 1, step)
              for dx in range(-reach, reach + 1, step)]
    _, _, by, bx = scan(coarse)
    # The artwork is thin outlines, so being one pixel out already costs a third
    # of the overlap; refine around the coarse winner a pixel at a time.
    fine = [(by + dy, bx + dx) for dy in range(-step, step + 1)
            for dx in range(-step, step + 1)]
    best, best_score, _, _ = scan(coarse + fine)
    return best, best_score


def match(window: np.ndarray, templates: dict[str, np.ndarray]) -> tuple[str, float]:
    best, best_score = "", 0.0
    for kind, tpl in templates.items():
        side = min(window.shape[0], tpl.shape[0]), min(window.shape[1], tpl.shape[1])
        a = window[:side[0], :side[1]]
        b = tpl[:side[0], :side[1]]
        # Score how much of the part's artwork is present, so ink bleeding in
        # from a neighbouring part cannot veto a correct match.
        total = np.count_nonzero(b)
        if not total:
            continue
        score = np.count_nonzero(a & b) / total
        if score > best_score:
            best, best_score = kind, score
    return best, best_score


def classify(blob: dict, unit: float) -> tuple[str, float] | None:
    """Part type plus the column offset its glyph is drawn with."""
    width = blob["x1"] - blob["x0"]
    height = blob["y1"] - blob["y0"]
    if width < unit * 0.6 or height < unit * 0.5:
        return None

    pts = blob["pts"]
    # The pivot ring of a ramp sits at its low end; other parts are symmetric.
    band = [p for p in pts if p[0] >= blob["y1"] - height * 0.35]
    if not band:
        return None
    pivot = float(np.mean([p[1] for p in band])) - (blob["x0"] + blob["x1"]) / 2

    fill = blob["n"] / float(width * height)
    if abs(pivot) > unit * 0.15:
        return ("ramp_right", 0.07) if pivot < 0 else ("ramp_left", -0.07)
    if fill > 0.55:
        return ("crossover", 0.0)
    return ("unknown", 0.0)


def parts_on_page(page_no: int, cache: Path,
                  templates: dict[str, np.ndarray] | None = None,
                  threshold: float = 0.65) -> list[Part]:
    """Every part on one page, by matching each peg against the part artwork."""
    templates = templates if templates is not None else build_templates(cache)
    page = render(page_no, cache)

    # Row clustering is the shakiest part of the fit, so try both groupings and
    # keep whichever explains more of the page as actual parts.
    best_fit, best_parts = None, -1
    for row_tol in (24, 14):
        try:
            fit = fit_lattice(page, row_tol)
        except ValueError:
            continue
        n = sum(
            1 for row in range(GRID) for col in range(GRID)
            if best_at(page, *fit, col, row, templates)[1] >= threshold
        )
        if n > best_parts:
            best_fit, best_parts = fit, n
    if best_fit is None:
        raise ValueError("no regular lattice found")
    x0, y0, unit = best_fit

    found = []
    for row in range(GRID):
        for col in range(GRID):
            if np.count_nonzero(_crop(page, x0, y0, unit, col, row)) < unit * 3:
                continue
            kind, score = best_at(page, x0, y0, unit, col, row, templates)
            if score < threshold:
                continue
            if kind == "bit_either":
                found.append(Part("bit", col, row, None))
            elif kind.startswith("bit_"):
                found.append(Part("bit", col, row, 0 if kind.endswith("left") else 1))
            else:
                found.append(Part(kind, col, row))
    return sorted(found, key=lambda p: (p.y, p.x))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pages", nargs="+", type=int)
    parser.add_argument("--cache", default=None)
    args = parser.parse_args()
    cache = Path(args.cache) if args.cache else REPO / ".guide-cache"
    for page in args.pages:
        print(f"page {page}:")
        for part in parts_on_page(page, cache):
            print(f"   {part.type:12s} ({part.x},{part.y})")


if __name__ == "__main__":
    main()
