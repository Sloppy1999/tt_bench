#!/usr/bin/env python3
"""
Board Analytics Module for the Turing Tumble Benchmark
=======================================================

Extracts board-level features from puzzle JSONs, integrates them with
LLM benchmark scoring results, and produces correlation analytics,
visualizations, and structured academic reports.

The module is organised into five independently callable components:

    BoardFeatureExtractor  — Feature engineering from board JSONs
    BenchmarkIntegrator    — Merging features with benchmark outputs
    CorrelationAnalyzer    — Statistical correlation & importance analysis
    BoardVisualizer        — matplotlib/seaborn publication-quality plots
    ReportGenerator        — Structured markdown/tabular report export

All public classes and functions carry NumPy-style docstrings.  The
optional ``main()`` entry point runs the full end-to-end pipeline
(extract → merge → analyse → visualise → report) from a directory of
puzzle JSONs and a benchmark-results file.

Coordinate system (from ``tasks/official/COORDINATES.md``):
    - Origin (0,0) = top-left, x increases rightward, y increases downward
    - Board dimensions: 11×11 by default (configurable)
    - Hoppers at y=-1, trigger levers at y=11

Author : Turing Tumble Benchmark Team
Version: 1.0.0
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Headless backend — must be set before pyplot import
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.preprocessing import StandardScaler

# Suppress sklearn warnings about continuous y in mutual_info_classif
# (some performance metrics are continuous; MI still provides useful signal)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
# Suppress scipy ConstantInputWarning for features with near-zero variance
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero.*")
warnings.filterwarnings(
    "ignore", message="An input array is constant",
)

# --- Imports ----------------------------------------------------------------
from tt_bench import simulator as tt_sim  # noqa: E402
from tt_bench.analytics import metrics as cx  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# ============================================================================
# Constants
# ============================================================================

# All component types recognised by the simulator
ALL_COMPONENT_TYPES: Tuple[str, ...] = (
    "ramp_right", "ramp_left", "crossover",
    "bit", "gear_bit", "gear", "interceptor", "trigger",
)

# Performance metric column names in merged data
_PERF_METRICS: Tuple[str, ...] = (
    "success", "trace_accuracy", "state_precision",
    "tool_calls_count", "turns", "valid",
    "latency_ms", "tokens_used",
)

# Default seaborn/matplotlib style for publication-quality plots
_PLOT_CONTEXT: Dict[str, Any] = {
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}
_DEFAULT_FIGSIZE: Tuple[float, float] = (12, 9)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class BoardFeatures:
    """Container for all extracted board-level features of a single puzzle.

    Attributes
    ----------
    task_id : str
        Unique task identifier (e.g. ``"tt-official-ch07"``).
    component_counts : Dict[str, int]
        Raw count of each component type present on the board (fixed + solution).
    component_diversity : float
        Shannon entropy of the component-type distribution [0, log₂(8)].
        Higher values indicate more heterogeneous component usage.
    board_fill_ratio : float
        Fraction of board cells occupied by a component [0, 1].
        Proxy for visual/clutter complexity.
    spatial_entropy : float
        Entropy of component presence across 4×4 spatial blocks.
        Captures irregularity of spatial layout (0 = all in one block).
    clustering_coefficient : float
        Mean normalised nearest-neighbour distance among placed components.
        Low values → dense clusters; high values → dispersed layout.
    graph_edge_count : int
        Number of edges in the component interaction graph (gear connections,
        ramp adjacency, crossover adjacency).
    connected_components : int
        Number of weakly connected components in the interaction graph.
        More components → more independent subsystems to reason about.
    longest_dependency_chain : int
        Maximum number of stateful components (bit/gear_bit) encountered
        along any single marble path — a proxy for sequential reasoning depth.
    graph_density : float
        Edge density of the interaction graph [0, 1].
    avg_graph_degree : float
        Mean degree (neighbour count) in the interaction graph.
    symmetry_horizontal : float
        Horizontal reflection symmetry score [0, 1] — how well left half
        mirrors right half in component-type distribution.
    symmetry_vertical : float
        Vertical reflection symmetry score [0, 1].
    path_length_blue : int
        Number of cells visited by the first blue marble in simulation.
    path_length_red : int
        Number of cells visited by the first red marble in simulation.
    branching_factor : int
        Number of distinct path choices a marble can encounter (counts
        cells where multiple adjacent components are downstream-reachable).
    cyclomatic_complexity : int
        McCabe cyclomatic complexity M = E − N + 2P of the component
        interaction graph.  Quantifies the number of linearly independent
        marble-routing paths through the board — a direct control-flow
        complexity metric for procedural reasoning load.
    total_path_length : int
        Sum of cell-visit steps across ALL marbles in the full solution
        run (not just the first).  Captures the total simulation work
        an LLM must trace for procedural understanding.
    state_transition_count : int
        Total number of bit/gear_bit state flips across the full marble
        sequence.  Each flip is a discrete state change the LLM must
        track at timestep t → t+1.
    state_space_size : int
        Number of unique bit-state configurations visited during the
        full run.  Approximates the size of the reachable state space.
    temporal_depth : int
        Number of marbles released before the board either (a) completes
        all marbles, (b) hits a repeated state (limit cycle), or
        (c) reaches the 500-step safety limit.  Measures how deep the
        temporal reasoning chain extends.
    tier : int
        Puzzle tier (1-5) from the index.
    tags : List[str]
        Semantic tags from the index (e.g. "routing", "gear_bit").
    complexity_metrics : Dict[str, float]
        All metrics from ``complexity_metrics.compute_all_metrics()``.
    num_fixed_components : int
        Number of pre-placed (fixed) components on the board.
    num_solution_components : int
        Number of components in the reference solution.
    available_parts_total : int
        Total number of parts available for placement.
    available_parts_types : int
        Number of distinct part types available for placement.
    """

    task_id: str = ""
    component_counts: Dict[str, int] = field(default_factory=dict)
    component_diversity: float = 0.0
    board_fill_ratio: float = 0.0
    spatial_entropy: float = 0.0
    clustering_coefficient: float = 0.0
    graph_edge_count: int = 0
    connected_components: int = 0
    longest_dependency_chain: int = 0
    graph_density: float = 0.0
    avg_graph_degree: float = 0.0
    symmetry_horizontal: float = 0.0
    symmetry_vertical: float = 0.0
    path_length_blue: int = 0
    path_length_red: int = 0
    branching_factor: int = 0
    cyclomatic_complexity: int = 0
    total_path_length: int = 0
    state_transition_count: int = 0
    state_space_size: int = 0
    temporal_depth: int = 0
    tier: int = 0
    tags: List[str] = field(default_factory=list)
    complexity_metrics: Dict[str, float] = field(default_factory=dict)
    num_fixed_components: int = 0
    num_solution_components: int = 0
    available_parts_total: int = 0
    available_parts_types: int = 0
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Flatten features into a single-level dictionary for DataFrame export."""
        d: Dict[str, Any] = {"task_id": self.task_id, "source": self.source}
        # Component counts (one-hot-like columns)
        for ctype in ALL_COMPONENT_TYPES:
            d[f"n_{ctype}"] = self.component_counts.get(ctype, 0)
        # Basic board stats
        d["component_diversity"] = self.component_diversity
        d["board_fill_ratio"] = self.board_fill_ratio
        d["spatial_entropy"] = self.spatial_entropy
        d["clustering_coefficient"] = self.clustering_coefficient
        # Graph metrics
        d["graph_edge_count"] = self.graph_edge_count
        d["connected_components"] = self.connected_components
        d["longest_dependency_chain"] = self.longest_dependency_chain
        d["graph_density"] = self.graph_density
        d["avg_graph_degree"] = self.avg_graph_degree
        # Symmetry
        d["symmetry_horizontal"] = self.symmetry_horizontal
        d["symmetry_vertical"] = self.symmetry_vertical
        # Path metrics
        d["path_length_blue"] = self.path_length_blue
        d["path_length_red"] = self.path_length_red
        d["branching_factor"] = self.branching_factor
        d["cyclomatic_complexity"] = self.cyclomatic_complexity
        d["total_path_length"] = self.total_path_length
        d["state_transition_count"] = self.state_transition_count
        d["state_space_size"] = self.state_space_size
        d["temporal_depth"] = self.temporal_depth
        # Metadata
        d["tier"] = self.tier
        d["num_tags"] = len(self.tags)
        d["num_fixed_components"] = self.num_fixed_components
        d["num_solution_components"] = self.num_solution_components
        d["available_parts_total"] = self.available_parts_total
        d["available_parts_types"] = self.available_parts_types
        # Flatten complexity metrics
        for key, val in self.complexity_metrics.items():
            d[f"cx_{key}"] = val
        # Tag one-hot columns
        all_tags = self._all_known_tags()
        for tag in all_tags:
            d[f"tag_{tag}"] = 1 if tag in self.tags else 0
        return d

    @staticmethod
    def _all_known_tags() -> List[str]:
        """Return the union of all tags observed across the benchmark index."""
        return sorted({
            "routing", "color_symmetry", "triggering", "state_bit",
            "gear_bit", "gear", "looping", "latching", "latch",
            "binary_counting", "overflow", "branching", "merging",
            "interception", "timing", "bit_counting",
        })


# ============================================================================
# Board Feature Extractor
# ============================================================================


class BoardFeatureExtractor:
    """Extract board-level features from Turing Tumble puzzle JSON files.

    Computes structural, spatial, graph-theoretic, symmetry, path, and
    metadata features.  All computations are deterministic and reproducible.

    Parameters
    ----------
    challenges_dir : str or Path
        Directory containing challenge JSON files.
    index_path : str or Path, optional
        Path to INDEX.json for tier/tag metadata.  Defaults to the
        standard location under ``tasks/official/``.
    compute_complexity : bool
        Whether to delegate to ``complexity_metrics.compute_all_metrics()``
        for additional domain-specific metrics (default: True).
    """

    def __init__(
        self,
        challenges_dir: Union[str, Path],
        index_path: Optional[Union[str, Path]] = None,
        compute_complexity: bool = True,
    ) -> None:
        self.challenges_dir = Path(challenges_dir)
        self.compute_complexity = compute_complexity

        # Load index for tier/tag lookups
        if index_path is None:
            index_path = (
                _PROJECT_ROOT / "tasks" / "official" / "INDEX.json"
            )
        self.index: Dict[str, Dict[str, Any]] = {}
        try:
            with open(index_path) as f:
                raw = json.load(f)
            for entry in raw.get("tasks", []):
                self.index[entry["task_id"]] = entry
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.warning("Could not load INDEX.json: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_all(self, pattern: str = "*.json", source: str = "") -> pd.DataFrame:
        """Extract features from all matching challenge JSONs.

        Parameters
        ----------
        pattern : str
            Glob pattern for challenge files (default: ``"*.json"``).
        source : str
            Optional source tag for cross-dataset aggregation (e.g. ``"full"``,
            ``"1comp"``, ``"2comp"``).

        Returns
        -------
        pd.DataFrame
            One row per puzzle, columns = feature names (includes ``source``
            column when specified).
        """
        files = sorted(self.challenges_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No files matching '{pattern}' found in {self.challenges_dir}"
            )
        logger.info("Extracting features from %d puzzle(s)...", len(files))

        rows: List[Dict[str, Any]] = []
        for fp in files:
            try:
                features = self.extract_single(str(fp), source=source)
                rows.append(features.to_dict())
            except Exception as exc:
                logger.warning("Skipping %s: %s", fp.name, exc)

        if not rows:
            raise RuntimeError("No features extracted — all files failed.")

        return pd.DataFrame(rows)

    def extract_single(
        self, task_path: Union[str, Path], source: str = "",
    ) -> BoardFeatures:
        """Extract all features from a single challenge JSON file.

        Parameters
        ----------
        task_path : str or Path
            Path to the challenge JSON file.

        Returns
        -------
        BoardFeatures
            Container with all computed features.

        Raises
        ------
        ValueError
            If the JSON is structurally invalid or missing required fields.
        """
        task_path = Path(task_path)
        with open(task_path) as f:
            task = json.load(f)

        # --- Basic validation -----------------------------------------------
        board_data = task.get("board")
        if not board_data:
            raise ValueError(f"Missing 'board' key in {task_path}")

        task_id = task.get("task_id", task_path.stem)
        width = board_data.get("width", 11)
        height = board_data.get("height", 11)

        # --- Build the board for simulation-based metrics -------------------
        try:
            board = tt_sim.Board.from_task_json(str(task_path))
        except Exception as exc:
            raise ValueError(
                f"Failed to construct Board from {task_path}: {exc}"
            ) from exc

        # --- Component counts -----------------------------------------------
        fixed = board_data.get("fixed_components", [])
        solution = task.get("solution", {}).get("placed_components", [])
        all_comps = fixed + solution
        comp_counts = Counter(
            comp.get("type", "unknown") for comp in all_comps
        )

        # Component-type diversity (Shannon entropy of type distribution)
        # Reasoning: H quantifies uncertainty — a board heavy on one type
        # is conceptually simpler than one mixing many types.
        total = sum(comp_counts.values())
        diversity = 0.0
        if total > 0:
            for count in comp_counts.values():
                p = count / total
                if p > 0:
                    diversity -= p * math.log2(p)

        # --- Board fill ratio -----------------------------------------------
        # The fraction of cells occupied: higher = less room for placement
        # error, more visual clutter to parse.
        fill = len(all_comps) / (width * height) if width * height > 0 else 0.0

        # --- Spatial entropy ------------------------------------------------
        # Divide the board into 4×4 blocks and compute entropy of occupancy
        # counts — captures how evenly components are spread vs. concentrated.
        block_size = 4
        block_rows = max(1, height // block_size)
        block_cols = max(1, width // block_size)
        block_counts: List[int] = []
        for br in range(block_rows):
            for bc in range(block_cols):
                y0, y1 = br * block_size, min((br + 1) * block_size, height)
                x0, x1 = bc * block_size, min((bc + 1) * block_size, width)
                n_in_block = 0
                for comp in all_comps:
                    cx_val, cy_val = comp.get("x", -1), comp.get("y", -1)
                    if x0 <= cx_val < x1 and y0 <= cy_val < y1:
                        n_in_block += 1
                block_counts.append(n_in_block)
        sp_entropy = 0.0
        total_blocks = sum(block_counts)
        if total_blocks > 0:
            for bc_val in block_counts:
                if bc_val > 0:
                    p = bc_val / total_blocks
                    sp_entropy -= p * math.log2(p)

        # --- Clustering coefficient (nearest-neighbour) ---------------------
        # Lower values → components clustered together; higher → dispersed.
        # Meaningful: dense clusters suggest a single "engine," dispersed
        # layouts suggest multiple independent subsystems.
        coords = np.array(
            [[comp["x"], comp["y"]] for comp in all_comps], dtype=float
        )
        if len(coords) > 1:
            dists = squareform(pdist(coords, metric="cityblock"))
            np.fill_diagonal(dists, np.inf)
            nn_dists = np.min(dists, axis=1)  # nearest-neighbour per point
            # Normalise by board diagonal so clustering is scale-invariant
            max_dist = math.sqrt(width**2 + height**2)
            clustering = float(
                np.mean(nn_dists) / max_dist if max_dist > 0 else 0.0
            )
        else:
            clustering = 0.0

        # --- Component interaction graph ------------------------------------
        # Build a graph where edges represent mechanical interaction:
        #   - gear connections (explicit in board.gear_connections)
        #   - ramp adjacency (components within 1 Manhattan step)
        #   - crossovers connect to adjacent cells
        # The graph captures which components can affect each other's
        # behaviour, making it a proxy for reasoning coupling complexity.
        G = _build_interaction_graph(board)

        graph_edges = G.number_of_edges()
        n_nodes = G.number_of_nodes()
        cc_count = (
            nx.number_weakly_connected_components(G)
            if n_nodes > 0
            else 0
        )
        g_density = (
            2.0 * graph_edges / (n_nodes * (n_nodes - 1))
            if n_nodes > 1
            else 0.0
        )
        avg_deg = (
            sum(d for _, d in G.degree()) / n_nodes if n_nodes > 0 else 0.0
        )

        # McCabe cyclomatic complexity of the interaction graph.
        # M = E − N + 2P: linearly independent paths through the board.
        # P = number of weakly connected components (each is an entry
        # point to an independent routing subsystem).
        # Rationale: this is the standard control-flow complexity metric
        # from software engineering, applied to the board's marble-routing
        # graph.  Higher M → more distinct behavioural paths the LLM must
        # mentally simulate during procedural reasoning.
        cyclomatic = (
            graph_edges - n_nodes + 2 * max(cc_count, 1)
            if n_nodes > 0
            else 0
        )

        # --- Longest dependency chain (via path simulation) -----------------
        # Uses the complexity_metrics helper which counts stateful
        # components along the simulated marble path.
        dep_chain = 0
        try:
            dep_chain = cx.dependency_depth(board)
        except Exception:
            pass

        # --- Symmetry scores ------------------------------------------------
        # How well the component-type distribution mirrors across axes.
        # High symmetry = board is "regular" = possibly easier to reason
        # about because patterns repeat.
        sym_h = _compute_horizontal_symmetry(all_comps, width, height)
        sym_v = _compute_vertical_symmetry(all_comps, width, height)

        # --- Path length & branching factor ---------------------------------
        # Path length: how many cells a marble traverses → longer paths
        # demand more steps of procedural reasoning.
        # Branching factor: number of cells where a marble has >1 valid
        # "next component" — quantifies decision-point density.
        path_blue, path_red = 0, 0
        bf = 0
        try:
            board.reset()
            result_b = board.release_marble("blue")
            path_blue = len(result_b.path) if result_b.path else 0
            board.reset()
            result_r = board.release_marble("red")
            path_red = len(result_r.path) if result_r.path else 0
            board.reset()
            bf = _compute_branching_factor(board, width, height)
        except Exception:
            pass

        # --- Temporal state-transition metrics --------------------------------
        # Run the full marble sequence from the challenge's input_sequence
        # and collect state-evolution data: total path length across all
        # marbles, total bit/gear_bit flips, unique state configurations
        # visited, and temporal depth (marbles until completion/limit cycle).
        total_path = 0
        state_flips = 0
        state_space = 0
        temp_depth = 0
        try:
            board.reset()
            input_seq = task.get("input_sequence", ["blue"])
            # Normalise to Side enum values
            norm_seq = [
                "blue" if s in ("blue", "b") else "red" if s in ("red", "r") else s
                for s in input_seq
            ]
            # snapshot initial state
            initial_state = _snapshot_bit_states(board)
            visited_states: set = {initial_state}
            prev_state = dict(initial_state)

            results = board.run(norm_seq)
            temp_depth = len(results)

            for r in results:
                total_path += len(r.path) if r.path else 0
                # Count flips by comparing per-bit states before vs after
                curr_state = r.final_state
                for pos, new_val in curr_state.items():
                    old_val = prev_state.get(pos, -1)
                    if old_val != -1 and old_val != new_val:
                        state_flips += 1
                # Snapshot for state-space tracking
                snap = _snapshot_bit_states(board)
                visited_states.add(snap)
                prev_state = dict(curr_state)

            state_space = len(visited_states)
        except Exception:
            pass

        # --- Metadata from index --------------------------------------------
        idx_entry = self.index.get(task_id, {})
        tier = idx_entry.get("tier", 0)
        tags: List[str] = idx_entry.get("tags", [])

        # --- Available parts ------------------------------------------------
        avail = task.get("available_parts", {})
        avail_total = sum(avail.values())
        avail_types = sum(1 for v in avail.values() if v > 0)

        # --- Complexity metrics (delegate) ----------------------------------
        cx_metrics: Dict[str, float] = {}
        if self.compute_complexity:
            try:
                cx_metrics = cx.compute_all_metrics(board, task)
            except Exception as exc:
                logger.warning(
                    "Complexity metrics failed for %s: %s", task_id, exc
                )

        return BoardFeatures(
            task_id=task_id,
            component_counts=dict(comp_counts),
            component_diversity=round(diversity, 6),
            board_fill_ratio=round(fill, 6),
            spatial_entropy=round(sp_entropy, 6),
            clustering_coefficient=round(clustering, 6),
            graph_edge_count=graph_edges,
            connected_components=cc_count,
            longest_dependency_chain=dep_chain,
            graph_density=round(g_density, 6),
            avg_graph_degree=round(avg_deg, 4),
            symmetry_horizontal=round(sym_h, 6),
            symmetry_vertical=round(sym_v, 6),
            path_length_blue=path_blue,
            path_length_red=path_red,
            branching_factor=bf,
            cyclomatic_complexity=cyclomatic,
            total_path_length=total_path,
            state_transition_count=state_flips,
            state_space_size=state_space,
            temporal_depth=temp_depth,
            tier=tier,
            tags=tags,
            complexity_metrics=cx_metrics,
            num_fixed_components=len(fixed),
            num_solution_components=len(solution),
            available_parts_total=avail_total,
            available_parts_types=avail_types,
            source=source,
        )


# ============================================================================
# Benchmark Integrator
# ============================================================================


class BenchmarkIntegrator:
    """Merge board-level features with benchmark scoring results.

    Accepts a benchmark report JSON (as produced by
    ``scorer/run_benchmark.py --save-report``) and a features DataFrame
    (from ``BoardFeatureExtractor.extract_all()``), and produces a unified
    DataFrame ready for statistical analysis.

    Parameters
    ----------
    features_df : pd.DataFrame
        Board features, one row per task_id.
    """

    def __init__(self, features_df: pd.DataFrame) -> None:
        self.features_df = features_df.set_index("task_id", drop=False)
        # Cache the feature column names (exclude non-feature columns)
        self._feature_cols = self._identify_feature_columns(features_df)

    @staticmethod
    def _identify_feature_columns(df: pd.DataFrame) -> List[str]:
        """Return column names that represent numeric board features."""
        exclude = {"task_id", "success", "component_score", "task_type", "latency_ms", "tokens_used", "source", "benchmark_source"}
        cols: List[str] = []
        for c in df.columns:
            if c in exclude:
                continue
            if c.startswith("metric_"):
                continue
            if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
                cols.append(c)
        return cols

    def load_benchmark_json(self, report_path: Union[str, Path]) -> pd.DataFrame:
        """Load a benchmark report JSON and return a flat results DataFrame.

        Parameters
        ----------
        report_path : str or Path
            Path to the JSON report produced by ``--save-report``.

        Returns
        -------
        pd.DataFrame
            One row per task result, with columns: task_id, task_type,
            success, and per-metric columns.
        """
        with open(report_path) as f:
            report = json.load(f)

        rows: List[Dict[str, Any]] = []
        for result in report.get("results", []):
            raw_success = result.get("success")
            if raw_success is True or raw_success == 1:
                success_val = 1
            elif raw_success is False or raw_success == 0:
                success_val = 0
            else:
                success_val = None
            row: Dict[str, Any] = {
                "task_id": result.get("task_id", ""),
                "task_type": result.get("task_type", ""),
                "success": success_val,
            }
            # Extract continuous component-placement score (0--1) when present.
            # The field is a top-level JSON key (parallel to ``success``) for
            # agentic_synthesis tasks; understanding tasks omit it.
            comp_score = result.get("component_score")
            if comp_score is not None:
                row["component_score"] = float(comp_score)
            # Preserve source tag from benchmark results (but rename to
            # avoid collision with features_df.source — the merge will
            # produce source_x/source_y otherwise).
            src = result.get("source", "")
            if src:
                row["benchmark_source"] = src
            # Flatten metrics dict (skip keys already present as top-level columns)
            _TOP_LEVEL = {"component_score", "component_correct", "component_placed", "component_gt"}
            metrics = result.get("metrics", {})
            for key, val in metrics.items():
                if key in _TOP_LEVEL:
                    continue
                if isinstance(val, (int, float)):
                    row[f"metric_{key}"] = val
                elif isinstance(val, bool):
                    row[f"metric_{key}"] = 1 if val else 0
            # Also copy latency and tokens
            row["latency_ms"] = result.get("latency_ms", 0)
            row["tokens_used"] = result.get("tokens_used", 0)
            rows.append(row)

        return pd.DataFrame(rows)

    def load_benchmark_csv(self, csv_path: Union[str, Path]) -> pd.DataFrame:
        """Load benchmark results from a CSV file.

        Expected columns: ``task_id``, ``task_type``, ``success``, plus
        any metric columns.

        Parameters
        ----------
        csv_path : str or Path
            Path to the CSV file.

        Returns
        -------
        pd.DataFrame
        """
        return pd.read_csv(csv_path)

    def merge(
        self,
        results_df: pd.DataFrame,
        on: str = "task_id",
        how: str = "inner",
    ) -> pd.DataFrame:
        """Merge board features with benchmark results.

        Parameters
        ----------
        results_df : pd.DataFrame
            Benchmark results (from ``load_benchmark_json`` or
            ``load_benchmark_csv``).
        on : str
            Column to join on (default: ``"task_id"``).
        how : str
            Join type (``"inner"``, ``"left"``, ``"right"``, ``"outer"``).

        Returns
        -------
        pd.DataFrame
            Merged DataFrame with both feature and performance columns.
        """
        merged = results_df.merge(
            self.features_df.reset_index(drop=True), on=on, how=how
        )
        # Update feature column list for the merged frame
        self._feature_cols = self._identify_feature_columns(merged)
        logger.info(
            "Merged: %d rows, %d feature columns",
            len(merged), len(self._feature_cols),
        )
        return merged

    @property
    def feature_columns(self) -> List[str]:
        """List of numeric feature column names."""
        return self._feature_cols


# ============================================================================
# Correlation Analyzer
# ============================================================================


@dataclass
class CorrelationResult:
    """Container for a single feature–performance correlation pair.

    Attributes
    ----------
    feature : str
        Name of the board feature.
    performance_metric : str
        Name of the performance metric (e.g. ``"success"``).
    pearson_r : float
        Pearson correlation coefficient [-1, 1].
    pearson_p : float
        Two-tailed p-value for Pearson r.
    spearman_rho : float
        Spearman rank correlation [-1, 1].
    spearman_p : float
        Two-tailed p-value for Spearman rho.
    mutual_info : float
        Normalised mutual information [0, 1] between binned feature and target.
    f_score : float
        ANOVA F-score (only meaningful for classification targets).
    f_pvalue : float
        ANOVA p-value.
    """

    feature: str = ""
    performance_metric: str = ""
    pearson_r: float = 0.0
    pearson_p: float = 1.0
    spearman_rho: float = 0.0
    spearman_p: float = 1.0
    mutual_info: float = 0.0
    f_score: float = 0.0
    f_pvalue: float = 1.0


class CorrelationAnalyzer:
    """Statistical correlation and feature-importance analysis.

    Computes Pearson/Spearman correlations, mutual information, and ANOVA
    F-scores between board features and LLM performance metrics.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged DataFrame from ``BenchmarkIntegrator.merge()``.
    feature_cols : List[str]
        Column names of board features.
    performance_cols : List[str]
        Column names of performance targets.
    """

    def __init__(
        self,
        merged_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        performance_cols: Optional[List[str]] = None,
    ) -> None:
        self.df = merged_df
        self.feature_cols = list(feature_cols or [])
        if not self.feature_cols:
            # Auto-detect: exclude known non-feature columns
            exclude = {
                "task_id", "task_type", "success", "component_score",
                "latency_ms", "tokens_used",
            }
            self.feature_cols = [
                c for c in merged_df.columns
                if c not in exclude
                and pd.api.types.is_numeric_dtype(merged_df[c])
            ]
        self.performance_cols = list(
            performance_cols
            or [c for c in ["component_score", "success"] if c in merged_df.columns]
            or ["success"]
        )

    # ------------------------------------------------------------------
    # Core correlation computation
    # ------------------------------------------------------------------

    def compute_all_correlations(self) -> List[CorrelationResult]:
        """Compute all pairwise feature–performance correlations.

        Returns
        -------
        List[CorrelationResult]
        """
        results: List[CorrelationResult] = []

        for feat in self.feature_cols:
            for perf in self.performance_cols:
                # Drop NaN only for this specific (feature, perf) pair —
                # metric columns may be task-type-specific (e.g.
                # trace_accuracy only defined for understanding tasks).
                pair_cols = [feat, perf]
                df_pair = self.df.dropna(subset=pair_cols, how="any")

                x = df_pair[feat].values.astype(float)
                y = df_pair[perf].values.astype(float)

                if len(x) < 3 or np.std(x) == 0:
                    continue

                # Pearson
                try:
                    pr, pp = stats.pearsonr(x, y)
                except Exception:
                    pr, pp = 0.0, 1.0

                # Spearman
                try:
                    sr, sp = stats.spearmanr(x, y)
                except Exception:
                    sr, sp = 0.0, 1.0

                # Mutual information (binned into deciles)
                mi = 0.0
                try:
                    x_binned = pd.qcut(
                        x, q=min(10, len(np.unique(x))),
                        labels=False, duplicates="drop",
                    )
                    mi = float(
                        mutual_info_classif(
                            x_binned.reshape(-1, 1), y.astype(int),
                            discrete_features=True,
                        )[0]
                    )
                    # Normalise by max possible (log2(n_classes))
                    n_classes = len(np.unique(y))
                    if n_classes > 1:
                        mi = mi / math.log2(n_classes)
                except Exception:
                    mi = 0.0

                # ANOVA F-score (binary classification only)
                f_val, f_p = 0.0, 1.0
                if len(np.unique(y)) == 2:
                    try:
                        f_val, f_p = f_classif(x.reshape(-1, 1), y)
                        f_val = float(f_val[0])
                        f_p = float(f_p[0])
                    except Exception:
                        pass

                results.append(
                    CorrelationResult(
                        feature=feat,
                        performance_metric=perf,
                        pearson_r=round(float(pr), 6),
                        pearson_p=round(float(pp), 8),
                        spearman_rho=round(float(sr), 6),
                        spearman_p=round(float(sp), 8),
                        mutual_info=round(mi, 6),
                        f_score=round(f_val, 4),
                        f_pvalue=round(f_p, 8),
                    )
                )

        return results

    def to_dataframe(
        self, results: List[CorrelationResult],
    ) -> pd.DataFrame:
        """Convert correlation results to a sorted DataFrame.

        Parameters
        ----------
        results : List[CorrelationResult]

        Returns
        -------
        pd.DataFrame
            Sorted by absolute Pearson r descending.
        """
        df = pd.DataFrame([r.__dict__ for r in results])
        if not df.empty:
            df = df.sort_values("pearson_r", key=abs, ascending=False)
        return df

    # ------------------------------------------------------------------
    # Tier-stratified analysis
    # ------------------------------------------------------------------

    def tier_stratified(
        self,
        feature_col: str,
        perf_col: str = "component_score",
    ) -> pd.DataFrame:
        """Compute mean performance by tier for a given feature.

        Parameters
        ----------
        feature_col : str
            Feature to bin into tertiles (low / mid / high).
        perf_col : str
            Performance metric to aggregate.

        Returns
        -------
        pd.DataFrame
            Columns: tier, feature_tertile, mean_perf, std_perf, count.
        """
        df = self.df.dropna(subset=["tier", feature_col, perf_col])
        if df.empty:
            return pd.DataFrame()

        # Bin feature into tertiles within each tier
        results: List[Dict[str, Any]] = []
        for tier_val in sorted(df["tier"].unique()):
            tier_df = df[df["tier"] == tier_val].copy()
            try:
                tier_df["feature_tertile"] = pd.qcut(
                    tier_df[feature_col], q=3, labels=["low", "mid", "high"],
                    duplicates="drop",
                )
            except Exception:
                tier_df["feature_tertile"] = "all"

            for label, group in tier_df.groupby("feature_tertile"):
                results.append({
                    "tier": tier_val,
                    "feature_tertile": str(label),
                    "mean_perf": group[perf_col].mean(),
                    "std_perf": group[perf_col].std(),
                    "count": len(group),
                })

        return pd.DataFrame(results)

    def task_type_split(
        self, results_df: pd.DataFrame,
    ) -> Dict[str, List[CorrelationResult]]:
        """Compute separate correlations for understanding vs synthesis tasks.

        Parameters
        ----------
        results_df : pd.DataFrame
            Merged results.

        Returns
        -------
        Dict[str, List[CorrelationResult]]
            Keys: ``"understanding"``, ``"agentic_synthesis"``, ``"all"``.
        """
        outputs: Dict[str, List[CorrelationResult]] = {}

        for task_type in ["understanding", "agentic_synthesis"]:
            subset = results_df[results_df["task_type"] == task_type]
            if subset.empty:
                outputs[task_type] = []
                continue
            analyzer = CorrelationAnalyzer(
                subset,
                feature_cols=self.feature_cols,
                performance_cols=self.performance_cols,
            )
            outputs[task_type] = analyzer.compute_all_correlations()

        # All combined
        outputs["all"] = self.compute_all_correlations()
        return outputs


# ============================================================================
# Visualizer
# ============================================================================


class BoardVisualizer:
    """Generate publication-quality visualisations for the analytics report.

    All plots use matplotlib with seaborn styling and a consistent
    serif-font aesthetic suitable for academic papers.

    Parameters
    ----------
    output_dir : str or Path
        Directory where plots are saved.
    style : Dict[str, Any], optional
        matplotlib rcParams overrides.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "analytics_output",
        style: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        plot_style = {**_PLOT_CONTEXT}
        if style:
            plot_style.update(style)
        matplotlib.rcParams.update(plot_style)

    # ------------------------------------------------------------------
    # 1. Correlation Heatmap
    # ------------------------------------------------------------------

    def correlation_heatmap(
        self,
        merged_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        perf_cols: Optional[List[str]] = None,
        filename: str = "correlation_heatmap.png",
        title: str = "Board Feature–Performance Correlation Heatmap",
    ) -> Path:
        """Plot a heatmap of Pearson correlations between features and performance.

        Parameters
        ----------
        merged_df : pd.DataFrame
        feature_cols : List[str], optional
            Feature columns. Auto-detected if None.
        perf_cols : List[str], optional
            Performance columns. Defaults to ``["success"]``.
        filename : str
        title : str

        Returns
        -------
        Path
            Output file path.
        """
        if perf_cols is None:
            # Find any column starting with 'metric_' or named 'component_score' / 'success'
            perf_cols = [
                c for c in merged_df.columns
                if c in ("component_score", "success") or c.startswith("metric_")
            ]
            if not perf_cols:
                perf_cols = ["component_score"]
        if feature_cols is None:
            exclude = set(perf_cols) | {"task_id", "task_type", "benchmark_source", "source"}
            feature_cols = [
                c for c in merged_df.columns
                if c not in exclude
                and not c.startswith("metric_")   # skip benchmark-internal metrics
                and pd.api.types.is_numeric_dtype(merged_df[c])
            ]

        # Correlation matrix
        corr_df = merged_df[feature_cols + perf_cols].corr(method="pearson")
        # Extract the feature × performance submatrix
        corr_sub = corr_df.loc[feature_cols, perf_cols]

        # Drop rows and columns that are entirely NaN — these happen when
        # features are constant (zero variance) or performance metrics are
        # undefined for a given task type.  This eliminates white-space
        # gaps in the heatmap.
        corr_sub = corr_sub.dropna(how="all", axis=0).dropna(how="all", axis=1)

        # Select top-N features by max absolute correlation for readability
        if len(feature_cols) > 30:
            top_n = 30
            abs_max = corr_sub.abs().max(axis=1).sort_values(ascending=False)
            top_features = abs_max.head(top_n).index.tolist()
            corr_sub = corr_sub.loc[top_features]

        # Dynamic figure size: scale by number of features (rows) and
        # performance columns so annotation values remain legible.
        n_rows, n_cols = corr_sub.shape
        cell_h = 0.55   # inches per row — enough for .2f annotations
        cell_w = 1.1    # inches per column
        fig_w = max(8, n_cols * cell_w + 2)
        fig_h = max(6, n_rows * cell_h + 1.5)
        annot_fs = min(11, max(7, 120 // max(n_rows, 1)))

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.heatmap(
            corr_sub,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            square=False,
            linewidths=0.5,
            annot_kws={"fontsize": annot_fs},
            cbar_kws={"shrink": 0.8, "label": "Pearson r"},
            ax=ax,
        )
        ax.set_title(title, fontweight="bold", pad=16)
        ax.set_xlabel("Performance Metric")
        ax.set_ylabel("Board Feature")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)

        path = self.output_dir / filename
        fig.savefig(path)
        plt.close(fig)
        logger.info("Saved heatmap: %s", path)
        return path

    # ------------------------------------------------------------------
    # 2. Feature Distribution by Tier
    # ------------------------------------------------------------------

    def feature_distribution_by_tier(
        self,
        merged_df: pd.DataFrame,
        feature_col: str,
        filename: Optional[str] = None,
    ) -> Path:
        """Plot a feature's distribution faceted by tier.

        Parameters
        ----------
        merged_df : pd.DataFrame
        feature_col : str
        filename : str, optional

        Returns
        -------
        Path
        """
        if filename is None:
            filename = f"dist_{feature_col}_by_tier.png"

        df = merged_df.dropna(subset=[feature_col, "tier"])
        tiers = sorted(df["tier"].unique())
        n_tiers = len(tiers)

        fig, axes = plt.subplots(
            1, n_tiers, figsize=(4 * n_tiers, 4), sharey=True,
        )
        if n_tiers == 1:
            axes = [axes]

        for ax, tier_val in zip(axes, tiers):
            data = df[df["tier"] == tier_val][feature_col]
            sns.histplot(data, kde=True, bins=12, ax=ax, color="steelblue")
            ax.set_title(f"Tier {tier_val} (n={len(data)})")
            ax.set_xlabel(feature_col)

        fig.suptitle(
            f"Distribution of {feature_col} by Tier",
            fontweight="bold", y=1.02,
        )
        fig.tight_layout()

        path = self.output_dir / filename
        fig.savefig(path)
        plt.close(fig)
        logger.info("Saved distribution plot: %s", path)
        return path

    # ------------------------------------------------------------------
    # 3. Top-Correlated Scatter Plots
    # ------------------------------------------------------------------

    def scatter_top_correlations(
        self,
        merged_df: pd.DataFrame,
        perf_col: str = "component_score",
        top_n: int = 6,
        filename: str = "scatter_top_correlations.png",
    ) -> Path:
        """Scatter plots of top-N most correlated features against a performance metric.

        Parameters
        ----------
        merged_df : pd.DataFrame
        perf_col : str
        top_n : int
        filename : str

        Returns
        -------
        Path
        """
        # Compute correlations
        feature_cols = [
            c for c in merged_df.columns
            if c not in {"task_id", "task_type", "tier", perf_col}
            and pd.api.types.is_numeric_dtype(merged_df[c])
        ]
        corrs: List[Tuple[str, float]] = []
        for feat in feature_cols:
            try:
                r, _ = stats.pearsonr(
                    merged_df[feat].dropna(),
                    merged_df[perf_col].loc[merged_df[feat].notna()],
                )
                if not np.isnan(r):
                    corrs.append((feat, abs(r)))
            except Exception:
                pass

        corrs.sort(key=lambda x: x[1], reverse=True)
        top_features = [f for f, _ in corrs[:top_n]]

        if not top_features:
            logger.warning("No features with valid correlations for scatter plot; skipping.")
            return self.output_dir / filename

        n_cols = min(3, len(top_features))
        n_rows = math.ceil(len(top_features) / n_cols)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(5 * n_cols, 4 * n_rows),
        )
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        i = -1
        for i, feat in enumerate(top_features):
            ax = axes_flat[i]
            df_plot = merged_df.dropna(subset=[feat, perf_col])
            # Jitter for binary targets
            y_vals = df_plot[perf_col]
            if len(np.unique(y_vals)) <= 2:
                y_vals = y_vals + np.random.uniform(-0.05, 0.05, size=len(y_vals))
            ax.scatter(
                df_plot[feat], y_vals,
                alpha=0.6, s=40, c="steelblue", edgecolors="white",
                linewidth=0.3,
            )
            # Trend line
            try:
                slope, intercept, r_val, _, _ = stats.linregress(
                    df_plot[feat], df_plot[perf_col],
                )
                xs = np.linspace(df_plot[feat].min(), df_plot[feat].max(), 50)
                ax.plot(xs, slope * xs + intercept, "r--", lw=1.5, alpha=0.7)
                ax.set_title(
                    f"{feat}\nr={r_val:.3f}", fontsize=10,
                )
            except Exception:
                ax.set_title(feat, fontsize=10)

            ax.set_xlabel(feat, fontsize=9)
            ax.set_ylabel(perf_col, fontsize=9)

        # Hide unused subplots
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(
            f"Top-{top_n} Features vs {perf_col}",
            fontweight="bold", y=1.01,
        )
        fig.tight_layout()

        path = self.output_dir / filename
        fig.savefig(path)
        plt.close(fig)
        logger.info("Saved scatter plot: %s", path)
        return path

    # ------------------------------------------------------------------
    # 4. Performance Degradation Curve
    # ------------------------------------------------------------------

    def degradation_curve(
        self,
        merged_df: pd.DataFrame,
        complexity_col: str,
        perf_col: str = "component_score",
        n_bins: int = 8,
        filename: Optional[str] = None,
    ) -> Path:
        """Plot mean performance as a function of a complexity metric.

        Parameters
        ----------
        merged_df : pd.DataFrame
        complexity_col : str
            Feature to use as x-axis (complexity measure).
        perf_col : str
            Performance metric on y-axis.
        n_bins : int
            Number of bins for the complexity axis.
        filename : str, optional

        Returns
        -------
        Path
        """
        if filename is None:
            filename = f"degradation_{complexity_col}_vs_{perf_col}.png"

        df = merged_df.dropna(subset=[complexity_col, perf_col])
        df = df.copy()
        try:
            df["bin"] = pd.qcut(df[complexity_col], q=n_bins, duplicates="drop")
        except Exception:
            df["bin"] = pd.cut(df[complexity_col], bins=n_bins)

        summary = df.groupby("bin").agg(
            mean=(perf_col, "mean"),
            sem=(perf_col, "sem"),
        ).reset_index()
        summary["bin_mid"] = summary["bin"].apply(
            lambda iv: iv.mid if hasattr(iv, "mid") else 0,
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(
            summary["bin_mid"], summary["mean"], yerr=summary["sem"],
            fmt="o-", capsize=5, capthick=1.5, color="steelblue",
            markersize=8, linewidth=2,
        )
        ax.fill_between(
            summary["bin_mid"],
            summary["mean"] - summary["sem"],
            summary["mean"] + summary["sem"],
            alpha=0.2, color="steelblue",
        )
        ax.set_xlabel(complexity_col)
        ax.set_ylabel(f"Mean {perf_col}")
        ax.set_title(
            f"Performance Degradation by {complexity_col}",
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = self.output_dir / filename
        fig.savefig(path)
        plt.close(fig)
        logger.info("Saved degradation curve: %s", path)
        return path

    # ------------------------------------------------------------------
    # 5. Multi-feature degradation grid
    # ------------------------------------------------------------------

    def degradation_grid(
        self,
        merged_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        perf_col: str = "component_score",
        n_top: int = 9,
        filename: str = "degradation_grid.png",
    ) -> Path:
        """Grid of degradation curves for the top-N most correlated features.

        Parameters
        ----------
        merged_df : pd.DataFrame
        feature_cols : List[str], optional
        perf_col : str
        n_top : int
        filename : str

        Returns
        -------
        Path
        """
        if feature_cols is None:
            exclude = {"task_id", "task_type", "tier", perf_col}
            feature_cols = [
                c for c in merged_df.columns
                if c not in exclude
                and pd.api.types.is_numeric_dtype(merged_df[c])
            ]

        # Rank by absolute Pearson correlation
        rankings: List[Tuple[str, float]] = []
        df_clean = merged_df.dropna(subset=[perf_col])
        for feat in feature_cols:
            sub = df_clean.dropna(subset=[feat])
            if len(sub) < 5:
                continue
            try:
                r, _ = stats.pearsonr(sub[feat], sub[perf_col])
                if not np.isnan(r):
                    rankings.append((feat, abs(r)))
            except Exception:
                pass
        rankings.sort(key=lambda x: x[1], reverse=True)
        top_features = [f for f, _ in rankings[:n_top]]

        if not top_features:
            logger.warning(
                "No features with valid correlations for degradation grid; skipping.",
            )
            return self.output_dir / filename

        n_cols = 3
        n_rows = math.ceil(len(top_features) / n_cols)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(5 * n_cols, 3.5 * n_rows),
        )
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        i = -1
        for i, feat in enumerate(top_features):
            ax = axes_flat[i]
            sub = merged_df.dropna(subset=[feat, perf_col]).copy()
            try:
                sub["bin"] = pd.qcut(sub[feat], q=6, duplicates="drop")
            except Exception:
                sub["bin"] = pd.cut(sub[feat], bins=6)
            summ = sub.groupby("bin").agg(
                mean=(perf_col, "mean"),
                sem=(perf_col, "sem"),
            ).reset_index()
            summ["mid"] = summ["bin"].apply(
                lambda iv: iv.mid if hasattr(iv, "mid") else 0,
            )
            ax.errorbar(
                summ["mid"], summ["mean"], yerr=summ["sem"],
                fmt="o-", capsize=3, color="steelblue", markersize=5,
            )
            r_val = stats.pearsonr(sub[feat], sub[perf_col])[0]
            ax.set_title(f"{feat} (r={r_val:.2f})", fontsize=9)
            ax.tick_params(labelsize=7)

        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(
            f"Performance Degradation by Top-{n_top} Complexity Metrics",
            fontweight="bold", y=1.01,
        )
        fig.tight_layout()

        path = self.output_dir / filename
        fig.savefig(path)
        plt.close(fig)
        logger.info("Saved degradation grid: %s", path)
        return path


# ============================================================================
# Report Generator
# ============================================================================


class ReportGenerator:
    """Generate structured markdown/tabular reports for academic use.

    Parameters
    ----------
    output_dir : str or Path
        Directory where reports are saved.
    """

    def __init__(
        self, output_dir: Union[str, Path] = "analytics_output",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def summary_table(
        self,
        merged_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        filename: str = "summary_statistics.csv",
    ) -> pd.DataFrame:
        """Compute and save a summary statistics table.

        Parameters
        ----------
        merged_df : pd.DataFrame
        feature_cols : List[str], optional
        filename : str

        Returns
        -------
        pd.DataFrame
            Columns: feature, mean, std, min, q25, median, q75, max.
        """
        if feature_cols is None:
            exclude = {"task_id", "task_type"}
            feature_cols = [
                c for c in merged_df.columns
                if c not in exclude
                and pd.api.types.is_numeric_dtype(merged_df[c])
            ]

        stats_df = merged_df[feature_cols].describe().T
        stats_df = stats_df.rename(
            columns={
                "mean": "mean", "std": "std", "min": "min",
                "25%": "q25", "50%": "median", "75%": "q75", "max": "max",
            }
        )
        # Select and reorder
        stats_df = stats_df[["mean", "std", "min", "q25", "median", "q75", "max"]]
        stats_df = stats_df.round(4)

        path = self.output_dir / filename
        stats_df.to_csv(path)
        logger.info("Saved summary stats: %s", path)
        return stats_df

    def ranked_correlations(
        self,
        corr_results: List[CorrelationResult],
        filename: str = "ranked_correlations.csv",
        top_n: int = 30,
    ) -> pd.DataFrame:
        """Save the top-N features ranked by absolute Pearson r.

        Parameters
        ----------
        corr_results : List[CorrelationResult]
        filename : str
        top_n : int

        Returns
        -------
        pd.DataFrame
        """
        df = pd.DataFrame([r.__dict__ for r in corr_results])
        if df.empty:
            return df

        # Keep only the max-abs-r row per feature–perf pair
        df["abs_r"] = df["pearson_r"].abs()
        df = df.sort_values("abs_r", ascending=False).head(top_n)
        df = df.drop(columns=["abs_r"])

        path = self.output_dir / filename
        df.to_csv(path, index=False)
        logger.info("Saved ranked correlations: %s", path)
        return df

    def academic_report(
        self,
        corr_results: List[CorrelationResult],
        summary_df: pd.DataFrame,
        merged_df: pd.DataFrame,
        perf_col: str = "component_score",
        filename: str = "academic_results.md",
    ) -> Path:
        """Generate a markdown report formatted for academic methods/results sections.

        Parameters
        ----------
        corr_results : List[CorrelationResult]
        summary_df : pd.DataFrame
        merged_df : pd.DataFrame
        perf_col : str
        filename : str

        Returns
        -------
        Path
        """
        df_corr = pd.DataFrame([r.__dict__ for r in corr_results])
        if df_corr.empty:
            logger.warning("No correlation results to report.")
            path = self.output_dir / filename
            path.write_text("# No data available.\n")
            return path

        # Top features
        df_corr["abs_r"] = df_corr["pearson_r"].abs()
        top10 = df_corr.sort_values("abs_r", ascending=False).head(10)

        # Significant features (p < 0.05)
        sig = df_corr[df_corr["pearson_p"] < 0.05].sort_values(
            "abs_r", ascending=False,
        )

        n_puzzles = merged_df["task_id"].nunique() if "task_id" in merged_df.columns else len(merged_df)
        n_features = len(df_corr["feature"].unique())

        lines = [
            "# Board Feature Analysis — Academic Results",
            "",
            f"**Dataset:** {n_puzzles} puzzles, {n_features} board features extracted.",
            "",
            "## Summary Statistics",
            "",
            "Descriptive statistics for all extracted board features are presented in ",
            f"Table 1 (see `summary_statistics.csv`).  The feature set encompasses "
            f"component-type distributions, spatial metrics (fill ratio, spatial entropy, "
            f"nearest-neighbour clustering), graph-theoretic measures (interaction-graph "
            f"density, connected components), symmetry indices, marble-path characteristics, "
            f"and the BICI composite complexity index.",
            "",
            "## Correlation Analysis",
            "",
            f"We computed Pearson and Spearman correlation coefficients between each board "
            f"feature and LLM performance ({perf_col}), along with mutual information and "
            f"ANOVA F-scores.  Table 2 (see `ranked_correlations.csv`) lists the top-30 "
            f"features ranked by absolute Pearson *r*.",
            "",
            "### Top-10 Features by |Pearson r|",
            "",
            "| Rank | Feature | Pearson r | Spearman ρ | MI | p-value |",
            "|------|---------|-----------|-------------|-----|---------|",
        ]
        for i, (_, row) in enumerate(top10.iterrows(), 1):
            lines.append(
                f"| {i} | `{row['feature']}` | {row['pearson_r']:.3f} "
                f"| {row['spearman_rho']:.3f} | {row['mutual_info']:.3f} "
                f"| {row['pearson_p']:.4f} |"
            )

        n_sig = len(sig)
        lines += [
            "",
            f"**{n_sig} features** showed statistically significant Pearson correlations "
            f"(*p* < 0.05) with {perf_col}.",
            "",
            "### Interpretation",
            "",
            "Features with high absolute correlation magnitudes are strong candidates for "
            "predictive modeling of LLM puzzle-solving difficulty.  Positive correlations "
            "suggest that higher feature values coincide with higher success rates, while "
            "negative correlations indicate that the feature captures an aspect of puzzle "
            "complexity that degrades LLM performance.",
            "",
            "Correlation does not imply causation; multimodel experiments across different "
            "LLM architectures and sizes are needed to establish robust complexity predictors.",
            "",
            "---",
            "",
            "*Report generated by `analytics/board_analytics.py`*",
        ]

        path = self.output_dir / filename
        path.write_text("\n".join(lines))
        logger.info("Saved academic report: %s", path)
        return path


# ============================================================================
# Pipeline & CLI
# ============================================================================


def run_pipeline(
    challenges_dir: Union[str, Path],
    results_path: Union[str, Path],
    output_dir: Union[str, Path] = "analytics_output",
    pattern: str = "*.json",
    perf_col: str = "component_score",
    compute_complexity: bool = True,
) -> Dict[str, Any]:
    """Run the full analytics pipeline end-to-end.

    Parameters
    ----------
    challenges_dir : str or Path
        Directory containing challenge JSON files.
    results_path : str or Path
        Path to benchmark results (JSON or CSV).
    output_dir : str or Path
        Directory for output artefacts.
    pattern : str
        Glob pattern for challenge files.
    perf_col : str
        Primary performance metric for correlation analysis.
    compute_complexity : bool
        Whether to compute complexity_metrics features.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys: ``features_df``, ``merged_df``, ``correlations``,
        ``corr_df``, ``summary_df``, ``output_dir``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. Feature Extraction --------------------------------------------
    logger.info("=== Step 1/5: Extracting board features ===")
    extractor = BoardFeatureExtractor(
        challenges_dir=challenges_dir,
        compute_complexity=compute_complexity,
    )
    features_df = extractor.extract_all(pattern=pattern)
    logger.info("Extracted %d features for %d puzzles", features_df.shape[1], features_df.shape[0])

    # ---- 2. Merge with Benchmark Results ----------------------------------
    logger.info("=== Step 2/5: Merging with benchmark results ===")
    integrator = BenchmarkIntegrator(features_df)
    results_path = Path(results_path)
    if results_path.suffix == ".csv":
        results_df = integrator.load_benchmark_csv(results_path)
    else:
        results_df = integrator.load_benchmark_json(results_path)
    merged_df = integrator.merge(results_df)

    # Fall back to success if the preferred performance column is absent
    # (e.g. pre-component_score benchmark reports).
    if perf_col not in merged_df.columns:
        logger.info(
            "perf_col=%r not found in results; falling back to 'success'", perf_col,
        )
        perf_col = "success"

    # ---- 3. Correlation Analysis ------------------------------------------
    logger.info("=== Step 3/5: Computing correlations ===")
    # Use core performance metrics — exclude metric_* columns which are
    # also present as features (avoid tautological self-correlations).
    perf_cols = [c for c in [perf_col, "latency_ms", "tokens_used"] if c in merged_df.columns]
    # Also include any metric_* columns that exist but NOT as feature columns
    for c in merged_df.columns:
        if c.startswith("metric_") and c not in integrator.feature_columns:
            perf_cols.append(c)
    perf_cols = list(dict.fromkeys(perf_cols))  # unique, order-preserving

    analyzer = CorrelationAnalyzer(
        merged_df,
        feature_cols=integrator.feature_columns,
        performance_cols=perf_cols,
    )
    corr_results = analyzer.compute_all_correlations()
    corr_df = analyzer.to_dataframe(corr_results)

    # Task-type split
    type_split = analyzer.task_type_split(merged_df)

    # ---- 4. Visualizations ------------------------------------------------
    logger.info("=== Step 4/5: Generating visualizations ===")
    viz = BoardVisualizer(output_dir=out)

    viz.correlation_heatmap(merged_df, perf_cols=perf_cols)

    # Feature distribution by tier for key metrics
    key_features = ["component_diversity", "board_fill_ratio", "graph_density",
                    "longest_dependency_chain"]
    for kf in key_features:
        if kf in merged_df.columns:
            viz.feature_distribution_by_tier(merged_df, kf)

    viz.scatter_top_correlations(merged_df, perf_col=perf_col)

    # Degradation curves
    if corr_df is not None and not corr_df.empty:
        top_corr_features = corr_df["feature"].head(9).tolist()
        viz.degradation_grid(
            merged_df,
            feature_cols=top_corr_features,
            perf_col=perf_col,
        )
        # Single detailed curve for the top feature
        if top_corr_features:
            viz.degradation_curve(
                merged_df,
                complexity_col=top_corr_features[0],
                perf_col=perf_col,
            )

    # ---- 5. Report Generation ---------------------------------------------
    logger.info("=== Step 5/5: Generating reports ===")
    reporter = ReportGenerator(output_dir=out)
    summary_df = reporter.summary_table(merged_df)
    reporter.ranked_correlations(corr_results)
    reporter.academic_report(corr_results, summary_df, merged_df, perf_col=perf_col)

    logger.info("=== Pipeline complete. Output: %s ===", out)

    return {
        "features_df": features_df,
        "merged_df": merged_df,
        "correlations": corr_results,
        "corr_df": corr_df,
        "summary_df": summary_df,
        "task_type_split": type_split,
        "output_dir": out,
    }


# ============================================================================
# Aggregated Multi-Source Pipeline
# ============================================================================


def run_aggregated_pipeline(
    source_dirs: Dict[str, Union[str, Path]],
    aggregated_results_path: Union[str, Path],
    output_dir: Union[str, Path] = "analytics_output/aggregated",
    pattern: str = "*.json",
    perf_col: str = "component_score",
    compute_complexity: bool = True,
) -> Dict[str, Any]:
    """Run analytics across multiple challenge sources and aggregate results.

    Extracts features from each source directory (tagged with the source
    key), merges with a single aggregated benchmark results file (which
    must include a ``source`` column), and produces a combined analysis
    with source-aware visualisations.

    Parameters
    ----------
    source_dirs : Dict[str, str | Path]
        Mapping of source name → challenge directory path.
        e.g. ``{"full": "tasks/official/challenges/json",
        "1comp": "tasks/challenges_1comp"}``.
    aggregated_results_path : str or Path
        Path to benchmark results JSON/CSV containing all sources.
        Each result row must include a ``"source"`` field matching one
        of the keys in ``source_dirs``.
    output_dir : str or Path
    pattern : str
    perf_col : str
    compute_complexity : bool

    Returns
    -------
    Dict[str, Any]
        Same shape as ``run_pipeline``, plus ``"source_comparison"``
        DataFrame with per-source aggregated stats.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. Extract features from each source -------------------------------
    logger.info("=== Aggregated Step 1/6: Extracting features from %d sources ===", len(source_dirs))
    all_features_dfs: List[pd.DataFrame] = []
    for src_name, src_dir in source_dirs.items():
        src_path = Path(src_dir)
        if not src_path.is_dir():
            logger.warning("Skipping non-existent source dir: %s", src_path)
            continue
        extractor = BoardFeatureExtractor(
            challenges_dir=src_path,
            compute_complexity=compute_complexity,
        )
        src_df = extractor.extract_all(pattern=pattern, source=src_name)
        logger.info("  %s: %d puzzles, %d features", src_name, len(src_df), src_df.shape[1])
        all_features_dfs.append(src_df)

    if not all_features_dfs:
        raise RuntimeError("No source directories produced features.")

    features_df_all = pd.concat(all_features_dfs, ignore_index=True)
    logger.info("Combined features: %d rows × %d columns", *features_df_all.shape)

    # ---- 2. Merge with aggregated benchmark results -------------------------
    logger.info("=== Aggregated Step 2/6: Merging with aggregated benchmark results ===")
    integrator = BenchmarkIntegrator(features_df_all)
    results_path = Path(aggregated_results_path)
    if results_path.suffix == ".csv":
        results_df = integrator.load_benchmark_csv(results_path)
    else:
        results_df = integrator.load_benchmark_json(results_path)
    merged_df = integrator.merge(results_df)
    logger.info("Merged: %d rows × %d columns", *merged_df.shape)

    # Fall back to success if the preferred performance column is absent.
    if perf_col not in merged_df.columns:
        logger.info(
            "perf_col=%r not found in results; falling back to 'success'", perf_col,
        )
        perf_col = "success"

    # ---- 3. Source-comparison table -----------------------------------------
    logger.info("=== Aggregated Step 3/6: Source comparison ===")
    source_comparison = _build_source_comparison(merged_df, perf_col)

    # ---- 4. Correlation Analysis (source-aware) -----------------------------
    logger.info("=== Aggregated Step 4/6: Computing correlations ===")
    perf_cols = [c for c in [perf_col, "latency_ms", "tokens_used"] if c in merged_df.columns]
    for c in merged_df.columns:
        if c.startswith("metric_") and c not in integrator.feature_columns:
            perf_cols.append(c)
    perf_cols = list(dict.fromkeys(perf_cols))

    analyzer = CorrelationAnalyzer(
        merged_df,
        feature_cols=integrator.feature_columns,
        performance_cols=perf_cols,
    )
    corr_results = analyzer.compute_all_correlations()
    corr_df = analyzer.to_dataframe(corr_results)

    # Per-source correlation splits
    source_splits: Dict[str, List[CorrelationResult]] = {}
    for src_name in source_dirs:
        source_subset = merged_df[merged_df["source"] == src_name]
        if len(source_subset) < 5:
            source_splits[src_name] = []
            continue
        src_analyzer = CorrelationAnalyzer(
            source_subset,
            feature_cols=integrator.feature_columns,
            performance_cols=perf_cols,
        )
        source_splits[src_name] = src_analyzer.compute_all_correlations()

    type_split = analyzer.task_type_split(merged_df)

    # ---- 5. Visualizations --------------------------------------------------
    logger.info("=== Aggregated Step 5/6: Generating visualizations ===")
    viz = BoardVisualizer(output_dir=out)

    # Overall heatmap
    viz.correlation_heatmap(merged_df, perf_cols=perf_cols)

    # Per-source performance distributions
    _plot_source_performance_boxplot(merged_df, perf_col, out, viz)

    # Feature distributions by source
    for kf in ["component_diversity", "board_fill_ratio", "graph_density",
               "longest_dependency_chain"]:
        if kf in merged_df.columns:
            _plot_feature_by_source(merged_df, kf, out, viz)

    # Scatter and degradation
    viz.scatter_top_correlations(merged_df, perf_col=perf_col)
    if corr_df is not None and not corr_df.empty:
        top_corr_features = corr_df["feature"].head(9).tolist()
        viz.degradation_grid(
            merged_df, feature_cols=top_corr_features, perf_col=perf_col,
        )
        if top_corr_features:
            viz.degradation_curve(
                merged_df, complexity_col=top_corr_features[0],
                perf_col=perf_col,
            )

    # ---- 6. Reports ---------------------------------------------------------
    logger.info("=== Aggregated Step 6/6: Generating reports ===")
    reporter = ReportGenerator(output_dir=out)
    summary_df = reporter.summary_table(merged_df)
    reporter.ranked_correlations(corr_results)
    reporter.academic_report(corr_results, summary_df, merged_df, perf_col=perf_col)

    # Save source-comparison CSV
    sc_path = out / "source_comparison.csv"
    source_comparison.to_csv(sc_path, index=False)
    logger.info("Saved source comparison: %s", sc_path)

    logger.info("=== Aggregated pipeline complete. Output: %s ===", out)
    return {
        "features_df": features_df_all,
        "merged_df": merged_df,
        "correlations": corr_results,
        "corr_df": corr_df,
        "summary_df": summary_df,
        "task_type_split": type_split,
        "source_splits": source_splits,
        "source_comparison": source_comparison,
        "output_dir": out,
    }


def _build_source_comparison(merged_df: pd.DataFrame, perf_col: str) -> pd.DataFrame:
    """Build a per-source summary table."""
    if "source" not in merged_df.columns:
        return pd.DataFrame()

    rows = []
    for src in sorted(merged_df["source"].dropna().unique()):
        subset = merged_df[merged_df["source"] == src]
        rows.append({
            "source": src,
            "n_tasks": len(subset),
            "n_puzzles": subset["task_id"].nunique(),
            f"mean_{perf_col}": subset[perf_col].mean(),
            f"std_{perf_col}": subset[perf_col].std(),
            "mean_latency_ms": subset["latency_ms"].mean() if "latency_ms" in subset.columns else 0,
            "mean_tokens": subset["tokens_used"].mean() if "tokens_used" in subset.columns else 0,
        })
    return pd.DataFrame(rows)


def _plot_source_performance_boxplot(
    merged_df: pd.DataFrame, perf_col: str, out_dir: Path, viz: BoardVisualizer,
) -> None:
    """Box plot of performance by source."""
    if "source" not in merged_df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    df_plot = merged_df.dropna(subset=[perf_col, "source"])
    order = sorted(df_plot["source"].unique())
    sns.boxplot(
        data=df_plot, x="source", y=perf_col, order=order,
        palette="Set2", ax=ax,
    )
    sns.stripplot(
        data=df_plot, x="source", y=perf_col, order=order,
        color="black", alpha=0.3, size=4, ax=ax,
    )
    ax.set_title(f"{perf_col} by Challenge Source", fontweight="bold")
    ax.set_xlabel("Source")
    fig.tight_layout()
    path = out_dir / f"boxplot_{perf_col}_by_source.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved source boxplot: %s", path)


def _plot_feature_by_source(
    merged_df: pd.DataFrame, feature_col: str, out_dir: Path, viz: BoardVisualizer,
) -> None:
    """KDE distribution of a feature by source."""
    if "source" not in merged_df.columns:
        return
    df_plot = merged_df.dropna(subset=[feature_col, "source"])
    fig, ax = plt.subplots(figsize=(8, 5))
    for src in sorted(df_plot["source"].unique()):
        data = df_plot[df_plot["source"] == src][feature_col]
        if len(data) > 1:
            sns.kdeplot(data, label=src, fill=True, alpha=0.3, ax=ax)
    ax.set_title(f"{feature_col} by Source", fontweight="bold")
    ax.set_xlabel(feature_col)
    ax.legend(title="Source")
    fig.tight_layout()
    path = out_dir / f"kde_{feature_col}_by_source.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved source KDE: %s", path)


def main() -> int:
    """Command-line entry point for the board analytics pipeline."""
    parser = argparse.ArgumentParser(
        description="Turing Tumble Board Analytics Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with default paths
  python analytics/board_analytics.py

  # Custom paths
  python analytics/board_analytics.py \\
      --challenges-dir tasks/official/challenges/json \\
      --results scorer/scorer/benchmark_results/benchmark_2025-06-13.json \\
      --output-dir analytics_output

  # Without complexity metrics (faster)
  python analytics/board_analytics.py --no-complexity
        """,
    )
    parser.add_argument(
        "--challenges-dir",
        type=Path,
        default=_PROJECT_ROOT / "tasks" / "official" / "challenges" / "json",
        help="Directory containing challenge JSON files",
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to benchmark results JSON or CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analytics_output"),
        help="Output directory for plots and reports",
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern for challenge files (default: *.json)",
    )
    parser.add_argument(
        "--perf-col",
        default="component_score",
        help="Primary performance metric column (default: success)",
    )
    parser.add_argument(
        "--no-complexity",
        action="store_true",
        help="Skip complexity_metrics computation for faster runs",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization generation",
    )

    args = parser.parse_args()

    if not args.challenges_dir.is_dir():
        logger.error("Challenges directory not found: %s", args.challenges_dir)
        return 1
    if not Path(args.results).is_file():
        logger.error("Results file not found: %s", args.results)
        return 1

    result = run_pipeline(
        challenges_dir=args.challenges_dir,
        results_path=args.results,
        output_dir=args.output_dir,
        pattern=args.pattern,
        perf_col=args.perf_col,
        compute_complexity=not args.no_complexity,
    )

    # Print summary to stdout
    corr_df = result["corr_df"]
    if corr_df is not None and not corr_df.empty:
        print("\nTop-10 Features by |Pearson r|:")
        print(corr_df[["feature", "performance_metric", "pearson_r", "pearson_p"]].head(10).to_string(index=False))
    else:
        print("\nNo correlations computed (insufficient data).")

    print(f"\nAll outputs saved to: {result['output_dir']}")
    return 0


# ============================================================================
# Helper Functions (module-private)
# ============================================================================


def _snapshot_bit_states(board: tt_sim.Board) -> Tuple[Tuple[Tuple[int, int], int], ...]:
    """Return a frozen snapshot of all bit/gear_bit (position, state) pairs.

    Used for state-space tracking during temporal dynamics analysis.
    The result is hashable (tuple of tuples) so it can be stored in a set.
    """
    items: List[Tuple[Tuple[int, int], int]] = []
    for pos, comp in board.components.items():
        if isinstance(comp, (tt_sim.Bit, tt_sim.GearBit)):
            items.append((pos, comp.state))
    return tuple(sorted(items))


def _build_interaction_graph(board: tt_sim.Board) -> nx.DiGraph:
    """Build a directed interaction graph from board components.

    Edges represent potential mechanical interaction:
    - Gear connections (from ``board.gear_connections``)
    - Ramp adjacency (ramps point diagonally to next cell)
    - Crossover adjacency (crossovers connect to diagonal neighbours)
    - Bit/gear_bit adjacency (both exit directions)

    The graph is directed because some interactions are directional
    (e.g., ramp_right goes to lower-right, not vice versa).

    Returns
    -------
    nx.DiGraph
    """
    G = nx.DiGraph()

    # Add nodes for all components
    for (x, y), comp in board.components.items():
        G.add_node((x, y), type=comp.component_type.value)

    # Gear connections (bidirectional)
    for (x, y), neighbours in board.gear_connections.items():
        for (nx_val, ny_val) in neighbours:
            if (nx_val, ny_val) in board.components:
                G.add_edge((x, y), (nx_val, ny_val))

    # Ramp / crossover / bit directed edges
    for (x, y), comp in board.components.items():
        comp_type = comp.component_type.value

        if comp_type in ("ramp_right",):
            dest = (x + 1, y + 1)
            if dest in board.components:
                G.add_edge((x, y), dest)
        elif comp_type in ("ramp_left",):
            dest = (x - 1, y + 1)
            if dest in board.components:
                G.add_edge((x, y), dest)
        elif comp_type in ("crossover",):
            for dx, dy in [(1, 1), (-1, 1)]:
                dest = (x + dx, y + dy)
                if dest in board.components:
                    G.add_edge((x, y), dest)
        elif comp_type in ("bit", "gear_bit"):
            # Bits can redirect in both directions depending on state
            for dx in (-1, 1):
                dest = (x + dx, y + 1)
                if dest in board.components:
                    G.add_edge((x, y), dest)

    return G


def _compute_horizontal_symmetry(
    components: List[Dict[str, Any]],
    width: int,
    height: int,
) -> float:
    """Compute horizontal reflection symmetry of component distribution.

    Mirrors the board left-right and compares the component-type
    distribution of the left half with the right half using cosine
    similarity of type-count vectors.

    Range [0, 1] where 1 = perfect symmetry.

    Parameters
    ----------
    components : List[Dict]
        Placed components with ``"x"``, ``"y"``, ``"type"`` keys.
    width : int
        Board width in columns.
    height : int
        Board height in rows.

    Returns
    -------
    float
    """
    mid = width / 2

    left_counts: Counter[str] = Counter()
    right_counts: Counter[str] = Counter()

    for comp in components:
        cx_val = comp.get("x", -1)
        ctype = comp.get("type", "unknown")
        if cx_val < mid:
            left_counts[ctype] += 1
        elif cx_val > mid:
            # Mirror to left for comparison
            right_counts[ctype] += 1
        # Components exactly on the midline are ignored (self-symmetric)

    # Cosine similarity between type-count vectors
    all_types = sorted(set(left_counts.keys()) | set(right_counts.keys()))
    v_left = np.array([left_counts.get(t, 0) for t in all_types], dtype=float)
    v_right = np.array([right_counts.get(t, 0) for t in all_types], dtype=float)

    norm = np.linalg.norm(v_left) * np.linalg.norm(v_right)
    if norm == 0:
        return 0.0

    return float(np.dot(v_left, v_right) / norm)


def _compute_vertical_symmetry(
    components: List[Dict[str, Any]],
    width: int,
    height: int,
) -> float:
    """Compute vertical reflection symmetry of component distribution.

    Mirrors the board top-bottom and compares type-count vectors via
    cosine similarity.

    Range [0, 1].

    Parameters
    ----------
    components : List[Dict]
    width : int
    height : int

    Returns
    -------
    float
    """
    mid = height / 2

    top_counts: Counter[str] = Counter()
    bottom_counts: Counter[str] = Counter()

    for comp in components:
        cy_val = comp.get("y", -1)
        ctype = comp.get("type", "unknown")
        if cy_val < mid:
            top_counts[ctype] += 1
        elif cy_val > mid:
            bottom_counts[ctype] += 1  # noqa: SIM114

    all_types = sorted(set(top_counts.keys()) | set(bottom_counts.keys()))
    v_top = np.array([top_counts.get(t, 0) for t in all_types], dtype=float)
    v_bottom = np.array([bottom_counts.get(t, 0) for t in all_types], dtype=float)

    norm = np.linalg.norm(v_top) * np.linalg.norm(v_bottom)
    if norm == 0:
        return 0.0

    return float(np.dot(v_top, v_bottom) / norm)


def _compute_branching_factor(
    board: tt_sim.Board,
    width: int,
    height: int,
) -> int:
    """Count cells where a marble has multiple valid downstream neighbours.

    A cell is a "branch point" if, from that cell, there are at least two
    occupied cells that are reachable according to component routing rules
    (ramp diagonals, crossover diagonals, bit dual-exit).

    A higher branching factor means more decision points in the marble
    path → more control-flow complexity in procedural reasoning.

    Parameters
    ----------
    board : tt_sim.Board
    width : int
    height : int

    Returns
    -------
    int
    """
    branch_points = 0
    for (x, y), comp in board.components.items():
        reachable = 0
        comp_type = comp.component_type.value

        if comp_type in ("ramp_right",):
            if (x + 1, y + 1) in board.components:
                reachable += 1
        elif comp_type in ("ramp_left",):
            if (x - 1, y + 1) in board.components:
                reachable += 1
        elif comp_type in ("crossover",):
            for dx in (-1, 1):
                if (x + dx, y + 1) in board.components:
                    reachable += 1
        elif comp_type in ("bit", "gear_bit"):
            # A bit has two possible exits depending on state
            exits = 0
            for dx in (-1, 1):
                if (x + dx, y + 1) in board.components:
                    exits += 1
            # Count as branch point if both exits are occupied (the bit
            # can actually go either way depending on state)
            if exits >= 2:
                reachable = exits

        if reachable >= 2:
            branch_points += 1

    return branch_points


# ============================================================================
# Run as script
# ============================================================================

if __name__ == "__main__":
    sys.exit(main())
