#!/usr/bin/env python3
"""Logprobs confidence analysis for Turing Tumble benchmark results.

Analyses token-level log probabilities captured by --capture-logprobs.
Produces per-task and aggregate confidence metrics, grouped by task type,
question type, and success/failure status.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TokenStats:
    """Per-token or per-sequence logprob statistics."""

    mean_logprob: float
    min_logprob: float
    max_logprob: float
    token_count: int
    perplexity_proxy: float  # exp(-mean_logprob), lower = more confident
    logprobs_raw: List[float] = field(default_factory=list)


@dataclass
class TaskLogprobSummary:
    """Logprob summary for a single benchmark task."""

    task_id: str
    task_type: str
    success: Optional[bool]
    question_type: str  # for understanding; empty for agentic
    token_stats: Optional[TokenStats] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _extract_logprobs_flat(lp_list: Optional[List[Dict]]) -> Optional[TokenStats]:
    """Extract stats from a flat list of per-token logprob dicts."""
    if not lp_list:
        return None
    logprobs = [item["logprob"] for item in lp_list if isinstance(item, dict) and "logprob" in item]
    if not logprobs:
        return None
    return TokenStats(
        mean_logprob=statistics.mean(logprobs),
        min_logprob=min(logprobs),
        max_logprob=max(logprobs),
        token_count=len(logprobs),
        perplexity_proxy=math.exp(-statistics.mean(logprobs)),
        logprobs_raw=logprobs,
    )


def _extract_logprobs_multiturn(
    turn_lps: Optional[List[Optional[List[Dict]]]],
) -> Optional[TokenStats]:
    """Extract stats from agentic multi-turn logprobs (list of per-turn lists)."""
    if not turn_lps:
        return None
    all_logprobs: List[float] = []
    for turn in turn_lps:
        if turn:
            for item in turn:
                if isinstance(item, dict) and "logprob" in item:
                    all_logprobs.append(item["logprob"])
    if not all_logprobs:
        return None
    return TokenStats(
        mean_logprob=statistics.mean(all_logprobs),
        min_logprob=min(all_logprobs),
        max_logprob=max(all_logprobs),
        token_count=len(all_logprobs),
        perplexity_proxy=math.exp(-statistics.mean(all_logprobs)),
        logprobs_raw=all_logprobs,
    )


def extract_task_logprobs(task: Dict[str, Any]) -> Optional[TaskLogprobSummary]:
    """Extract logprob stats from a single benchmark task result."""
    logprobs = task.get("logprobs")
    if logprobs is None:
        return None

    task_type = task.get("task_type", "unknown")
    task_id = task.get("task_id", "unknown")
    success = task.get("success")
    error = task.get("error")

    # Determine question type for understanding tasks
    q_type = ""
    if task_type == "understanding":
        predicted = task.get("predicted", {})
        q_type = predicted.get("question_type", "")

    stats: Optional[TokenStats] = None
    if task_type == "understanding":
        stats = _extract_logprobs_flat(logprobs)
    elif task_type == "agentic_synthesis":
        stats = _extract_logprobs_multiturn(logprobs)

    return TaskLogprobSummary(
        task_id=task_id,
        task_type=task_type,
        success=success,
        question_type=q_type,
        token_stats=stats,
        error=error,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _compute_aggregate_stats(
    summaries: List[TaskLogprobSummary],
) -> Dict[str, Any]:
    """Compute aggregate statistics across a group of task summaries."""
    valid = [s for s in summaries if s.token_stats is not None]
    if not valid:
        return {"count": len(summaries), "with_logprobs": 0}

    means = [s.token_stats.mean_logprob for s in valid]
    mins = [s.token_stats.min_logprob for s in valid]
    ppls = [s.token_stats.perplexity_proxy for s in valid]
    tcounts = [s.token_stats.token_count for s in valid]

    return {
        "count": len(summaries),
        "with_logprobs": len(valid),
        "mean_logprob_avg": statistics.mean(means),
        "mean_logprob_median": statistics.median(means),
        "mean_logprob_stdev": statistics.stdev(means) if len(means) > 1 else 0.0,
        "min_logprob_avg": statistics.mean(mins),
        "min_logprob_min": min(mins),
        "perplexity_avg": statistics.mean(ppls),
        "perplexity_median": statistics.median(ppls),
        "total_tokens_analysed": sum(tcounts),
        "avg_tokens_per_response": statistics.mean(tcounts),
    }


def _analyse_reasoning(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Extract reasoning/thinking metrics from agentic task transcripts.

    Analyses the ``assistant_text`` field in tool-call transcripts to
    measure reasoning verbosity, repetition, and quality signals.
    """
    reasoning_tasks: List[Dict[str, Any]] = []
    for r in results:
        if r is None:
            continue
        predicted = r.get("predicted", {}) or {}
        transcript = predicted.get("transcript", [])
        if not transcript:
            continue

        thinking_turns: List[Dict[str, Any]] = []
        total_chars = 0
        for t in transcript:
            atext = (t.get("assistant_text") or "").strip()
            if atext:
                thinking_turns.append({
                    "turn": t.get("turn", 0),
                    "tool": t.get("tool_name", ""),
                    "chars": len(atext),
                    "preview": atext[:200],
                    "full_text": atext,
                })
                total_chars += len(atext)

        if thinking_turns:
            char_counts = [tt["chars"] for tt in thinking_turns]
            reasoning_tasks.append({
                "task_id": r.get("task_id", ""),
                "success": r.get("success"),
                "turns_with_thinking": len(thinking_turns),
                "total_thinking_chars": total_chars,
                "avg_chars_per_turn": total_chars / len(thinking_turns),
                "max_chars_turn": max(char_counts),
                "min_chars_turn": min(char_counts),
                "thinking_turns": thinking_turns,
            })

    if not reasoning_tasks:
        return {"tasks_with_reasoning": 0}

    all_chars = [rt["total_thinking_chars"] for rt in reasoning_tasks]
    all_avg = [rt["avg_chars_per_turn"] for rt in reasoning_tasks]

    return {
        "tasks_with_reasoning": len(reasoning_tasks),
        "total_thinking_chars": sum(all_chars),
        "avg_thinking_per_task": statistics.mean(all_chars) if all_chars else 0,
        "avg_chars_per_turn": statistics.mean(all_avg) if all_avg else 0,
        "per_task": reasoning_tasks,
    }


def analyse(results_path: Path) -> Dict[str, Any]:
    """Run full logprobs analysis on a benchmark results JSON.

    Returns a structured dict suitable for printing or JSON serialisation.
    """
    with open(results_path) as f:
        report = json.load(f)

    tasks = report.get("results", [])
    summaries = []
    for task in tasks:
        s = extract_task_logprobs(task)
        if s is not None:
            summaries.append(s)

    if not summaries:
        return {
            "error": "No logprobs data found in results. "
                     "Re-run with --capture-logprobs.",
            "source": str(results_path),
        }

    # ── Global stats ──────────────────────────────────────────────────
    global_stats = _compute_aggregate_stats(summaries)

    # ── By task type ──────────────────────────────────────────────────
    by_type: Dict[str, Any] = {}
    for tt in ["understanding", "agentic_synthesis"]:
        group = [s for s in summaries if s.task_type == tt]
        if group:
            by_type[tt] = _compute_aggregate_stats(group)

    # ── Success vs failure ────────────────────────────────────────────
    by_success: Dict[str, Any] = {}
    for label, pred in [("successful", True), ("failed", False), ("unknown", None)]:
        group = [s for s in summaries if s.success is pred]
        if group:
            by_success[label] = _compute_aggregate_stats(group)

    # ── By question type (understanding only) ─────────────────────────
    by_question: Dict[str, Any] = {}
    understanding = [s for s in summaries if s.task_type == "understanding"]
    qtype_groups: Dict[str, List[TaskLogprobSummary]] = defaultdict(list)
    for s in understanding:
        qtype_groups[s.question_type or "unknown"].append(s)
    for qt, group in sorted(qtype_groups.items()):
        by_question[qt] = _compute_aggregate_stats(group)

    # ── Per-task details ──────────────────────────────────────────────
    task_details = []
    for s in summaries:
        detail = {
            "task_id": s.task_id,
            "task_type": s.task_type,
            "success": s.success,
            "question_type": s.question_type,
        }
        if s.token_stats:
            detail.update({
                "mean_logprob": round(s.token_stats.mean_logprob, 4),
                "min_logprob": round(s.token_stats.min_logprob, 4),
                "perplexity": round(s.token_stats.perplexity_proxy, 2),
                "tokens": s.token_stats.token_count,
            })
        if s.error:
            detail["error"] = s.error
        task_details.append(detail)

    # ── Confidence gap analysis ───────────────────────────────────────
    success_group = [s for s in summaries if s.success is True and s.token_stats]
    failure_group = [s for s in summaries if s.success is False and s.token_stats]
    confidence_gap = None
    if success_group and failure_group:
        success_mean = statistics.mean(s.token_stats.mean_logprob for s in success_group)
        failure_mean = statistics.mean(s.token_stats.mean_logprob for s in failure_group)
        confidence_gap = {
            "success_mean_logprob": round(success_mean, 4),
            "failure_mean_logprob": round(failure_mean, 4),
            "gap": round(success_mean - failure_mean, 4),
            "interpretation": (
                "Model is MORE confident on successful tasks"
                if success_mean > failure_mean
                else "Model is MORE confident on failed tasks (overconfidence)"
                if failure_mean > success_mean
                else "No confidence difference between success and failure"
            ),
        }

    # ── Reasoning analysis (agentic tasks with transcripts) ────────────
    reasoning_analysis = _analyse_reasoning(report.get("results", []))

    return {
        "source": str(results_path),
        "model": report.get("model", "unknown"),
        "provider": report.get("provider", "unknown"),
        "total_tasks_in_report": report.get("total_tasks", 0),
        "tasks_with_logprobs": len(summaries),
        "global": global_stats,
        "by_task_type": by_type,
        "by_success": by_success,
        "by_question_type": by_question,
        "confidence_gap": confidence_gap,
        "task_details": task_details,
        "reasoning": reasoning_analysis,
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def print_report(analysis: Dict[str, Any]) -> None:
    """Print a human-readable analysis report to stdout."""
    if "error" in analysis:
        print(f"\nERROR: {analysis['error']}")
        return

    print(f"\n{'=' * 70}")
    print(f"Logprobs Confidence Analysis")
    print(f"{'=' * 70}")
    print(f"Source:   {Path(analysis['source']).name}")
    print(f"Model:    {analysis['model']} ({analysis['provider']})")
    print(f"Tasks with logprobs: {analysis['tasks_with_logprobs']} / {analysis['total_tasks_in_report']}")

    g = analysis["global"]
    print(f"\n── Global Confidence ──")
    print(f"  Mean logprob:     {g['mean_logprob_avg']:.4f}  (avg across tokens)")
    print(f"  Median logprob:   {g['mean_logprob_median']:.4f}")
    print(f"  StDev logprob:    {g['mean_logprob_stdev']:.4f}")
    print(f"  Min logprob:      {g['min_logprob_min']:.4f}  (most uncertain token)")
    print(f"  Perplexity:       {g['perplexity_avg']:.2f}  (lower = more confident)")
    print(f"  Total tokens:     {g['total_tokens_analysed']}")
    print(f"  Avg tokens/resp:  {g['avg_tokens_per_response']:.1f}")

    if analysis["by_task_type"]:
        print(f"\n── By Task Type ──")
        for tt, stats in analysis["by_task_type"].items():
            print(f"  {tt:25s}  mean_logprob={stats['mean_logprob_avg']:.4f}  "
                  f"perplexity={stats['perplexity_avg']:.1f}  n={stats['with_logprobs']}")

    if analysis["by_success"]:
        print(f"\n── Success vs Failure ──")
        for label, stats in analysis["by_success"].items():
            print(f"  {label:12s}  mean_logprob={stats['mean_logprob_avg']:.4f}  "
                  f"perplexity={stats['perplexity_avg']:.1f}  n={stats['with_logprobs']}")

    if analysis.get("confidence_gap"):
        cg = analysis["confidence_gap"]
        print(f"\n── Confidence Gap ──")
        print(f"  Success mean logprob:  {cg['success_mean_logprob']:.4f}")
        print(f"  Failure mean logprob:  {cg['failure_mean_logprob']:.4f}")
        print(f"  Gap:                   {cg['gap']:.4f}")
        print(f"  → {cg['interpretation']}")

    if analysis["by_question_type"]:
        print(f"\n── By Question Type (understanding) ──")
        for qt, stats in sorted(analysis["by_question_type"].items()):
            print(f"  {qt:20s}  mean_logprob={stats['mean_logprob_avg']:.4f}  "
                  f"perplexity={stats['perplexity_avg']:.1f}  n={stats['with_logprobs']}")

    print(f"\n── Per-Task Confidence (first 20) ──")
    print(f"  {'Task':35s} {'Type':12s} {'OK':5s} {'mean_lp':>8s} {'min_lp':>8s} {'ppl':>6s} {'tokens':>6s}")
    print(f"  {'-'*35} {'-'*12} {'-'*5} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")
    for d in analysis["task_details"][:20]:
        ok = "✓" if d["success"] is True else "✗" if d["success"] is False else "?"
        mlp = d.get("mean_logprob", "N/A")
        minlp = d.get("min_logprob", "N/A")
        ppl = d.get("perplexity", "N/A")
        tok = d.get("tokens", "N/A")
        mlp_s = f"{mlp:.4f}" if isinstance(mlp, float) else str(mlp)
        minlp_s = f"{minlp:.4f}" if isinstance(minlp, float) else str(minlp)
        ppl_s = f"{ppl:.1f}" if isinstance(ppl, float) else str(ppl)
        tok_s = str(tok)
        print(f"  {d['task_id']:35s} {d['task_type']:12s} {ok:5s} {mlp_s:>8s} {minlp_s:>8s} {ppl_s:>6s} {tok_s:>6s}")

    print(f"\n{'=' * 70}")

    # ── Reasoning analysis (agentic) ──────────────────────────────────
    reasoning = analysis.get("reasoning", {})
    if reasoning and reasoning.get("tasks_with_reasoning", 0) > 0:
        print(f"\n── Reasoning Analysis (agentic thinking) ──")
        print(f"  Tasks with reasoning:  {reasoning['tasks_with_reasoning']}")
        print(f"  Total thinking chars:  {reasoning['total_thinking_chars']:,}")
        print(f"  Avg thinking / task:   {reasoning['avg_thinking_per_task']:,.0f} chars")
        print(f"  Avg chars / turn:      {reasoning['avg_chars_per_turn']:,.0f}")
        for pt in reasoning.get("per_task", []):
            print(f"\n  Task: {pt['task_id']}  {'✓' if pt['success'] else '✗'}")
            print(f"    Turns with thinking: {pt['turns_with_thinking']}")
            print(f"    Total thinking:      {pt['total_thinking_chars']:,} chars")
            print(f"    Avg / turn:          {pt['avg_chars_per_turn']:,.0f} chars")
            print(f"    Max turn:            {pt['max_chars_turn']:,} chars")
            for tt in pt.get("thinking_turns", []):
                print(f"      Turn {tt['turn']}: {tt['tool']:20s} {tt['chars']:>6,} chars  {tt['preview'][:80]}...")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_thinking(analysis: Dict[str, Any]) -> None:
    """Print full reasoning/thinking text from agentic transcripts."""
    reasoning = analysis.get("reasoning", {})
    if not reasoning or not reasoning.get("per_task"):
        print("\nNo reasoning data available.")
        return

    print(f"\n{'=' * 70}")
    print("Full Reasoning / Thinking Text")
    print(f"{'=' * 70}")
    for pt in reasoning["per_task"]:
        print(f"\n── {pt['task_id']} {'✓' if pt['success'] else '✗'} "
              f"({pt['total_thinking_chars']:,} total chars) ──")
        for tt in pt.get("thinking_turns", []):
            # Load full thinking from original results if available
            print(f"\n  [Turn {tt['turn']}] {tt['tool']} ({tt['chars']:,} chars)")
            print(f"  {'─' * 60}")
            print(f"  {tt.get('full_text', tt.get('preview', ''))}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Analyse logprobs confidence data from Turing Tumble benchmark results."
    )
    parser.add_argument(
        "results_file", type=Path,
        help="Path to a benchmark_results/*.json file generated with --capture-logprobs",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output analysis as JSON instead of a human-readable report",
    )
    parser.add_argument(
        "--save", type=Path, default=None,
        help="Save analysis JSON to this path",
    )
    parser.add_argument(
        "--show-thinking", action="store_true",
        help="Show full reasoning/thinking text from agentic task transcripts",
    )
    args = parser.parse_args()

    analysis = analyse(args.results_file)

    if args.json:
        print(json.dumps(analysis, indent=2, default=str))
    else:
        print_report(analysis)

    if args.show_thinking:
        _print_thinking(analysis)

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"\nAnalysis saved to {args.save}")


if __name__ == "__main__":
    main()
