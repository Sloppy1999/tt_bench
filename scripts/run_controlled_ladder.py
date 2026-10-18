#!/usr/bin/env python3
"""Run one explicitly configured, resumable cell against an existing vLLM server.

This does not submit jobs. A fresh client/executor is created for each attempt.
Never use an output directory with changed config, code, or task contents.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from functools import partial
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid

import requests

from tt_bench.benchmark.runner import TuringTumbleBenchmark
from tt_bench.llm.client import LLMConfig, create_llm_client
from tt_bench.tools.executor import create_executor_from_task
from tt_bench.tools.interventions import (
    ARMS, InterventionExecutor, select_candidate, structured_feedback,
)
from tt_bench.simulator.inventory import used_inventory
from tt_bench.simulator.targets import validate_targets


class MeasuredBenchmark(TuringTumbleBenchmark):
    scoring_simulator_calls = 0

    def _validate_against_expected(self, board, expected, task_info):
        self.scoring_simulator_calls += 1
        return super()._validate_against_expected(board, expected, task_info)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_once(path, value):
    # Atomic publication; never replace prior completed records.
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def evaluate_candidate(benchmark, info, placements):
    """Separate observable ranking inputs from official evaluation."""
    normalized = benchmark._normalize_placements(placements)
    inventory = info.get("available_parts", {})
    # The scorer's rule: shared ramp pool, editable bits free, every editable bit set.
    inventory_ok, _ = benchmark._validate_available_parts(info, normalized)
    used = used_inventory(normalized, inventory, info["board"])
    validator = create_executor_from_task(
        info["board"], info["board"].get("fixed_components", []),
        available_parts=inventory, executor_type=partial(InterventionExecutor, legality=True))
    structural = []
    for index, placement in enumerate(placements):
        check = validator.execute("place_component", {
            "component_type": placement.get("component_type", placement.get("type")),
            "x": placement.get("x"), "y": placement.get("y"), "state": placement.get("state", 0)})
        if check.get("success") is False:
            structural.append({"type": "component_constraint", "entity_id": f"component_{index}",
                               "location": [placement.get("x"), placement.get("y")],
                               "cause": check.get("error")})
    try:
        board = benchmark._build_board(info, placements=normalized)
    except Exception as exc:
        raw = {"success": False, "error": f"Cannot construct candidate board: {exc}"}
        return {"feedback": structured_feedback(raw, {}), "simulator_output": raw,
                "inventory_compliance": inventory_ok, "physical_legality": False,
                "functional_correctness": None, "functional_reason": "Board construction failed",
                "simulator_calls": 0}
    verifier = InterventionExecutor(board)
    sequence = benchmark._normalize_input_sequence(info.get("input_sequence", ["blue"]))
    raw = verifier.run_simulation(sequence)
    feedback = structured_feedback(raw, {k: v - used[k] for k, v in inventory.items()})
    feedback["violations"].extend(structural)
    if structural:
        feedback["valid"] = False
    if not inventory_ok:
        feedback["valid"] = False
        feedback["violations"].append({"type": "inventory", "entity_id": None,
                                       "location": None, "cause": "Declared inventory exceeded"})
    # Official target comparison is recorded, never passed to select_candidate.
    results = board.run(sequence)
    functional, reason = validate_targets(info, board, results)
    return {"feedback": feedback, "simulator_output": raw,
            "inventory_compliance": inventory_ok,
            "physical_legality": not any(v["type"] != "inventory" for v in feedback["violations"]),
            "functional_correctness": functional, "functional_reason": reason, "simulator_calls": 2}


def attempt(config, task, arm, candidate, output):
    captured = []
    transport_failures = []
    meter = {"tokens": 0, "steps": 0, "malformed_tool_calls": 0, "malformed_placements": 0}
    level = ARMS.index(arm)

    def factory(*args, **kwargs):
        kwargs["executor_type"] = partial(
            InterventionExecutor, structured=level >= 2, legality=level >= 3)
        executor = create_executor_from_task(*args, **kwargs)
        captured.append(executor)
        return executor

    # Candidate zero is paired exactly across arms. Further seeds are recorded.
    seed = (config["seed"] + candidate * 1000003) % (2**31)
    client = create_llm_client(LLMConfig(
        provider="vllm", model=config["model"], temperature=config["temperature"],
        max_tokens=config["max_tokens"], seed=seed,
        base_url=os.environ.get("TTBENCH_BASE_URL", "http://localhost:8000"),
        api_key="none", timeout=config["timeout"],
    ))
    request_path = output / task.stem / f"candidate_{candidate}_{uuid.uuid4().hex}.requests.jsonl"

    def record_post(url, **kwargs):
        # Preserve incomplete attempts too. Authentication headers are never saved.
        request_id = uuid.uuid4().hex
        def append(record):
            with request_path.open("a") as handle:
                handle.write(json.dumps({"request_id": request_id, "time": time.time(),
                                         "slurm_job_id": os.environ.get("SLURM_JOB_ID"), **record}) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        append({"request": deepcopy(kwargs.get("json"))})
        try:
            response = requests.post(url, **kwargs)
        except Exception as exc:
            transport_failures.append({"exception": str(exc)})
            append({"exception": str(exc)})
            raise
        try:
            body = response.json()
        except ValueError:
            body = response.text
        append({"status_code": response.status_code, "response": body})
        if response.status_code >= 400:
            transport_failures.append({"status_code": response.status_code})
        elif isinstance(body, dict):
            usage = body.get("usage") or {}
            meter["tokens"] += usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            choices = body.get("choices") or []
            meter["steps"] += int(bool(choices))
            for choice in choices:
                for call in choice.get("message", {}).get("tool_calls") or []:
                    name = None
                    try:
                        name = call["function"]["name"]
                        json.loads(call["function"]["arguments"])
                    except (KeyError, ValueError, TypeError):
                        meter["malformed_tool_calls"] += 1
                        meter["malformed_placements"] += int(name == "place_component")
        return response

    client.http_post = record_post
    benchmark = MeasuredBenchmark(
        client, task.parent, output, max_turns=config["max_turns"],
        max_tokens=config["max_tokens"], auto_simulate=level >= 1,
        prompt_variant="urged" if level == 0 else "autosim", executor_factory=factory,
    )
    result = benchmark.run_agentic_task(task)
    if not captured:
        raise RuntimeError(f"Executor not created: {result.error}")
    executor = captured[0]
    info, _ = benchmark.load_task(task)
    final = executor.get_placed_components()
    evaluation = evaluate_candidate(benchmark, info, final)
    credited = evaluate_candidate(benchmark, info, result.predicted.get("final_solution", []))
    return {"candidate": candidate, "seed": seed, "result": asdict(result),
            "final_placements": final, "selection_evaluation": evaluation,
            "credited_evaluation": credited, "events": executor.events,
            "simulations": executor.simulations,
            "requests_path": str(request_path),
            "transport_failures": transport_failures,
            "meter": meter, "scoring_simulator_calls": benchmark.scoring_simulator_calls,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID")}


def episode(config, task, arm, output):
    candidates = []
    directory = output / task.stem
    directory.mkdir(exist_ok=True)
    for index in range(4 if arm in ARMS[4:] else 1):
        path = directory / f"candidate_{index}.json"
        if path.exists():
            value = json.loads(path.read_text())
        else:
            value = attempt(config, task, arm, index, output)
            write_once(path, value)
        candidates.append(value)
        first = value["selection_evaluation"]["feedback"]
        # Fixed adaptive rule: stop after candidate zero only when all observed
        # released marbles are routed legally. No official success predicate.
        if arm == "A5_adaptive_search" and index == 0 and (
            first["valid"] and first["progress"]["released"] > 0
            and first["progress"]["caught"] == first["progress"]["released"]
        ):
            break
    selected = select_candidate([c["selection_evaluation"]["feedback"] for c in candidates])
    chosen = candidates[selected]
    result = chosen["result"]
    search = arm in ARMS[4:]
    evaluation = chosen["selection_evaluation" if search else "credited_evaluation"]
    # Search submits the selected FINAL board, not the baseline's oracle-backed
    # best-board fallback. Keep both records for interpreting this difference.
    success = (evaluation["inventory_compliance"] and evaluation["physical_legality"]
               and evaluation["functional_correctness"]) if search else result["success"]
    events = [e for c in candidates for e in c["events"]]
    simulations = [s for c in candidates for s in c["simulations"]]
    malformed = sum(c["meter"]["malformed_tool_calls"] for c in candidates)
    malformed_placements = sum(c["meter"]["malformed_placements"] for c in candidates)
    invalid = sum(e["invalid"] for e in events) + malformed
    placements = [e for e in events if e["tool"] == "place_component"]
    removals = [e for e in events if e["tool"] == "remove_component"]
    failed_first = bool(simulations) and not structured_feedback(simulations[0]["output"], {})["valid"]
    infrastructure = any(c["transport_failures"] for c in candidates)
    scoring_calls = sum(c["scoring_simulator_calls"] for c in candidates)
    posthoc_calls = sum(c[k]["simulator_calls"] for c in candidates
                        for k in ("selection_evaluation", "credited_evaluation"))
    failure = ("none" if success else "infrastructure_review" if infrastructure
               else "no_tool_calls" if not events
               else "inventory" if not evaluation["inventory_compliance"]
               else "physical" if not evaluation["physical_legality"] else "functional")
    return {"task_id": result["task_id"], "model": config["model"], "tier": config["tier"],
            "arm": arm, "seed": config["seed"], "success": success,
            "selected_candidate": selected, "candidate_count": len(candidates),
            "inventory_compliance": evaluation["inventory_compliance"],
            "physical_legality": evaluation["physical_legality"],
            "functional_correctness": evaluation["functional_correctness"],
            "failure_category": failure, "invalid_tool_call_rate": invalid / max(1, len(events) + malformed),
            "invalid_placement_rate": (sum(e["invalid"] for e in placements) + malformed_placements)
                / max(1, len(placements) + malformed_placements),
            "simulator_calls": len(simulations) + scoring_calls + posthoc_calls,
            "interaction_simulator_calls": len(simulations), "scoring_simulator_calls": scoring_calls,
            "posthoc_verifier_calls": posthoc_calls,
            "placements": sum(not e["invalid"] for e in placements),
            "removals": sum(not e["invalid"] for e in removals),
            "rollback_count": 0,  # no transaction rollback operation in this harness
            "steps": sum(c["meter"]["steps"] for c in candidates),
            "tokens": sum(c["meter"]["tokens"] for c in candidates),
            "wall_seconds": sum(c["result"]["latency_ms"] for c in candidates) / 1000,
            "gpu_hours": None,  # allocated GPU-hours require Slurm accounting
            "first_simulation_failed": failed_first if simulations else None,
            "success_after_first_failed_simulation": success if failed_first else None,
            "credit_route": "selected_final_board" if search else result["metrics"].get("credit_route"),
            "slurm_job_ids": sorted({c["slurm_job_id"] for c in candidates if c["slurm_job_id"]}),
            "infrastructure_review_required": infrastructure}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    required = {"model", "model_revision", "tier", "seed", "temperature", "max_turns",
                "max_tokens", "timeout", "tasks", "experiment_date", "hardware_class",
                "quantization", "serving_config"}
    if required - config.keys():
        parser.error(f"Missing explicit settings: {sorted(required - config.keys())}")
    if any("key" in k.lower() or "token" in k.lower() and k != "max_tokens" for k in config):
        parser.error("Do not store credentials in experiment configs")
    tasks = [Path(p) for p in config["tasks"]]
    if not tasks or len({t.stem for t in tasks}) != len(tasks):
        parser.error("Task roster must be nonempty with unique IDs")
    if any(json.loads(t.read_text()).get("tier", 1) != config["tier"] for t in tasks):
        parser.error("Task tier differs from configured tier")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
               for root in (Path("src"), Path("scripts")) for p in sorted(root.rglob("*.py"))}
    for path in (Path("pyproject.toml"), Path("uv.lock"), Path("jureca/controlled_ladder.sbatch")):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = {"config": config, "commit": commit, "sources": sources,
                "tasks": {str(t): hashlib.sha256(t.read_bytes()).hexdigest() for t in tasks}}
    model_name = config["model"].replace("/", "--")
    output = (args.output_root / f"{config['experiment_date']}_{commit[:12]}"
              / model_name / f"tier_{config['tier']}" / args.arm / f"seed_{config['seed']}")
    if config.get("shard_id"):
        shard = config["shard_id"]
        if not isinstance(shard, str) or not shard.replace("_", "").isalnum():
            parser.error("shard_id must contain only letters, digits and underscores")
        output /= shard
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = output / "manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text())["identity"] != identity:
                raise RuntimeError("Existing shard has different code/config/tasks; use a new namespace")
        else:
            write_once(manifest, {"identity": identity, "identity_hash": digest(identity),
                "arm": args.arm, "python": sys.version, "platform": platform.platform(),
                "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
                "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
        def run_task(task):
            path = output / task.stem / "episode.json"
            if not path.exists():
                write_once(path, episode(config, task, args.arm, output))
                return 1
            return 0
        started = time.time()
        completed = []
        status = "failed"
        try:
            with ThreadPoolExecutor(max_workers=config.get("workers", 1)) as pool:
                # Consume every result: infrastructure exceptions must fail the job.
                completed = list(pool.map(run_task, tasks))
            status = "complete"
        finally:
            write_once(output / f"invocation_{uuid.uuid4().hex}.json", {
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "started": started,
                "wall_seconds": time.time() - started, "status": status,
                "new_episodes": sum(completed), "expected_episodes": len(tasks),
                "workers": config.get("workers", 1)})
    print(output)


if __name__ == "__main__":
    main()
