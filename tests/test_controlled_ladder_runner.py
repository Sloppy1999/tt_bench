import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load("run_controlled_ladder")
smoke = load("smoke_controlled_ladder")
analysis = load("analyze_controlled_ladder")
prepare = load("prepare_controlled_study")


def test_one_puzzle_all_arms_and_resume(tmp_path, monkeypatch):
    monkeypatch.chdir(SCRIPTS.parent)
    smoke.smoke(tmp_path / "smoke")
    summary = json.loads((tmp_path / "smoke/smoke_summary.json").read_text())
    assert summary["resume_verified"]
    search = next(r for r in summary["episodes"] if r["arm"] == "A4_best_of_4")
    assert search["selected_candidate"] == 0
    assert search["candidate_count"] == 4
    baseline = json.loads(next((tmp_path / "smoke/A0_baseline").rglob("candidate_0.json")).read_text())
    requests = Path(baseline["requests_path"]).read_text()
    assert '"request"' in requests and '"response"' in requests
    assert 'Authorization' not in requests
    analysis.analyze(tmp_path / "smoke", tmp_path / "analysis")
    assert (tmp_path / "analysis/episodes.csv").exists()


def test_analysis_refuses_missing_pairs(tmp_path):
    root = tmp_path / "input"
    for i, (arm, task) in enumerate([("A0_baseline", "a"), ("A1_step_sim", "b")]):
        directory = root / str(i)
        directory.mkdir(parents=True)
        (directory / "episode.json").write_text(json.dumps({
            "model": "test", "tier": 1, "arm": arm, "seed": 1,
            "task_id": task, "success": False,
        }))
    (root / "smoke_summary.json").write_text('{"synthetic": true}')
    with pytest.raises(ValueError, match="Incomplete paired roster"):
        analysis.analyze(root, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_puzzle_cluster_bootstrap_keeps_seeds_together():
    # Each puzzle has opposite outcomes across seeds: every puzzle mean is .5.
    matrix = np.array([[0, 1], [1, 0], [0, 1]])
    assert analysis.bootstrap(matrix, np.random.default_rng(0)) == [.5, .5]


def test_randomization_and_holm_keep_puzzle_blocks():
    matrix = np.tile([[1, -1]], (30, 1))
    assert analysis.paired_randomization(matrix, np.random.default_rng(0)) == 1.0
    assert analysis.paired_randomization(matrix[:2], np.random.default_rng(0)) is None
    assert analysis.holm([.01, .04, .03, None]) == [.03, .06, .06, None]


def test_write_once_does_not_replace_existing_artifacts(tmp_path):
    path = tmp_path / "result.json"
    runner.write_once(path, {"value": 1})
    with pytest.raises(FileExistsError):
        runner.write_once(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 1}


def test_gpu_accounting_includes_whole_node_cost(tmp_path):
    accounting = tmp_path / "accounting.txt"
    accounting.write_text("JobIDRaw|State|ElapsedRaw|AllocTRES|ExitCode\n"
                          "42|COMPLETED|3600|cpu=128,gres/gpu=4|0:0\n")
    frame = pd.DataFrame({"slurm_job_ids": [["42"], ["42"]], "wall_seconds": [1.0, 3.0]})
    jobs = analysis.attach_accounting(frame, accounting)
    assert list(frame.gpu_hours) == [1.0, 3.0]
    assert jobs.allocated_gpu_hours.sum() == 4.0


def test_parallel_shards_pair_and_refuse_changed_resume(tmp_path, monkeypatch):
    monkeypatch.chdir(SCRIPTS.parent)
    task = json.loads(Path("data/tasks/challenges_1comp/tt-official-ch01-1comp.json").read_text())
    paths = []
    for index in range(3):
        path = tmp_path / f"fixture_{index}.json"
        path.write_text(json.dumps({**task, "task_id": path.stem}))
        paths.append(str(path))
    config = {"model": "scripted", "model_revision": "fixture", "tier": 1,
              "seed": 1001, "temperature": 0.7, "max_turns": 3, "max_tokens": 128,
              "timeout": 10, "experiment_date": "fixture", "hardware_class": "mock",
              "quantization": "none", "serving_config": [], "workers": 2}
    filename = tmp_path / "config.json"
    output = tmp_path / "results"
    monkeypatch.setattr(runner.requests, "post", smoke.transport)
    for shard, tasks in enumerate((paths[:2], paths[2:])):
        config.update(shard_id=f"shard_{shard}", tasks=tasks)
        filename.write_text(json.dumps(config))
        for arm in ("A0_baseline", "A1_step_sim"):
            monkeypatch.setattr(sys, "argv", ["run_controlled_ladder.py", "--config", str(filename),
                "--arm", arm, "--output-root", str(output)])
            runner.main()
    assert len(list(output.rglob("episode.json"))) == 6
    analysis.analyze(output, tmp_path / "report", expected_arms=["A0_baseline", "A1_step_sim"])
    config["temperature"] = 0.8
    filename.write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="different code/config/tasks"):
        runner.main()


def test_study_freeze_preserves_scaled_set_glob(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    directory = Path("data/tasks/challenges_1comp")
    directory.mkdir(parents=True)
    rows = []
    for name, tier in [("tt-official-ch01", 1), ("tt-official-ch12", 2), ("tt-official-12", 2)]:
        path = directory / f"{name}.json"
        path.write_text(json.dumps({"task_id": name, "tier": tier,
                                   "board": {"width": 11, "height": 11}}))
        rows.append({"path": str(path.relative_to("data/tasks")), "task_id": name,
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "status": "positive_candidate", "reasons": []})
    audit = Path("audit.json")
    audit.write_text(json.dumps({"source_sha256": {}, "files": rows, "scope": "test fixture"}))
    models = Path("models.json")
    models.write_text(json.dumps([{"model": f"fixture_{i}", "revision": "test",
        "native_context": 131072, "tensor_parallel_size": 1, "quantization": None,
        "tool_parser": "hermes"} for i in range(7)]))
    prepare.prepare(audit, models, Path("frozen"))
    eligible = json.loads(Path("frozen/eligible_tasks.json").read_text())
    assert {r["task_id"] for r in eligible} == {"tt-official-ch01", "tt-official-ch12"}
    design = json.loads(Path("frozen/design.json").read_text())
    assert design["episodes"] == 7 * 2 * 3 * 5
    assert design["candidate_attempts"] == 7 * 2 * 3 * 8
