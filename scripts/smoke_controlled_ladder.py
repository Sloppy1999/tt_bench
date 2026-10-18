#!/usr/bin/env python3
"""One-puzzle, scripted-transport smoke test of every arm; no model inference."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from run_controlled_ladder import episode, write_once
from tt_bench.tools.interventions import ARMS


class Response:
    status_code = 200
    text = "scripted smoke response"

    def __init__(self, message):
        self.message = message

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": self.message, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


def transport(url, **kwargs):
    messages = kwargs["json"]["messages"]
    if any(m["role"] == "tool" for m in messages):
        return Response({"role": "assistant", "content": "Finished scripted attempt."})
    return Response({"role": "assistant", "content": "Scripted fixture placement.",
                     "tool_calls": [{"id": "smoke_call", "type": "function", "function": {
                         "name": "place_component", "arguments": json.dumps({
                             "component_type": "ramp_left", "x": 3, "y": 9})}}]})


def smoke(output):
    output.mkdir(parents=True, exist_ok=False)
    task = Path("data/tasks/challenges_1comp/tt-official-ch01-1comp.json")
    config = {"model": "SCRIPTED_SMOKE_NOT_A_MODEL", "tier": 1, "seed": 1001,
              "temperature": 0.7, "max_tokens": 128, "max_turns": 3, "timeout": 10}
    summaries = []
    with patch("requests.post", transport):
        for arm in ARMS:
            arm_output = output / arm
            arm_output.mkdir()
            result = episode(config, task, arm, arm_output)
            write_once(arm_output / task.stem / "episode.json", result)
            # Resuming reads cached attempts; must issue no HTTP requests.
            with patch("requests.post", side_effect=AssertionError("resume made a request")):
                resumed = episode(config, task, arm, arm_output)
            assert resumed == result
            assert result["candidate_count"] == (4 if arm == "A4_best_of_4" else 1)
            assert result["interaction_simulator_calls"] == (0 if arm == "A0_baseline" else result["candidate_count"])
            summaries.append(result)
    write_once(output / "smoke_summary.json", {
        "synthetic": True, "model_inference": False, "resume_verified": True,
        "episodes": summaries})
    print(f"Scripted smoke passed for {len(summaries)} arms: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    smoke(parser.parse_args().output)
