"""Opt-in controlled harness interventions; no reference solutions in feedback.

``valid`` denotes observable physical legality, not official task success.
Candidate ranking deliberately cannot access TaskResult.success or target labels.
"""
from __future__ import annotations

from copy import deepcopy
import inspect
import re

from tt_bench.simulator import Component, ComponentType
from tt_bench.tools.executor import TuringTumbleToolExecutor

ARMS = (
    "A0_baseline", "A1_step_sim", "A2_structured_feedback",
    "A3_legality_masking", "A4_best_of_4", "A5_adaptive_search",
)


def structured_feedback(result, inventory, delta=None):
    """Translate existing simulator observations, without inspecting task targets."""
    violations = []
    for error in result.get("free_fall_errors", []):
        cell = re.search(r"\((-?\d+),\s*(-?\d+)\)", error)
        marble = re.search(r"marble (\d+)", error)
        violations.append({
            "type": "unsupported_marble",
            "entity_id": f"marble_{marble[1]}" if marble else None,
            "location": [int(cell[1]), int(cell[2])] if cell else None,
            "cause": error,
        })
    for trace in result.get("execution_traces", []):
        reason = trace.get("termination_reason")
        if not trace.get("final_destination") and reason not in (
            "no_blue_balls", "no_red_balls",
        ):
            violations.append({
                "type": "uncaught_marble", "entity_id": f"marble_{trace['marble']}",
                "location": list(trace["path"][-1]) if trace.get("path") else None,
                "cause": reason or "No catcher reached",
                "affected_count": trace.get("_path_group_size", 1),
            })
    if not result.get("success"):
        violations.append({"type": "simulation_error", "entity_id": None,
                           "location": None, "cause": result.get("error", "Unknown error")})
    caught = sum(result.get(k, 0) for k in ("left_catcher", "right_catcher", "interceptor"))
    exhausted_releases = sum(
        trace.get("_path_group_size", 1) for trace in result.get("execution_traces", [])
        if trace.get("termination_reason") in ("no_blue_balls", "no_red_balls")
    )
    released = result.get("total_marbles", 0) - exhausted_releases
    # No target count is inferred from hidden labels. The observed release count
    # is a routing denominator, not the required output count.
    return {
        "schema_version": 1,
        "valid": bool(result.get("success")) and not violations,
        "validity_scope": "observable_physical_legality",
        "violations": violations,
        "progress": {"caught": caught, "released": released,
                     "target": None,
                     "left_catcher": result.get("left_catcher", 0),
                     "right_catcher": result.get("right_catcher", 0),
                     "interceptor": result.get("interceptor", 0)},
        "inventory_remaining": inventory,
        "state_delta": delta or {},
        "recovery_hint": (
            f"Inspect the observed violation near {violations[0]['location']}."
            if violations else "Compare the observed output with the stated puzzle objective."
        ),
    }


def candidate_score(feedback):
    """Predefined label-free ranking: legal, caught fraction, caught, fewer violations.

    The earliest candidate wins ties. No claim of functional success follows
    from this score; official evaluation remains a separate measurement.
    """
    progress = feedback["progress"]
    return (int(feedback["valid"]),
            progress["caught"] / max(1, progress["released"]),
            progress["caught"], -len(feedback["violations"]))


def select_candidate(feedbacks):
    if not feedbacks:
        raise ValueError("No candidates")
    return max(range(len(feedbacks)), key=lambda i: candidate_score(feedbacks[i]))


class InterventionExecutor(TuringTumbleToolExecutor):
    """Instrument all arms; mutate behavior only with explicit intervention flags."""

    def __init__(self, *args, structured=False, legality=False, **kwargs):
        self.inventory_declared = kwargs.get("available_parts") is not None
        super().__init__(*args, **kwargs)
        self.structured = structured
        self.legality = legality
        self.events = []
        self.simulations = []
        self._delta = {}

    def inventory_remaining(self):
        return {k: v - self._used_parts.get(k, 0) for k, v in self.available_parts.items()}

    def run_simulation(self, input_sequence=None):
        result = super().run_simulation(input_sequence)
        self.simulations.append({"input_sequence": deepcopy(input_sequence),
                                 "output": deepcopy(result)})
        return result

    def _auto_simulation_payload(self):
        payload = super()._auto_simulation_payload()
        if self.structured:
            # Preserve A1 observations and its prompt contract. Only add the
            # structured representation; do not perform a second simulation.
            payload.update(structured_feedback(
                self.simulations[-1]["output"], self.inventory_remaining(), self._delta))
        return payload

    def _preflight(self, name, arguments):
        method = getattr(super(), name, None)
        if name not in ("place_component", "remove_component", "run_simulation", "get_board_state"):
            return "Unknown tool"
        try:
            inspect.signature(method).bind(**arguments)
        except (TypeError, ValueError) as exc:
            return str(exc)
        if name not in ("place_component", "remove_component"):
            return None
        x, y = arguments["x"], arguments["y"]
        if type(x) is not int or type(y) is not int:
            return "Coordinates must be integers"
        if not (0 <= x < self.board.cols and 0 <= y < self.board.rows):
            return "Coordinates out of bounds"
        if name == "remove_component":
            if (x, y) in self._fixed_positions:
                return "Cannot remove fixed component"
            return None if (x, y) in self.board.components else "Cell is empty"
        kind = arguments["component_type"]
        if not isinstance(kind, str) or kind not in {c.value for c in ComponentType}:
            return "Unavailable component type"
        state = arguments.get("state", 0)
        if type(state) is not int or state not in (0, 1):
            return "State must be integer 0 or 1"
        if kind not in ("bit", "gear_bit") and state != 0:
            return "Only bit and gear_bit accept a nonzero state; ramps encode orientation in type"
        existing = self.board.components.get((x, y))
        if existing is not None:
            if ((x,y) in self.board.editable_bit_states and kind in ('bit','gear_bit')
                    and existing.component_type.value == kind):
                return None
            return 'Cell is occupied'
        from tt_bench.simulator.inventory import inventory_key
        pool = inventory_key(kind, self.available_parts)
        if self.inventory_declared and self.inventory_remaining().get(pool, 0) <= 0:
            return "Inventory exhausted or component type unavailable"
        try:
            Component.from_dict({"type": kind, "x": x, "y": y, "state": state})
        except (TypeError, ValueError) as exc:
            return str(exc)
        return None

    def execute(self, tool_name, arguments):
        before = len(self.simulations)
        self._delta = {"operation": tool_name, "arguments": deepcopy(arguments)}
        rejection = self._preflight(tool_name, arguments) if self.legality else None
        if rejection:
            result = {"success": False, "error": rejection, "rejected_before_execution": True}
        else:
            try:
                result = super().execute(tool_name, arguments)
            except Exception as exc:
                self.events.append({"tool": tool_name, "arguments": deepcopy(arguments),
                                    "exception": str(exc), "invalid": True})
                raise
        self.events.append({"tool": tool_name, "arguments": deepcopy(arguments),
                            "result": deepcopy(result),
                            "invalid": bool(result.get("error")) or result.get("success") is False,
                            "simulator_calls": len(self.simulations) - before})
        return result
