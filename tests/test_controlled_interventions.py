import pytest

from tt_bench.simulator import Board
from tt_bench.tools.interventions import (
    InterventionExecutor, structured_feedback, select_candidate,
)


def executor(**kwargs):
    return InterventionExecutor(
        Board(rows=2, cols=5, blue_hopper_x=2, blue_hopper_count=1),
        target_sequence=["blue"], **kwargs,
    )


def test_auto_simulation_after_success_only_and_non_destructive():
    e = executor(auto_simulate=True)
    for name, args in [
        ("place_component", {"component_type": "ramp_right", "x": 2, "y": 0}),
        ("remove_component", {"x": 2, "y": 0}),
    ]:
        assert "simulation" in e.execute(name, args)
        assert e.board.blue_balls_remaining == 1
    assert len(e.simulations) == 2
    assert not e.execute("remove_component", {"x": 2, "y": 0})["success"]
    assert len(e.simulations) == 2


def test_baseline_instrumentation_does_not_add_simulation():
    e = executor()
    assert "simulation" not in e.execute(
        "place_component", {"component_type": "ramp_right", "x": 2, "y": 0})
    assert not e.simulations


def test_structured_feedback_has_observed_coordinates_and_no_target():
    result = structured_feedback({
        "success": True, "free_fall_errors": ["marble 3 traversed empty cell (8, 12)"],
        "total_marbles": 8, "left_catcher": 5,
    }, {"ramp_left": 1})
    assert not result["valid"]
    assert result["violations"][0]["location"] == [8, 12]
    assert result["violations"][0]["entity_id"] == "marble_3"
    assert result["progress"] == {
        "caught": 5, "released": 8, "target": None,
        "left_catcher": 5, "right_catcher": 0, "interceptor": 0,
    }


@pytest.mark.parametrize("change", [
    {"x": 1.5}, {"x": True}, {"x": -1}, {"y": 8},
    {"state": 2}, {"state": True}, {"state": 1},
    {"component_type": "unknown"}, {"orientation": "up"},
])
def test_preflight_rejections_leave_inventory_and_board_unchanged(change):
    e = executor(legality=True, available_parts={"ramp_right": 1})
    args = {"component_type": "ramp_right", "x": 2, "y": 0, **change}
    assert e.execute("place_component", args)["rejected_before_execution"]
    assert e.inventory_remaining() == {"ramp_right": 1}
    assert not e.placed_components
    assert not e.board.components


def test_preflight_occupied_fixed_and_exhausted_inventory():
    e = executor(legality=True, available_parts={"ramp_right": 1})
    assert e.execute("place_component", {"component_type": "ramp_right", "x": 2, "y": 0})["success"]
    for x in (2, 3):
        assert e.execute("place_component", {
            "component_type": "ramp_right", "x": x, "y": 0})["rejected_before_execution"]
    e._fixed_positions.add((2, 0))
    assert e.execute("remove_component", {"x": 2, "y": 0})["rejected_before_execution"]


def test_candidate_selection_ignores_official_success_and_breaks_ties_by_index():
    good = structured_feedback({"success": True, "left_catcher": 1, "total_marbles": 1}, {})
    bad = structured_feedback({"success": True, "left_catcher": 2, "total_marbles": 2,
                               "free_fall_errors": ["marble 1 traversed empty cell (2, 1)"]}, {})
    bad["official_success"] = True
    good["official_success"] = False
    assert select_candidate([bad, good, good]) == 1


def test_structured_automatic_feedback_contains_delta():
    e = executor(auto_simulate=True, structured=True)
    result = e.execute("place_component", {"component_type": "ramp_right", "x": 2, "y": 0})
    assert result["simulation"]["state_delta"]["operation"] == "place_component"
    assert len(e.simulations) == 1


def test_exhausted_hopper_events_do_not_reduce_observed_progress():
    feedback = structured_feedback({
        "success": True, "total_marbles": 15, "left_catcher": 8,
        "execution_traces": [
            {"marble": 1, "final_destination": "left_catcher", "_path_group_size": 8},
            {"marble": 9, "termination_reason": "no_blue_balls", "_path_group_size": 7},
        ],
    }, {})
    assert feedback["valid"]
    assert feedback["progress"]["released"] == feedback["progress"]["caught"] == 8
