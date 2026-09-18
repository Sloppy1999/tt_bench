"""All three validation entry points must enforce the same explicit targets."""
import json

import pytest

from tt_bench.benchmark.runner import TuringTumbleBenchmark
from tt_bench.simulator import Board, verify_task
from tt_bench.tools.executor import create_executor_from_task


def task():
    return {
        "task_id": "tiny", "input_sequence": ["blue"],
        "board": {"width": 5, "height": 1, "hopper_entry_mode": "column",
                  "fixed_components": [],
                  "ball_hoppers": {"blue": {"x": 2, "count": 1}, "red": {"x": 4, "count": 0}},
                  "trigger_levers": {"left": {"x": 2}, "right": {"x": 4}}},
        "solution": {"placed_components": [], "final_marble_state": ["blue"]},
        "required_output": ["blue"],
        "expected_output": {"left_catcher": 1, "right_catcher": 0, "intercepted": 0},
    }


def verdicts(t):
    runner = TuringTumbleBenchmark.__new__(TuringTumbleBenchmark)
    board = Board.from_task_dict(t)
    scored, _ = runner._validate_simulation_results(board, t, board.run(t['input_sequence']))
    executor = create_executor_from_task(
        t['board'], t['board']['fixed_components'],
        target_sequence=t['input_sequence'],
        target_final_state=t.get('solution', {}).get('final_marble_state'),
        required_output=t.get('required_output'), expected_output=t.get('expected_output'),
    )
    executor.run_simulation(t['input_sequence'])
    return verify_task(t), scored, executor.is_solution_found()


def test_agreeing_targets_pass_all_entry_points():
    assert verdicts(task()) == (True, True, True)


@pytest.mark.parametrize('field,value', [
    ('required_output', ['red']), ('required_output', []),
    ('required_output', ['...']), ('final_marble_state', []),
    ('final_marble_state', ['...']), ('left_catcher', 0),
    ('left_catcher', True), ('left_catcher', '1'), ('left_catcher', -1),
    ('intercepted', 1),
])
def test_conflicting_or_malformed_targets_never_short_circuit(field, value):
    t = task()
    if field == 'required_output': t[field] = value
    elif field == 'final_marble_state': t['solution'][field] = value
    else: t['expected_output'][field] = value
    assert verdicts(t) == (False, False, False)


def test_missing_targets_do_not_fall_back_to_objective():
    t = task(); t['solution'].pop('final_marble_state'); t.pop('required_output')
    t['expected_output'] = {'description': 'blue goes left'}
    t['objective'] = 'Make all blue marbles reach the left exit'
    assert verdicts(t) == (False, False, False)


@pytest.mark.parametrize('keep', ['required_output', 'expected_output'])
def test_each_supported_target_can_stand_alone(keep):
    t = task(); t['solution'].pop('final_marble_state')
    t.pop('expected_output' if keep == 'required_output' else 'required_output')
    assert verdicts(t) == (True, True, True)


def test_loader_preserves_required_output(tmp_path):
    p = tmp_path / 'tiny.json'; p.write_text(json.dumps(task()))
    runner = TuringTumbleBenchmark.__new__(TuringTumbleBenchmark)
    info, _ = runner.load_task(p)
    assert info['required_output'] == ['blue']
    info['required_output'] = ['red']
    assert not runner.validate_synthesis(info, [])[0]


def test_final_bit_states_are_checked_even_with_matching_sequence():
    t = task(); t['expected_output']['final_bit_states'] = {'bit_99_99': 0}
    assert verdicts(t) == (False, False, False)


def test_empty_final_sequence_cannot_match_an_empty_run():
    t = task(); t['input_sequence'] = []; t['solution']['final_marble_state'] = []
    t.pop('required_output'); t['expected_output'] = {}
    assert verdicts(t) == (False, False, False)


def test_interception_with_counts_and_required_output_passes_all_entry_points():
    t = task(); t['board']['fixed_components'] = [{'type': 'interceptor', 'x': 2, 'y': 0}]
    t['solution']['final_marble_state'] = t['required_output'] = ['intercepted']
    t['expected_output'] = {'left_catcher': 0, 'right_catcher': 0, 'intercepted': 1}
    assert verdicts(t) == (True, True, True)


@pytest.mark.parametrize('expected,valid', [(1, True), (0, False)])
def test_final_bit_state_uses_simulated_state_before_tool_restoration(expected, valid):
    t = task(); t['board']['height'] = 2
    t['board']['fixed_components'] = [{'type': 'bit', 'x': 2, 'y': 0, 'state': 0},
                                       {'type': 'interceptor', 'x': 3, 'y': 1}]
    t['solution']['final_marble_state'] = t['required_output'] = ['intercepted']
    t['expected_output'] = {'intercepted': 1, 'final_bit_states': {'bit_2_0': expected}}
    assert verdicts(t) == (valid, valid, valid)


def test_tool_approach_slots_follow_current_catcher_geometry():
    t = task(); t['board']['height'] = 2
    t['board']['fixed_components'] = [{'type': 'ramp_left', 'x': 2, 'y': 0}]
    # Exit x=1 lies on the left bar even though the stored cup is at x=2.
    assert verdicts(t) == (True, True, True)
