"""Ground-truth invariants and scorer/tool agreement for the reviewed guide."""
import json
from copy import deepcopy
from pathlib import Path

import pytest

from tt_bench.benchmark.runner import TuringTumbleBenchmark
from tt_bench.simulator import Board, GearBit, build_gear_connections, verify_task
from tt_bench.simulator.targets import validate_targets
from tt_bench.tools.executor import TuringTumbleToolExecutor, create_executor_from_task
from tt_bench.tools.interventions import InterventionExecutor

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'data/tasks/official/challenges/json'
FILES = sorted(DIRECTORY.glob('*.json'))


def load(suffix):
    return json.loads((DIRECTORY / f'tt-official-ch{suffix}.json').read_text())


def executor(task, kind=TuringTumbleToolExecutor):
    return create_executor_from_task(
        task['board'], task['board']['fixed_components'], task['available_parts'],
        target_sequence=task['input_sequence'],
        target_final_state=task['solution'].get('final_marble_state'),
        expected_output=task.get('expected_output'), required_output=task.get('required_output'),
        trials=task.get('trials'), registers=task.get('registers'), executor_type=kind,
    )


@pytest.mark.parametrize('path', FILES, ids=lambda p:p.stem)
def test_reference_passes_simulator_scorer_and_agent_tools(path):
    task = json.loads(path.read_text())
    assert verify_task(task)
    runner = TuringTumbleBenchmark.__new__(TuringTumbleBenchmark)
    placements = task['solution']['placed_components']
    ok, why = runner.validate_synthesis(task, placements)
    assert ok, why
    agent = executor(task)
    for p in placements:
        result = agent.place_component(p['type'], p['x'], p['y'], p.get('state', 0))
        assert result['success'], result
    agent.run_simulation(task['input_sequence'])
    assert agent.is_solution_found()


def test_guide_membership_and_inventory():
    assert len(FILES) == 55
    assert not (DIRECTORY / 'tt-official-ch01-pA.json').exists()
    for suffix in ('21-pA','22-pA'):
        exercise = json.loads((ROOT / f'data/tasks/official/exercises/tt-official-ch{suffix}.json').read_text())
        assert exercise['task_type'] == 'paper_exercise'
        assert 'board' not in exercise
    assert load('09')['available_parts']['ramp'] == 18
    assert load('11')['available_parts']['ramp'] == 15
    assert load('24')['available_parts']['ramp'] == 14
    for suffix,count in [('09',10),('24',12),('26',10)]:
        assert {h['count'] for h in load(suffix)['board']['ball_hoppers'].values()} == {count}


def test_ch24_runs_through_the_counter_and_intercepts_tenth():
    task = load('24')
    board = Board.from_task_dict(task)
    results = board.run(task['input_sequence'])
    assert len(results) == 10
    assert all(r.colour == 'blue' and r.caught_by == 'left_catcher' for r in results[:9])
    assert results[9].caught_by == 'interceptor'
    assert (3, 0) in results[0].path
    broken = deepcopy(task)
    broken['board']['hopper_entry_mode'] = 'column'
    assert not verify_task(broken)


def test_ramp_orientations_share_one_inventory_pool():
    agent = TuringTumbleToolExecutor(Board(), {'ramp':2})
    assert agent.place_component('ramp_left',0,0)['success']
    assert agent.place_component('ramp_right',1,1)['success']
    assert not agent.place_component('ramp_left',2,2)['success']
    assert agent.remove_component(0,0)['success']
    assert agent.place_component('ramp_right',2,2)['success']
    runner = TuringTumbleBenchmark.__new__(TuringTumbleBenchmark)
    task = {'available_parts':{'ramp':2}, 'solution':{'placed_components':[]}}
    assert not runner._validate_available_parts(task,[
        {'type':'ramp_left','x':0,'y':0}, {'type':'ramp_right','x':1,'y':1},
        {'type':'ramp_left','x':2,'y':2}])[0]


def test_state_selection_requires_an_explicit_choice_without_consuming_parts():
    task = load('23-pA')
    agent = executor(task, InterventionExecutor)
    agent.legality = True
    agent.run_simulation(['blue'])
    assert not agent.is_solution_found()
    for p in task['solution']['placed_components']:
        result = agent.execute('place_component',dict(component_type=p['type'],x=p['x'],y=p['y'],state=p['state']))
        assert result['success'], result
    assert agent._used_parts == {}
    agent.run_simulation(['blue'])
    assert agent.is_solution_found()
    assert not agent.remove_component(3,0)['success']
    assert agent.get_board_state()['components'][0]['source'] == 'fixed'


def test_absent_gears_never_get_created_or_connect_separate_latches():
    board = Board()
    board.place(2,2,GearBit(2,2))
    board.place(4,4,GearBit(4,4))
    build_gear_connections(board)
    assert set(board.components) == {(2,2),(4,4)}
    assert not board.gear_connections
    task = load('29-bA')
    board = Board.from_task_dict(task)
    assert len(board.to_llm_dict()['gear_groups']) == 2


@pytest.mark.parametrize('suffix',['19','20'])
def test_interception_colour_is_enforced_in_every_logic_case(suffix):
    task = load(suffix)
    assert len(task['trials']) == 4
    wrong = deepcopy(task)
    wrong['trials'][0]['expect']['intercepted_colours'] = ['blue']
    assert not verify_task(wrong)


def test_trial_rejects_a_gap_even_if_its_colour_target_matches():
    task = load('18')
    board = Board.from_task_dict(task)
    # Blue enters (3,0); removing its ramp creates a gap even with a catcher below.
    board.remove(3,0)
    ok, reason = validate_targets(task,board,[])
    assert not ok
    assert 'free fall' in reason
