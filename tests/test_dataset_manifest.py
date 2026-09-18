"""Manifest generation must quarantine without modifying source tasks."""
import hashlib
import json

from scripts.build_dataset_manifest import KNOWN, build


def write_task(root, stem='tiny', *, negative=False):
    path = root / 'official/challenges/json' / (stem + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    task = {
        'task_id': stem, 'tier': 1, 'challenge_number': 1, 'input_sequence': ['blue'],
        'board': {'width': 5, 'height': 3, 'hopper_entry_mode': 'column',
                  'ball_hoppers': {'blue': {'x': 1, 'count': 1}, 'red': {'x': 4, 'count': 0}},
                  'trigger_levers': {'left': {'x': 1, 'y': 3}, 'right': {'x': 4, 'y': 3}},
                  'fixed_components': [{'type': 'ramp_right', 'x': 1, 'y': 0}]},
        'available_parts': {k: int(k == 'ramp_left') for k in KNOWN},
        'solution': {'placed_components': [{'type': 'ramp_left', 'x': 2, 'y': 1}],
                     'final_marble_state': ['blue']},
        'required_output': ['blue'],
    }
    if negative: task['_meta'] = {'variant_type': 'unsolvable'}
    path.write_text(json.dumps(task))
    return path


def test_manifest_is_repeatable_and_preserves_dataset(tmp_path):
    path = write_task(tmp_path); before = path.read_bytes()
    first = build(tmp_path)
    assert first == build(tmp_path)
    assert first['counts'] == {'positive_candidate': 1}
    assert first['files'][0]['sha256'] == hashlib.sha256(before).hexdigest()
    assert path.read_bytes() == before


def test_conflicting_target_is_quarantined(tmp_path):
    path = write_task(tmp_path); task = json.loads(path.read_text())
    task['required_output'] = ['red']; path.write_text(json.dumps(task))
    row = build(tmp_path)['files'][0]
    assert row['status'] == 'quarantined'
    assert any('required_output' in r for r in row['reasons'])


def test_negative_label_never_becomes_eligible_without_proof(tmp_path):
    write_task(tmp_path, negative=True)
    assert build(tmp_path)['counts'] == {'negative_review': 1}


def test_duplicate_ids_quarantine_both_files(tmp_path):
    one = write_task(tmp_path)
    two = write_task(tmp_path, 'second')
    task = json.loads(two.read_text()); task['task_id'] = 'tiny'; two.write_text(json.dumps(task))
    manifest = build(tmp_path)
    assert manifest['counts'] == {'quarantined': 2}
    assert all('duplicate_task_id' in row['reasons'] for row in manifest['files'])


def test_invalid_json_is_quarantined_without_aborting_sweep(tmp_path):
    path = write_task(tmp_path); path.write_text('{broken')
    row = build(tmp_path)['files'][0]
    assert row['status'] == 'quarantined'
    assert row['reasons'][0].startswith('invalid_task:JSONDecodeError')


def test_concurrent_edit_prevents_publishing_mixed_manifest(tmp_path, monkeypatch):
    import pytest
    from scripts import build_dataset_manifest as module
    path = write_task(tmp_path)
    original = module.classify

    def classify_then_edit(p, root):
        row = original(p, root)
        p.write_text(p.read_text() + '\n')
        return row

    monkeypatch.setattr(module, 'classify', classify_then_edit)
    with pytest.raises(RuntimeError, match='changed during scan'):
        module.build(tmp_path)
