#!/usr/bin/env python3
"""Classify positive-task eligibility without modifying the dataset.

Negative labels require separate proof review. A candidate is mechanically eligible,
not certified for objective semantics, provenance, question quality or split isolation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from tt_bench.benchmark.runner import TuringTumbleBenchmark
from tt_bench.simulator import verify_task
from tt_bench.simulator.inventory import used_inventory

KNOWN = {'ramp_left', 'ramp_right', 'bit', 'gear_bit', 'gear', 'crossover', 'interceptor', 'trigger'}


def classify(path: Path, root: Path) -> dict:
    raw = path.read_bytes()
    row = {'path': str(path.relative_to(root)), 'sha256': hashlib.sha256(raw).hexdigest(),
           'status': 'quarantined', 'reasons': []}
    reasons = row['reasons']
    try:
        task = json.loads(raw)
        row['task_id'] = task.get('task_id')
        if task.get('_meta', {}).get('variant_type') == 'unsolvable' or 'unsolvable' in path.parts:
            reasons.append('negative_label_requires_current_proof')
            row['status'] = 'negative_review'
            return row
        if task.get('task_id') != path.stem: reasons.append('task_id_filename_mismatch')
        if task.get('tier') not in (1, 2, 3, 4): reasons.append('invalid_tier')
        if type(task.get('challenge_number')) is not int or task['challenge_number'] < 1:
            reasons.append('invalid_challenge_number')
        seq = task.get('input_sequence')
        if not isinstance(seq, list) or not seq or any(c not in ('blue', 'red') for c in seq):
            reasons.append('invalid_input_sequence')
        board = task['board']; placed = task['solution']['placed_components']
        editable = {tuple(p) for p in board.get('editable_bit_states', [])}
        components = board['fixed_components'] + [p for p in placed if (p['x'],p['y']) not in editable]
        cells = [(c['x'], c['y']) for c in components]
        if len(set(cells)) != len(cells): reasons.append('overlapping_components')
        if any(c['type'] not in KNOWN or not 0 <= c['x'] < board['width']
               or not 0 <= c['y'] < board['height'] for c in components):
            reasons.append('invalid_component')
        inventory = task['available_parts']
        pooled = KNOWN - {'ramp_left','ramp_right'} | {'ramp'}
        if set(inventory) not in (KNOWN, pooled) or any(type(n) is not int or n < 0 for n in inventory.values()):
            reasons.append('invalid_inventory')
        for kind, count in used_inventory(placed, inventory, board).items():
            if type(inventory.get(kind)) is not int or inventory[kind] < count:
                reasons.append('insufficient_inventory:' + kind)
        for label, count in [('challenges_1comp', 1), ('challenges_2comp', 2)]:
            if label in path.parts and len(placed) != count: reasons.append('wrong_component_count')
        runner = TuringTumbleBenchmark.__new__(TuringTumbleBenchmark)
        info, _ = runner.load_task(path)
        valid, detail = runner.validate_synthesis(info, placed)
        if not valid: reasons.append('reference_rejected:' + detail)
        if not verify_task(task): reasons.append('simulator_reference_rejected')
        if runner.validate_synthesis(info, [])[0]: reasons.append('solves_without_placement')
        row['status'] = 'positive_candidate' if not reasons else 'quarantined'
    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        reasons.append(f'invalid_task:{type(exc).__name__}:{exc}')
    return row


def build(root: Path) -> dict:
    source_root = Path(__file__).resolve().parents[1]
    sources = ['src/tt_bench/simulator/targets.py', 'src/tt_bench/simulator/validation.py',
               'src/tt_bench/simulator/board.py', 'src/tt_bench/simulator/components.py',
               'src/tt_bench/simulator/inventory.py',
               'src/tt_bench/benchmark/runner.py', 'src/tt_bench/tools/executor.py',
               'scripts/build_dataset_manifest.py']
    source_hashes = {s: hashlib.sha256((source_root / s).read_bytes()).hexdigest() for s in sources}
    dirs = ['official/challenges/json', 'challenges_1comp', 'challenges_2comp', 'scaled']
    paths = [p for directory in dirs for p in sorted((root / directory).rglob('*.json'))]
    rows = [classify(p, root) for p in paths]
    ids = defaultdict(list)
    for row in rows:
        if isinstance(row.get('task_id'), str): ids[row['task_id']].append(row)
    for group in ids.values():
        if len(group) > 1:
            for row in group:
                row['reasons'].append('duplicate_task_id')
                if row['status'] == 'positive_candidate': row['status'] = 'quarantined'
    current_paths = [p for directory in dirs for p in sorted((root / directory).rglob('*.json'))]
    if paths != current_paths or any(
        hashlib.sha256((root / row['path']).read_bytes()).hexdigest() != row['sha256']
        for row in rows
    ) or any(hashlib.sha256((source_root / s).read_bytes()).hexdigest() != digest
             for s, digest in source_hashes.items()):
        raise RuntimeError('Dataset or validator changed during scan; rerun for a stable manifest')
    return {'schema_version': 1, 'scope': 'mechanical positive eligibility; not full dataset certification',
            'source_sha256': source_hashes,
            'counts': dict(Counter(row['status'] for row in rows)), 'files': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/tasks'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fail-on-quarantine', action='store_true')
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.root.resolve()):
        parser.error('Write the manifest outside the dataset directory')
    manifest = build(args.root)
    if not manifest['files']: parser.error('No challenge files found')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest['counts'], sort_keys=True))
    return int(args.fail_on_quarantine and any(row['status'] != 'positive_candidate' for row in manifest['files']))


if __name__ == '__main__':
    raise SystemExit(main())
