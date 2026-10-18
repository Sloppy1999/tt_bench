"""Physical inventories: either ramp orientation consumes the same part pool."""
from collections import Counter


def inventory_key(kind: str, available: dict) -> str:
    return 'ramp' if kind in ('ramp_left', 'ramp_right') and 'ramp' in available else kind


def used_inventory(placements: list[dict], available: dict, board_data: dict | None = None) -> Counter:
    editable = {tuple(p) for p in (board_data or {}).get('editable_bit_states', [])}
    return Counter(
        inventory_key(p.get('type', p.get('component_type')), available)
        for p in placements if (p['x'], p['y']) not in editable
    )
