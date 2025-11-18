# src/explicit_reachability.py
import argparse
import json
import time
from collections import deque
from pathlib import Path
import sys

# ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import parse_pnml, validate_net
from src.memory_utils import start_memory_tracking, stop_memory_tracking

def marking_to_tuple(mark):
    return tuple(int(x) for x in mark)

def tuple_to_list(tup):
    return [int(x) for x in tup]

def is_fireable(transition, marking_tuple):
    for p in transition['pre']:
        if marking_tuple[p] == 0:
            return False
    return True

def fire_transition(transition, marking_tuple, n_places):
    m = list(marking_tuple)
    for p in transition['pre']:
        m[p] = 0
    for p in transition['post']:
        m[p] = 1
    if len(m) != n_places:
        raise RuntimeError("marking length mismatch")
    return tuple(m)

def bfs_reachability(net):
    validate_net(net)
    places = net['places']
    transitions = net['transitions']
    n_places = len(places)

    # special-case: no transitions => only initial marking reachable
    start = marking_to_tuple(net['initial_marking'])
    if len(transitions) == 0:
        return {
            'places': [{'id': p['id'], 'name': p.get('name'), 'index': p['index']} for p in places],
            'transitions': [],
            'initial_marking': list(start),
            'reachable_count': 1,
            'reachable_markings': [list(start)],
            'edges': []
        }

    seen = set([start])
    parent = {start: None}
    parent_trans = {start: None}
    order = [start]
    q = deque([start])

    while q:
        m = q.popleft()
        for t in transitions:
            if is_fireable(t, m):
                m2 = fire_transition(t, m, n_places)
                if m2 not in seen:
                    seen.add(m2)
                    parent[m2] = m
                    parent_trans[m2] = t['id']
                    order.append(m2)
                    q.append(m2)
    edges = []
    for child, p in parent.items():
        if p is None:
            continue
        edges.append({
            'from': tuple_to_list(p),
            'to': tuple_to_list(child),
            'by_transition': parent_trans[child]
        })
    return {
        'places': [{'id': p['id'], 'name': p.get('name'), 'index': p['index']} for p in places],
        'transitions': [{'id': t['id'], 'name': t.get('name'), 'index': t['index']} for t in transitions],
        'initial_marking': tuple_to_list(start),
        'reachable_count': len(seen),
        'reachable_markings': [tuple_to_list(m) for m in order],
        'edges': edges
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pnml', help='path to pnml file')
    ap.add_argument('--out', help='output json path', default=None)
    ap.add_argument('--mem', action='store_true', help='measure memory for parse and BFS')
    args = ap.parse_args()

    pnml_path = Path(args.pnml)
    if not pnml_path.exists():
        print("ERROR: PNML file not found:", pnml_path)
        return

    parser_tracker = start_memory_tracking() if args.mem else {}
    t0 = time.perf_counter()
    try:
        net = parse_pnml(str(pnml_path))
        validate_net(net)
    except Exception as e:
        print("Error parsing or validating PNML:", e)
        return
    t1 = time.perf_counter()
    parser_mem = stop_memory_tracking(parser_tracker) if args.mem else {}

    bfs_tracker = start_memory_tracking() if args.mem else {}
    t2 = time.perf_counter()
    res = bfs_reachability(net)
    t3 = time.perf_counter()
    bfs_mem = stop_memory_tracking(bfs_tracker) if args.mem else {}

    res_meta = {
        'parser_time_s': t1 - t0,
        'bfs_time_s': t3 - t2,
        'parser_memory': parser_mem,
        'bfs_memory': bfs_mem
    }
    res.update(res_meta)

    out_path = args.out or (pnml_path.with_suffix('.reach.json').name)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    print(f"Reachability analysis written to {out_path}")
    print(f"Places: {len(net['places'])}, Transitions: {len(net['transitions'])}")
    print(f"Reachable markings: {res['reachable_count']}")
    if args.mem:
        print("Parser memory:", parser_mem)
        print("BFS memory:", bfs_mem)

if __name__ == '__main__':
    main()
