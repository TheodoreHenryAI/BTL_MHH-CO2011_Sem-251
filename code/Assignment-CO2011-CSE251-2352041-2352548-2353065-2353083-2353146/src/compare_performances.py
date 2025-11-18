# src/compare_performances.py
import argparse
import json
import time
import sys
from pathlib import Path

# ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import parse_pnml, validate_net, compute_reachable_bdd, enumerate_markings_from_bdd
from src.memory_utils import start_memory_tracking, stop_memory_tracking

def bfs_limited(net, enum_limit=None):
    places = net['places']
    transitions = net['transitions']
    n_places = len(places)

    start = tuple(int(x) for x in net['initial_marking'])
    if len(transitions) == 0:
        return {
            'reachable_count': 1,
            'reachable_markings': [list(start)],
            'edges': []
        }

    from collections import deque
    seen = set([start])
    order = [start]
    parent_trans = {start: None}
    q = deque([start])

    while q:
        m = q.popleft()
        for t in transitions:
            enabled = True
            for p in t['pre']:
                if m[p] == 0:
                    enabled = False
                    break
            if not enabled:
                continue
            m2 = list(m)
            for p in t['pre']:
                m2[p] = 0
            for p in t['post']:
                m2[p] = 1
            m2 = tuple(m2)
            if m2 not in seen:
                seen.add(m2)
                order.append(m2)
                parent_trans[m2] = t['id']
                q.append(m2)
                if enum_limit and len(seen) >= enum_limit:
                    return {
                        'reachable_count': f">={enum_limit}",
                        'reachable_markings': [list(x) for x in order],
                        'edges': []
                    }
    edges = []
    for m in order:
        for t in transitions:
            enabled = True
            for pr in t['pre']:
                if m[pr] == 0:
                    enabled = False
                    break
            if not enabled:
                continue
            m2 = list(m)
            for pr in t['pre']:
                m2[pr] = 0
            for po in t['post']:
                m2[po] = 1
            edges.append({'from': list(m), 'to': m2, 'by_transition': t['id']})
    return {
        'reachable_count': len(seen),
        'reachable_markings': [list(x) for x in order],
        'edges': edges
    }

def run_compare(pnml_path: str, out_path: str, bfs_limit: int = None, bdd_enum_limit: int = 10000, mem: bool = False, verbose: bool = False):
    parse_tracker = start_memory_tracking() if mem else {}
    t0 = time.perf_counter()
    try:
        net = parse_pnml(pnml_path)
    except Exception as e:
        raise RuntimeError(f"Failed to parse PNML {pnml_path}: {e}")
    t1 = time.perf_counter()
    parse_mem = stop_memory_tracking(parse_tracker) if mem else {}

    try:
        validate_net(net)
    except Exception as e:
        raise RuntimeError(f"Net validation failed: {e}")

    if verbose:
        print("[compare] Starting explicit BFS (enumeration limit: {})".format(bfs_limit))
    bfs_tracker = start_memory_tracking() if mem else {}
    tb0 = time.perf_counter()
    bfs_res = bfs_limited(net, enum_limit=bfs_limit)
    tb1 = time.perf_counter()
    bfs_mem = stop_memory_tracking(bfs_tracker) if mem else {}

    if verbose:
        print("[compare] Starting BDD build (fixpoint)")
    bdd_build_tracker = start_memory_tracking() if mem else {}
    tbd0 = time.perf_counter()
    bdd_mgr, reachable_bdd, old_vars, new_vars = compute_reachable_bdd(net, var_order='interleaved', verbose=verbose)
    tbd1 = time.perf_counter()
    bdd_build_mem = stop_memory_tracking(bdd_build_tracker) if mem else {}

    bdd_enum_res = {'reachable_count': None, 'reachable_markings': None}
    enum_time = None
    enum_mem = None
    if bdd_enum_limit is not None and bdd_enum_limit > 0:
        if verbose:
            print(f"[compare] Enumerating up to {bdd_enum_limit} markings from BDD...")
        bdd_enum_tracker = start_memory_tracking() if mem else {}
        ten0 = time.perf_counter()
        markings = enumerate_markings_from_bdd(bdd_mgr, reachable_bdd, old_vars, limit=bdd_enum_limit)
        ten1 = time.perf_counter()
        enum_time = ten1 - ten0
        enum_mem = stop_memory_tracking(bdd_enum_tracker) if mem else {}
        bdd_enum_res['reachable_markings'] = markings
        bdd_enum_res['reachable_count'] = (f">={bdd_enum_limit}" if len(markings) >= bdd_enum_limit else len(markings))
    else:
        bdd_enum_res['reachable_markings'] = []
        bdd_enum_res['reachable_count'] = None

    explicit_count = bfs_res['reachable_count']
    symbolic_count = bdd_enum_res['reachable_count'] if bdd_enum_res['reachable_count'] is not None else 'computed_bdd_only'

    comparison = {'equal_sets': None, 'notes': []}
    try:
        if isinstance(explicit_count, int) and isinstance(bdd_enum_res['reachable_count'], int):
            setE = set(tuple(x) for x in bfs_res['reachable_markings'])
            setB = set(tuple(x) for x in bdd_enum_res['reachable_markings'])
            comparison['equal_sets'] = (setE == setB)
            if not comparison['equal_sets']:
                diffE = [list(x) for x in (setE - setB)]
                diffB = [list(x) for x in (setB - setE)]
                comparison['notes'].append({'in_explicit_not_bdd': diffE, 'in_bdd_not_explicit': diffB})
        else:
            comparison['notes'].append("One or both enumerations were truncated/not performed; cannot fully compare sets.")
    except Exception as e:
        comparison['notes'].append(f"Comparison failed: {e}")

    result = {
        'pnml': str(pnml_path),
        'parser': {
            'time_s': t1 - t0,
            'memory': parse_mem
        },
        'explicit': {
            'time_s': tb1 - tb0,
            'memory': bfs_mem,
            'reachable_count': bfs_res['reachable_count'],
            'enumerated_markings_sample_count': len(bfs_res.get('reachable_markings') or [])
        },
        'symbolic_bdd': {
            'build_time_s': tbd1 - tbd0,
            'build_memory': bdd_build_mem,
            'enumeration_time_s': enum_time,
            'enumeration_memory': enum_mem,
            'reachable_count': bdd_enum_res['reachable_count'],
            'enumerated_markings_sample_count': len(bdd_enum_res.get('reachable_markings') or [])
        },
        'comparison': comparison,
        'params': {
            'bfs_limit': bfs_limit,
            'bdd_enum_limit': bdd_enum_limit,
            'mem': mem
        },
        'timestamp': time.time()
    }

    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    print("=== Performance comparison summary ===")
    print("PNML:", pnml_path)
    print("Parser: time {:.6f}s mem-tracked: {}".format(result['parser']['time_s'], bool(result['parser']['memory'])))
    print("Explicit BFS: time {:.6f}s reachable_count: {} mem-tracked: {}".format(
        result['explicit']['time_s'], result['explicit']['reachable_count'], bool(result['explicit']['memory'])))
    print("BDD build: time {:.6f}s mem-tracked: {}".format(result['symbolic_bdd']['build_time_s'], bool(result['symbolic_bdd']['build_memory'])))
    print("BDD enumeration: time {} reachable_count: {} mem-tracked: {}".format(
        ("{:.6f}s".format(result['symbolic_bdd']['enumeration_time_s']) if result['symbolic_bdd']['enumeration_time_s'] is not None else "n/a"),
        result['symbolic_bdd']['reachable_count'],
        bool(result['symbolic_bdd']['enumeration_memory'])))
    if comparison['equal_sets'] is not None:
        print("Sets equal:", comparison['equal_sets'])
    else:
        print("Sets comparison: not performed (truncated or not enumerated)")
    print("Result JSON written to:", outp.resolve())
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pnml', help='path to pnml file')
    ap.add_argument('--out', help='output json', default='results/compare.json')
    ap.add_argument('--bfs_limit', type=int, default=1000000, help='max markings to discover in BFS (use large number for no truncation)')
    ap.add_argument('--bdd_enum_limit', type=int, default=10000, help='max markings to enumerate from BDD (0 disables enumeration)')
    ap.add_argument('--mem', action='store_true', help='measure memory (requires psutil)')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    pnml = Path(args.pnml)
    if not pnml.exists():
        print("ERROR: PNML not found:", pnml)
        sys.exit(2)

    bfs_limit = args.bfs_limit if args.bfs_limit > 0 else None
    bdd_enum_limit = args.bdd_enum_limit if args.bdd_enum_limit > 0 else None

    try:
        res = run_compare(str(pnml), args.out, bfs_limit=bfs_limit, bdd_enum_limit=bdd_enum_limit, mem=args.mem, verbose=args.verbose)
    except Exception as e:
        print("ERROR during comparison:", e)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(3)

if __name__ == '__main__':
    main()
