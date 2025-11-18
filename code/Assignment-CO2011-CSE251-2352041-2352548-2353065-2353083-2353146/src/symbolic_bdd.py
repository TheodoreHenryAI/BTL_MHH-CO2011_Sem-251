# src/symbolic_bdd.py
import argparse
import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# import helpers from common
from src.common import (
    parse_pnml,
    validate_net,
    build_transition_relation,
    marking_to_bdd,
    enumerate_markings_from_bdd,
    varname,
    varname_prime,
)
from src.memory_utils import start_memory_tracking, stop_memory_tracking


def symbolic_reachability(net, enum_limit=10000, verbose=False, var_order='interleaved', memtrack=False):
    """
    Compute reachable set symbolically using per-transition image computation.
    """
    validate_net(net)

    try:
        from dd import autoref as _bdd
    except Exception as e:
        raise RuntimeError("Missing 'dd' package. Install with: python -m pip install dd") from e

    places = net['places']
    transitions = net['transitions']
    n = len(places)

    # declare variable order
    if var_order == 'interleaved':
        ordered_vars = []
        for i in range(n):
            ordered_vars.append(varname(i))
            ordered_vars.append(varname_prime(i))
    else:  # grouped: all old then all primed
        ordered_vars = [varname(i) for i in range(n)] + [varname_prime(i) for i in range(n)]

    bdd = _bdd.BDD()
    bdd.declare(*ordered_vars)

    old_vars = [varname(i) for i in range(n)]
    new_vars = [varname_prime(i) for i in range(n)]

    if verbose:
        print("BDD declared vars:", ordered_vars)

    # Build per-transition relations R_t
    R_list = []
    for t in transitions:
        Rt = build_transition_relation(bdd, n, t)
        R_list.append(Rt)

    # initial marking as BDD over old vars
    start = tuple(int(x) for x in net['initial_marking'])
    reachable = marking_to_bdd(bdd, old_vars, start)

    # fixpoint iteration (per-transition image)
    iteration = 0
    bdd_tracker = start_memory_tracking() if memtrack else {}
    t0 = time.perf_counter()
    while True:
        iteration += 1
        if verbose:
            print(f"[BDD iter {iteration}] computing image (per-transition)...")
        img_old_total = bdd.false
        # compute image for each transition separately
        for Rt in R_list:
            conj = reachable & Rt  # conjunction over old & new variables
            # existentially quantify old_vars -> results over primed vars
            img_new = bdd.exist(old_vars, conj)
            # rename primed -> old
            rename_map = {varname_prime(i): varname(i) for i in range(n)}
            img_old = bdd.let(rename_map, img_new)
            img_old_total = img_old_total | img_old

        next_reachable = reachable | img_old_total
        if next_reachable == reachable:
            if verbose:
                print(f"[BDD iter {iteration}] fixpoint reached.")
            break
        reachable = next_reachable
    t1 = time.perf_counter()
    bdd_mem = stop_memory_tracking(bdd_tracker) if memtrack else {}

    # enumeration (up to limit)
    enum_tracker = start_memory_tracking() if memtrack else {}
    t2 = time.perf_counter()
    markings = []
    if enum_limit:
        markings = enumerate_markings_from_bdd(bdd, reachable, old_vars, limit=enum_limit)
    t3 = time.perf_counter()
    enum_mem = stop_memory_tracking(enum_tracker) if memtrack else {}

    reachable_count = (f">={enum_limit}" if enum_limit and len(markings) >= enum_limit else len(markings))

    # build edges between enumerated markings (for visualization)
    edges = []
    for m in markings:
        for t in transitions:
            enabled = True
            for p in t['pre']:
                if m[p] == 0:
                    enabled = False
                    break
            if not enabled:
                continue
            m2 = m.copy()
            for p in t['pre']:
                m2[p] = 0
            for p in t['post']:
                m2[p] = 1
            edges.append({'from': m, 'to': m2, 'by_transition': t['id']})

    return {
        'places': [{'id': p['id'], 'name': p.get('name'), 'index': p['index']} for p in places],
        'transitions': [{'id': t['id'], 'name': t.get('name'), 'index': t['index']} for t in transitions],
        'initial_marking': list(start),
        'reachable_count': reachable_count,
        'reachable_markings': markings,
        'edges': edges,
        'bdd_time_s': t1 - t0,
        'bdd_memory': bdd_mem,
        'enumeration_time_s': t3 - t2,
        'enumeration_memory': enum_mem
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pnml', help='path to pnml file')
    ap.add_argument('--out', help='output json path', default=None)
    ap.add_argument('--enum_limit', type=int, default=10000)
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--var_order', choices=['interleaved', 'grouped'], default='interleaved')
    ap.add_argument('--mem', action='store_true', help='measure memory for BDD build & enumeration')
    args = ap.parse_args()

    pnml_path = Path(args.pnml)
    if not pnml_path.exists():
        print("ERROR: PNML file not found:", pnml_path)
        return

    if args.enum_limit is not None and args.enum_limit < 0:
        print("ERROR: --enum_limit must be non-negative (use 0 to disable enumeration).")
        return

    try:
        net = parse_pnml(str(pnml_path))
        validate_net(net)
    except Exception as e:
        print("Error parsing or validating PNML:", e)
        return

    try:
        t_total0 = time.perf_counter()
        res = symbolic_reachability(net,
                                    enum_limit=(args.enum_limit if args.enum_limit > 0 else None),
                                    verbose=args.verbose,
                                    var_order=args.var_order,
                                    memtrack=args.mem)
        t_total1 = time.perf_counter()
    except RuntimeError as e:
        print("Runtime error:", e)
        return
    except Exception as e:
        print("Unexpected error during symbolic reachability:", e)
        return

    res['symbolic_time_s'] = (t_total1 - t_total0)

    out_path = args.out or (pnml_path.with_suffix('.reach_bdd.json').name)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    print(f"BDD reachability written to {out_path}")
    print(f"Places: {len(net['places'])}, Transitions: {len(net['transitions'])}")
    print(f"Reachable markings (reported): {res['reachable_count']}, total time: {res['symbolic_time_s']:.6f}s")
    if args.mem:
        print("BDD memory:", res.get('bdd_memory'))
        print("Enumeration memory:", res.get('enumeration_memory'))


if __name__ == '__main__':
    main()
