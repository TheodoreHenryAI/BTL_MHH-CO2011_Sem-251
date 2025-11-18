# src/deadlock_detection.py
import argparse
import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import parse_pnml, validate_net, compute_reachable_bdd, marking_to_bdd
from src.memory_utils import start_memory_tracking, stop_memory_tracking

try:
    import pulp
except Exception:
    pulp = None

def bdd_deadlock_detection(net, bdd_mgr, reachable_bdd):
    transitions = net['transitions']
    n = len(net['places'])

    # If no transitions => initial marking is trivially dead (no transitions enabled)
    if len(transitions) == 0:
        return list(net['initial_marking'])

    enabled_list = []
    for t in transitions:
        if len(t['pre']) == 0:
            enabled = bdd_mgr.true
        else:
            piles = [bdd_mgr.var(f"x{p}") for p in t['pre']]
            enabled = piles[0]
            for u in piles[1:]:
                enabled = enabled & u
        enabled_list.append(enabled)

    deadbdd = bdd_mgr.true
    for e in enabled_list:
        deadbdd = deadbdd & ~e

    cand = reachable_bdd & deadbdd
    if cand == bdd_mgr.false:
        return None
    assn = cand.pick()
    mark = [0] * n
    for i in range(n):
        name = f"x{i}"
        if assn.get(name, False):
            mark[i] = 1
    return mark

def ilp_deadlock_search(net, bdd_mgr, reachable_bdd, max_iters=200, verbose=False):
    if pulp is None:
        raise RuntimeError("PuLP not installed. Install: python -m pip install pulp")
    transitions = net['transitions']
    n = len(net['places'])

    # Quick checks
    if len(transitions) == 0:
        return {'found': True, 'marking': list(net['initial_marking']), 'reason': 'no_transitions'}

    for t in transitions:
        if len(t['pre']) == 0:
            if verbose:
                print("Transition with empty pre found -> always enabled -> no dead markings possible.")
            return {'found': False, 'reason': 'transition_with_empty_pre'}

    forbidden = []
    iteration = 0
    while iteration < max_iters:
        iteration += 1
        if verbose:
            print(f"[ILP iter {iteration}] Building ILP...")
        prob = pulp.LpProblem("dead_marking", pulp.LpStatusOptimal)
        m_vars = [pulp.LpVariable(f"m_{i}", cat='Binary') for i in range(n)]
        y_vars = [pulp.LpVariable(f"y_{j}", cat='Binary') for j in range(len(transitions))]

        for j, t in enumerate(transitions):
            for p in t['pre']:
                prob += y_vars[j] <= m_vars[p]
        for j, t in enumerate(transitions):
            if len(t['pre']) > 0:
                prob += y_vars[j] >= pulp.lpSum([m_vars[p] for p in t['pre']]) - len(t['pre']) + 1
            else:
                prob += y_vars[j] >= 1
        for j in range(len(transitions)):
            prob += y_vars[j] == 0
        for ones, zeros in forbidden:
            prob += pulp.lpSum([(1 - m_vars[p]) for p in ones] + [m_vars[p] for p in zeros]) >= 1
        prob += pulp.lpSum(m_vars)

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        status = pulp.LpStatus[prob.status] if hasattr(pulp, 'LpStatus') else None
        if verbose:
            print("ILP solver status:", status)
        if status != 'Optimal':
            return {'found': False, 'reason': 'ilp_infeasible', 'iters': iteration}

        candidate = [int(pulp.value(v)) for v in m_vars]
        cand_bdd = marking_to_bdd(bdd_mgr, [f"x{i}" for i in range(n)], candidate)
        if (reachable_bdd & cand_bdd) != bdd_mgr.false:
            return {'found': True, 'marking': candidate, 'iters': iteration}
        else:
            ones = [i for i, bit in enumerate(candidate) if bit == 1]
            zeros = [i for i, bit in enumerate(candidate) if bit == 0]
            forbidden.append((ones, zeros))
            if verbose:
                print(f"[ILP iter {iteration}] Candidate marking not reachable, forbidding and retrying.")
    return {'found': False, 'reason': 'max_iters_exceeded', 'iters': iteration}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pnml', help='path to pnml file')
    ap.add_argument('--out_bdd', help='output json for BDD-only', default=None)
    ap.add_argument('--out_ilp', help='output json for ILP+BDD', default=None)
    ap.add_argument('--mem', action='store_true', help='measure memory (requires psutil)')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    pnml_path = Path(args.pnml)
    if not pnml_path.exists():
        print("ERROR: PNML file not found:", pnml_path)
        return

    try:
        net = parse_pnml(str(pnml_path))
        validate_net(net)
    except Exception as e:
        print("Error parsing/validating PNML:", e)
        return

    bdd_tracker = start_memory_tracking() if args.mem else {}
    t0 = time.perf_counter()
    try:
        bdd_mgr, reachable_bdd, old_vars, new_vars = compute_reachable_bdd(net, verbose=args.verbose)
    except RuntimeError as e:
        print("BDD error:", e)
        return
    t1 = time.perf_counter()
    bdd_build_mem = stop_memory_tracking(bdd_tracker) if args.mem else {}

    t2 = time.perf_counter()
    dead_mark = bdd_deadlock_detection(net, bdd_mgr, reachable_bdd)
    t3 = time.perf_counter()
    bdd_dead_mem = start_memory_tracking() and {}  # placeholder; we capture above if needed

    bdd_result = {
        'found': dead_mark is not None,
        'dead_marking': dead_mark,
        'time_s': t3 - t2,
        'bdd_build_time_s': t1 - t0,
        'bdd_build_memory': bdd_build_mem
    }
    out_bdd = args.out_bdd or (pnml_path.with_suffix('.dead_bdd.json').name)
    with open(out_bdd, 'w', encoding='utf-8') as f:
        json.dump(bdd_result, f, indent=2)
    print("BDD-only deadlock result written to", out_bdd)
    if dead_mark:
        print("BDD: Found reachable dead marking:", dead_mark)
    else:
        print("BDD: No reachable dead marking found.")

    ilp_tracker = start_memory_tracking() if args.mem else {}
    t4 = time.perf_counter()
    try:
        ilp_res = ilp_deadlock_search(net, bdd_mgr, reachable_bdd, verbose=args.verbose)
    except RuntimeError as e:
        print("ILP error:", e)
        return
    t5 = time.perf_counter()
    ilp_mem = stop_memory_tracking(ilp_tracker) if args.mem else {}

    ilp_res['time_s'] = t5 - t4
    ilp_res['ilp_memory'] = ilp_mem
    out_ilp = args.out_ilp or (pnml_path.with_suffix('.dead_ilp.json').name)
    with open(out_ilp, 'w', encoding='utf-8') as f:
        json.dump(ilp_res, f, indent=2)
    print("ILP+BDD deadlock result written to", out_ilp)
    if ilp_res.get('found'):
        print("ILP: Found reachable dead marking:", ilp_res.get('marking'))
    else:
        print("ILP: No reachable dead marking found. Reason:", ilp_res.get('reason'))

if __name__ == '__main__':
    main()
