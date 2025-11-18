# src/optimization.py
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

def marking_to_tuple(mark):
    return tuple(int(x) for x in mark)

def bfs_enumerate(net, enum_limit=None):
    from collections import deque
    places = net['places']
    transitions = net['transitions']
    n_places = len(places)
    start = marking_to_tuple(net['initial_marking'])
    seen = set([start])
    order = [start]
    q = deque([start])
    # special: no transitions => only start
    if len(transitions) == 0:
        return [start]
    while q:
        m = q.popleft()
        for t in transitions:
            enabled = all(m[p] == 1 for p in t['pre'])
            if enabled:
                m2 = list(m)
                for p in t['pre']:
                    m2[p] = 0
                for p in t['post']:
                    m2[p] = 1
                m2 = tuple(m2)
                if m2 not in seen:
                    seen.add(m2)
                    order.append(m2)
                    q.append(m2)
        if enum_limit and len(seen) > enum_limit:
            return None
    return order

def ilp_bdd_maximize(net, bdd_mgr, reachable_bdd, c_vector, max_iters=200, verbose=False):
    if pulp is None:
        raise RuntimeError("PuLP not installed. Install: python -m pip install pulp")
    places = net['places']
    transitions = net['transitions']
    n = len(places)
    forbidden = []
    iteration = 0
    while iteration < max_iters:
        iteration += 1
        if verbose:
            print(f"[ILP iter {iteration}] building ILP...")
        prob = pulp.LpProblem("max_obj", pulp.LpMaximize)
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
        prob += pulp.lpSum([c_vector[i] * m_vars[i] for i in range(n)])
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        status = pulp.LpStatus[prob.status] if hasattr(pulp, 'LpStatus') else None
        if verbose:
            print("ILP status:", status)
        if status != 'Optimal':
            return {'found': False, 'reason': 'ilp_infeasible', 'iters': iteration}
        candidate = [int(pulp.value(v)) for v in m_vars]
        cand_bdd = marking_to_bdd(bdd_mgr, [f"x{i}" for i in range(n)], candidate)
        if (reachable_bdd & cand_bdd) != bdd_mgr.false:
            val = sum(ci * xi for ci, xi in zip(c_vector, candidate))
            return {'found': True, 'marking': candidate, 'value': val, 'iters': iteration}
        else:
            ones = [i for i, bit in enumerate(candidate) if bit == 1]
            zeros = [i for i, bit in enumerate(candidate) if bit == 0]
            forbidden.append((ones, zeros))
            if verbose:
                print(f"[ILP iter {iteration}] candidate unreachable; forbidding and continuing.")
    return {'found': False, 'reason': 'max_iters_exceeded', 'iters': iteration}

def parse_weights_arg(weights_arg, n_places):
    if weights_arg is None:
        return [1] * n_places
    if ',' in weights_arg:
        parts = [p.strip() for p in weights_arg.split(',')]
        vals = []
        try:
            vals = [float(p) for p in parts]
        except Exception:
            raise ValueError("Invalid numeric value in --weights")
        if len(vals) != n_places:
            raise ValueError(f"Weights length ({len(vals)}) mismatch with number of places ({n_places})")
        return vals
    else:
        p = Path(weights_arg)
        if p.exists():
            data = json.loads(p.read_text(encoding='utf-8'))
            if isinstance(data, list) and len(data) == n_places:
                return [float(x) for x in data]
        raise ValueError("Invalid --weights argument: must be comma list or path to json list")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pnml', help='pnml path')
    ap.add_argument('--weights', help='comma-separated weights or path to json list', default=None)
    ap.add_argument('--enum_limit', type=int, default=10000, help='0 to disable enumeration and force ILP+BDD')
    ap.add_argument('--out', help='output json path', default=None)
    ap.add_argument('--mem', action='store_true', help='measure memory during operations')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--max_iters', type=int, default=200)
    args = ap.parse_args()

    pnml_path = Path(args.pnml)
    if not pnml_path.exists():
        print("PNML file not found:", pnml_path)
        return

    if args.enum_limit < 0:
        print("ERROR: --enum_limit must be non-negative.")
        return

    try:
        net = parse_pnml(str(pnml_path))
        validate_net(net)
    except Exception as e:
        print("Error parsing/validating PNML:", e)
        return

    n_places = len(net['places'])
    try:
        c = parse_weights_arg(args.weights, n_places)
    except Exception as e:
        print("Error parsing weights:", e)
        return

    enum_tracker = start_memory_tracking() if args.mem else {}
    t0 = time.perf_counter()
    enum_list = bfs_enumerate(net, enum_limit=(None if args.enum_limit == 0 else args.enum_limit))
    t1 = time.perf_counter()
    enum_mem = stop_memory_tracking(enum_tracker) if args.mem else {}

    if enum_list is not None:
        best_val = None
        best_mark = None
        for m in enum_list:
            val = sum(ci * mi for ci, mi in zip(c, m))
            if best_val is None or val > best_val:
                best_val = val
                best_mark = list(m)
        result = {
            'method': 'enumeration',
            'optimal_value': best_val,
            'optimal_marking': best_mark,
            'enumerated': len(enum_list),
            'time_s': time.perf_counter() - t0,
            'enum_memory': enum_mem
        }
        out_path = args.out or (pnml_path.with_suffix('.opt.json').name)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        if args.verbose:
            print("Enumeration mode result:", result)
        print("Optimization (enumeration) written to", out_path)
        return

    # ILP+BDD fallback
    bdd_tracker = start_memory_tracking() if args.mem else {}
    t2 = time.perf_counter()
    try:
        bdd_mgr, reachable_bdd, old_vars, new_vars = compute_reachable_bdd(net, verbose=args.verbose)
    except Exception as e:
        print("BDD error while building reachable set:", e)
        return
    t3 = time.perf_counter()
    bdd_mem = stop_memory_tracking(bdd_tracker) if args.mem else {}

    ilp_tracker = start_memory_tracking() if args.mem else {}
    t4 = time.perf_counter()
    try:
        ilp_res = ilp_bdd_maximize(net, bdd_mgr, reachable_bdd, c_vector=c, max_iters=args.max_iters, verbose=args.verbose)
    except Exception as e:
        print("ILP error:", e)
        return
    t5 = time.perf_counter()
    ilp_mem = stop_memory_tracking(ilp_tracker) if args.mem else {}

    ilp_res['method'] = 'ilp_bdd_loop'
    ilp_res['bdd_build_time_s'] = t3 - t2
    ilp_res['bdd_build_memory'] = bdd_mem
    ilp_res['ilp_memory'] = ilp_mem
    ilp_res['total_time_s'] = t5 - t0

    out_path = args.out or (pnml_path.with_suffix('.opt.json').name)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(ilp_res, f, indent=2)
    if args.verbose:
        print("ILP+BDD result:", ilp_res)
    print("Optimization (ILP+BDD) written to", out_path)

if __name__ == '__main__':
    main()
