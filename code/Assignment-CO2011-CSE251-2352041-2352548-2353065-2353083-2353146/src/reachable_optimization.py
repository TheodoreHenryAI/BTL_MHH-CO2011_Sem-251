import argparse
import json
import time
from pathlib import Path
import sys

# ensure project root is importable, and import pnml_parser robustly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src import pnml_parser as parser_module
except Exception:
    import importlib.util
    spec = importlib.util.spec_from_file_location("pnml_parser", str(ROOT / "src" / "pnml_parser.py"))
    pnml = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pnml)
    parser_module = pnml

# dd BDD library
try:
    from dd import autoref as _bdd
except Exception:
    raise RuntimeError("Missing 'dd' library. Install with: pip install dd")

# ILP library
try:
    import pulp
except Exception:
    raise RuntimeError("Missing 'pulp' library. Install with: pip install pulp")

# -------------------------
# Helper BDD / PNML helpers
# -------------------------
def varname(p):
    return f"x{p}"

def varname_prime(p):
    return f"x{p}p"

def build_transition_relation(bdd, n_places, transition):
    """
    Build relation R_t for a single transition (same semantics as earlier scripts).
    """
    # enabled = AND of pre variables
    if len(transition['pre']) == 0:
        enabled = bdd.true
    else:
        piles = [bdd.var(varname(p)) for p in transition['pre']]
        enabled = piles[0]
        for u in piles[1:]:
            enabled = enabled & u

    eqs = bdd.true
    pre_set = set(transition['pre'])
    post_set = set(transition['post'])
    for p in range(n_places):
        xp = bdd.var(varname_prime(p))
        if p in post_set:
            expr = bdd.true
        else:
            if p in pre_set:
                expr = bdd.false
            else:
                expr = bdd.var(varname(p))
        # XNOR: (xp & expr) | (~xp & ~expr)
        xnor = (xp & expr) | (~xp & ~expr)
        eqs = eqs & xnor

    return enabled & eqs

def marking_to_bdd(bdd, marking):
    conj = bdd.true
    for i,bit in enumerate(marking):
        v = bdd.var(varname(i))
        conj = conj & v if int(bit) else conj & ~v
    return conj

def compute_reachable_bdd(net, verbose=False):
    """
    Build BDD manager and compute reachable set as a BDD.
    Returns (bdd_mgr, reachable_bdd)
    """
    bdd = _bdd.BDD()
    places = net['places']
    transitions = net['transitions']
    n = len(places)

    # declare variables (old then new)
    old_vars = [varname(i) for i in range(n)]
    new_vars = [varname_prime(i) for i in range(n)]
    bdd.declare(*old_vars, *new_vars)

    # build transition relation R = OR_t R_t
    R = bdd.false
    for t in transitions:
        Rt = build_transition_relation(bdd, n, t)
        R = R | Rt

    # initial
    start = tuple(int(x) for x in net['initial_marking'])
    reachable = marking_to_bdd(bdd, start)

    # fixpoint loop
    iteration = 0
    while True:
        iteration += 1
        if verbose:
            print(f"[BDD iter {iteration}] computing image...")
        conj = reachable & R
        img_new = bdd.exist(old_vars, conj)  # exists over old vars -> BDD on new vars
        rename_map = { varname_prime(i): varname(i) for i in range(n) }
        img_old = bdd.let(rename_map, img_new)
        next_reachable = reachable | img_old
        if next_reachable == reachable:
            if verbose:
                print(f"[BDD iter {iteration}] fixpoint reached.")
            break
        reachable = next_reachable
    return bdd, reachable

def bdd_enumerate_markings(bdd_mgr, f, n_places, limit=None):
    """
    Enumerate satisfying assignments of BDD f to lists of 0/1. Limit caps enumeration.
    """
    res = []
    try:
        it = f.pick_iter()
    except Exception:
        return res
    count = 0
    for assn in it:
        mark = [0]*n_places
        for p in range(n_places):
            name = varname(p)
            if name in assn and assn[name]:
                mark[p] = 1
            else:
                mark[p] = 0
        res.append(mark)
        count += 1
        if limit and count >= limit:
            break
    return res

# -------------------------
# Explicit BFS (small, copied inline)
# -------------------------
from collections import deque
def bfs_enumerate(net):
    places = net['places']
    transitions = net['transitions']
    n = len(places)
    start = tuple(int(x) for x in net['initial_marking'])

    def is_fireable(t, marking):
        for p in t['pre']:
            if marking[p] == 0:
                return False
        return True

    def fire(t, marking):
        m = list(marking)
        for p in t['pre']: m[p] = 0
        for p in t['post']: m[p] = 1
        return tuple(m)

    seen = set([start])
    q = deque([start])
    order = [start]
    while q:
        m = q.popleft()
        for t in transitions:
            if is_fireable(t, m):
                m2 = fire(t, m)
                if m2 not in seen:
                    seen.add(m2)
                    q.append(m2)
                    order.append(m2)
    return [list(m) for m in order]

# -------------------------
# ILP + BDD loop for optimization
# -------------------------
def ilp_bdd_optimize(net, bdd_mgr, reachable_bdd, c_vec, max_iters=200, verbose=False):
    """
    ILP loop:
      - Solve ILP maximize c·m (m binary)
      - If candidate reachable (via BDD) return it
      - else forbid exact marking and repeat
    """
    n = len(net['places'])
    forbidden = []
    iteration = 0
    while iteration < max_iters:
        iteration += 1
        if verbose:
            print(f"[ILP iter {iteration}] building ILP...")
        prob = pulp.LpProblem("opt_marking", pulp.LpMaximize)
        m_vars = [pulp.LpVariable(f"m_{i}", cat='Binary') for i in range(n)]

        # objective
        prob += pulp.lpSum([c_vec[i] * m_vars[i] for i in range(n)])

        # forbidden exact-marking cuts
        for ones, zeros in forbidden:
            # sum_{p in ones} (1 - m_p) + sum_{p in zeros} m_p >= 1
            prob += pulp.lpSum([(1 - m_vars[p]) for p in ones] + [m_vars[p] for p in zeros]) >= 1

        # solve
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[prob.status] != 'Optimal':
            if verbose:
                print("ILP infeasible or no optimal solution.")
            return {'found': False, 'reason': 'ilp_infeasible', 'iters': iteration}

        candidate = [int(round(pulp.value(v))) for v in m_vars]
        # compute candidate objective
        obj_val = sum(c_vec[i]*candidate[i] for i in range(n))

        # check reachability via BDD
        cand_bdd = marking_to_bdd(bdd_mgr, candidate)
        if (reachable_bdd & cand_bdd) != bdd_mgr.false:
            return {'found': True, 'marking': candidate, 'value': obj_val, 'iters': iteration}
        else:
            # forbid exact candidate and continue
            ones = [i for i,b in enumerate(candidate) if b==1]
            zeros = [i for i,b in enumerate(candidate) if b==0]
            forbidden.append((ones, zeros))
            if verbose:
                print(f"[ILP iter {iteration}] candidate {candidate} (value={obj_val}) not reachable; forbidding and retrying.")
    return {'found': False, 'reason': 'max_iters_exceeded', 'iters': iteration}

# -------------------------
# Main driver
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pnml', help='path to pnml file')
    ap.add_argument('--coeffs', nargs='*', type=float, help='objective coefficients c0 c1 ... (length must equal #places). Default: all ones')
    ap.add_argument('--out', help='output json path', default=None)
    ap.add_argument('--enum_limit', type=int, default=10000, help='cap for full enumeration')
    ap.add_argument('--max_ilp_iters', type=int, default=200, help='max ILP iterations when using ILP loop')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    pnml_path = Path(args.pnml)
    if not pnml_path.exists():
        print("ERROR: PNML file not found:", pnml_path)
        return

    net = parser_module.parse_pnml(str(pnml_path))
    n = len(net['places'])

    # coefficients
    if args.coeffs:
        if len(args.coeffs) != n:
            print("ERROR: number of coefficients must match number of places (n=%d)" % n)
            return
        c_vec = [float(x) for x in args.coeffs]
    else:
        c_vec = [1.0]*n  # maximize total tokens by default

    # compute reachable set symbolically (BDD)
    t0 = time.perf_counter()
    bdd_mgr, reachable_bdd = compute_reachable_bdd(net, verbose=args.verbose)
    t1 = time.perf_counter()
    bdd_time = t1 - t0

    # attempt to enumerate reachable markings up to enum_limit
    t2 = time.perf_counter()
    markings = bdd_enumerate_markings(bdd_mgr, reachable_bdd, n, limit=args.enum_limit+1)
    t3 = time.perf_counter()
    enum_time = t3 - t2

    result = {
        'places': [{'id':p['id'],'index':p['index'],'name':p.get('name')} for p in net['places']],
        'transitions': [{'id':t['id'],'index':t['index'],'name':t.get('name')} for t in net['transitions']],
        'initial_marking': list(net['initial_marking']),
        'coeffs': c_vec,
        'bdd_time_s': bdd_time,
        'enum_time_s': enum_time
    }

    # If fully enumerated (<= enum_limit)
    if len(markings) <= args.enum_limit:
        # compute optimum by enumeration
        best = None
        best_val = None
        for m in markings:
            val = sum(c_vec[i]*m[i] for i in range(n))
            if best is None or val > best_val:
                best = m
                best_val = val
        result.update({
            'method': 'enumeration',
            'reachable_count': len(markings),
            'best_marking': best,
            'best_value': best_val,
            'all_reachable_markings': markings
        })
    else:
        # need ILP+BDD loop (do not enumerate)
        if args.verbose:
            print(f"Reachable set too large to enumerate (>{args.enum_limit}). Using ILP+BDD loop.")
        t4 = time.perf_counter()
        ilp_res = ilp_bdd_optimize(net, bdd_mgr, reachable_bdd, c_vec, max_iters=args.max_ilp_iters, verbose=args.verbose)
        t5 = time.perf_counter()
        ilp_time = t5 - t4
        result.update({
            'method': 'ilp_bdd_loop',
            'reachable_count': None,
            'best_marking': ilp_res.get('marking'),
            'best_value': ilp_res.get('value'),
            'ilp_result': ilp_res,
            'ilp_time_s': ilp_time
        })

    out_path = args.out or (pnml_path.with_suffix('.opt.json').name)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print("Optimization result written to", out_path)
    print("Method:", result['method'], "Best value:", result.get('best_value'), "Best marking:", result.get('best_marking'))

if __name__ == '__main__':
    main()