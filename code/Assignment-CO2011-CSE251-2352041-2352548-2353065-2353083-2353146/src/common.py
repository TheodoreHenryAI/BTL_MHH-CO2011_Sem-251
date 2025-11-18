# src/common.py
import sys
from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# centralized PNML parser import (single place)
try:
    from src import pnml_parser as _pnml_parser
except Exception:
    import importlib.util
    spec = importlib.util.spec_from_file_location("pnml_parser", str(ROOT / "src" / "pnml_parser.py"))
    _pnml_parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_pnml_parser)

parse_pnml = getattr(_pnml_parser, "parse_pnml")

def validate_net(net: Dict[str, Any]) -> None:
    if not isinstance(net, dict):
        raise ValueError("Parsed net must be a dict.")

    places = net.get('places')
    transitions = net.get('transitions')
    initial = net.get('initial_marking')

    if places is None or transitions is None or initial is None:
        raise ValueError("Parsed net missing required keys: 'places', 'transitions', or 'initial_marking'.")

    if not isinstance(places, list):
        raise ValueError("'places' must be a list.")
    if not isinstance(transitions, list):
        raise ValueError("'transitions' must be a list.")
    if not isinstance(initial, (list, tuple)):
        raise ValueError("'initial_marking' must be a list or tuple.")

    n_places = len(places)
    if n_places == 0:
        raise ValueError("The net contains zero places; at least one place is required.")

    # check unique ids and index continuity for places
    place_ids = set()
    place_indices = set()
    for p in places:
        pid = p.get('id')
        idx = p.get('index')
        if pid in place_ids:
            raise ValueError(f"Duplicate place id found: {pid}")
        place_ids.add(pid)
        if not isinstance(idx, int) or idx < 0:
            raise ValueError(f"Invalid place index for place {pid}: {idx}")
        place_indices.add(idx)
    if place_indices != set(range(n_places)):
        raise ValueError(f"Place indices must be 0..{n_places-1}. Found indices: {sorted(place_indices)}")

    # check unique ids and index continuity for transitions
    trans_ids = set()
    trans_indices = set()
    for t in transitions:
        tid = t.get('id')
        idx = t.get('index')
        if tid in trans_ids:
            raise ValueError(f"Duplicate transition id found: {tid}")
        trans_ids.add(tid)
        if not isinstance(idx, int) or idx < 0:
            raise ValueError(f"Invalid transition index for transition {tid}: {idx}")
        trans_indices.add(idx)
    if trans_indices and trans_indices != set(range(len(transitions))):
        raise ValueError(f"Transition indices must be 0..{len(transitions)-1}. Found indices: {sorted(trans_indices)}")

    # initial marking length & values
    if len(initial) != n_places:
        raise ValueError(f"Initial marking length ({len(initial)}) does not match number of places ({n_places}).")
    for i, v in enumerate(initial):
        if v not in (0, 1, True, False):
            raise ValueError(f"Initial marking value for place index {i} must be 0 or 1 (found: {v}).")

    # check pre/post references
    for t in transitions:
        pre = t.get('pre', [])
        post = t.get('post', [])
        if not isinstance(pre, list) or not isinstance(post, list):
            raise ValueError(f"Transition {t.get('id')} has invalid pre/post structure (must be lists).")
        for p in pre + post:
            if not isinstance(p, int) or p < 0 or p >= n_places:
                raise ValueError(f"Transition {t.get('id')} references invalid place index {p} (valid 0..{n_places-1}).")

def varname(p: int) -> str:
    return f"x{p}"

def varname_prime(p: int) -> str:
    return f"x{p}p"

def build_transition_relation(bdd, n_places: int, transition: dict):
    if len(transition.get('pre', [])) == 0:
        enabled = bdd.true
    else:
        piles = [bdd.var(varname(p)) for p in transition['pre']]
        enabled = piles[0]
        for u in piles[1:]:
            enabled = enabled & u

    eqs = bdd.true
    pre_set = set(transition.get('pre', []))
    post_set = set(transition.get('post', []))
    for p in range(n_places):
        xp = bdd.var(varname_prime(p))
        if p in post_set:
            expr = bdd.true
        else:
            if p in pre_set:
                expr = bdd.false
            else:
                expr = bdd.var(varname(p))
        xnor = (xp & expr) | (~xp & ~expr)
        eqs = eqs & xnor

    return enabled & eqs

def marking_to_bdd(bdd, old_var_names, marking):
    node = bdd.true
    for name, bit in zip(old_var_names, marking):
        v = bdd.var(name)
        node = node & v if int(bit) else node & ~v
    return node

def enumerate_markings_from_bdd(bdd, node, old_var_names, limit=None):
    res = []
    work = node
    count = 0

    # quick check
    if work == bdd.false:
        return []

    while True:
        try:
            assn = work.pick()  # picks one satisfying assignment (returns dict var->bool)
        except Exception:
            # if pick() fails, return what we have
            break
        if not assn:
            break
        # build full marking in specified order
        mark = [1 if assn.get(var, False) else 0 for var in old_var_names]
        res.append(mark)
        count += 1
        if limit and count >= limit:
            break
        # build blocking cube for this assignment: cube = AND_{v in old_var_names} (v if bit==1 else ~v)
        cube = bdd.true
        for var, bit in zip(old_var_names, mark):
            vnode = bdd.var(var)
            cube = cube & (vnode if bit else ~vnode)
        # exclude this exact assignment from work
        work = work & ~cube
        if work == bdd.false:
            break

    return res

def compute_reachable_bdd(net: Dict[str, Any], var_order: str = "interleaved", verbose: bool = False):
    """
    Build a BDD manager and compute the reachable set as a BDD.
    Returns (bdd_manager, reachable_bdd, old_var_names, new_var_names).
    """
    validate_net(net)  # raise early if invalid

    try:
        from dd import autoref as _bdd
    except Exception as exc:
        raise RuntimeError("Missing required package 'dd'. Install with: python -m pip install dd") from exc

    places = net['places']
    transitions = net['transitions']
    n = len(places)

    if var_order == "interleaved":
        ordered = []
        for i in range(n):
            ordered.append(varname(i))
            ordered.append(varname_prime(i))
    else:
        ordered = [varname(i) for i in range(n)] + [varname_prime(i) for i in range(n)]

    bdd = _bdd.BDD()
    bdd.declare(*ordered)

    old_vars = [varname(i) for i in range(n)]
    new_vars = [varname_prime(i) for i in range(n)]

    if verbose:
        print("BDD declared vars:", ordered)

    # build per-transition relations list
    R_list = []
    for t in transitions:
        Rt = build_transition_relation(bdd, n, t)
        R_list.append(Rt)

    # initial marking
    start = tuple(int(x) for x in net['initial_marking'])
    reachable = marking_to_bdd(bdd, old_vars, start)

    # fixpoint (per-transition image)
    iteration = 0
    while True:
        iteration += 1
        if verbose:
            print(f"[BDD iter {iteration}] computing image...")
        img_old_total = bdd.false
        for Rt in R_list:
            conj = reachable & Rt
            img_new = bdd.exist(old_vars, conj)
            rename_map = {varname_prime(i): varname(i) for i in range(n)}
            img_old = bdd.let(rename_map, img_new)
            img_old_total = img_old_total | img_old
        next_reachable = reachable | img_old_total
        if next_reachable == reachable:
            if verbose:
                print(f"[BDD iter {iteration}] fixpoint reached.")
            break
        reachable = next_reachable

    return bdd, reachable, old_vars, new_vars
