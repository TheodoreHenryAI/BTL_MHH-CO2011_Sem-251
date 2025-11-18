# src/pnml_parser.py
import xml.etree.ElementTree as ET
import json
import sys
from pathlib import Path
import argparse
import traceback

def local_name(tag):
    if tag is None:
        return None
    return tag.split('}')[-1] if '}' in tag else tag

def find_child_text(parent, child_local_names):
    for c in parent:
        if local_name(c.tag) in child_local_names:
            # direct text
            if (c.text is not None) and c.text.strip():
                return c.text.strip()
            # nested text nodes
            for g in c:
                if (g.text is not None) and g.text.strip():
                    return g.text.strip()
    return None

def parse_pnml(path: str, verbose: bool = False):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PNML file not found: {path}")
    if verbose:
        print(f"[parse_pnml] reading file: {p.resolve()}")
    try:
        tree = ET.parse(str(p))
    except ET.ParseError as e:
        # include first 200 chars to help debugging
        snippet = p.read_text(encoding='utf-8', errors='replace')[:200]
        raise ValueError(f"Failed to parse PNML XML: {e}\nFile snippet (first 200 chars):\n{snippet}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to open/parse PNML: {e}") from e

    root = tree.getroot()

    place_elems = []
    trans_elems = []
    arc_elems = []

    # collect elements: tolerant to PNML variants
    for elem in root.iter():
        ln = local_name(elem.tag)
        if ln == 'place':
            place_elems.append(elem)
        elif ln == 'transition':
            trans_elems.append(elem)
        elif ln == 'arc':
            arc_elems.append(elem)

    if verbose:
        print(f"[parse_pnml] found places={len(place_elems)} transitions={len(trans_elems)} arcs={len(arc_elems)}")

    # build places list with indices assigned in discovery order
    places = []
    place_id_to_index = {}
    for i, pe in enumerate(place_elems):
        pid = pe.get('id') or f"p_{i}"
        name = find_child_text(pe, ['name', 'label']) or pid
        init_text = find_child_text(pe, ['initialMarking', 'initialMark', 'initial', 'marking'])
        init_val = 0
        if init_text:
            try:
                init_val = int(init_text.strip())
            except Exception:
                digits = ''.join(ch for ch in init_text if ch.isdigit())
                init_val = int(digits) if digits else 0
        # ensure unique id if duplicates
        if pid in place_id_to_index:
            pid = f"{pid}_{i}"
            if verbose:
                print(f"[parse_pnml] duplicate place id detected; renamed to {pid}")
        places.append({'id': pid, 'index': i, 'name': name, 'initial': int(bool(init_val))})
        place_id_to_index[pid] = i

    # transitions
    transitions = []
    trans_id_to_index = {}
    for i, te in enumerate(trans_elems):
        tid = te.get('id') or f"t_{i}"
        name = find_child_text(te, ['name', 'label']) or tid
        if tid in trans_id_to_index:
            tid = f"{tid}_{i}"
            if verbose:
                print(f"[parse_pnml] duplicate transition id detected; renamed to {tid}")
        transitions.append({'id': tid, 'index': i, 'name': name})
        trans_id_to_index[tid] = i

    # pre/post mappings from arcs
    pre = {t['index']: set() for t in transitions}
    post = {t['index']: set() for t in transitions}

    for arc in arc_elems:
        src = arc.get('source')
        tgt = arc.get('target')
        if not src or not tgt:
            if verbose:
                print(f"[parse_pnml] warning: arc with missing source/target ignored.")
            continue
        # arc from place -> transition
        if src in place_id_to_index and tgt in trans_id_to_index:
            t_i = trans_id_to_index[tgt]
            p_i = place_id_to_index[src]
            pre[t_i].add(p_i)
        # arc from transition -> place
        elif src in trans_id_to_index and tgt in place_id_to_index:
            t_i = trans_id_to_index[src]
            p_i = place_id_to_index[tgt]
            post[t_i].add(p_i)
        else:
            # arc references unknown id: warn and skip
            if verbose:
                print(f"[parse_pnml] warning: arc references unknown ids: src={src} tgt={tgt}")

    transitions_out = []
    for t in transitions:
        idx = t['index']
        transitions_out.append({
            'id': t['id'],
            'index': idx,
            'name': t['name'],
            'pre': sorted(list(pre.get(idx, []))),
            'post': sorted(list(post.get(idx, [])))
        })

    initial_marking = [p['initial'] for p in places]

    net = {
        'places': places,
        'transitions': transitions_out,
        'initial_marking': initial_marking
    }
    if verbose:
        print(f"[parse_pnml] parsed net: places={len(places)} transitions={len(transitions_out)}")
    return net

def main():
    ap = argparse.ArgumentParser(description="PNML parser -> JSON (verbose and robust)")
    ap.add_argument('pnml', help='path to pnml file')
    ap.add_argument('--out', help='output json path', default=None)
    ap.add_argument('--verbose', action='store_true', help='print debug messages')
    args = ap.parse_args()

    try:
        p = Path(args.pnml)
        if not p.exists():
            print(f"ERROR: PNML file not found: {p}", file=sys.stderr)
            sys.exit(2)

        net = parse_pnml(str(p), verbose=args.verbose)
        out_path = Path(args.out) if args.out else p.with_suffix('.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(net, f, indent=2)
        print(f"Parsed net written to {out_path}")
        print(f"Places: {len(net['places'])}, Transitions: {len(net['transitions'])}")
    except Exception as e:
        # print full traceback for debugging
        print("ERROR: PNML parsing failed:", file=sys.stderr)
        print(str(e), file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        sys.exit(3)

if __name__ == '__main__':
    main()
