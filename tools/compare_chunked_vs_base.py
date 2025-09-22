import argparse
import glob
import os
import pickle
import re
from collections import defaultdict

import numpy as np


KV_RE = re.compile(r"layer(?P<layer>\d+)_cp(?P<cp>\d+)_sp(?P<tp>\d+)_kv_prefill_.*\\.pkl$")
ATTN_RE = re.compile(r"layer(?P<layer>\d+)_cp(?P<cp>\d+)_sp(?P<tp>\d+)_attn_prefill_out_.*\\.pkl$")


def load_pkls(root: str, pattern: str):
    files = sorted(glob.glob(os.path.join(root, pattern)))
    blobs = []
    for f in files:
        try:
            with open(f, "rb") as fh:
                blobs.append((f, pickle.load(fh)))
        except Exception as e:
            print(f"WARN: failed to load {f}: {e}")
    return blobs


def group_by_layer(blobs, regex: re.Pattern):
    grouped = defaultdict(list)
    for path, obj in blobs:
        m = regex.search(os.path.basename(path))
        if not m:
            continue
        layer = int(m.group("layer"))
        cp = int(m.group("cp"))
        tp = int(m.group("tp"))
        grouped[layer].append((cp, tp, path, obj))
    return grouped


def compare_arrays(a: np.ndarray, b: np.ndarray, name: str, atol=1e-2, rtol=1e-2):
    ok = np.allclose(a, b, atol=atol, rtol=rtol)
    diff = np.abs(a - b)
    print(f"  {name}: shapeA={a.shape} shapeB={b.shape} max={diff.max():.6f} mean={diff.mean():.6f} ok={ok}")
    return ok


def assemble_attn(group):
    """Assemble attention output by concatenating across CP (tokens) then TP (heads).

    group: list of (cp, tp, path, obj)
    obj expects keys: out_shape, out_fp32_head (np.array)
    """
    by_tp = defaultdict(list)
    for cp, tp, path, obj in group:
        arr = obj.get("out_fp32_head")
        if arr is None:
            continue
        by_tp[tp].append((cp, arr))

    # concat CP by cp rank order along token dimension
    cp_concat_per_tp = {}
    for tp, items in by_tp.items():
        items_sorted = [arr for _, arr in sorted(items, key=lambda x: x[0])]
        cp_concat_per_tp[tp] = np.concatenate(items_sorted, axis=0) if items_sorted else None

    # concat TP by tp rank order along head dimension (last dim or middle dim depending on layout)
    # Our dump stores flattened [tokens, heads*V] rows, so concat along last dim
    tps = sorted(cp_concat_per_tp.keys())
    if not tps:
        return None
    arrays = [cp_concat_per_tp[t] for t in tps if cp_concat_per_tp[t] is not None]
    if not arrays:
        return None
    return np.concatenate(arrays, axis=-1)


def main():
    ap = argparse.ArgumentParser(description="Compare KV and attention output between base and chunked runs.")
    ap.add_argument("--base_dir", required=True, help="Directory of base (no-chunk) dumps")
    ap.add_argument("--chunked_dir", required=True, help="Directory of chunked dumps")
    ap.add_argument("--compare", choices=["kv", "attn", "both"], default="both")
    ap.add_argument("--layers", type=str, default=None, help="Comma-separated layer ids to compare (default: all)")
    ap.add_argument("--atol", type=float, default=1e-2)
    ap.add_argument("--rtol", type=float, default=1e-2)
    args = ap.parse_args()

    layers_filter = None
    if args.layers:
        layers_filter = set(int(x) for x in args.layers.split(","))

    if args.compare in ("kv", "both"):
        base_kv = group_by_layer(load_pkls(args.base_dir, "*kv_prefill*.pkl"), KV_RE)
        chk_kv = group_by_layer(load_pkls(args.chunked_dir, "*kv_prefill*.pkl"), KV_RE)
        print("=== Compare KV blocks (cp0 only recommended) ===")
        common_layers = sorted(set(base_kv.keys()) & set(chk_kv.keys()))
        for layer in common_layers:
            if layers_filter and layer not in layers_filter:
                continue
            # Prefer cp=0, aggregate multiple tp if present by concatenation along last dim
            def pick_and_concat_kv(group):
                group_cp0 = [(tp, path, obj) for cp, tp, path, obj in group if cp == 0]
                if not group_cp0:
                    return None, None
                # concatenate kv_nope and kv_rope along last dim if tp>1
                nope_list = []
                rope_list = []
                for tp, path, obj in sorted(group_cp0, key=lambda x: x[0]):
                    nope = obj.get("kv_nope_blocks")
                    rope = obj.get("kv_rope_blocks")
                    if nope is not None:
                        nope_list.append(nope)
                    if rope is not None:
                        rope_list.append(rope)
                nope_cat = np.concatenate(nope_list, axis=-1) if len(nope_list) > 1 else (nope_list[0] if nope_list else None)
                rope_cat = np.concatenate(rope_list, axis=-1) if len(rope_list) > 1 else (rope_list[0] if rope_list else None)
                return nope_cat, rope_cat

            base_nope, base_rope = pick_and_concat_kv(base_kv[layer])
            chk_nope, chk_rope = pick_and_concat_kv(chk_kv[layer])
            print(f"Layer {layer}:")
            if base_nope is None or chk_nope is None:
                print("  WARN: missing kv_nope blocks")
            else:
                compare_arrays(base_nope, chk_nope, name="kv_nope", atol=args.atol, rtol=args.rtol)
            if base_rope is None or chk_rope is None:
                print("  WARN: missing kv_rope blocks")
            else:
                compare_arrays(base_rope, chk_rope, name="kv_rope", atol=args.atol, rtol=args.rtol)

    if args.compare in ("attn", "both"):
        base_attn = group_by_layer(load_pkls(args.base_dir, "*attn_prefill_out*.pkl"), ATTN_RE)
        chk_attn = group_by_layer(load_pkls(args.chunked_dir, "*attn_prefill_out*.pkl"), ATTN_RE)
        print("=== Compare prefill attention outputs (assembled CP+TP) ===")
        common_layers = sorted(set(base_attn.keys()) & set(chk_attn.keys()))
        for layer in common_layers:
            if layers_filter and layer not in layers_filter:
                continue
            base_assembled = assemble_attn(base_attn[layer])
            chk_assembled = assemble_attn(chk_attn[layer])
            print(f"Layer {layer}:")
            if base_assembled is None or chk_assembled is None:
                print("  WARN: missing attention dumps")
                continue
            # Align by min length in token dim
            n = min(base_assembled.shape[0], chk_assembled.shape[0])
            compare_arrays(base_assembled[:n], chk_assembled[:n], name="attn_prefill_out", atol=args.atol, rtol=args.rtol)


if __name__ == "__main__":
    main()


