#!/usr/bin/env python3

import argparse
import sys
from typing import Tuple, Optional

import h5py
import numpy as np


DATASETS = ("/field", "/step")


def format_index(idx_tuple) -> str:
    return "(" + ", ".join(str(int(x)) for x in idx_tuple) + ")"


def first_mismatch_index_exact(a: np.ndarray, b: np.ndarray) -> Optional[Tuple[int, ...]]:
    diff = (a != b)
    if not np.any(diff):
        return None
    return tuple(np.argwhere(diff)[0])


def first_mismatch_index_close(
    a: np.ndarray,
    b: np.ndarray,
    rtol: float,
    atol: float
) -> Optional[Tuple[int, ...]]:
    close = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    diff = ~close
    if not np.any(diff):
        return None
    return tuple(np.argwhere(diff)[0])


def max_abs_rel_diff(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    abs_diff = np.abs(a - b)
    max_abs = float(np.max(abs_diff)) if abs_diff.size else 0.0

    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), np.finfo(np.float64).tiny)
    rel_diff = abs_diff / denom
    max_rel = float(np.max(rel_diff)) if rel_diff.size else 0.0

    return max_abs, max_rel


def compare_scalar(
    a,
    b,
    name: str,
    rtol: float,
    atol: float,
    exact: bool
) -> bool:
    if exact:
        if a != b:
            print(f"[FAIL] {name}: scalar mismatch: {a} vs {b}")
            return False
        print(f"[ OK ] {name}: exact scalar match")
        return True

    if not np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
        abs_diff = abs(a - b)
        denom = max(abs(a), abs(b), np.finfo(np.float64).tiny)
        rel_diff = abs_diff / denom
        print(f"[FAIL] {name}: scalar mismatch: {a} vs {b}")
        print(f"       abs_diff={abs_diff:.17g}, rel_diff={rel_diff:.17g}")
        return False

    print(f"[ OK ] {name}: scalar matched within tolerance")
    return True


def compare_dataset(
    ds_a: h5py.Dataset,
    ds_b: h5py.Dataset,
    name: str,
    rtol: float,
    atol: float,
    exact: bool,
    chunksize: int,
) -> bool:
    if ds_a.shape != ds_b.shape:
        print(f"[FAIL] {name}: shape mismatch: {ds_a.shape} vs {ds_b.shape}")
        return False

    if ds_a.dtype != ds_b.dtype:
        print(f"[FAIL] {name}: dtype mismatch: {ds_a.dtype} vs {ds_b.dtype}")
        return False

    shape = ds_a.shape
    ndim = len(shape)

    mode = "exact" if exact else f"rtol={rtol}, atol={atol}"
    print(f"[INFO] {name}: shape={shape}, dtype={ds_a.dtype}, mode={mode}")

    # Scalar dataset
    if ndim == 0:
        a = ds_a[()]
        b = ds_b[()]
        return compare_scalar(a, b, name, rtol, atol, exact)

    # Compare chunk-by-chunk along first axis
    n0 = shape[0]
    for start in range(0, n0, chunksize):
        stop = min(start + chunksize, n0)
        sl = (slice(start, stop),) + (slice(None),) * (ndim - 1)

        a = ds_a[sl]
        b = ds_b[sl]

        if exact:
            if not np.array_equal(a, b):
                local_idx = first_mismatch_index_exact(a, b)
                global_idx = (start + local_idx[0],) + local_idx[1:]
                print(f"[FAIL] {name}: exact mismatch at index {format_index(global_idx)}")
                print(f"       values: {a[local_idx]} vs {b[local_idx]}")
                return False
        else:
            if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
                local_idx = first_mismatch_index_close(a, b, rtol, atol)
                global_idx = (start + local_idx[0],) + local_idx[1:]

                max_abs, max_rel = max_abs_rel_diff(a, b)

                print(f"[FAIL] {name}: mismatch at index {format_index(global_idx)}")
                print(f"       values: {a[local_idx]} vs {b[local_idx]}")
                print(f"       chunk max_abs_diff={max_abs:.17g}, chunk max_rel_diff={max_rel:.17g}")
                return False

    if np.issubdtype(ds_a.dtype, np.floating) and not exact:
        print(f"[ OK ] {name}: matched within tolerance")
    else:
        print(f"[ OK ] {name}: exact match")

    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two cooling solver HDF5 outputs dataset-by-dataset."
    )
    parser.add_argument("file_a", help="Reference file (e.g. CUDA output)")
    parser.add_argument("file_b", help="Candidate file (e.g. Python/Numba output)")
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-12,
        help="Relative tolerance for /field. Default: 1e-12"
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-14,
        help="Absolute tolerance for /field. Default: 1e-14"
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=4,
        help="Number of frames to compare at once. Default: 4"
    )
    parser.add_argument(
        "--exact-field",
        action="store_true",
        help="Require exact equality also for /field"
    )
    parser.add_argument(
        "--ignore-dtype",
        action="store_true",
        help="Ignore dtype mismatch and compare only values"
    )

    args = parser.parse_args()

    if args.chunksize <= 0:
        print("ERROR: --chunksize must be > 0", file=sys.stderr)
        return 2

    try:
        with h5py.File(args.file_a, "r") as fa, h5py.File(args.file_b, "r") as fb:
            missing = [name for name in DATASETS if name not in fa or name not in fb]
            if missing:
                for name in missing:
                    print(f"[FAIL] Missing dataset in one of the files: {name}")
                return 1

            overall_ok = True

            for name in DATASETS:
                ds_a = fa[name]
                ds_b = fb[name]

                if ds_a.shape != ds_b.shape:
                    print(f"[FAIL] {name}: shape mismatch: {ds_a.shape} vs {ds_b.shape}")
                    overall_ok = False
                    continue

                if (ds_a.dtype != ds_b.dtype) and (not args.ignore_dtype):
                    print(f"[FAIL] {name}: dtype mismatch: {ds_a.dtype} vs {ds_b.dtype}")
                    overall_ok = False
                    continue

                # If dtypes differ but user wants to ignore them, compare through numpy arrays
                # using temporary in-memory wrappers via slicing logic inside compare_dataset-like flow.
                if args.ignore_dtype and ds_a.dtype != ds_b.dtype:
                    print(f"[INFO] {name}: dtype mismatch ignored: {ds_a.dtype} vs {ds_b.dtype}")

                if name == "/step":
                    # Steps should match exactly
                    ok = compare_dataset(
                        ds_a, ds_b, name,
                        rtol=0.0,
                        atol=0.0,
                        exact=True,
                        chunksize=args.chunksize
                    )
                elif name == "/field":
                    if args.ignore_dtype and ds_a.dtype != ds_b.dtype:
                        # Custom comparison path when dtype differs but values should still be compared
                        shape = ds_a.shape
                        ndim = len(shape)
                        mode = "exact" if args.exact_field else f"rtol={args.rtol}, atol={args.atol}"
                        print(f"[INFO] {name}: shape={shape}, dtype={ds_a.dtype} vs {ds_b.dtype}, mode={mode}")

                        ok = True
                        n0 = shape[0]
                        for start in range(0, n0, args.chunksize):
                            stop = min(start + args.chunksize, n0)
                            sl = (slice(start, stop),) + (slice(None),) * (ndim - 1)

                            a = np.asarray(ds_a[sl], dtype=np.float64)
                            b = np.asarray(ds_b[sl], dtype=np.float64)

                            if args.exact_field:
                                if not np.array_equal(a, b):
                                    local_idx = first_mismatch_index_exact(a, b)
                                    global_idx = (start + local_idx[0],) + local_idx[1:]
                                    print(f"[FAIL] {name}: exact mismatch at index {format_index(global_idx)}")
                                    print(f"       values: {a[local_idx]} vs {b[local_idx]}")
                                    ok = False
                                    break
                            else:
                                if not np.allclose(a, b, rtol=args.rtol, atol=args.atol, equal_nan=True):
                                    local_idx = first_mismatch_index_close(a, b, args.rtol, args.atol)
                                    global_idx = (start + local_idx[0],) + local_idx[1:]
                                    max_abs, max_rel = max_abs_rel_diff(a, b)
                                    print(f"[FAIL] {name}: mismatch at index {format_index(global_idx)}")
                                    print(f"       values: {a[local_idx]} vs {b[local_idx]}")
                                    print(f"       chunk max_abs_diff={max_abs:.17g}, chunk max_rel_diff={max_rel:.17g}")
                                    ok = False
                                    break

                        if ok:
                            if args.exact_field:
                                print(f"[ OK ] {name}: exact match")
                            else:
                                print(f"[ OK ] {name}: matched within tolerance")
                    else:
                        ok = compare_dataset(
                            ds_a, ds_b, name,
                            rtol=args.rtol,
                            atol=args.atol,
                            exact=args.exact_field,
                            chunksize=args.chunksize
                        )
                else:
                    print(f"[FAIL] Unexpected dataset: {name}")
                    ok = False

                overall_ok = overall_ok and ok

            if overall_ok:
                print("\nSUCCESS: files match for all checked datasets.")
                return 0
            else:
                print("\nFAILURE: files differ.")
                return 1

    except OSError as e:
        print(f"ERROR: cannot open file: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

