#!/usr/bin/env python3

import argparse
import sys
from typing import Optional, Tuple, Sequence

import h5py
import numpy as np


DEFAULT_DATASETS = ("/step", "/pos", "/vel", "/screen")


def format_index(idx_tuple: Sequence[int]) -> str:
    return "(" + ", ".join(str(int(x)) for x in idx_tuple) + ")"


def dataset_exists(file_handle: h5py.File, name: str) -> bool:
    return name in file_handle and isinstance(file_handle[name], h5py.Dataset)


def first_mismatch_index_exact(a: np.ndarray, b: np.ndarray) -> Optional[Tuple[int, ...]]:
    diff = a != b
    if not np.any(diff):
        return None
    return tuple(int(x) for x in np.argwhere(diff)[0])


def first_mismatch_index_close(
    a: np.ndarray,
    b: np.ndarray,
    rtol: float,
    atol: float,
) -> Optional[Tuple[int, ...]]:
    close = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    diff = ~close
    if not np.any(diff):
        return None
    return tuple(int(x) for x in np.argwhere(diff)[0])


def max_abs_rel_diff(
    a: np.ndarray,
    b: np.ndarray,
) -> Tuple[float, float, Tuple[int, ...], Tuple[int, ...]]:
    """
    Return:
      max_abs_diff,
      max_rel_diff,
      local_index_of_max_abs,
      local_index_of_max_rel
    """
    if a.size == 0:
        return 0.0, 0.0, tuple(), tuple()

    a64 = a.astype(np.float64, copy=False)
    b64 = b.astype(np.float64, copy=False)

    abs_diff = np.abs(a64 - b64)

    denom = np.maximum(
        np.maximum(np.abs(a64), np.abs(b64)),
        np.finfo(np.float64).tiny,
    )

    rel_diff = abs_diff / denom

    max_abs_flat = int(np.argmax(abs_diff))
    max_rel_flat = int(np.argmax(rel_diff))

    max_abs_idx = tuple(int(x) for x in np.unravel_index(max_abs_flat, abs_diff.shape))
    max_rel_idx = tuple(int(x) for x in np.unravel_index(max_rel_flat, rel_diff.shape))

    max_abs = float(abs_diff[max_abs_idx])
    max_rel = float(rel_diff[max_rel_idx])

    return max_abs, max_rel, max_abs_idx, max_rel_idx


def check_finite_dataset(
    ds: h5py.Dataset,
    name: str,
    chunksize: int,
    label: str,
) -> bool:
    if not np.issubdtype(ds.dtype, np.floating):
        return True

    shape = ds.shape

    if len(shape) == 0:
        value = ds[()]
        if not np.isfinite(value):
            print(f"[FAIL] {label}:{name}: non-finite scalar value: {value}")
            return False
        return True

    n0 = shape[0]

    for start in range(0, n0, chunksize):
        stop = min(start + chunksize, n0)
        sl = (slice(start, stop),) + (slice(None),) * (len(shape) - 1)

        data = ds[sl]
        finite = np.isfinite(data)

        if not np.all(finite):
            local_idx = tuple(int(x) for x in np.argwhere(~finite)[0])
            global_idx = (start + local_idx[0],) + local_idx[1:]

            print(f"[FAIL] {label}:{name}: non-finite value at index {format_index(global_idx)}")
            print(f"       value: {data[local_idx]}")
            return False

    return True


def compare_dataset(
    ds_a: h5py.Dataset,
    ds_b: h5py.Dataset,
    name: str,
    rtol: float,
    atol: float,
    exact: bool,
    chunksize: int,
    ignore_dtype: bool,
) -> bool:
    if ds_a.shape != ds_b.shape:
        print(f"[FAIL] {name}: shape mismatch: {ds_a.shape} vs {ds_b.shape}")
        return False

    if ds_a.dtype != ds_b.dtype and not ignore_dtype:
        print(f"[FAIL] {name}: dtype mismatch: {ds_a.dtype} vs {ds_b.dtype}")
        return False

    shape = ds_a.shape
    ndim = len(shape)

    mode = "exact" if exact else f"rtol={rtol:g}, atol={atol:g}"
    dtype_note = f"{ds_a.dtype}" if ds_a.dtype == ds_b.dtype else f"{ds_a.dtype} vs {ds_b.dtype}"

    print(f"[INFO] {name}: shape={shape}, dtype={dtype_note}, mode={mode}")

    if ndim == 0:
        a = ds_a[()]
        b = ds_b[()]

        if exact:
            if a != b:
                print(f"[FAIL] {name}: scalar mismatch: {a} vs {b}")
                return False
        else:
            if not np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
                abs_diff = abs(float(a) - float(b))
                denom = max(abs(float(a)), abs(float(b)), np.finfo(np.float64).tiny)
                rel_diff = abs_diff / denom

                print(f"[FAIL] {name}: scalar mismatch: {a} vs {b}")
                print(f"       abs_diff={abs_diff:.17g}, rel_diff={rel_diff:.17g}")
                return False

        print(f"[ OK ] {name}")
        return True

    n0 = shape[0]

    global_max_abs = 0.0
    global_max_rel = 0.0
    global_max_abs_idx: Optional[Tuple[int, ...]] = None
    global_max_rel_idx: Optional[Tuple[int, ...]] = None

    for start in range(0, n0, chunksize):
        stop = min(start + chunksize, n0)

        sl = (slice(start, stop),) + (slice(None),) * (ndim - 1)

        a = ds_a[sl]
        b = ds_b[sl]

        if exact:
            if not np.array_equal(a, b):
                local_idx = first_mismatch_index_exact(a, b)
                assert local_idx is not None

                global_idx = (start + local_idx[0],) + local_idx[1:]

                print(f"[FAIL] {name}: exact mismatch at index {format_index(global_idx)}")
                print(f"       values: {a[local_idx]} vs {b[local_idx]}")
                return False

        else:
            max_abs, max_rel, max_abs_idx, max_rel_idx = max_abs_rel_diff(a, b)

            if max_abs > global_max_abs:
                global_max_abs = max_abs
                global_max_abs_idx = (start + max_abs_idx[0],) + max_abs_idx[1:]

            if max_rel > global_max_rel:
                global_max_rel = max_rel
                global_max_rel_idx = (start + max_rel_idx[0],) + max_rel_idx[1:]

            if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
                local_idx = first_mismatch_index_close(a, b, rtol, atol)
                assert local_idx is not None

                global_idx = (start + local_idx[0],) + local_idx[1:]

                print(f"[FAIL] {name}: mismatch at index {format_index(global_idx)}")
                print(f"       values: {a[local_idx]} vs {b[local_idx]}")
                print(f"       chunk max_abs_diff={max_abs:.17g}")
                print(f"       chunk max_rel_diff={max_rel:.17g}")

                if global_max_abs_idx is not None:
                    print(f"       global max_abs_diff_so_far={global_max_abs:.17g} at {format_index(global_max_abs_idx)}")

                if global_max_rel_idx is not None:
                    print(f"       global max_rel_diff_so_far={global_max_rel:.17g} at {format_index(global_max_rel_idx)}")

                return False

    if exact:
        print(f"[ OK ] {name}: exact match")
    else:
        print(f"[ OK ] {name}: matched within tolerance")
        print(f"       max_abs_diff={global_max_abs:.17g}")
        print(f"       max_rel_diff={global_max_rel:.17g}")

        if global_max_abs_idx is not None:
            print(f"       max_abs_index={format_index(global_max_abs_idx)}")

        if global_max_rel_idx is not None:
            print(f"       max_rel_index={format_index(global_max_rel_idx)}")

    return True


def read_step_dataset(file_handle: h5py.File, label: str) -> Optional[np.ndarray]:
    if not dataset_exists(file_handle, "/step"):
        print(f"[FAIL] {label}: missing /step dataset")
        return None

    step = file_handle["/step"][()]

    if step.ndim != 1:
        print(f"[FAIL] {label}:/step: expected 1D dataset, got shape {step.shape}")
        return None

    return step


def validate_steps(
    step_a: np.ndarray,
    step_b: np.ndarray,
    max_steps: Optional[int],
) -> bool:
    ok = True

    print(f"[INFO] /step: frames={step_a.size}")

    if step_a.shape != step_b.shape:
        print(f"[FAIL] /step: shape mismatch: {step_a.shape} vs {step_b.shape}")
        return False

    if not np.array_equal(step_a, step_b):
        idx = first_mismatch_index_exact(step_a, step_b)
        assert idx is not None

        print(f"[FAIL] /step: mismatch at index {format_index(idx)}")
        print(f"       values: {step_a[idx]} vs {step_b[idx]}")
        ok = False

    if step_a.size == 0:
        print("[FAIL] /step: no frames written")
        return False

    if int(step_a[0]) != 0:
        print(f"[FAIL] /step: first saved step is {step_a[0]}, expected 0")
        ok = False

    diffs = np.diff(step_a)

    if np.any(diffs <= 0):
        bad = int(np.argwhere(diffs <= 0)[0, 0])
        print("[FAIL] /step: steps are not strictly increasing")
        print(f"       step[{bad}]={step_a[bad]}, step[{bad + 1}]={step_a[bad + 1]}")
        ok = False

    unique_count = np.unique(step_a).size
    if unique_count != step_a.size:
        print(f"[FAIL] /step: duplicate step values detected: unique={unique_count}, total={step_a.size}")
        ok = False

    if max_steps is not None:
        if int(step_a[-1]) != int(max_steps):
            print(f"[FAIL] /step: last saved step is {step_a[-1]}, expected max_steps={max_steps}")
            ok = False
        else:
            print(f"[ OK ] /step: final saved step is max_steps={max_steps}")

    if ok:
        print("[ OK ] /step: identical, strictly increasing, no duplicates")

    return ok


def expected_steps(max_steps: int, output_every: int) -> np.ndarray:
    steps = list(range(0, max_steps, output_every))

    if not steps or steps[-1] != max_steps:
        steps.append(max_steps)

    return np.asarray(steps, dtype=np.int64)


def validate_expected_steps(
    step: np.ndarray,
    max_steps: int,
    output_every: int,
) -> bool:
    exp = expected_steps(max_steps, output_every)

    if not np.array_equal(step, exp):
        print("[FAIL] /step: does not match expected output cadence")
        print(f"       expected number of frames: {exp.size}")
        print(f"       actual number of frames:   {step.size}")

        n = min(exp.size, step.size)
        mismatch = np.argwhere(exp[:n] != step[:n])

        if mismatch.size:
            i = int(mismatch[0, 0])
            print(f"       first mismatch at frame {i}: expected step {exp[i]}, got {step[i]}")
        elif exp.size != step.size:
            print("       common prefix matches, but lengths differ")

        print(f"       expected first/last: {exp[0]} / {exp[-1]}")
        print(f"       actual first/last:   {step[0]} / {step[-1]}")
        return False

    print("[ OK ] /step: matches expected cadence")
    return True


def parse_dataset_list(value: str) -> Tuple[str, ...]:
    names = []

    for item in value.split(","):
        item = item.strip()
        if not item:
            continue

        if not item.startswith("/"):
            item = "/" + item

        names.append(item)

    if not names:
        raise argparse.ArgumentTypeError("dataset list cannot be empty")

    return tuple(names)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two particles.h5 files dataset-by-dataset."
    )

    parser.add_argument("file_a", help="Reference file")
    parser.add_argument("file_b", help="Candidate file")

    parser.add_argument(
        "--datasets",
        type=parse_dataset_list,
        default=DEFAULT_DATASETS,
        help="Comma-separated datasets to compare. Default: /step,/pos,/vel,/screen",
    )

    parser.add_argument(
        "--rtol",
        type=float,
        default=1.0e-12,
        help="Relative tolerance for floating-point datasets. Default: 1e-12",
    )

    parser.add_argument(
        "--atol",
        type=float,
        default=1.0e-14,
        help="Absolute tolerance for floating-point datasets. Default: 1e-14",
    )

    parser.add_argument(
        "--screen-rtol",
        type=float,
        default=0.0,
        help="Relative tolerance for /screen if --relaxed-screen is used. Default: 0",
    )

    parser.add_argument(
        "--screen-atol",
        type=float,
        default=0.0,
        help="Absolute tolerance for /screen if --relaxed-screen is used. Default: 0",
    )

    parser.add_argument(
        "--relaxed-screen",
        action="store_true",
        help="Compare /screen with tolerance instead of exact equality.",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=8,
        help="Number of frames to compare at once. Default: 8",
    )

    parser.add_argument(
        "--exact-floats",
        action="store_true",
        help="Require exact equality for floating-point datasets.",
    )

    parser.add_argument(
        "--ignore-dtype",
        action="store_true",
        help="Ignore dtype mismatch and compare values only.",
    )

    parser.add_argument(
        "--check-finite",
        action="store_true",
        help="Fail if floating-point datasets contain NaN or Inf.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Expected final physical step. If provided, /step[-1] must equal this.",
    )

    parser.add_argument(
        "--output-every",
        type=int,
        default=None,
        help="Expected output cadence. Requires --max-steps. Validates exact /step sequence.",
    )

    args = parser.parse_args()

    if args.chunksize <= 0:
        print("ERROR: --chunksize must be > 0", file=sys.stderr)
        return 2

    if args.rtol < 0.0 or args.atol < 0.0:
        print("ERROR: --rtol and --atol must be >= 0", file=sys.stderr)
        return 2

    if args.screen_rtol < 0.0 or args.screen_atol < 0.0:
        print("ERROR: --screen-rtol and --screen-atol must be >= 0", file=sys.stderr)
        return 2

    if args.output_every is not None:
        if args.output_every <= 0:
            print("ERROR: --output-every must be > 0", file=sys.stderr)
            return 2

        if args.max_steps is None:
            print("ERROR: --output-every requires --max-steps", file=sys.stderr)
            return 2

    if args.max_steps is not None and args.max_steps <= 0:
        print("ERROR: --max-steps must be > 0", file=sys.stderr)
        return 2

    try:
        with h5py.File(args.file_a, "r") as fa, h5py.File(args.file_b, "r") as fb:
            overall_ok = True

            for name in args.datasets:
                if not dataset_exists(fa, name):
                    print(f"[FAIL] Missing dataset in reference file: {name}")
                    overall_ok = False

                if not dataset_exists(fb, name):
                    print(f"[FAIL] Missing dataset in candidate file: {name}")
                    overall_ok = False

            if not overall_ok:
                print("\nFAILURE: missing datasets.")
                return 1

            if "/step" in args.datasets:
                step_a = read_step_dataset(fa, "reference")
                step_b = read_step_dataset(fb, "candidate")

                if step_a is None or step_b is None:
                    overall_ok = False
                else:
                    ok = validate_steps(
                        step_a=step_a,
                        step_b=step_b,
                        max_steps=args.max_steps,
                    )
                    overall_ok = overall_ok and ok

                    if args.max_steps is not None and args.output_every is not None:
                        ok = validate_expected_steps(
                            step=step_a,
                            max_steps=args.max_steps,
                            output_every=args.output_every,
                        )
                        overall_ok = overall_ok and ok

            if args.check_finite:
                for name in args.datasets:
                    if name in ("/pos", "/vel"):
                        ok_a = check_finite_dataset(
                            fa[name],
                            name,
                            args.chunksize,
                            label="reference",
                        )

                        ok_b = check_finite_dataset(
                            fb[name],
                            name,
                            args.chunksize,
                            label="candidate",
                        )

                        overall_ok = overall_ok and ok_a and ok_b

            for name in args.datasets:
                if name == "/step":
                    # Already handled with stronger step-specific logic.
                    continue

                ds_a = fa[name]
                ds_b = fb[name]

                if name == "/screen":
                    if args.relaxed_screen:
                        ok = compare_dataset(
                            ds_a=ds_a,
                            ds_b=ds_b,
                            name=name,
                            rtol=args.screen_rtol,
                            atol=args.screen_atol,
                            exact=False,
                            chunksize=args.chunksize,
                            ignore_dtype=args.ignore_dtype,
                        )
                    else:
                        ok = compare_dataset(
                            ds_a=ds_a,
                            ds_b=ds_b,
                            name=name,
                            rtol=0.0,
                            atol=0.0,
                            exact=True,
                            chunksize=args.chunksize,
                            ignore_dtype=args.ignore_dtype,
                        )
                else:
                    floating = np.issubdtype(ds_a.dtype, np.floating) or np.issubdtype(ds_b.dtype, np.floating)

                    if floating and not args.exact_floats:
                        ok = compare_dataset(
                            ds_a=ds_a,
                            ds_b=ds_b,
                            name=name,
                            rtol=args.rtol,
                            atol=args.atol,
                            exact=False,
                            chunksize=args.chunksize,
                            ignore_dtype=args.ignore_dtype,
                        )
                    else:
                        ok = compare_dataset(
                            ds_a=ds_a,
                            ds_b=ds_b,
                            name=name,
                            rtol=0.0,
                            atol=0.0,
                            exact=True,
                            chunksize=args.chunksize,
                            ignore_dtype=args.ignore_dtype,
                        )

                overall_ok = overall_ok and ok

            if overall_ok:
                print("\nSUCCESS: files match for all checked datasets.")
                return 0

            print("\nFAILURE: files differ.")
            return 1

    except OSError as exc:
        print(f"ERROR: cannot open file: {exc}", file=sys.stderr)
        return 2

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
