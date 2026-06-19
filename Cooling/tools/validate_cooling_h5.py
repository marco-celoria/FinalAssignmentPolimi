#!/usr/bin/env python3
"""
Validate two Cooling Solver HDF5 output files.

Expected HDF5 structure:

  /field : float dataset with shape (nframes, ny, nx)
  /step  : integer dataset with shape (nframes,)

Comparison policy:

  - /step is compared exactly.
  - /field is compared with tolerance by default.
  - /field can be compared exactly with --exact-field.
  - Dataset shapes and dtypes must match.

Default tolerance is aligned with the project validation rule:

  abs(candidate - reference) <= atol + rtol * abs(reference)

with:

  rtol = 1.0e-8
  atol = 1.0e-8
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Tuple

import h5py
import numpy as np


DATASETS = ("/field", "/step")


def format_index(idx_tuple: Tuple[int, ...]) -> str:
    return "(" + ", ".join(str(int(x)) for x in idx_tuple) + ")"


def project_isclose(
    reference: np.ndarray,
    candidate: np.ndarray,
    rtol: float,
    atol: float,
) -> np.ndarray:
    """
    Project-specific asymmetric tolerance rule:

        abs(candidate - reference) <= atol + rtol * abs(reference)

    This intentionally uses the reference as the relative-tolerance scale.
    """
    return np.abs(candidate - reference) <= (atol + rtol * np.abs(reference))


def project_scalar_isclose(
    reference,
    candidate,
    rtol: float,
    atol: float,
) -> bool:
    return abs(candidate - reference) <= (atol + rtol * abs(reference))


def first_mismatch_index_exact(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> Optional[Tuple[int, ...]]:
    diff = reference != candidate

    if not np.any(diff):
        return None

    return tuple(int(x) for x in np.argwhere(diff)[0])


def first_mismatch_index_close(
    reference: np.ndarray,
    candidate: np.ndarray,
    rtol: float,
    atol: float,
) -> Optional[Tuple[int, ...]]:
    close = project_isclose(
        reference=reference,
        candidate=candidate,
        rtol=rtol,
        atol=atol,
    )

    diff = ~close

    if not np.any(diff):
        return None

    return tuple(int(x) for x in np.argwhere(diff)[0])


def max_abs_rel_diff(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> Tuple[float, float]:
    """
    Diagnostic max absolute and reference-relative differences.

    Relative difference is reported as:

        abs(candidate - reference) / max(abs(reference), tiny)

    so it is consistent with the project validation rule.
    """
    abs_diff = np.abs(candidate - reference)
    max_abs = float(np.max(abs_diff)) if abs_diff.size else 0.0

    denom = np.maximum(np.abs(reference), np.finfo(np.float64).tiny)
    rel_diff = abs_diff / denom
    max_rel = float(np.max(rel_diff)) if rel_diff.size else 0.0

    return max_abs, max_rel


def compare_scalar(
    reference,
    candidate,
    name: str,
    rtol: float,
    atol: float,
    exact: bool,
) -> bool:
    if exact:
        if reference != candidate:
            print(f"[FAIL] {name}: scalar mismatch: {reference} vs {candidate}")
            return False

        print(f"[ OK ] {name}: exact scalar match")
        return True

    if not project_scalar_isclose(reference, candidate, rtol=rtol, atol=atol):
        abs_diff = abs(candidate - reference)
        denom = max(abs(reference), np.finfo(np.float64).tiny)
        rel_diff = abs_diff / denom
        threshold = atol + rtol * abs(reference)

        print(f"[FAIL] {name}: scalar mismatch: {reference} vs {candidate}")
        print(f"       abs_diff={abs_diff:.17g}")
        print(f"       rel_diff_vs_reference={rel_diff:.17g}")
        print(f"       tolerance_threshold={threshold:.17g}")
        return False

    print(f"[ OK ] {name}: scalar matched within tolerance")
    return True


def compare_dataset(
    ds_reference: h5py.Dataset,
    ds_candidate: h5py.Dataset,
    name: str,
    rtol: float,
    atol: float,
    exact: bool,
    chunksize: int,
) -> bool:
    if ds_reference.shape != ds_candidate.shape:
        print(
            f"[FAIL] {name}: shape mismatch: "
            f"{ds_reference.shape} vs {ds_candidate.shape}"
        )
        return False

    if ds_reference.dtype != ds_candidate.dtype:
        print(
            f"[FAIL] {name}: dtype mismatch: "
            f"{ds_reference.dtype} vs {ds_candidate.dtype}"
        )
        return False

    shape = ds_reference.shape
    ndim = len(shape)

    mode = "exact" if exact else f"rtol={rtol}, atol={atol}, reference-scaled"
    print(
        f"[INFO] {name}: "
        f"shape={shape}, dtype={ds_reference.dtype}, mode={mode}"
    )

    if ndim == 0:
        reference = ds_reference[()]
        candidate = ds_candidate[()]
        return compare_scalar(reference, candidate, name, rtol, atol, exact)

    n0 = shape[0]

    for start in range(0, n0, chunksize):
        stop = min(start + chunksize, n0)
        sl = (slice(start, stop),) + (slice(None),) * (ndim - 1)

        reference = ds_reference[sl]
        candidate = ds_candidate[sl]

        if exact:
            if not np.array_equal(reference, candidate):
                local_idx = first_mismatch_index_exact(reference, candidate)

                if local_idx is None:
                    print(f"[FAIL] {name}: exact mismatch detected")
                    return False

                global_idx = (start + local_idx[0],) + local_idx[1:]

                print(
                    f"[FAIL] {name}: exact mismatch at index "
                    f"{format_index(global_idx)}"
                )
                print(
                    f"       values: reference={reference[local_idx]}, "
                    f"candidate={candidate[local_idx]}"
                )
                return False

        else:
            close = project_isclose(
                reference=reference,
                candidate=candidate,
                rtol=rtol,
                atol=atol,
            )

            if not bool(np.all(close)):
                local_idx = first_mismatch_index_close(
                    reference,
                    candidate,
                    rtol,
                    atol,
                )

                if local_idx is None:
                    print(f"[FAIL] {name}: tolerance mismatch detected")
                    return False

                global_idx = (start + local_idx[0],) + local_idx[1:]
                max_abs, max_rel = max_abs_rel_diff(reference, candidate)

                ref_value = reference[local_idx]
                cand_value = candidate[local_idx]
                abs_diff = abs(cand_value - ref_value)
                threshold = atol + rtol * abs(ref_value)

                print(
                    f"[FAIL] {name}: mismatch at index "
                    f"{format_index(global_idx)}"
                )
                print(f"       reference={ref_value}")
                print(f"       candidate={cand_value}")
                print(f"       abs_diff={abs_diff:.17g}")
                print(f"       tolerance_threshold={threshold:.17g}")
                print(
                    f"       chunk max_abs_diff={max_abs:.17g}, "
                    f"chunk max_rel_diff_vs_reference={max_rel:.17g}"
                )
                return False

    if exact:
        print(f"[ OK ] {name}: exact match")
    else:
        print(f"[ OK ] {name}: matched within project tolerance")

    return True


def validate_hdf5_files(
    reference_file: str,
    candidate_file: str,
    rtol: float,
    atol: float,
    chunksize: int,
    exact_field: bool,
) -> bool:
    with h5py.File(reference_file, "r") as reference_h5, h5py.File(candidate_file, "r") as candidate_h5:
        overall_ok = True

        for name in DATASETS:
            reference_has_dataset = name in reference_h5
            candidate_has_dataset = name in candidate_h5

            if not reference_has_dataset or not candidate_has_dataset:
                print(f"[FAIL] Missing dataset: {name}")

                if not reference_has_dataset:
                    print(f"       Reference file missing {name}")

                if not candidate_has_dataset:
                    print(f"       Candidate file missing {name}")

                overall_ok = False
                continue

            ds_reference = reference_h5[name]
            ds_candidate = candidate_h5[name]

            if name == "/step":
                ok = compare_dataset(
                    ds_reference=ds_reference,
                    ds_candidate=ds_candidate,
                    name=name,
                    rtol=0.0,
                    atol=0.0,
                    exact=True,
                    chunksize=chunksize,
                )

            elif name == "/field":
                ok = compare_dataset(
                    ds_reference=ds_reference,
                    ds_candidate=ds_candidate,
                    name=name,
                    rtol=rtol,
                    atol=atol,
                    exact=exact_field,
                    chunksize=chunksize,
                )

            else:
                print(f"[FAIL] Unexpected dataset configured for validation: {name}")
                ok = False

            overall_ok = overall_ok and ok

        return overall_ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two Cooling Solver HDF5 outputs."
    )

    parser.add_argument(
        "reference",
        help="Reference HDF5 file.",
    )

    parser.add_argument(
        "candidate",
        help="Candidate HDF5 file.",
    )

    parser.add_argument(
        "--rtol",
        type=float,
        default=1.0e-8,
        help="Relative tolerance for /field. Default: 1e-8.",
    )

    parser.add_argument(
        "--atol",
        type=float,
        default=1.0e-8,
        help="Absolute tolerance for /field. Default: 1e-8.",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=4,
        help="Number of frames to compare at once. Default: 4.",
    )

    parser.add_argument(
        "--exact-field",
        action="store_true",
        help="Require exact equality for /field instead of tolerance comparison.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.rtol < 0.0:
        print("ERROR: --rtol must be >= 0", file=sys.stderr)
        return 2

    if args.atol < 0.0:
        print("ERROR: --atol must be >= 0", file=sys.stderr)
        return 2

    if args.chunksize <= 0:
        print("ERROR: --chunksize must be > 0", file=sys.stderr)
        return 2

    try:
        ok = validate_hdf5_files(
            reference_file=args.reference,
            candidate_file=args.candidate,
            rtol=args.rtol,
            atol=args.atol,
            chunksize=args.chunksize,
            exact_field=args.exact_field,
        )

        if ok:
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

