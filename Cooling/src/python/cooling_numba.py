#!/usr/bin/env python3

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple

import h5py
import numpy as np
from numba import njit, prange, get_num_threads, set_num_threads


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class MeasuredPoint:
    x: float
    y: float
    v: float


@dataclass
class Config:
    nx: int
    ny: int
    Sreal: float
    Simag: float
    Dreal: float
    Dimag: float
    maxIters: int
    steps: int
    outputEvery: int
    measured: List[MeasuredPoint]


# ============================================================
# VALIDATION
# ============================================================

def validated_grid_size(nx: int, ny: int) -> int:
    if nx <= 0 or ny <= 0:
        raise ValueError("Grid dimensions must be > 0")

    n = nx * ny

    if n <= 0:
        raise ValueError("Invalid total grid size")

    return n


def validate_output_every(output_every: int) -> None:
    if output_every <= 0:
        raise ValueError("outputEvery must be > 0")


def validate_stats_mode(mode: str) -> None:
    if mode not in ("accurate", "fast"):
        raise ValueError("stats mode must be either 'accurate' or 'fast'")


# ============================================================
# INPUT PARSER
# ============================================================

def read_input(fname: str) -> Config:
    tokens: List[str] = []

    try:
        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                line = line.split("#", 1)[0]
                parts = line.split()
                if parts:
                    tokens.extend(parts)
    except OSError as e:
        raise RuntimeError(f"Cannot open input file: {fname}") from e

    if not tokens:
        raise RuntimeError(f"Input file is empty or contains no numeric tokens: {fname}")

    pos = 0

    def next_int() -> int:
        nonlocal pos

        if pos >= len(tokens):
            raise RuntimeError("Malformed input: missing integer token")

        tok = tokens[pos]
        pos += 1

        try:
            return int(tok)
        except Exception as e:
            raise RuntimeError(f"Malformed input: invalid integer token '{tok}'") from e

    def next_float() -> float:
        nonlocal pos

        if pos >= len(tokens):
            raise RuntimeError("Malformed input: missing floating-point token")

        tok = tokens[pos]
        pos += 1

        try:
            return float(tok)
        except Exception as e:
            raise RuntimeError(f"Malformed input: invalid floating-point token '{tok}'") from e

    nx = next_int()
    ny = next_int()

    if nx < 3 or ny < 3:
        raise RuntimeError("Grid dimensions must be at least 3 x 3")

    n_measured = next_int()

    if n_measured < 0:
        raise RuntimeError("Number of measured points cannot be negative")

    measured: List[MeasuredPoint] = []

    for _ in range(n_measured):
        measured.append(
            MeasuredPoint(
                x=next_float(),
                y=next_float(),
                v=next_float(),
            )
        )

    sreal = next_float()
    simag = next_float()
    dreal = next_float()
    dimag = next_float()
    max_iters = next_int()
    steps = next_int()

    if max_iters <= 0:
        raise RuntimeError("maxIters must be > 0")

    if steps < 0:
        raise RuntimeError("steps must be >= 0")

    if dreal <= 0.0 or dimag <= 0.0:
        raise RuntimeError("Domain extents Dreal and Dimag must be > 0")

    if pos != len(tokens):
        raise RuntimeError("Malformed input: unexpected extra tokens at end of file")

    return Config(
        nx=nx,
        ny=ny,
        Sreal=sreal,
        Simag=simag,
        Dreal=dreal,
        Dimag=dimag,
        maxIters=max_iters,
        steps=steps,
        outputEvery=1,
        measured=measured,
    )


# ============================================================
# HOST-SIDE PHYSICS HELPERS
# ============================================================

def build_domain_params(cfg: Config) -> Tuple[float, float, float, float]:
    """
    Node-centered geometry including both boundaries:

        x(i) = x0 + i * dx,   dx = Dreal / (nx - 1)
        y(j) = y0 + j * dy,   dy = Dimag / (ny - 1)
    """
    x0 = cfg.Sreal
    y0 = cfg.Simag
    dx = cfg.Dreal / float(cfg.nx - 1)
    dy = cfg.Dimag / float(cfg.ny - 1)

    return x0, y0, dx, dy


def analytical_field(x: float, y: float) -> float:
    return (x * x * x + y * y * y) / 6.0


def compute_discrepancy(cfg: Config) -> float:
    """
    Computes:

        mean(measured_value - analytical_field(x, y))
    """
    if not cfg.measured:
        return 0.0

    acc = np.longdouble(0.0)

    for m in cfg.measured:
        acc += np.longdouble(m.v - analytical_field(m.x, m.y))

    return float(acc / np.longdouble(len(cfg.measured)))


def build_cooling_coeffs(
    dx: float,
    dy: float,
    dd: float = 100.0,
) -> Tuple[float, float, float, float, float, float, float]:
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("build_cooling_coeffs: dx and dy must be > 0")

    if dd <= 0.0:
        raise ValueError("build_cooling_coeffs: dd must be > 0")

    hx = dx
    hy = dy

    dgx = -2.0 * (1.0 + dd * hx / (hx * hx + dd))
    dgy = -2.0 * (1.0 + dd * hy / (hy * hy + dd))

    CX = (hx + dd * math.exp(hx)) / (15.0 * dd + hx)
    CY = (hy + dd * math.exp(hy)) / (15.0 * dd + hy)

    return dd, hx, hy, dgx, dgy, CX, CY


# ============================================================
# NUMBA CPU KERNELS
#
# These are intentionally CUDA-shaped:
# - flat arrays
# - explicit scalar parameters
# - no Python objects in hot loops
# - interior and boundary work separated
# ============================================================

@njit(parallel=True, fastmath=False)
def compute_weight_kernel(
    weight: np.ndarray,
    nx: int,
    ny: int,
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    max_iters: int,
) -> None:
    for j in prange(ny):
        row = j * nx
        y = y0 + dy * float(j)

        for i in range(nx):
            x = x0 + dx * float(i)
            p = row + i

            za = 0.0
            zb = 0.0
            it = 0

            while it < max_iters:
                if za * za + zb * zb > 4.0:
                    break

                tmp = za * za - zb * zb + x
                zb = 2.0 * za * zb + y
                za = tmp

                it += 1

            weight[p] = it


@njit(parallel=True, fastmath=False)
def initialize_field_kernel(
    u: np.ndarray,
    weight: np.ndarray,
    nx: int,
    ny: int,
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    discrepancy: float,
    wmin: int,
    wmax: int,
) -> None:
    denom = float(wmax - wmin) if wmax > wmin else 1.0

    for j in prange(ny):
        y = y0 + dy * j
        row = j * nx

        for i in range(nx):
            x = x0 + dx * i
            p = row + i

            F = (x * x * x + y * y * y) / 6.0
            wnorm = (weight[p] - wmin) / denom

            u[p] = 293.16 + 80.0 * (discrepancy + F) * wnorm


@njit(parallel=True, fastmath=False)
def update_interior_kernel(
    u1: np.ndarray,
    u2: np.ndarray,
    nx: int,
    ny: int,
    dgx: float,
    dgy: float,
    CX: float,
    CY: float,
) -> None:
    for j in prange(1, ny - 1):
        row = j * nx
        row_up = (j - 1) * nx
        row_dn = (j + 1) * nx

        for i in range(1, nx - 1):
            p = row + i

            u2[p] = (
                CX * (
                    u1[p - 1]
                    + u1[p + 1]
                    + (dgx + 0.5 / CX) * u1[p]
                )
                + CY * (
                    u1[row_up + i]
                    + u1[row_dn + i]
                    + (dgy + 0.5 / CY) * u1[p]
                )
            )


@njit(parallel=True, fastmath=False)
def apply_boundary_lr_kernel(u: np.ndarray, nx: int, ny: int) -> None:
    """
    Left/right boundaries excluding corners.
    """
    for j in prange(1, ny - 1):
        row = j * nx
        u[row] = u[row + 1]
        u[row + nx - 1] = u[row + nx - 2]


@njit(parallel=True, fastmath=False)
def apply_boundary_tb_kernel(u: np.ndarray, nx: int, ny: int) -> None:
    """
    Top/bottom boundaries including corners.

    This intentionally runs after apply_boundary_lr_kernel to preserve
    the same corner behavior as the original CUDA-style version.
    """
    row_top = 0
    row_1 = nx
    row_nm2 = (ny - 2) * nx
    row_nm1 = (ny - 1) * nx

    for i in prange(nx):
        u[row_top + i] = u[row_1 + i]
        u[row_nm1 + i] = u[row_nm2 + i]


def update_field(
    u1: np.ndarray,
    u2: np.ndarray,
    nx: int,
    ny: int,
    dgx: float,
    dgy: float,
    CX: float,
    CY: float,
) -> None:
    """
    Host-side launch sequence.

    Kept intentionally close to CUDA launch ordering:
      1. update interior
      2. apply left/right boundaries
      3. apply top/bottom boundaries
    """
    update_interior_kernel(u1, u2, nx, ny, dgx, dgy, CX, CY)
    apply_boundary_lr_kernel(u2, nx, ny)
    apply_boundary_tb_kernel(u2, nx, ny)


# ============================================================
# HDF5 WRITER
# ============================================================

class H5Writer:
    def __init__(
        self,
        fname: str,
        nx: int,
        ny: int,
        batch: int = 32,
        tile_y: int = 256,
        tile_x: int = 256,
    ):
        if nx <= 0 or ny <= 0:
            raise ValueError("H5Writer: nx and ny must be > 0")

        if batch <= 0:
            raise ValueError("H5Writer: batch must be > 0")

        if tile_y <= 0 or tile_x <= 0:
            raise ValueError("H5Writer: tile sizes must be > 0")

        self.nx = nx
        self.ny = ny
        self.batch = batch
        self.frame = 0
        self.capacity = batch
        self.closed = False

        chunk_y = min(ny, tile_y)
        chunk_x = min(nx, tile_x)

        self.file = h5py.File(fname, "w")

        self.field = self.file.create_dataset(
            "field",
            shape=(0, ny, nx),
            maxshape=(None, ny, nx),
            chunks=(1, chunk_y, chunk_x),
            dtype=np.float64,
        )

        self.step = self.file.create_dataset(
            "step",
            shape=(0,),
            maxshape=(None,),
            chunks=(batch,),
            dtype=np.int32,
        )

        self._extend(self.capacity)

    def _extend(self, n: int) -> None:
        self.field.resize((n, self.ny, self.nx))
        self.step.resize((n,))

    def write(self, step_number: int, field_1d: np.ndarray) -> None:
        if self.closed:
            raise RuntimeError("H5Writer: write() called after close()")

        if field_1d.size != self.nx * self.ny:
            raise RuntimeError("H5Writer: field size mismatch")

        if self.frame >= self.capacity:
            self.capacity += self.batch
            self._extend(self.capacity)

        self.field[self.frame, :, :] = field_1d.reshape(self.ny, self.nx)
        self.step[self.frame] = np.int32(step_number)
        self.frame += 1

    def close(self) -> None:
        if self.closed:
            return

        self.closed = True

        if self.frame != self.capacity:
            self._extend(self.frame)

        self.file.flush()
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


# ============================================================
# STATISTICS
# ============================================================

def compute_stats_accurate(u: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Numerically careful statistics.

    This is close to your original implementation:
    - min/max in float64
    - mean/std using longdouble accumulation

    More accurate, but slower and allocates a temporary longdouble array.
    """
    if u.size == 0:
        raise RuntimeError("compute_stats_accurate: empty field")

    mn = float(np.min(u))
    mx = float(np.max(u))

    mean_ld = np.sum(u, dtype=np.longdouble) / np.longdouble(u.size)
    diff_ld = u.astype(np.longdouble) - mean_ld
    std_ld = np.sqrt(
        np.sum(diff_ld * diff_ld, dtype=np.longdouble) / np.longdouble(u.size)
    )

    return float(mn), float(mean_ld), float(mx), float(std_ld)


def compute_stats_fast(u: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Fast statistics.

    Useful for performance runs where exact longdouble agreement is not required.
    """
    if u.size == 0:
        raise RuntimeError("compute_stats_fast: empty field")

    mn = float(np.min(u))
    mx = float(np.max(u))
    mean = float(np.mean(u))
    std = float(np.std(u))

    return mn, mean, mx, std


def compute_stats(u: np.ndarray, mode: str) -> Tuple[float, float, float, float]:
    if mode == "accurate":
        return compute_stats_accurate(u)

    if mode == "fast":
        return compute_stats_fast(u)

    raise ValueError("Unknown stats mode")


def write_stats_header(f) -> None:
    f.write("Step;Min;Mean;Max;Std_dev\n")


def write_stats_line(
    f,
    step: int,
    stats: Tuple[float, float, float, float],
) -> None:
    mn, mean, mx, std = stats
    f.write(f"{step};{mn:.15g};{mean:.15g};{mx:.15g};{std:.15g}\n")


# ============================================================
# NUMBA WARM-UP
# ============================================================

def warmup_numba() -> None:
    """
    Forces Numba to compile kernels before measured sections.

    The array dtypes match the real simulation:
      weight : int32
      u      : float64
    """
    print("Warming up Numba JIT compiler...")

    nx, ny = 3, 3
    n = nx * ny

    x0 = 0.0
    y0 = 0.0
    dx = 0.1
    dy = 0.1

    dgx = 1.0
    dgy = 1.0
    CX = 1.0
    CY = 1.0

    discrepancy = 0.0
    wmin = 0
    wmax = 10
    max_iters = 2

    weight = np.empty(n, dtype=np.int32)
    u_curr = np.zeros(n, dtype=np.float64)
    u_next = np.zeros(n, dtype=np.float64)

    compute_weight_kernel(weight, nx, ny, x0, y0, dx, dy, max_iters)
    initialize_field_kernel(
        u_curr,
        weight,
        nx,
        ny,
        x0,
        y0,
        dx,
        dy,
        discrepancy,
        wmin,
        wmax,
    )

    update_interior_kernel(u_curr, u_next, nx, ny, dgx, dgy, CX, CY)
    apply_boundary_lr_kernel(u_next, nx, ny)
    apply_boundary_tb_kernel(u_next, nx, ny)

    print("JIT warmup completed; reported timings exclude compilation.")


# ============================================================
# HOST-SIDE DRIVER
# ============================================================

def run_simulation(
    cfg: Config,
    h5_file: str,
    csv_file: str,
    stats_mode: str,
    h5_tile_y: int,
    h5_tile_x: int,
) -> None:
    validate_stats_mode(stats_mode)

    n = validated_grid_size(cfg.nx, cfg.ny)

    x0, y0, dx, dy = build_domain_params(cfg)
    _, _, _, dgx, dgy, CX, CY = build_cooling_coeffs(dx, dy, 100.0)
    discrepancy = compute_discrepancy(cfg)

    # Flat, GPU-friendly arrays.
    weight = np.empty(n, dtype=np.int32)
    u_curr = np.empty(n, dtype=np.float64)
    u_next = np.empty(n, dtype=np.float64)

    # Warm-up before measured work.
    warmup_numba()

    total_wall_t0 = time.perf_counter()

    # --------------------------------------------------------
    # Weight field
    # --------------------------------------------------------
    t0 = time.perf_counter()

    compute_weight_kernel(
        weight,
        cfg.nx,
        cfg.ny,
        x0,
        y0,
        dx,
        dy,
        cfg.maxIters,
    )

    t1 = time.perf_counter()

    wmin = int(np.min(weight))
    wmax = int(np.max(weight))

    weight_time_s = t1 - t0

    # --------------------------------------------------------
    # Initialization
    # --------------------------------------------------------
    t2 = time.perf_counter()

    initialize_field_kernel(
        u_curr,
        weight,
        cfg.nx,
        cfg.ny,
        x0,
        y0,
        dx,
        dy,
        discrepancy,
        wmin,
        wmax,
    )

    t3 = time.perf_counter()

    init_time_s = t3 - t2

    # --------------------------------------------------------
    # Dynamics + output loop
    # --------------------------------------------------------
    pure_dynamics_time_s = 0.0
    output_stats_io_time_s = 0.0
    output_frames = 0

    loop_t0 = time.perf_counter()

    with open(csv_file, "w", encoding="utf-8") as csvf:
        write_stats_header(csvf)

        with H5Writer(
            h5_file,
            cfg.nx,
            cfg.ny,
            batch=32,
            tile_y=h5_tile_y,
            tile_x=h5_tile_x,
        ) as writer:
            # Step 0 output.
            out_t0 = time.perf_counter()

            stats0 = compute_stats(u_curr, stats_mode)
            write_stats_line(csvf, 0, stats0)
            writer.write(0, u_curr)
            output_frames += 1

            out_t1 = time.perf_counter()
            output_stats_io_time_s += out_t1 - out_t0

            for step in range(1, cfg.steps + 1):
                dyn_t0 = time.perf_counter()

                update_field(
                    u_curr,
                    u_next,
                    cfg.nx,
                    cfg.ny,
                    dgx,
                    dgy,
                    CX,
                    CY,
                )

                dyn_t1 = time.perf_counter()
                pure_dynamics_time_s += dyn_t1 - dyn_t0

                u_curr, u_next = u_next, u_curr

                if (step % cfg.outputEvery) == 0 or step == cfg.steps:
                    out_t0 = time.perf_counter()

                    stats = compute_stats(u_curr, stats_mode)
                    write_stats_line(csvf, step, stats)
                    writer.write(step, u_curr)
                    output_frames += 1

                    out_t1 = time.perf_counter()
                    output_stats_io_time_s += out_t1 - out_t0

    loop_t1 = time.perf_counter()
    total_wall_t1 = time.perf_counter()

    loop_total_time_s = loop_t1 - loop_t0
    total_wall_time_s = total_wall_t1 - total_wall_t0

    updates = float(cfg.nx - 2) * float(cfg.ny - 2) * float(cfg.steps)

    # --------------------------------------------------------
    # Reporting
    # --------------------------------------------------------
    print(f"Grid:                         {cfg.nx} x {cfg.ny}")
    print(f"Measured points:              {len(cfg.measured)}")
    print(f"Max iterations:               {cfg.maxIters}")
    print(f"Time steps:                   {cfg.steps}")
    print(f"Snapshot every:               {cfg.outputEvery} step(s)")
    print(f"Output frames:                {output_frames}")
    print(f"Stats mode:                   {stats_mode}")
    print(f"Numba threads:                {get_num_threads()}")
    print(f"NUMBA_NUM_THREADS env:        {os.environ.get('NUMBA_NUM_THREADS', '(not set)')}")
    print(f"HDF5 chunk tile:              {min(cfg.ny, h5_tile_y)} x {min(cfg.nx, h5_tile_x)}")
    print(f"Weight field time:            {weight_time_s:.6f} s")
    print(f"Init field time:              {init_time_s:.6f} s")
    print(f"Pure dynamics compute time:   {pure_dynamics_time_s:.6f} s")
    print(f"Stats + CSV + HDF5 time:      {output_stats_io_time_s:.6f} s")
    print(f"Dynamics loop total time:     {loop_total_time_s:.6f} s")
    print(f"Total measured wall time:     {total_wall_time_s:.6f} s")

    if cfg.steps > 0 and pure_dynamics_time_s > 0.0:
        print(
            f"Pure dynamics performance:    "
            f"{updates / pure_dynamics_time_s / 1e9:.6f} GLUP/s"
        )

    if cfg.steps > 0 and loop_total_time_s > 0.0:
        print(
            f"End-to-end loop performance:  "
            f"{updates / loop_total_time_s / 1e9:.6f} GLUP/s"
        )

    print(f"Mean discrepancy:             {discrepancy:.15g}")
    print("Simulation completed successfully.")


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="CPU-only CUDA-shaped Numba multicore cooling solver"
    )

    parser.add_argument(
        "input",
        nargs="?",
        default="Cooling.inp",
        help="Input file",
    )

    parser.add_argument(
        "h5",
        nargs="?",
        default="cooling.h5",
        help="Output HDF5 file",
    )

    parser.add_argument(
        "csv",
        nargs="?",
        default="Statistics.csv",
        help="Output CSV file",
    )

    parser.add_argument(
        "outputEvery",
        nargs="?",
        type=int,
        default=None,
        help="Snapshot cadence override",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of Numba CPU threads",
    )

    parser.add_argument(
        "--stats",
        choices=("accurate", "fast"),
        default="fast",
        help="Statistics mode: accurate uses longdouble, fast uses NumPy mean/std",
    )

    parser.add_argument(
        "--h5-tile-y",
        type=int,
        default=256,
        help="HDF5 chunk tile size in y",
    )

    parser.add_argument(
        "--h5-tile-x",
        type=int,
        default=256,
        help="HDF5 chunk tile size in x",
    )

    args = parser.parse_args()

    if args.threads is not None:
        if args.threads <= 0:
            raise ValueError("--threads must be > 0")

        set_num_threads(args.threads)

    if args.h5_tile_y <= 0 or args.h5_tile_x <= 0:
        raise ValueError("HDF5 tile sizes must be > 0")

    cfg = read_input(args.input)

    if args.outputEvery is not None:
        cfg.outputEvery = args.outputEvery

    validate_output_every(cfg.outputEvery)
    validate_stats_mode(args.stats)

    print(f"Input file:                   {args.input}")
    print(f"HDF5 output:                  {args.h5}")
    print(f"CSV output:                   {args.csv}")

    run_simulation(
        cfg=cfg,
        h5_file=args.h5,
        csv_file=args.csv,
        stats_mode=args.stats,
        h5_tile_y=args.h5_tile_y,
        h5_tile_x=args.h5_tile_x,
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"CRITICAL ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)

