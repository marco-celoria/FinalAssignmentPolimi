#!/usr/bin/env python3
import argparse
import math
import time
from dataclasses import dataclass
from typing import List, Tuple

import h5py
import numpy as np
from numba import njit, prange


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
# SAFE GRID SIZE
# ============================================================

def safe_grid_size(nx: int, ny: int) -> int:
    if nx <= 0 or ny <= 0:
        raise ValueError("Grid dimensions must be > 0")
    n = nx * ny
    if n <= 0:
        raise OverflowError("Grid size overflow")
    return n


# ============================================================
# INPUT PARSER
# ============================================================

def read_input(fname: str) -> Config:
    tokens: List[str] = []

    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0]
            parts = line.split()
            if parts:
                tokens.extend(parts)

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
        measured.append(MeasuredPoint(next_float(), next_float(), next_float()))

    sreal = next_float()
    simag = next_float()
    dreal = next_float()
    dimag = next_float()
    max_iters = next_int()
    steps = next_int()
    _legacy_ppm_flag = next_int()  # parsed and ignored

    if max_iters <= 0:
        raise RuntimeError("maxIters must be > 0")
    if steps < 0:
        raise RuntimeError("steps must be >= 0")

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
    Mean(measured_value - analytical_field(x,y))
    """
    if not cfg.measured:
        return 0.0

    acc = np.longdouble(0.0)
    for m in cfg.measured:
        acc += np.longdouble(m.v - analytical_field(m.x, m.y))

    return float(acc / np.longdouble(len(cfg.measured)))


def build_cooling_coeffs(nx: int, ny: int, dd: float = 100.0) -> Tuple[float, float, float, float, float, float, float]:
    """
    Matches the cleaned C++ baseline:
      hx = 1 / (nx - 1)
      hy = 1 / (ny - 1)
    """
    if nx < 3 or ny < 3:
        raise ValueError("build_cooling_coeffs: nx and ny must be at least 3")

    hx = 1.0 / float(nx - 1)
    hy = 1.0 / float(ny - 1)

    dgx = -2.0 * (1.0 + dd * hx / (hx * hx + dd))
    dgy = -2.0 * (1.0 + dd * hy / (hy * hy + dd))

    CX = (hx + dd * math.exp(hx)) / (15.0 * dd + hx)
    CY = (hy + dd * math.exp(hy)) / (15.0 * dd + hy)

    return dd, hx, hy, dgx, dgy, CX, CY


# ============================================================
# CUDA-SHAPED NUMBA KERNELS
#
# These are written to resemble future CUDA kernels:
# - flat arrays
# - explicit scalar parameters
# - no Python objects in hot loops
# - interior and boundaries separated
# ============================================================

@njit(parallel=False, fastmath=False)
def compute_weight_kernel(weight: np.ndarray,
                          nx: int, ny: int,
                          x0: float, y0: float,
                          dx: float, dy: float,
                          max_iters: int) -> None:
    """
    Standard Mandelbrot escape-time count:
        z_{n+1} = z_n^2 + c, z_0 = 0
    mapped over the structured grid.
    """
    for j in prange(ny):
        y = y0 + dy * j
        row = j * nx

        for i in range(nx):
            x = x0 + dx * i
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


@njit(parallel=False, fastmath=False)
def initialize_field_kernel(u: np.ndarray,
                            weight: np.ndarray,
                            nx: int, ny: int,
                            x0: float, y0: float,
                            dx: float, dy: float,
                            discrepancy: float,
                            wmin: int, wmax: int) -> None:
    """
    u0(x,y) = 293.16 + 80 * ( discrepancy + analytical_field(x,y) ) * w_norm
    """
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
def update_interior_kernel(u1: np.ndarray,
                           u2: np.ndarray,
                           nx: int, ny: int,
                           dgx: float, dgy: float,
                           CX: float, CY: float) -> None:
    """
    Interior stencil only.
    This maps directly to a future CUDA interior kernel.
    """
    for j in prange(1, ny - 1):
        row = j * nx
        row_up = (j - 1) * nx
        row_dn = (j + 1) * nx

        for i in range(1, nx - 1):
            p = row + i

            u2[p] = (
                CX * (
                    u1[p - 1] +
                    u1[p + 1] +
                    (dgx + 0.5 / CX) * u1[p]
                )
                +
                CY * (
                    u1[row_up + i] +
                    u1[row_dn + i] +
                    (dgy + 0.5 / CY) * u1[p]
                )
            )


@njit(parallel=True, fastmath=False)
def apply_boundary_lr_kernel(u: np.ndarray, nx: int, ny: int) -> None:
    """
    Left/right boundaries excluding corners.
    Separate kernel for CUDA-style decomposition.
    """
    for j in prange(1, ny - 1):
        row = j * nx
        u[row + 0] = u[row + 1]
        u[row + (nx - 1)] = u[row + (nx - 2)]


@njit(parallel=True, fastmath=False)
def apply_boundary_tb_kernel(u: np.ndarray, nx: int, ny: int) -> None:
    """
    Top/bottom boundaries including corners.
    Separate kernel for CUDA-style decomposition.
    """
    top_row = 0
    row1 = nx
    row_nm2 = (ny - 2) * nx
    row_nm1 = (ny - 1) * nx

    for i in prange(nx):
        u[top_row + i] = u[row1 + i]
        u[row_nm1 + i] = u[row_nm2 + i]


def update_field(u1: np.ndarray, u2: np.ndarray,
                 nx: int, ny: int,
                 dgx: float, dgy: float,
                 CX: float, CY: float) -> None:
    """
    Host-side launch sequence.
    This is intentionally structured like future CUDA launch code.
    """
    update_interior_kernel(u1, u2, nx, ny, dgx, dgy, CX, CY)
    apply_boundary_lr_kernel(u2, nx, ny)
    apply_boundary_tb_kernel(u2, nx, ny)


# ============================================================
# HDF5 WRITER
# ============================================================

class H5Writer:
    def __init__(self, fname: str, nx: int, ny: int, batch: int = 32):
        if nx <= 0 or ny <= 0:
            raise ValueError("H5Writer: nx and ny must be > 0")
        if batch <= 0:
            raise ValueError("H5Writer: batch must be > 0")

        self.nx = nx
        self.ny = ny
        self.batch = batch
        self.frame = 0
        self.capacity = batch
        self.closed = False

        self.file = h5py.File(fname, "w")
        self.field = self.file.create_dataset(
            "field",
            shape=(0, ny, nx),
            maxshape=(None, ny, nx),
            chunks=(1, ny, nx),
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

def compute_stats(u: np.ndarray) -> Tuple[float, float, float, float]:
    if u.size == 0:
        raise RuntimeError("compute_stats: empty field")

    mn = float(np.min(u))
    mx = float(np.max(u))

    mean_ld = np.sum(u, dtype=np.longdouble) / np.longdouble(u.size)
    diff_ld = u.astype(np.longdouble) - mean_ld
    std_ld = np.sqrt(np.sum(diff_ld * diff_ld, dtype=np.longdouble) / np.longdouble(u.size))

    return float(mn), float(mean_ld), float(mx), float(std_ld)


def write_stats_header(f) -> None:
    f.write("Step;Min;Mean;Max;Std_dev\n")


def write_stats_line(f, step: int, stats: Tuple[float, float, float, float]) -> None:
    mn, mean, mx, std = stats
    f.write(f"{step};{mn:.15g};{mean:.15g};{mx:.15g};{std:.15g}\n")


# ============================================================
# HOST-SIDE DRIVER
#
# This is intentionally structured so the future CUDA version
# will mainly replace the Numba kernel calls with CUDA kernel
# launches and move arrays to device memory.
# ============================================================

def run_simulation(cfg: Config, h5_file: str, csv_file: str) -> None:
    n = safe_grid_size(cfg.nx, cfg.ny)

    x0, y0, dx, dy = build_domain_params(cfg)
    _dd, _hx, _hy, dgx, dgy, CX, CY = build_cooling_coeffs(cfg.nx, cfg.ny, 100.0)
    discrepancy = compute_discrepancy(cfg)

    # Flat arrays only: GPU-friendly design
    weight = np.empty(n, dtype=np.int32)
    u_curr = np.empty(n, dtype=np.float64)
    u_next = np.empty(n, dtype=np.float64)

    with open(csv_file, "w", encoding="utf-8") as csvf:
        write_stats_header(csvf)

        t0 = time.perf_counter()
        compute_weight_kernel(weight, cfg.nx, cfg.ny, x0, y0, dx, dy, cfg.maxIters)
        t1 = time.perf_counter()

        wmin = int(weight.min())
        wmax = int(weight.max())

        initialize_field_kernel(u_curr, weight, cfg.nx, cfg.ny, x0, y0, dx, dy,
                                discrepancy, wmin, wmax)
        t2 = time.perf_counter()

        with H5Writer(h5_file, cfg.nx, cfg.ny, 32) as writer:
            # Step 0
            writer.write(0, u_curr)
            write_stats_line(csvf, 0, compute_stats(u_curr))

            for step in range(1, cfg.steps + 1):
                update_field(u_curr, u_next, cfg.nx, cfg.ny, dgx, dgy, CX, CY)
                u_curr, u_next = u_next, u_curr

                if (step % cfg.outputEvery) == 0 or step == cfg.steps:
                    writer.write(step, u_curr)
                    write_stats_line(csvf, step, compute_stats(u_curr))

        t3 = time.perf_counter()

    print(f"Grid:              {cfg.nx} x {cfg.ny}")
    print(f"Measured points:   {len(cfg.measured)}")
    print(f"Max iterations:    {cfg.maxIters}")
    print(f"Time steps:        {cfg.steps}")
    print(f"Snapshot every:    {cfg.outputEvery} step(s)")
    print(f"Weight field time: {t1 - t0} s")
    print(f"Init field time:   {t2 - t1} s")
    print(f"Dynamics + I/O:    {t3 - t2} s")

    if cfg.steps > 0 and (t3 - t2) > 0.0:
        updates = float(cfg.nx - 2) * float(cfg.ny - 2) * float(cfg.steps)
        print(f"Performance:       {updates / (t3 - t2) / 1e9} GLUP/s")

    print(f"Mean discrepancy:  {discrepancy}")
    print("Simulation completed successfully.")


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="CUDA-shaped Python + Numba multicore cooling solver"
    )
    parser.add_argument("input", nargs="?", default="Cooling.inp",
                        help="Input file (default: Cooling.inp)")
    parser.add_argument("h5", nargs="?", default="cooling.h5",
                        help="Output HDF5 file (default: cooling.h5)")
    parser.add_argument("csv", nargs="?", default="Statistics.csv",
                        help="Output CSV file (default: Statistics.csv)")
    parser.add_argument("outputEvery", nargs="?", type=int, default=None,
                        help="Snapshot cadence override")
    args = parser.parse_args()

    cfg = read_input(args.input)

    if args.outputEvery is not None:
        cfg.outputEvery = args.outputEvery

    if cfg.outputEvery <= 0:
        raise ValueError("outputEvery must be > 0")

    print(f"Input file:        {args.input}")
    print(f"HDF5 output:       {args.h5}")
    print(f"CSV output:        {args.csv}")

    run_simulation(cfg, args.h5, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
