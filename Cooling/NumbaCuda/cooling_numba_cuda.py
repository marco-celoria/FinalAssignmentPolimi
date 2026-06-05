#!/usr/bin/env python3
import argparse
import math
import time
from dataclasses import dataclass
from typing import List, Tuple
import sys

import h5py
import numpy as np
from numba import cuda


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
# VALIDATE GRID SIZE
# ============================================================

def validated_grid_size(nx: int, ny: int) -> int:
    if nx <= 0 or ny <= 0:
        raise ValueError("Grid dimensions must be > 0")
    return nx * ny

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
    x0 = cfg.Sreal
    y0 = cfg.Simag
    dx = cfg.Dreal / float(cfg.nx - 1)
    dy = cfg.Dimag / float(cfg.ny - 1)
    return x0, y0, dx, dy


def analytical_field(x: float, y: float) -> float:
    return (x * x * x + y * y * y) / 6.0


def compute_discrepancy(cfg: Config) -> float:
    if not cfg.measured:
        return 0.0

    acc = np.longdouble(0.0)
    for m in cfg.measured:
        acc += np.longdouble(m.v - analytical_field(m.x, m.y))

    return float(acc / np.longdouble(len(cfg.measured)))


#def build_cooling_coeffs(nx: int, ny: int, dd: float = 100.0) -> Tuple[float, float, float, float, float, float, float]:
#    if nx < 3 or ny < 3:
#        raise ValueError("build_cooling_coeffs: nx and ny must be at least 3")
#    if dd <= 0.0:
#        raise ValueError("build_cooling_coeffs: dd must be > 0")
#
#    hx = 1.0 / float(nx - 1)
#    hy = 1.0 / float(ny - 1)
#
#    dgx = -2.0 * (1.0 + dd * hx / (hx * hx + dd))
#    dgy = -2.0 * (1.0 + dd * hy / (hy * hy + dd))
#
#    CX = (hx + dd * math.exp(hx)) / (15.0 * dd + hx)
#    CY = (hy + dd * math.exp(hy)) / (15.0 * dd + hy)
#
#    return dd, hx, hy, dgx, dgy, CX, CY

def build_cooling_coeffs(dx: float, dy: float, dd: float = 100.0):
    # 1. Validation Checks
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("build_cooling_coeffs: dx and dy must be > 0")
    if dd <= 0.0:
        raise ValueError("build_cooling_coeffs: dd must be > 0")

    # 2. Assign Grid Spacing
    hx = dx
    hy = dy

    # 3. Calculate Coefficients
    dgx = -2.0 * (1.0 + dd * hx / (hx * hx + dd))
    dgy = -2.0 * (1.0 + dd * hy / (hy * hy + dd))

    CX = (hx + dd * math.exp(hx)) / (15.0 * dd + hx)
    CY = (hy + dd * math.exp(hy)) / (15.0 * dd + hy)

    return dd, hx, hy, dgx, dgy, CX, CY

# ============================================================
# CUDA KERNELS
# ============================================================

@cuda.jit
def compute_weight_kernel(weight, nx, ny, x0, y0, dx, dy, max_iters):
    i, j = cuda.grid(2)
    if i >= nx or j >= ny:
        return

    p = i + j * nx
    x = x0 + dx * i
    y = y0 + dy * j

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


@cuda.jit
def initialize_field_kernel(u, weight, nx, ny, x0, y0, dx, dy, discrepancy, wmin, wmax):
    i, j = cuda.grid(2)
    if i >= nx or j >= ny:
        return

    p = i + j * nx
    x = x0 + dx * i
    y = y0 + dy * j

    denom = float(wmax - wmin) if wmax > wmin else 1.0
    F = (x * x * x + y * y * y) / 6.0
    wnorm = (weight[p] - wmin) / denom

    u[p] = 293.16 + 80.0 * (discrepancy + F) * wnorm


@cuda.jit
def update_interior_kernel(u1, u2, nx, ny, dgx, dgy, CX, CY):
    i, j = cuda.grid(2)
    if i < 1 or i >= nx - 1 or j < 1 or j >= ny - 1:
        return

    p = i + j * nx

    u2[p] = (
        CX * (
            u1[p - 1] +
            u1[p + 1] +
            (dgx + 0.5 / CX) * u1[p]
        )
        +
        CY * (
            u1[p - nx] +
            u1[p + nx] +
            (dgy + 0.5 / CY) * u1[p]
        )
    )


# Left/right edges are updated first.
# Top/bottom then copies from the already-updated edge-adjacent values,
# so corners are implicitly determined by the edge-update order.

@cuda.jit
def apply_boundary_lr_kernel(u, nx, ny):
    j = cuda.grid(1)
    if j < 1 or j >= ny - 1:
        return

    row = j * nx
    u[row + 0] = u[row + 1]
    u[row + (nx - 1)] = u[row + (nx - 2)]


@cuda.jit
def apply_boundary_tb_kernel(u, nx, ny):
    i = cuda.grid(1)
    if i >= nx:
        return

    u[i] = u[nx + i]
    u[(ny - 1) * nx + i] = u[(ny - 2) * nx + i]


# ============================================================
# KERNEL LAUNCH HELPERS
# ============================================================

def launch_compute_weight(d_weight, nx, ny, x0, y0, dx, dy, max_iters, block2d=(16, 16)) -> None:
    grid2d = (
        (nx + block2d[0] - 1) // block2d[0],
        (ny + block2d[1] - 1) // block2d[1],
    )
    compute_weight_kernel[grid2d, block2d](d_weight, nx, ny, x0, y0, dx, dy, max_iters)


def launch_initialize_field(d_u, d_weight, nx, ny, x0, y0, dx, dy,
                            discrepancy, wmin, wmax, block2d=(16, 16)) -> None:
    grid2d = (
        (nx + block2d[0] - 1) // block2d[0],
        (ny + block2d[1] - 1) // block2d[1],
    )
    initialize_field_kernel[grid2d, block2d](
        d_u, d_weight, nx, ny, x0, y0, dx, dy, discrepancy, wmin, wmax
    )


def launch_update_field(d_u1, d_u2, nx, ny, dgx, dgy, CX, CY,
                        block2d=(16, 16), block1d=256) -> None:
    grid2d = (
        (nx + block2d[0] - 1) // block2d[0],
        (ny + block2d[1] - 1) // block2d[1],
    )
    update_interior_kernel[grid2d, block2d](d_u1, d_u2, nx, ny, dgx, dgy, CX, CY)

    grid1d_y = ((ny + block1d - 1) // block1d,)
    apply_boundary_lr_kernel[grid1d_y, block1d](d_u2, nx, ny)

    grid1d_x = ((nx + block1d - 1) // block1d,)
    apply_boundary_tb_kernel[grid1d_x, block1d](d_u2, nx, ny)


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
# ============================================================

def run_simulation(cfg: Config,
                   h5_file: str,
                   csv_file: str,
                   block2d=(16, 16),
                   block1d=256) -> None:
    if not cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Check NVIDIA driver / CUDA runtime / Numba installation."
        )

    n = validated_grid_size(cfg.nx, cfg.ny)
#    x0, y0, dx, dy = build_domain_params(cfg)
#    _dd, _hx, _hy, dgx, dgy, CX, CY = build_cooling_coeffs(cfg.nx, cfg.ny, 100.0)
    x0, y0, dx, dy = build_domain_params(cfg)
    _, _, _, dgx, dgy, CX, CY = build_cooling_coeffs(dx, dy, 100.0)
    discrepancy = compute_discrepancy(cfg)

    # Host array used only when needed for reductions / output
    weight_host = np.empty(n, dtype=np.int32)

    # Device arrays
    d_weight = cuda.device_array(n, dtype=np.int32)
    d_u_curr = cuda.device_array(n, dtype=np.float64)
    d_u_next = cuda.device_array(n, dtype=np.float64)

    with open(csv_file, "w", encoding="utf-8") as csvf:
        write_stats_header(csvf)

        # Weight field
        t0 = time.perf_counter()
        launch_compute_weight(d_weight, cfg.nx, cfg.ny, x0, y0, dx, dy, cfg.maxIters, block2d)
        cuda.synchronize()
        t1 = time.perf_counter()

        # Min/max currently done on host (simple + reliable baseline)
        d_weight.copy_to_host(weight_host)
        wmin = int(weight_host.min())
        wmax = int(weight_host.max())

        # Initialization field
        launch_initialize_field(
            d_u_curr, d_weight,
            cfg.nx, cfg.ny,
            x0, y0, dx, dy,
            discrepancy, wmin, wmax,
            block2d
        )
        cuda.synchronize()
        t2 = time.perf_counter()

        with H5Writer(h5_file, cfg.nx, cfg.ny, 32) as writer:
            # Step 0 output
            u_host = d_u_curr.copy_to_host()
            writer.write(0, u_host)
            write_stats_line(csvf, 0, compute_stats(u_host))

            for step in range(1, cfg.steps + 1):
                launch_update_field(
                    d_u_curr, d_u_next,
                    cfg.nx, cfg.ny,
                    dgx, dgy, CX, CY,
                    block2d, block1d
                )
                d_u_curr, d_u_next = d_u_next, d_u_curr

                if (step % cfg.outputEvery) == 0 or step == cfg.steps:
                    cuda.synchronize()
                    u_host = d_u_curr.copy_to_host()
                    writer.write(step, u_host)
                    write_stats_line(csvf, step, compute_stats(u_host))

        t3 = time.perf_counter()

    print(f"Grid:              {cfg.nx} x {cfg.ny}")
    print(f"Measured points:   {len(cfg.measured)}")
    print(f"Max iterations:    {cfg.maxIters}")
    print(f"Time steps:        {cfg.steps}")
    print(f"Snapshot every:    {cfg.outputEvery} step(s)")
    print(f"CUDA block2d:      {block2d}")
    print(f"CUDA block1d:      {block1d}")
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
        description="Numba CUDA version of the cleaned cooling solver"
    )
    parser.add_argument("input", nargs="?", default="Cooling.inp",
                        help="Input file (default: Cooling.inp)")
    parser.add_argument("h5", nargs="?", default="cooling.h5",
                        help="Output HDF5 file (default: cooling.h5)")
    parser.add_argument("csv", nargs="?", default="Statistics.csv",
                        help="Output CSV file (default: Statistics.csv)")
    parser.add_argument("outputEvery", nargs="?", type=int, default=None,
                        help="Snapshot cadence override")
    parser.add_argument("--block-x", type=int, default=16,
                        help="CUDA block size in x for 2D kernels")
    parser.add_argument("--block-y", type=int, default=16,
                        help="CUDA block size in y for 2D kernels")
    parser.add_argument("--block-1d", type=int, default=256,
                        help="CUDA block size for 1D boundary kernels")
    args = parser.parse_args()

    cfg = read_input(args.input)

    if args.outputEvery is not None:
        cfg.outputEvery = args.outputEvery

    if cfg.outputEvery <= 0:
        raise ValueError("outputEvery must be > 0")
    if args.block_x <= 0 or args.block_y <= 0 or args.block_1d <= 0:
        raise ValueError("Block sizes must be > 0")

    print(f"Input file:        {args.input}")
    print(f"HDF5 output:       {args.h5}")
    print(f"CSV output:        {args.csv}")

    run_simulation(
        cfg,
        args.h5,
        args.csv,
        block2d=(args.block_x, args.block_y),
        block1d=args.block_1d
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"CRITICAL ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)


