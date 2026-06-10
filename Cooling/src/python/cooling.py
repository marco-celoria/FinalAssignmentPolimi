#!/usr/bin/env python3

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import h5py
import numpy as np


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

    # outputEvery is optional.
    # If hasOutputEvery is False, only the final state is written.
    hasOutputEvery: bool
    outputEvery: Optional[int]

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

    has_output_every = False
    output_every: Optional[int] = None

    # Optional outputEvery.
    #
    # Supported endings:
    #
    #   maxIters steps
    #
    # or:
    #
    #   maxIters steps outputEvery
    #
    # If outputEvery is absent, only the final state is written.
    if pos < len(tokens):
        output_every = next_int()
        has_output_every = True

        validate_output_every(output_every)

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
        hasOutputEvery=has_output_every,
        outputEvery=output_every,
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
# PURE NUMPY KERNELS
# ============================================================

def compute_weight_field(
    nx: int,
    ny: int,
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    max_iters: int,
) -> np.ndarray:
    """
    Pure NumPy version of the Mandelbrot-style iteration kernel.

    Returns:
        weight: int32 array of shape (ny, nx)
    """
    x = x0 + dx * np.arange(nx, dtype=np.float64)
    y = y0 + dy * np.arange(ny, dtype=np.float64)

    c_real = np.broadcast_to(x[None, :], (ny, nx)).copy()
    c_imag = np.broadcast_to(y[:, None], (ny, nx)).copy()

    z_real = np.zeros((ny, nx), dtype=np.float64)
    z_imag = np.zeros((ny, nx), dtype=np.float64)
    weight = np.zeros((ny, nx), dtype=np.int32)

    active = np.ones((ny, nx), dtype=bool)

    for it in range(max_iters):
        if not np.any(active):
            break

        zr = z_real[active]
        zi = z_imag[active]
        cr = c_real[active]
        ci = c_imag[active]

        zr_new = zr * zr - zi * zi + cr
        zi_new = 2.0 * zr * zi + ci

        z_real[active] = zr_new
        z_imag[active] = zi_new

        weight[active] = it + 1
        active[active] = (zr_new * zr_new + zi_new * zi_new) <= 4.0

    return weight


def initialize_field(
    weight: np.ndarray,
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    discrepancy: float,
    wmin: int,
    wmax: int,
) -> np.ndarray:
    """
    Initialize the temperature/field using the analytical field and weight normalization.
    """
    ny, nx = weight.shape
    x = x0 + dx * np.arange(nx, dtype=np.float64)
    y = y0 + dy * np.arange(ny, dtype=np.float64)

    X, Y = np.meshgrid(x, y, indexing="xy")
    F = (X * X * X + Y * Y * Y) / 6.0

    denom = float(wmax - wmin) if wmax > wmin else 1.0
    wnorm = (weight.astype(np.float64) - float(wmin)) / denom

    u = 293.16 + 80.0 * (discrepancy + F) * wnorm
    return u.astype(np.float64, copy=False)


def update_field(
    u1: np.ndarray,
    u2: np.ndarray,
    dgx: float,
    dgy: float,
    CX: float,
    CY: float,
) -> None:
    """
    One explicit update step using vectorized NumPy slicing.
    Boundary handling matches the original order:
      1) interior update
      2) left/right boundaries excluding corners
      3) top/bottom boundaries including corners
    """
    # Interior
    u2[1:-1, 1:-1] = (
        CX * (
            u1[1:-1, :-2]
            + u1[1:-1, 2:]
            + (dgx + 0.5 / CX) * u1[1:-1, 1:-1]
        )
        + CY * (
            u1[:-2, 1:-1]
            + u1[2:, 1:-1]
            + (dgy + 0.5 / CY) * u1[1:-1, 1:-1]
        )
    )

    # Left/right boundaries excluding corners
    u2[1:-1, 0] = u2[1:-1, 1]
    u2[1:-1, -1] = u2[1:-1, -2]

    # Top/bottom boundaries including corners
    u2[0, :] = u2[1, :]
    u2[-1, :] = u2[-2, :]


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

    def write(self, step_number: int, field_2d: np.ndarray) -> None:
        if self.closed:
            raise RuntimeError("H5Writer: write() called after close()")

        if field_2d.shape != (self.ny, self.nx):
            raise RuntimeError("H5Writer: field shape mismatch")

        if self.frame >= self.capacity:
            self.capacity += self.batch
            self._extend(self.capacity)

        self.field[self.frame, :, :] = field_2d
        self.step[self.frame] = np.int32(step_number)
        self.frame += 1

    def close(self) -> None:
        if self.closed:
            return

        if self.frame != self.capacity:
            self._extend(self.frame)

        self.file.flush()
        self.file.close()
        self.closed = True

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

    - min/max in float64
    - mean/std using longdouble accumulation
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

    if cfg.hasOutputEvery:
        if cfg.outputEvery is None:
            raise ValueError("Internal error: hasOutputEvery is true but outputEvery is None")
        validate_output_every(cfg.outputEvery)

    n = validated_grid_size(cfg.nx, cfg.ny)

    x0, y0, dx, dy = build_domain_params(cfg)
    _, _, _, dgx, dgy, CX, CY = build_cooling_coeffs(dx, dy, 100.0)
    discrepancy = compute_discrepancy(cfg)

    # Field arrays
    weight = np.empty((cfg.ny, cfg.nx), dtype=np.int32)
    u_curr = np.empty((cfg.ny, cfg.nx), dtype=np.float64)
    u_next = np.empty((cfg.ny, cfg.nx), dtype=np.float64)

    total_wall_t0 = time.perf_counter()

    # --------------------------------------------------------
    # Weight field
    # --------------------------------------------------------
    t0 = time.perf_counter()

    weight[:, :] = compute_weight_field(
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

    u_curr[:, :] = initialize_field(
        weight,
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
    last_written_step: Optional[int] = None

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

            def write_output_frame(step: int) -> None:
                nonlocal output_frames
                nonlocal output_stats_io_time_s
                nonlocal last_written_step

                if last_written_step == step:
                    return

                out_t0 = time.perf_counter()

                stats = compute_stats(u_curr, stats_mode)
                write_stats_line(csvf, step, stats)
                writer.write(step, u_curr)
                output_frames += 1

                out_t1 = time.perf_counter()
                output_stats_io_time_s += out_t1 - out_t0

                last_written_step = step

            # If outputEvery was explicitly specified, keep traditional behavior:
            # write the initial condition at step 0.
            if cfg.hasOutputEvery:
                write_output_frame(0)

            for step in range(1, cfg.steps + 1):
                dyn_t0 = time.perf_counter()

                update_field(
                    u_curr,
                    u_next,
                    dgx,
                    dgy,
                    CX,
                    CY,
                )

                dyn_t1 = time.perf_counter()
                pure_dynamics_time_s += dyn_t1 - dyn_t0

                u_curr, u_next = u_next, u_curr

                if cfg.hasOutputEvery:
                    assert cfg.outputEvery is not None

                    if (step % cfg.outputEvery) == 0:
                        write_output_frame(step)

            # Always write the final state.
            write_output_frame(cfg.steps)

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

    if cfg.hasOutputEvery:
        assert cfg.outputEvery is not None
        print(f"Snapshot every:               {cfg.outputEvery} step(s)")
    else:
        print("Snapshot every:               final step only")

    print(f"Output frames:                {output_frames}")
    print(f"Stats mode:                   {stats_mode}")
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
        description="CPU-only NumPy/HDF5 cooling solver (no Numba)"
    )

    parser.add_argument(
        "input",
        nargs="?",
        default="input/Cooling.in",
        help="Input file",
    )

    parser.add_argument(
        "h5",
        nargs="?",
        default="output/Cooling.h5",
        help="Output HDF5 file",
    )

    parser.add_argument(
        "csv",
        nargs="?",
        default="output/Statistics.csv",
        help="Output CSV file",
    )

    parser.add_argument(
        "outputEvery",
        nargs="?",
        type=int,
        default=None,
        help=(
            "Optional snapshot cadence override. "
            "If omitted and absent from input file, only the final state is written."
        ),
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

    if args.h5_tile_y <= 0 or args.h5_tile_x <= 0:
        raise ValueError("HDF5 tile sizes must be > 0")

    cfg = read_input(args.input)

    # Command-line outputEvery overrides input-file outputEvery.
    # If neither specifies outputEvery, only the final state is written.
    if args.outputEvery is not None:
        validate_output_every(args.outputEvery)
        cfg.outputEvery = args.outputEvery
        cfg.hasOutputEvery = True

    if cfg.hasOutputEvery:
        assert cfg.outputEvery is not None
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
