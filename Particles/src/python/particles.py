#!/usr/bin/env python3
"""
================================================================================
Particle System Solver - Vectorized Python/NumPy Baseline
================================================================================

Teaching baseline for students less familiar with C/C++. This version uses
NumPy vectorization for reasonable baseline performance while keeping the main
algorithmic structure visible enough to be reimplemented with:

  * Numba CPU:      @numba.njit, parallel=True, numba.prange
  * Numba CUDA:     @numba.cuda.jit kernels

The numerical model and command-line behavior are aligned with the C++17 serial
baseline used in the course.

Official benchmark/no-output mode:

  python3 particles_python_numpy_baseline.py input_final.in none 0

Optional HDF5 correctness/debug run:

  python3 particles_python_numpy_baseline.py input_medium.in particles_python.h5 10
  python3 particles_python_numpy_baseline.py input_final.in reference_python.h5 1000

Command line:

  python3 particles_python_numpy_baseline.py [inputFile] [h5File|none|--no-hdf5] [outputEvery]

Input file format, after removing empty lines and comment-only lines beginning
with '#':

  generatingGridNx
  generatingGridNy
  generatingGridXs
  generatingGridXe
  generatingGridYs
  generatingGridYe
  screenGridNx
  screenGridNy
  screenGridXs
  screenGridXe
  screenGridYs
  screenGridYe
  maxFractalIterations
  timeSteps
  dt
  outputEvery

outputEvery:
  0  means final HDF5 frame only, if HDF5 output is enabled.
  >0 means step 0, every outputEvery steps, and the final step.

Notes for students:
  * The main target remains compute_forces(...), an O(N^2) all-pairs interaction.
  * This NumPy implementation computes the force in i-blocks to avoid building
    full NxN temporary matrices for large particle counts.
  * A Numba/Numba-CUDA solution will usually replace compute_forces(...),
    integrate_vv(...), and possibly compute_generating_field(...).
================================================================================
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple
import math
import numpy as np

try:
    import h5py  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    h5py = None


# ============================================================
# CONSTANTS: part of the numerical model. Do not change.
# ============================================================

K_FORCE = 1.0e-3
EPS = 1.0e-2
EPS2 = EPS * EPS

# Number of target particles processed per block in the vectorized force kernel.
# Memory use is approximately several arrays of shape (force_block_size, N).
DEFAULT_FORCE_BLOCK_SIZE = 256


# ============================================================
# DATA STRUCTURES
# ============================================================


@dataclass
class Grid:
    nx: int = 0
    ny: int = 0
    xs: float = 0.0
    xe: float = 0.0
    ys: float = 0.0
    ye: float = 0.0
    values: Optional[np.ndarray] = None

    def allocate(self) -> None:
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("Grid dimensions must be > 0")
        self.values = np.zeros((self.ny, self.nx), dtype=np.uint64)


@dataclass
class Config:
    max_iters: int = 0
    max_steps: int = 0
    output_every: int = 0
    dt: float = 0.0


@dataclass
class Particles:
    # Structure-of-arrays layout, matching the C++ baseline.
    w: np.ndarray
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray

    @property
    def n(self) -> int:
        return int(self.w.shape[0])


@dataclass
class ValidationQuantities:
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_vx: float = 0.0
    sum_vy: float = 0.0
    weighted_sum_x: float = 0.0
    weighted_sum_y: float = 0.0
    momentum_x: float = 0.0
    momentum_y: float = 0.0
    kinetic_energy: float = 0.0
    potential_like: float = 0.0
    energy_like: float = 0.0


# ============================================================
# INPUT PARSER
# ============================================================


def _clean_input_lines(filename: str) -> list[str]:
    lines: list[str] = []
    with open(filename, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if line:
                lines.append(line)
    return lines


def read_input(filename: str) -> Tuple[Config, Grid, Grid]:
    lines = _clean_input_lines(filename)
    if len(lines) < 16:
        raise RuntimeError(
            f"Input file '{filename}' contains {len(lines)} data lines; expected at least 16"
        )

    g = Grid()
    screen = Grid()
    cfg = Config()

    try:
        g.nx = int(lines[0])
        g.ny = int(lines[1])
        g.xs = float(lines[2])
        g.xe = float(lines[3])
        g.ys = float(lines[4])
        g.ye = float(lines[5])

        screen.nx = int(lines[6])
        screen.ny = int(lines[7])
        screen.xs = float(lines[8])
        screen.xe = float(lines[9])
        screen.ys = float(lines[10])
        screen.ye = float(lines[11])

        cfg.max_iters = int(lines[12])
        cfg.max_steps = int(lines[13])
        cfg.dt = float(lines[14])
        cfg.output_every = int(lines[15])
    except ValueError as e:
        raise RuntimeError(f"Parse error in input file '{filename}': {e}") from e

    if g.nx < 2 or g.ny < 2 or screen.nx < 2 or screen.ny < 2:
        raise RuntimeError("Grids must have at least 2 points in each direction")
    if g.xe <= g.xs or g.ye <= g.ys:
        raise RuntimeError("Invalid generating domain")
    if screen.xe <= screen.xs or screen.ye <= screen.ys:
        raise RuntimeError("Invalid particle/screen domain")
    if cfg.max_iters <= 0 or cfg.max_steps <= 0:
        raise RuntimeError("maxIters and maxSteps must be > 0")
    if cfg.dt <= 0.0:
        raise RuntimeError("dt must be > 0")
    if cfg.output_every < 0:
        raise RuntimeError("outputEvery must be >= 0")

    g.allocate()
    screen.allocate()
    return cfg, g, screen


# ============================================================
# OUTPUT POLICY
# ============================================================


def is_no_hdf5_token(text: str) -> bool:
    return text in {"none", "NONE", "-", "--no-hdf5"}


def should_write_step(step: int, final_step: int, output_every: int) -> bool:
    if step == final_step:
        return True
    if output_every == 0:
        return False
    if step == 0:
        return True
    return (step % output_every) == 0


# ============================================================
# GENERATING FIELD
# ============================================================

def compute_generating_field(g: Grid, max_iter: int) -> None:
    """
    C++-matching scalar Mandelbrot-like generating field.

    This intentionally avoids NumPy vectorized masking/reductions so that the
    iteration counts, thresholding, and particle generation match the C++ serial
    baseline as closely as possible.

    Matches the C++ logic:

        while (iter < maxIter) {
            a = za*za - zb*zb + ca;
            b = 2*za*zb + cb;
            za = a;
            zb = b;

            if (za*za + zb*zb > 4.0) break;

            ++iter;
        }
    """
    if g.values is None:
        raise RuntimeError("compute_generating_field: grid not allocated")

    dx = (g.xe - g.xs) / float(g.nx - 1)
    dy = (g.ye - g.ys) / float(g.ny - 1)

    # Supports both storage layouts:
    #   NumPy baseline: values shape is (ny, nx)
    #   Numba/CUDA Python codes: values shape is flattened (ny*nx,)
    values = g.values

    if values.ndim == 2:
        for j in range(g.ny):
            cb = g.ys + float(j) * dy

            for i in range(g.nx):
                ca = g.xs + float(i) * dx

                za = 0.0
                zb = 0.0
                it = 0

                while it < max_iter:
                    a = za * za - zb * zb + ca
                    b = 2.0 * za * zb + cb

                    za = a
                    zb = b

                    if za * za + zb * zb > 4.0:
                        break

                    it += 1

                values[j, i] = np.uint64(it)

    elif values.ndim == 1:
        expected = g.nx * g.ny
        if values.size != expected:
            raise RuntimeError(
                f"compute_generating_field: flat grid has size {values.size}, "
                f"expected {expected}"
            )

        for j in range(g.ny):
            cb = g.ys + float(j) * dy
            base = j * g.nx

            for i in range(g.nx):
                ca = g.xs + float(i) * dx

                za = 0.0
                zb = 0.0
                it = 0

                while it < max_iter:
                    a = za * za - zb * zb + ca
                    b = 2.0 * za * zb + cb

                    za = a
                    zb = b

                    if za * za + zb * zb > 4.0:
                        break

                    it += 1

                values[base + i] = np.uint64(it)

    else:
        raise RuntimeError(
            f"compute_generating_field: unsupported values.ndim={values.ndim}"
        )




# ============================================================
# PARTICLE GENERATION
# ============================================================


def generate_particles(g: Grid, screen: Grid) -> Particles:
    if g.values is None:
        raise RuntimeError("generate_particles: generating field not allocated")

    values = g.values
    vmax = int(values.max())
    vmin0 = int(values.min())

    # floor((29*vmax + vmin0)/30). Python integers avoid unsigned overflow.
    threshold = (29 * vmax + vmin0) // 30

    jj, ii = np.nonzero(values >= np.uint64(threshold))
    count = int(ii.size)
    if count == 0:
        raise RuntimeError("No particles generated")

    selected_values = values[jj, ii].astype(np.float64)
    w = np.maximum(1.0, 10.0 * selected_values)
    x = screen.xs + (screen.xe - screen.xs) * ii.astype(np.float64) / float(g.nx - 1)
    y = screen.ys + (screen.ye - screen.ys) * jj.astype(np.float64) / float(g.ny - 1)
    vx = np.zeros(count, dtype=np.float64)
    vy = np.zeros(count, dtype=np.float64)

    return Particles(w=w, x=x, y=y, vx=vx, vy=vy)


# ============================================================
# PRIMARY KERNEL: FORCE COMPUTATION
# ============================================================

def compute_forces(
    P: Particles,
    fx: np.ndarray,
    fy: np.ndarray,
    block_size: int = DEFAULT_FORCE_BLOCK_SIZE,
) -> None:
    """
    C++-matching scalar O(N^2) all-pairs force calculation.

    This intentionally avoids NumPy vectorized reductions so that each force
    component is accumulated in the same j-loop order as the C++ serial
    baseline.

    The block_size argument is accepted for drop-in API compatibility but is
    not used.
    """
    N = P.n

    if fx.shape[0] != N or fy.shape[0] != N:
        raise RuntimeError("compute_forces: force array size mismatch")

    x = P.x
    y = P.y
    w = P.w

    for i in range(N):
        xi = x[i]
        yi = y[i]
        wi = w[i]

        fxi = 0.0
        fyi = 0.0

        for j in range(N):
            if i == j:
                continue

            dx = x[j] - xi
            dy = y[j] - yi

            r2 = dx * dx + dy * dy + EPS2
            invr = 1.0 / math.sqrt(r2)
            invr2 = invr * invr
            invr3 = invr2 * invr

            coeff = K_FORCE * wi * w[j] * invr3

            fxi += coeff * dx
            fyi += coeff * dy

        fx[i] = fxi
        fy[i] = fyi



# ============================================================
# VELOCITY-VERLET INTEGRATION
# ============================================================


def integrate_vv(
    P: Particles,
    fx: np.ndarray,
    fy: np.ndarray,
    fx_new: np.ndarray,
    fy_new: np.ndarray,
    dt: float,
    force_block_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    N = P.n
    if fx.shape[0] != N or fy.shape[0] != N or fx_new.shape[0] != N or fy_new.shape[0] != N:
        raise RuntimeError("integrate_vv: force array size mismatch")

    invm = 1.0 / P.w
    P.vx += 0.5 * fx * invm * dt
    P.vy += 0.5 * fy * invm * dt
    P.x += P.vx * dt
    P.y += P.vy * dt

    compute_forces(P, fx_new, fy_new, force_block_size)

    P.vx += 0.5 * fx_new * invm * dt
    P.vy += 0.5 * fy_new * invm * dt

    return fx_new, fy_new, fx, fy


# ============================================================
# SCREEN BUILDING - visualization/debugging only
# ============================================================


def build_screen(screen: Grid, P: Particles, wmin: float, wr: float) -> None:
    if screen.values is None:
        raise RuntimeError("build_screen: screen grid not allocated")
    if wr <= 0.0:
        raise RuntimeError("build_screen: invalid weight range")

    values = screen.values
    values.fill(0)

    invdx = float(screen.nx - 1) / (screen.xe - screen.xs)
    invdy = float(screen.ny - 1) / (screen.ye - screen.ys)

    ix = ((P.x - screen.xs) * invdx).astype(np.int64)
    iy = ((P.y - screen.ys) * invdy).astype(np.int64)
    ix = np.clip(ix, 0, screen.nx - 1)
    iy = np.clip(iy, 0, screen.ny - 1)

    wp = (10.0 * (P.w - wmin) / wr).astype(np.int64)
    wp = np.clip(wp, 0, 1000).astype(np.uint64)

    # Reproduce the 3x3 stencil used by the C++/CUDA references. np.add.at
    # handles repeated indices correctly.
    for dj in (-1, 0, 1):
        jy = iy + dj
        valid_y = (jy >= 0) & (jy < screen.ny)
        for di in (-1, 0, 1):
            jx = ix + di
            valid = valid_y & (jx >= 0) & (jx < screen.nx)
            np.add.at(values, (jy[valid], jx[valid]), wp[valid])


# ============================================================
# VALIDATION QUANTITIES
# ============================================================


def compute_validation_quantities(
    P: Particles,
    block_size: int = DEFAULT_FORCE_BLOCK_SIZE,
) -> ValidationQuantities:
    q = ValidationQuantities()

    q.sum_x = float(np.sum(P.x))
    q.sum_y = float(np.sum(P.y))
    q.sum_vx = float(np.sum(P.vx))
    q.sum_vy = float(np.sum(P.vy))
    q.weighted_sum_x = float(np.sum(P.w * P.x))
    q.weighted_sum_y = float(np.sum(P.w * P.y))
    q.momentum_x = float(np.sum(P.w * P.vx))
    q.momentum_y = float(np.sum(P.w * P.vy))
    q.kinetic_energy = float(0.5 * np.sum(P.w * (P.vx * P.vx + P.vy * P.vy)))

    # Potential-like term is also O(N^2). Compute it in blocks to avoid an NxN
    # temporary matrix. Only upper-triangular pairs i<j are counted.
    N = P.n
    potential = 0.0
    for start in range(0, N, block_size):
        stop = min(start + block_size, N)
        xb = P.x[start:stop, None]
        yb = P.y[start:stop, None]
        wb = P.w[start:stop, None]

        dx = P.x[None, :] - xb
        dy = P.y[None, :] - yb
        r2 = dx * dx + dy * dy + EPS2
        pair = K_FORCE * wb * P.w[None, :] / np.sqrt(r2)

        rows = np.arange(stop - start)[:, None]
        global_i = start + rows
        global_j = np.arange(N)[None, :]
        mask = global_j > global_i
        potential += float(np.sum(pair[mask]))

    q.potential_like = potential
    q.energy_like = q.kinetic_energy + q.potential_like
    return q


def print_validation_quantities(q: ValidationQuantities) -> None:
    print("Final validation quantities:")
    print(f"  sum_x:            {q.sum_x:.17g}")
    print(f"  sum_y:            {q.sum_y:.17g}")
    print(f"  sum_vx:           {q.sum_vx:.17g}")
    print(f"  sum_vy:           {q.sum_vy:.17g}")
    print(f"  weighted_sum_x:   {q.weighted_sum_x:.17g}")
    print(f"  weighted_sum_y:   {q.weighted_sum_y:.17g}")
    print(f"  momentum_x:       {q.momentum_x:.17g}")
    print(f"  momentum_y:       {q.momentum_y:.17g}")
    print(f"  kinetic_energy:   {q.kinetic_energy:.17g}")
    print(f"  potential_like:   {q.potential_like:.17g}")
    print(f"  energy_like:      {q.energy_like:.17g}")


# ============================================================
# HDF5 STREAM WRITER
# ============================================================


class H5StreamWriter:
    def __init__(self, filename: str, nparticles: int, nx: int, ny: int, chunk_frames: int = 64):
        if h5py is None:
            raise RuntimeError("h5py is not available. Use 'none' as output file or install h5py.")
        if nparticles <= 0:
            raise ValueError("H5StreamWriter: nparticles must be > 0")
        if nx <= 0 or ny <= 0:
            raise ValueError("H5StreamWriter: nx and ny must be > 0")

        self.nparticles = nparticles
        self.nx = nx
        self.ny = ny
        self.frame = 0
        self.closed = False

        self.file = h5py.File(filename, "w")
        self.pos = self.file.create_dataset(
            "pos",
            shape=(0, nparticles, 2),
            maxshape=(None, nparticles, 2),
            chunks=(1, nparticles, 2),
            dtype="f8",
        )
        self.vel = self.file.create_dataset(
            "vel",
            shape=(0, nparticles, 2),
            maxshape=(None, nparticles, 2),
            chunks=(1, nparticles, 2),
            dtype="f8",
        )
        self.screen = self.file.create_dataset(
            "screen",
            shape=(0, ny, nx),
            maxshape=(None, ny, nx),
            chunks=(1, min(ny, 256), min(nx, 256)),
            dtype="u8",
        )
        self.step = self.file.create_dataset(
            "step",
            shape=(0,),
            maxshape=(None,),
            chunks=(chunk_frames,),
            dtype="i8",
        )
        self.weight = self.file.create_dataset("weight", shape=(nparticles,), dtype="f8")

    def write_metadata(self, input_file: str, cfg: Config, gen: Grid, screen_grid: Grid) -> None:
        attrs = self.file.attrs
        attrs["application"] = "Particle System Solver - Vectorized Python/NumPy Baseline"
        attrs["format_version"] = "2.0"
        attrs["input_file"] = input_file
        attrs["screen_dataset_note"] = "For visualization/debugging; not recommended for strict grading."
        attrs["particles"] = self.nparticles
        attrs["generating_grid_nx"] = gen.nx
        attrs["generating_grid_ny"] = gen.ny
        attrs["generating_grid_xs"] = gen.xs
        attrs["generating_grid_xe"] = gen.xe
        attrs["generating_grid_ys"] = gen.ys
        attrs["generating_grid_ye"] = gen.ye
        attrs["screen_grid_nx"] = screen_grid.nx
        attrs["screen_grid_ny"] = screen_grid.ny
        attrs["screen_grid_xs"] = screen_grid.xs
        attrs["screen_grid_xe"] = screen_grid.xe
        attrs["screen_grid_ys"] = screen_grid.ys
        attrs["screen_grid_ye"] = screen_grid.ye
        attrs["max_iters"] = cfg.max_iters
        attrs["max_steps"] = cfg.max_steps
        attrs["output_every"] = cfg.output_every
        attrs["dt"] = cfg.dt
        attrs["kForce"] = K_FORCE
        attrs["eps"] = EPS
        attrs["eps2"] = EPS2

    def write_weights(self, P: Particles) -> None:
        if P.n != self.nparticles:
            raise RuntimeError("H5StreamWriter: particle size mismatch in write_weights")
        self.weight[...] = P.w

    def write_frame(self, step_number: int, P: Particles, screen_grid: Grid) -> None:
        if self.closed:
            raise RuntimeError("H5StreamWriter: write after close")
        if screen_grid.values is None:
            raise RuntimeError("H5StreamWriter: screen grid not allocated")
        if P.n != self.nparticles:
            raise RuntimeError("H5StreamWriter: particle size mismatch")

        f = self.frame
        new_size = f + 1
        self.pos.resize((new_size, self.nparticles, 2))
        self.vel.resize((new_size, self.nparticles, 2))
        self.screen.resize((new_size, self.ny, self.nx))
        self.step.resize((new_size,))

        self.pos[f, :, 0] = P.x
        self.pos[f, :, 1] = P.y
        self.vel[f, :, 0] = P.vx
        self.vel[f, :, 1] = P.vy
        self.screen[f, :, :] = screen_grid.values
        self.step[f] = step_number
        self.frame += 1

    def close(self) -> None:
        if not self.closed:
            self.file.flush()
            self.file.close()
            self.closed = True

    def __enter__(self) -> "H5StreamWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ============================================================
# MAIN PROGRAM
# ============================================================


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vectorized Python/NumPy baseline for the Particle System Solver assignment"
    )
    parser.add_argument("input_file", nargs="?", default="Particles.inp")
    parser.add_argument("h5_file", nargs="?", default="none")
    parser.add_argument("output_every", nargs="?", type=int, default=None)
    parser.add_argument(
        "--force-block-size",
        type=int,
        default=DEFAULT_FORCE_BLOCK_SIZE,
        help="target-particle block size used by the vectorized O(N^2) force kernel",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    input_file = args.input_file
    output_file = args.h5_file
    write_hdf5 = not is_no_hdf5_token(output_file)
    if write_hdf5 and h5py is None:
        raise RuntimeError("HDF5 output requested, but h5py is not available. Use 'none' or install h5py.")

    force_block_size = int(args.force_block_size)
    if force_block_size <= 0:
        raise RuntimeError("--force-block-size must be > 0")

    cfg, gen, screen = read_input(input_file)
    if args.output_every is not None:
        if args.output_every < 0:
            raise RuntimeError("outputEvery must be >= 0")
        cfg.output_every = args.output_every

    print(f"Input file:                 {input_file}")
    print(f"HDF5 available:             {'yes' if h5py is not None else 'no'}")
    print(f"HDF5 output:                {output_file if write_hdf5 else 'disabled'}")
    print(f"Benchmark/no-output mode:   {'yes' if not write_hdf5 else 'no'}")
    print(f"Generating grid:            {gen.nx} x {gen.ny}")
    print(f"Screen grid:                {screen.nx} x {screen.ny}")
    print(f"Max iterations:             {cfg.max_iters}")
    print(f"Steps:                      {cfg.max_steps}")
    print(f"dt:                         {cfg.dt:.17g}")
    print(f"Force block size:           {force_block_size}")
    if cfg.output_every == 0:
        print("Output policy:              final frame only, if HDF5 is enabled")
    else:
        print(
            f"Output policy:              step 0, every {cfg.output_every} step(s), "
            "and final step, if HDF5 is enabled"
        )

    # --------------------------------------------------------
    # 1. Generate field
    # --------------------------------------------------------
    t0 = time.perf_counter()
    compute_generating_field(gen, cfg.max_iters)
    mandel_seconds = time.perf_counter() - t0

    # --------------------------------------------------------
    # 2. Generate particles
    # --------------------------------------------------------
    t0 = time.perf_counter()
    P = generate_particles(gen, screen)
    particle_generation_seconds = time.perf_counter() - t0
    N = P.n
    print(f"Particles:                  {N}")

    # --------------------------------------------------------
    # 3. Force arrays and initial force
    # --------------------------------------------------------
    fx = np.zeros(N, dtype=np.float64)
    fy = np.zeros(N, dtype=np.float64)
    fx_new = np.zeros(N, dtype=np.float64)
    fy_new = np.zeros(N, dtype=np.float64)

    t0 = time.perf_counter()
    compute_forces(P, fx, fy, force_block_size)
    init_force_seconds = time.perf_counter() - t0

    wmin = float(P.w.min())
    wmax = float(P.w.max())
    wr = max(wmax - wmin, 1.0)

    h5: Optional[H5StreamWriter] = None
    if write_hdf5:
        h5 = H5StreamWriter(output_file, N, screen.nx, screen.ny)
        h5.write_metadata(input_file, cfg, gen, screen)
        h5.write_weights(P)

    output_frames = 0
    has_last_written_step = False
    last_written_step = -1
    screen_build_seconds = 0.0
    hdf5_write_seconds = 0.0

    def write_output_frame(step: int) -> None:
        nonlocal output_frames, has_last_written_step, last_written_step
        nonlocal screen_build_seconds, hdf5_write_seconds

        if h5 is None:
            return
        if has_last_written_step and step == last_written_step:
            return

        t_screen = time.perf_counter()
        build_screen(screen, P, wmin, wr)
        screen_build_seconds += time.perf_counter() - t_screen

        t_h5 = time.perf_counter()
        h5.write_frame(step, P, screen)
        hdf5_write_seconds += time.perf_counter() - t_h5

        has_last_written_step = True
        last_written_step = step
        output_frames += 1

    # --------------------------------------------------------
    # 4. Simulation loop
    # --------------------------------------------------------
    loop_start = time.perf_counter()
    pure_dynamics_seconds = 0.0

    for step in range(cfg.max_steps):
        if h5 is not None and should_write_step(step, cfg.max_steps, cfg.output_every):
            write_output_frame(step)

        t_dyn = time.perf_counter()
        fx, fy, fx_new, fy_new = integrate_vv(
            P, fx, fy, fx_new, fy_new, cfg.dt, force_block_size
        )
        pure_dynamics_seconds += time.perf_counter() - t_dyn

    if h5 is not None:
        write_output_frame(cfg.max_steps)
        h5.close()

    loop_wall_seconds = time.perf_counter() - loop_start

    # --------------------------------------------------------
    # 5. Validation
    # --------------------------------------------------------
    t0 = time.perf_counter()
    validation = compute_validation_quantities(P, force_block_size)
    validation_seconds = time.perf_counter() - t0

    # --------------------------------------------------------
    # 6. Reporting
    # --------------------------------------------------------
    interactions = float(N) * float(N - 1) * float(cfg.max_steps)
    giga_interactions = interactions / 1.0e9

    print("Simulation completed successfully.")
    print(f"Output frames:                   {output_frames}")
    print(f"Mandelbrot wall time:            {mandel_seconds:.17g} s")
    print(f"Particle generation wall time:   {particle_generation_seconds:.17g} s")
    print(f"Initial force wall time:         {init_force_seconds:.17g} s")
    print(f"Pure dynamics time:              {pure_dynamics_seconds:.17g} s")
    print(f"Screen build time:               {screen_build_seconds:.17g} s")
    print(f"HDF5 write time:                 {hdf5_write_seconds:.17g} s")
    print(f"Validation time:                 {validation_seconds:.17g} s")
    print(f"Loop wall time:                  {loop_wall_seconds:.17g} s")

    if cfg.max_steps > 0 and pure_dynamics_seconds > 0.0:
        print(
            f"Pure dynamics performance:  "
            f"{giga_interactions / pure_dynamics_seconds:.17g} GInteractions/s"
        )
    if cfg.max_steps > 0 and loop_wall_seconds > 0.0:
        print(
            f"Loop end-to-end performance: "
            f"{giga_interactions / loop_wall_seconds:.17g} GInteractions/s"
        )

    print_validation_quantities(validation)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

