#!/usr/bin/env python3
"""
================================================================================
Particle System Solver - Python/NumPy Baseline
================================================================================

Teaching baseline for students less familiar with C/C++, to be reimplemented with:

  * Numba CPU:      @numba.njit, parallel=True, numba.prange and/or
  * Numba CUDA:     @numba.cuda.jit kernels

Official benchmark/no-output mode:

  python3 ./path/to/particles.py input/Particles.in none 0

Optional HDF5 correctness/debug run:

  python3 ./path/to/particles.py input/Particles.in output/Particles_python.h5
  python3 ./path/to/particles.py input/Particles.in output/Particles_python.h5 1000

Command line:

  python3 ./path/to/particles.py [inputFile] [h5File|none|--no-hdf5] [outputEvery]

Input file format:

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
================================================================================
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

try:
    import h5py  # type: ignore
except ImportError:
    h5py = None


# ============================================================
# CONSTANTS
# ============================================================

K_FORCE = 1.0e-3
EPS = 1.0e-2
EPS2 = EPS * EPS


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
    values: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint64))

    def allocate(self) -> None:
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("Grid dimensions must be > 0")
        self.values = np.zeros(self.nx * self.ny, dtype=np.uint64)


@dataclass
class Config:
    max_iters: int = 0
    max_steps: int = 0
    output_every: int = 0
    dt: float = 0.0


@dataclass
class Particles:
    n: int = 0
    w:  np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    x:  np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    y:  np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    vx: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    vy: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    def resize(self, n_particles: int) -> None:
        if n_particles <= 0:
            raise ValueError("Particles.resize: number of particles must be > 0")

        self.n  = int(n_particles)
        self.w  = np.empty(self.n, dtype=np.float64)
        self.x  = np.empty(self.n, dtype=np.float64)
        self.y  = np.empty(self.n, dtype=np.float64)
        self.vx = np.zeros(self.n, dtype=np.float64)
        self.vy = np.zeros(self.n, dtype=np.float64)


@dataclass
class ValidationQuantities:
    sum_x:  float = 0.0
    sum_y:  float = 0.0
    sum_vx: float = 0.0
    sum_vy: float = 0.0
    weighted_sum_x: float = 0.0
    weighted_sum_y: float = 0.0
    momentum_x:     float = 0.0
    momentum_y:     float = 0.0
    kinetic_energy: float = 0.0
    potential_like: float = 0.0
    energy_like:    float = 0.0


# ============================================================
# INPUT PARSER
# ============================================================

def _clean_input_lines(filename: str) -> list[str]:
    lines: list[str] = []

    try:
        with open(filename, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.split("#", 1)[0].strip()
                if line:
                    lines.append(line)
    except OSError as exc:
        raise RuntimeError(f"Cannot open input file: {filename}") from exc

    return lines


def read_input(filename: str) -> Tuple[Config, Grid, Grid]:
    lines = _clean_input_lines(filename)

    if len(lines) < 16:
        raise RuntimeError(f"Input file '{filename}' contains {len(lines)} data lines; expected at least 16")

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
    except ValueError as exc:
        raise RuntimeError(f"Parse error in input file '{filename}': {exc}") from exc

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
    Mandelbrot-like generating field.
    """
    if g.values.size != g.nx * g.ny:
        raise RuntimeError("compute_generating_field: grid not allocated or size mismatch")
    dx = (g.xe - g.xs) / float(g.nx - 1)
    dy = (g.ye - g.ys) / float(g.ny - 1)
    values = g.values
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


# ============================================================
# PARTICLE GENERATION
# ============================================================

def generate_particles(g: Grid, screen: Grid) -> Particles:
    if g.values.size == 0:
        raise RuntimeError("generate_particles: generating field not allocated")

    vmax = int(np.max(g.values))
    vmin0 = int(np.min(g.values))
    # floor((29*vmax + vmin0)/30). Python integers avoid unsigned overflow.
    threshold = (29 * vmax + vmin0) // 30
    vals2 = g.values.reshape(g.ny, g.nx)
    j_idx, i_idx = np.nonzero(vals2 >= np.uint64(threshold))
    count = int(i_idx.size)
    if count == 0:
        raise RuntimeError("No particles generated")

    p = Particles()
    p.resize(count)
    selected_vals = vals2[j_idx, i_idx].astype(np.float64)
    p.w[:] = np.maximum(1.0, 10.0 * selected_vals)
    p.x[:] = screen.xs + (screen.xe - screen.xs) * i_idx.astype(np.float64) / float(g.nx - 1)
    p.y[:] = screen.ys + (screen.ye - screen.ys) * j_idx.astype(np.float64) / float(g.ny - 1)
    return p


# ============================================================
# PRIMARY KERNEL: FORCE COMPUTATION
# ============================================================

def compute_forces(
    p: Particles,
    fx: np.ndarray,
    fy: np.ndarray,
) -> None:
    """
    O(N^2) all-pairs force calculation.
    """
    n_particles = p.n

    if fx.shape[0] != n_particles or fy.shape[0] != n_particles:
        raise RuntimeError("compute_forces: force array size mismatch")

    x = p.x
    y = p.y
    w = p.w
    for i in range(n_particles):
        xi  = x[i]
        yi  = y[i]
        wi  = w[i]
        fxi = 0.0
        fyi = 0.0
        for j in range(n_particles):
            if i == j:
                continue
            dx = x[j] - xi
            dy = y[j] - yi
            r2 = dx * dx + dy * dy + EPS2
            invr  = 1.0 / math.sqrt(r2)
            invr3 = invr * invr * invr
            coeff = K_FORCE * wi * w[j] * invr3
            fxi  += coeff * dx
            fyi  += coeff * dy
        fx[i] = fxi
        fy[i] = fyi


# ============================================================
# VELOCITY-VERLET INTEGRATION
# ============================================================

def half_kick_drift(
    p: Particles,
    fx: np.ndarray,
    fy: np.ndarray,
    dt: float,
) -> None:
    n_particles = p.n
    for i in range(n_particles):
        invm = 1.0 / p.w[i]
        p.vx[i] += 0.5 * fx[i] * invm * dt
        p.vy[i] += 0.5 * fy[i] * invm * dt
        p.x[i]  += p.vx[i] * dt
        p.y[i]  += p.vy[i] * dt


def half_kick(
    p: Particles,
    fx_new: np.ndarray,
    fy_new: np.ndarray,
    dt: float,
) -> None:
    n_particles = p.n
    for i in range(n_particles):
        invm = 1.0 / p.w[i]
        p.vx[i] += 0.5 * fx_new[i] * invm * dt
        p.vy[i] += 0.5 * fy_new[i] * invm * dt


def integrate_vv(
    p: Particles,
    fx: np.ndarray,
    fy: np.ndarray,
    fx_new: np.ndarray,
    fy_new: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if (
        fx.shape[0] != p.n
        or fy.shape[0] != p.n
        or fx_new.shape[0] != p.n
        or fy_new.shape[0] != p.n
    ):
        raise RuntimeError("integrate_vv: force array size mismatch")

    half_kick_drift(p, fx, fy, dt)
    compute_forces(p, fx_new, fy_new)
    half_kick(p, fx_new, fy_new, dt)
    # Swap old/new force buffers.
    return fx_new, fy_new, fx, fy


# ============================================================
# SCREEN BUILDING
# ============================================================

def build_screen(screen: Grid, p: Particles, wmin: float, wr: float) -> None:
    """
    Build visualization/debugging screen.
    Uses np.add.at to correctly handle repeated particle contributions.
    """
    if screen.values.size != screen.nx * screen.ny:
        raise RuntimeError("build_screen: screen grid not allocated or size mismatch")
    if wr <= 0.0:
        raise RuntimeError("build_screen: invalid weight range")

    values = screen.values
    values.fill(np.uint64(0))
    invdx = float(screen.nx - 1) / (screen.xe - screen.xs)
    invdy = float(screen.ny - 1) / (screen.ye - screen.ys)
    ix = ((p.x - screen.xs) * invdx).astype(np.int64)
    iy = ((p.y - screen.ys) * invdy).astype(np.int64)
    ix = np.clip(ix, 0, screen.nx - 1)
    iy = np.clip(iy, 0, screen.ny - 1)
    wp = (10.0 * (p.w - wmin) / wr).astype(np.int64)
    wp = np.clip(wp, 0, 1000).astype(np.uint64)
    for dj in (-1, 0, 1):
        jy = iy + dj
        valid_y = (jy >= 0) & (jy < screen.ny)
        for di in (-1, 0, 1):
            jx = ix + di
            valid = valid_y & (jx >= 0) & (jx < screen.nx)
            flat_idx = jy[valid] * screen.nx + jx[valid]
            np.add.at(values, flat_idx, wp[valid])


# ============================================================
# VALIDATION QUANTITIES
# ============================================================

def compute_validation_quantities(p: Particles, block_size: int = 1024) -> ValidationQuantities:
    q = ValidationQuantities()
    q.sum_x  = float(np.sum(p.x,  dtype=np.float64))
    q.sum_y  = float(np.sum(p.y,  dtype=np.float64))
    q.sum_vx = float(np.sum(p.vx, dtype=np.float64))
    q.sum_vy = float(np.sum(p.vy, dtype=np.float64))
    q.weighted_sum_x = float(np.sum(p.w * p.x, dtype=np.float64))
    q.weighted_sum_y = float(np.sum(p.w * p.y, dtype=np.float64))
    q.momentum_x = float(np.sum(p.w * p.vx, dtype=np.float64))
    q.momentum_y = float(np.sum(p.w * p.vy, dtype=np.float64))
    q.kinetic_energy = float(0.5 * np.sum(p.w * (p.vx * p.vx + p.vy * p.vy), dtype=np.float64))
    # Potential-like term is O(N^2). Compute in blocks to avoid NxN memory.
    n_particles = p.n
    potential   = 0.0
    for start in range(0, n_particles, block_size):
        stop = min(start + block_size, n_particles)
        xb = p.x[start:stop, None]
        yb = p.y[start:stop, None]
        wb = p.w[start:stop, None]
        dx = p.x[None, :] - xb
        dy = p.y[None, :] - yb
        r2 = dx * dx + dy * dy + EPS2
        pair = K_FORCE * wb * p.w[None, :] / np.sqrt(r2)
        rows = np.arange(stop - start)[:, None]
        global_i = start + rows
        global_j = np.arange(n_particles)[None, :]
        mask = global_j > global_i
        potential += float(np.sum(pair[mask], dtype=np.float64))
    q.potential_like = potential
    q.energy_like    = q.kinetic_energy + q.potential_like
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
    def __init__(
        self,
        filename:   str,
        nparticles: int,
        nx: int,
        ny: int,
        chunk_frames:  int = 64,
        screen_tile_y: int = 256,
        screen_tile_x: int = 256,
    ):
        if h5py is None:
            raise RuntimeError("h5py is not available. Use 'none' as output file or install h5py.")
        if nparticles <= 0:
            raise ValueError("H5StreamWriter: nparticles must be > 0")
        if nx <= 0 or ny <= 0:
            raise ValueError("H5StreamWriter: nx and ny must be > 0")
        if chunk_frames <= 0:
            raise ValueError("H5StreamWriter: chunk_frames must be > 0")
        if screen_tile_y <= 0 or screen_tile_x <= 0:
            raise ValueError("H5StreamWriter: screen tile sizes must be > 0")
        self.nparticles = int(nparticles)
        self.nx = int(nx)
        self.ny = int(ny)
        self.chunk_frames = int(chunk_frames)
        self.capacity = int(chunk_frames)
        self.frame = 0
        self.closed = False
        self.file = h5py.File(filename, "w")

        screen_chunk_y = min(self.ny, screen_tile_y)
        screen_chunk_x = min(self.nx, screen_tile_x)

        self.pos = self.file.create_dataset(
            "/pos",
            shape=(0, self.nparticles, 2),
            maxshape=(None, self.nparticles, 2),
            chunks=(1, self.nparticles, 2),
            dtype=np.float64,
        )

        self.vel = self.file.create_dataset(
            "/vel",
            shape=(0, self.nparticles, 2),
            maxshape=(None, self.nparticles, 2),
            chunks=(1, self.nparticles, 2),
            dtype=np.float64,
        )

        self.screen = self.file.create_dataset(
            "/screen",
            shape=(0, self.ny, self.nx),
            maxshape=(None, self.ny, self.nx),
            chunks=(1, screen_chunk_y, screen_chunk_x),
            dtype=np.uint64,
        )

        self.step = self.file.create_dataset(
            "/step",
            shape=(0,),
            maxshape=(None,),
            chunks=(self.chunk_frames,),
            dtype=np.int64,
        )

        self.weight = self.file.create_dataset(
            "/weight",
            shape=(self.nparticles,),
            dtype=np.float64,
        )

        self._extend(self.capacity)

    def _extend(self, new_size: int) -> None:
        self.pos.resize((new_size, self.nparticles, 2))
        self.vel.resize((new_size, self.nparticles, 2))
        self.screen.resize((new_size, self.ny, self.nx))
        self.step.resize((new_size,))

    def _shrink_to_fit(self) -> None:
        self.pos.resize((self.frame, self.nparticles, 2))
        self.vel.resize((self.frame, self.nparticles, 2))
        self.screen.resize((self.frame, self.ny, self.nx))
        self.step.resize((self.frame,))

    def write_metadata(self, input_file: str, cfg: Config, gen: Grid, screen_grid: Grid) -> None:
        attrs = self.file.attrs
        attrs["application"] = "Particle System Solver - Python/NumPy Baseline"
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
        attrs["eps"]  = EPS
        attrs["eps2"] = EPS2

    def write_weights(self, p: Particles) -> None:
        if p.n != self.nparticles:
            raise RuntimeError("H5StreamWriter: particle size mismatch in write_weights")
        self.weight[...] = p.w

    def write_frame(self, step_number: int, p: Particles, screen_grid: Grid) -> None:
        if self.closed:
            raise RuntimeError("H5StreamWriter: write after close")
        if p.n != self.nparticles:
            raise RuntimeError("H5StreamWriter: particle size mismatch")
        if screen_grid.values.size != self.nx * self.ny:
            raise RuntimeError("H5StreamWriter: screen grid size mismatch")

        if self.frame >= self.capacity:
            self.capacity += self.chunk_frames
            self._extend(self.capacity)

        f = self.frame
        self.pos[f, :, 0] = p.x
        self.pos[f, :, 1] = p.y
        self.vel[f, :, 0] = p.vx
        self.vel[f, :, 1] = p.vy
        self.screen[f, :, :] = screen_grid.values.reshape(self.ny, self.nx)
        self.step[f] = np.int64(step_number)
        self.frame += 1

    def close(self) -> None:
        if not self.closed:
            self._shrink_to_fit()
            self.file.flush()
            self.file.close()
            self.closed = True

    def __enter__(self) -> "H5StreamWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


# ============================================================
# CLI
# ============================================================

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python/NumPy baseline for the Particle System Solver assignment")
    parser.add_argument("input_file",   nargs="?", default="input/Particles.in")
    parser.add_argument("h5_file",      nargs="?", default="none")
    parser.add_argument("output_every", nargs="?", type=int, default=None)
    return parser.parse_args(argv)


# ============================================================
# MAIN PROGRAM
# ============================================================

def main(argv: list[str]) -> int:
    args = parse_args(argv)
    input_file  = args.input_file
    output_file = args.h5_file
    write_hdf5 = not is_no_hdf5_token(output_file)
    if write_hdf5 and h5py is None:
        raise RuntimeError("HDF5 output requested, but h5py is not available. Use 'none' or install h5py.")

    cfg, gen, screen = read_input(input_file)
    if args.output_every is not None:
        if args.output_every < 0:
            raise RuntimeError("outputEvery must be >= 0")
        cfg.output_every = int(args.output_every)

    print(f"Input file:                 {input_file}")
    print(f"HDF5 available:             {'yes' if h5py is not None else 'no'}")
    print(f"HDF5 output:                {output_file if write_hdf5 else 'disabled'}")
    print(f"Benchmark/no-output mode:   {'yes' if not write_hdf5 else 'no'}")
    print(f"Generating grid:            {gen.nx} x {gen.ny}")
    print(f"Screen grid:                {screen.nx} x {screen.ny}")
    print(f"Max iterations:             {cfg.max_iters}")
    print(f"Steps:                      {cfg.max_steps}")
    print(f"dt:                         {cfg.dt:.17g}")

    if cfg.output_every == 0:
        print(f"Output policy:              final frame only, if HDF5 is enabled")
    else:
        print(f"Output policy:              step 0, every {cfg.output_every} step(s), and final step, if HDF5 is enabled")

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
    p = generate_particles(gen, screen)
    particle_generation_seconds = time.perf_counter() - t0
    print(f"Particles:                  {p.n}")

    # --------------------------------------------------------
    # 3. Force arrays and initial force
    # --------------------------------------------------------
    fx = np.empty(p.n, dtype=np.float64)
    fy = np.empty(p.n, dtype=np.float64)
    fx_new = np.empty(p.n, dtype=np.float64)
    fy_new = np.empty(p.n, dtype=np.float64)
    t0 = time.perf_counter()
    compute_forces(p, fx, fy)
    init_force_seconds = time.perf_counter() - t0
    wmin = float(np.min(p.w))
    wmax = float(np.max(p.w))
    wr   = max(wmax - wmin, 1.0)

    # --------------------------------------------------------
    # 4. Optional HDF5 writer
    # --------------------------------------------------------
    h5: Optional[H5StreamWriter] = None

    if write_hdf5:
        h5 = H5StreamWriter(output_file, p.n, screen.nx, screen.ny)
        h5.write_metadata(input_file, cfg, gen, screen)
        h5.write_weights(p)

    output_frames = 0
    has_last_written_step = False
    last_written_step = -1
    screen_build_seconds = 0.0
    hdf5_write_seconds   = 0.0

    def write_output_frame(step: int) -> None:
        nonlocal output_frames, has_last_written_step, last_written_step
        nonlocal screen_build_seconds, hdf5_write_seconds
        if h5 is None:
            return

        if has_last_written_step and step == last_written_step:
            return

        t_screen = time.perf_counter()
        build_screen(screen, p, wmin, wr)
        screen_build_seconds += time.perf_counter() - t_screen

        t_h5 = time.perf_counter()
        h5.write_frame(step, p, screen)
        hdf5_write_seconds += time.perf_counter() - t_h5

        has_last_written_step = True
        last_written_step = step
        output_frames += 1

    # --------------------------------------------------------
    # 5. Simulation loop
    # --------------------------------------------------------
    loop_start = time.perf_counter()
    pure_dynamics_seconds = 0.0

    for step in range(cfg.max_steps):
        if h5 is not None and should_write_step(step, cfg.max_steps, cfg.output_every):
            write_output_frame(step)

        t_dyn = time.perf_counter()
        fx, fy, fx_new, fy_new = integrate_vv(p, fx, fy, fx_new, fy_new, cfg.dt)
        pure_dynamics_seconds += time.perf_counter() - t_dyn

    if h5 is not None:
        write_output_frame(cfg.max_steps)
        h5.close()

    loop_wall_seconds = time.perf_counter() - loop_start

    # --------------------------------------------------------
    # 6. Validation
    # --------------------------------------------------------
    t0 = time.perf_counter()
    validation = compute_validation_quantities(p)
    validation_seconds = time.perf_counter() - t0

    # --------------------------------------------------------
    # 7. Reporting
    # --------------------------------------------------------
    interactions = float(p.n) * float(p.n - 1) * float(cfg.max_steps)
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
        print(f"Pure dynamics performance:   {giga_interactions / pure_dynamics_seconds:.17g} GInteractions/s")

    if cfg.max_steps > 0 and loop_wall_seconds > 0.0:
        print(f"Loop end-to-end performance: {giga_interactions / loop_wall_seconds:.17g} GInteractions/s")

    print_validation_quantities(validation)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

