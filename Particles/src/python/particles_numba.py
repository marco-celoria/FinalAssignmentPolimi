#!/usr/bin/env python3
"""
================================================================================
Particle System Solver - Numba Multicore Reference Solution
================================================================================

Official CPU multicore reference solution for the Python track of the HPC final
assignment. It is aligned with the C++17 serial baseline and with the NumPy
teaching baseline, while using Numba for thread-level parallelism.

Primary Numba targets
---------------------
  1. compute_forces_numba(...): O(N^2) all-pairs interaction kernel.
  2. half_kick_drift_numba(...) and half_kick_numba(...): Velocity-Verlet update.
  3. compute_generating_field_numba(...): embarrassingly parallel field generation.
  4. build_screen_numba(...): serial by design, because several particles may
     update the same screen cells.

Official benchmark/no-output mode
---------------------------------
  python3 particles_numba_reference.py input_final.in none 0

Optional HDF5 correctness/debug run
-----------------------------------
  python3 particles_numba_reference.py input_medium.in particles_numba.h5 10
  python3 particles_numba_reference.py input_final.in reference_numba.h5 1000

Command line
------------
  python3 particles_numba_reference.py [inputFile] [h5File|none|--no-hdf5] [outputEvery]
                                      [--threads N]
                                      [--screen-tile-y NY]
                                      [--screen-tile-x NX]
                                      [--no-warmup]

Input format
------------
Same as the serial C++17 baseline:

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

Notes for instructors
---------------------
  * This is a conservative, race-free Numba CPU reference.
  * The force kernel parallelizes over target particle i. Each thread computes
    and writes fx[i], fy[i] exactly once.
  * Pair-symmetry is deliberately not used here. It is a valid advanced strategy
    but needs careful race-free accumulation.
  * JIT warm-up is enabled by default so reported timings exclude compilation.
  * HDF5 output is optional and excluded from benchmark/no-output mode.
================================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional, Tuple

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

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

    # Flattened row-major storage: values[j*nx + i]. This maps naturally to
    # Numba kernels and is compatible with reshape(ny, nx) for HDF5 output.
    values: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint64))

    def allocate(self) -> None:
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("Grid dimensions must be > 0")
        self.values = np.zeros(self.nx * self.ny, dtype=np.uint64)


@dataclass
class Particles:
    n: int = 0

    # Structure-of-arrays layout, matching the C++ baseline.
    w: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    x: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    y: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    vx: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    vy: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    def resize(self, n_particles: int) -> None:
        if n_particles <= 0:
            raise ValueError("Particles.resize: number of particles must be > 0")

        self.n = int(n_particles)
        self.w = np.empty(self.n, dtype=np.float64)
        self.x = np.empty(self.n, dtype=np.float64)
        self.y = np.empty(self.n, dtype=np.float64)
        self.vx = np.zeros(self.n, dtype=np.float64)
        self.vy = np.zeros(self.n, dtype=np.float64)


@dataclass
class Config:
    max_iters: int = 0
    max_steps: int = 0
    output_every: int = 0
    dt: float = 0.0


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
# INPUT PARSER AND VALIDATION
# ============================================================


def _parse_scalar(token: str, typ):
    if typ is int:
        return int(token)
    if typ is float:
        return float(token)
    return typ(token)


def parse_line(lines: Iterator[str], typ):
    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.lstrip(" \t\r\n")

        if not stripped or stripped.startswith("#"):
            continue

        content = stripped.split("#", 1)[0].rstrip()
        if not content:
            continue

        tokens = content.split()
        if len(tokens) != 1:
            raise RuntimeError(f"Trailing junk: {line}")

        try:
            return _parse_scalar(tokens[0], typ)
        except Exception as exc:
            raise RuntimeError(f"Parse error: {line}") from exc

    raise EOFError("Unexpected EOF")


def validate_config(g: Grid, screen: Grid, cfg: Config) -> None:
    if g.nx < 2 or g.ny < 2 or screen.nx < 2 or screen.ny < 2:
        raise RuntimeError("Grids must have at least 2 points in each direction")
    if g.xe <= g.xs or g.ye <= g.ys:
        raise RuntimeError("Invalid generating domain")
    if screen.xe <= screen.xs or screen.ye <= screen.ys:
        raise RuntimeError("Invalid particle/screen domain")
    if cfg.dt <= 0.0:
        raise RuntimeError("dt must be > 0")
    if cfg.max_steps <= 0 or cfg.max_iters <= 0:
        raise RuntimeError("maxSteps and maxIters must be > 0")
    if cfg.output_every < 0:
        raise RuntimeError("outputEvery must be >= 0")


def validate_h5_tiles(tile_y: int, tile_x: int) -> None:
    if tile_y <= 0 or tile_x <= 0:
        raise ValueError("HDF5 tile sizes must be > 0")


def read_input(file_name: str) -> Tuple[Config, Grid, Grid]:
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            lines = iter(f.readlines())
    except OSError as exc:
        raise RuntimeError(f"Cannot open input file: {file_name}") from exc

    g = Grid()
    screen = Grid()
    cfg = Config()

    g.nx = parse_line(lines, int)
    g.ny = parse_line(lines, int)
    g.xs = parse_line(lines, float)
    g.xe = parse_line(lines, float)
    g.ys = parse_line(lines, float)
    g.ye = parse_line(lines, float)

    screen.nx = parse_line(lines, int)
    screen.ny = parse_line(lines, int)
    screen.xs = parse_line(lines, float)
    screen.xe = parse_line(lines, float)
    screen.ys = parse_line(lines, float)
    screen.ye = parse_line(lines, float)

    cfg.max_iters = parse_line(lines, int)
    cfg.max_steps = parse_line(lines, int)
    cfg.dt = parse_line(lines, float)
    cfg.output_every = parse_line(lines, int)

    validate_config(g, screen, cfg)

    g.allocate()
    screen.allocate()
    return cfg, g, screen


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
# NUMBA KERNELS
# ============================================================


@njit(parallel=True, fastmath=False)
def compute_generating_field_numba(values, nx, ny, xs, xe, ys, ye, max_iter):
    dx = (xe - xs) / (nx - 1)
    dy = (ye - ys) / (ny - 1)

    for j in prange(ny):
        cb = ys + j * dy
        base = j * nx

        for i in range(nx):
            ca = xs + i * dx
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


@njit(parallel=True, fastmath=False)
def compute_forces_numba(x, y, w, fx, fy):
    n_particles = x.shape[0]

    for i in prange(n_particles):
        xi = x[i]
        yi = y[i]
        wi = w[i]

        fxi = 0.0
        fyi = 0.0

        for j in range(n_particles):
            if i == j:
                continue

            dx = x[j] - xi
            dy = y[j] - yi
            r2 = dx * dx + dy * dy + EPS2
            invr = 1.0 / np.sqrt(r2)
            invr3 = invr * invr * invr
            coeff = K_FORCE * wi * w[j] * invr3

            fxi += coeff * dx
            fyi += coeff * dy

        fx[i] = fxi
        fy[i] = fyi


@njit(parallel=True, fastmath=False)
def half_kick_drift_numba(x, y, vx, vy, w, fx, fy, dt):
    n_particles = x.shape[0]

    for i in prange(n_particles):
        invm = 1.0 / w[i]
        vx[i] += 0.5 * fx[i] * invm * dt
        vy[i] += 0.5 * fy[i] * invm * dt
        x[i] += vx[i] * dt
        y[i] += vy[i] * dt


@njit(parallel=True, fastmath=False)
def half_kick_numba(vx, vy, w, fx_new, fy_new, dt):
    n_particles = vx.shape[0]

    for i in prange(n_particles):
        invm = 1.0 / w[i]
        vx[i] += 0.5 * fx_new[i] * invm * dt
        vy[i] += 0.5 * fy_new[i] * invm * dt


@njit(fastmath=False)
def build_screen_numba(values, nx, ny, xs, xe, ys, ye, x, y, w, wmin, wr):
    # Kept serial intentionally: several particles may update the same screen
    # cell. A naive prange would introduce races.
    values[:] = np.uint64(0)

    n_particles = x.shape[0]
    invdx = (nx - 1) / (xe - xs)
    invdy = (ny - 1) / (ye - ys)

    for n in range(n_particles):
        ix = int((x[n] - xs) * invdx)
        iy = int((y[n] - ys) * invdy)

        if ix < 0:
            ix = 0
        elif ix > nx - 1:
            ix = nx - 1

        if iy < 0:
            iy = 0
        elif iy > ny - 1:
            iy = ny - 1

        wp = int(10.0 * (w[n] - wmin) / wr)
        if wp < 0:
            wp = 0
        elif wp > 1000:
            wp = 1000

        wp_u64 = np.uint64(wp)

        for dj in (-1, 0, 1):
            jy = iy + dj
            if jy < 0 or jy >= ny:
                continue
            row = jy * nx

            for di in (-1, 0, 1):
                jx = ix + di
                if jx < 0 or jx >= nx:
                    continue
                values[row + jx] += wp_u64


@njit(parallel=True, fastmath=False)
def validation_basic_numba(w, x, y, vx, vy):
    n_particles = x.shape[0]
    sum_x = 0.0
    sum_y = 0.0
    sum_vx = 0.0
    sum_vy = 0.0
    weighted_sum_x = 0.0
    weighted_sum_y = 0.0
    momentum_x = 0.0
    momentum_y = 0.0
    kinetic_energy = 0.0

    for i in prange(n_particles):
        wi = w[i]
        xi = x[i]
        yi = y[i]
        vxi = vx[i]
        vyi = vy[i]

        sum_x += xi
        sum_y += yi
        sum_vx += vxi
        sum_vy += vyi
        weighted_sum_x += wi * xi
        weighted_sum_y += wi * yi
        momentum_x += wi * vxi
        momentum_y += wi * vyi
        kinetic_energy += 0.5 * wi * (vxi * vxi + vyi * vyi)

    return (
        sum_x,
        sum_y,
        sum_vx,
        sum_vy,
        weighted_sum_x,
        weighted_sum_y,
        momentum_x,
        momentum_y,
        kinetic_energy,
    )


@njit(parallel=True, fastmath=False)
def validation_potential_numba(w, x, y):
    n_particles = x.shape[0]
    potential_like = 0.0

    for i in prange(n_particles):
        local = 0.0
        xi = x[i]
        yi = y[i]
        wi = w[i]

        for j in range(i + 1, n_particles):
            dx = x[j] - xi
            dy = y[j] - yi
            r2 = dx * dx + dy * dy + EPS2
            local += K_FORCE * wi * w[j] / np.sqrt(r2)

        potential_like += local

    return potential_like


# ============================================================
# HIGH-LEVEL PHYSICS WRAPPERS
# ============================================================


def compute_generating_field(g: Grid, max_iter: int) -> None:
    compute_generating_field_numba(g.values, g.nx, g.ny, g.xs, g.xe, g.ys, g.ye, max_iter)


def generate_particles(g: Grid, screen: Grid) -> Particles:
    if g.values.size == 0:
        raise RuntimeError("generate_particles: empty generating field")

    vmax = int(np.max(g.values))
    vmin0 = int(np.min(g.values))

    # floor((29*vmax + vmin0)/30). Python integers avoid unsigned overflow.
    threshold = (29 * vmax + vmin0) // 30

    vals2 = g.values.reshape(g.ny, g.nx)
    mask = vals2 >= np.uint64(threshold)
    j_idx, i_idx = np.nonzero(mask)
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


def compute_forces(p: Particles, fx: np.ndarray, fy: np.ndarray) -> None:
    if fx.shape[0] != p.n or fy.shape[0] != p.n:
        raise RuntimeError("compute_forces: force array size mismatch")
    compute_forces_numba(p.x, p.y, p.w, fx, fy)


def integrate_vv(
    p: Particles,
    fx: np.ndarray,
    fy: np.ndarray,
    fx_new: np.ndarray,
    fy_new: np.ndarray,
    dt: float,
):
    if fx.shape[0] != p.n or fy.shape[0] != p.n or fx_new.shape[0] != p.n or fy_new.shape[0] != p.n:
        raise RuntimeError("integrate_vv: force array size mismatch")

    half_kick_drift_numba(p.x, p.y, p.vx, p.vy, p.w, fx, fy, dt)
    compute_forces_numba(p.x, p.y, p.w, fx_new, fy_new)
    half_kick_numba(p.vx, p.vy, p.w, fx_new, fy_new, dt)

    # Return swapped force buffers.
    return fx_new, fy_new, fx, fy


def build_screen(g: Grid, p: Particles, wmin: float, wr: float) -> None:
    if wr <= 0.0:
        raise RuntimeError("build_screen: invalid weight range")
    build_screen_numba(g.values, g.nx, g.ny, g.xs, g.xe, g.ys, g.ye, p.x, p.y, p.w, wmin, wr)


def compute_validation_quantities(p: Particles) -> ValidationQuantities:
    basic = validation_basic_numba(p.w, p.x, p.y, p.vx, p.vy)
    potential_like = validation_potential_numba(p.w, p.x, p.y)

    q = ValidationQuantities()
    q.sum_x = float(basic[0])
    q.sum_y = float(basic[1])
    q.sum_vx = float(basic[2])
    q.sum_vy = float(basic[3])
    q.weighted_sum_x = float(basic[4])
    q.weighted_sum_y = float(basic[5])
    q.momentum_x = float(basic[6])
    q.momentum_y = float(basic[7])
    q.kinetic_energy = float(basic[8])
    q.potential_like = float(potential_like)
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
# HDF5 WRITER
# ============================================================


class H5StreamWriter:
    def __init__(
        self,
        name: str,
        nparticles: int,
        nx: int,
        ny: int,
        chunk_frames: int = 64,
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

        self.file = h5py.File(name, "w")
        self.np = int(nparticles)
        self.nx = int(nx)
        self.ny = int(ny)
        self.chunk_frames = int(chunk_frames)
        self.capacity = int(chunk_frames)
        self.current_frame = 0
        self.closed = False

        self.Pbuf = np.empty((self.np, 2), dtype=np.float64)
        self.Vbuf = np.empty((self.np, 2), dtype=np.float64)

        screen_chunk_y = min(self.ny, screen_tile_y)
        screen_chunk_x = min(self.nx, screen_tile_x)

        self.pos = self.file.create_dataset(
            "/pos",
            shape=(0, self.np, 2),
            maxshape=(None, self.np, 2),
            chunks=(1, self.np, 2),
            dtype=np.float64,
        )
        self.vel = self.file.create_dataset(
            "/vel",
            shape=(0, self.np, 2),
            maxshape=(None, self.np, 2),
            chunks=(1, self.np, 2),
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
        self.weight = self.file.create_dataset("/weight", shape=(self.np,), dtype=np.float64)

        self._extend_datasets(self.capacity)

    def write_metadata(self, input_file: str, cfg: Config, gen: Grid, screen_grid: Grid) -> None:
        attrs = self.file.attrs
        attrs["application"] = "Particle System Solver - Numba Multicore Reference"
        attrs["format_version"] = "2.0"
        attrs["input_file"] = input_file
        attrs["screen_dataset_note"] = "For visualization/debugging; not recommended for strict grading."
        attrs["particles"] = self.np
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

    def write_weights(self, p: Particles) -> None:
        if p.n != self.np:
            raise RuntimeError("H5StreamWriter: particle size mismatch in write_weights")
        self.weight[...] = p.w

    def _extend_datasets(self, new_size: int) -> None:
        self.pos.resize((new_size, self.np, 2))
        self.vel.resize((new_size, self.np, 2))
        self.screen.resize((new_size, self.ny, self.nx))
        self.step.resize((new_size,))

    def _shrink_to_fit(self) -> None:
        self.pos.resize((self.current_frame, self.np, 2))
        self.vel.resize((self.current_frame, self.np, 2))
        self.screen.resize((self.current_frame, self.ny, self.nx))
        self.step.resize((self.current_frame,))

    def write_frame(self, step_number: int, p: Particles, g: Grid) -> None:
        if self.closed:
            raise RuntimeError("H5StreamWriter: write_frame called after close")
        if p.n != self.np:
            raise RuntimeError("H5StreamWriter: particle size mismatch")
        if g.nx != self.nx or g.ny != self.ny:
            raise RuntimeError("H5StreamWriter: screen grid size mismatch")
        if g.values.size != self.nx * self.ny:
            raise RuntimeError("H5StreamWriter: screen buffer size mismatch")

        if self.current_frame >= self.capacity:
            self.capacity += self.chunk_frames
            self._extend_datasets(self.capacity)

        self.Pbuf[:, 0] = p.x
        self.Pbuf[:, 1] = p.y
        self.Vbuf[:, 0] = p.vx
        self.Vbuf[:, 1] = p.vy

        self.pos[self.current_frame, :, :] = self.Pbuf
        self.vel[self.current_frame, :, :] = self.Vbuf
        self.screen[self.current_frame, :, :] = g.values.reshape(g.ny, g.nx)
        self.step[self.current_frame] = np.int64(step_number)
        self.current_frame += 1

    def close(self) -> None:
        if self.closed:
            return
        self._shrink_to_fit()
        self.file.flush()
        self.file.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


# ============================================================
# JIT WARM-UP
# ============================================================


def warmup_numba() -> None:
    """Compile hot kernels before timing so reported runtime excludes JIT cost."""
    values = np.zeros(4, dtype=np.uint64)
    compute_generating_field_numba(values, 2, 2, -2.0, 1.0, -1.0, 1.0, 4)

    x = np.array([0.0, 1.0], dtype=np.float64)
    y = np.array([0.0, 0.5], dtype=np.float64)
    w = np.array([1.0, 2.0], dtype=np.float64)
    vx = np.zeros(2, dtype=np.float64)
    vy = np.zeros(2, dtype=np.float64)
    fx = np.zeros(2, dtype=np.float64)
    fy = np.zeros(2, dtype=np.float64)
    fx_new = np.zeros(2, dtype=np.float64)
    fy_new = np.zeros(2, dtype=np.float64)

    compute_forces_numba(x, y, w, fx, fy)
    half_kick_drift_numba(x, y, vx, vy, w, fx, fy, 1.0e-3)
    compute_forces_numba(x, y, w, fx_new, fy_new)
    half_kick_numba(vx, vy, w, fx_new, fy_new, 1.0e-3)

    screen = np.zeros(4, dtype=np.uint64)
    build_screen_numba(screen, 2, 2, -1.0, 1.0, -1.0, 1.0, x, y, w, 1.0, 1.0)

    validation_basic_numba(w, x, y, vx, vy)
    validation_potential_numba(w, x, y)


# ============================================================
# CLI
# ============================================================


def parse_args(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(
        description="Numba multicore reference solution for the Particle System Solver assignment"
    )

    parser.add_argument("input", nargs="?", default="Particles.inp", help="Input file")
    parser.add_argument("output", nargs="?", default="none", help="HDF5 file, or none/-/--no-hdf5")
    parser.add_argument("output_every", nargs="?", type=int, default=None, help="Override outputEvery from input file")

    parser.add_argument("--threads", type=int, default=None, help="Number of Numba CPU threads")
    parser.add_argument("--screen-tile-y", type=int, default=256, help="HDF5 screen chunk tile size in y")
    parser.add_argument("--screen-tile-x", type=int, default=256, help="HDF5 screen chunk tile size in x")
    parser.add_argument("--no-warmup", action="store_true", help="Do not compile Numba kernels before measured sections")

    return parser.parse_args(argv)


# ============================================================
# MAIN
# ============================================================


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.threads is not None:
        if args.threads <= 0:
            raise ValueError("--threads must be > 0")
        set_num_threads(args.threads)

    output_file = args.output
    write_hdf5 = not is_no_hdf5_token(output_file)

    if write_hdf5:
        validate_h5_tiles(args.screen_tile_y, args.screen_tile_x)

    if write_hdf5 and h5py is None:
        raise RuntimeError("HDF5 output requested, but h5py is not available. Use 'none' or install h5py.")

    cfg, gen, screen = read_input(args.input)

    if args.output_every is not None:
        if args.output_every < 0:
            raise ValueError("outputEvery override must be >= 0")
        cfg.output_every = int(args.output_every)

    if not args.no_warmup:
        warmup_numba()

    print(f"Input file:                 {args.input}")
    print(f"HDF5 available:             {'yes' if h5py is not None else 'no'}")
    print(f"HDF5 output:                {output_file if write_hdf5 else 'disabled'}")
    print(f"Benchmark/no-output mode:   {'yes' if not write_hdf5 else 'no'}")
    print(f"Generating grid:            {gen.nx} x {gen.ny}")
    print(f"Screen grid:                {screen.nx} x {screen.ny}")
    print(f"Max iterations:             {cfg.max_iters}")
    print(f"Steps:                      {cfg.max_steps}")
    print(f"dt:                         {cfg.dt:.17g}")
    if cfg.output_every == 0:
        print("Output policy:              final frame only, if HDF5 is enabled")
    else:
        print(f"Output policy:              step 0, every {cfg.output_every} step(s), and final step, if HDF5 is enabled")
    print(f"Numba threads:              {get_num_threads()}")
    print(f"NUMBA_NUM_THREADS env:      {os.environ.get('NUMBA_NUM_THREADS', '(not set)')}")
    print(f"Numba JIT warm-up:          {'disabled' if args.no_warmup else 'enabled'}")

    # --------------------------------------------------------
    # 1. Generate field
    # --------------------------------------------------------
    gen_t0 = time.perf_counter()
    compute_generating_field(gen, cfg.max_iters)
    gen_t1 = time.perf_counter()

    # --------------------------------------------------------
    # 2. Generate particles
    # --------------------------------------------------------
    part_t0 = time.perf_counter()
    p = generate_particles(gen, screen)
    part_t1 = time.perf_counter()

    print(f"Particles:                  {p.n}")

    # --------------------------------------------------------
    # 3. Initial force computation
    # --------------------------------------------------------
    fx = np.empty(p.n, dtype=np.float64)
    fy = np.empty(p.n, dtype=np.float64)
    fx_new = np.empty(p.n, dtype=np.float64)
    fy_new = np.empty(p.n, dtype=np.float64)

    init_force_t0 = time.perf_counter()
    compute_forces(p, fx, fy)
    init_force_t1 = time.perf_counter()

    wmin = float(np.min(p.w))
    wmax = float(np.max(p.w))
    wr = max(wmax - wmin, 1.0)

    # --------------------------------------------------------
    # 4. Optional HDF5 writer
    # --------------------------------------------------------
    h5: Optional[H5StreamWriter] = None
    if write_hdf5:
        h5 = H5StreamWriter(
            output_file,
            p.n,
            screen.nx,
            screen.ny,
            screen_tile_y=args.screen_tile_y,
            screen_tile_x=args.screen_tile_x,
        )
        h5.write_metadata(args.input, cfg, gen, screen)
        h5.write_weights(p)

    pure_dynamics_time_s = 0.0
    screen_build_time_s = 0.0
    hdf5_write_time_s = 0.0
    output_frames = 0
    has_last_written_step = False
    last_written_step = -1

    def write_output_frame(step: int) -> None:
        nonlocal output_frames, screen_build_time_s, hdf5_write_time_s
        nonlocal has_last_written_step, last_written_step

        if h5 is None:
            return
        if has_last_written_step and step == last_written_step:
            return

        t0 = time.perf_counter()
        build_screen(screen, p, wmin, wr)
        t1 = time.perf_counter()
        h5.write_frame(step, p, screen)
        t2 = time.perf_counter()

        screen_build_time_s += t1 - t0
        hdf5_write_time_s += t2 - t1
        output_frames += 1
        has_last_written_step = True
        last_written_step = step

    # --------------------------------------------------------
    # 5. Simulation loop
    # --------------------------------------------------------
    loop_t0 = time.perf_counter()

    for step in range(cfg.max_steps):
        if h5 is not None and should_write_step(step, cfg.max_steps, cfg.output_every):
            write_output_frame(step)

        dyn_t0 = time.perf_counter()
        fx, fy, fx_new, fy_new = integrate_vv(p, fx, fy, fx_new, fy_new, cfg.dt)
        dyn_t1 = time.perf_counter()
        pure_dynamics_time_s += dyn_t1 - dyn_t0

    if h5 is not None:
        write_output_frame(cfg.max_steps)
        h5.close()

    loop_t1 = time.perf_counter()

    # --------------------------------------------------------
    # 6. Validation
    # --------------------------------------------------------
    validation_t0 = time.perf_counter()
    validation = compute_validation_quantities(p)
    validation_t1 = time.perf_counter()

    # --------------------------------------------------------
    # 7. Reporting
    # --------------------------------------------------------
    generating_field_s = gen_t1 - gen_t0
    particle_generation_s = part_t1 - part_t0
    initial_force_s = init_force_t1 - init_force_t0
    loop_wall_s = loop_t1 - loop_t0
    validation_s = validation_t1 - validation_t0

    interactions = float(p.n) * float(p.n - 1) * float(cfg.max_steps)
    giga_interactions = interactions / 1.0e9

    print("Simulation completed successfully.")
    print(f"Output frames:                   {output_frames}")
    print(f"Generating field wall time:      {generating_field_s:.17g} s")
    print(f"Particle generation wall time:   {particle_generation_s:.17g} s")
    print(f"Initial force wall time:         {initial_force_s:.17g} s")
    print(f"Pure dynamics time:              {pure_dynamics_time_s:.17g} s")
    print(f"Screen build time:               {screen_build_time_s:.17g} s")
    print(f"HDF5 write time:                 {hdf5_write_time_s:.17g} s")
    print(f"Validation time:                 {validation_s:.17g} s")
    print(f"Loop wall time:                  {loop_wall_s:.17g} s")
    print(f"Dynamics time per step:          {pure_dynamics_time_s / cfg.max_steps:.17g} s")
    print(f"Loop wall time per step:         {loop_wall_s / cfg.max_steps:.17g} s")

    if cfg.max_steps > 0 and pure_dynamics_time_s > 0.0:
        print(f"Pure dynamics performance:  {giga_interactions / pure_dynamics_time_s:.17g} GInteractions/s")
    if cfg.max_steps > 0 and loop_wall_s > 0.0:
        print(f"Loop end-to-end performance: {giga_interactions / loop_wall_s:.17g} GInteractions/s")

    print_validation_quantities(validation)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

