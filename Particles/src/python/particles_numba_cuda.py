#!/usr/bin/env python3
"""
================================================================================
Particle System Solver - Numba CUDA Reference Solution
================================================================================

Official Numba CUDA offloading reference solution for the Python track of the HPC
final assignment. It is aligned with the C++17 serial baseline, the NumPy
baseline, and the Numba multicore reference solution.

Primary GPU offload targets
---------------------------
  1. mandelbrot_kernel(...): initial generating field, one GPU thread per grid
     point.
  2. compute_forces_tiled_kernel(...): O(N^2) all-pairs force kernel, one GPU
     thread per target particle and shared-memory tiling over source particles.
  3. half_kick_drift_kernel(...) and half_kick_kernel(...): Velocity-Verlet
     integration update.
  4. build_screen_kernel(...): optional visualization/debug screen for HDF5 mode.

Official benchmark/no-output mode
---------------------------------
  python3 particles_numba_cuda_reference.py input_final.in none 0

Optional HDF5 correctness/debug run
-----------------------------------
  python3 particles_numba_cuda_reference.py input_medium.in particles_numba_cuda.h5 10
  python3 particles_numba_cuda_reference.py input_final.in reference_numba_cuda.h5 1000

Command line
------------
  python3 particles_numba_cuda_reference.py [inputFile] [h5File|none|--no-hdf5] [outputEvery]
                                           [--device ID]
                                           [--threads-per-block N]
                                           [--chunk-frames N]
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
  * This is a conservative, race-free GPU reference implementation.
  * The force kernel deliberately does not exploit pair symmetry. Each GPU
    thread owns one target particle and writes fx[i], fy[i] exactly once.
  * Particle arrays remain resident on the GPU during the simulation loop.
  * HDF5 output is optional and excluded from benchmark/no-output mode.
  * Final validation quantities are computed on the host after copying the final
    particle state back from the GPU.
================================================================================
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional, Tuple

import numpy as np
from numba import cuda, float64

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

# The shared-memory force kernel below is specialized for TILE_SIZE. Keep this
# constant equal to the launch block size used for the force kernel.
TILE_SIZE = 256
MANDELBROT_BLOCK = (16, 16)


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

    # Flattened row-major storage: values[j*nx + i].
    values: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint64))

    def allocate(self) -> None:
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("Grid.allocate: grid dimensions must be > 0")
        self.values = np.zeros(self.nx * self.ny, dtype=np.uint64)


@dataclass
class Particles:
    n: int = 0

    # Structure-of-arrays layout, matching all reference implementations.
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
class HostOutputBuffers:
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    screen_i64: np.ndarray
    screen_u64: np.ndarray


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


def blocks_1d(n: int, threads: int) -> int:
    n = int(n)
    threads = int(threads)

    if n < 0:
        raise ValueError("blocks_1d: n must be >= 0")
    if threads <= 0:
        raise ValueError("blocks_1d: threads must be > 0")

    return (n + threads - 1) // threads


# ============================================================
# CUDA KERNELS
# ============================================================


@cuda.jit
def mandelbrot_kernel(values, nx, ny, xs, ys, dx, dy, max_iter):
    i, j = cuda.grid(2)

    if i >= nx or j >= ny:
        return

    ca = xs + i * dx
    cb = ys + j * dy

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

    values[i + j * nx] = np.uint64(it)


@cuda.jit
def compute_forces_tiled_kernel(x, y, w, fx, fy, n_particles):
    sh_x = cuda.shared.array(shape=TILE_SIZE, dtype=float64)
    sh_y = cuda.shared.array(shape=TILE_SIZE, dtype=float64)
    sh_w = cuda.shared.array(shape=TILE_SIZE, dtype=float64)

    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    tid = cuda.threadIdx.x

    active = i < n_particles

    if active:
        xi = x[i]
        yi = y[i]
        wi = w[i]
    else:
        xi = 0.0
        yi = 0.0
        wi = 0.0

    fxi = 0.0
    fyi = 0.0

    tiles = (n_particles + TILE_SIZE - 1) // TILE_SIZE

    for tile in range(tiles):
        j = tile * TILE_SIZE + tid

        if j < n_particles:
            sh_x[tid] = x[j]
            sh_y[tid] = y[j]
            sh_w[tid] = w[j]
        else:
            sh_x[tid] = 0.0
            sh_y[tid] = 0.0
            sh_w[tid] = 0.0

        cuda.syncthreads()

        if active:
            for k in range(TILE_SIZE):
                global_j = tile * TILE_SIZE + k
                if global_j >= n_particles or global_j == i:
                    continue

                dx = sh_x[k] - xi
                dy = sh_y[k] - yi
                r2 = dx * dx + dy * dy + EPS2
                invr = 1.0 / math.sqrt(r2)
                invr3 = invr * invr * invr
                coeff = K_FORCE * wi * sh_w[k] * invr3

                fxi += coeff * dx
                fyi += coeff * dy

        cuda.syncthreads()

    if active:
        fx[i] = fxi
        fy[i] = fyi


@cuda.jit
def half_kick_drift_kernel(x, y, vx, vy, w, fx, fy, dt, n_particles):
    i = cuda.grid(1)

    if i >= n_particles:
        return

    invm = 1.0 / w[i]
    vx[i] += 0.5 * fx[i] * invm * dt
    vy[i] += 0.5 * fy[i] * invm * dt
    x[i] += vx[i] * dt
    y[i] += vy[i] * dt


@cuda.jit
def half_kick_kernel(vx, vy, w, fx_new, fy_new, dt, n_particles):
    i = cuda.grid(1)

    if i >= n_particles:
        return

    invm = 1.0 / w[i]
    vx[i] += 0.5 * fx_new[i] * invm * dt
    vy[i] += 0.5 * fy_new[i] * invm * dt


@cuda.jit
def zero_int64_kernel(arr, n):
    i = cuda.grid(1)
    if i < n:
        arr[i] = 0


@cuda.jit
def build_screen_kernel(screen, x, y, w, nx, ny, xs, ys, invdx, invdy, wmin, wr, n_particles):
    n = cuda.grid(1)

    if n >= n_particles:
        return

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

    for dj in range(-1, 2):
        jy = iy + dj
        if jy < 0 or jy >= ny:
            continue
        row = jy * nx

        for di in range(-1, 2):
            jx = ix + di
            if jx < 0 or jx >= nx:
                continue
            # int64 screen is used because Numba CUDA int64 atomics are broadly
            # supported. Values are non-negative and converted to uint64 for HDF5.
            cuda.atomic.add(screen, row + jx, wp)


# ============================================================
# HOST-SIDE PHYSICS HELPERS
# ============================================================


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


def compute_validation_quantities(p: Particles) -> ValidationQuantities:
    q = ValidationQuantities()
    n = p.n

    q.sum_x = float(np.sum(p.x))
    q.sum_y = float(np.sum(p.y))
    q.sum_vx = float(np.sum(p.vx))
    q.sum_vy = float(np.sum(p.vy))
    q.weighted_sum_x = float(np.sum(p.w * p.x))
    q.weighted_sum_y = float(np.sum(p.w * p.y))
    q.momentum_x = float(np.sum(p.w * p.vx))
    q.momentum_y = float(np.sum(p.w * p.vy))
    q.kinetic_energy = float(0.5 * np.sum(p.w * (p.vx * p.vx + p.vy * p.vy)))

    potential = 0.0
    # Host-side blocked validation avoids allocating a full NxN matrix.
    block_size = 256
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        xb = p.x[start:stop, None]
        yb = p.y[start:stop, None]
        wb = p.w[start:stop, None]

        dx = p.x[None, :] - xb
        dy = p.y[None, :] - yb
        r2 = dx * dx + dy * dy + EPS2
        pair = K_FORCE * wb * p.w[None, :] / np.sqrt(r2)

        rows = np.arange(stop - start)[:, None]
        global_i = start + rows
        global_j = np.arange(n)[None, :]
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


def allocate_host_output_buffers(nparticles: int, screen_size: int) -> HostOutputBuffers:
    if nparticles <= 0:
        raise ValueError("allocate_host_output_buffers: nparticles must be > 0")
    if screen_size <= 0:
        raise ValueError("allocate_host_output_buffers: screen_size must be > 0")

    return HostOutputBuffers(
        x=cuda.pinned_array(nparticles, dtype=np.float64),
        y=cuda.pinned_array(nparticles, dtype=np.float64),
        vx=cuda.pinned_array(nparticles, dtype=np.float64),
        vy=cuda.pinned_array(nparticles, dtype=np.float64),
        screen_i64=cuda.pinned_array(screen_size, dtype=np.int64),
        screen_u64=np.empty(screen_size, dtype=np.uint64),
    )


def launch_output_copy(screen_d, x_d, y_d, vx_d, vy_d, host: HostOutputBuffers, stream) -> None:
    x_d.copy_to_host(host.x, stream=stream)
    y_d.copy_to_host(host.y, stream=stream)
    vx_d.copy_to_host(host.vx, stream=stream)
    vy_d.copy_to_host(host.vy, stream=stream)
    screen_d.copy_to_host(host.screen_i64, stream=stream)


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
        attrs["application"] = "Particle System Solver - Numba CUDA Reference"
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

    def write_frame_arrays(self, step_number: int, x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray, screen_values: np.ndarray) -> None:
        if self.closed:
            raise RuntimeError("H5StreamWriter: write after close")
        if x.size != self.np or y.size != self.np:
            raise RuntimeError("H5StreamWriter: particle position size mismatch")
        if vx.size != self.np or vy.size != self.np:
            raise RuntimeError("H5StreamWriter: particle velocity size mismatch")
        if screen_values.size != self.nx * self.ny:
            raise RuntimeError("H5StreamWriter: screen size mismatch")

        if self.current_frame >= self.capacity:
            self.capacity += self.chunk_frames
            self._extend_datasets(self.capacity)

        self.Pbuf[:, 0] = x
        self.Pbuf[:, 1] = y
        self.Vbuf[:, 0] = vx
        self.Vbuf[:, 1] = vy

        self.pos[self.current_frame, :, :] = self.Pbuf
        self.vel[self.current_frame, :, :] = self.Vbuf
        self.screen[self.current_frame, :, :] = screen_values.reshape(self.ny, self.nx)
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
# CUDA WARM-UP
# ============================================================


def warmup_cuda() -> None:
    """Compile CUDA kernels before timing the real workload."""
    d_vals = cuda.device_array(4, dtype=np.uint64)
    mandelbrot_kernel[(1, 1), (2, 2)](d_vals, 2, 2, -2.0, -1.0, 3.0, 2.0, 4)

    x = np.array([0.0, 1.0], dtype=np.float64)
    y = np.array([0.0, 0.5], dtype=np.float64)
    w = np.array([1.0, 2.0], dtype=np.float64)
    vx = np.zeros(2, dtype=np.float64)
    vy = np.zeros(2, dtype=np.float64)

    x_d = cuda.to_device(x)
    y_d = cuda.to_device(y)
    w_d = cuda.to_device(w)
    vx_d = cuda.to_device(vx)
    vy_d = cuda.to_device(vy)
    fx_d = cuda.device_array(2, dtype=np.float64)
    fy_d = cuda.device_array(2, dtype=np.float64)
    fx_new_d = cuda.device_array(2, dtype=np.float64)
    fy_new_d = cuda.device_array(2, dtype=np.float64)

    compute_forces_tiled_kernel[1, TILE_SIZE](x_d, y_d, w_d, fx_d, fy_d, 2)
    half_kick_drift_kernel[1, TILE_SIZE](x_d, y_d, vx_d, vy_d, w_d, fx_d, fy_d, 1.0e-3, 2)
    compute_forces_tiled_kernel[1, TILE_SIZE](x_d, y_d, w_d, fx_new_d, fy_new_d, 2)
    half_kick_kernel[1, TILE_SIZE](vx_d, vy_d, w_d, fx_new_d, fy_new_d, 1.0e-3, 2)

    screen_d = cuda.device_array(4, dtype=np.int64)
    zero_int64_kernel[1, TILE_SIZE](screen_d, 4)
    build_screen_kernel[1, TILE_SIZE](screen_d, x_d, y_d, w_d, 2, 2, -1.0, -1.0, 0.5, 0.5, 1.0, 1.0, 2)
    cuda.synchronize()


# ============================================================
# CUDA LAUNCH HELPERS
# ============================================================


def queue_screen_build_and_output(
    screen_d,
    x_d,
    y_d,
    vx_d,
    vy_d,
    w_d,
    host: HostOutputBuffers,
    screen: Grid,
    blocks_screen: int,
    blocks_particles: int,
    threads: int,
    wmin: float,
    wr: float,
    invdx_screen: float,
    invdy_screen: float,
    nparticles: int,
    stream,
) -> None:
    zero_int64_kernel[blocks_screen, threads, stream](screen_d, screen.nx * screen.ny)
    build_screen_kernel[blocks_particles, threads, stream](
        screen_d,
        x_d,
        y_d,
        w_d,
        screen.nx,
        screen.ny,
        screen.xs,
        screen.ys,
        invdx_screen,
        invdy_screen,
        wmin,
        wr,
        nparticles,
    )
    launch_output_copy(screen_d, x_d, y_d, vx_d, vy_d, host, stream)


def queue_verlet_step(x_d, y_d, vx_d, vy_d, w_d, fx_d, fy_d, fx_new_d, fy_new_d, dt: float, nparticles: int, blocks_particles: int, threads: int, stream) -> None:
    half_kick_drift_kernel[blocks_particles, threads, stream](x_d, y_d, vx_d, vy_d, w_d, fx_d, fy_d, dt, nparticles)
    compute_forces_tiled_kernel[blocks_particles, threads, stream](x_d, y_d, w_d, fx_new_d, fy_new_d, nparticles)
    half_kick_kernel[blocks_particles, threads, stream](vx_d, vy_d, w_d, fx_new_d, fy_new_d, dt, nparticles)


# ============================================================
# CLI
# ============================================================


def parse_args(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(
        description="Numba CUDA reference solution for the Particle System Solver assignment"
    )

    parser.add_argument("input", nargs="?", default="Particles.inp", help="Input file")
    parser.add_argument("output", nargs="?", default="none", help="HDF5 file, or none/-/--no-hdf5")
    parser.add_argument("output_every", nargs="?", type=int, default=None, help="Override outputEvery from input file")

    parser.add_argument("--device", type=int, default=None, help="CUDA device id")
    parser.add_argument(
        "--threads-per-block",
        type=int,
        default=TILE_SIZE,
        help=f"CUDA threads per block. For this reference force kernel, use {TILE_SIZE}.",
    )
    parser.add_argument("--chunk-frames", type=int, default=64, help="HDF5 frame chunk size")
    parser.add_argument("--screen-tile-y", type=int, default=256, help="HDF5 /screen chunk tile size in y")
    parser.add_argument("--screen-tile-x", type=int, default=256, help="HDF5 /screen chunk tile size in x")
    parser.add_argument("--no-warmup", action="store_true", help="Do not compile CUDA kernels before measured sections")

    return parser.parse_args(argv)


# ============================================================
# MAIN
# ============================================================


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if not cuda.is_available():
        raise RuntimeError("CUDA is not available")
     
    if args.device is not None:
        cuda.select_device(args.device)
     
    if args.threads_per_block != TILE_SIZE:
        raise ValueError(
            f"This reference implementation requires --threads-per-block {TILE_SIZE}, "
            "because the shared-memory tile size is fixed at compile time."
        )
    
    input_file = args.input
    output_file = args.output
    write_hdf5 = not is_no_hdf5_token(output_file)
    
    if write_hdf5:
        if args.chunk_frames <= 0:
            raise ValueError("--chunk-frames must be > 0")
        validate_h5_tiles(args.screen_tile_y, args.screen_tile_x)
     
    if write_hdf5 and h5py is None:
        raise RuntimeError("HDF5 output requested, but h5py is not available. Use 'none' or install h5py.")

    cfg, gen, screen = read_input(input_file)

    if args.output_every is not None:
        if args.output_every < 0:
            raise ValueError("outputEvery override must be >= 0")
        cfg.output_every = int(args.output_every)

    if not args.no_warmup:
        warmup_cuda()

    main_stream = cuda.stream()
    device = cuda.get_current_device()

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
        print("Output policy:              final frame only, if HDF5 is enabled")
    else:
        print(f"Output policy:              step 0, every {cfg.output_every} step(s), and final step, if HDF5 is enabled")
    print(f"CUDA device id:             {cuda.current_context().device.id}")
    print(f"CUDA device:                {device.name.decode() if isinstance(device.name, bytes) else device.name}")
    print(f"CUDA compute capability:    {device.compute_capability[0]}.{device.compute_capability[1]}")
    print(f"Threads per block:          {TILE_SIZE}")
    print(f"CUDA JIT warm-up:           {'disabled' if args.no_warmup else 'enabled'}")
    print(f"NUMBA_CUDA_DRIVER:          {os.environ.get('NUMBA_CUDA_DRIVER', '(not set)')}")

    # --------------------------------------------------------
    # 1. Generate Mandelbrot field on GPU
    # --------------------------------------------------------
    d_vals = cuda.device_array(gen.nx * gen.ny, dtype=np.uint64, stream=main_stream)
    grid2d = (
        (gen.nx + MANDELBROT_BLOCK[0] - 1) // MANDELBROT_BLOCK[0],
        (gen.ny + MANDELBROT_BLOCK[1] - 1) // MANDELBROT_BLOCK[1],
    )
    dx_gen = (gen.xe - gen.xs) / float(gen.nx - 1)
    dy_gen = (gen.ye - gen.ys) / float(gen.ny - 1)

    mandel_start = cuda.event()
    mandel_stop = cuda.event()
    mandel_start.record(main_stream)
    mandelbrot_kernel[grid2d, MANDELBROT_BLOCK, main_stream](d_vals, gen.nx, gen.ny, gen.xs, gen.ys, dx_gen, dy_gen, cfg.max_iters)
    mandel_stop.record(main_stream)
    mandel_stop.synchronize()
    mandel_gpu_s = cuda.event_elapsed_time(mandel_start, mandel_stop) / 1000.0

    gen.values = d_vals.copy_to_host(stream=main_stream)
    main_stream.synchronize()

    # --------------------------------------------------------
    # 2. Generate particles on host
    # --------------------------------------------------------
    particle_t0 = time.perf_counter()
    particles = generate_particles(gen, screen)
    particle_t1 = time.perf_counter()
    particle_generation_s = particle_t1 - particle_t0
    print(f"Particles:                  {particles.n}")

    # --------------------------------------------------------
    # 3. Copy particles to GPU and allocate device arrays
    # --------------------------------------------------------
    x_d = cuda.to_device(particles.x, stream=main_stream)
    y_d = cuda.to_device(particles.y, stream=main_stream)
    vx_d = cuda.to_device(particles.vx, stream=main_stream)
    vy_d = cuda.to_device(particles.vy, stream=main_stream)
    w_d = cuda.to_device(particles.w, stream=main_stream)

    fx_d = cuda.device_array(particles.n, dtype=np.float64, stream=main_stream)
    fy_d = cuda.device_array(particles.n, dtype=np.float64, stream=main_stream)
    fx_new_d = cuda.device_array(particles.n, dtype=np.float64, stream=main_stream)
    fy_new_d = cuda.device_array(particles.n, dtype=np.float64, stream=main_stream)

    screen_size = screen.nx * screen.ny
    # int64 screen is used because Numba CUDA int64 atomics are broadly supported.
    screen_d = cuda.device_array(screen_size, dtype=np.int64, stream=main_stream)

    threads = TILE_SIZE
    blocks_particles = blocks_1d(particles.n, threads)
    blocks_screen = blocks_1d(screen_size, threads)

    # --------------------------------------------------------
    # 4. Initial force computation
    # --------------------------------------------------------
    init_force_start = cuda.event()
    init_force_stop = cuda.event()
    init_force_start.record(main_stream)
    compute_forces_tiled_kernel[blocks_particles, threads, main_stream](x_d, y_d, w_d, fx_d, fy_d, particles.n)
    init_force_stop.record(main_stream)
    init_force_stop.synchronize()
    init_force_gpu_s = cuda.event_elapsed_time(init_force_start, init_force_stop) / 1000.0

    # --------------------------------------------------------
    # 5. Constants and optional HDF5 output setup
    # --------------------------------------------------------
    wmin = float(np.min(particles.w))
    wmax = float(np.max(particles.w))
    wr = max(wmax - wmin, 1.0)
    invdx_screen = float(screen.nx - 1) / (screen.xe - screen.xs)
    invdy_screen = float(screen.ny - 1) / (screen.ye - screen.ys)

    host: Optional[HostOutputBuffers] = None
    h5: Optional[H5StreamWriter] = None
    if write_hdf5:
        host = allocate_host_output_buffers(particles.n, screen_size)
        h5 = H5StreamWriter(
            output_file,
            particles.n,
            screen.nx,
            screen.ny,
            chunk_frames=args.chunk_frames,
            screen_tile_y=args.screen_tile_y,
            screen_tile_x=args.screen_tile_x,
        )
        h5.write_metadata(input_file, cfg, gen, screen)
        h5.write_weights(particles)

    # --------------------------------------------------------
    # 6. Simulation loop
    # --------------------------------------------------------
    pure_dynamics_gpu_ms = 0.0
    screen_and_copy_ms = 0.0
    hdf5_write_s = 0.0
    output_frames = 0
    has_last_written_step = False
    last_written_step = -1

    dyn_start = cuda.event()
    dyn_stop = cuda.event()
    out_start = cuda.event()
    out_stop = cuda.event()

    def time_pending_dynamics() -> None:
        nonlocal pure_dynamics_gpu_ms
        dyn_stop.record(main_stream)
        dyn_stop.synchronize()
        pure_dynamics_gpu_ms += cuda.event_elapsed_time(dyn_start, dyn_stop)

    def start_dynamics_segment() -> None:
        dyn_start.record(main_stream)

    def write_output_frame(step: int) -> None:
        nonlocal output_frames, screen_and_copy_ms, hdf5_write_s
        nonlocal has_last_written_step, last_written_step

        if h5 is None or host is None:
            return
        if has_last_written_step and step == last_written_step:
            return

        out_start.record(main_stream)
        queue_screen_build_and_output(
            screen_d=screen_d,
            x_d=x_d,
            y_d=y_d,
            vx_d=vx_d,
            vy_d=vy_d,
            w_d=w_d,
            host=host,
            screen=screen,
            blocks_screen=blocks_screen,
            blocks_particles=blocks_particles,
            threads=threads,
            wmin=wmin,
            wr=wr,
            invdx_screen=invdx_screen,
            invdy_screen=invdy_screen,
            nparticles=particles.n,
            stream=main_stream,
        )
        out_stop.record(main_stream)
        out_stop.synchronize()
        screen_and_copy_ms += cuda.event_elapsed_time(out_start, out_stop)

        np.copyto(host.screen_u64, host.screen_i64, casting="unsafe")

        t0 = time.perf_counter()
        h5.write_frame_arrays(
            step_number=step,
            x=host.x,
            y=host.y,
            vx=host.vx,
            vy=host.vy,
            screen_values=host.screen_u64,
        )
        hdf5_write_s += time.perf_counter() - t0

        output_frames += 1
        has_last_written_step = True
        last_written_step = step

    loop_wall_t0 = time.perf_counter()
    start_dynamics_segment()

    for step in range(cfg.max_steps):
        if h5 is not None and should_write_step(step, cfg.max_steps, cfg.output_every):
            time_pending_dynamics()
            write_output_frame(step)
            start_dynamics_segment()

        queue_verlet_step(
            x_d=x_d,
            y_d=y_d,
            vx_d=vx_d,
            vy_d=vy_d,
            w_d=w_d,
            fx_d=fx_d,
            fy_d=fy_d,
            fx_new_d=fx_new_d,
            fy_new_d=fy_new_d,
            dt=cfg.dt,
            nparticles=particles.n,
            blocks_particles=blocks_particles,
            threads=threads,
            stream=main_stream,
        )

        fx_d, fx_new_d = fx_new_d, fx_d
        fy_d, fy_new_d = fy_new_d, fy_d

    time_pending_dynamics()

    if h5 is not None:
        write_output_frame(cfg.max_steps)
        h5.close()

    loop_wall_s = time.perf_counter() - loop_wall_t0
    pure_dynamics_gpu_s = pure_dynamics_gpu_ms / 1000.0
    screen_and_copy_s = screen_and_copy_ms / 1000.0

    # --------------------------------------------------------
    # 7. Copy final state and validate
    # --------------------------------------------------------
    final_copy_t0 = time.perf_counter()
    particles.x = x_d.copy_to_host()
    particles.y = y_d.copy_to_host()
    particles.vx = vx_d.copy_to_host()
    particles.vy = vy_d.copy_to_host()
    cuda.synchronize()
    final_copy_s = time.perf_counter() - final_copy_t0

    validation_t0 = time.perf_counter()
    validation = compute_validation_quantities(particles)
    validation_s = time.perf_counter() - validation_t0

    # --------------------------------------------------------
    # 8. Reporting
    # --------------------------------------------------------
    interactions = float(particles.n) * float(particles.n - 1) * float(cfg.max_steps)
    giga_interactions = interactions / 1.0e9

    print("Simulation completed successfully.")
    print(f"Output frames:                   {output_frames}")
    print(f"Mandelbrot GPU time:             {mandel_gpu_s:.17g} s")
    print(f"Particle generation wall time:   {particle_generation_s:.17g} s")
    print(f"Initial force GPU time:          {init_force_gpu_s:.17g} s")
    print(f"Pure dynamics GPU time:          {pure_dynamics_gpu_s:.17g} s")
    print(f"Screen+copy GPU time:            {screen_and_copy_s:.17g} s")
    print(f"Final state copy time:           {final_copy_s:.17g} s")
    print(f"HDF5 write time:                 {hdf5_write_s:.17g} s")
    print(f"Validation time:                 {validation_s:.17g} s")
    print(f"Loop wall time:                  {loop_wall_s:.17g} s")
    print(f"GPU dynamics time per step:      {pure_dynamics_gpu_s / cfg.max_steps:.17g} s")
    print(f"Loop wall time per step:         {loop_wall_s / cfg.max_steps:.17g} s")

    if cfg.max_steps > 0 and pure_dynamics_gpu_s > 0.0:
        print(f"Pure dynamics performance:  {giga_interactions / pure_dynamics_gpu_s:.17g} GInteractions/s")
    if cfg.max_steps > 0 and loop_wall_s > 0.0:
        print(f"Loop end-to-end performance: {giga_interactions / loop_wall_s:.17g} GInteractions/s")

    print_validation_quantities(validation)
    cuda.synchronize()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

