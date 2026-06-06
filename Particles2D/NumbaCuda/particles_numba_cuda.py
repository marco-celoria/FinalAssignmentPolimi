#!/usr/bin/env python3

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Iterator, Tuple

import h5py
import numpy as np
from numba import cuda, float64


# ============================================================
# CONSTANTS
# ============================================================

K_FORCE = 1.0e-3
EPS = 1.0e-2
EPS2 = EPS * EPS
BLOCK_SIZE = 256


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
            raise ValueError("Grid.allocate: grid dimensions must be > 0")

        self.values = np.zeros(self.nx * self.ny, dtype=np.uint64)


@dataclass
class Particles:
    n: int = 0
    w: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    x: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    y: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    vx: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    vy: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    def resize(self, n_particles: int) -> None:
        if n_particles <= 0:
            raise ValueError("Particles.resize: number of particles must be > 0")

        self.n = n_particles

        self.w = np.empty(n_particles, dtype=np.float64)
        self.x = np.empty(n_particles, dtype=np.float64)
        self.y = np.empty(n_particles, dtype=np.float64)
        self.vx = np.zeros(n_particles, dtype=np.float64)
        self.vy = np.zeros(n_particles, dtype=np.float64)


@dataclass
class Config:
    maxIters: int = 0
    maxSteps: int = 0
    outputEvery: int = 10
    dt: float = 0.0


@dataclass
class HostOutputBuffers:
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    screen_i64: np.ndarray
    screen_u64: np.ndarray


# ============================================================
# VALIDATION
# ============================================================

def validate(g: Grid, pg: Grid, cfg: Config) -> None:
    if g.nx < 2 or g.ny < 2 or pg.nx < 2 or pg.ny < 2:
        raise RuntimeError("Grids must have at least 2 points")

    if g.xe <= g.xs or g.ye <= g.ys:
        raise RuntimeError("Invalid generating domain")

    if pg.xe <= pg.xs or pg.ye <= pg.ys:
        raise RuntimeError("Invalid particle domain")

    if cfg.dt <= 0.0:
        raise RuntimeError("dt must be > 0")

    if cfg.maxSteps <= 0 or cfg.maxIters <= 0:
        raise RuntimeError("maxSteps and maxIters must be > 0")

    if cfg.outputEvery <= 0:
        raise RuntimeError("outputEvery must be > 0")


def validate_h5_tiles(tile_y: int, tile_x: int) -> None:
    if tile_y <= 0 or tile_x <= 0:
        raise ValueError("HDF5 tile sizes must be > 0")


# ============================================================
# INPUT PARSER
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

        if "#" in stripped:
            content, _comment = stripped.split("#", 1)
            content = content.rstrip()

            if not content:
                continue

            tokens = content.split()
        else:
            tokens = stripped.split()

        if len(tokens) != 1:
            raise RuntimeError(f"Trailing junk: {line}")

        try:
            return _parse_scalar(tokens[0], typ)
        except Exception as exc:
            raise RuntimeError(f"Parse error: {line}") from exc

    raise EOFError("Unexpected EOF")


def read_input(file_name: str) -> Tuple[Config, Grid, Grid]:
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            lines = iter(f.readlines())
    except OSError as exc:
        raise RuntimeError(f"Cannot open input file: {file_name}") from exc

    g = Grid()
    pg = Grid()
    cfg = Config()

    g.nx = parse_line(lines, int)
    g.ny = parse_line(lines, int)

    g.xs = parse_line(lines, float)
    g.xe = parse_line(lines, float)
    g.ys = parse_line(lines, float)
    g.ye = parse_line(lines, float)

    pg.nx = parse_line(lines, int)
    pg.ny = parse_line(lines, int)

    pg.xs = parse_line(lines, float)
    pg.xe = parse_line(lines, float)
    pg.ys = parse_line(lines, float)
    pg.ye = parse_line(lines, float)

    cfg.maxIters = parse_line(lines, int)
    cfg.maxSteps = parse_line(lines, int)
    cfg.dt = parse_line(lines, float)
    cfg.outputEvery = parse_line(lines, int)

    validate(g, pg, cfg)

    g.allocate()
    pg.allocate()

    return cfg, g, pg


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

    # Matches CUDA C++ semantics:
    # update z, then test escape, then increment if still bounded.
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
    sh_x = cuda.shared.array(shape=BLOCK_SIZE, dtype=float64)
    sh_y = cuda.shared.array(shape=BLOCK_SIZE, dtype=float64)
    sh_w = cuda.shared.array(shape=BLOCK_SIZE, dtype=float64)

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

    tiles = (n_particles + BLOCK_SIZE - 1) // BLOCK_SIZE

    for tile in range(tiles):
        j = tile * BLOCK_SIZE + tid

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
            for k in range(BLOCK_SIZE):
                global_j = tile * BLOCK_SIZE + k

                if global_j >= n_particles or global_j == i:
                    continue

                dx = sh_x[k] - xi
                dy = sh_y[k] - yi

                r2 = dx * dx + dy * dy + EPS2

                invr = 1.0 / math.sqrt(r2)
                invr2 = invr * invr
                invr3 = invr2 * invr

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
def build_screen_kernel(
    screen,
    x,
    y,
    w,
    nx,
    ny,
    xs,
    ys,
    invdx,
    invdy,
    wmin,
    wr,
    n_particles,
):
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

            # screen is int64 intentionally.
            # Values are non-negative and converted to uint64 before HDF5 write.
            cuda.atomic.add(screen, row + jx, wp)


# ============================================================
# HIGH-LEVEL HELPERS
# ============================================================

def generate_particles(g: Grid, pg: Grid) -> Particles:
    if g.values.size == 0:
        raise RuntimeError("generate_particles: empty generating field")

    p = Particles()

    vmax = int(np.max(g.values))
    vmin = int(np.min(g.values))

    vmin = (29 * vmax + vmin) // 30

    vals2 = g.values.reshape(g.ny, g.nx)
    mask = vals2 >= np.uint64(vmin)

    js, is_ = np.nonzero(mask)
    count = int(is_.size)

    if count == 0:
        raise RuntimeError("No particles generated")

    p.resize(count)

    selected_vals = vals2[js, is_].astype(np.float64)

    p.w[:] = np.maximum(1.0, 10.0 * selected_vals)

    p.x[:] = pg.xs + (pg.xe - pg.xs) * (
        is_.astype(np.float64) / float(g.nx - 1)
    )

    p.y[:] = pg.ys + (pg.ye - pg.ys) * (
        js.astype(np.float64) / float(g.ny - 1)
    )

    p.vx.fill(0.0)
    p.vy.fill(0.0)

    return p


def allocate_host_output_buffers(
    nparticles: int,
    screen_size: int,
) -> HostOutputBuffers:
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


def launch_output_copy(
    screen_d,
    x_d,
    y_d,
    vx_d,
    vy_d,
    host: HostOutputBuffers,
    stream,
) -> None:
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
        if nparticles <= 0:
            raise ValueError("H5StreamWriter: nparticles must be > 0")
        if nx <= 0 or ny <= 0:
            raise ValueError("H5StreamWriter: nx and ny must be > 0")
        if chunk_frames <= 0:
            raise ValueError("H5StreamWriter: chunk_frames must be > 0")
        if screen_tile_y <= 0 or screen_tile_x <= 0:
            raise ValueError("H5StreamWriter: screen tile sizes must be > 0")

        self.file = h5py.File(name, "w")

        self.np = nparticles
        self.nx = nx
        self.ny = ny
        self.chunk_frames = chunk_frames
        self.capacity = chunk_frames
        self.current_frame = 0
        self.closed = False

        self.Pbuf = np.empty((self.np, 2), dtype=np.float64)
        self.Vbuf = np.empty((self.np, 2), dtype=np.float64)

        screen_chunk_y = min(self.ny, screen_tile_y)
        screen_chunk_x = min(self.nx, screen_tile_x)

        # /pos and /vel: [frame, particle, component]
        # Safe one-frame chunking, matching the corrected C++ version.
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

        # /screen: [frame, y, x]
        # Tile the spatial dimensions to avoid enormous HDF5 chunks.
        self.screen = self.file.create_dataset(
            "/screen",
            shape=(0, self.ny, self.nx),
            maxshape=(None, self.ny, self.nx),
            chunks=(1, screen_chunk_y, screen_chunk_x),
            dtype=np.uint64,
        )

        # /step: [frame]
        self.step = self.file.create_dataset(
            "/step",
            shape=(0,),
            maxshape=(None,),
            chunks=(self.chunk_frames,),
            dtype=np.int64,
        )

        self._extend_datasets(self.capacity)

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

    def write_frame_arrays(
        self,
        step_number: int,
        x: np.ndarray,
        y: np.ndarray,
        vx: np.ndarray,
        vy: np.ndarray,
        screen_values: np.ndarray,
    ) -> None:
        if self.closed:
            raise RuntimeError("H5StreamWriter: write after close")

        if x.size != self.np or y.size != self.np:
            raise RuntimeError("Particle position size mismatch")

        if vx.size != self.np or vy.size != self.np:
            raise RuntimeError("Particle velocity size mismatch")

        if screen_values.size != self.nx * self.ny:
            raise RuntimeError("Screen size mismatch")

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
    """
    Compile CUDA kernels before timing the real workload.
    """
    d_vals = cuda.device_array(4, dtype=np.uint64)

    mandelbrot_kernel[(1, 1), (2, 2)](
        d_vals,
        2,
        2,
        -2.0,
        -1.0,
        3.0,
        2.0,
        4,
    )

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

    compute_forces_tiled_kernel[1, BLOCK_SIZE](
        x_d,
        y_d,
        w_d,
        fx_d,
        fy_d,
        2,
    )

    half_kick_drift_kernel[1, BLOCK_SIZE](
        x_d,
        y_d,
        vx_d,
        vy_d,
        w_d,
        fx_d,
        fy_d,
        1.0e-3,
        2,
    )

    compute_forces_tiled_kernel[1, BLOCK_SIZE](
        x_d,
        y_d,
        w_d,
        fx_new_d,
        fy_new_d,
        2,
    )

    half_kick_kernel[1, BLOCK_SIZE](
        vx_d,
        vy_d,
        w_d,
        fx_new_d,
        fy_new_d,
        1.0e-3,
        2,
    )

    screen_d = cuda.device_array(4, dtype=np.int64)

    zero_int64_kernel[1, BLOCK_SIZE](screen_d, 4)

    build_screen_kernel[1, BLOCK_SIZE](
        screen_d,
        x_d,
        y_d,
        w_d,
        2,
        2,
        -1.0,
        -1.0,
        0.5,
        0.5,
        1.0,
        1.0,
        2,
    )

    cuda.synchronize()


# ============================================================
# MAIN SIMULATION HELPERS
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
    zero_int64_kernel[blocks_screen, threads, stream](
        screen_d,
        screen.nx * screen.ny,
    )

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

    launch_output_copy(
        screen_d,
        x_d,
        y_d,
        vx_d,
        vy_d,
        host,
        stream,
    )


def queue_verlet_step(
    x_d,
    y_d,
    vx_d,
    vy_d,
    w_d,
    fx_d,
    fy_d,
    fx_new_d,
    fy_new_d,
    dt: float,
    nparticles: int,
    blocks_particles: int,
    threads: int,
    stream,
) -> None:
    half_kick_drift_kernel[blocks_particles, threads, stream](
        x_d,
        y_d,
        vx_d,
        vy_d,
        w_d,
        fx_d,
        fy_d,
        dt,
        nparticles,
    )

    compute_forces_tiled_kernel[blocks_particles, threads, stream](
        x_d,
        y_d,
        w_d,
        fx_new_d,
        fy_new_d,
        nparticles,
    )

    half_kick_kernel[blocks_particles, threads, stream](
        vx_d,
        vy_d,
        w_d,
        fx_new_d,
        fy_new_d,
        dt,
        nparticles,
    )


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="CUDA Numba GPU particle solver"
    )

    parser.add_argument(
        "input",
        nargs="?",
        default="Particles.inp",
        help="Input file",
    )

    parser.add_argument(
        "output",
        nargs="?",
        default="particles.h5",
        help="Output HDF5 file",
    )

    parser.add_argument(
        "output_every",
        nargs="?",
        type=int,
        default=None,
        help="Override outputEvery from input file",
    )

    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=64,
        help="HDF5 frame chunk size for /step and dataset extension capacity",
    )

    parser.add_argument(
        "--screen-tile-y",
        type=int,
        default=256,
        help="HDF5 /screen chunk tile size in y",
    )

    parser.add_argument(
        "--screen-tile-x",
        type=int,
        default=256,
        help="HDF5 /screen chunk tile size in x",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    try:
        args = parse_args()

        if not cuda.is_available():
            raise RuntimeError("CUDA is not available")

        if args.chunk_frames <= 0:
            raise ValueError("--chunk-frames must be > 0")

        validate_h5_tiles(args.screen_tile_y, args.screen_tile_x)

        input_file = args.input
        output_file = args.output

        cfg, gen, screen = read_input(input_file)

        if args.output_every is not None:
            if args.output_every <= 0:
                raise ValueError("outputEvery override must be > 0")
            cfg.outputEvery = args.output_every

        # Compile kernels before timing the real workload.
        warmup_cuda()

        main_stream = cuda.stream()

        print("Input file:                 ", input_file)
        print("HDF5 output:                ", output_file)
        print("Generating grid:            ", f"{gen.nx} x {gen.ny}")
        print("Screen grid:                ", f"{screen.nx} x {screen.ny}")
        print("Max iterations:             ", cfg.maxIters)
        print("Steps:                      ", cfg.maxSteps)
        print("Output every:               ", cfg.outputEvery)
        print("CUDA device:                ", cuda.get_current_device().name.decode())

        # ----------------------------------------------------
        # 1. Generate Mandelbrot field on GPU
        # ----------------------------------------------------
        d_vals = cuda.device_array(
            gen.nx * gen.ny,
            dtype=np.uint64,
            stream=main_stream,
        )

        block2d = (16, 16)
        grid2d = (
            (gen.nx + block2d[0] - 1) // block2d[0],
            (gen.ny + block2d[1] - 1) // block2d[1],
        )

        dx_gen = (gen.xe - gen.xs) / float(gen.nx - 1)
        dy_gen = (gen.ye - gen.ys) / float(gen.ny - 1)

        mandel_start = cuda.event()
        mandel_stop = cuda.event()

        mandel_start.record(main_stream)

        mandelbrot_kernel[grid2d, block2d, main_stream](
            d_vals,
            gen.nx,
            gen.ny,
            gen.xs,
            gen.ys,
            dx_gen,
            dy_gen,
            cfg.maxIters,
        )

        mandel_stop.record(main_stream)
        mandel_stop.synchronize()

        mandel_gpu_s = cuda.event_elapsed_time(
            mandel_start,
            mandel_stop,
        ) / 1000.0

        gen.values = d_vals.copy_to_host()

        # ----------------------------------------------------
        # 2. Generate particles on host
        # ----------------------------------------------------
        particle_t0 = time.perf_counter()

        particles = generate_particles(gen, screen)

        particle_t1 = time.perf_counter()

        particle_generation_s = particle_t1 - particle_t0

        print("Particles:                  ", particles.n)

        # ----------------------------------------------------
        # 3. Copy particles to GPU
        # ----------------------------------------------------
        x_d = cuda.to_device(particles.x, stream=main_stream)
        y_d = cuda.to_device(particles.y, stream=main_stream)
        vx_d = cuda.to_device(particles.vx, stream=main_stream)
        vy_d = cuda.to_device(particles.vy, stream=main_stream)
        w_d = cuda.to_device(particles.w, stream=main_stream)

        fx_d = cuda.device_array(
            particles.n,
            dtype=np.float64,
            stream=main_stream,
        )

        fy_d = cuda.device_array(
            particles.n,
            dtype=np.float64,
            stream=main_stream,
        )

        fx_new_d = cuda.device_array(
            particles.n,
            dtype=np.float64,
            stream=main_stream,
        )

        fy_new_d = cuda.device_array(
            particles.n,
            dtype=np.float64,
            stream=main_stream,
        )

        screen_size = screen.nx * screen.ny

        # int64 screen is used because Numba CUDA int64 atomics are generally
        # better supported than uint64 atomics. Values are non-negative.
        screen_d = cuda.device_array(
            screen_size,
            dtype=np.int64,
            stream=main_stream,
        )

        threads = BLOCK_SIZE
        blocks_particles = (particles.n + threads - 1) // threads
        blocks_screen = (screen_size + threads - 1) // threads

        # ----------------------------------------------------
        # 4. Initial force computation
        # ----------------------------------------------------
        init_force_start = cuda.event()
        init_force_stop = cuda.event()

        init_force_start.record(main_stream)

        compute_forces_tiled_kernel[blocks_particles, threads, main_stream](
            x_d,
            y_d,
            w_d,
            fx_d,
            fy_d,
            particles.n,
        )

        init_force_stop.record(main_stream)
        init_force_stop.synchronize()

        init_force_gpu_s = cuda.event_elapsed_time(
            init_force_start,
            init_force_stop,
        ) / 1000.0

        # ----------------------------------------------------
        # 5. Host constants and pinned buffers
        # ----------------------------------------------------
        wmin = float(np.min(particles.w))
        wmax = float(np.max(particles.w))
        wr = max(wmax - wmin, 1.0)

        invdx_screen = float(screen.nx - 1) / (screen.xe - screen.xs)
        invdy_screen = float(screen.ny - 1) / (screen.ye - screen.ys)

        host = allocate_host_output_buffers(
            particles.n,
            screen_size,
        )

        # ----------------------------------------------------
        # 6. Simulation loop
        #
        # gpu_timed_s includes:
        #   - integration kernels
        #   - force kernels
        #   - screen build kernels
        #   - device-to-host output copies
        #
        # gpu_timed_s excludes:
        #   - HDF5 write time
        #
        # wall_s includes HDF5.
        # ----------------------------------------------------
        segment_start = cuda.event()
        segment_stop = cuda.event()

        gpu_ms = 0.0
        output_frames = 0

        # Guard against accidentally writing the same step twice.
        has_last_written_step = False
        last_written_step = -1

        def start_gpu_segment() -> None:
            segment_start.record(main_stream)

        def stop_gpu_segment_and_sync() -> float:
            segment_stop.record(main_stream)
            segment_stop.synchronize()
            return cuda.event_elapsed_time(segment_start, segment_stop)

        wall_t0 = time.perf_counter()

        with H5StreamWriter(
            output_file,
            particles.n,
            screen.nx,
            screen.ny,
            chunk_frames=args.chunk_frames,
            screen_tile_y=args.screen_tile_y,
            screen_tile_x=args.screen_tile_x,
        ) as h5:

            def write_output_frame(step: int, restart_segment: bool) -> None:
                nonlocal gpu_ms
                nonlocal output_frames
                nonlocal has_last_written_step
                nonlocal last_written_step

                if has_last_written_step and step == last_written_step:
                    return

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

                # This synchronizes the queued GPU work and D2H copies.
                gpu_ms += stop_gpu_segment_and_sync()

                # Convert non-negative int64 screen to uint64 for HDF5.
                np.copyto(
                    host.screen_u64,
                    host.screen_i64,
                    casting="unsafe",
                )

                h5.write_frame_arrays(
                    step_number=step,
                    x=host.x,
                    y=host.y,
                    vx=host.vx,
                    vy=host.vy,
                    screen_values=host.screen_u64,
                )

                has_last_written_step = True
                last_written_step = step
                output_frames += 1

                if restart_segment:
                    start_gpu_segment()

            start_gpu_segment()

            for step in range(cfg.maxSteps):
                # Output current state before advancing,
                # matching CUDA C++ logic.
                if step % cfg.outputEvery == 0:
                    write_output_frame(
                        step=step,
                        restart_segment=True,
                    )

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

            # Always save the final physical state after cfg.maxSteps integrations.
            # The duplicate guard prevents accidental duplicate frames.
            write_output_frame(
                step=cfg.maxSteps,
                restart_segment=False,
            )

        wall_t1 = time.perf_counter()
        gpu_timed_s = gpu_ms / 1000.0
        wall_s = wall_t1 - wall_t0

        interactions = (
            float(particles.n)
            * float(particles.n - 1)
            * float(cfg.maxSteps)
        )

        giga_interactions = interactions / 1.0e9

        print("Simulation completed successfully.")
        print(f"Output frames:              {output_frames}")
        print(f"Mandelbrot GPU time:        {mandel_gpu_s:.6f} s")
        print(f"Particle generation wall:   {particle_generation_s:.6f} s")
        print(f"Initial force GPU time:     {init_force_gpu_s:.6f} s")
        print(f"Timed GPU pipeline time:    {gpu_timed_s:.6f} s")
        print(f"Wall time including HDF5:   {wall_s:.6f} s")
        print(f"GPU time per step:          {gpu_timed_s / cfg.maxSteps:.6e} s")
        print(f"Wall time per step:         {wall_s / cfg.maxSteps:.6e} s")

        if gpu_timed_s > 0.0:
            print(
                f"GPU pipeline performance:   "
                f"{giga_interactions / gpu_timed_s:.6f} GInteractions/s"
            )

        if wall_s > 0.0:
            print(
                f"End-to-end performance:     "
                f"{giga_interactions / wall_s:.6f} GInteractions/s"
            )

        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
