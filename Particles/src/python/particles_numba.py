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
from numba import njit, prange, get_num_threads, set_num_threads


# ============================================================
# CONSTANTS
# ============================================================

K_FORCE = 1e-3
EPS = 1e-2
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
# PARSER
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
        except Exception:
            raise RuntimeError(f"Parse error: {line}")

    raise EOFError("Unexpected EOF")


def read_input(file_name: str) -> Tuple[Config, Grid, Grid]:
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            lines = iter(f.readlines())
    except OSError as e:
        raise RuntimeError(f"Cannot open input file: {file_name}") from e

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
# NUMBA KERNELS
# ============================================================

@njit(cache=True, parallel=True, fastmath=False)
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

            # This intentionally matches the CUDA particle version:
            # update z, then test escape, then increment if still bounded.
            while it < max_iter:
                a = za * za - zb * zb + ca
                b = 2.0 * za * zb + cb

                za = a
                zb = b

                if za * za + zb * zb > 4.0:
                    break

                it += 1

            values[base + i] = np.uint64(it)


@njit(cache=True, parallel=True, fastmath=False)
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

            invr = 1.0 / math.sqrt(r2)
            invr2 = invr * invr
            invr3 = invr2 * invr

            coeff = K_FORCE * wi * w[j] * invr3

            fxi += coeff * dx
            fyi += coeff * dy

        fx[i] = fxi
        fy[i] = fyi



@njit(cache=True, parallel=True, fastmath=False)
def half_kick_drift_numba(x, y, vx, vy, w, fx, fy, dt):
    n_particles = x.shape[0]

    for i in prange(n_particles):
        invm = 1.0 / w[i]

        vx[i] += 0.5 * fx[i] * invm * dt
        vy[i] += 0.5 * fy[i] * invm * dt

        x[i] += vx[i] * dt
        y[i] += vy[i] * dt

@njit(cache=True, parallel=True, fastmath=False)
def half_kick_numba(vx, vy, w, fx_new, fy_new, dt):
    n_particles = vx.shape[0]

    for i in prange(n_particles):
        invm = 1.0 / w[i]

        vx[i] += 0.5 * fx_new[i] * invm * dt
        vy[i] += 0.5 * fy_new[i] * invm * dt

@njit(cache=True, fastmath=False)
def build_screen_numba(values, nx, ny, xs, xe, ys, ye, x, y, w, wmin, wr):
    # Kept serial intentionally.
    # Multiple particles can update the same screen pixel, so naive prange would race.
    values[:] = np.uint64(0)

    n_particles = x.shape[0]

    if n_particles == 0:
        return

    invdx = (nx - 1) / (xe - xs) if xe != xs else 0.0
    invdy = (ny - 1) / (ye - ys) if ye != ys else 0.0

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


# ============================================================
# HIGH-LEVEL PHYSICS WRAPPERS
# ============================================================

def compute_generating_field(g: Grid, max_iter: int) -> None:
    compute_generating_field_numba(
        g.values,
        g.nx,
        g.ny,
        g.xs,
        g.xe,
        g.ys,
        g.ye,
        max_iter,
    )


def generate_particles(g: Grid, pg: Grid) -> Particles:
    if g.values.size == 0:
        raise RuntimeError("generate_particles: empty generating field")
    p = Particles()

    vmax = int(np.max(g.values))
    vmin = int(np.min(g.values))
    vmin = (29 * vmax + vmin) // 30

    vals2 = g.values.reshape(g.ny, g.nx)
    mask = vals2 >= np.uint64(vmin)

    j_idx, i_idx = np.nonzero(mask)

    count = int(i_idx.size)

    if count == 0:
        raise RuntimeError("No particles generated")

    p.resize(count)

    selected_vals = vals2[j_idx, i_idx].astype(np.float64)
    dx_range = pg.xe - pg.xs
    dy_range = pg.ye - pg.ys

    denom_x = float(g.nx - 1)
    denom_y = float(g.ny - 1)

    p.w[:] = np.maximum(1.0, 10.0 * selected_vals)
    p.x[:] = pg.xs + (dx_range * i_idx.astype(np.float64)) / denom_x
    p.y[:] = pg.ys + (dy_range * j_idx.astype(np.float64)) / denom_y

    # p.vx and p.vy are already zero-filled by resize().
    return p


def compute_forces(p: Particles, fx: np.ndarray, fy: np.ndarray) -> None:
    compute_forces_numba(p.x, p.y, p.w, fx, fy)


def integrate_vv(
    p: Particles,
    fx: np.ndarray,
    fy: np.ndarray,
    fx_new: np.ndarray,
    fy_new: np.ndarray,
    dt: float,
):
    half_kick_drift_numba(p.x, p.y, p.vx, p.vy, p.w, fx, fy, dt)

    compute_forces_numba(p.x, p.y, p.w, fx_new, fy_new)

    half_kick_numba(p.vx, p.vy, p.w, fx_new, fy_new, dt)

    # Return swapped force buffers.
    return fx_new, fy_new, fx, fy


def build_screen(g: Grid, p: Particles, wmin: float, wr: float) -> None:
    build_screen_numba(
        g.values,
        g.nx,
        g.ny,
        g.xs,
        g.xe,
        g.ys,
        g.ye,
        p.x,
        p.y,
        p.w,
        wmin,
        wr,
    )


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

        self.grid = self.file.create_dataset(
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

        self._extend_datasets(self.capacity)

    def _extend_datasets(self, new_size: int) -> None:
        self.pos.resize((new_size, self.np, 2))
        self.vel.resize((new_size, self.np, 2))
        self.grid.resize((new_size, self.ny, self.nx))
        self.step.resize((new_size,))

    def _shrink_to_fit(self) -> None:
        self.pos.resize((self.current_frame, self.np, 2))
        self.vel.resize((self.current_frame, self.np, 2))
        self.grid.resize((self.current_frame, self.ny, self.nx))
        self.step.resize((self.current_frame,))

    def write_frame(self, step_number: int, p: Particles, g: Grid) -> None:
        if self.closed:
            raise RuntimeError("H5StreamWriter: write_frame called after close")

        if p.n != self.np:
            raise RuntimeError("Particle size mismatch")

        if g.nx != self.nx or g.ny != self.ny:
            raise RuntimeError("Grid size mismatch")

        if g.values.size != self.nx * self.ny:
            raise RuntimeError("Screen buffer size mismatch")

        if self.current_frame >= self.capacity:
            self.capacity += self.chunk_frames
            self._extend_datasets(self.capacity)

        self.Pbuf[:, 0] = p.x
        self.Pbuf[:, 1] = p.y

        self.Vbuf[:, 0] = p.vx
        self.Vbuf[:, 1] = p.vy

        self.pos[self.current_frame, :, :] = self.Pbuf
        self.vel[self.current_frame, :, :] = self.Vbuf
        self.grid[self.current_frame, :, :] = g.values.reshape(g.ny, g.nx)
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
    """
    Compile hot kernels before timing so reported runtime excludes JIT cost.
    """
    values = np.zeros(4, dtype=np.uint64)

    compute_generating_field_numba(
        values,
        2,
        2,
        -2.0,
        1.0,
        -1.0,
        1.0,
        4,
    )

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
    half_kick_drift_numba(x, y, vx, vy, w, fx, fy, 1e-3)
    compute_forces_numba(x, y, w, fx_new, fy_new)
    half_kick_numba(vx, vy, w, fx_new, fy_new, 1e-3)

    screen = np.zeros(4, dtype=np.uint64)

    build_screen_numba(
        screen,
        2,
        2,
        -1.0,
        1.0,
        -1.0,
        1.0,
        x,
        y,
        w,
        1.0,
        1.0,
    )


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="CPU-only Numba particle solver fallback"
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
        "--threads",
        type=int,
        default=None,
        help="Number of Numba CPU threads",
    )

    parser.add_argument(
        "--screen-tile-y",
        type=int,
        default=256,
        help="HDF5 screen chunk tile size in y",
    )

    parser.add_argument(
        "--screen-tile-x",
        type=int,
        default=256,
        help="HDF5 screen chunk tile size in x",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    try:
        args = parse_args()

        if args.threads is not None:
            if args.threads <= 0:
                raise ValueError("--threads must be > 0")

            try:
                set_num_threads(args.threads)
            except ValueError as e:
                raise ValueError(
                    f"Cannot set Numba threads to {args.threads}."
                ) from e

        validate_h5_tiles(args.screen_tile_y, args.screen_tile_x)

        cfg, gen, screen = read_input(args.input)

        if args.output_every is not None:
            if args.output_every <= 0:
                raise ValueError("outputEvery override must be > 0")
            cfg.outputEvery = args.output_every

        # JIT compile before timing / real work.
        warmup_numba()

        # ----------------------------------------------------
        # Pre-loop setup
        # ----------------------------------------------------
        gen_t0 = time.perf_counter()

        compute_generating_field(gen, cfg.maxIters)

        gen_t1 = time.perf_counter()

        part_t0 = time.perf_counter()

        p = generate_particles(gen, screen)

        part_t1 = time.perf_counter()

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

        # ----------------------------------------------------
        # Timed simulation loop
        # ----------------------------------------------------
        pure_dynamics_time_s = 0.0
        output_time_s = 0.0
        output_frames = 0

        # Used to avoid duplicate output frames with the same step number.
        has_last_written_step = False
        last_written_step = -1

        loop_t0 = time.perf_counter()

        with H5StreamWriter(
            args.output,
            p.n,
            screen.nx,
            screen.ny,
            screen_tile_y=args.screen_tile_y,
            screen_tile_x=args.screen_tile_x,
        ) as h5:

            def write_output_frame(step: int) -> None:
                nonlocal output_frames
                nonlocal output_time_s
                nonlocal has_last_written_step
                nonlocal last_written_step

                if has_last_written_step and step == last_written_step:
                    return

                out_t0 = time.perf_counter()

                build_screen(screen, p, wmin, wr)
                h5.write_frame(step, p, screen)

                out_t1 = time.perf_counter()

                output_time_s += out_t1 - out_t0
                output_frames += 1

                has_last_written_step = True
                last_written_step = step

            for step in range(cfg.maxSteps):
                # Output current state before advancing.
                if step % cfg.outputEvery == 0:
                    write_output_frame(step)

                dyn_t0 = time.perf_counter()

                fx, fy, fx_new, fy_new = integrate_vv(
                    p,
                    fx,
                    fy,
                    fx_new,
                    fy_new,
                    cfg.dt,
                )

                dyn_t1 = time.perf_counter()
                pure_dynamics_time_s += dyn_t1 - dyn_t0

            # Always save the final state after cfg.maxSteps integrations.
            # The write_output_frame guard prevents accidental duplication.
            write_output_frame(cfg.maxSteps)

        loop_t1 = time.perf_counter()

        loop_wall_s = loop_t1 - loop_t0

        generating_field_s = gen_t1 - gen_t0
        particle_generation_s = part_t1 - part_t0
        initial_force_s = init_force_t1 - init_force_t0

        interactions = float(p.n) * float(p.n - 1) * float(cfg.maxSteps)
        giga_interactions = interactions / 1e9

        print("Simulation completed successfully.")
        print(f"Input file:                 {args.input}")
        print(f"HDF5 output:                {args.output}")
        print(f"Particles:                  {p.n}")
        print(f"Steps:                      {cfg.maxSteps}")
        print(f"Output every:               {cfg.outputEvery}")
        print(f"Output frames:              {output_frames}")
        print(f"Numba threads:              {get_num_threads()}")
        print(f"NUMBA_NUM_THREADS env:      {os.environ.get('NUMBA_NUM_THREADS', '(not set)')}")
        print(f"Generating field time:      {generating_field_s:.6f} s")
        print(f"Particle generation time:   {particle_generation_s:.6f} s")
        print(f"Initial force time:         {initial_force_s:.6f} s")
        print(f"Pure dynamics time:         {pure_dynamics_time_s:.6f} s")
        print(f"Screen + HDF5 output time:  {output_time_s:.6f} s")
        print(f"Loop wall time:             {loop_wall_s:.6f} s")
        print(f"Dynamics time per step:     {pure_dynamics_time_s / cfg.maxSteps:.6e} s")
        print(f"Wall time per step:         {loop_wall_s / cfg.maxSteps:.6e} s")

        if pure_dynamics_time_s > 0.0:
            print(
                f"Pure dynamics performance:  "
                f"{giga_interactions / pure_dynamics_time_s:.6f} GInteractions/s"
            )

        if loop_wall_s > 0.0:
            print(
                f"End-to-end performance:     "
                f"{giga_interactions / loop_wall_s:.6f} GInteractions/s"
            )

        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
