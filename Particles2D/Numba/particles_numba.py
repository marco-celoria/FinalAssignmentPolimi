#!/usr/bin/env python3

import sys
import math
import time
from dataclasses import dataclass, field
from typing import Iterator, Tuple

import numpy as np
import h5py
from numba import njit, prange


# ============================================================
# CONSTANTS
# ============================================================

K_FORCE = 1e-3
EPS = 1e-2
EPS2 = EPS * EPS


# ============================================================
# UTIL
# ============================================================

def idx2d(i: int, j: int, nx: int) -> int:
    return i + j * nx


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

    def resize(self, N: int) -> None:
        self.n = N
        self.w = np.zeros(N, dtype=np.float64)
        self.x = np.zeros(N, dtype=np.float64)
        self.y = np.zeros(N, dtype=np.float64)
        self.vx = np.zeros(N, dtype=np.float64)
        self.vy = np.zeros(N, dtype=np.float64)


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

    if cfg.maxSteps == 0 or cfg.maxIters == 0:
        raise RuntimeError("Invalid iteration counts")

    if cfg.outputEvery == 0:
        raise RuntimeError("outputEvery must be > 0")


# ============================================================
# PARSER (STRICT + ROBUST)
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
    except OSError:
        raise RuntimeError("Cannot open input")

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


@njit(cache=True, parallel=True, fastmath=False)
def compute_forces_numba(x, y, w, fx, fy):
    N = x.shape[0]

    for i in prange(N):
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

            invr = 1.0 / np.sqrt(r2)
            invr2 = invr * invr
            invr3 = invr2 * invr
            coeff = K_FORCE * wi * w[j] * invr3

            fxi += coeff * dx
            fyi += coeff * dy

        fx[i] = fxi
        fy[i] = fyi


@njit(cache=True, parallel=True, fastmath=False)
def half_kick_drift_numba(x, y, vx, vy, w, fx, fy, dt):
    N = x.shape[0]
    half_dt = 0.5 * dt

    for i in prange(N):
        invm = 1.0 / w[i]

        # Match CUDA:
        # v += 0.5 * a_old * dt
        vx[i] += half_dt * fx[i] * invm
        vy[i] += half_dt * fy[i] * invm

        # x += v * dt
        x[i] += vx[i] * dt
        y[i] += vy[i] * dt


@njit(cache=True, parallel=True, fastmath=False)
def half_kick_numba(vx, vy, w, fx_new, fy_new, dt):
    N = vx.shape[0]
    half_dt = 0.5 * dt

    for i in prange(N):
        invm = 1.0 / w[i]

        # Match CUDA:
        # v += 0.5 * a_new * dt
        vx[i] += half_dt * fx_new[i] * invm
        vy[i] += half_dt * fy_new[i] * invm


@njit(cache=True, fastmath=False)
def build_screen_numba(values, nx, ny, xs, xe, ys, ye, x, y, w, wmin, wr):
    values[:] = 0

    N = x.shape[0]
    if N == 0:
        return

    invdx = (nx - 1) / (xe - xs) if xe != xs else 0.0
    invdy = (ny - 1) / (ye - ys) if ye != ys else 0.0

    for n in range(N):
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
    compute_generating_field_numba(g.values, g.nx, g.ny, g.xs, g.xe, g.ys, g.ye, max_iter)


def generate_particles(g: Grid, pg: Grid) -> Particles:
    P = Particles()

    vmax = int(np.max(g.values))
    vmin = int(np.min(g.values))
    vmin = (29 * vmax + vmin) // 30

    vals2 = g.values.reshape(g.ny, g.nx)
    mask = vals2 >= np.uint64(vmin)

    js, is_ = np.nonzero(mask)
    count = int(is_.size)

    if count == 0:
        raise RuntimeError("No particles generated")

    P.resize(count)

    selected_vals = vals2[js, is_].astype(np.float64)
    P.w[:] = np.maximum(1.0, 10.0 * selected_vals)

    P.x[:] = pg.xs + (pg.xe - pg.xs) * (is_.astype(np.float64) / float(g.nx - 1))
    P.y[:] = pg.ys + (pg.ye - pg.ys) * (js.astype(np.float64) / float(g.ny - 1))

    return P


def compute_forces(P: Particles, fx: np.ndarray, fy: np.ndarray) -> None:
    compute_forces_numba(P.x, P.y, P.w, fx, fy)


def integrate_vv(P: Particles,
                 fx: np.ndarray,
                 fy: np.ndarray,
                 fx_new: np.ndarray,
                 fy_new: np.ndarray,
                 dt: float):
    half_kick_drift_numba(P.x, P.y, P.vx, P.vy, P.w, fx, fy, dt)
    compute_forces_numba(P.x, P.y, P.w, fx_new, fy_new)
    half_kick_numba(P.vx, P.vy, P.w, fx_new, fy_new, dt)
    # Swap force buffers, same spirit as CUDA std::swap
    return fx_new, fy_new, fx, fy


def build_screen(g: Grid, P: Particles, wmin: float, wr: float) -> None:
    build_screen_numba(g.values, g.nx, g.ny, g.xs, g.xe, g.ys, g.ye,
                       P.x, P.y, P.w, wmin, wr)


# ============================================================
# HDF5 WRITER
# ============================================================

class H5StreamWriter:
    def __init__(self,
                 name: str,
                 nparticles: int,
                 nx: int,
                 ny: int,
                 chunk_frames: int = 64):
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

        self.pos = self.file.create_dataset(
            "/pos",
            shape=(0, self.np, 2),
            maxshape=(None, self.np, 2),
            chunks=(self.chunk_frames, self.np, 2),
            dtype=np.float64
        )

        self.vel = self.file.create_dataset(
            "/vel",
            shape=(0, self.np, 2),
            maxshape=(None, self.np, 2),
            chunks=(self.chunk_frames, self.np, 2),
            dtype=np.float64
        )

        self.grid = self.file.create_dataset(
            "/screen",
            shape=(0, self.ny, self.nx),
            maxshape=(None, self.ny, self.nx),
            chunks=(self.chunk_frames, self.ny, self.nx),
            dtype=np.uint64
        )

        self._extend_datasets(self.capacity)

    def _extend_datasets(self, new_size: int) -> None:
        self.pos.resize((new_size, self.np, 2))
        self.vel.resize((new_size, self.np, 2))
        self.grid.resize((new_size, self.ny, self.nx))

    def _shrink_to_fit(self) -> None:
        self.pos.resize((self.current_frame, self.np, 2))
        self.vel.resize((self.current_frame, self.np, 2))
        self.grid.resize((self.current_frame, self.ny, self.nx))

    def write_frame(self, P: Particles, G: Grid) -> None:
        if P.n != self.np:
            raise RuntimeError("Particle size mismatch")

        if G.nx != self.nx or G.ny != self.ny:
            raise RuntimeError("Grid size mismatch")

        if self.current_frame >= self.capacity:
            self.capacity += self.chunk_frames
            self._extend_datasets(self.capacity)

        self.Pbuf[:, 0] = P.x
        self.Pbuf[:, 1] = P.y
        self.Vbuf[:, 0] = P.vx
        self.Vbuf[:, 1] = P.vy

        self.pos[self.current_frame, :, :] = self.Pbuf
        self.vel[self.current_frame, :, :] = self.Vbuf
        self.grid[self.current_frame, :, :] = G.values.reshape(G.ny, G.nx)

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


# ============================================================
# JIT WARM-UP
# ============================================================

def warmup_numba() -> None:
    """
    Compile hot kernels before timing, so reported runtime excludes JIT cost.
    """
    # Tiny dummy arrays
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
    half_kick_drift_numba(x, y, vx, vy, w, fx, fy, 1e-3)
    compute_forces_numba(x, y, w, fx_new, fy_new)
    half_kick_numba(vx, vy, w, fx_new, fy_new, 1e-3)

    screen = np.zeros(4, dtype=np.uint64)
    build_screen_numba(screen, 2, 2, -1.0, 1.0, -1.0, 1.0, x, y, w, 1.0, 1.0)


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    try:
        input_file = sys.argv[1] if len(sys.argv) > 1 else "Particles.inp"

        cfg, gen, screen = read_input(input_file)

        # JIT compile before timing / real work where possible
        warmup_numba()

        # Pre-loop work (same spirit as CUDA: initial setup before timing)
        compute_generating_field(gen, cfg.maxIters)
        P = generate_particles(gen, screen)

        fx = np.empty(P.n, dtype=np.float64)
        fy = np.empty(P.n, dtype=np.float64)
        fx_new = np.empty(P.n, dtype=np.float64)
        fy_new = np.empty(P.n, dtype=np.float64)

        compute_forces(P, fx, fy)

        wmin = float(np.min(P.w))
        wmax = float(np.max(P.w))
        wr = max(wmax - wmin, 1.0)

        with H5StreamWriter("particles.h5", P.n, screen.nx, screen.ny) as h5:
            t0 = time.perf_counter()

            for step in range(cfg.maxSteps):
                if step % cfg.outputEvery == 0:
                    build_screen(screen, P, wmin, wr)
                    h5.write_frame(P, screen)

                fx, fy, fx_new, fy_new = integrate_vv(P, fx, fy, fx_new, fy_new, cfg.dt)

            t1 = time.perf_counter()

            if (cfg.maxSteps - 1) % cfg.outputEvery != 0:
                build_screen(screen, P, wmin, wr)
                h5.write_frame(P, screen)

        elapsed_s = t1 - t0
        interactions = float(P.n) * float(P.n - 1) * float(cfg.maxSteps)
        giga_interactions = interactions / 1e9

        print("Simulation completed successfully.")
        print(f"Total time: {elapsed_s} s")
        print(f"Time per step: {elapsed_s / cfg.maxSteps} s")
        print(f"Particles: {P.n}")
        print(f"Performance: {giga_interactions / elapsed_s} GInteractions/s")

        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

