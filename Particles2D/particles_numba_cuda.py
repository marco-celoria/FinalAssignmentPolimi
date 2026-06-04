#!/usr/bin/env python3

import sys
import math
import time
from dataclasses import dataclass, field
from typing import Tuple, Iterator

import numpy as np
import h5py
from numba import cuda, float64


# ============================================================
# CONSTANTS
# ============================================================

K_FORCE = 1e-3
EPS = 1e-2
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

    def allocate(self):
        self.values = np.zeros(self.nx * self.ny, dtype=np.uint64)


@dataclass
class Particles:
    n: int = 0
    w: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    x: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    y: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    vx: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    vy: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    def resize(self, N):
        self.n = N
        self.w = np.empty(N, dtype=np.float64)
        self.x = np.empty(N, dtype=np.float64)
        self.y = np.empty(N, dtype=np.float64)
        self.vx = np.empty(N, dtype=np.float64)
        self.vy = np.empty(N, dtype=np.float64)


@dataclass
class Config:
    maxIters: int = 0
    maxSteps: int = 0
    outputEvery: int = 10
    dt: float = 0.0


# ============================================================
# PARSER
# ============================================================

def parse_line(lines: Iterator[str], typ):
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        s = s.split("#")[0].strip()
        return typ(s)
    raise EOFError("Unexpected EOF")


def read_input(fname: str) -> Tuple[Config, Grid, Grid]:
    with open(fname) as f:
        lines = iter(f.readlines())

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

    g.allocate()
    pg.allocate()

    return cfg, g, pg


# ============================================================
# GPU KERNELS
# ============================================================

@cuda.jit
def mandelbrot_kernel(values, nx, ny, xs, xe, ys, ye, maxIter):
    i, j = cuda.grid(2)

    if i < nx and j < ny:
        dx = (xe - xs) / (nx - 1)
        dy = (ye - ys) / (ny - 1)

        ca = xs + i * dx
        cb = ys + j * dy

        za = 0.0
        zb = 0.0
        it = 0

        while it < maxIter:
            a = za * za - zb * zb + ca
            b = 2.0 * za * zb + cb
            za = a
            zb = b
            it += 1
            if za * za + zb * zb > 4.0:
                break

        values[i + j * nx] = np.uint64(it)


# ------------------------------------------------------------
# SHARED-MEMORY TILED N-BODY (IMPORTANT)
# ------------------------------------------------------------

@cuda.jit
def compute_forces_tiled(x, y, w, fx, fy, N):
    sh_x = cuda.shared.array(BLOCK_SIZE, float64)
    sh_y = cuda.shared.array(BLOCK_SIZE, float64)
    sh_w = cuda.shared.array(BLOCK_SIZE, float64)

    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    tid = cuda.threadIdx.x

    active = i < N

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

    tiles = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    for tile in range(tiles):
        j = tile * BLOCK_SIZE + tid

        if j < N:
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
                if global_j >= N or global_j == i:
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


# ------------------------------------------------------------

@cuda.jit
def update_position(x, y, vx, vy, fx, fy, w, dt, N):
    i = cuda.grid(1)
    if i >= N:
        return

    invm = 1.0 / w[i]

    vx[i] += 0.5 * fx[i] * invm * dt
    vy[i] += 0.5 * fy[i] * invm * dt

    x[i] += vx[i] * dt
    y[i] += vy[i] * dt


@cuda.jit
def update_velocity(vx, vy, fx, fy, w, dt, N):
    i = cuda.grid(1)
    if i >= N:
        return

    invm = 1.0 / w[i]

    vx[i] += 0.5 * fx[i] * invm * dt
    vy[i] += 0.5 * fy[i] * invm * dt


@cuda.jit
def build_screen(screen, x, y, w, nx, ny, xs, xe, ys, ye, wmin, wr, N):
    n = cuda.grid(1)
    if n >= N:
        return

    invdx = (nx - 1) / (xe - xs) if xe != xs else 0.0
    invdy = (ny - 1) / (ye - ys) if ye != ys else 0.0

    ix = int((x[n] - xs) * invdx)
    iy = int((y[n] - ys) * invdy)

    ix = max(0, min(ix, nx - 1))
    iy = max(0, min(iy, ny - 1))

    wp = int(10.0 * (w[n] - wmin) / wr)
    wp = max(0, min(wp, 1000))

    wp = np.uint64(wp)

    for dj in range(-1, 2):
        jy = iy + dj
        if jy < 0 or jy >= ny:
            continue

        for di in range(-1, 2):
            jx = ix + di
            if jx < 0 or jx >= nx:
                continue

            cuda.atomic.add(screen, jx + jy * nx, wp)


# ============================================================
# MAIN
# ============================================================

def main():
    cfg, gen, screen = read_input("Particles.inp")

    # Mandelbrot on GPU
    d_vals = cuda.device_array(gen.nx * gen.ny, dtype=np.uint64)

    block2d = (16, 16)
    grid2d = ((gen.nx + 15)//16, (gen.ny + 15)//16)

    mandelbrot_kerneld_vals, gen.nx, gen.ny,
        gen.xs, gen.xe, gen.ys, gen.ye, cfg.maxIters
    cuda.synchronize()

    gen.values = d_vals.copy_to_host()

    # Generate particles
    vals = gen.values.reshape(gen.ny, gen.nx)
    vmax = int(np.max(vals))
    vmin = (29 * vmax + int(np.min(vals))) // 30

    js, is_ = np.nonzero(vals >= vmin)
    N = len(is_)

    P = Particles()
    P.resize(N)

    P.w[:] = np.maximum(1.0, 10.0 * vals[js, is_])
    P.x[:] = screen.xs + (screen.xe - screen.xs) * is_ / (gen.nx - 1)
    P.y[:] = screen.ys + (screen.ye - screen.ys) * js / (gen.ny - 1)
    P.vx.fill(0.0)
    P.vy.fill(0.0)

    # GPU buffers
    x_d = cuda.to_device(P.x)
    y_d = cuda.to_device(P.y)
    vx_d = cuda.to_device(P.vx)
    vy_d = cuda.to_device(P.vy)
    w_d = cuda.to_device(P.w)

    fx_d = cuda.device_array(N)
    fy_d = cuda.device_array(N)
    fx_new_d = cuda.device_array(N)
    fy_new_d = cuda.device_array(N)

    screen_d = cuda.device_array(screen.nx * screen.ny, dtype=np.uint64)

    threads = BLOCK_SIZE
    blocks = (N + threads - 1) // threads

    compute_forces_tiledx_d, y_d, w_d, fx_d, fy_d, N
    cuda.synchronize()

    wmin = float(np.min(P.w))
    wmax = float(np.max(P.w))
    wr = max(wmax - wmin, 1.0)

    t0 = time.time()

    for step in range(cfg.maxSteps):

        if step % cfg.outputEvery == 0:
            screen_d[:] = 0

            build_screenscreen_d, x_d, y_d, w_d,
                screen.nx, screen.ny,
                screen.xs, screen.xe,
                screen.ys, screen.ye,
                wmin, wr, N
            
            cuda.synchronize()

        update_positionx_d, y_d, vx_d, vy_d, fx_d, fy_d, w_d, cfg.dt, N
        compute_forces_tiledx_d, y_d, w_d, fx_new_d, fy_new_d, N
        update_velocityvx_d, vy_d, fx_new_d, fy_new_d, w_d, cfg.dt, N

        fx_d, fx_new_d = fx_new_d, fx_d
        fy_d, fy_new_d = fy_new_d, fy_d

    t1 = time.time()

    print("Done")
    print("Time:", t1 - t0)


if __name__ == "__main__":
    main()
