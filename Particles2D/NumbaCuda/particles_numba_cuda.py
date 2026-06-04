#!/usr/bin/env python3

import sys
import math
import time
from dataclasses import dataclass, field
from typing import Iterator, Tuple

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
# GPU KERNELS
# ============================================================

@cuda.jit
def mandelbrot_kernel(values, nx, ny, xs, xe, ys, ye, max_iter):
    i, j = cuda.grid(2)

    if i < nx and j < ny:
        dx = (xe - xs) / (nx - 1)
        dy = (ye - ys) / (ny - 1)

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
def compute_forces_tiled_kernel(x, y, w, fx, fy, N):
    sh_x = cuda.shared.array(shape=BLOCK_SIZE, dtype=float64)
    sh_y = cuda.shared.array(shape=BLOCK_SIZE, dtype=float64)
    sh_w = cuda.shared.array(shape=BLOCK_SIZE, dtype=float64)

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
def half_kick_drift_kernel(x, y, vx, vy, w, fx, fy, dt, N):
    i = cuda.grid(1)
    if i >= N:
        return

    invm = 1.0 / w[i]

    # Match CPU/CUDA reference:
    # 1) v += 0.5 * a_old * dt
    vx[i] += 0.5 * fx[i] * invm * dt
    vy[i] += 0.5 * fy[i] * invm * dt

    # 2) x += v * dt
    x[i] += vx[i] * dt
    y[i] += vy[i] * dt


@cuda.jit
def half_kick_kernel(vx, vy, w, fx_new, fy_new, dt, N):
    i = cuda.grid(1)
    if i >= N:
        return

    invm = 1.0 / w[i]

    # 3) v += 0.5 * a_new * dt
    vx[i] += 0.5 * fx_new[i] * invm * dt
    vy[i] += 0.5 * fy_new[i] * invm * dt


@cuda.jit
def zero_int64_kernel(arr, n):
    i = cuda.grid(1)
    if i < n:
        arr[i] = 0


@cuda.jit
def build_screen_kernel(screen, x, y, w, nx, ny, xs, xe, ys, ye, wmin, wr, N):
    n = cuda.grid(1)
    if n >= N:
        return

    invdx = (nx - 1) / (xe - xs) if xe != xs else 0.0
    invdy = (ny - 1) / (ye - ys) if ye != ys else 0.0

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

            cuda.atomic.add(screen, row + jx, wp)


# ============================================================
# HIGH-LEVEL HELPERS
# ============================================================

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

    P.vx.fill(0.0)
    P.vy.fill(0.0)

    return P


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
# CUDA WARM-UP
# ============================================================

def warmup_cuda() -> None:
    # tiny Mandelbrot
    d_vals = cuda.device_array(4, dtype=np.uint64)
    mandelbrot_kernel[(1, 1), (2, 2)](d_vals, 2, 2, -2.0, 1.0, -1.0, 1.0, 4)

    # tiny particle arrays
    x = np.array([0.0, 1.0], dtype=np.float64)
    y = np.array([0.0, 0.5], dtype=np.float64)
    w = np.array([1.0, 2.0], dtype=np.float64)
    vx = np.zeros(2, dtype=np.float64)
    vy = np.zeros(2, dtype=np.float64)
    fx = np.zeros(2, dtype=np.float64)
    fy = np.zeros(2, dtype=np.float64)
    fx_new = np.zeros(2, dtype=np.float64)
    fy_new = np.zeros(2, dtype=np.float64)

    x_d = cuda.to_device(x)
    y_d = cuda.to_device(y)
    w_d = cuda.to_device(w)
    vx_d = cuda.to_device(vx)
    vy_d = cuda.to_device(vy)
    fx_d = cuda.to_device(fx)
    fy_d = cuda.to_device(fy)
    fx_new_d = cuda.to_device(fx_new)
    fy_new_d = cuda.to_device(fy_new)

    compute_forces_tiled_kernel[1, BLOCK_SIZE](x_d, y_d, w_d, fx_d, fy_d, 2)
    half_kick_drift_kernel[1, BLOCK_SIZE](x_d, y_d, vx_d, vy_d, w_d, fx_d, fy_d, 1e-3, 2)
    compute_forces_tiled_kernel[1, BLOCK_SIZE](x_d, y_d, w_d, fx_new_d, fy_new_d, 2)
    half_kick_kernel[1, BLOCK_SIZE](vx_d, vy_d, w_d, fx_new_d, fy_new_d, 1e-3, 2)

    screen_d = cuda.device_array(4, dtype=np.int64)
    zero_int64_kernel[1, BLOCK_SIZE](screen_d, 4)
    build_screen_kernel[1, BLOCK_SIZE](screen_d, x_d, y_d, w_d, 2, 2, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 2)

    cuda.synchronize()


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    try:
        if not cuda.is_available():
            raise RuntimeError("CUDA is not available")

        input_file = sys.argv[1] if len(sys.argv) > 1 else "Particles.inp"

        cfg, gen, screen = read_input(input_file)

        # Warm-up so timing excludes most JIT overhead
        warmup_cuda()

        # ----------------------------------------------------
        # 1) Generating field on GPU
        # ----------------------------------------------------
        d_vals = cuda.device_array(gen.nx * gen.ny, dtype=np.uint64)

        block2d = (16, 16)
        grid2d = (
            (gen.nx + block2d[0] - 1) // block2d[0],
            (gen.ny + block2d[1] - 1) // block2d[1],
        )

        mandelbrot_kernel[grid2d, block2d](
            d_vals,
            gen.nx, gen.ny,
            gen.xs, gen.xe,
            gen.ys, gen.ye,
            cfg.maxIters
        )
        cuda.synchronize()

        gen.values = d_vals.copy_to_host()

        # ----------------------------------------------------
        # 2) Particle generation on host (same logic as CPU)
        # ----------------------------------------------------
        P = generate_particles(gen, screen)

        # ----------------------------------------------------
        # 3) Copy particles to GPU
        # ----------------------------------------------------
        x_d = cuda.to_device(P.x)
        y_d = cuda.to_device(P.y)
        vx_d = cuda.to_device(P.vx)
        vy_d = cuda.to_device(P.vy)
        w_d = cuda.to_device(P.w)

        fx_d = cuda.device_array(P.n, dtype=np.float64)
        fy_d = cuda.device_array(P.n, dtype=np.float64)
        fx_new_d = cuda.device_array(P.n, dtype=np.float64)
        fy_new_d = cuda.device_array(P.n, dtype=np.float64)

        # Screen buffer on device:
        # use int64 for broad atomic-add compatibility, cast to uint64 on host
        screen_d = cuda.device_array(screen.nx * screen.ny, dtype=np.int64)

        threads = BLOCK_SIZE
        blocks_particles = (P.n + threads - 1) // threads
        blocks_screen = (screen.nx * screen.ny + threads - 1) // threads

        # Initial forces before timing (matches CPU logic)
        compute_forces_tiled_kernel[blocks_particles, threads](x_d, y_d, w_d, fx_d, fy_d, P.n)
        cuda.synchronize()

        wmin = float(np.min(P.w))
        wmax = float(np.max(P.w))
        wr = max(wmax - wmin, 1.0)

        with H5StreamWriter("particles.h5", P.n, screen.nx, screen.ny) as h5:
            t0 = time.perf_counter()

            for step in range(cfg.maxSteps):
                if step % cfg.outputEvery == 0:
                    zero_int64_kernel[blocks_screen, threads](screen_d, screen.nx * screen.ny)
                    build_screen_kernel[blocks_particles, threads](
                        screen_d,
                        x_d, y_d, w_d,
                        screen.nx, screen.ny,
                        screen.xs, screen.xe,
                        screen.ys, screen.ye,
                        wmin, wr, P.n
                    )
                    cuda.synchronize()

                    # copy output state to host for HDF5 writing
                    x_d.copy_to_host(P.x)
                    y_d.copy_to_host(P.y)
                    vx_d.copy_to_host(P.vx)
                    vy_d.copy_to_host(P.vy)

                    screen_host_i64 = screen_d.copy_to_host()
                    screen.values[:] = screen_host_i64.astype(np.uint64, copy=False)

                    h5.write_frame(P, screen)

                # Velocity-Verlet on GPU
                half_kick_drift_kernel[blocks_particles, threads](
                    x_d, y_d, vx_d, vy_d, w_d, fx_d, fy_d, cfg.dt, P.n
                )
                compute_forces_tiled_kernel[blocks_particles, threads](
                    x_d, y_d, w_d, fx_new_d, fy_new_d, P.n
                )
                half_kick_kernel[blocks_particles, threads](
                    vx_d, vy_d, w_d, fx_new_d, fy_new_d, cfg.dt, P.n
                )

                # swap force buffers
                fx_d, fx_new_d = fx_new_d, fx_d
                fy_d, fy_new_d = fy_new_d, fy_d

            cuda.synchronize()
            t1 = time.perf_counter()

            # Final catch-up frame (same as CPU code)
            if (cfg.maxSteps - 1) % cfg.outputEvery != 0:
                zero_int64_kernel[blocks_screen, threads](screen_d, screen.nx * screen.ny)
                build_screen_kernel[blocks_particles, threads](
                    screen_d,
                    x_d, y_d, w_d,
                    screen.nx, screen.ny,
                    screen.xs, screen.xe,
                    screen.ys, screen.ye,
                    wmin, wr, P.n
                )
                cuda.synchronize()

                x_d.copy_to_host(P.x)
                y_d.copy_to_host(P.y)
                vx_d.copy_to_host(P.vx)
                vy_d.copy_to_host(P.vy)

                screen_host_i64 = screen_d.copy_to_host()
                screen.values[:] = screen_host_i64.astype(np.uint64, copy=False)

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
