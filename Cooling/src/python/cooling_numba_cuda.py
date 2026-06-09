#!/usr/bin/env python3

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import h5py
import numpy as np
from numba import cuda


# ============================================================
# OPTIONAL CUPY SUPPORT
# ============================================================

try:
    import cupy as cp
    HAVE_CUPY = True
except ImportError:
    cp = None
    HAVE_CUPY = False


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
# VALIDATION HELPERS
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


def is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def validate_block_sizes(block2d: Tuple[int, int], block1d: int) -> None:
    bx, by = block2d

    if bx <= 0 or by <= 0 or block1d <= 0:
        raise ValueError("CUDA block sizes must be > 0")

    if bx * by > 1024:
        raise ValueError("2D CUDA block size must have at most 1024 threads")

    if block1d > 1024:
        raise ValueError("1D CUDA block size must be <= 1024")

    if not is_power_of_two(block1d):
        raise ValueError("block-1d must be a power of two for the reduction kernel")


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
    # Supported input endings:
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
    x0 = cfg.Sreal
    y0 = cfg.Simag
    dx = cfg.Dreal / float(cfg.nx - 1)
    dy = cfg.Dimag / float(cfg.ny - 1)

    return x0, y0, dx, dy


def analytical_field(x: float, y: float) -> float:
    return (x * x * x + y * y * y) / 6.0


def compute_discrepancy(cfg: Config) -> float:
    if not cfg.measured:
        return 0.0

    acc = np.longdouble(0.0)

    for m in cfg.measured:
        acc += np.longdouble(m.v - analytical_field(m.x, m.y))

    return float(acc / np.longdouble(len(cfg.measured)))


def build_cooling_coeffs(dx: float, dy: float, dd: float = 100.0):
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
# CUDA KERNELS
# ============================================================

@cuda.jit
def compute_weight_kernel(weight, nx, ny, x0, y0, dx, dy, max_iters):
    i, j = cuda.grid(2)

    if i >= nx or j >= ny:
        return

    p = i + j * nx

    x = x0 + dx * i
    y = y0 + dy * j

    za = 0.0
    zb = 0.0
    it = 0

    while it < max_iters:
        if za * za + zb * zb > 4.0:
            break

        tmp = za * za - zb * zb + x
        zb = 2.0 * za * zb + y
        za = tmp

        it += 1

    weight[p] = it


@cuda.jit
def initialize_field_kernel(u, weight, nx, ny, x0, y0, dx, dy, discrepancy, wmin, wmax):
    i, j = cuda.grid(2)

    if i >= nx or j >= ny:
        return

    p = i + j * nx

    x = x0 + dx * i
    y = y0 + dy * j

    denom = float(wmax - wmin) if wmax > wmin else 1.0

    F = (x * x * x + y * y * y) / 6.0
    wnorm = (weight[p] - wmin) / denom

    u[p] = 293.16 + 80.0 * (discrepancy + F) * wnorm


@cuda.jit
def update_interior_kernel(u1, u2, nx, ny, dgx, dgy, CX, CY):
    i, j = cuda.grid(2)

    if i < 1 or i >= nx - 1 or j < 1 or j >= ny - 1:
        return

    p = i + j * nx

    u2[p] = (
        CX * (
            u1[p - 1]
            + u1[p + 1]
            + (dgx + 0.5 / CX) * u1[p]
        )
        + CY * (
            u1[p - nx]
            + u1[p + nx]
            + (dgy + 0.5 / CY) * u1[p]
        )
    )


@cuda.jit
def apply_boundary_all_kernel(u, nx, ny):
    """
    Applies the same boundary behavior as the original two-kernel sequence:

      1) left/right edges:
         u[j,0]     = u[j,1]
         u[j,nx-1]  = u[j,nx-2]

      2) top/bottom edges after that.

    Corners are written explicitly so the result does not rely on
    cross-block synchronization.
    """
    idx = cuda.grid(1)

    # Left/right edges, excluding corners.
    if idx < ny - 2:
        j = idx + 1
        row = j * nx

        u[row] = u[row + 1]
        u[row + nx - 1] = u[row + nx - 2]

    # Top/bottom edges, including explicit corner behavior.
    if idx < nx:
        top = idx
        bottom = (ny - 1) * nx + idx

        if idx == 0:
            u[top] = u[nx + 1]
            u[bottom] = u[(ny - 2) * nx + 1]
        elif idx == nx - 1:
            u[top] = u[nx + nx - 2]
            u[bottom] = u[(ny - 2) * nx + nx - 2]
        else:
            u[top] = u[nx + idx]
            u[bottom] = u[(ny - 2) * nx + idx]


# ============================================================
# GPU STATS REDUCTION
# ============================================================

def make_stats_kernel(block_size: int):
    @cuda.jit
    def stats_kernel(u, n, block_min, block_max, block_sum, block_sum2):
        s_min = cuda.shared.array(block_size, dtype=np.float64)
        s_max = cuda.shared.array(block_size, dtype=np.float64)
        s_sum = cuda.shared.array(block_size, dtype=np.float64)
        s_sum2 = cuda.shared.array(block_size, dtype=np.float64)

        tid = cuda.threadIdx.x
        gid = cuda.grid(1)

        if gid < n:
            val = u[gid]
            s_min[tid] = val
            s_max[tid] = val
            s_sum[tid] = val
            s_sum2[tid] = val * val
        else:
            s_min[tid] = 1.0e300
            s_max[tid] = -1.0e300
            s_sum[tid] = 0.0
            s_sum2[tid] = 0.0

        cuda.syncthreads()

        s = cuda.blockDim.x // 2

        while s > 0:
            if tid < s:
                other_min = s_min[tid + s]
                other_max = s_max[tid + s]

                if other_min < s_min[tid]:
                    s_min[tid] = other_min

                if other_max > s_max[tid]:
                    s_max[tid] = other_max

                s_sum[tid] += s_sum[tid + s]
                s_sum2[tid] += s_sum2[tid + s]

            cuda.syncthreads()
            s //= 2

        if tid == 0:
            b = cuda.blockIdx.x

            block_min[b] = s_min[0]
            block_max[b] = s_max[0]
            block_sum[b] = s_sum[0]
            block_sum2[b] = s_sum2[0]

    return stats_kernel


class GpuStatsReducer:
    def __init__(self, n: int, block_size: int):
        if n <= 0:
            raise ValueError("GpuStatsReducer: n must be > 0")

        if not is_power_of_two(block_size):
            raise ValueError("GpuStatsReducer: block_size must be a power of two")

        if block_size > 1024:
            raise ValueError("GpuStatsReducer: block_size must be <= 1024")

        self.n = n
        self.block_size = block_size
        self.grid_size = (n + block_size - 1) // block_size

        self.kernel = make_stats_kernel(block_size)

        self.d_min = cuda.device_array(self.grid_size, dtype=np.float64)
        self.d_max = cuda.device_array(self.grid_size, dtype=np.float64)
        self.d_sum = cuda.device_array(self.grid_size, dtype=np.float64)
        self.d_sum2 = cuda.device_array(self.grid_size, dtype=np.float64)

        self.h_min = cuda.pinned_array(self.grid_size, dtype=np.float64)
        self.h_max = cuda.pinned_array(self.grid_size, dtype=np.float64)
        self.h_sum = cuda.pinned_array(self.grid_size, dtype=np.float64)
        self.h_sum2 = cuda.pinned_array(self.grid_size, dtype=np.float64)

    def queue(self, d_u, stream) -> None:
        self.kernel[self.grid_size, self.block_size, stream](
            d_u,
            self.n,
            self.d_min,
            self.d_max,
            self.d_sum,
            self.d_sum2,
        )

        self.d_min.copy_to_host(self.h_min, stream=stream)
        self.d_max.copy_to_host(self.h_max, stream=stream)
        self.d_sum.copy_to_host(self.h_sum, stream=stream)
        self.d_sum2.copy_to_host(self.h_sum2, stream=stream)

    def finish_host_reduction(self) -> Tuple[float, float, float, float]:
        mn = float(np.min(self.h_min))
        mx = float(np.max(self.h_max))

        total_sum = float(np.sum(self.h_sum))
        total_sum2 = float(np.sum(self.h_sum2))

        mean = total_sum / float(self.n)
        var = max(0.0, total_sum2 / float(self.n) - mean * mean)
        std = math.sqrt(var)

        return mn, mean, mx, std


# ============================================================
# KERNEL LAUNCH HELPERS
# ============================================================

def grid2d_for(nx: int, ny: int, block2d: Tuple[int, int]) -> Tuple[int, int]:
    return (
        (nx + block2d[0] - 1) // block2d[0],
        (ny + block2d[1] - 1) // block2d[1],
    )


def launch_compute_weight(
    d_weight,
    nx: int,
    ny: int,
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    max_iters: int,
    block2d: Tuple[int, int],
    stream,
) -> None:
    compute_weight_kernel[grid2d_for(nx, ny, block2d), block2d, stream](
        d_weight,
        nx,
        ny,
        x0,
        y0,
        dx,
        dy,
        max_iters,
    )


def launch_initialize_field(
    d_u,
    d_weight,
    nx: int,
    ny: int,
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    discrepancy: float,
    wmin: int,
    wmax: int,
    block2d: Tuple[int, int],
    stream,
) -> None:
    initialize_field_kernel[grid2d_for(nx, ny, block2d), block2d, stream](
        d_u,
        d_weight,
        nx,
        ny,
        x0,
        y0,
        dx,
        dy,
        discrepancy,
        wmin,
        wmax,
    )


def launch_update_field(
    d_u1,
    d_u2,
    nx: int,
    ny: int,
    dgx: float,
    dgy: float,
    CX: float,
    CY: float,
    block2d: Tuple[int, int],
    block1d: int,
    stream,
) -> None:
    update_interior_kernel[grid2d_for(nx, ny, block2d), block2d, stream](
        d_u1,
        d_u2,
        nx,
        ny,
        dgx,
        dgy,
        CX,
        CY,
    )

    n_boundary_work = max(nx, ny - 2)
    grid1d = (n_boundary_work + block1d - 1) // block1d

    apply_boundary_all_kernel[grid1d, block1d, stream](d_u2, nx, ny)


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

    def write(self, step_number: int, field_1d: np.ndarray) -> None:
        if self.closed:
            raise RuntimeError("H5Writer: write() called after close()")

        if field_1d.size != self.nx * self.ny:
            raise RuntimeError("H5Writer: field size mismatch")

        if self.frame >= self.capacity:
            self.capacity += self.batch
            self._extend(self.capacity)

        self.field[self.frame, :, :] = field_1d.reshape(self.ny, self.nx)
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
# CSV STATS WRITER
# ============================================================

def write_stats_header(f) -> None:
    f.write("Step;Min;Mean;Max;Std_dev\n")


def write_stats_line(f, step: int, stats: Tuple[float, float, float, float]) -> None:
    mn, mean, mx, std = stats
    f.write(f"{step};{mn:.15g};{mean:.15g};{mx:.15g};{std:.15g}\n")


# ============================================================
# TIMING HELPERS
# ============================================================

def elapsed_event_seconds(start_evt, stop_evt) -> float:
    return cuda.event_elapsed_time(start_evt, stop_evt) / 1000.0


def get_weight_minmax(
    d_weight,
    n: int,
    pinned_fallback: Optional[np.ndarray],
) -> Tuple[int, int, float, str]:
    """
    Returns:
      wmin, wmax, wall_time_seconds, method_name
    """
    t0 = time.perf_counter()

    if HAVE_CUPY:
        cp_weight = cp.asarray(d_weight)
        wmin = int(cp_weight.min().get())
        wmax = int(cp_weight.max().get())
        method = "CuPy device reduction"
    else:
        if pinned_fallback is None or pinned_fallback.size != n:
            pinned_fallback = cuda.pinned_array(n, dtype=np.int32)

        d_weight.copy_to_host(pinned_fallback)

        wmin = int(np.min(pinned_fallback))
        wmax = int(np.max(pinned_fallback))
        method = "host fallback reduction"

    t1 = time.perf_counter()

    return wmin, wmax, t1 - t0, method


# ============================================================
# HOST-SIDE DRIVER
# ============================================================

def run_simulation(
    cfg: Config,
    h5_file: str,
    csv_file: str,
    block2d: Tuple[int, int] = (16, 16),
    block1d: int = 256,
) -> None:
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available")

    validate_block_sizes(block2d, block1d)

    if cfg.hasOutputEvery:
        if cfg.outputEvery is None:
            raise ValueError("Internal error: hasOutputEvery is true but outputEvery is None")
        validate_output_every(cfg.outputEvery)

    n = validated_grid_size(cfg.nx, cfg.ny)

    x0, y0, dx, dy = build_domain_params(cfg)
    _, _, _, dgx, dgy, CX, CY = build_cooling_coeffs(dx, dy, 100.0)
    discrepancy = compute_discrepancy(cfg)

    stream = cuda.stream()

    d_weight = cuda.device_array(n, dtype=np.int32)
    d_u_curr = cuda.device_array(n, dtype=np.float64)
    d_u_next = cuda.device_array(n, dtype=np.float64)

    u_host_pinned = cuda.pinned_array(n, dtype=np.float64)
    weight_host_pinned = None if HAVE_CUPY else cuda.pinned_array(n, dtype=np.int32)

    stats_reducer = GpuStatsReducer(n=n, block_size=block1d)

    # --------------------------------------------------------
    # Warm-up / JIT compilation
    # --------------------------------------------------------
    print("Warming up CUDA compiler...")

    launch_compute_weight(
        d_weight=d_weight,
        nx=3,
        ny=3,
        x0=x0,
        y0=y0,
        dx=dx,
        dy=dy,
        max_iters=2,
        block2d=block2d,
        stream=stream,
    )

    launch_initialize_field(
        d_u=d_u_curr,
        d_weight=d_weight,
        nx=3,
        ny=3,
        x0=x0,
        y0=y0,
        dx=dx,
        dy=dy,
        discrepancy=0.0,
        wmin=0,
        wmax=1,
        block2d=block2d,
        stream=stream,
    )

    launch_update_field(
        d_u1=d_u_curr,
        d_u2=d_u_next,
        nx=3,
        ny=3,
        dgx=dgx,
        dgy=dgy,
        CX=CX,
        CY=CY,
        block2d=block2d,
        block1d=block1d,
        stream=stream,
    )

    stream.synchronize()

    # --------------------------------------------------------
    # Main timed execution
    # --------------------------------------------------------
    total_wall_t0 = time.perf_counter()

    # -----------------------------
    # Weight field
    # -----------------------------
    weight_start = cuda.event()
    weight_stop = cuda.event()

    weight_start.record(stream)

    launch_compute_weight(
        d_weight=d_weight,
        nx=cfg.nx,
        ny=cfg.ny,
        x0=x0,
        y0=y0,
        dx=dx,
        dy=dy,
        max_iters=cfg.maxIters,
        block2d=block2d,
        stream=stream,
    )

    weight_stop.record(stream)
    weight_stop.synchronize()

    weight_gpu_s = elapsed_event_seconds(weight_start, weight_stop)

    wmin, wmax, minmax_wall_s, minmax_method = get_weight_minmax(
        d_weight=d_weight,
        n=n,
        pinned_fallback=weight_host_pinned,
    )

    # -----------------------------
    # Initialization field
    # -----------------------------
    init_start = cuda.event()
    init_stop = cuda.event()

    init_start.record(stream)

    launch_initialize_field(
        d_u=d_u_curr,
        d_weight=d_weight,
        nx=cfg.nx,
        ny=cfg.ny,
        x0=x0,
        y0=y0,
        dx=dx,
        dy=dy,
        discrepancy=discrepancy,
        wmin=wmin,
        wmax=wmax,
        block2d=block2d,
        stream=stream,
    )

    init_stop.record(stream)
    init_stop.synchronize()

    init_gpu_s = elapsed_event_seconds(init_start, init_stop)

    # -----------------------------
    # Output/timestep pipeline
    # -----------------------------
    pipeline_gpu_ms = 0.0
    output_frames = 0
    last_written_step: Optional[int] = None

    segment_start = cuda.event()
    segment_stop = cuda.event()
    segment_running = False

    def start_gpu_segment() -> None:
        nonlocal segment_running

        if not segment_running:
            segment_start.record(stream)
            segment_running = True

    def stop_gpu_segment_and_sync() -> float:
        nonlocal segment_running

        if not segment_running:
            return 0.0

        segment_stop.record(stream)
        segment_stop.synchronize()

        segment_running = False

        return cuda.event_elapsed_time(segment_start, segment_stop)

    loop_wall_t0 = time.perf_counter()

    with open(csv_file, "w", encoding="utf-8") as csvf:
        write_stats_header(csvf)

        with H5Writer(h5_file, cfg.nx, cfg.ny, batch=32) as writer:

            def write_output_frame(step: int) -> None:
                nonlocal pipeline_gpu_ms
                nonlocal output_frames
                nonlocal last_written_step

                if last_written_step == step:
                    return

                # Queue stats reduction and device-to-host copy in the same stream.
                stats_reducer.queue(d_u_curr, stream=stream)
                d_u_curr.copy_to_host(u_host_pinned, stream=stream)

                # Stop and synchronize the active GPU segment so host-side
                # reduction/HDF5/CSV see valid data.
                pipeline_gpu_ms += stop_gpu_segment_and_sync()

                stats = stats_reducer.finish_host_reduction()
                write_stats_line(csvf, step, stats)
                writer.write(step, u_host_pinned)

                last_written_step = step
                output_frames += 1

            # If outputEvery was explicitly specified, write initial condition.
            # If absent, do not output step 0; only final state is written later.
            start_gpu_segment()

            if cfg.hasOutputEvery:
                write_output_frame(0)

                # Start a fresh timing segment for subsequent dynamics.
                start_gpu_segment()

            for step in range(1, cfg.steps + 1):
                launch_update_field(
                    d_u1=d_u_curr,
                    d_u2=d_u_next,
                    nx=cfg.nx,
                    ny=cfg.ny,
                    dgx=dgx,
                    dgy=dgy,
                    CX=CX,
                    CY=CY,
                    block2d=block2d,
                    block1d=block1d,
                    stream=stream,
                )

                d_u_curr, d_u_next = d_u_next, d_u_curr

                # Periodic output only when outputEvery was explicitly specified.
                if cfg.hasOutputEvery:
                    assert cfg.outputEvery is not None

                    if (step % cfg.outputEvery) == 0:
                        write_output_frame(step)

                        # Continue timing subsequent GPU work, unless this was
                        # the last step and the final write would be duplicate.
                        if step < cfg.steps:
                            start_gpu_segment()

            # Always write final state.
            # If it was already written by periodic output, this skips it.
            write_output_frame(cfg.steps)

            # If no output occurred inside the loop before final output, the final
            # write already stopped/synchronized the segment. If the final write was
            # skipped as duplicate, there should be no active segment unless one was
            # intentionally restarted, but this guard is harmless and avoids leaving
            # pending work unsynchronized.
            pipeline_gpu_ms += stop_gpu_segment_and_sync()

    loop_wall_s = time.perf_counter() - loop_wall_t0
    total_wall_s = time.perf_counter() - total_wall_t0

    pipeline_gpu_s = pipeline_gpu_ms / 1000.0

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
    print(f"CUDA block2d:                 {block2d}")
    print(f"CUDA block1d:                 {block1d}")
    print(f"CuPy available:               {HAVE_CUPY}")
    print(f"Weight min/max method:        {minmax_method}")
    print(f"Weight kernel GPU time:       {weight_gpu_s:.6f} s")
    print(f"Weight min/max wall time:     {minmax_wall_s:.6f} s")
    print(f"Init field GPU time:          {init_gpu_s:.6f} s")
    print(f"Pipeline GPU time:            {pipeline_gpu_s:.6f} s")
    print(f"Loop wall time incl. I/O:     {loop_wall_s:.6f} s")
    print(f"Total wall time:              {total_wall_s:.6f} s")

    if cfg.steps > 0 and pipeline_gpu_s > 0.0:
        print(f"GPU pipeline performance:     {updates / pipeline_gpu_s / 1e9:.6f} GLUP/s")

    if cfg.steps > 0 and loop_wall_s > 0.0:
        print(f"End-to-end loop performance:  {updates / loop_wall_s / 1e9:.6f} GLUP/s")

    print(f"Mean discrepancy:             {discrepancy:.15g}")
    print("Simulation completed successfully.")


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Numba CUDA cooling solver with robust timing, pinned buffers, "
            "optional CuPy reductions, and optional output cadence"
        )
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
        "--block-x",
        type=int,
        default=16,
        help="CUDA block size in x for 2D kernels",
    )

    parser.add_argument(
        "--block-y",
        type=int,
        default=16,
        help="CUDA block size in y for 2D kernels",
    )

    parser.add_argument(
        "--block-1d",
        type=int,
        default=256,
        help="CUDA block size for 1D kernels and reductions. Must be a power of two.",
    )

    args = parser.parse_args()

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

    block2d = (args.block_x, args.block_y)
    block1d = args.block_1d

    validate_block_sizes(block2d, block1d)

    print(f"Input file:                   {args.input}")
    print(f"HDF5 output:                  {args.h5}")
    print(f"CSV output:                   {args.csv}")

    run_simulation(
        cfg=cfg,
        h5_file=args.h5,
        csv_file=args.csv,
        block2d=block2d,
        block1d=block1d,
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"CRITICAL ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
