#!/usr/bin/env python3
"""
Cooling / field-evolution solver - Numba CUDA reference implementation.

This version is aligned with the Python/NumPy teaching baseline and the C++
serial baseline.

GPU-accelerated parts:

  1) temperature initialization
  2) stencil update + boundary kernels

Host-side by design:

  - fractal weight field initialization, using the same scalar algorithm as the
    strict Python/NumPy baseline and the C++ serial baseline
  - weight min/max reduction
  - HDF5 output
  - CSV output
  - snapshot statistics and checksum

Important design choices:

  - The fractal/Mandelbrot-style weight initialization is intentionally computed
    on the host. This avoids CPU/GPU boundary-classification differences and
    makes this Numba CUDA implementation produce the same initial field as the
    NumPy baseline.
  - HDF5 output is optional and disabled by default.
  - Statistics are computed on the host after copying requested snapshots.
  - outputEvery = 0 means final frame/statistics only.
  - outputEvery > 0 means step 0, every outputEvery steps, and final step.
  - CSV statistics include L2_norm and a deterministic checksum.

Official performance mode:

  python ./path/to/cooling_numba_cuda.py input/Cooling.in none output/Cooling_numba_cuda.csv 0

or with explicit CUDA device selection:

  python ./path/to/cooling_numba_cuda.py --device 0 input/Cooling.in none output/Cooling_numba_cuda.csv 0

Command line:

  python ./path/to/cooling_numba_cuda.py [options] [inputFile] [h5File|none] [csvFile] [outputEvery]

Examples:

  python ./path/to/cooling_numba_cuda.py --device 0           input/Cooling.in none                         output/Cooling_numba_cuda.csv  0
  python ./path/to/cooling_numba_cuda.py --device 0           input/Cooling.in output/Cooling_numba_cuda.h5 output/Cooling_numba_cuda.csv 50
  python ./path/to/cooling_numba_cuda.py --device 0 --no-hdf5 input/Cooling.in none                         output/Cooling_numba_cuda.csv  0
"""

from __future__ import annotations

import argparse
import math
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numba
import numpy as np
from numba import cuda, njit, prange

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class SamplePoint:
    x: float
    y: float
    value: float


@dataclass
class SimulationConfig:
    grid_width: int
    grid_height: int

    domain_start_x: float
    domain_start_y: float
    domain_width: float
    domain_height: float

    max_fractal_iterations: int
    time_steps: int
    output_every: int

    measured_points: List[SamplePoint]


@dataclass
class GridMapping:
    x0: float
    y0: float
    dx: float
    dy: float


@dataclass
class UpdateCoefficients:
    damping: float
    step_x: float
    step_y: float
    laplace_x: float
    laplace_y: float
    coeff_x: float
    coeff_y: float
    center_x: float
    center_y: float


@dataclass
class FieldStatistics:
    min_value: float
    mean_value: float
    max_value: float
    std_dev: float
    l2_norm: float
    checksum: float


@dataclass
class RunTimings:
    warmup_time: float = 0.0
    weight_host_time: float = 0.0
    weight_h2d_time: float = 0.0
    weight_range_time: float = 0.0
    init_kernel_time: float = 0.0
    pure_dynamics_gpu_time: float = 0.0
    device_to_host_copy_time: float = 0.0
    statistics_time: float = 0.0
    csv_time: float = 0.0
    hdf5_time: float = 0.0
    loop_wall_time: float = 0.0
    total_wall_time: float = 0.0


# ============================================================
# VALIDATION / UTILITY
# ============================================================

_INTEGER_RE = re.compile(r"^[+-]?[0-9]+$")


def parse_strict_int(token: str) -> int:
    """
    Strict integer parser aligned with the C++ baseline behavior.
    Rejects strings such as '1.0' or '1_000'.
    """
    if not _INTEGER_RE.match(token):
        raise RuntimeError(f"Malformed input: invalid integer token '{token}'")
    return int(token)


def checked_grid_size(grid_width: int, grid_height: int) -> int:
    if grid_width <= 0 or grid_height <= 0:
        raise ValueError("Grid dimensions must be > 0")

    total_cells = grid_width * grid_height

    if total_cells <= 0:
        raise ValueError("Invalid total grid size")

    return total_cells


def validate_output_every(output_every: int) -> None:
    if output_every < 0:
        raise ValueError("outputEvery must be >= 0; use 0 for final-only output")


def validate_block_sizes(block2d: Tuple[int, int], block1d: int) -> None:
    block_x, block_y = block2d

    if block_x <= 0 or block_y <= 0:
        raise ValueError("CUDA 2D block dimensions must be > 0")

    if block1d <= 0:
        raise ValueError("CUDA 1D block size must be > 0")

    if block_x > 1024 or block_y > 1024 or block1d > 1024:
        raise ValueError("CUDA block dimensions must not exceed 1024")

    threads_2d = int(block_x) * int(block_y)

    if threads_2d > 1024:
        raise ValueError("2D CUDA block size must have at most 1024 threads")


def is_no_hdf5_token(text: str) -> bool:
    return text in {"none", "NONE", "-", "--no-hdf5"}


def ensure_parent_directory_exists(file_name: str) -> None:
    if not file_name or is_no_hdf5_token(file_name):
        return

    parent = Path(file_name).expanduser().parent

    if str(parent) in {"", "."}:
        return

    parent.mkdir(parents=True, exist_ok=True)

    if not parent.is_dir():
        raise RuntimeError(f"Output path parent exists but is not a directory: {parent}")


def cuda_device_name() -> str:
    device = cuda.get_current_device()
    name = device.name
    return name.decode("utf-8", errors="replace") if isinstance(name, bytes) else str(name)


def build_output_schedule(final_step: int, output_every: int) -> List[int]:
    validate_output_every(output_every)

    steps: List[int] = []

    if output_every > 0:
        steps.append(0)

        step = output_every
        while step < final_step:
            steps.append(step)
            step += output_every

    if not steps or steps[-1] != final_step:
        steps.append(final_step)

    return steps


def should_write_step(step: int, final_step: int, output_every: int) -> bool:
    if step == final_step:
        return True

    if output_every <= 0:
        return False

    if step == 0:
        return True

    return (step % output_every) == 0


# ============================================================
# INPUT PARSER
# ============================================================

def read_configuration_file(file_name: str) -> SimulationConfig:
    tokens: List[str] = []

    try:
        with open(file_name, "r", encoding="utf-8") as f:
            for line in f:
                line = line.split("#", 1)[0]
                tokens.extend(line.split())
    except OSError as e:
        raise RuntimeError(f"Cannot open input file: {file_name}") from e

    if not tokens:
        raise RuntimeError(f"Input file contains no numeric tokens: {file_name}")

    pos = 0

    def next_int() -> int:
        nonlocal pos

        if pos >= len(tokens):
            raise RuntimeError("Malformed input: missing integer token")

        token = tokens[pos]
        pos += 1

        return parse_strict_int(token)

    def next_float() -> float:
        nonlocal pos

        if pos >= len(tokens):
            raise RuntimeError("Malformed input: missing floating-point token")

        token = tokens[pos]
        pos += 1

        try:
            value = float(token)
        except Exception as e:
            raise RuntimeError(f"Malformed input: invalid floating-point token '{token}'") from e

        if not math.isfinite(value):
            raise RuntimeError(f"Malformed input: non-finite floating-point token '{token}'")

        return value

    grid_width = next_int()
    grid_height = next_int()

    if grid_width < 3 or grid_height < 3:
        raise RuntimeError("Grid dimensions must be at least 3 x 3")

    measured_count = next_int()

    if measured_count < 0:
        raise RuntimeError("Number of measured points cannot be negative")

    measured_points: List[SamplePoint] = []

    for _ in range(measured_count):
        measured_points.append(
            SamplePoint(
                x=next_float(),
                y=next_float(),
                value=next_float(),
            )
        )

    domain_start_x = next_float()
    domain_start_y = next_float()
    domain_width = next_float()
    domain_height = next_float()
    max_fractal_iterations = next_int()
    time_steps = next_int()

    if max_fractal_iterations <= 0:
        raise RuntimeError("maxFractalIterations must be > 0")

    if time_steps < 0:
        raise RuntimeError("timeSteps must be >= 0")

    if domain_width <= 0.0 or domain_height <= 0.0:
        raise RuntimeError("domainWidth and domainHeight must be > 0")

    output_every = 0

    if pos < len(tokens):
        output_every = next_int()
        validate_output_every(output_every)

    if pos != len(tokens):
        raise RuntimeError("Malformed input: unexpected extra tokens at end of file")

    return SimulationConfig(
        grid_width=grid_width,
        grid_height=grid_height,
        domain_start_x=domain_start_x,
        domain_start_y=domain_start_y,
        domain_width=domain_width,
        domain_height=domain_height,
        max_fractal_iterations=max_fractal_iterations,
        time_steps=time_steps,
        output_every=output_every,
        measured_points=measured_points,
    )


# ============================================================
# HOST-SIDE MODEL HELPERS
# ============================================================

def build_grid_mapping(cfg: SimulationConfig) -> GridMapping:
    return GridMapping(
        x0=cfg.domain_start_x,
        y0=cfg.domain_start_y,
        dx=cfg.domain_width / float(cfg.grid_width - 1),
        dy=cfg.domain_height / float(cfg.grid_height - 1),
    )


def reference_field(x: float, y: float) -> float:
    return (x * x * x + y * y * y) / 6.0


def compute_mean_discrepancy(cfg: SimulationConfig) -> float:
    if not cfg.measured_points:
        return 0.0

    total = 0.0

    for point in cfg.measured_points:
        total += point.value - reference_field(point.x, point.y)

    return total / float(len(cfg.measured_points))


def build_update_coefficients(
    dx: float,
    dy: float,
    damping: float = 100.0,
) -> UpdateCoefficients:
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("build_update_coefficients: dx and dy must be > 0")

    if damping <= 0.0:
        raise ValueError("build_update_coefficients: damping must be > 0")

    step_x = dx
    step_y = dy

    laplace_x = -2.0 * (1.0 + damping * step_x / (step_x * step_x + damping))
    laplace_y = -2.0 * (1.0 + damping * step_y / (step_y * step_y + damping))

    coeff_x = (step_x + damping * math.exp(step_x)) / (15.0 * damping + step_x)
    coeff_y = (step_y + damping * math.exp(step_y)) / (15.0 * damping + step_y)

    center_x = laplace_x + 0.5 / coeff_x
    center_y = laplace_y + 0.5 / coeff_y

    return UpdateCoefficients(
        damping=damping,
        step_x=step_x,
        step_y=step_y,
        laplace_x=laplace_x,
        laplace_y=laplace_y,
        coeff_x=coeff_x,
        coeff_y=coeff_y,
        center_x=center_x,
        center_y=center_y,
    )


# ============================================================
# HOST-SIDE STRICT FRACTAL INITIALIZATION
# ============================================================

@njit(parallel=True, fastmath=False)
def compute_fractal_weights_host_kernel(
    grid_width: int,
    grid_height: int,
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    max_iterations: int,
) -> np.ndarray:
    """
    Numba-compiled host-side fractal weight computation.

    Uses only primitive arguments so that Numba can compile in nopython mode.
    The loop order and escape-before-update logic match the C++ baseline.
    """
    weight = np.empty(grid_width * grid_height, dtype=np.int32)

    for j in prange(grid_height):
        c_imag = y0 + dy * float(j)
        row = j * grid_width

        for i in range(grid_width):
            c_real = x0 + dx * float(i)

            z_real = 0.0
            z_imag = 0.0
            iteration = 0

            while iteration < max_iterations:
                if z_real * z_real + z_imag * z_imag > 4.0:
                    break

                tmp = z_real * z_real - z_imag * z_imag + c_real
                z_imag = 2.0 * z_real * z_imag + c_imag
                z_real = tmp
                iteration += 1

            weight[row + i] = iteration

    return weight


def compute_fractal_weights_host(
    grid_width: int,
    grid_height: int,
    mapping: GridMapping,
    max_iterations: int,
) -> np.ndarray:
    """
    Python wrapper around the Numba host kernel.

    Validation stays outside @njit because checked_grid_size() and GridMapping
    are normal Python objects.
    """
    checked_grid_size(grid_width, grid_height)

    if max_iterations <= 0:
        raise ValueError("max_iterations must be > 0")

    return compute_fractal_weights_host_kernel(
        grid_width,
        grid_height,
        mapping.x0,
        mapping.y0,
        mapping.dx,
        mapping.dy,
        max_iterations,
    )



# ============================================================
# NUMBA CUDA KERNELS
# ============================================================

@cuda.jit
def initialize_temperature_field_kernel(
    temperature,
    weight_field,
    grid_width,
    grid_height,
    x0,
    y0,
    dx,
    dy,
    mean_discrepancy,
    min_weight,
    max_weight,
):
    i, j = cuda.grid(2)
    if i >= grid_width or j >= grid_height:
        return

    p = i + j * grid_width
    x = x0 + dx * i
    y = y0 + dy * j

    denom = 1.0
    if max_weight > min_weight:
        denom = float(max_weight - min_weight)

    field_value = (x * x * x + y * y * y) / 6.0
    normalized_weight = float(weight_field[p] - min_weight) / denom
    temperature[p] = 293.16 + 80.0 * (mean_discrepancy + field_value) * normalized_weight


@cuda.jit
def update_interior_kernel(
    current_field,
    next_field,
    grid_width,
    grid_height,
    center_x,
    center_y,
    coeff_x,
    coeff_y,
):
    i, j = cuda.grid(2)
    if i < 1 or i >= grid_width - 1 or j < 1 or j >= grid_height - 1:
        return

    p = i + j * grid_width
    next_field[p] = (
        coeff_x * (current_field[p - 1] + current_field[p + 1] + center_x * current_field[p])
        + coeff_y * (current_field[p - grid_width] + current_field[p + grid_width] + center_y * current_field[p])
    )


@cuda.jit
def apply_boundary_left_right_kernel(field, grid_width, grid_height):
    j = cuda.grid(1)
    if j < 1 or j >= grid_height - 1:
        return

    row = j * grid_width
    field[row] = field[row + 1]
    field[row + grid_width - 1] = field[row + grid_width - 2]


@cuda.jit
def apply_boundary_top_bottom_kernel(field, grid_width, grid_height):
    i = cuda.grid(1)
    if i >= grid_width:
        return

    field[i] = field[grid_width + i]
    field[(grid_height - 1) * grid_width + i] = field[(grid_height - 2) * grid_width + i]


# ============================================================
# CUDA LAUNCH HELPERS
# ============================================================

def grid2d_for(
    grid_width: int,
    grid_height: int,
    block2d: Tuple[int, int],
) -> Tuple[int, int]:
    """Calculate the 2D CUDA grid dimensions from problem size and block size."""
    return (
        (grid_width + block2d[0] - 1) // block2d[0],
        (grid_height + block2d[1] - 1) // block2d[1],
    )


def launch_initialize_temperature_field(
    d_temperature: Any,
    d_weight_field: Any,
    grid_width: int,
    grid_height: int,
    mapping: GridMapping,
    mean_discrepancy: float,
    min_weight: int,
    max_weight: int,
    block2d: Tuple[int, int],
    stream: Any,
) -> None:
    grid = grid2d_for(grid_width, grid_height, block2d)

    initialize_temperature_field_kernel[grid, block2d, stream](
        d_temperature,
        d_weight_field,
        grid_width,
        grid_height,
        mapping.x0,
        mapping.y0,
        mapping.dx,
        mapping.dy,
        mean_discrepancy,
        min_weight,
        max_weight,
    )


def launch_advance_temperature_field(
    d_current_field: Any,
    d_next_field: Any,
    grid_width: int,
    grid_height: int,
    coeffs: UpdateCoefficients,
    block2d: Tuple[int, int],
    block1d: int,
    stream: Any,
) -> None:
    # 1. Update interior.
    grid2d = grid2d_for(grid_width, grid_height, block2d)

    update_interior_kernel[grid2d, block2d, stream](
        d_current_field,
        d_next_field,
        grid_width,
        grid_height,
        coeffs.center_x,
        coeffs.center_y,
        coeffs.coeff_x,
        coeffs.coeff_y,
    )

    # 2. Apply left/right boundaries, excluding corners.
    grid_y = (grid_height + block1d - 1) // block1d

    apply_boundary_left_right_kernel[grid_y, block1d, stream](
        d_next_field,
        grid_width,
        grid_height,
    )

    # 3. Apply top/bottom boundaries, including corners.
    grid_x = (grid_width + block1d - 1) // block1d

    apply_boundary_top_bottom_kernel[grid_x, block1d, stream](
        d_next_field,
        grid_width,
        grid_height,
    )


def advance_temperature_field_steps(
    d_current_field: Any,
    d_next_field: Any,
    number_of_steps: int,
    grid_width: int,
    grid_height: int,
    coeffs: UpdateCoefficients,
    block2d: Tuple[int, int],
    block1d: int,
    stream: Any,
) -> Tuple[Any, Any]:
    """Run multiple timesteps and return the current/next device buffers."""
    if number_of_steps < 0:
        raise ValueError("number_of_steps must be >= 0")

    current = d_current_field
    nxt = d_next_field

    for _ in range(number_of_steps):
        launch_advance_temperature_field(
            current,
            nxt,
            grid_width,
            grid_height,
            coeffs,
            block2d,
            block1d,
            stream,
        )

        current, nxt = nxt, current

    return current, nxt


# ============================================================
# HDF5 WRITER
# ============================================================

class TimeSeriesWriter:
    def __init__(
        self,
        file_name: str,
        grid_width: int,
        grid_height: int,
        batch: int = 32,
        tile_y: int = 256,
        tile_x: int = 256,
    ):
        if grid_width <= 0 or grid_height <= 0:
            raise ValueError("TimeSeriesWriter: grid dimensions must be > 0")

        if batch <= 0:
            raise ValueError("TimeSeriesWriter: batch must be > 0")

        if tile_y <= 0 or tile_x <= 0:
            raise ValueError("TimeSeriesWriter: tile sizes must be > 0")

        try:
            import h5py
        except ImportError as e:
            raise RuntimeError(
                "h5py is required for HDF5 output; use --no-hdf5 to disable it"
            ) from e

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.batch = batch
        self.frame_count = 0
        self.capacity = batch
        self.closed = False

        chunk_y = min(grid_height, tile_y)
        chunk_x = min(grid_width, tile_x)

        self.file = h5py.File(file_name, "w")

        self.field = self.file.create_dataset(
            "field",
            shape=(0, grid_height, grid_width),
            maxshape=(None, grid_height, grid_width),
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

    def _extend(self, new_size: int) -> None:
        self.field.resize((new_size, self.grid_height, self.grid_width))
        self.step.resize((new_size,))

    def write_frame(self, step_number: int, field_1d: np.ndarray) -> None:
        if self.closed:
            raise RuntimeError("TimeSeriesWriter: write_frame() called after close()")

        if field_1d.size != self.grid_width * self.grid_height:
            raise RuntimeError("TimeSeriesWriter: field size mismatch")

        if self.frame_count >= self.capacity:
            self.capacity += self.batch
            self._extend(self.capacity)

        self.field[self.frame_count, :, :] = field_1d.reshape(
            self.grid_height,
            self.grid_width,
        )
        self.step[self.frame_count] = np.int32(step_number)
        self.frame_count += 1

    def close(self) -> None:
        if self.closed:
            return

        if self.frame_count != self.capacity:
            self._extend(self.frame_count)

        self.file.flush()
        self.file.close()
        self.closed = True

    def __enter__(self) -> "TimeSeriesWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


# ============================================================
# STATISTICS
# ============================================================

def compute_field_statistics(field: np.ndarray) -> FieldStatistics:
    if field.size == 0:
        raise RuntimeError("compute_field_statistics: empty field")

    min_value = float(np.min(field))
    max_value = float(np.max(field))

    sum_value = float(np.sum(field, dtype=np.float64))
    mean_value = sum_value / float(field.size)

    sum_squares = float(np.sum(field * field, dtype=np.float64))
    l2_norm = float(np.sqrt(sum_squares))

    checksum_weights = (
        (np.arange(field.size, dtype=np.uint64) % np.uint64(1009))
        + np.uint64(1)
    )
    checksum = float(
        np.sum(field * checksum_weights.astype(np.float64), dtype=np.float64)
    )

    diff = field - mean_value
    std_dev = float(
        np.sqrt(np.sum(diff * diff, dtype=np.float64) / float(field.size))
    )

    return FieldStatistics(
        min_value=min_value,
        mean_value=mean_value,
        max_value=max_value,
        std_dev=std_dev,
        l2_norm=l2_norm,
        checksum=checksum,
    )


def write_statistics_header(f) -> None:
    f.write("Step;Min;Mean;Max;Std_dev;L2_norm;Checksum\n")


def write_statistics_row(f, step: int, stats: FieldStatistics) -> None:
    f.write(
        f"{step};"
        f"{stats.min_value:.15g};"
        f"{stats.mean_value:.15g};"
        f"{stats.max_value:.15g};"
        f"{stats.std_dev:.15g};"
        f"{stats.l2_norm:.15g};"
        f"{stats.checksum:.15g}\n"
    )


# ============================================================
# CUDA TIMING / WARMUP
# ============================================================

def elapsed_event_seconds(start_event, stop_event) -> float:
    return cuda.event_elapsed_time(start_event, stop_event) / 1000.0


def time_cuda_segment(callable_fn, stream) -> float:
    start_event = cuda.event()
    stop_event = cuda.event()

    start_event.record(stream)
    callable_fn()
    stop_event.record(stream)
    stop_event.synchronize()

    return elapsed_event_seconds(start_event, stop_event)


def warmup_cuda(
    block2d: Tuple[int, int],
    block1d: int,
    verbose: bool = True,
) -> float:
    if verbose:
        print("Warming up Numba CUDA compiler...")

    start = time.perf_counter()

    grid_width = 4
    grid_height = 4
    total_cells = grid_width * grid_height

    mapping = GridMapping(0.0, 0.0, 0.1, 0.1)
    coeffs = build_update_coefficients(mapping.dx, mapping.dy, 100.0)

    host_weight = compute_fractal_weights_host(grid_width, grid_height, mapping, 2)

    d_weight_field = cuda.to_device(host_weight)
    d_current_field = cuda.device_array(total_cells, dtype=np.float64)
    d_next_field = cuda.device_array(total_cells, dtype=np.float64)

    stream = cuda.stream()
    min_weight = int(np.min(host_weight))
    max_weight = int(np.max(host_weight))

    launch_initialize_temperature_field(
        d_current_field,
        d_weight_field,
        grid_width,
        grid_height,
        mapping,
        0.0,
        min_weight,
        max_weight,
        block2d,
        stream,
    )

    launch_advance_temperature_field(
        d_current_field,
        d_next_field,
        grid_width,
        grid_height,
        coeffs,
        block2d,
        block1d,
        stream,
    )

    stream.synchronize()

    elapsed = time.perf_counter() - start

    if verbose:
        print("CUDA warmup completed; reported GPU timings exclude compilation.")

    return elapsed


def maybe_get_h5py_version(write_hdf5: bool) -> str:
    if not write_hdf5:
        return "disabled"

    try:
        import h5py
        return h5py.__version__
    except Exception:
        return "unavailable"


# ============================================================
# SIMULATION DRIVER
# ============================================================

def run_simulation(
    cfg: SimulationConfig,
    h5_file: str,
    csv_file: str,
    h5_tile_y: int,
    h5_tile_x: int,
    block2d: Tuple[int, int],
    block1d: int,
    write_hdf5: bool,
    skip_warmup: bool,
) -> None:
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available")

    validate_block_sizes(block2d, block1d)
    validate_output_every(cfg.output_every)

    ensure_parent_directory_exists(csv_file)

    if write_hdf5:
        ensure_parent_directory_exists(h5_file)

    total_cells = checked_grid_size(cfg.grid_width, cfg.grid_height)

    mapping = build_grid_mapping(cfg)
    coeffs = build_update_coefficients(mapping.dx, mapping.dy, 100.0)
    mean_discrepancy = compute_mean_discrepancy(cfg)

    stream = cuda.stream()

    timings = RunTimings()

    if not skip_warmup:
        timings.warmup_time = warmup_cuda(block2d, block1d, verbose=True)
    else:
        print("Skipping explicit CUDA warmup; first-call compilation time may be included in timings.")

    total_wall_start = time.perf_counter()

    # --------------------------------------------------------
    # Host-side strict fractal initialization.
    # --------------------------------------------------------
    weight_start = time.perf_counter()
    host_weight_field = compute_fractal_weights_host(
        cfg.grid_width,
        cfg.grid_height,
        mapping,
        cfg.max_fractal_iterations,
    )
    timings.weight_host_time = time.perf_counter() - weight_start

    range_start = time.perf_counter()
    min_weight = int(np.min(host_weight_field))
    max_weight = int(np.max(host_weight_field))
    timings.weight_range_time = time.perf_counter() - range_start

    h2d_start = time.perf_counter()
    d_weight_field = cuda.to_device(host_weight_field, stream=stream)
    stream.synchronize()
    timings.weight_h2d_time = time.perf_counter() - h2d_start

    d_current_field = cuda.device_array(total_cells, dtype=np.float64, stream=stream)
    d_next_field = cuda.device_array(total_cells, dtype=np.float64, stream=stream)
    host_field = cuda.pinned_array(total_cells, dtype=np.float64)

    timings.init_kernel_time = time_cuda_segment(
        lambda: launch_initialize_temperature_field(
            d_current_field,
            d_weight_field,
            cfg.grid_width,
            cfg.grid_height,
            mapping,
            mean_discrepancy,
            min_weight,
            max_weight,
            block2d,
            stream,
        ),
        stream,
    )

    output_schedule = build_output_schedule(cfg.time_steps, cfg.output_every)

    output_frames = 0
    final_stats = FieldStatistics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    writer_ctx = (
        TimeSeriesWriter(
            h5_file,
            cfg.grid_width,
            cfg.grid_height,
            32,
            h5_tile_y,
            h5_tile_x,
        )
        if write_hdf5
        else None
    )

    loop_start = time.perf_counter()

    try:
        with open(csv_file, "w", encoding="utf-8") as csvf:
            csv_start = time.perf_counter()
            write_statistics_header(csvf)
            timings.csv_time += time.perf_counter() - csv_start

            writer = writer_ctx

            def write_output_frame(step: int) -> None:
                nonlocal output_frames, final_stats, d_current_field

                copy_start2 = time.perf_counter()
                d_current_field.copy_to_host(host_field, stream=stream)
                stream.synchronize()
                timings.device_to_host_copy_time += time.perf_counter() - copy_start2

                stats_start = time.perf_counter()
                stats = compute_field_statistics(host_field)
                timings.statistics_time += time.perf_counter() - stats_start

                final_stats = stats

                csv_frame_start = time.perf_counter()
                write_statistics_row(csvf, step, stats)
                timings.csv_time += time.perf_counter() - csv_frame_start

                if writer is not None:
                    hdf5_start = time.perf_counter()
                    writer.write_frame(step, host_field)
                    timings.hdf5_time += time.perf_counter() - hdf5_start

                output_frames += 1

            current_step = 0

            for target_step in output_schedule:
                if target_step < current_step or target_step > cfg.time_steps:
                    raise RuntimeError("Internal error: invalid output schedule")

                steps_to_advance = target_step - current_step

                if steps_to_advance > 0:
                    def do_steps() -> None:
                        nonlocal d_current_field, d_next_field

                        d_current_field, d_next_field = advance_temperature_field_steps(
                            d_current_field,
                            d_next_field,
                            steps_to_advance,
                            cfg.grid_width,
                            cfg.grid_height,
                            coeffs,
                            block2d,
                            block1d,
                            stream,
                        )

                    timings.pure_dynamics_gpu_time += time_cuda_segment(do_steps, stream)
                    current_step = target_step

                if should_write_step(target_step, cfg.time_steps, cfg.output_every):
                    write_output_frame(target_step)

            csvf.flush()

            if writer is not None:
                writer.close()

    finally:
        if writer_ctx is not None and not writer_ctx.closed:
            writer_ctx.close()

    stream.synchronize()

    timings.loop_wall_time = time.perf_counter() - loop_start
    timings.total_wall_time = time.perf_counter() - total_wall_start

    updates = (
        float(cfg.grid_width - 2)
        * float(cfg.grid_height - 2)
        * float(cfg.time_steps)
    )

    print(f"Grid:                         {cfg.grid_width} x {cfg.grid_height}")
    print(f"Measured points:              {len(cfg.measured_points)}")
    print(f"Max fractal iterations:       {cfg.max_fractal_iterations}")
    print(f"Time steps:                   {cfg.time_steps}")

    if cfg.output_every == 0:
        print("Snapshot/statistics policy:    final step only")
    else:
        print(
            "Snapshot/statistics policy:    "
            f"step 0, every {cfg.output_every} step(s), and final step"
        )

    print(f"Output frames:                {output_frames}")
    print(f"Python version:               {platform.python_version()}")
    print(f"NumPy version:                {np.__version__}")
    print(f"Numba version:                {numba.__version__}")
    print(f"h5py version:                 {maybe_get_h5py_version(write_hdf5)}")
    print(f"CUDA available:               {cuda.is_available()}")
    print(f"CUDA device:                  {cuda_device_name()}")
    print(f"HDF5 chunk tile:              {min(cfg.grid_height, h5_tile_y)} x {min(cfg.grid_width, h5_tile_x)}")
    print(f"CUDA block2d:                 {block2d}")
    print(f"CUDA block1d:                 {block1d}")
    print("CUDA backend:                 Numba CUDA")
    print("Weight initialization:        host scalar baseline-compatible")
    print(f"CUDA warmup time:             {timings.warmup_time:.6f} s")
    print(f"Weight field host time:       {timings.weight_host_time:.6f} s")
    print(f"Weight H2D copy time:         {timings.weight_h2d_time:.6f} s")
    print(f"Weight range reduction time:  {timings.weight_range_time:.6f} s")
    print(f"Init kernel GPU time:         {timings.init_kernel_time:.6f} s")
    print(f"Pure dynamics GPU time:       {timings.pure_dynamics_gpu_time:.6f} s")
    print(f"Device-to-host copy time:     {timings.device_to_host_copy_time:.6f} s")
    print(f"Statistics time:              {timings.statistics_time:.6f} s")
    print(f"CSV write time:               {timings.csv_time:.6f} s")
    print(f"HDF5 write time:              {timings.hdf5_time:.6f} s")
    print(f"Dynamics loop wall time:      {timings.loop_wall_time:.6f} s")
    print(f"Total measured wall time:     {timings.total_wall_time:.6f} s")

    if cfg.time_steps > 0 and timings.pure_dynamics_gpu_time > 0.0:
        print(
            f"Pure dynamics performance:    "
            f"{updates / timings.pure_dynamics_gpu_time / 1e9:.6f} GLUP/s"
        )

    if cfg.time_steps > 0 and timings.loop_wall_time > 0.0:
        print(
            f"End-to-end loop performance:  "
            f"{updates / timings.loop_wall_time / 1e9:.6f} GLUP/s"
        )

    print(f"Mean discrepancy:             {mean_discrepancy:.15g}")
    print(f"Final min:                    {final_stats.min_value:.15g}")
    print(f"Final mean:                   {final_stats.mean_value:.15g}")
    print(f"Final max:                    {final_stats.max_value:.15g}")
    print(f"Final std.dev.:               {final_stats.std_dev:.15g}")
    print(f"Final L2 norm:                {final_stats.l2_norm:.15g}")
    print(f"Final checksum:               {final_stats.checksum:.15g}")
    print(f"Weight range:                 {min_weight} ... {max_weight}")
    print("Simulation completed successfully.")


# ============================================================
# COMMAND LINE
# ============================================================

def parse_command_line(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Numba CUDA cooling solver"
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
        default="none",
        help="Output HDF5 file, or 'none' to disable HDF5 output",
    )

    parser.add_argument(
        "csv",
        nargs="?",
        default="output/Cooling_numba_cuda.csv",
        help="Output CSV file",
    )

    parser.add_argument(
        "outputEvery",
        nargs="?",
        type=int,
        default=None,
        help=(
            "Optional snapshot cadence override. "
            "0 means final-only; >0 means step 0, periodic snapshots, and final."
        ),
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index",
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
        "--block1d",
        "--block-1d",
        dest="block1d",
        type=int,
        default=256,
        help="CUDA block size for 1D boundary kernels",
    )

    parser.add_argument(
        "--no-hdf5",
        action="store_true",
        help="Disable HDF5 output",
    )

    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Do not run explicit Numba CUDA warmup",
    )

    parser.add_argument(
        "--h5-tile-y",
        type=int,
        default=256,
        help="HDF5 chunk tile size in y",
    )

    parser.add_argument(
        "--h5-tile-x",
        type=int,
        default=256,
        help="HDF5 chunk tile size in x",
    )

    return parser.parse_args(argv)


# ============================================================
# MAIN
# ============================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_command_line(argv)

    if args.h5_tile_y <= 0 or args.h5_tile_x <= 0:
        raise ValueError("HDF5 tile sizes must be > 0")

    block2d = (args.block_x, args.block_y)
    block1d = args.block1d

    validate_block_sizes(block2d, block1d)

    if not cuda.is_available():
        raise RuntimeError("CUDA is not available")

    cuda.select_device(args.device)

    cfg = read_configuration_file(args.input)

    if args.outputEvery is not None:
        validate_output_every(args.outputEvery)
        cfg.output_every = args.outputEvery

    validate_output_every(cfg.output_every)

    write_hdf5 = (not args.no_hdf5) and (not is_no_hdf5_token(args.h5))

    print(f"Input file:                   {args.input}")
    print(f"CSV output:                   {args.csv}")
    print(f"HDF5 output:                  {args.h5 if write_hdf5 else 'disabled'}")
    print(f"Official grading mode:        {'yes' if not write_hdf5 else 'no'}")

    run_simulation(
        cfg=cfg,
        h5_file=args.h5,
        csv_file=args.csv,
        h5_tile_y=args.h5_tile_y,
        h5_tile_x=args.h5_tile_x,
        block2d=block2d,
        block1d=block1d,
        write_hdf5=write_hdf5,
        skip_warmup=args.skip_warmup,
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"CRITICAL ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

