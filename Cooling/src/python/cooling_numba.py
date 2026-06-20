#!/usr/bin/env python3
"""
Cooling / field-evolution solver - Numba CPU reference implementation.

This file is the official/reference Numba multicore CPU version aligned with
the Python/NumPy, C++17, OpenMP, and CUDA versions of the assignment.

It keeps the same numerical model and output conventions while replacing the
main NumPy compute kernels with explicit Numba-compiled loops.

Parallelized with Numba CPU:

  1) compute_fractal_weights_kernel()
  2) initialize_temperature_field_kernel()
  3) update_interior_kernel()
  4) boundary kernels

Host-side by design:

  - HDF5 output
  - CSV output
  - snapshot statistics and checksum

Important design choices:

  - HDF5 output is optional and disabled by default.
  - Statistics are computed using a two-pass float64 implementation.
  - outputEvery = 0 means final frame/statistics only.
  - outputEvery > 0 means step 0, every outputEvery steps, and final step.
  - CSV statistics include L2_norm and a deterministic checksum.

Official performance mode:

  python ./path/to/cooling_numba.py input/Cooling.in none output/Cooling_numba.csv 0

or with explicit thread count:

  python ./path/to/cooling_numba.py --threads 16 input/Cooling.in none output/Cooling_numba.csv 0

Command line:

  python ./path/to/cooling_numba.py [options] [inputFile] [h5File|none] [csvFile] [outputEvery]

Examples:

  python ./path/to/cooling_numba.py --threads 8           input/Cooling.in none                    output/Cooling_numba.csv  0
  python ./path/to/cooling_numba.py --threads 8           input/Cooling.in output/Cooling_numba.h5 output/Cooling_numba.csv 50
  python ./path/to/cooling_numba.py --threads 8 --no-hdf5 input/Cooling.in none                    output/Cooling_numba.csv  0
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numba
import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads


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
    output_every: int  # 0 means final-only output

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
    weight_time: float = 0.0
    weight_range_time: float = 0.0
    init_time: float = 0.0
    pure_dynamics_time: float = 0.0
    statistics_time: float = 0.0
    csv_time: float = 0.0
    hdf5_time: float = 0.0
    loop_wall_time: float = 0.0
    total_wall_time: float = 0.0
    warmup_time: float = 0.0


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
# NUMBA CPU KERNELS
# ============================================================

@njit(parallel=True, fastmath=False, cache=True)
def compute_fractal_weights_kernel(
    weight_field: np.ndarray,
    grid_width: int,
    grid_height: int,
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    max_fractal_iterations: int,
) -> None:
    for j in prange(grid_height):
        row = j * grid_width
        y = y0 + dy * float(j)

        for i in range(grid_width):
            x = x0 + dx * float(i)
            p = row + i

            z_real = 0.0
            z_imag = 0.0
            iteration = 0

            while iteration < max_fractal_iterations:
                if z_real * z_real + z_imag * z_imag > 4.0:
                    break

                tmp = z_real * z_real - z_imag * z_imag + x
                z_imag = 2.0 * z_real * z_imag + y
                z_real = tmp

                iteration += 1

            weight_field[p] = iteration


@njit(parallel=True, fastmath=False, cache=True)
def initialize_temperature_field_kernel(
    temperature: np.ndarray,
    weight_field: np.ndarray,
    grid_width: int,
    grid_height: int,
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    mean_discrepancy: float,
    min_weight: int,
    max_weight: int,
) -> None:
    denom = float(max_weight - min_weight) if max_weight > min_weight else 1.0

    for j in prange(grid_height):
        y = y0 + dy * float(j)
        row = j * grid_width

        for i in range(grid_width):
            x = x0 + dx * float(i)
            p = row + i

            field_value = (x * x * x + y * y * y) / 6.0
            normalized_weight = float(weight_field[p] - min_weight) / denom

            temperature[p] = (
                293.16
                + 80.0 * (mean_discrepancy + field_value) * normalized_weight
            )


@njit(parallel=True, fastmath=False, cache=True)
def update_interior_kernel(
    current_field: np.ndarray,
    next_field: np.ndarray,
    grid_width: int,
    grid_height: int,
    center_x: float,
    center_y: float,
    coeff_x: float,
    coeff_y: float,
) -> None:
    for j in prange(1, grid_height - 1):
        row = j * grid_width
        row_up = (j - 1) * grid_width
        row_down = (j + 1) * grid_width

        for i in range(1, grid_width - 1):
            p = row + i

            next_field[p] = (
                coeff_x
                * (
                    current_field[p - 1]
                    + current_field[p + 1]
                    + center_x * current_field[p]
                )
                + coeff_y
                * (
                    current_field[row_up + i]
                    + current_field[row_down + i]
                    + center_y * current_field[p]
                )
            )


@njit(parallel=True, fastmath=False, cache=True)
def apply_boundary_left_right_kernel(
    field: np.ndarray,
    grid_width: int,
    grid_height: int,
) -> None:
    for j in prange(1, grid_height - 1):
        row = j * grid_width

        field[row] = field[row + 1]
        field[row + grid_width - 1] = field[row + grid_width - 2]


@njit(parallel=True, fastmath=False, cache=True)
def apply_boundary_top_bottom_kernel(
    field: np.ndarray,
    grid_width: int,
    grid_height: int,
) -> None:
    row_top = 0
    row_1 = grid_width
    row_nm2 = (grid_height - 2) * grid_width
    row_nm1 = (grid_height - 1) * grid_width

    for i in prange(grid_width):
        field[row_top + i] = field[row_1 + i]
        field[row_nm1 + i] = field[row_nm2 + i]


def advance_temperature_field(
    current_field: np.ndarray,
    next_field: np.ndarray,
    grid_width: int,
    grid_height: int,
    coeffs: UpdateCoefficients,
) -> None:
    update_interior_kernel(
        current_field,
        next_field,
        grid_width,
        grid_height,
        coeffs.center_x,
        coeffs.center_y,
        coeffs.coeff_x,
        coeffs.coeff_y,
    )

    apply_boundary_left_right_kernel(next_field, grid_width, grid_height)
    apply_boundary_top_bottom_kernel(next_field, grid_width, grid_height)


def advance_temperature_field_steps(
    current_field: np.ndarray,
    next_field: np.ndarray,
    number_of_steps: int,
    grid_width: int,
    grid_height: int,
    coeffs: UpdateCoefficients,
) -> Tuple[np.ndarray, np.ndarray]:
    if number_of_steps < 0:
        raise ValueError("number_of_steps must be >= 0")

    current = current_field
    nxt = next_field

    for _ in range(number_of_steps):
        advance_temperature_field(current, nxt, grid_width, grid_height, coeffs)
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
# NUMBA WARM-UP
# ============================================================

def warmup_numba(verbose: bool = True) -> float:
    """
    Force Numba to compile kernels before measured sections.

    Returned wall time is reported separately and excluded from kernel timings.
    """
    if verbose:
        print("Warming up Numba JIT compiler...")

    start = time.perf_counter()

    grid_width = 4
    grid_height = 4
    total_cells = grid_width * grid_height

    mapping = GridMapping(x0=0.0, y0=0.0, dx=0.1, dy=0.1)
    coeffs = build_update_coefficients(mapping.dx, mapping.dy, 100.0)

    weight_field = np.empty(total_cells, dtype=np.int32)
    current_field = np.zeros(total_cells, dtype=np.float64)
    next_field = np.zeros(total_cells, dtype=np.float64)

    compute_fractal_weights_kernel(
        weight_field,
        grid_width,
        grid_height,
        mapping.x0,
        mapping.y0,
        mapping.dx,
        mapping.dy,
        2,
    )

    initialize_temperature_field_kernel(
        current_field,
        weight_field,
        grid_width,
        grid_height,
        mapping.x0,
        mapping.y0,
        mapping.dx,
        mapping.dy,
        0.0,
        0,
        10,
    )

    advance_temperature_field(
        current_field,
        next_field,
        grid_width,
        grid_height,
        coeffs,
    )

    elapsed = time.perf_counter() - start

    if verbose:
        print("JIT warmup completed; reported timings exclude compilation.")

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
    write_hdf5: bool,
    skip_warmup: bool,
) -> None:
    validate_output_every(cfg.output_every)

    ensure_parent_directory_exists(csv_file)

    if write_hdf5:
        ensure_parent_directory_exists(h5_file)

    total_cells = checked_grid_size(cfg.grid_width, cfg.grid_height)

    mapping = build_grid_mapping(cfg)
    coeffs = build_update_coefficients(mapping.dx, mapping.dy, 100.0)
    mean_discrepancy = compute_mean_discrepancy(cfg)

    weight_field = np.empty(total_cells, dtype=np.int32)
    current_field = np.empty(total_cells, dtype=np.float64)
    next_field = np.empty(total_cells, dtype=np.float64)

    timings = RunTimings()

    if not skip_warmup:
        timings.warmup_time = warmup_numba(verbose=True)
    else:
        print("Skipping explicit Numba warmup; first-call JIT time may be included in timings.")

    total_wall_start = time.perf_counter()

    weight_start = time.perf_counter()
    compute_fractal_weights_kernel(
        weight_field,
        cfg.grid_width,
        cfg.grid_height,
        mapping.x0,
        mapping.y0,
        mapping.dx,
        mapping.dy,
        cfg.max_fractal_iterations,
    )
    timings.weight_time = time.perf_counter() - weight_start

    weight_range_start = time.perf_counter()
    min_weight = int(np.min(weight_field))
    max_weight = int(np.max(weight_field))
    timings.weight_range_time = time.perf_counter() - weight_range_start

    init_start = time.perf_counter()
    initialize_temperature_field_kernel(
        current_field,
        weight_field,
        cfg.grid_width,
        cfg.grid_height,
        mapping.x0,
        mapping.y0,
        mapping.dx,
        mapping.dy,
        mean_discrepancy,
        min_weight,
        max_weight,
    )
    timings.init_time = time.perf_counter() - init_start

    output_frames = 0
    final_stats = FieldStatistics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    output_schedule = build_output_schedule(cfg.time_steps, cfg.output_every)

    writer_ctx = (
        TimeSeriesWriter(
            h5_file,
            cfg.grid_width,
            cfg.grid_height,
            batch=32,
            tile_y=h5_tile_y,
            tile_x=h5_tile_x,
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
                nonlocal output_frames, final_stats

                stats_start = time.perf_counter()
                stats = compute_field_statistics(current_field)
                timings.statistics_time += time.perf_counter() - stats_start

                final_stats = stats

                csv_frame_start = time.perf_counter()
                write_statistics_row(csvf, step, stats)
                timings.csv_time += time.perf_counter() - csv_frame_start

                if writer is not None:
                    hdf5_start = time.perf_counter()
                    writer.write_frame(step, current_field)
                    timings.hdf5_time += time.perf_counter() - hdf5_start

                output_frames += 1

            current_step = 0

            for target_step in output_schedule:
                if target_step < current_step or target_step > cfg.time_steps:
                    raise RuntimeError("Internal error: invalid output schedule")

                steps_to_advance = target_step - current_step

                if steps_to_advance > 0:
                    dyn_start = time.perf_counter()
                    current_field, next_field = advance_temperature_field_steps(
                        current_field,
                        next_field,
                        steps_to_advance,
                        cfg.grid_width,
                        cfg.grid_height,
                        coeffs,
                    )
                    timings.pure_dynamics_time += time.perf_counter() - dyn_start
                    current_step = target_step

                if should_write_step(target_step, cfg.time_steps, cfg.output_every):
                    write_output_frame(target_step)

            csvf.flush()

            if writer is not None:
                writer.close()

    finally:
        if writer_ctx is not None and not writer_ctx.closed:
            writer_ctx.close()

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
    print(f"Numba threads:                {get_num_threads()}")
    print(f"NUMBA_NUM_THREADS env:        {os.environ.get('NUMBA_NUM_THREADS', '(not set)')}")
    print(f"HDF5 chunk tile:              {min(cfg.grid_height, h5_tile_y)} x {min(cfg.grid_width, h5_tile_x)}")
    print("Python backend:               Numba CPU multicore")
    print(f"Numba warmup time:            {timings.warmup_time:.6f} s")
    print(f"Weight field time:            {timings.weight_time:.6f} s")
    print(f"Weight range reduction time:  {timings.weight_range_time:.6f} s")
    print(f"Init field time:              {timings.init_time:.6f} s")
    print(f"Pure dynamics compute time:   {timings.pure_dynamics_time:.6f} s")
    print(f"Statistics time:              {timings.statistics_time:.6f} s")
    print(f"CSV write time:               {timings.csv_time:.6f} s")
    print(f"HDF5 write time:              {timings.hdf5_time:.6f} s")
    print(f"Dynamics loop wall time:      {timings.loop_wall_time:.6f} s")
    print(f"Total measured wall time:     {timings.total_wall_time:.6f} s")

    if cfg.time_steps > 0 and timings.pure_dynamics_time > 0.0:
        print(
            f"Pure dynamics performance:    "
            f"{updates / timings.pure_dynamics_time / 1e9:.6f} GLUP/s"
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
        description="CPU-only Numba multicore cooling solver"
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
        default="output/Cooling_numba.csv",
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
        "--threads",
        type=int,
        default=None,
        help="Number of Numba CPU threads",
    )

    parser.add_argument(
        "--no-hdf5",
        action="store_true",
        help="Disable HDF5 output",
    )

    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Do not run explicit Numba JIT warmup",
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

    if args.threads is not None:
        if args.threads <= 0:
            raise ValueError("--threads must be > 0")
        set_num_threads(args.threads)

    if args.h5_tile_y <= 0 or args.h5_tile_x <= 0:
        raise ValueError("HDF5 tile sizes must be > 0")

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

