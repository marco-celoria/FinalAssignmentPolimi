#!/usr/bin/env python3
"""
Cooling / field-evolution solver - Python/NumPy teaching baseline.

This file is the Python teaching baseline aligned with the polished C++,
OpenMP, and CUDA versions of the assignment.

It is intentionally a CPU NumPy implementation. Students can use it as a
starting point for:

  1) Numba CPU multicore parallelization
  2) Numba CUDA/CuPy GPU offloading

Important design choices:

  - HDF5 output is host-side and optional.
  - HDF5 is disabled by default, matching the official C++ baseline policy.
  - Statistics are computed using a two-pass float64 implementation.
  - outputEvery = 0 means final frame/statistics only.
  - outputEvery > 0 means step 0, every outputEvery steps, and final step.
  - CSV statistics include L2_norm and a deterministic checksum to help validation.

Official performance mode:

  python ./path/to/cooling.py input/Cooling.in none output/Cooling_python.csv 0

Command line:

  python ./path/to/cooling.py [options] [inputFile] [h5File|none] [csvFile] [outputEvery]

Examples:

  python ./path/to/cooling.py           input/Cooling.in none                     output/Cooling_python.csv  0
  python ./path/to/cooling.py           input/Cooling.in output/Cooling_python.h5 output/Cooling_python.csv 50
  python ./path/to/cooling.py --no-hdf5 input/Cooling.in none                     output/Cooling_python.csv  0

"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

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


def checked_grid_size(width: int, height: int) -> int:
    if width <= 0 or height <= 0:
        raise ValueError("Grid dimensions must be > 0")

    total = width * height
    if total <= 0:
        raise ValueError("Invalid total grid size")

    return total


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
#
# Input file format, after removing comments beginning with '#':
#
#   grid_width
#   grid_height
#   number_of_measured_points
#   x y value      repeated number_of_measured_points times
#   domain_start_x
#   domain_start_y
#   domain_width
#   domain_height
#   max_fractal_iterations
#   time_steps
#   output_every   optional; 0 means final-only output
#
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
# MODEL HELPERS
# ============================================================

def build_grid_mapping(cfg: SimulationConfig) -> GridMapping:
    """
    Node-centered geometry including both boundaries:

        x(i) = x0 + i * dx,   dx = domain_width  / (grid_width  - 1)
        y(j) = y0 + j * dy,   dy = domain_height / (grid_height - 1)
    """
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

    return UpdateCoefficients(
        damping=damping,
        step_x=step_x,
        step_y=step_y,
        laplace_x=laplace_x,
        laplace_y=laplace_y,
        coeff_x=coeff_x,
        coeff_y=coeff_y,
    )


# ============================================================
# NUMPY COMPUTE KERNELS
# ============================================================

def compute_fractal_weights_numpy(
    grid_width: int,
    grid_height: int,
    mapping: GridMapping,
    max_iterations: int,
) -> np.ndarray:
    """
    NumPy version of the Mandelbrot-style weight kernel.

    Returns:
        int32 array of shape (grid_height, grid_width)
    """
    x = mapping.x0 + mapping.dx * np.arange(grid_width, dtype=np.float64)
    y = mapping.y0 + mapping.dy * np.arange(grid_height, dtype=np.float64)

    c_real = np.broadcast_to(x[None, :], (grid_height, grid_width))
    c_imag = np.broadcast_to(y[:, None], (grid_height, grid_width))

    z_real = np.zeros((grid_height, grid_width), dtype=np.float64)
    z_imag = np.zeros((grid_height, grid_width), dtype=np.float64)
    weight = np.zeros((grid_height, grid_width), dtype=np.int32)
    active = np.ones((grid_height, grid_width), dtype=bool)

    for iteration in range(max_iterations):
        if not bool(np.any(active)):
            break

        zr = z_real[active]
        zi = z_imag[active]
        cr = c_real[active]
        ci = c_imag[active]

        zr_new = zr * zr - zi * zi + cr
        zi_new = 2.0 * zr * zi + ci

        z_real[active] = zr_new
        z_imag[active] = zi_new

        weight[active] = iteration + 1
        active[active] = (zr_new * zr_new + zi_new * zi_new) <= 4.0

    return weight


def initialize_temperature_field_numpy(
    weight_field: np.ndarray,
    mapping: GridMapping,
    mean_discrepancy: float,
    min_weight: int,
    max_weight: int,
) -> np.ndarray:
    if weight_field.ndim != 2:
        raise ValueError("weight_field must be a 2-D array")

    grid_height, grid_width = weight_field.shape

    x = mapping.x0 + mapping.dx * np.arange(grid_width, dtype=np.float64)
    y = mapping.y0 + mapping.dy * np.arange(grid_height, dtype=np.float64)

    x3 = x[None, :] * x[None, :] * x[None, :]
    y3 = y[:, None] * y[:, None] * y[:, None]

    ref = (x3 + y3) / 6.0

    denom = float(max_weight - min_weight) if max_weight > min_weight else 1.0
    normalized_weight = (weight_field.astype(np.float64) - float(min_weight)) / denom

    field = 293.16 + 80.0 * (mean_discrepancy + ref) * normalized_weight

    return np.asarray(field, dtype=np.float64)


def update_temperature_field_numpy(
    current: np.ndarray,
    next_field: np.ndarray,
    coeffs: UpdateCoefficients,
) -> None:
    """
    One timestep update using NumPy slicing.

    Boundary order matches the C++ baseline:
      1) update interior
      2) update left/right boundaries, excluding corners
      3) update top/bottom boundaries, including corners
    """
    if current.shape != next_field.shape or current.ndim != 2:
        raise ValueError("current and next_field must be 2-D arrays with identical shape")

    cx = coeffs.coeff_x
    cy = coeffs.coeff_y
    lx = coeffs.laplace_x
    ly = coeffs.laplace_y

    next_field[1:-1, 1:-1] = (
        cx * (
            current[1:-1, :-2]
            + current[1:-1, 2:]
            + (lx + 0.5 / cx) * current[1:-1, 1:-1]
        )
        + cy * (
            current[:-2, 1:-1]
            + current[2:, 1:-1]
            + (ly + 0.5 / cy) * current[1:-1, 1:-1]
        )
    )

    next_field[1:-1, 0] = next_field[1:-1, 1]
    next_field[1:-1, -1] = next_field[1:-1, -2]

    next_field[0, :] = next_field[1, :]
    next_field[-1, :] = next_field[-2, :]


def advance_temperature_field_steps_numpy(
    current_field: np.ndarray,
    next_field: np.ndarray,
    number_of_steps: int,
    coeffs: UpdateCoefficients,
) -> Tuple[np.ndarray, np.ndarray]:
    if number_of_steps < 0:
        raise ValueError("number_of_steps must be >= 0")

    current = current_field
    nxt = next_field

    for _ in range(number_of_steps):
        update_temperature_field_numpy(current, nxt, coeffs)
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
            raise RuntimeError("h5py is required for HDF5 output; use --no-hdf5 to disable it") from e

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

    def write_frame(self, step_number: int, field_2d: np.ndarray) -> None:
        if self.closed:
            raise RuntimeError("TimeSeriesWriter: write_frame() called after close()")

        if field_2d.shape != (self.grid_height, self.grid_width):
            raise RuntimeError("TimeSeriesWriter: field shape mismatch")

        if self.frame_count >= self.capacity:
            self.capacity += self.batch
            self._extend(self.capacity)

        self.field[self.frame_count, :, :] = field_2d
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

    flat = np.ravel(field)

    min_value = float(np.min(flat))
    max_value = float(np.max(flat))

    sum_value = float(np.sum(flat, dtype=np.float64))
    mean_value = sum_value / float(flat.size)

    sum_squares = float(np.sum(flat * flat, dtype=np.float64))
    l2_norm = float(np.sqrt(sum_squares))

    checksum_weights = (
        (np.arange(flat.size, dtype=np.uint64) % np.uint64(1009))
        + np.uint64(1)
    )
    checksum = float(
        np.sum(flat * checksum_weights.astype(np.float64), dtype=np.float64)
    )

    diff = flat - mean_value
    std_dev = float(
        np.sqrt(np.sum(diff * diff, dtype=np.float64) / float(flat.size))
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
# SIMULATION DRIVER
# ============================================================

def run_simulation(
    cfg: SimulationConfig,
    h5_file: str,
    csv_file: str,
    h5_tile_y: int,
    h5_tile_x: int,
    write_hdf5: bool,
) -> None:
    validate_output_every(cfg.output_every)
    checked_grid_size(cfg.grid_width, cfg.grid_height)

    ensure_parent_directory_exists(csv_file)

    if write_hdf5:
        ensure_parent_directory_exists(h5_file)

    mapping = build_grid_mapping(cfg)
    coeffs = build_update_coefficients(mapping.dx, mapping.dy, 100.0)
    mean_discrepancy = compute_mean_discrepancy(cfg)

    timings = RunTimings()
    total_wall_start = time.perf_counter()

    weight_start = time.perf_counter()
    weight_field = compute_fractal_weights_numpy(
        cfg.grid_width,
        cfg.grid_height,
        mapping,
        cfg.max_fractal_iterations,
    )
    timings.weight_time = time.perf_counter() - weight_start

    weight_range_start = time.perf_counter()
    min_weight = int(np.min(weight_field))
    max_weight = int(np.max(weight_field))
    timings.weight_range_time = time.perf_counter() - weight_range_start

    init_start = time.perf_counter()
    current_field = initialize_temperature_field_numpy(
        weight_field,
        mapping,
        mean_discrepancy,
        min_weight,
        max_weight,
    )
    timings.init_time = time.perf_counter() - init_start

    next_field = np.empty_like(current_field)

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
            csv_header_start = time.perf_counter()
            write_statistics_header(csvf)
            timings.csv_time += time.perf_counter() - csv_header_start
            writer = writer_ctx

            def write_output_frame(step: int) -> None:
                nonlocal output_frames, final_stats

                stats_start = time.perf_counter()
                stats = compute_field_statistics(current_field)
                timings.statistics_time += time.perf_counter() - stats_start

                final_stats = stats

                csv_start = time.perf_counter()
                write_statistics_row(csvf, step, stats)
                timings.csv_time += time.perf_counter() - csv_start

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
                    current_field, next_field = advance_temperature_field_steps_numpy(
                        current_field,
                        next_field,
                        steps_to_advance,
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

    print(f"HDF5 chunk tile:              {min(cfg.grid_height, h5_tile_y)} x {min(cfg.grid_width, h5_tile_x)}")
    print("Explicit parallelism:          no")
    print("Python backend:                NumPy CPU")
    print(f"Weight field time:            {timings.weight_time:.6f} s")
    print(f"Weight range reduction time:  {timings.weight_range_time:.6f} s")
    print(f"Init field time:              {timings.init_time:.6f} s")
    print(f"Pure dynamics compute time:   {timings.pure_dynamics_time:.6f} s")
    print(f"Statistics time:              {timings.statistics_time:.6f} s")
    print(f"CSV write time:               {timings.csv_time:.6f} s")
    print(f"HDF5 write time:              {timings.hdf5_time:.6f} s")
    print(f"Dynamics loop wall time:      {timings.loop_wall_time:.6f} s")
    print(f"Total measured wall time:     {timings.total_wall_time:.6f} s")
    print(f"Output frames:                {output_frames}")

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
        description="Python NumPy baseline cooling solver"
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
        default="output/Cooling_python.csv",
        help="Output CSV file",
    )

    parser.add_argument(
        "outputEvery",
        nargs="?",
        type=int,
        default=None,
        help=(
            "Optional snapshot cadence override. "
            "0 means final-only output; >0 means step 0, periodic snapshots, and final."
        ),
    )

    parser.add_argument(
        "--no-hdf5",
        action="store_true",
        help="Disable HDF5 output",
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

    cfg = read_configuration_file(args.input)

    if args.outputEvery is not None:
        validate_output_every(args.outputEvery)
        cfg.output_every = args.outputEvery

    validate_output_every(cfg.output_every)

    write_hdf5 = (not args.no_hdf5) and (not is_no_hdf5_token(args.h5))
    print(f"Python version:               {sys.version.split()[0]}")
    print(f"NumPy version:                {np.__version__}")
    print(f"Input file:                   {args.input}")
    print(f"CSV output:                   {args.csv}")
    print(f"HDF5 output:                  {args.h5 if write_hdf5 else 'disabled'}")

    run_simulation(
        cfg=cfg,
        h5_file=args.h5,
        csv_file=args.csv,
        h5_tile_y=args.h5_tile_y,
        h5_tile_x=args.h5_tile_x,
        write_hdf5=write_hdf5,
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"CRITICAL ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
