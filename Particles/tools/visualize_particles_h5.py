#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Optional, Sequence

import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize


# ---------------------------------------------------------------------
# HDF5 dataset names
# ---------------------------------------------------------------------

POS_DATASET = "/pos"
VEL_DATASET = "/vel"
SCREEN_DATASET = "/screen"
STEP_DATASET = "/step"


# ---------------------------------------------------------------------
# Visualization defaults
# ---------------------------------------------------------------------

TITLE = "Mandelbrot-seeded N-body simulation"

FIGSIZE = (9.0, 7.0)
ASPECT = "equal"

SCREEN_CMAP = "inferno"
SCREEN_PERCENTILES = (1.0, 99.5)
SAMPLE_FRAMES = 32
MAX_PIXELS_PER_SAMPLED_FRAME = 500_000

PARTICLE_COLOR = "cyan"
PARTICLE_SIZE = 4.0
PARTICLE_ALPHA = 0.85

EXTENT_PADDING = 0.05


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def require_dataset(h5: h5py.File, name: str) -> h5py.Dataset:
    if name not in h5:
        raise ValueError(f"required dataset '{name}' not found in HDF5 file")
    return h5[name]


def validate_positions(pos: h5py.Dataset) -> tuple[int, int]:
    """
    Validate /pos.

    Expected shape:
        /pos: (nframes, nparticles, ndim)

    Only x/y coordinates are used, so ndim must be at least 2.
    """
    if pos.ndim != 3:
        raise ValueError(
            f"{POS_DATASET} must have shape (nframes, nparticles, ndim), "
            f"got {pos.shape}"
        )

    nframes, nparticles, ndim = pos.shape

    if nframes <= 0:
        raise ValueError(f"{POS_DATASET} contains no frames")

    if nparticles <= 0:
        raise ValueError(f"{POS_DATASET} contains no particles")

    if ndim < 2:
        raise ValueError(
            f"{POS_DATASET} must contain at least x/y coordinates, "
            f"got last dimension={ndim}"
        )

    return int(nframes), int(nparticles)


def validate_screen(screen: h5py.Dataset, expected_nframes: int) -> tuple[int, int]:
    """
    Validate /screen.

    Expected shape:
        /screen: (nframes, ny, nx)
    """
    if screen.ndim != 3:
        raise ValueError(
            f"{SCREEN_DATASET} must have shape (nframes, ny, nx), "
            f"got {screen.shape}"
        )

    nframes, ny, nx = screen.shape

    if nframes != expected_nframes:
        raise ValueError(
            "screen/frame count mismatch:\n"
            f"  {SCREEN_DATASET}: {nframes}\n"
            f"  expected:        {expected_nframes}"
        )

    if ny <= 0 or nx <= 0:
        raise ValueError(f"{SCREEN_DATASET} has invalid shape {screen.shape}")

    return int(ny), int(nx)


def validate_optional_velocity(
    vel: Optional[h5py.Dataset],
    expected_nframes: int,
    expected_nparticles: int,
) -> None:
    """
    Validate optional /vel if present.

    The animation does not use velocities, but validating them helps detect
    inconsistent simulation output.
    """
    if vel is None:
        return

    if vel.ndim != 3:
        raise ValueError(
            f"{VEL_DATASET} must have shape (nframes, nparticles, ndim), "
            f"got {vel.shape}"
        )

    nframes, nparticles, ndim = vel.shape

    if nframes != expected_nframes:
        raise ValueError(
            "velocity/frame count mismatch:\n"
            f"  {VEL_DATASET}: {nframes}\n"
            f"  expected:     {expected_nframes}"
        )

    if nparticles != expected_nparticles:
        raise ValueError(
            "velocity/particle count mismatch:\n"
            f"  {VEL_DATASET}: {nparticles}\n"
            f"  expected:     {expected_nparticles}"
        )

    if ndim < 2:
        raise ValueError(
            f"{VEL_DATASET} must contain at least vx/vy components, "
            f"got last dimension={ndim}"
        )


def validate_optional_steps(
    steps: Optional[h5py.Dataset],
    expected_nframes: int,
) -> None:
    """
    Validate optional /step.

    Expected shape:
        /step: (nframes,)
    """
    if steps is None:
        return

    if steps.ndim != 1:
        raise ValueError(
            f"{STEP_DATASET} must have shape (nframes,), got {steps.shape}"
        )

    if steps.shape[0] != expected_nframes:
        raise ValueError(
            "step/frame count mismatch:\n"
            f"  {STEP_DATASET}: {steps.shape[0]}\n"
            f"  expected:       {expected_nframes}"
        )


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise ValueError("--fps must be greater than zero")

    if args.stride <= 0:
        raise ValueError("--stride must be greater than zero")


# ---------------------------------------------------------------------
# Frame and data helpers
# ---------------------------------------------------------------------

def make_frame_indices(nframes: int, stride: int) -> list[int]:
    frames = list(range(0, nframes, stride))

    if not frames:
        raise ValueError("no frames selected; check --stride")

    return frames


def choose_sampled_frames(
    frame_indices: Sequence[int],
    sample_count: int,
) -> list[int]:
    """
    Select approximately equally spaced frames for cheap global statistics.
    """
    if sample_count <= 0:
        raise ValueError("sample_count must be greater than zero")

    if len(frame_indices) <= sample_count:
        return list(frame_indices)

    positions = np.linspace(0, len(frame_indices) - 1, sample_count)
    return [frame_indices[int(round(pos))] for pos in positions]


def load_positions(pos: h5py.Dataset, frame: int) -> np.ndarray:
    return pos[frame, :, :2]


def load_screen(screen: h5py.Dataset, frame: int) -> np.ndarray:
    return screen[frame]


def build_title(
    frame: int,
    nframes: int,
    steps: Optional[h5py.Dataset],
) -> str:
    if steps is None:
        return f"{TITLE} — frame {frame}/{nframes - 1}"

    return f"{TITLE} — frame {frame}/{nframes - 1} — step {int(steps[frame])}"


# ---------------------------------------------------------------------
# Extent and color scaling
# ---------------------------------------------------------------------

def read_screen_extent(screen: h5py.Dataset) -> Optional[tuple[float, float, float, float]]:
    """
    Read physical extent from /screen attributes if available.

    Expected attributes:
        xs, xe, ys, ye
    """
    required_attrs = ("xs", "xe", "ys", "ye")

    if not all(name in screen.attrs for name in required_attrs):
        return None

    xs = float(screen.attrs["xs"])
    xe = float(screen.attrs["xe"])
    ys = float(screen.attrs["ys"])
    ye = float(screen.attrs["ye"])

    if xe <= xs or ye <= ys:
        raise ValueError("invalid /screen extent attributes: require xe > xs and ye > ys")

    return xs, xe, ys, ye


def infer_extent_from_positions(
    pos: h5py.Dataset,
    frame_indices: Sequence[int],
) -> tuple[float, float, float, float]:
    """
    Infer plotting extent from sampled particle positions.
    """
    sampled_frames = choose_sampled_frames(frame_indices, SAMPLE_FRAMES)

    xmin = float("inf")
    xmax = float("-inf")
    ymin = float("inf")
    ymax = float("-inf")

    for frame in sampled_frames:
        xy = load_positions(pos, frame)

        finite = np.isfinite(xy).all(axis=1)
        if not np.any(finite):
            continue

        xy = xy[finite]

        xmin = min(xmin, float(np.min(xy[:, 0])))
        xmax = max(xmax, float(np.max(xy[:, 0])))
        ymin = min(ymin, float(np.min(xy[:, 1])))
        ymax = max(ymax, float(np.max(xy[:, 1])))

    if not all(np.isfinite(v) for v in (xmin, xmax, ymin, ymax)):
        raise ValueError("could not infer finite plotting extent from /pos")

    if xmax <= xmin:
        delta = max(abs(xmin), 1.0) * 1.0e-6
        xmin -= delta
        xmax += delta

    if ymax <= ymin:
        delta = max(abs(ymin), 1.0) * 1.0e-6
        ymin -= delta
        ymax += delta

    dx = xmax - xmin
    dy = ymax - ymin

    return (
        xmin - EXTENT_PADDING * dx,
        xmax + EXTENT_PADDING * dx,
        ymin - EXTENT_PADDING * dy,
        ymax + EXTENT_PADDING * dy,
    )


def determine_extent(
    pos: h5py.Dataset,
    screen: h5py.Dataset,
    frame_indices: Sequence[int],
) -> tuple[float, float, float, float]:
    """
    Determine plotting extent.

    Priority:
      1. /screen attributes xs, xe, ys, ye
      2. inferred particle-position extent
    """
    extent = read_screen_extent(screen)

    if extent is not None:
        return extent

    return infer_extent_from_positions(pos, frame_indices)


def compute_screen_bounds(
    screen: h5py.Dataset,
    frame_indices: Sequence[int],
    *,
    log_scale: bool,
) -> tuple[float, float]:
    """
    Compute robust color bounds for /screen using fixed percentiles.

    For log-scale rendering, non-positive values are ignored because LogNorm
    requires strictly positive values.
    """
    pmin, pmax = SCREEN_PERCENTILES
    sampled_frames = choose_sampled_frames(frame_indices, SAMPLE_FRAMES)

    sampled_values: list[np.ndarray] = []

    for frame in sampled_frames:
        values = np.ravel(load_screen(screen, frame))

        if values.size > MAX_PIXELS_PER_SAMPLED_FRAME:
            step = max(1, values.size // MAX_PIXELS_PER_SAMPLED_FRAME)
            values = values[::step]

        values = values[np.isfinite(values)]

        if log_scale:
            values = values[values > 0.0]

        if values.size > 0:
            sampled_values.append(values)

    if not sampled_values:
        if log_scale:
            return 1.0, 10.0
        raise ValueError(f"no finite values found in {SCREEN_DATASET}")

    values = np.concatenate(sampled_values)

    vmin = float(np.percentile(values, pmin))
    vmax = float(np.percentile(values, pmax))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("non-finite /screen color bounds detected")

    if log_scale:
        vmin = max(vmin, np.finfo(np.float64).tiny)

    if vmax <= vmin:
        if log_scale:
            vmax = vmin * 10.0
        else:
            eps = max(abs(vmin), 1.0) * 1.0e-12
            vmin -= eps
            vmax += eps

    return vmin, vmax


# ---------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------

def create_animation(
    fig: plt.Figure,
    ax: plt.Axes,
    pos: h5py.Dataset,
    screen: h5py.Dataset,
    steps: Optional[h5py.Dataset],
    frame_indices: Sequence[int],
    extent: tuple[float, float, float, float],
    screen_bounds: tuple[float, float],
    *,
    fps: int,
    log_scale: bool,
) -> animation.FuncAnimation:
    first_frame = frame_indices[0]
    nframes = int(pos.shape[0])

    screen0 = load_screen(screen, first_frame)
    pos0 = load_positions(pos, first_frame)

    vmin, vmax = screen_bounds

    if log_scale:
        norm = LogNorm(vmin=vmin, vmax=vmax)
        colorbar_label = "Weighted particle deposition, log scale"
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
        colorbar_label = "Weighted particle deposition"

    image = ax.imshow(
        screen0,
        origin="lower",
        extent=extent,
        cmap=SCREEN_CMAP,
        norm=norm,
        interpolation="bilinear",
        animated=True,
    )

    scatter = ax.scatter(
        pos0[:, 0],
        pos0[:, 1],
        color=PARTICLE_COLOR,
        s=PARTICLE_SIZE,
        alpha=PARTICLE_ALPHA,
        edgecolors="none",
        animated=True,
    )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect(ASPECT, adjustable="box")

    title = ax.set_title(build_title(first_frame, nframes, steps))

    def update(frame_number: int):
        frame = frame_indices[frame_number]

        image.set_data(load_screen(screen, frame))
        scatter.set_offsets(load_positions(pos, frame))
        title.set_text(build_title(frame, nframes, steps))

        return image, scatter, title

    return animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000.0 / fps,
        blit=False,
    )


def save_or_show(
    ani: animation.FuncAnimation,
    output: Optional[Path],
    fps: int,
) -> None:
    if output is None:
        plt.tight_layout()
        plt.show()
        return

    print(f"Saving animation to '{output}'...")

    suffix = output.suffix.lower()

    if suffix == ".gif":
        ani.save(output, writer="pillow", fps=fps)
    elif suffix == ".mp4":
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "cannot save MP4 because ffmpeg was not found in PATH; "
                "install ffmpeg or save as .gif instead"
            )

        writer = animation.FFMpegWriter(
            fps=fps,
            codec="libx264",
            extra_args=["-pix_fmt", "yuv420p"],
        )
        ani.save(output, writer=writer)
    else:
        raise ValueError("output file must have extension .gif or .mp4")

    print("Save complete.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an animation from N-body HDF5 output."
    )

    parser.add_argument(
        "filename",
        nargs="?",
        default="particles.h5",
        help="input HDF5 file, default: particles.h5",
    )

    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="output animation path, supported extensions: .gif, .mp4",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="animation frames per second, default: 30",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="use every Nth saved frame, default: 1",
    )

    parser.add_argument(
        "--log-screen",
        action="store_true",
        help="use logarithmic color scaling for /screen",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    validate_cli_args(args)

    input_path = Path(args.filename)

    if not input_path.exists():
        raise FileNotFoundError(f"file '{input_path}' not found")

    with h5py.File(input_path, "r") as h5:
        pos = require_dataset(h5, POS_DATASET)
        screen = require_dataset(h5, SCREEN_DATASET)

        vel = h5[VEL_DATASET] if VEL_DATASET in h5 else None
        steps = h5[STEP_DATASET] if STEP_DATASET in h5 else None

        nframes, nparticles = validate_positions(pos)
        screen_ny, screen_nx = validate_screen(screen, nframes)
        validate_optional_velocity(vel, nframes, nparticles)
        validate_optional_steps(steps, nframes)

        frame_indices = make_frame_indices(nframes, args.stride)
        extent = determine_extent(pos, screen, frame_indices)

        print(f"Loaded file:        {input_path}")
        print(f"Frames displayed:   {len(frame_indices)} / {nframes}")
        print(f"Particles:          {nparticles}")
        print(f"Screen grid:        {screen_nx} x {screen_ny}")
        print(f"Position dataset:   shape={pos.shape}, dtype={pos.dtype}")
        print(f"Screen dataset:     shape={screen.shape}, dtype={screen.dtype}")

        if vel is not None:
            print(f"Velocity dataset:   shape={vel.shape}, dtype={vel.dtype}")
        else:
            print("Velocity dataset:   not present")

        if steps is not None:
            print(f"Step dataset:       shape={steps.shape}, dtype={steps.dtype}")

        print(
            "Extent:             "
            f"x=[{extent[0]}, {extent[1]}], "
            f"y=[{extent[2]}, {extent[3]}]"
        )

        print("Computing /screen color bounds...")

        screen_bounds = compute_screen_bounds(
            screen,
            frame_indices,
            log_scale=args.log_screen,
        )

        print(
            "Screen color scale: "
            f"vmin={screen_bounds[0]:.8g}, vmax={screen_bounds[1]:.8g}"
        )
        print(f"Log screen:         {args.log_screen}")

        fig, ax = plt.subplots(figsize=FIGSIZE)

        ani = create_animation(
            fig,
            ax,
            pos,
            screen,
            steps,
            frame_indices,
            extent,
            screen_bounds,
            fps=args.fps,
            log_scale=args.log_screen,
        )

        save_or_show(ani, args.save, args.fps)


def main() -> int:
    try:
        run(parse_args())
        return 0

    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    except OSError as exc:
        print(f"ERROR: failed to open or read HDF5 file: {exc}")
        return 2

    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 3

    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 4

    except Exception as exc:
        print(f"ERROR: unexpected failure: {exc}")
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
