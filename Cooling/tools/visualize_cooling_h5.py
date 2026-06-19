#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import SymLogNorm


FIELD_DATASET = "/field"
STEP_DATASET = "/step"


# ============================================================
# VALIDATION
# ============================================================

def validate_field_dataset(field_data: h5py.Dataset) -> Tuple[int, int, int]:
    """
    Validate the /field dataset.

    Expected shape:
        /field: (nframes, ny, nx)

    Returns
    -------
    nframes, ny, nx
    """
    if field_data.ndim != 3:
        raise ValueError(
            f"{FIELD_DATASET} must have shape (nframes, ny, nx), "
            f"got {field_data.shape}"
        )

    nframes, ny, nx = field_data.shape

    if nframes <= 0:
        raise ValueError(f"{FIELD_DATASET} contains no frames")

    if ny <= 0 or nx <= 0:
        raise ValueError(
            f"{FIELD_DATASET} has invalid spatial shape: {(ny, nx)}"
        )

    return int(nframes), int(ny), int(nx)


def validate_step_dataset(
    step_data: Optional[h5py.Dataset],
    nframes: int,
) -> None:
    """
    Validate optional /step dataset.

    Expected shape:
        /step: (nframes,)
    """
    if step_data is None:
        return

    if step_data.ndim != 1:
        raise ValueError(
            f"{STEP_DATASET} must have shape (nframes,), "
            f"got {step_data.shape}"
        )

    if step_data.shape[0] != nframes:
        raise ValueError(
            f"{STEP_DATASET} length mismatch: expected {nframes}, "
            f"got {step_data.shape[0]}"
        )


def validate_args(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")

    if args.stride <= 0:
        raise ValueError("--stride must be > 0")

    if args.amplify <= 0.0:
        raise ValueError("--amplify must be > 0")

    if args.relative_floor <= 0.0:
        raise ValueError("--relative-floor must be > 0")

    if args.sample_frames <= 0:
        raise ValueError("--sample-frames must be > 0")

    pmin, pmax = args.percentile
    if not (0.0 <= pmin < pmax <= 100.0):
        raise ValueError("--percentile must satisfy 0 <= PMIN < PMAX <= 100")

    if args.figsize[0] <= 0.0 or args.figsize[1] <= 0.0:
        raise ValueError("--figsize values must be > 0")


# ============================================================
# FRAME AND NUMERICAL HELPERS
# ============================================================

def make_frame_indices(nframes: int, stride: int) -> List[int]:
    frame_indices = list(range(0, nframes, stride))

    if not frame_indices:
        raise ValueError("No frames selected. Check --stride.")

    return frame_indices


def choose_sampled_frames(
    frame_indices: Sequence[int],
    sample_frames: int,
) -> List[int]:
    """
    Select approximately equally spaced frames for robust statistics.
    """
    if len(frame_indices) <= sample_frames:
        return list(frame_indices)

    positions = np.linspace(0, len(frame_indices) - 1, sample_frames)
    return [frame_indices[int(round(pos))] for pos in positions]


def load_frame(
    field_data: h5py.Dataset,
    frame_index: int,
) -> np.ndarray:
    """
    Load one field frame as float64.

    Converting to float64 makes display transformations numerically stable,
    even if the HDF5 dataset is stored as float32 or integer.
    """
    return np.asarray(field_data[frame_index], dtype=np.float64)


def finite_values(frame: np.ndarray) -> np.ndarray:
    """
    Return finite values from a frame as a flat 1D array.
    """
    flat = frame.ravel()
    return flat[np.isfinite(flat)]


def downsample_values(
    values: np.ndarray,
    max_values: int = 500_000,
) -> np.ndarray:
    """
    Deterministically downsample a flat array to limit memory use.
    """
    if values.size <= max_values:
        return values

    stride = max(1, values.size // max_values)
    return values[::stride]


def robust_median(frame: np.ndarray) -> float:
    values = finite_values(frame)

    if values.size == 0:
        raise ValueError("Cannot compute median: frame contains no finite values")

    return float(np.median(values))


def compute_relative_floor(
    initial_frame: np.ndarray,
    relative_floor_fraction: float,
) -> float:
    """
    Compute a robust denominator floor for relative mode.

    This avoids huge relative values where the initial field is zero or tiny.

    The scale is estimated from the 95th percentile of abs(initial_frame).
    If the initial field is exactly zero everywhere, a fallback scale of 1.0
    is used.
    """
    values = finite_values(np.abs(initial_frame))

    if values.size == 0:
        reference_scale = 1.0
    else:
        reference_scale = float(np.percentile(values, 95.0))

        if not np.isfinite(reference_scale) or reference_scale <= 0.0:
            max_abs = float(np.max(values)) if values.size > 0 else 0.0
            reference_scale = max_abs if max_abs > 0.0 else 1.0

    return max(
        relative_floor_fraction * reference_scale,
        np.finfo(np.float64).eps,
    )


# ============================================================
# DISPLAY TRANSFORMS
# ============================================================

def make_display_frame(
    raw_frame: np.ndarray,
    initial_frame: np.ndarray,
    initial_median: float,
    mode: str,
    amplify: float,
    relative_floor_value: float,
) -> np.ndarray:
    """
    Convert physical field values into display values.

    Modes
    -----
    raw:
        Display the field directly.

    anomaly:
        Display amplify * (field - initial_field).

    relative:
        Display amplify * (field - initial_field) / denominator.
        The denominator has a robust floor to avoid explosions near zero.

    centered:
        Display amplify * (field - median(initial_field)).
    """
    if mode == "raw":
        return raw_frame

    if mode == "anomaly":
        return amplify * (raw_frame - initial_frame)

    if mode == "relative":
        denominator = np.maximum(np.abs(initial_frame), relative_floor_value)
        return amplify * (raw_frame - initial_frame) / denominator

    if mode == "centered":
        return amplify * (raw_frame - initial_median)

    raise ValueError(f"Unknown display mode: {mode}")


def colorbar_label(mode: str) -> str:
    labels = {
        "raw": "Field value",
        "anomaly": "Field change from initial",
        "relative": "Relative field change",
        "centered": "Centered field value",
    }

    return labels[mode]


def default_colormap(mode: str) -> str:
    if mode in ("anomaly", "relative", "centered"):
        return "coolwarm"

    return "magma"


def build_title(
    base_title: str,
    frame_index: int,
    total_frames: int,
    step_data: Optional[h5py.Dataset],
    mode: str,
) -> str:
    if step_data is not None:
        step = int(step_data[frame_index])
        return (
            f"{base_title} — {mode} — "
            f"frame {frame_index}/{total_frames - 1} — step {step}"
        )

    return f"{base_title} — {mode} — frame {frame_index}/{total_frames - 1}"


# ============================================================
# COLOR LIMITS
# ============================================================

def compute_color_bounds(
    field_data: h5py.Dataset,
    frame_indices: Sequence[int],
    mode: str,
    amplify: float,
    relative_floor_fraction: float,
    percentile: Tuple[float, float],
    sample_frames: int,
) -> Tuple[float, float]:
    """
    Compute robust global color bounds from sampled display frames.
    """
    sampled_frames = choose_sampled_frames(frame_indices, sample_frames)

    initial_frame = load_frame(field_data, frame_indices[0])
    initial_median = robust_median(initial_frame)
    relative_floor_value = compute_relative_floor(
        initial_frame,
        relative_floor_fraction,
    )

    sampled_values = []

    for frame_index in sampled_frames:
        raw_frame = load_frame(field_data, frame_index)

        display_frame = make_display_frame(
            raw_frame=raw_frame,
            initial_frame=initial_frame,
            initial_median=initial_median,
            mode=mode,
            amplify=amplify,
            relative_floor_value=relative_floor_value,
        )

        values = finite_values(display_frame)
        values = downsample_values(values)

        if values.size > 0:
            sampled_values.append(values)

    if not sampled_values:
        raise ValueError("No finite display values found for color bounds")

    all_values = np.concatenate(sampled_values)

    pmin, pmax = percentile
    vmin = float(np.percentile(all_values, pmin))
    vmax = float(np.percentile(all_values, pmax))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("Non-finite color bounds detected")

    if vmax <= vmin:
        eps = max(abs(vmin), 1.0) * 1.0e-12
        vmin -= eps
        vmax += eps

    if mode in ("anomaly", "relative", "centered"):
        max_abs = max(abs(vmin), abs(vmax))

        if max_abs <= 0.0 or not np.isfinite(max_abs):
            max_abs = 1.0

        vmin = -max_abs
        vmax = max_abs

    return vmin, vmax


def make_norm(
    vmin: float,
    vmax: float,
    symlog: bool,
) -> Optional[SymLogNorm]:
    """
    Build optional symmetric logarithmic normalization.
    """
    if not symlog:
        return None

    max_abs = max(abs(vmin), abs(vmax))

    if max_abs <= 0.0 or not np.isfinite(max_abs):
        max_abs = 1.0

    linthresh = max(max_abs * 1.0e-3, np.finfo(np.float64).eps)

    return SymLogNorm(
        linthresh=linthresh,
        vmin=-max_abs,
        vmax=max_abs,
        base=10,
    )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a simple animation of a cooling-solver field."
    )

    parser.add_argument(
        "filename",
        nargs="?",
        default="output/Cooling.h5",
        help="Path to HDF5 file. Default: output/Cooling.h5.",
    )

    parser.add_argument(
        "--save",
        type=str,
        help="Save animation to .mp4 or .gif. If omitted, show interactively.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Playback frames per second. Default: 20.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every Nth frame. Default: 1.",
    )

    parser.add_argument(
        "--mode",
        choices=["raw", "anomaly", "relative", "centered"],
        default="anomaly",
        help=(
            "Display mode. raw = field, anomaly = field - initial, "
            "relative = normalized change, centered = field - median(initial). "
            "Default: anomaly."
        ),
    )

    parser.add_argument(
        "--amplify",
        type=float,
        default=1.0,
        help="Multiply anomaly/relative/centered values. Default: 1.",
    )

    parser.add_argument(
        "--relative-floor",
        type=float,
        default=1.0e-6,
        help=(
            "Relative-mode denominator floor as a fraction of a robust "
            "initial-field scale. Default: 1e-6."
        ),
    )

    parser.add_argument(
        "--percentile",
        type=float,
        nargs=2,
        default=(1.0, 99.0),
        metavar=("PMIN", "PMAX"),
        help="Robust color percentiles. Default: 1 99.",
    )

    parser.add_argument(
        "--sample-frames",
        type=int,
        default=16,
        help="Number of frames sampled for color bounds. Default: 16.",
    )

    parser.add_argument(
        "--symlog",
        action="store_true",
        help="Use symmetric logarithmic color scaling.",
    )

    parser.add_argument(
        "--cmap",
        type=str,
        default=None,
        help="Matplotlib colormap. Default depends on mode.",
    )

    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(9.0, 7.0),
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches. Default: 9 7.",
    )

    parser.add_argument(
        "--title",
        type=str,
        default="Cooling field evolution",
        help="Base plot title.",
    )

    return parser.parse_args()


# ============================================================
# ANIMATION
# ============================================================

def create_animation(
    fig: plt.Figure,
    ax: plt.Axes,
    field_data: h5py.Dataset,
    step_data: Optional[h5py.Dataset],
    frame_indices: Sequence[int],
    *,
    mode: str,
    amplify: float,
    relative_floor_fraction: float,
    cmap: str,
    vmin: float,
    vmax: float,
    symlog: bool,
    fps: int,
    title_text: str,
) -> animation.FuncAnimation:
    first_frame = frame_indices[0]
    total_frames = field_data.shape[0]

    initial_frame = load_frame(field_data, first_frame)
    initial_median = robust_median(initial_frame)
    relative_floor_value = compute_relative_floor(
        initial_frame,
        relative_floor_fraction,
    )

    first_display = make_display_frame(
        raw_frame=initial_frame,
        initial_frame=initial_frame,
        initial_median=initial_median,
        mode=mode,
        amplify=amplify,
        relative_floor_value=relative_floor_value,
    )

    norm = make_norm(vmin, vmax, symlog)

    image = ax.imshow(
        first_display,
        origin="lower",
        cmap=cmap,
        vmin=None if norm is not None else vmin,
        vmax=None if norm is not None else vmax,
        norm=norm,
        interpolation="bilinear",
        aspect="auto",
        animated=True,
    )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label(mode))

    ax.set_xlabel("x index")
    ax.set_ylabel("y index")

    title = ax.set_title(
        build_title(
            base_title=title_text,
            frame_index=first_frame,
            total_frames=total_frames,
            step_data=step_data,
            mode=mode,
        )
    )

    def update(k: int):
        frame_index = frame_indices[k]
        raw_frame = load_frame(field_data, frame_index)

        display_frame = make_display_frame(
            raw_frame=raw_frame,
            initial_frame=initial_frame,
            initial_median=initial_median,
            mode=mode,
            amplify=amplify,
            relative_floor_value=relative_floor_value,
        )

        image.set_data(display_frame)

        title.set_text(
            build_title(
                base_title=title_text,
                frame_index=frame_index,
                total_frames=total_frames,
                step_data=step_data,
                mode=mode,
            )
        )

        return image, title

    return animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000.0 / fps,
        blit=False,
    )


def save_or_show(
    ani: animation.FuncAnimation,
    output_path: Optional[str],
    fps: int,
) -> None:
    if output_path is None:
        plt.tight_layout()
        plt.show()
        return

    print(f"Saving animation to '{output_path}'...")

    lower = output_path.lower()

    if lower.endswith(".gif"):
        ani.save(output_path, writer="pillow", fps=fps)
    else:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "Cannot save MP4 because ffmpeg was not found in PATH. "
                "Install ffmpeg, or save as .gif instead."
            )

        writer = animation.FFMpegWriter(
            fps=fps,
            codec="libx264",
            extra_args=["-pix_fmt", "yuv420p"],
        )
        ani.save(output_path, writer=writer)

    print("Save complete.")


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    try:
        args = parse_args()
        validate_args(args)

        file_path = Path(args.filename)

        if not file_path.exists():
            print(f"ERROR: file '{file_path}' not found.")
            return 1

        with h5py.File(file_path, "r") as h5:
            if FIELD_DATASET not in h5:
                print(f"ERROR: dataset '{FIELD_DATASET}' not found in HDF5 file.")
                return 2

            field_data = h5[FIELD_DATASET]
            step_data = h5[STEP_DATASET] if STEP_DATASET in h5 else None

            nframes, ny, nx = validate_field_dataset(field_data)
            validate_step_dataset(step_data, nframes)

            frame_indices = make_frame_indices(
                nframes=nframes,
                stride=args.stride,
            )

            cmap = args.cmap or default_colormap(args.mode)

            print(
                f"Loaded field dataset: shape={field_data.shape}, "
                f"dtype={field_data.dtype}"
            )

            if step_data is not None:
                print(
                    f"Loaded step dataset:  shape={step_data.shape}, "
                    f"dtype={step_data.dtype}"
                )

            print(f"Grid shape:           {ny} x {nx}")
            print(f"Frames displayed:     {len(frame_indices)} / {nframes}")
            print(f"Mode:                 {args.mode}")
            print(f"Amplification:        {args.amplify}")
            print(f"Relative floor:       {args.relative_floor}")
            print(f"Colormap:             {cmap}")
            print(f"SymLog scale:         {args.symlog}")

            print("Computing robust color bounds...")

            vmin, vmax = compute_color_bounds(
                field_data=field_data,
                frame_indices=frame_indices,
                mode=args.mode,
                amplify=args.amplify,
                relative_floor_fraction=args.relative_floor,
                percentile=tuple(args.percentile),
                sample_frames=args.sample_frames,
            )

            print(f"Color scale:          vmin={vmin:.8g}, vmax={vmax:.8g}")

            fig, ax = plt.subplots(figsize=tuple(args.figsize))

            ani = create_animation(
                fig=fig,
                ax=ax,
                field_data=field_data,
                step_data=step_data,
                frame_indices=frame_indices,
                mode=args.mode,
                amplify=args.amplify,
                relative_floor_fraction=args.relative_floor,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                symlog=args.symlog,
                fps=args.fps,
                title_text=args.title,
            )

            save_or_show(
                ani=ani,
                output_path=args.save,
                fps=args.fps,
            )

        return 0

    except OSError as exc:
        print(f"ERROR: failed to open or read HDF5 file: {exc}")
        return 3

    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 4

    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 5

    except Exception as exc:
        print(f"ERROR: unexpected failure: {exc}")
        return 6


if __name__ == "__main__":
    raise SystemExit(main())
