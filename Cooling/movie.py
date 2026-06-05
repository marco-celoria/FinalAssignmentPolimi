#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


FIELD_DATASET = "/field"
STEP_DATASET = "/step"


def compute_global_field_bounds(
    field_dataset: h5py.Dataset,
    chunk_size: int = 8
) -> Tuple[float, float]:
    """
    Compute global min/max over the field dataset without loading
    the entire dataset into memory.

    Parameters
    ----------
    field_dataset : h5py.Dataset
        Dataset with shape (nframes, ny, nx).
    chunk_size : int
        Number of frames processed at once.

    Returns
    -------
    (vmin, vmax) : tuple of float
        Global minimum and maximum field values.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    if field_dataset.ndim != 3:
        raise ValueError(
            f"{FIELD_DATASET} must have shape (nframes, ny, nx), got {field_dataset.shape}"
        )

    nframes = field_dataset.shape[0]
    global_min = float("inf")
    global_max = float("-inf")

    for start in range(0, nframes, chunk_size):
        stop = min(start + chunk_size, nframes)
        chunk = field_dataset[start:stop]

        chunk_min = float(np.min(chunk))
        chunk_max = float(np.max(chunk))

        if chunk_min < global_min:
            global_min = chunk_min
        if chunk_max > global_max:
            global_max = chunk_max

    return global_min, global_max


def validate_field_dataset(field_data: h5py.Dataset) -> Tuple[int, int, int]:
    """
    Validate the /field dataset.

    Parameters
    ----------
    field_data : h5py.Dataset
        Dataset expected to have shape (nframes, ny, nx).

    Returns
    -------
    (nframes, ny, nx) : tuple of int
        Parsed dataset dimensions.
    """
    if field_data.ndim != 3:
        raise ValueError(
            f"{FIELD_DATASET} must have shape (nframes, ny, nx), got {field_data.shape}"
        )

    nframes, ny, nx = field_data.shape

    if nframes <= 0:
        raise ValueError(f"{FIELD_DATASET} contains no frames")
    if ny <= 0 or nx <= 0:
        raise ValueError(f"{FIELD_DATASET} has invalid spatial shape: {(ny, nx)}")

    return nframes, ny, nx


def validate_step_dataset(
    step_data: Optional[h5py.Dataset],
    nframes: int
) -> None:
    """
    Validate the optional /step dataset.

    Parameters
    ----------
    step_data : h5py.Dataset or None
        Optional dataset expected to have shape (nframes,).
    nframes : int
        Number of frames from /field.
    """
    if step_data is None:
        return

    if step_data.ndim != 1:
        raise ValueError(
            f"{STEP_DATASET} must have shape (nframes,), got {step_data.shape}"
        )

    if step_data.shape[0] != nframes:
        raise ValueError(
            f"{STEP_DATASET} length mismatch: expected {nframes}, got {step_data.shape[0]}"
        )


def parse_args() -> argparse.Namespace:
    """
    Build and parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Animate cooling solver field evolution from an HDF5 file."
    )
    parser.add_argument(
        "filename",
        nargs="?",
        default="cooling.h5",
        help="Path to HDF5 file (default: cooling.h5)"
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Save animation to file (e.g. cooling.mp4 or cooling.gif)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Playback frames per second (default: 20)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Display every Nth frame (default: 1)"
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="inferno",
        help="Matplotlib colormap (default: inferno)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8,
        help="Chunk size used for global min/max computation (default: 8)"
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Manual lower color bound (default: compute globally)"
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Manual upper color bound (default: compute globally)"
    )
    parser.add_argument(
        "--transpose",
        action="store_true",
        help="Transpose each frame before plotting"
    )
    parser.add_argument(
        "--origin",
        type=str,
        default="lower",
        choices=["lower", "upper"],
        help="imshow origin (default: lower)"
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(8.0, 6.0),
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches (default: 8 6)"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Cooling field evolution",
        help="Base plot title"
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate CLI arguments.
    """
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.stride <= 0:
        raise ValueError("--stride must be > 0")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")
    if args.figsize[0] <= 0 or args.figsize[1] <= 0:
        raise ValueError("--figsize values must be > 0")
    if args.vmin is not None and args.vmax is not None and args.vmin > args.vmax:
        raise ValueError("--vmin must be <= --vmax")


def resolve_color_bounds(
    field_data: h5py.Dataset,
    user_vmin: Optional[float],
    user_vmax: Optional[float],
    chunk_size: int
) -> Tuple[float, float]:
    """
    Resolve color scale bounds from user input or dataset-wide statistics.
    """
    if user_vmin is not None and user_vmax is not None:
        return user_vmin, user_vmax

    print("Computing global field bounds...")
    auto_vmin, auto_vmax = compute_global_field_bounds(field_data, chunk_size=chunk_size)

    vmin = auto_vmin if user_vmin is None else user_vmin
    vmax = auto_vmax if user_vmax is None else user_vmax

    return vmin, vmax


def maybe_transpose(frame: np.ndarray, transpose: bool) -> np.ndarray:
    """
    Return frame transposed if requested.
    """
    return frame.T if transpose else frame


def load_frame(
    field_data: h5py.Dataset,
    frame_index: int,
    transpose: bool
) -> np.ndarray:
    """
    Load a single frame from /field, applying transpose if requested.
    """
    frame = field_data[frame_index]
    return maybe_transpose(frame, transpose)


def build_title(
    base_title: str,
    frame_index: int,
    total_frames: int,
    step_data: Optional[h5py.Dataset]
) -> str:
    """
    Build the displayed title for a frame.
    """
    if step_data is not None:
        step_value = int(step_data[frame_index])
        return f"{base_title} — frame {frame_index}/{total_frames - 1} — step {step_value}"
    return f"{base_title} — frame {frame_index}/{total_frames - 1}"


def create_animation(
    fig: plt.Figure,
    ax: plt.Axes,
    field_data: h5py.Dataset,
    step_data: Optional[h5py.Dataset],
    frame_indices: Sequence[int],
    *,
    cmap: str,
    origin: str,
    transpose: bool,
    vmin: float,
    vmax: float,
    fps: int,
    title_text: str,
):
    """
    Create the matplotlib animation objects.

    Returns
    -------
    ani, im, title
        Animation object, image artist, and title artist.
    """
    first_frame_index = frame_indices[0]
    first_frame = load_frame(field_data, first_frame_index, transpose=transpose)

    im = ax.imshow(
        first_frame,
        origin=origin,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        animated=True,
        aspect="auto"
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Field value")

    ax.set_xlabel("x index" if not transpose else "y index")
    ax.set_ylabel("y index" if not transpose else "x index")

    title = ax.set_title(
        build_title(title_text, first_frame_index, field_data.shape[0], step_data)
    )

    def update(k: int):
        frame_index = frame_indices[k]
        frame = load_frame(field_data, frame_index, transpose=transpose)
        im.set_data(frame)
        title.set_text(build_title(title_text, frame_index, field_data.shape[0], step_data))
        return im, title

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000 // fps,
        blit=False
    )

    return ani, im, title


def save_or_show_animation(
    ani: animation.FuncAnimation,
    output_path: Optional[str],
    fps: int
) -> None:
    """
    Save or show the animation depending on user options.
    """
    if output_path:
        print(f"Saving animation to '{output_path}'...")
        lower = output_path.lower()

        if lower.endswith(".gif"):
            ani.save(output_path, writer="pillow", fps=fps)
        else:
            # Usually MP4; requires ffmpeg to be installed
            ani.save(output_path, fps=fps, extra_args=["-vcodec", "libx264"])

        print("Save complete.")
    else:
        plt.tight_layout()
        plt.show()


def main() -> int:
    try:
        args = parse_args()
        validate_args(args)

        file_path = Path(args.filename)
        if not file_path.exists():
            print(f"ERROR: file '{file_path}' not found.")
            return 1

        with h5py.File(file_path, "r") as f:
            if FIELD_DATASET not in f:
                print(f"ERROR: dataset '{FIELD_DATASET}' not found in HDF5 file.")
                return 2

            field_data = f[FIELD_DATASET]
            step_data = f[STEP_DATASET] if STEP_DATASET in f else None

            nframes, ny, nx = validate_field_dataset(field_data)
            validate_step_dataset(step_data, nframes)

            frame_indices = list(range(0, nframes, args.stride))
            if not frame_indices:
                print("ERROR: no frames selected. Check --stride.")
                return 2

            print(f"Loaded field dataset: shape={field_data.shape}, dtype={field_data.dtype}")
            if step_data is not None:
                print(f"Loaded step dataset:  shape={step_data.shape}, dtype={step_data.dtype}")
            print(f"Grid shape: {ny} x {nx}")
            print(f"Displaying {len(frame_indices)} frame(s) out of {nframes} with stride={args.stride}")

            vmin, vmax = resolve_color_bounds(
                field_data,
                user_vmin=args.vmin,
                user_vmax=args.vmax,
                chunk_size=args.chunk_size
            )
            print(f"Color scale: vmin={vmin}, vmax={vmax}")

            fig, ax = plt.subplots(figsize=tuple(args.figsize))

            ani, _, _ = create_animation(
                fig,
                ax,
                field_data,
                step_data,
                frame_indices,
                cmap=args.cmap,
                origin=args.origin,
                transpose=args.transpose,
                vmin=vmin,
                vmax=vmax,
                fps=args.fps,
                title_text=args.title,
            )

            save_or_show_animation(ani, args.save, args.fps)

        return 0

    except OSError as e:
        print(f"ERROR: failed to open or read HDF5 file: {e}")
        return 3
    except ValueError as e:
        print(f"ERROR: {e}")
        return 4
    except Exception as e:
        print(f"ERROR: unexpected failure: {e}")
        return 5


if __name__ == "__main__":
    raise SystemExit(main())

