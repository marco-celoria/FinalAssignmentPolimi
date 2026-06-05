#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


POS_DATASET = "/pos"
VEL_DATASET = "/vel"
SCREEN_DATASET = "/screen"


def compute_global_speed_bounds(vel_dataset: h5py.Dataset, chunk_size: int = 50) -> Tuple[float, float]:
    """
    Compute global min/max particle speed by processing the velocity dataset in chunks.

    Parameters
    ----------
    vel_dataset : h5py.Dataset
        Dataset with shape (nframes, nparticles, ndim).
    chunk_size : int
        Number of frames to process at once.

    Returns
    -------
    (vmin, vmax) : tuple of float
        Global minimum and maximum speed magnitude across all frames.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    if vel_dataset.ndim != 3:
        raise ValueError(f"{VEL_DATASET} must have shape (nframes, nparticles, ndim), got {vel_dataset.shape}")

    num_frames = vel_dataset.shape[0]
    global_vmin = float("inf")
    global_vmax = float("-inf")

    for start in range(0, num_frames, chunk_size):
        stop = min(start + chunk_size, num_frames)
        chunk_vel = vel_dataset[start:stop]          # shape: (chunk, nparticles, ndim)
        chunk_speed = np.linalg.norm(chunk_vel, axis=2)

        local_min = float(np.min(chunk_speed))
        local_max = float(np.max(chunk_speed))

        if local_min < global_vmin:
            global_vmin = local_min
        if local_max > global_vmax:
            global_vmax = local_max

    return global_vmin, global_vmax


def validate_datasets(pos_data: h5py.Dataset, vel_data: h5py.Dataset, screen_data: h5py.Dataset) -> None:
    """
    Validate dataset presence and shape compatibility.
    """
    if pos_data.ndim != 3:
        raise ValueError(f"{POS_DATASET} must have shape (nframes, nparticles, 2), got {pos_data.shape}")

    if vel_data.ndim != 3:
        raise ValueError(f"{VEL_DATASET} must have shape (nframes, nparticles, 2), got {vel_data.shape}")

    if screen_data.ndim != 3:
        raise ValueError(f"{SCREEN_DATASET} must have shape (nframes, ny, nx), got {screen_data.shape}")

    if pos_data.shape[0] != vel_data.shape[0] or pos_data.shape[0] != screen_data.shape[0]:
        raise ValueError(
            "Frame count mismatch:\n"
            f"  {POS_DATASET}:    {pos_data.shape[0]}\n"
            f"  {VEL_DATASET}:    {vel_data.shape[0]}\n"
            f"  {SCREEN_DATASET}: {screen_data.shape[0]}"
        )

    if pos_data.shape[1] != vel_data.shape[1]:
        raise ValueError(
            "Particle count mismatch:\n"
            f"  {POS_DATASET}: {pos_data.shape[1]}\n"
            f"  {VEL_DATASET}: {vel_data.shape[1]}"
        )

    if pos_data.shape[2] < 2:
        raise ValueError(f"{POS_DATASET} must contain at least x/y coordinates, got last dimension={pos_data.shape[2]}")

    if vel_data.shape[2] < 2:
        raise ValueError(f"{VEL_DATASET} must contain at least vx/vy components, got last dimension={vel_data.shape[2]}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Animate an N-body simulation stored in HDF5."
    )
    parser.add_argument(
        "filename",
        nargs="?",
        default="build/particles.h5",
        help="Path to the HDF5 file (default: build/particles.h5)"
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Save animation to a file (e.g. output.mp4 or output.gif)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Playback frames per second (default: 30)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Animate every Nth frame (default: 1)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Chunk size used to compute global speed bounds (default: 50)"
    )
    parser.add_argument(
        "--particle-size",
        type=float,
        default=5.0,
        help="Scatter marker size (default: 5.0)"
    )
    parser.add_argument(
        "--grid-cmap",
        type=str,
        default="inferno",
        help="Colormap for background grid (default: inferno)"
    )
    parser.add_argument(
        "--particle-cmap",
        type=str,
        default="plasma",
        help="Colormap for particle speeds (default: plasma)"
    )
    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        default=(0.0, 1.0),
        metavar=("XMIN", "XMAX"),
        help="X-axis limits (default: 0 1)"
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=(0.0, 1.0),
        metavar=("YMIN", "YMAX"),
        help="Y-axis limits (default: 0 1)"
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.fps <= 0:
        print("ERROR: --fps must be > 0")
        return 2
    if args.stride <= 0:
        print("ERROR: --stride must be > 0")
        return 2
    if args.chunk_size <= 0:
        print("ERROR: --chunk-size must be > 0")
        return 2
    if args.particle_size <= 0:
        print("ERROR: --particle-size must be > 0")
        return 2

    file_path = Path(args.filename)
    if not file_path.exists():
        print(f"ERROR: file '{file_path}' not found.")
        return 1

    f = h5py.File(file_path, "r")

    try:
        # Check required datasets exist
        for name in (POS_DATASET, VEL_DATASET, SCREEN_DATASET):
            if name not in f:
                print(f"ERROR: dataset '{name}' not found in '{file_path}'.")
                return 1

        pos_data = f[POS_DATASET]
        vel_data = f[VEL_DATASET]
        screen_data = f[SCREEN_DATASET]

        validate_datasets(pos_data, vel_data, screen_data)

        num_frames = pos_data.shape[0]
        num_particles = pos_data.shape[1]
        frame_indices = list(range(0, num_frames, args.stride))

        print(f"Loaded simulation from: {file_path}")
        print(f"Frames:     {num_frames}")
        print(f"Particles:  {num_particles}")
        print(f"Grid shape: {screen_data.shape[1]} x {screen_data.shape[2]}")
        print(f"Stride:     {args.stride}")
        print("Computing global particle-speed bounds...")

        vmin, vmax = compute_global_speed_bounds(vel_data, chunk_size=args.chunk_size)
        print(f"Speed range: [{vmin:.6g}, {vmax:.6g}]")

        fig, ax = plt.subplots(figsize=(8, 6))

        # Load first frame lazily
        frame0 = frame_indices[0]
        grid0 = screen_data[frame0]
        pos0 = pos_data[frame0]
        vel0 = vel_data[frame0]
        speed0 = np.linalg.norm(vel0, axis=1)

        # Background image
        im = ax.imshow(
            grid0,
            origin="lower",
            extent=[args.xlim[0], args.xlim[1], args.ylim[0], args.ylim[1]],
            cmap=args.grid_cmap,
            animated=False,
            aspect="auto"
        )

        # Particle overlay
        scat = ax.scatter(
            pos0[:, 0],
            pos0[:, 1],
            c=speed0,
            cmap=args.particle_cmap,
            s=args.particle_size,
            vmin=vmin,
            vmax=vmax,
            edgecolors="none"
        )

        cbar = fig.colorbar(scat, ax=ax)
        cbar.set_label("Velocity magnitude")

        ax.set_xlim(*args.xlim)
        ax.set_ylim(*args.ylim)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"N-Body Simulation — frame {frame0}/{num_frames - 1}")
        ax.set_aspect("equal", adjustable="box")

        title = ax.title

        def update(k: int):
            frame = frame_indices[k]

            # Lazy load only the current frame
            grid = screen_data[frame]
            pos = pos_data[frame]
            vel = vel_data[frame]
            speed = np.linalg.norm(vel, axis=1)

            im.set_data(grid)
            scat.set_offsets(pos[:, :2])
            scat.set_array(speed)
            title.set_text(f"N-Body Simulation — frame {frame}/{num_frames - 1}")

            return im, scat, title

        # blit=False is more robust across backends and with text/colorbar updates
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(frame_indices),
            interval=1000 // args.fps,
            blit=False
        )

        if args.save:
            print(f"Saving animation to '{args.save}'...")

            output = args.save.lower()
            if output.endswith(".gif"):
                ani.save(args.save, writer="pillow", fps=args.fps)
            else:
                # Usually MP4; requires ffmpeg installed
                ani.save(args.save, fps=args.fps, extra_args=["-vcodec", "libx264"])

            print("Save complete.")
        else:
            plt.tight_layout()
            plt.show()

    except OSError as e:
        print(f"ERROR: failed to read HDF5 file: {e}")
        return 3
    except Exception as e:
        print(f"ERROR: {e}")
        return 4
    finally:
        f.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

