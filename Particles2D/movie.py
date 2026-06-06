#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Tuple, Optional

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm


POS_DATASET = "/pos"
VEL_DATASET = "/vel"
SCREEN_DATASET = "/screen"
STEP_DATASET = "/step"


def compute_global_speed_bounds(
    vel_dataset: h5py.Dataset,
    chunk_size: int = 50,
) -> Tuple[float, float]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    if vel_dataset.ndim != 3:
        raise ValueError(
            f"{VEL_DATASET} must have shape (nframes, nparticles, ndim), "
            f"got {vel_dataset.shape}"
        )

    if vel_dataset.shape[2] < 2:
        raise ValueError(
            f"{VEL_DATASET} must contain at least vx/vy components, "
            f"got last dimension={vel_dataset.shape[2]}"
        )

    num_frames = vel_dataset.shape[0]

    if num_frames <= 0:
        raise ValueError(f"{VEL_DATASET} contains no frames")

    global_vmin = float("inf")
    global_vmax = float("-inf")

    for start in range(0, num_frames, chunk_size):
        stop = min(start + chunk_size, num_frames)

        chunk_vel = vel_dataset[start:stop, :, :2]
        chunk_speed = np.linalg.norm(chunk_vel, axis=2)

        local_min = float(np.min(chunk_speed))
        local_max = float(np.max(chunk_speed))

        global_vmin = min(global_vmin, local_min)
        global_vmax = max(global_vmax, local_max)

    if not np.isfinite(global_vmin) or not np.isfinite(global_vmax):
        raise ValueError("Non-finite speed bounds detected")

    if global_vmax <= global_vmin:
        # Avoid a degenerate color scale.
        global_vmax = global_vmin + 1.0e-30

    return global_vmin, global_vmax


def compute_screen_log_bounds(
    screen_dataset: h5py.Dataset,
    chunk_size: int = 10,
) -> Tuple[float, float]:
    """
    Compute positive min/max for LogNorm.
    Zeros are ignored for vmin.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    num_frames = screen_dataset.shape[0]

    positive_min = float("inf")
    global_max = 0.0

    for start in range(0, num_frames, chunk_size):
        stop = min(start + chunk_size, num_frames)

        chunk = screen_dataset[start:stop]
        local_max = float(np.max(chunk))

        if local_max > global_max:
            global_max = local_max

        positive = chunk[chunk > 0]

        if positive.size:
            local_min = float(np.min(positive))
            if local_min < positive_min:
                positive_min = local_min

    if global_max <= 0.0 or not np.isfinite(positive_min):
        return 1.0, 1.0

    return positive_min, global_max


def validate_datasets(
    pos_data: h5py.Dataset,
    vel_data: h5py.Dataset,
    screen_data: h5py.Dataset,
    step_data: Optional[h5py.Dataset],
) -> None:
    if pos_data.ndim != 3:
        raise ValueError(
            f"{POS_DATASET} must have shape (nframes, nparticles, 2), "
            f"got {pos_data.shape}"
        )

    if vel_data.ndim != 3:
        raise ValueError(
            f"{VEL_DATASET} must have shape (nframes, nparticles, 2), "
            f"got {vel_data.shape}"
        )

    if screen_data.ndim != 3:
        raise ValueError(
            f"{SCREEN_DATASET} must have shape (nframes, ny, nx), "
            f"got {screen_data.shape}"
        )

    if pos_data.shape[0] <= 0:
        raise ValueError("No frames found in HDF5 file")

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
        raise ValueError(
            f"{POS_DATASET} must contain at least x/y coordinates, "
            f"got last dimension={pos_data.shape[2]}"
        )

    if vel_data.shape[2] < 2:
        raise ValueError(
            f"{VEL_DATASET} must contain at least vx/vy components, "
            f"got last dimension={vel_data.shape[2]}"
        )

    if step_data is not None:
        if step_data.ndim != 1:
            raise ValueError(f"{STEP_DATASET} must be 1D, got shape {step_data.shape}")

        if step_data.shape[0] != pos_data.shape[0]:
            raise ValueError(
                "Step/frame count mismatch:\n"
                f"  {STEP_DATASET}: {step_data.shape[0]}\n"
                f"  {POS_DATASET}:  {pos_data.shape[0]}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Animate an N-body simulation stored in HDF5."
    )

    parser.add_argument(
        "filename",
        nargs="?",
        default="particles.h5",
        help="Path to the HDF5 file. Default: particles.h5",
    )

    parser.add_argument(
        "--save",
        type=str,
        help="Save animation to a file, e.g. output.mp4 or output.gif",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Playback frames per second. Default: 30",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Animate every Nth saved frame. Default: 1",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Chunk size used to compute global speed bounds. Default: 50",
    )

    parser.add_argument(
        "--particle-size",
        type=float,
        default=5.0,
        help="Scatter marker size. Default: 5.0",
    )

    parser.add_argument(
        "--grid-cmap",
        type=str,
        default="inferno",
        help="Colormap for background screen. Default: inferno",
    )

    parser.add_argument(
        "--particle-cmap",
        type=str,
        default="plasma",
        help="Colormap for particle speeds. Default: plasma",
    )

    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="X-axis limits. Default: 0 screen_nx",
    )

    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Y-axis limits. Default: 0 screen_ny",
    )

    parser.add_argument(
        "--log-screen",
        action="store_true",
        help="Use logarithmic color scaling for the background screen.",
    )

    parser.add_argument(
        "--hide-particles",
        action="store_true",
        help="Show only the screen image, without particle scatter overlay.",
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

    try:
        with h5py.File(file_path, "r") as f:
            for name in (POS_DATASET, VEL_DATASET, SCREEN_DATASET):
                if name not in f:
                    print(f"ERROR: dataset '{name}' not found in '{file_path}'.")
                    return 1

            pos_data = f[POS_DATASET]
            vel_data = f[VEL_DATASET]
            screen_data = f[SCREEN_DATASET]
            step_data = f[STEP_DATASET] if STEP_DATASET in f else None

            validate_datasets(pos_data, vel_data, screen_data, step_data)

            num_frames = pos_data.shape[0]
            num_particles = pos_data.shape[1]
            screen_ny = screen_data.shape[1]
            screen_nx = screen_data.shape[2]

            frame_indices = list(range(0, num_frames, args.stride))

            if not frame_indices:
                raise ValueError("No frames selected. Check --stride.")

            xlim = tuple(args.xlim) if args.xlim is not None else (0.0, float(screen_nx))
            ylim = tuple(args.ylim) if args.ylim is not None else (0.0, float(screen_ny))

            if xlim[1] <= xlim[0]:
                raise ValueError("--xlim must satisfy XMAX > XMIN")

            if ylim[1] <= ylim[0]:
                raise ValueError("--ylim must satisfy YMAX > YMIN")

            print(f"Loaded simulation from: {file_path}")
            print(f"Frames:      {num_frames}")
            print(f"Particles:   {num_particles}")
            print(f"Screen grid: {screen_nx} x {screen_ny}")
            print(f"Stride:      {args.stride}")
            print(f"X limits:    {xlim}")
            print(f"Y limits:    {ylim}")

            print("Computing global particle-speed bounds...")
            vmin, vmax = compute_global_speed_bounds(
                vel_data,
                chunk_size=args.chunk_size,
            )
            print(f"Speed range: [{vmin:.6g}, {vmax:.6g}]")

            screen_norm = None

            if args.log_screen:
                print("Computing logarithmic screen color bounds...")
                smin, smax = compute_screen_log_bounds(
                    screen_data,
                    chunk_size=max(1, args.chunk_size),
                )
                screen_norm = LogNorm(vmin=smin, vmax=smax)
                print(f"Screen positive range: [{smin:.6g}, {smax:.6g}]")

            fig, ax = plt.subplots(figsize=(9, 7))

            frame0 = frame_indices[0]

            grid0 = screen_data[frame0]
            pos0 = pos_data[frame0, :, :2]
            vel0 = vel_data[frame0, :, :2]
            speed0 = np.linalg.norm(vel0, axis=1)

            if step_data is not None:
                physical_step0 = int(step_data[frame0])
            else:
                physical_step0 = frame0

            im = ax.imshow(
                grid0,
                origin="lower",
                extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                cmap=args.grid_cmap,
                norm=screen_norm,
                animated=False,
                aspect="auto",
            )

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Screen intensity")

            scat = None
            particle_cbar = None

            if not args.hide_particles:
                scat = ax.scatter(
                    pos0[:, 0],
                    pos0[:, 1],
                    c=speed0,
                    cmap=args.particle_cmap,
                    s=args.particle_size,
                    vmin=vmin,
                    vmax=vmax,
                    edgecolors="none",
                )

                particle_cbar = fig.colorbar(
                    scat,
                    ax=ax,
                    fraction=0.046,
                    pad=0.10,
                )
                particle_cbar.set_label("Velocity magnitude")

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal", adjustable="box")

            title = ax.set_title(
                f"N-body simulation — frame {frame0}/{num_frames - 1}, step {physical_step0}"
            )

            def update(k: int):
                frame = frame_indices[k]

                grid = screen_data[frame]
                pos = pos_data[frame, :, :2]
                vel = vel_data[frame, :, :2]
                speed = np.linalg.norm(vel, axis=1)

                if step_data is not None:
                    physical_step = int(step_data[frame])
                else:
                    physical_step = frame

                im.set_data(grid)

                artists = [im, title]

                if scat is not None:
                    scat.set_offsets(pos)
                    scat.set_array(speed)
                    artists.append(scat)

                title.set_text(
                    f"N-body simulation — frame {frame}/{num_frames - 1}, step {physical_step}"
                )

                return artists

            ani = animation.FuncAnimation(
                fig,
                update,
                frames=len(frame_indices),
                interval=1000.0 / args.fps,
                blit=False,
            )

            plt.tight_layout()

            if args.save:
                print(f"Saving animation to '{args.save}'...")

                output = args.save.lower()

                if output.endswith(".gif"):
                    ani.save(args.save, writer="pillow", fps=args.fps)
                else:
                    ani.save(
                        args.save,
                        fps=args.fps,
                        extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"],
                    )

                print("Save complete.")
            else:
                plt.show()

    except OSError as exc:
        print(f"ERROR: failed to read HDF5 file: {exc}")
        return 3

    except Exception as exc:
        print(f"ERROR: {exc}")
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
