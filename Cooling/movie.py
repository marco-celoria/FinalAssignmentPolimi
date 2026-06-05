#!/usr/bin/env python3

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def compute_global_field_bounds(field_dataset, chunk_size=8):
    """
    Compute global min/max over /field without loading the whole dataset
    into memory at once.

    field_dataset shape is expected to be (nframes, ny, nx).
    """
    nframes = field_dataset.shape[0]
    global_min = float("inf")
    global_max = float("-inf")

    for start in range(0, nframes, chunk_size):
        stop = min(start + chunk_size, nframes)
        chunk = field_dataset[start:stop]   # shape: (chunk, ny, nx)

        cmin = np.min(chunk)
        cmax = np.max(chunk)

        if cmin < global_min:
            global_min = float(cmin)
        if cmax > global_max:
            global_max = float(cmax)

    return global_min, global_max


def main():
    parser = argparse.ArgumentParser(
        description="Animate cooling solver field evolution from HDF5."
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
        help="Show every Nth frame (default: 1)"
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
        help="Chunk size for global min/max computation (default: 8)"
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
        help="Transpose each frame before plotting (useful if axes appear swapped)"
    )
    parser.add_argument(
        "--origin",
        type=str,
        default="lower",
        choices=["lower", "upper"],
        help="imshow origin (default: lower)"
    )
    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.stride <= 0:
        raise ValueError("--stride must be > 0")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")

    file_path = Path(args.filename)
    if not file_path.exists():
        print(f"ERROR: file '{file_path}' not found.")
        return 1

    f = h5py.File(file_path, "r")

    try:
        if "/field" not in f:
            print("ERROR: dataset '/field' not found in HDF5 file.")
            return 2

        field_data = f["/field"]
        step_data = f["/step"] if "/step" in f else None

        if field_data.ndim != 3:
            print(f"ERROR: '/field' must have shape (nframes, ny, nx), got {field_data.shape}")
            return 2

        nframes, ny, nx = field_data.shape
        frame_indices = list(range(0, nframes, args.stride))

        print(f"Loaded field dataset: shape={field_data.shape}, dtype={field_data.dtype}")
        if step_data is not None:
            print(f"Loaded step dataset:  shape={step_data.shape}, dtype={step_data.dtype}")
        print(f"Displaying {len(frame_indices)} frame(s) out of {nframes} with stride={args.stride}")

        # Use manually provided bounds or compute global bounds lazily
        if args.vmin is None or args.vmax is None:
            print("Computing global field bounds...")
            auto_vmin, auto_vmax = compute_global_field_bounds(field_data, chunk_size=args.chunk_size)
            vmin = auto_vmin if args.vmin is None else args.vmin
            vmax = auto_vmax if args.vmax is None else args.vmax
        else:
            vmin = args.vmin
            vmax = args.vmax

        print(f"Color scale: vmin={vmin}, vmax={vmax}")

        fig, ax = plt.subplots(figsize=(8, 6))

        # Load first frame lazily
        first_frame = field_data[frame_indices[0]]
        if args.transpose:
            first_frame = first_frame.T

        im = ax.imshow(
            first_frame,
            origin=args.origin,
            cmap=args.cmap,
            vmin=vmin,
            vmax=vmax,
            animated=True,
            aspect="auto"
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Field value")

        ax.set_xlabel("x index")
        ax.set_ylabel("y index")

        title = ax.set_title("Cooling field evolution")

        def update(k):
            frame = frame_indices[k]
            arr = field_data[frame]
            if args.transpose:
                arr = arr.T

            im.set_data(arr)

            if step_data is not None:
                step_value = int(step_data[frame])
                title.set_text(f"Cooling field evolution — frame {frame} — step {step_value}")
            else:
                title.set_text(f"Cooling field evolution — frame {frame}")

            return (im, title)

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(frame_indices),
            interval=1000 // args.fps,
            blit=False
        )

        if args.save:
            print(f"Saving animation to '{args.save}'...")
            output = str(args.save).lower()

            if output.endswith(".gif"):
                ani.save(args.save, writer="pillow", fps=args.fps)
            else:
                # Typically MP4; requires ffmpeg installed
                ani.save(args.save, fps=args.fps, extra_args=["-vcodec", "libx264"])

            print("Save complete.")
        else:
            plt.show()

    finally:
        f.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
