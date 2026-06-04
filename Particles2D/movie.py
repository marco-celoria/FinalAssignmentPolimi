import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from pathlib import Path

def compute_global_speed_bounds(vel_dataset, chunk_size=50):
    """
    Computes global min/max speeds by processing the data in chunks.
    This prevents Out-Of-Memory errors on massive simulation files.
    """
    num_frames = vel_dataset.shape[0]
    global_vmin, global_vmax = float('inf'), float('-inf')

    for start in range(0, num_frames, chunk_size):
        end = min(start + chunk_size, num_frames)
        chunk_vel = vel_dataset[start:end]
        
        # Calculate speeds for the current chunk
        speeds = np.linalg.norm(chunk_vel, axis=2)
        global_vmin = min(global_vmin, np.min(speeds))
        global_vmax = max(global_vmax, np.max(speeds))

    return global_vmin, global_vmax

def main():
    # 1. Replace sys.argv with argparse for a proper CLI interface
    parser = argparse.ArgumentParser(description="Animate N-Body Simulation from HDF5.")
    parser.add_argument("filename", nargs="?", default="build/particles.h5", help="Path to HDF5 file")
    parser.add_argument("--save", type=str, help="Save to video file (e.g., output.mp4)")
    parser.add_argument("--fps", type=int, default=30, help="Playback frames per second")
    args = parser.parse_args()

    file_path = Path(args.filename)
    if not file_path.exists():
        print(f"ERROR: File '{file_path}' not found.")
        return

    # 2. Keep the file open during animation to allow lazy-loading
    f = h5py.File(file_path, 'r')
    
    try:
        # Link to the datasets without pulling them into memory (no [:] slicing yet)
        pos_data = f['/pos']
        vel_data = f['/vel']
        grid_data = f['/screen']

        num_frames = pos_data.shape[0]
        print(f"Loaded simulation: {num_frames} frames, {pos_data.shape[1]} particles.")
        print("Calculating velocity bounds...")
        
        # Calculate vmin/vmax using the chunked helper function
        vmin, vmax = compute_global_speed_bounds(vel_data)

        fig, ax = plt.subplots(figsize=(8, 6))

        # Setup background grid for the first frame
        im = ax.imshow(grid_data[0], origin='lower', extent=[0, 1, 0, 1], cmap='inferno')

        # Setup initial particle scatter
        initial_pos = pos_data[0]
        initial_speed = np.linalg.norm(vel_data[0], axis=1)
        scat = ax.scatter(initial_pos[:, 0], initial_pos[:, 1], 
                          c=initial_speed, cmap='plasma', s=5, vmin=vmin, vmax=vmax)

        plt.colorbar(scat, label='Velocity Magnitude')

        def update(frame):
            # 3. Lazy load ONLY the data needed for the current frame
            im.set_data(grid_data[frame])
            scat.set_offsets(pos_data[frame])
            
            current_speed = np.linalg.norm(vel_data[frame], axis=1)
            scat.set_array(current_speed)
            
            return im, scat

        # 4. Tie playback speed to the --fps argument
        ani = animation.FuncAnimation(fig, update, frames=num_frames, 
                                      interval=1000 // args.fps, blit=True)

        # 5. Handle video exporting 
        if args.save:
            print(f"Saving animation to {args.save}...")
            # Note: Requires ffmpeg installed on your system
            ani.save(args.save, fps=args.fps, extra_args=['-vcodec', 'libx264'])
            print("Save complete.")
        else:
            plt.show()

    finally:
        # Guarantee the HDF5 file closes safely when the window is closed or script crashes
        f.close()

if __name__ == "__main__":
    main()
