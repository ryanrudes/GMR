import argparse
import pickle
import os

import numpy as np

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

def count_pkl_files(folder):
    count = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for name in filenames:
            if name.endswith(".pkl"):
                count += 1
    return count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GMR pickle files to CSV (for beyondmimic)")
    parser.add_argument(
        "--input-dir", type=str, help="Path to the folder containing pickle files from GMR",
    )
    parser.add_argument(
        "--output-dir", type=str, help="Path to the output folder to save CSV files",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Target frame rate for downsampling (default: 30 fps)",
    )
    args = parser.parse_args()

    out_folder = args.output_dir
    os.makedirs(out_folder, exist_ok=True)
    
    total = count_pkl_files(args.input_dir)
    
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.completed}/{task.total}"),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    )

    with progress:
        task = progress.add_task("Converting pickle to CSV...", total=total)
        
        for dirpath, dirnames, filenames in os.walk(args.input_dir):
            for name in filenames:
                filepath = os.path.join(dirpath, name)

                if not name.endswith(".pkl"):
                    continue
        
                with open(filepath, "rb") as f:
                    motion_data = pickle.load(f)

                dof_pos = motion_data["dof_pos"]
                frame_rate = motion_data["fps"]            
                motion = np.zeros((dof_pos.shape[0], dof_pos.shape[1] + 7), dtype=np.float32)
                motion[:, :3] = motion_data["root_pos"]
                motion[:, 3:7] = motion_data["root_rot"]
                motion[:, 7:] = dof_pos

                if frame_rate > args.fps:
                    # downsample to target fps
                    downsample_factor = frame_rate / args.fps
                    indices = np.arange(0, motion.shape[0], downsample_factor).astype(int)
                    old_length = motion.shape[0]
                    motion = motion[indices]
                    print(f"Downsampled from {old_length} to {motion.shape[0]} frames")

                save_path = os.path.join(out_folder, os.path.relpath(filepath, args.input_dir)).replace(".pkl", ".csv")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                np.savetxt(
                    save_path,
                    motion,
                    delimiter=",",
                )
                
                progress.update(task, advance=1)