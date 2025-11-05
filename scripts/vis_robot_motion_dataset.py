import argparse
import os
from pathlib import Path
from time import time
from typing import Iterable, List

from tqdm import tqdm


# Global state used only when running interactively with the viewer.
paused = False
motion_num = 0
motion_id = 0
current_motion_id = -1


def keyboard_callback(keycode: int) -> None:
    """Handle basic keyboard controls for the interactive viewer."""

    global paused, motion_id, motion_num

    key = chr(keycode)
    if key == " ":
        paused = not paused
    elif key == "[":
        motion_id = (motion_id - 1) % motion_num
    elif key == "]":
        motion_id = (motion_id + 1) % motion_num


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View or export robot motion datasets.")
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--robot_motion_folder", type=str, required=True)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, default="videos/example.mp4")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without launching the interactive viewer. Sets MUJOCO_GL=egl by default.",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default="videos",
        help="Root folder to mirror motion paths when saving per-motion videos in headless mode.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-render videos even if the output file already exists.",
    )
    return parser.parse_args()


def configure_headless_mode() -> None:
    # Ensure MuJoCo uses an offscreen backend before it is imported.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def collect_motion_files(root_dir: str) -> List[str]:
    motion_files: List[str] = []
    for root, _, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith(".pkl"):
                full_path = os.path.join(root, filename)
                motion_files.append(os.path.relpath(full_path, root_dir))
    motion_files.sort()
    return motion_files


def load_dataset(robot_motion_folder: str, motion_files: Iterable[str], video_root: str, force: bool) -> List[dict]:
    from general_motion_retargeting import load_robot_motion
    
    video_root_path = Path(video_root)

    dataset = []
    for motion_file in tqdm(motion_files, desc="Loading motions"):
        motion_path = os.path.join(robot_motion_folder, motion_file)
        motion_data = load_robot_motion(motion_path)
        (
            motion_tensor,
            motion_fps,
            motion_root_pos,
            motion_root_rot,
            motion_dof_pos,
            motion_local_body_pos,
            motion_link_body_list,
        ) = motion_data

        rel_motion_path = Path(motion_file).with_suffix(".mp4")
        output_path = video_root_path / rel_motion_path
        
        if output_path.exists() and not force:
            tqdm.write(f"Skipping existing {output_path}")
            continue

        dataset.append(
            {
                "motion_file": motion_file,
                "motion_data": motion_tensor,
                "motion_fps": motion_fps,
                "motion_root_pos": motion_root_pos,
                "motion_root_rot": motion_root_rot,
                "motion_dof_pos": motion_dof_pos,
                "motion_local_body_pos": motion_local_body_pos,
                "motion_link_body_list": motion_link_body_list,
            }
        )
    return dataset


def render_motion_to_video(
    robot_type: str,
    motion_data: dict,
    output_path: Path,
) -> None:
    from general_motion_retargeting import RobotMotionViewer

    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = RobotMotionViewer(
        robot_type=robot_type,
        motion_fps=motion_data["motion_fps"],
        camera_follow=False,
        record_video=True,
        video_path=str(output_path),
        keyboard_callback=None,
        launch_viewer=False,
        verbose=False,
    )

    for frame_idx in range(len(motion_data["motion_root_pos"])):
        env.step(
            motion_data["motion_root_pos"][frame_idx],
            motion_data["motion_root_rot"][frame_idx],
            motion_data["motion_dof_pos"][frame_idx],
            rate_limit=False,
        )
    env.close()


def run_headless_mode(robot_type: str, dataset: List[dict], video_root: str) -> None:
    video_root_path = Path(video_root)
    for motion_data in tqdm(dataset, desc="Rendering videos"):
        rel_motion_path = Path(motion_data["motion_file"]).with_suffix(".mp4")
        output_path = video_root_path / rel_motion_path
        render_motion_to_video(robot_type, motion_data, output_path)


def run_interactive_mode(
    robot_type: str,
    dataset: List[dict],
    record_video: bool,
    video_path: str,
    headless: bool,
) -> None:
    from general_motion_retargeting import RobotMotionViewer

    global motion_num, current_motion_id, motion_id

    motion_num = len(dataset)
    current_motion_id = motion_id
    current_motion = dataset[current_motion_id]

    env = RobotMotionViewer(
        robot_type=robot_type,
        motion_fps=current_motion["motion_fps"],
        camera_follow=False,
        record_video=record_video,
        video_path=video_path,
        keyboard_callback=None if headless else keyboard_callback,
        launch_viewer=not headless,
    )

    frame_idx = 0
    total_frames = 0
    start_time = time()

    try:
        while True:
            if current_motion_id != motion_id:
                current_motion_id = motion_id
                current_motion = dataset[current_motion_id]
                frame_idx = 0
                print(
                    f"Switched to motion {motion_id}: {current_motion['motion_file']}, "
                    f"fps: {current_motion['motion_fps']}, num_frames: {len(current_motion['motion_root_pos'])}"
                )

            if not paused:
                env.step(
                    current_motion["motion_root_pos"][frame_idx],
                    current_motion["motion_root_rot"][frame_idx],
                    current_motion["motion_dof_pos"][frame_idx],
                    rate_limit=True,
                )
                total_frames += 1

                # Report actual rendering throughput periodically.
                elapsed = time() - start_time
                if elapsed >= 5.0:
                    print(f"Rendering FPS: {total_frames / elapsed:.2f}")
                    total_frames = 0
                    start_time = time()

                frame_idx = (frame_idx + 1) % len(current_motion["motion_root_pos"])
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def main() -> None:
    args = parse_args()

    if args.headless:
        configure_headless_mode()

    robot_motion_folder = args.robot_motion_folder
    if not os.path.exists(robot_motion_folder):
        raise FileNotFoundError(f"Motion data dir {robot_motion_folder} does not exist.")

    motion_files = collect_motion_files(robot_motion_folder)
    if not motion_files:
        raise RuntimeError(f"No motion files (.pkl) found under {robot_motion_folder}.")

    print(f"Found {len(motion_files)} motion files in {robot_motion_folder}, loading...")
    dataset = load_dataset(robot_motion_folder, motion_files, args.video_root, args.force)
    print("Loading done.")

    if args.headless and not args.record_video:
        print("Headless mode requires --record_video to be enabled.")
        raise SystemExit(1)

    if args.headless:
        run_headless_mode(args.robot, dataset, args.video_root)
        return

    run_interactive_mode(
        robot_type=args.robot,
        dataset=dataset,
        record_video=args.record_video,
        video_path=args.video_path,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()