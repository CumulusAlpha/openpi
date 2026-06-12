#!/usr/bin/env python3
"""Convert ACT/Aloha-style HDF5 episodes to a LeRobot dataset.

Default input layout expected by this script:

    /home/arx/act/datasets/episode_0.hdf5
    /home/arx/act/datasets/episode_1.hdf5
    ...

Each HDF5 file is expected to contain:

    /observations/qpos          (T, 14)
    /observations/middle_image  (T, N) JPEG/PNG bytes or (T, H, W, C)
    /observations/right_image   (T, N) JPEG/PNG bytes or (T, H, W, C)
    /action                     (T, 14)

Example:

    .venv/bin/python utils/convert_act_hdf5_to_lerobot.py \
        raw_dir=/home/arx/act/datasets \
        repo_id=local/act_aloha \
        task="put the object into the container"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
from typing import Iterable

import cv2
import h5py
import hydra
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
import numpy as np
from omegaconf import DictConfig
import tqdm


DEFAULT_IMAGE_MAP = {
    "middle_image": "observation.images.top",
    "right_image": "observation.images.right_wrist",
}

MOTOR_NAMES = [
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
    "left_gripper_pos",
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
    "right_gripper_pos",
]


@dataclass(frozen=True)
class EpisodeInfo:
    path: Path
    num_frames: int
    state_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    image_shapes: dict[str, tuple[int, int, int]]


def natural_episode_key(path: Path) -> tuple[int, str]:
    match = re.search(r"episode_(\d+)\.hdf5$", path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**12, path.name


def find_hdf5_files(raw_dir: Path) -> list[Path]:
    files = sorted(raw_dir.glob("episode_*.hdf5"), key=natural_episode_key)
    if not files:
        raise FileNotFoundError(f"No episode_*.hdf5 files found under {raw_dir}")
    return files


def decode_image(value: np.ndarray) -> np.ndarray:
    """Return an RGB HWC uint8 image from compressed bytes or an existing image array."""
    arr = np.asarray(value)
    if arr.ndim == 1:
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not decode compressed image with shape {arr.shape}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if arr.ndim == 3:
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    raise ValueError(f"Unsupported image shape: {arr.shape}")


def read_attr_string(attrs: h5py.AttributeManager, key: str, default: str) -> str:
    value = attrs.get(key, default)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def parse_image_map(values: Iterable[str]) -> dict[str, str]:
    image_map = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Image map entries must be raw_key=lerobot_key, got: {value}")
        raw_key, lerobot_key = value.split("=", 1)
        raw_key = raw_key.strip()
        lerobot_key = lerobot_key.strip()
        if not raw_key or not lerobot_key:
            raise ValueError(f"Invalid image map entry: {value}")
        image_map[raw_key] = lerobot_key
    return image_map


def inspect_episode(path: Path, image_map: dict[str, str]) -> EpisodeInfo:
    with h5py.File(path, "r") as episode:
        qpos = episode["/observations/qpos"]
        action = episode["/action"]
        image_shapes = {}
        for raw_key in image_map:
            dataset = episode[f"/observations/{raw_key}"]
            image_shapes[raw_key] = tuple(decode_image(dataset[0]).shape)

        return EpisodeInfo(
            path=path,
            num_frames=int(qpos.shape[0]),
            state_shape=tuple(qpos.shape[1:]),
            action_shape=tuple(action.shape[1:]),
            image_shapes=image_shapes,
        )


def validate_episode(path: Path, image_map: dict[str, str], expected_state_dim: int, expected_action_dim: int) -> None:
    required = ["/observations/qpos", "/action", *(f"/observations/{key}" for key in image_map)]
    with h5py.File(path, "r") as episode:
        missing = [key for key in required if key not in episode]
        if missing:
            raise KeyError(f"{path} is missing required datasets: {missing}")

        length = episode["/observations/qpos"].shape[0]
        if episode["/action"].shape[0] != length:
            raise ValueError(f"{path}: action length does not match qpos length")

        if episode["/observations/qpos"].shape[1] != expected_state_dim:
            raise ValueError(f"{path}: expected state dim {expected_state_dim}, got {episode['/observations/qpos'].shape}")
        if episode["/action"].shape[1] != expected_action_dim:
            raise ValueError(f"{path}: expected action dim {expected_action_dim}, got {episode['/action'].shape}")

        for raw_key in image_map:
            if episode[f"/observations/{raw_key}"].shape[0] != length:
                raise ValueError(f"{path}: {raw_key} length does not match qpos length")


def make_features(
    image_map: dict[str, str],
    image_shapes: dict[str, tuple[int, int, int]],
    state_dim: int,
    action_dim: int,
) -> dict:
    state_names = MOTOR_NAMES if state_dim == len(MOTOR_NAMES) else [f"state_{i}" for i in range(state_dim)]
    action_names = MOTOR_NAMES if action_dim == len(MOTOR_NAMES) else [f"action_{i}" for i in range(action_dim)]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [state_names],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [action_names],
        },
    }

    for raw_key, lerobot_key in image_map.items():
        height, width, channels = image_shapes[raw_key]
        features[lerobot_key] = {
            "dtype": "image",
            "shape": (height, width, channels),
            "names": ["height", "width", "channel"],
        }

    return features


def convert(args: DictConfig) -> Path:
    raw_dir = Path(args.raw_dir).expanduser()
    root = Path(args.root).expanduser() if args.root else HF_LEROBOT_HOME / args.repo_id
    output_path = root
    image_map = parse_image_map(args.image_map)
    hdf5_files = find_hdf5_files(raw_dir)
    if args.max_episodes is not None:
        hdf5_files = hdf5_files[: args.max_episodes]

    first_info = inspect_episode(hdf5_files[0], image_map)
    state_dim = first_info.state_shape[0]
    action_dim = first_info.action_shape[0]

    with h5py.File(hdf5_files[0], "r") as first_episode:
        fps = int(round(float(first_episode.attrs.get("frame_rate", args.fps))))
        task_from_attrs = read_attr_string(first_episode.attrs, "task", "").strip()

    task = args.task if args.task is not None else task_from_attrs or args.default_task

    infos = [first_info]
    for path in hdf5_files[1:]:
        validate_episode(path, image_map, state_dim, action_dim)
        if args.dry_run:
            infos.append(inspect_episode(path, image_map))

    print(f"Found {len(hdf5_files)} episodes under {raw_dir}")
    print(f"FPS: {fps}")
    print(f"Task: {task!r}")
    print(f"State dim: {state_dim}, action dim: {action_dim}")
    print(f"Images: {image_map}")
    print(f"First episode frames: {first_info.num_frames}, image shapes: {first_info.image_shapes}")

    if args.dry_run:
        total_frames = sum(info.num_frames for info in infos)
        print(f"Dry run only. Scanned {len(infos)} episodes, total scanned frames: {total_frames}")
        return output_path

    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_path} already exists. Set overwrite=true to replace it.")
        shutil.rmtree(output_path)
    features = make_features(
        image_map=image_map,
        image_shapes=first_info.image_shapes,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=root,
        fps=fps,
        robot_type=args.robot_type,
        features=features,
        use_videos=not args.no_videos,
        tolerance_s=args.tolerance_s,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        video_backend=args.video_backend,
    )

    for path in tqdm.tqdm(hdf5_files, desc="episodes"):
        with h5py.File(path, "r") as episode:
            qpos = episode["/observations/qpos"]
            action = episode["/action"]
            num_frames = qpos.shape[0]

            for i in range(num_frames):
                frame = {
                    "observation.state": qpos[i].astype(np.float32),
                    "action": action[i].astype(np.float32),
                    "task": task,
                }

                for raw_key, lerobot_key in image_map.items():
                    frame[lerobot_key] = decode_image(episode[f"/observations/{raw_key}"][i])

                dataset.add_frame(frame)

        dataset.save_episode()

    if hasattr(dataset, "consolidate"):
        dataset.consolidate()
    print(f"Saved LeRobot dataset to: {output_path}")
    return output_path


@hydra.main(
    version_base="1.3",
    config_path="../config/dataset",
    config_name="convert_act_hdf5_to_lerobot",
)
def main(cfg: DictConfig) -> None:
    convert(cfg)


if __name__ == "__main__":
    main()
