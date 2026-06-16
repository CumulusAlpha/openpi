#!/usr/bin/env python3
"""Convert ACT/Aloha-style HDF5 episodes to a LeRobot dataset.

Input layout expected by this script:

    <input_path>/episode_0.hdf5
    <input_path>/episode_1.hdf5
    ...

Each HDF5 file is expected to contain:

    /observations/qpos          (T, 14)
    /observations/middle_image  (T, N) JPEG/PNG bytes or (T, H, W, C)
    /observations/right_image   (T, N) JPEG/PNG bytes or (T, H, W, C)
    /action                     (T, 14)

Example:

    .venv/bin/python utils/convert_act_hdf5_to_lerobot.py \
        input_path=/path/to/episodes \
        output_path=data/apple \
        repo_id=local/act_aloha \
        task="put the object into the container"
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import uuid

import cv2
import datasets
import h5py
import hydra
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import embed_images
from lerobot.common.datasets.lerobot_dataset import hf_transform_to_torch
import numpy as np
from omegaconf import DictConfig
import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_IMAGE_MAP = {
    "middle_image": "observation.images.top",
    "right_image": "observation.images.right_wrist",
}

RIGHT_GRIPPER_ACTION_DIM = 13

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


def _save_episode_table_patch(self, episode_buffer: dict, episode_index: int) -> None:
    # Avoid LeRobot's in-memory concatenate_datasets call when saving large episodes.
    episode_dict = {key: episode_buffer[key] for key in self.hf_features}
    ep_dataset = datasets.Dataset.from_dict(episode_dict, features=self.hf_features, split="train")
    ep_dataset = embed_images(ep_dataset)
    self.hf_dataset.set_transform(hf_transform_to_torch)
    ep_data_path = self.root / self.meta.get_data_file_path(ep_index=episode_index)
    ep_data_path.parent.mkdir(parents=True, exist_ok=True)
    ep_dataset.to_parquet(ep_data_path)


LeRobotDataset._save_episode_table = _save_episode_table_patch  # noqa: SLF001


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


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def get_input_path(args: DictConfig) -> Path:
    input_path = args.get("input_path", None)
    if input_path is None:
        input_path = args.get("raw_dir", None)
    if input_path is None:
        raise ValueError("Set input_path in config/dataset/convert_act_hdf5_to_lerobot.yaml")
    return resolve_repo_path(input_path)


def get_output_path(args: DictConfig) -> Path:
    output_path = args.get("output_path", None)
    if output_path is None:
        output_path = args.get("root", None)
    if output_path:
        return resolve_repo_path(output_path)
    return HF_LEROBOT_HOME / args.repo_id


def looks_like_lerobot_dataset(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return (path / "meta" / "info.json").exists() and (path / "meta" / "tasks.jsonl").exists()


def validate_output_path(input_path: Path, output_path: Path, overwrite: bool) -> None:
    input_path = input_path.resolve()
    output_path = output_path.resolve()

    if input_path == output_path:
        raise ValueError(f"input_path and output_path must be different: {input_path}")
    if input_path.is_relative_to(output_path):
        raise ValueError(
            "output_path must not contain input_path. "
            f"Overwriting {output_path} could delete raw HDF5 files under {input_path}."
        )
    if output_path.is_relative_to(input_path):
        raise ValueError(
            "output_path must not be inside input_path. "
            f"Keep raw HDF5 input ({input_path}) and converted output ({output_path}) in sibling folders."
        )

    if not output_path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"{output_path} already exists. Set overwrite=true to regenerate it.")
    if any(output_path.iterdir()) and not looks_like_lerobot_dataset(output_path):
        raise ValueError(
            f"Refusing to overwrite non-LeRobot output directory: {output_path}. "
            "Choose an empty/new output_path or remove the directory manually."
        )


def make_temp_output_path(output_path: Path) -> Path:
    return output_path.parent / f".tmp_{output_path.name}_convert_{uuid.uuid4().hex[:8]}"


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


def state2action(actions: np.ndarray) -> np.ndarray:
    """Shift state-like actions so action[i] targets the next frame's state."""
    actions = np.asarray(actions)
    if actions.shape[0] == 0:
        return actions.copy()
    return np.concatenate([actions[1:], actions[-1:]], axis=0)


def binarize_right_gripper_actions(actions: np.ndarray, cfg: DictConfig | None) -> np.ndarray:
    if cfg is None or not bool(cfg.get("enabled", False)):
        return actions

    actions = np.asarray(actions, dtype=np.float32).copy()
    if actions.ndim != 2:
        raise ValueError(f"Expected actions to have shape (T, D), got {actions.shape}")
    if RIGHT_GRIPPER_ACTION_DIM >= actions.shape[1]:
        raise ValueError(f"Right gripper dim is out of range for action shape {actions.shape}")

    close_below = float(cfg.get("close_below", 0.079))
    actions[:, RIGHT_GRIPPER_ACTION_DIM] = (actions[:, RIGHT_GRIPPER_ACTION_DIM] < close_below).astype(np.float32)
    return actions


def convert(args: DictConfig) -> Path:
    input_path = get_input_path(args)
    output_path = get_output_path(args)
    validate_output_path(input_path, output_path, bool(args.overwrite))
    write_output_path = make_temp_output_path(output_path)
    image_map = parse_image_map(args.image_map)
    right_gripper_binarization = args.get("right_gripper_binarization")
    hdf5_files = find_hdf5_files(input_path)
    if args.max_episodes is not None:
        hdf5_files = hdf5_files[: args.max_episodes]

    first_info = inspect_episode(hdf5_files[0], image_map)
    state_dim = first_info.state_shape[0]
    action_dim = first_info.action_shape[0]

    with h5py.File(hdf5_files[0], "r") as first_episode:
        fps = round(float(first_episode.attrs.get("frame_rate", args.fps)))
        task_from_attrs = read_attr_string(first_episode.attrs, "task", "").strip()

    task = args.task if args.task is not None else task_from_attrs or args.default_task

    for path in hdf5_files[1:]:
        validate_episode(path, image_map, state_dim, action_dim)

    print(f"Found {len(hdf5_files)} episodes under {input_path}")
    print(f"FPS: {fps}")
    print(f"Task: {task!r}")
    print(f"State dim: {state_dim}, action dim: {action_dim}")
    print(f"Images: {image_map}")
    print(f"First episode frames: {first_info.num_frames}, image shapes: {first_info.image_shapes}")
    if right_gripper_binarization is not None and bool(right_gripper_binarization.get("enabled", False)):
        print(
            "Right gripper binarization: "
            f"close_below={right_gripper_binarization.get('close_below')}, "
            "1.0=closed, 0.0=open"
        )

    features = make_features(
        image_map=image_map,
        image_shapes=first_info.image_shapes,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    if write_output_path.exists():
        shutil.rmtree(write_output_path)

    try:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            root=write_output_path,
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
                action = state2action(episode["/action"][()])
                action = binarize_right_gripper_actions(action, right_gripper_binarization)
                num_frames = qpos.shape[0]
                for i in range(num_frames):
                    try:
                        frame = {
                            "observation.state": qpos[i].astype(np.float32),
                            "action": action[i].astype(np.float32),
                            "task": task,
                        }

                        for raw_key, lerobot_key in image_map.items():
                            try:
                                frame[lerobot_key] = decode_image(episode[f"/observations/{raw_key}"][i])
                            except Exception as exc:
                                raise RuntimeError(
                                    f"Failed to decode image in {path} at frame {i}, camera {raw_key}"
                                ) from exc

                        dataset.add_frame(frame)
                    except Exception as exc:
                        raise RuntimeError(f"Failed while converting {path} at frame {i}") from exc

            dataset.save_episode()

        if hasattr(dataset, "consolidate"):
            dataset.consolidate()

        if output_path.exists():
            shutil.rmtree(output_path)
        write_output_path.rename(output_path)
    except Exception:
        if write_output_path.exists():
            shutil.rmtree(write_output_path)
        raise

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
