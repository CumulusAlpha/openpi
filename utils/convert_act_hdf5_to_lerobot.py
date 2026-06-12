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

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import re
import shutil

import cv2
import h5py
import hydra
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import check_timestamps_sync
from lerobot.common.datasets.utils import get_episode_data_index
import numpy as np
from omegaconf import DictConfig
import pyarrow as pa
import pyarrow.parquet as pq
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


class OnlineFeatureStats:
    def __init__(self, output_shape: tuple[int, ...]):
        self.output_shape = output_shape
        self.count = 0
        self.mean = None
        self.m2 = None
        self.min = None
        self.max = None

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1, *self.output_shape)
        batch_count = values.shape[0]
        batch_mean = values.mean(axis=0)
        batch_m2 = ((values - batch_mean) ** 2).sum(axis=0)
        batch_min = values.min(axis=0)
        batch_max = values.max(axis=0)

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            self.min = batch_min
            self.max = batch_max
            return

        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * batch_count / total_count
        self.m2 = self.m2 + batch_m2 + (delta**2) * self.count * batch_count / total_count
        self.min = np.minimum(self.min, batch_min)
        self.max = np.maximum(self.max, batch_max)
        self.count = total_count

    def finalize(self, stat_count: int | None = None) -> dict[str, np.ndarray]:
        if self.count == 0:
            raise ValueError("Cannot finalize empty stats.")
        return {
            "min": self.min.astype(np.float32),
            "max": self.max.astype(np.float32),
            "mean": self.mean.astype(np.float32),
            "std": np.sqrt(self.m2 / self.count).astype(np.float32),
            "count": np.array([self.count if stat_count is None else stat_count]),
        }


class StreamingEpisodeStats:
    def __init__(self, features: dict):
        self.features = features
        self.numeric_stats: dict[str, OnlineFeatureStats] = {}
        self.image_stats: dict[str, OnlineFeatureStats] = {}
        self.image_frame_counts: dict[str, int] = {}

        for key, ft in features.items():
            dtype = ft["dtype"]
            if dtype in ["image", "video"]:
                self.image_stats[key] = OnlineFeatureStats((3,))
                self.image_frame_counts[key] = 0
            elif dtype != "string":
                shape = tuple(ft["shape"])
                self.numeric_stats[key] = OnlineFeatureStats(shape)

    def update_numeric(self, key: str, value: np.ndarray | int | float) -> None:
        if key in self.numeric_stats:
            self.numeric_stats[key].update(np.asarray(value))

    def update_image(self, key: str, image: np.ndarray) -> None:
        if key not in self.image_stats:
            return
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HWC RGB image for {key}, got shape {image.shape}")
        pixels = image.reshape(-1, 3).astype(np.float32) / 255.0
        self.image_stats[key].update(pixels)
        self.image_frame_counts[key] += 1

    def finalize(self) -> dict[str, dict[str, np.ndarray]]:
        stats = {key: stat.finalize() for key, stat in self.numeric_stats.items()}
        for key, stat in self.image_stats.items():
            image_stats = stat.finalize(stat_count=self.image_frame_counts[key])
            stats[key] = {
                name: value if name == "count" else value.reshape(3, 1, 1)
                for name, value in image_stats.items()
            }
        return stats


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


def state2action(actions: np.ndarray) -> np.ndarray:
    """Shift state-like actions so action[i] targets the next frame's state."""
    actions = np.asarray(actions)
    if actions.shape[0] == 0:
        return actions.copy()
    return np.concatenate([actions[1:], actions[-1:]], axis=0)


def cfg_get(args: DictConfig, key: str, default):
    return args.get(key, default) if hasattr(args, "get") else getattr(args, key, default)


def encode_image_bytes(image: np.ndarray, image_encoding: str, jpeg_quality: int) -> bytes:
    image_encoding = image_encoding.lower().lstrip(".")
    if image_encoding in {"jpg", "jpeg"}:
        ext = ".jpg"
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    elif image_encoding == "png":
        ext = ".png"
        params = []
    else:
        raise ValueError(f"Unsupported image_encoding={image_encoding!r}. Use 'png' or 'jpg'.")

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(ext, bgr, params)
    if not ok:
        raise ValueError(f"Could not encode image as {image_encoding}")
    return encoded.tobytes()


def add_task_if_needed(dataset: LeRobotDataset, task: str) -> int:
    task_index = dataset.meta.get_task_index(task)
    if task_index is None:
        dataset.meta.add_task(task)
        task_index = dataset.meta.get_task_index(task)
    return int(task_index)


def flush_rows(writer: pq.ParquetWriter, rows: list[dict], schema: pa.Schema) -> None:
    if not rows:
        return
    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    rows.clear()


def save_hdf5_episode_streaming(
    dataset: LeRobotDataset,
    path: Path,
    image_map: dict[str, str],
    task: str,
    batch_size: int,
    image_encoding: str,
    jpeg_quality: int,
) -> None:
    episode_index = dataset.meta.total_episodes
    global_start_index = dataset.meta.total_frames
    task_index = add_task_if_needed(dataset, task)
    parquet_path = dataset.root / dataset.meta.get_data_file_path(ep_index=episode_index)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    schema = dataset.hf_features.arrow_schema
    rows: list[dict] = []
    episode_tasks = {task}
    stats = StreamingEpisodeStats(dataset.features)

    with h5py.File(path, "r") as episode, pq.ParquetWriter(parquet_path, schema=schema) as writer:
        qpos = episode["/observations/qpos"]
        action = episode["/action"]
        num_frames = int(qpos.shape[0])

        for frame_index in range(num_frames):
            timestamp = frame_index / dataset.fps
            action_index = frame_index + 1 if frame_index + 1 < num_frames else frame_index
            state_value = qpos[frame_index].astype(np.float32)
            action_value = action[action_index].astype(np.float32)

            row = {
                "observation.state": state_value.tolist(),
                "action": action_value.tolist(),
                "timestamp": float(timestamp),
                "frame_index": int(frame_index),
                "episode_index": int(episode_index),
                "index": int(global_start_index + frame_index),
                "task_index": int(task_index),
            }
            stats.update_numeric("observation.state", state_value)
            stats.update_numeric("action", action_value)
            stats.update_numeric("timestamp", timestamp)
            stats.update_numeric("frame_index", frame_index)
            stats.update_numeric("episode_index", episode_index)
            stats.update_numeric("index", global_start_index + frame_index)
            stats.update_numeric("task_index", task_index)

            for raw_key, lerobot_key in image_map.items():
                try:
                    image = decode_image(episode[f"/observations/{raw_key}"][frame_index])
                    row[lerobot_key] = {
                        "bytes": encode_image_bytes(image, image_encoding, jpeg_quality),
                        "path": None,
                    }
                    stats.update_image(lerobot_key, image)
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to process image in {path} at frame {frame_index}, camera {raw_key}"
                    ) from exc

            rows.append(row)
            if len(rows) >= batch_size:
                flush_rows(writer, rows, schema)

        flush_rows(writer, rows, schema)

    episode_stats = stats.finalize()
    dataset.meta.save_episode(episode_index, num_frames, list(episode_tasks), episode_stats)

    ep_data_index = get_episode_data_index(dataset.meta.episodes, [episode_index])
    ep_data_index_np = {key: value.numpy() for key, value in ep_data_index.items()}
    timestamps = np.arange(num_frames, dtype=np.float32) / dataset.fps
    episode_indices = np.full((num_frames,), episode_index)
    check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, dataset.fps, dataset.tolerance_s)


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
        fps = round(float(first_episode.attrs.get("frame_rate", args.fps)))
        task_from_attrs = read_attr_string(first_episode.attrs, "task", "").strip()

    task = args.task if args.task is not None else task_from_attrs or args.default_task

    for path in hdf5_files[1:]:
        validate_episode(path, image_map, state_dim, action_dim)

    print(f"Found {len(hdf5_files)} episodes under {raw_dir}")
    print(f"FPS: {fps}")
    print(f"Task: {task!r}")
    print(f"State dim: {state_dim}, action dim: {action_dim}")
    print(f"Images: {image_map}")
    print(f"First episode frames: {first_info.num_frames}, image shapes: {first_info.image_shapes}")
    print(f"Streaming parquet writer: {cfg_get(args, 'stream_to_parquet', default=True)}")

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

    stream_to_parquet = bool(cfg_get(args, "stream_to_parquet", default=True))
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=root,
        fps=fps,
        robot_type=args.robot_type,
        features=features,
        use_videos=not args.no_videos,
        tolerance_s=args.tolerance_s,
        image_writer_processes=0 if stream_to_parquet else args.image_writer_processes,
        image_writer_threads=0 if stream_to_parquet else args.image_writer_threads,
        video_backend=args.video_backend,
    )

    if stream_to_parquet:
        batch_size = int(cfg_get(args, "parquet_batch_size", 64))
        if batch_size <= 0:
            raise ValueError("parquet_batch_size must be > 0")
        image_encoding = str(cfg_get(args, "image_encoding", "png"))
        jpeg_quality = int(cfg_get(args, "jpeg_quality", 95))
        for path in tqdm.tqdm(hdf5_files, desc="episodes"):
            save_hdf5_episode_streaming(
                dataset=dataset,
                path=path,
                image_map=image_map,
                task=task,
                batch_size=batch_size,
                image_encoding=image_encoding,
                jpeg_quality=jpeg_quality,
            )
    else:
        for path in tqdm.tqdm(hdf5_files, desc="episodes"):
            with h5py.File(path, "r") as episode:
                qpos = episode["/observations/qpos"]
                action = state2action(episode["/action"][()])
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
