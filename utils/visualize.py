#!/usr/bin/env python3
"""Stream a LeRobot dataset episode into a Viser viewer.

Examples:
    python utils/visualize.py --repo-id local/act_aloha --root /path/to/lerobot_dataset
    python utils/visualize.py --root /home/arx/.cache/huggingface/lerobot/local/act_aloha --episode-index 0
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

DEFAULT_VISER_URDF_PATH = Path("/home/arx/arx5-sdk/models/X5.urdf")


def _qpos_to_arm_cfgs(qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
    if qpos.size < 13:
        raise RuntimeError(f"Expected qpos to have at least 13 values, got {qpos.size}")
    return qpos[:6], qpos[7:13]


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_scalar_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().reshape(-1)[0].item())
    return int(np.asarray(value).reshape(-1)[0])


def _coerce_gui_int(value: Any, *, fallback: int, min_value: int, max_value: int) -> int:
    """Coerce a GUI numeric value while tolerating transient browser NaN values."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(fallback)
    if not math.isfinite(number):
        number = float(fallback)
    return max(min_value, min(int(round(number)), max_value))


def _to_hwc_uint8_image(value: Any) -> np.ndarray:
    image = _to_numpy(value)

    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise RuntimeError(f"Expected an image with 3 dimensions, got shape {image.shape}")

    # LeRobot video decoding often returns CHW tensors, while image parquet values
    # are commonly HWC. Detect CHW by a small channel dimension in front.
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.max(initial=0) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]

    return np.ascontiguousarray(image)


def _format_vector(name: str, value: Any, max_items: int = 14) -> str | None:
    if value is None:
        return None
    arr = _to_numpy(value).astype(np.float32, copy=False).reshape(-1)
    shown = np.array2string(arr[:max_items], precision=4, suppress_small=True)
    suffix = "" if arr.size <= max_items else f" ... ({arr.size})"
    return f"**{name}** `{shown}{suffix}`"


def _episode_length(meta: Any, episode_index: int) -> int:
    episode = meta.episodes[episode_index]
    if isinstance(episode, dict):
        return int(episode["length"])
    return int(episode.length)


def _episode_indices(meta: Any) -> list[int]:
    return sorted(int(key) for key in meta.episodes)


class StreamingLeRobotViewer:
    def __init__(
        self,
        *,
        repo_id: str,
        root: Path | None,
        episode_index: int,
        urdf_path: Path,
        video_backend: str | None,
        no_robot: bool,
    ):
        try:
            import viser
            from viser.extras import ViserUrdf
        except ImportError as exc:
            raise ImportError("viser is required. Install it with: pip install viser") from exc

        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

        self._viser_urdf_cls = ViserUrdf
        self._dataset_cls = LeRobotDataset
        self._repo_id = repo_id
        self._root = root
        self._urdf_path = urdf_path
        self._video_backend = video_backend
        self._no_robot = no_robot

        self.meta = LeRobotDatasetMetadata(repo_id, root=root)
        self.episode_indices = _episode_indices(self.meta)
        if episode_index not in self.episode_indices:
            raise ValueError(f"Episode {episode_index} is not available. Available: {self.episode_indices}")

        self.camera_keys = list(self.meta.camera_keys)
        self.vector_keys = [
            key
            for key, feature in self.meta.features.items()
            if feature.get("dtype") not in {"image", "video"} and key not in {"timestamp", "episode_index", "index"}
        ]
        for preferred_key in ("observation.state", "action"):
            if preferred_key in self.meta.features and preferred_key not in self.vector_keys:
                self.vector_keys.append(preferred_key)

        self.dataset = None
        self.current_episode_index = episode_index
        self.current_episode_length = 0

        self.server = viser.ViserServer()
        self.urdfs = []
        self.image_handles = {}

        self.episode_slider = self.server.gui.add_slider(
            "Episode",
            min=float(min(self.episode_indices)),
            max=float(max(self.episode_indices)),
            step=1.0,
            initial_value=float(episode_index),
        )
        self.step_slider = self.server.gui.add_slider(
            "Step",
            min=0.0,
            max=0.0,
            step=1.0,
            initial_value=0.0,
        )

        with self.server.gui.add_folder("Playback", expand_by_default=True):
            self.play_handle = self.server.gui.add_checkbox("Play", initial_value=False)
            self.fps_handle = self.server.gui.add_slider(
                "FPS",
                min=1.0,
                max=float(max(int(self.meta.fps), 1)),
                step=1.0,
                initial_value=float(max(min(int(self.meta.fps), 30), 1)),
            )

        with self.server.gui.add_folder("State", expand_by_default=True):
            self.state_handle = self.server.gui.add_markdown("")

        if not no_robot and urdf_path.exists() and "observation.state" in self.meta.features:
            for root_node_name, position in (
                ("/left_x5", (0.0, 0.35, 0.0)),
                ("/right_x5", (0.0, -0.35, 0.0)),
            ):
                self.server.scene.add_frame(root_node_name, position=position, show_axes=False)
                self.urdfs.append(self._viser_urdf_cls(self.server, urdf_path, root_node_name=root_node_name))
        elif not no_robot and not urdf_path.exists():
            print(f"URDF path does not exist, skip robot visualization: {urdf_path}")

        self._load_episode(episode_index)
        self._ensure_image_handles()
        self.update_step()

        self.step_slider.on_update(lambda _: self.update_step())
        self.episode_slider.on_update(lambda _: self.update_episode())

    def _load_episode(self, episode_index: int) -> None:
        if episode_index not in self.episode_indices:
            print(f"Episode {episode_index} is not available; keep episode {self.current_episode_index}")
            self.episode_slider.value = self.current_episode_index
            return

        self.current_episode_index = episode_index
        self.current_episode_length = _episode_length(self.meta, episode_index)
        self.dataset = self._dataset_cls(
            self._repo_id,
            root=self._root,
            episodes=[episode_index],
            video_backend=self._video_backend,
        )
        self.step_slider.max = max(self.current_episode_length - 1, 0)
        self.step_slider.value = float(
            _coerce_gui_int(
                self.step_slider.value,
                fallback=0,
                min_value=0,
                max_value=int(self.step_slider.max),
            )
        )
        print(f"Loaded episode {episode_index} with {self.current_episode_length} frames")

    def _read_frame(self, step_index: int) -> dict:
        if self.dataset is None:
            raise RuntimeError("Dataset is not loaded")
        if self.current_episode_length <= 0:
            raise RuntimeError(f"Episode {self.current_episode_index} has no frames")
        step_index = max(0, min(step_index, self.current_episode_length - 1))
        return self.dataset[step_index]

    def _ensure_image_handles(self) -> None:
        if not self.camera_keys:
            return
        frame = self._read_frame(0)
        with self.server.gui.add_folder("Cameras", expand_by_default=True):
            for key in self.camera_keys:
                if key in self.image_handles or key not in frame:
                    continue
                self.image_handles[key] = self.server.gui.add_image(_to_hwc_uint8_image(frame[key]), label=key)

    def update_episode(self) -> None:
        episode_index = _coerce_gui_int(
            self.episode_slider.value,
            fallback=self.current_episode_index,
            min_value=min(self.episode_indices),
            max_value=max(self.episode_indices),
        )
        if float(episode_index) != self.episode_slider.value:
            self.episode_slider.value = float(episode_index)
        if episode_index == self.current_episode_index:
            return
        self._load_episode(episode_index)
        self._ensure_image_handles()
        self.update_step()

    def update_step(self) -> None:
        step_index = _coerce_gui_int(
            self.step_slider.value,
            fallback=0,
            min_value=0,
            max_value=max(self.current_episode_length - 1, 0),
        )
        if float(step_index) != self.step_slider.value:
            self.step_slider.value = float(step_index)
        frame = self._read_frame(step_index)

        for key, handle in self.image_handles.items():
            if key in frame:
                handle.image = _to_hwc_uint8_image(frame[key])

        state = frame.get("observation.state")
        if self.urdfs and state is not None:
            state_arr = _to_numpy(state).reshape(-1)
            if state_arr.size >= 13:
                for urdf, arm_cfg in zip(self.urdfs, _qpos_to_arm_cfgs(state_arr), strict=False):
                    urdf.update_cfg(arm_cfg)

        lines = [
            f"**Episode** `{self.current_episode_index}`",
            f"**Step** `{step_index}/{self.current_episode_length - 1}`",
        ]
        if "task" in frame:
            lines.append(f"**task** `{frame['task']}`")
        elif "task_index" in frame and _to_scalar_int(frame["task_index"]) in self.meta.tasks:
            lines.append(f"**task** `{self.meta.tasks[_to_scalar_int(frame['task_index'])]}`")

        for key in self.vector_keys:
            if key in frame and key not in {"task_index"}:
                line = _format_vector(key, frame[key])
                if line is not None:
                    lines.append(line)

        self.state_handle.content = "<br/>".join(lines)

    def spin(self) -> None:
        fps = _coerce_gui_int(
            self.fps_handle.value,
            fallback=max(min(int(self.meta.fps), 30), 1),
            min_value=1,
            max_value=max(int(self.meta.fps), 1),
        )
        seconds_per_frame = 1.0 / float(fps)
        last_time = time.monotonic()
        while True:
            now = time.monotonic()
            if self.play_handle.value and now - last_time >= seconds_per_frame:
                last_time = now
                step_index = _coerce_gui_int(
                    self.step_slider.value,
                    fallback=0,
                    min_value=0,
                    max_value=max(int(self.step_slider.max), 0),
                )
                next_step = step_index + 1
                self.step_slider.value = 0 if next_step > int(self.step_slider.max) else next_step
                self.update_step()
                fps = _coerce_gui_int(
                    self.fps_handle.value,
                    fallback=fps,
                    min_value=1,
                    max_value=max(int(self.fps_handle.max), 1),
                )
                if float(fps) != self.fps_handle.value:
                    self.fps_handle.value = float(fps)
                seconds_per_frame = 1.0 / float(fps)
            time.sleep(0.01)


def _infer_repo_id(root: Path | None, repo_id: str | None) -> str:
    if repo_id:
        return repo_id
    if root is None:
        raise ValueError("Either --repo-id or --root must be provided")
    if len(root.parts) >= 2:
        return f"{root.parts[-2]}/{root.parts[-1]}"
    return root.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a LeRobot dataset with Viser, one frame at a time.")
    parser.add_argument("--repo-id", type=str, default=None, help="LeRobot repo id, e.g. local/act_aloha.")
    parser.add_argument("--root", type=Path, default=None, help="Local LeRobot dataset root.")
    parser.add_argument("--episode-index", type=int, default=0, help="Episode index to load first.")
    parser.add_argument("--urdf-path", type=Path, default=DEFAULT_VISER_URDF_PATH, help="URDF path for robot qpos.")
    parser.add_argument("--video-backend", type=str, default=None, help="LeRobot video backend, e.g. pyav.")
    parser.add_argument("--no-robot", action="store_true", help="Disable URDF robot visualization.")
    args = parser.parse_args()

    root = args.root.expanduser() if args.root is not None else None
    if root is not None and not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")

    repo_id = _infer_repo_id(root, args.repo_id)
    viewer = StreamingLeRobotViewer(
        repo_id=repo_id,
        root=root,
        episode_index=args.episode_index,
        urdf_path=args.urdf_path.expanduser(),
        video_backend=args.video_backend,
        no_robot=args.no_robot,
    )
    print("Viser launched. Open the printed URL in your browser.")
    viewer.spin()


if __name__ == "__main__":
    main()
