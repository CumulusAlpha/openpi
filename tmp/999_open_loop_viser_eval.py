#!/usr/bin/env python3
"""Open-loop Viser visualization for a trained pi0 ARX checkpoint.

This script replays one LeRobot episode from the training set, runs policy.infer()
on the recorded observations, and visualizes predicted action targets against the
recorded dataset state/action using ViserUrdf.update_cfg(). It never connects to
or commands the real robot.
"""

from __future__ import annotations

import dataclasses
import io
import json
import logging
from pathlib import Path
import sys
import threading
import time
from typing import Annotated, Any

import numpy as np
import pandas as pd
from PIL import Image
import tyro

FILE = Path(__file__).resolve()
REPO_ROOT = FILE.parents[1]
for path in (
    REPO_ROOT,
    REPO_ROOT / "src",
    REPO_ROOT / "packages" / "openpi-client" / "src",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.shared import normalize as _normalize  # noqa: E402
from openpi.training import config as _config  # noqa: E402

DEFAULT_CONFIG = "pi0_arx_lora_chunk50_delta"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/pi0_arx_lora_chunk50_delta/apple_lora/29999"
DEFAULT_DATASET = REPO_ROOT / "data/apple"
DEFAULT_PROMPT = "pick up the apple and put it in the white bowl"
DEFAULT_FALLBACK_URDF = REPO_ROOT / "assets/X5.urdf"


@dataclasses.dataclass(frozen=True)
class Args:
    config: str = DEFAULT_CONFIG
    checkpoint: Path = DEFAULT_CHECKPOINT
    dataset: Path = DEFAULT_DATASET
    episode_index: int = 0
    prompt: str = DEFAULT_PROMPT
    urdf_path: Path | None = None
    host: str = "0.0.0.0"
    port: int = 8080
    fps: float = 10.0
    max_steps: int = 0
    """0 means replay the full episode."""
    start_frame: int = 0
    pytorch_device: str | None = None
    """For safetensors checkpoints only."""
    show_gt_action: bool = False
    """Reference arm uses recorded action instead of state."""
    no_policy: Annotated[bool, tyro.conf.FlagCreatePairsOff] = False
    """Only replay dataset state/action without loading the policy."""


def find_urdf(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"URDF path does not exist: {explicit}")
        return explicit

    if DEFAULT_FALLBACK_URDF.exists():
        return DEFAULT_FALLBACK_URDF

    raise FileNotFoundError(f"Could not find X5 URDF at {DEFAULT_FALLBACK_URDF}. Pass --urdf-path explicitly.")

def episode_path(dataset: Path, episode_index: int) -> Path:
    info = json.loads((dataset / "meta/info.json").read_text())
    chunks_size = int(info.get("chunks_size", 1000))
    pattern = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    return dataset / pattern.format(episode_chunk=episode_index // chunks_size, episode_index=episode_index)


def load_norm_stats(dataset: Path) -> dict[str, _normalize.NormStats]:
    norm_path = dataset / "norm_stats.json"
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing dataset norm stats: {norm_path}")
    return _normalize.deserialize_json(norm_path.read_text())


def decode_image(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            with Image.open(io.BytesIO(value["bytes"])) as img:
                return np.asarray(img.convert("RGB"))
        if value.get("path") is not None:
            with Image.open(value["path"]) as img:
                return np.asarray(img.convert("RGB"))
    if isinstance(value, (bytes, bytearray)):
        with Image.open(io.BytesIO(value)) as img:
            return np.asarray(img.convert("RGB"))
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def chw_uint8(image_hwc: np.ndarray) -> np.ndarray:
    if image_hwc.dtype != np.uint8:
        image_hwc = np.clip(image_hwc, 0, 255).astype(np.uint8)
    return np.moveaxis(image_hwc, -1, 0)


def make_policy_obs(row: pd.Series, prompt: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    top = decode_image(row["observation.images.top"])
    right = decode_image(row["observation.images.right_wrist"])
    obs = {
        "observation.images.top": chw_uint8(top),
        "observation.images.right_wrist": chw_uint8(right),
        "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
        # The train-time repack transform includes "actions". For plain
        # policy.infer() this is only a structural placeholder; the sampled
        # action chunk comes from the model output, not from this dummy value.
        "action": np.zeros((50, 14), dtype=np.float32),
        "prompt": prompt,
    }
    return obs, {"top": top, "right_wrist": right}


def gripper_opening(value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        return 0.0
    if 0.0 <= value <= 0.08:
        return float(np.clip((0.08 - value) * 0.5, 0.0, 0.04))
    if -1.0 <= value <= 1.0:
        return float(np.clip((1.0 - value) * 0.02, 0.0, 0.04))
    return float(np.clip(0.04 - value, 0.0, 0.04))


def split_arm_cfg(qpos: np.ndarray) -> tuple[tuple[np.ndarray, float, float], tuple[np.ndarray, float, float]]:
    qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
    if qpos.shape[0] < 14:
        raise ValueError(f"Expected 14-dim qpos/action, got {qpos.shape}")
    left_gripper = float(qpos[6])
    right_gripper = float(qpos[13])
    return (
        (qpos[:6], left_gripper, gripper_opening(left_gripper)),
        (qpos[7:13], right_gripper, gripper_opening(right_gripper)),
    )


class ViserOpenLoopView:
    LINK6_VISUAL_PATH = "visual/link1/link2/link3/link4/link5/link6"
    FINGER_DIMS = (0.07, 0.012, 0.018)
    FINGER_X = 0.18
    FINGER_BASE_Y = 0.018

    def __init__(self, urdf_path: Path, host: str, port: int):
        import viser
        from viser.extras import ViserUrdf

        self.server = viser.ViserServer(host=host, port=port)
        self.urdfs = {}
        self.fingers = {}
        roots = {
            "pred_left": ("/pred_left_x5", (0.0, 0.35, 0.0), (0.55, 0.55, 0.55, 0.28)),
            "pred_right": ("/pred_right_x5", (0.0, -0.35, 0.0), (0.55, 0.55, 0.55, 0.28)),
            "ref_left": ("/ref_left_x5", (0.0, 0.35, 0.0), (0.1, 0.75, 0.25, 1.0)),
            "ref_right": ("/ref_right_x5", (0.0, -0.35, 0.0), (0.1, 0.75, 0.25, 1.0)),
        }
        for key, (root, position, rgba) in roots.items():
            self.server.scene.add_frame(root, position=position, show_axes=False)
            self.urdfs[key] = ViserUrdf(self.server, urdf_path, root_node_name=root, mesh_color_override=rgba)
            self.fingers[key] = self._add_gripper_boxes(root, rgba)

        self._control_lock = threading.Lock()
        self._paused = False
        self._step_requested = False
        self._reset_requested = False

        with self.server.gui.add_folder("Controls", expand_by_default=True):
            self.pause_checkbox = self.server.gui.add_checkbox("pause", initial_value=False)
            self.step_button = self.server.gui.add_button("step")
            self.reset_button = self.server.gui.add_button("reset")

        @self.pause_checkbox.on_update
        def _(_) -> None:
            with self._control_lock:
                self._paused = bool(self.pause_checkbox.value)

        @self.step_button.on_click
        def _(_) -> None:
            with self._control_lock:
                self._step_requested = True

        @self.reset_button.on_click
        def _(_) -> None:
            with self._control_lock:
                self._reset_requested = True

        self.gripper_text = self.server.gui.add_text(
            "gripper debug",
            "Waiting for first frame",
            multiline=True,
            disabled=True,
        )
        with self.server.gui.add_folder("Images", expand_by_default=True):
            empty = np.zeros((240, 320, 3), dtype=np.uint8)
            self.top_image = self.server.gui.add_image(empty, label="top")
            self.right_image = self.server.gui.add_image(empty, label="right_wrist")
        self.status = self.server.gui.add_markdown("Starting")

    def consume_controls(self) -> tuple[bool, bool, bool]:
        with self._control_lock:
            paused = self._paused
            step_requested = self._step_requested
            reset_requested = self._reset_requested
            self._step_requested = False
            self._reset_requested = False
        return paused, step_requested, reset_requested

    def _add_gripper_boxes(self, root: str, rgba: tuple[float, float, float, float]) -> dict[str, Any]:
        link6 = f"{root}/{self.LINK6_VISUAL_PATH}"
        color = rgba[:3]
        opacity = rgba[3]
        return {
            "left": self.server.scene.add_box(
                f"{link6}/left_finger",
                color=color,
                dimensions=self.FINGER_DIMS,
                opacity=opacity,
                position=(self.FINGER_X, self.FINGER_BASE_Y, 0.0),
            ),
            "right": self.server.scene.add_box(
                f"{link6}/right_finger",
                color=color,
                dimensions=self.FINGER_DIMS,
                opacity=opacity,
                position=(self.FINGER_X, -self.FINGER_BASE_Y, 0.0),
            ),
        }

    def _update_arm(self, key: str, arm: np.ndarray, _raw_gripper: float, opening: float) -> None:
        self.urdfs[key].update_cfg(arm)
        self.fingers[key]["left"].position = (self.FINGER_X, self.FINGER_BASE_Y + opening, 0.0)
        self.fingers[key]["right"].position = (self.FINGER_X, -self.FINGER_BASE_Y - opening, 0.0)

    @staticmethod
    def _gripper_text(
        pred_left: tuple[np.ndarray, float, float],
        pred_right: tuple[np.ndarray, float, float],
        ref_left: tuple[np.ndarray, float, float],
        ref_right: tuple[np.ndarray, float, float],
    ) -> str:
        return (
            "pred gray\n"
            f"left raw: {pred_left[1]:.6f}, opening: {pred_left[2]:.6f} m\n"
            f"right raw: {pred_right[1]:.6f}, opening: {pred_right[2]:.6f} m\n\n"
            "ref green\n"
            f"left raw: {ref_left[1]:.6f}, opening: {ref_left[2]:.6f} m\n"
            f"right raw: {ref_right[1]:.6f}, opening: {ref_right[2]:.6f} m"
        )

    def update(self, *, pred: np.ndarray | None, ref: np.ndarray, images: dict[str, np.ndarray], text: str) -> None:
        if pred is None:
            pred = ref
        pred_left, pred_right = split_arm_cfg(pred)
        ref_left, ref_right = split_arm_cfg(ref)
        self._update_arm("pred_left", *pred_left)
        self._update_arm("pred_right", *pred_right)
        self._update_arm("ref_left", *ref_left)
        self._update_arm("ref_right", *ref_right)
        self.top_image.image = images["top"]
        self.right_image.image = images["right_wrist"]
        self.status.content = text
        self.gripper_text.value = self._gripper_text(pred_left, pred_right, ref_left, ref_right)


def load_policy(args: Args):
    train_config = _config.get_config(args.config)
    norm_stats = load_norm_stats(args.dataset)
    return _policy_config.create_trained_policy(
        train_config,
        args.checkpoint,
        repack_transforms=train_config.data.repack_transforms,
        default_prompt=args.prompt,
        norm_stats=norm_stats,
        pytorch_device=args.pytorch_device,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = tyro.cli(Args, description=__doc__)
    urdf_path = find_urdf(args.urdf_path)
    ep_path = episode_path(args.dataset, args.episode_index)
    if not ep_path.exists():
        raise FileNotFoundError(ep_path)

    print(f"URDF: {urdf_path}")
    print(f"Episode: {ep_path}")
    df = pd.read_parquet(ep_path)
    if args.start_frame < 0 or args.start_frame >= len(df):
        raise ValueError(f"--start-frame must be in [0, {len(df) - 1}], got {args.start_frame}")
    stop = len(df) if args.max_steps <= 0 else min(len(df), args.start_frame + args.max_steps)

    policy = None if args.no_policy else load_policy(args)
    view = ViserOpenLoopView(urdf_path, host=args.host, port=args.port)

    action_chunk: np.ndarray | None = None
    chunk_start = -1
    infer_count = 0
    dt = 1.0 / max(args.fps, 1e-6)

    print(f"Viser is serving on http://{args.host}:{args.port}")
    print("Press Ctrl-C to stop.")

    try:
        frame_idx = args.start_frame
        while True:
            paused, step_requested, reset_requested = view.consume_controls()
            if reset_requested:
                frame_idx = args.start_frame
                action_chunk = None
                chunk_start = -1

            if frame_idx >= stop:
                time.sleep(0.05)
                continue

            if paused and not step_requested:
                time.sleep(0.05)
                continue

            loop_t = time.perf_counter()
            row = df.iloc[frame_idx]
            obs, images = make_policy_obs(row, args.prompt)
            offset = frame_idx - chunk_start
            if policy is not None and (action_chunk is None or offset < 0 or offset >= len(action_chunk)):
                infer_t = time.perf_counter()
                result = policy.infer(obs)
                action_chunk = np.asarray(result["actions"], dtype=np.float32)
                chunk_start = frame_idx
                offset = 0
                infer_count += 1
                model_ms = result.get("policy_timing", {}).get("infer_ms", float("nan"))
                wall_ms = (time.perf_counter() - infer_t) * 1000.0
                print(
                    f"infer #{infer_count}: frame={frame_idx}, chunk={action_chunk.shape}, "
                    f"wall_ms={wall_ms:.1f}, model_ms={model_ms:.1f}"
                )

            pred_action = None if action_chunk is None else action_chunk[offset]
            ref = np.asarray(row["action" if args.show_gt_action else "observation.state"], dtype=np.float32)
            gt_action = np.asarray(row["action"], dtype=np.float32)
            state = np.asarray(row["observation.state"], dtype=np.float32)
            err = float("nan") if pred_action is None else float(np.linalg.norm(pred_action - gt_action))
            chunk_left = 0 if action_chunk is None else len(action_chunk) - offset - 1
            chunk_offset_text = str(offset) if action_chunk is not None else "none"
            ref_label = "recorded action" if args.show_gt_action else "recorded state"
            pred_head = (
                np.array2string(pred_action[:3], precision=4)
                if pred_action is not None
                else "policy disabled"
            )
            pred_gripper = (
                f"L={pred_action[6]:.6f}, R={pred_action[13]:.6f}"
                if pred_action is not None
                else "policy disabled"
            )
            status = (
                f"<b>frame</b>: {frame_idx}/{stop - 1}<br>"
                f"<b>episode</b>: {args.episode_index}<br>"
                f"<b>chunk offset</b>: {chunk_offset_text} ({chunk_left} remaining)<br>"
                f"<b>prediction vs recorded action L2</b>: {err:.6f}<br>"
                f"<b>left/right layout</b>: gray predicted/action and green reference are overlaid at the same origins<br>"
                f"<b>reference</b>: {ref_label}<br>"
                f"<b>state[:3]</b>: {np.array2string(state[:3], precision=4)}<br>"
                f"<b>pred[:3]</b>: {pred_head}<br>"
                f"<b>ref gripper raw</b>: L={ref[6]:.6f}, R={ref[13]:.6f}<br>"
                f"<b>pred gripper raw</b>: {pred_gripper}"
            )
            view.update(pred=pred_action, ref=ref, images=images, text=status)
            frame_idx += 1

            elapsed = time.perf_counter() - loop_t
            if elapsed < dt:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        print("Stopped by user.")


if __name__ == "__main__":
    main()
