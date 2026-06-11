# -- coding: UTF-8
import argparse
import gc
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    rs = None


class _RealSenseStreamWorker(threading.Thread):
    """
    Background reader for one RealSense pipeline.
    Keeps only the latest frame pack so callers can read without blocking on USB/camera IO.
    """

    def __init__(self, name, pipeline, capture_fn, verbose=True):
        super().__init__(daemon=True)
        self.name = name
        self.pipeline = pipeline
        self.capture_fn = capture_fn
        self.verbose = verbose

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_pack = None
        self._consecutive_failures = 0
        self._last_error = None

    @property
    def consecutive_failures(self):
        with self._lock:
            return self._consecutive_failures

    @property
    def last_error(self):
        with self._lock:
            return self._last_error

    @property
    def has_latest(self):
        with self._lock:
            return self._latest_pack is not None

    def stop(self):
        self._stop_event.set()

    def get_latest(self):
        with self._lock:
            if self._latest_pack is None:
                return None
            return {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in self._latest_pack.items()
            }

    def run(self):
        while not self._stop_event.is_set():
            try:
                pack = self.capture_fn(self.name, self.pipeline, img_size=None)
                if pack is None:
                    with self._lock:
                        self._consecutive_failures += 1
                        self._last_error = "capture returned no frames"
                    time.sleep(0.01)
                    continue

                with self._lock:
                    self._latest_pack = pack
                    self._consecutive_failures = 0
                    self._last_error = None
            except Exception as exc:
                with self._lock:
                    self._consecutive_failures += 1
                    self._last_error = str(exc)
                if self.verbose:
                    print(f"{self.name}: RealSense reader error: {exc}")
                time.sleep(0.05)


class RealSenseManager:
    """
    Manage one or more Intel RealSense cameras.

    Each camera gets its own rs.pipeline. When realsense_serials is a mapping
    such as {"left": "...", "right": "..."}, camera_names selects which named
    cameras are initialized.
    """

    def __init__(
        self,
        desired_width=640,
        desired_height=480,
        desired_fps=90,
        warmup_frames=30,
        wait_timeout_ms=1000,
        device_serials=None,
        camera_names=None,
        align_to_color=True,
        prefer_mjpeg=False,
        verbose=True,
    ):
        self.enabled = REALSENSE_AVAILABLE
        self.verbose = verbose
        if not self.enabled:
            print("pyrealsense2 not found; RealSense disabled.")
            return

        self.desired_w = int(desired_width)
        self.desired_h = int(desired_height)
        self.desired_fps = int(desired_fps)
        self.warmup_frames = int(warmup_frames)
        self.wait_timeout_ms = int(wait_timeout_ms)
        self.camera_names = _normalize_optional_list(camera_names)
        self.device_serials_by_name = None

        if isinstance(device_serials, Mapping):
            serials_by_name = {str(name): str(serial) for name, serial in device_serials.items()}
            if self.camera_names is None:
                self.camera_names = list(serials_by_name.keys())

            missing_names = [name for name in self.camera_names if name not in serials_by_name]
            if missing_names:
                raise ValueError(
                    "camera_names contains names not present in realsense_serials: "
                    f"{missing_names}. Available names: {list(serials_by_name.keys())}"
                )

            self.device_serials_by_name = OrderedDict(
                (name, serials_by_name[name]) for name in self.camera_names
            )
            self.device_serials = list(self.device_serials_by_name.values())
        else:
            self.device_serials = _normalize_optional_list(device_serials)

        if self.camera_names and self.device_serials and len(self.camera_names) != len(self.device_serials):
            raise ValueError(
                "camera_names and realsense_serials must have the same length when "
                "realsense_serials is a list. Use a name-to-serial mapping to select "
                "a subset by camera name: "
                f"{len(self.camera_names)} != {len(self.device_serials)}"
            )
        self.align_to_color = align_to_color
        self.prefer_mjpeg = prefer_mjpeg
        self.depth_display_max_mm = 5000

        self.ctx = rs.context()
        self.devices_info = OrderedDict()
        self.pipelines = OrderedDict()
        self.workers = OrderedDict()
        self.aligners = {}
        self.pipeline_profiles = {}
        self.depth_display_max_mm_by_name = {}

        self._fail_counts = {}
        self._fail_restart_thresh = 5

        self._discover_devices()
        if self.devices_info:
            self._start_all()
        else:
            self.enabled = False

    def is_enabled(self):
        return self.enabled

    def camera_keys(self):
        return list(self.devices_info.keys())

    @classmethod
    def from_collect_config(cls, config_path=None):
        if config_path is None:
            config_path = Path(__file__).resolve().parents[1] / "configs" / "collect.yaml"
        else:
            config_path = Path(config_path).expanduser().resolve()

        from omegaconf import OmegaConf

        cfg = OmegaConf.load(config_path)
        camera_names = cfg.get("camera_names", None)
        serials = cfg.get("realsense_serials", None)
        if camera_names is not None:
            camera_names = OmegaConf.to_container(camera_names, resolve=True)
        if serials is not None:
            serials = OmegaConf.to_container(serials, resolve=True)
        return cls(
            desired_width=cfg.get("realsense_width", 640),
            desired_height=cfg.get("realsense_height", 480),
            desired_fps=cfg.get("realsense_fps", 30),
            warmup_frames=cfg.get("realsense_warmup_frames", 30),
            wait_timeout_ms=cfg.get("realsense_wait_timeout_ms", 100),
            device_serials=serials,
            camera_names=camera_names,
            align_to_color=cfg.get("realsense_align_to_color", True),
            verbose=cfg.get("realsense_verbose", True),
        )

    def stop(self, settle_seconds=0.7):
        stopped_any = False
        for worker in list(self.workers.values()):
            worker.stop()
        for worker in list(self.workers.values()):
            worker.join(timeout=max(1.0, self.wait_timeout_ms / 1000.0 + 0.2))

        for name, pipe in list(self.pipelines.items()):
            try:
                pipe.stop()
                stopped_any = True
                if self.verbose:
                    print(f"Stopped RealSense {name}")
            except Exception:
                pass
        for worker in list(self.workers.values()):
            worker.join(timeout=0.5)
        self.workers.clear()
        self.pipelines.clear()
        self.pipeline_profiles.clear()
        self.aligners.clear()
        self._fail_counts.clear()
        gc.collect()
        if stopped_any and settle_seconds > 0:
            time.sleep(settle_seconds)

    def capture_frames(self, img_size=None):
        if not self.enabled:
            return {}

        out = OrderedDict()
        to_restart = []

        for name, worker in list(self.workers.items()):
            pack = worker.get_latest()
            fail_count = worker.consecutive_failures
            self._fail_counts[name] = fail_count

            if fail_count >= self._fail_restart_thresh:
                if self.verbose:
                    print(f"Restarting RealSense {name} after {fail_count} failed background captures")
                to_restart.append(name)

            if pack is None:
                if self.verbose and fail_count > 0:
                    print(f"{name}: no cached RealSense frame yet")
                continue

            out[name] = self._resize_pack(pack, img_size)

        for name in to_restart:
            try:
                self._restart_one(name)
            except Exception as exc:
                if self.verbose:
                    print(f"Restart RealSense {name} failed: {exc}")

        return out

    def _capture_one(self, name, pipe, img_size=None):
        frameset = self._wait_for_frames(pipe, self.wait_timeout_ms)
        if frameset is None:
            frameset = pipe.poll_for_frames()
            if frameset is None:
                if self.verbose:
                    print(f"{name}: no RealSense frames this cycle")
                return None

        depth0 = frameset.get_depth_frame()
        color0 = frameset.get_color_frame()
        if not depth0 or not color0:
            frameset2 = self._wait_for_frames(pipe, self.wait_timeout_ms)
            if frameset2:
                frameset = frameset2
                depth0 = frameset.get_depth_frame()
                color0 = frameset.get_color_frame()

        aligned = False
        if self.align_to_color and name in self.aligners and depth0 and color0:
            try:
                frameset_aligned = self.aligners[name].process(frameset)
                depth = frameset_aligned.get_depth_frame()
                color = frameset_aligned.get_color_frame()
                if depth and color:
                    frameset = frameset_aligned
                    aligned = True
                elif self.verbose:
                    print(f"{name}: aligned frames missing, using unaligned frames")
            except Exception as exc:
                if self.verbose:
                    print(f"{name}: align failed ({exc}); using unaligned frames")

        depth = frameset.get_depth_frame()
        color = frameset.get_color_frame()
        if not depth or not color:
            if self.verbose:
                suffix = " aligned" if aligned else ""
                print(f"{name}: missing depth/color after{suffix} capture")
            return None

        depth_np = np.asanyarray(depth.get_data())
        try:
            color_np = self._color_frame_to_bgr(color)
        except Exception as exc:
            if self.verbose:
                print(f"{name}: color conversion failed ({exc})")
            return None

        depth_display_max_mm = self.depth_display_max_mm_by_name.get(name, self.depth_display_max_mm)
        depth_display = np.clip(depth_np.astype(np.float32), 0, depth_display_max_mm)
        depth_display = (depth_display / float(depth_display_max_mm) * 255.0).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

        if img_size is not None:
            color_np = cv2.resize(color_np, img_size)
            depth_np = cv2.resize(depth_np, img_size)
            depth_colormap = cv2.resize(depth_colormap, img_size)

        return {
            "color": color_np,
            "depth": depth_np,
            "depth_colormap": depth_colormap,
            "timestamp": time.time(),
        }

    def _resize_pack(self, pack, img_size):
        if img_size is None:
            return pack

        resized = dict(pack)
        resized["color"] = cv2.resize(pack["color"], img_size)
        resized["depth"] = cv2.resize(pack["depth"], img_size)
        resized["depth_colormap"] = cv2.resize(pack["depth_colormap"], img_size)
        return resized

    def _discover_devices(self):
        try:
            devs = list(self.ctx.query_devices())
        except Exception as exc:
            print(f"query_devices failed: {exc}")
            return

        if not devs:
            print("No RealSense devices found.")
            return

        serial_rank = {}
        serial_to_name = {}
        if self.device_serials:
            serial_rank = {serial: i for i, serial in enumerate(self.device_serials)}
            devs = sorted(
                devs,
                key=lambda dev: serial_rank.get(dev.get_info(rs.camera_info.serial_number), len(serial_rank)),
            )
        if self.device_serials_by_name:
            serial_to_name = {serial: name for name, serial in self.device_serials_by_name.items()}

        if self.verbose:
            print(f"Found {len(devs)} RealSense device(s)")

        camera_idx = 0
        for i, dev in enumerate(devs):
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)

            if self.device_serials and serial not in self.device_serials:
                continue
            if self.camera_names and not serial_to_name and camera_idx >= len(self.camera_names):
                break

            base_key = self._device_key(name)
            if serial_to_name:
                key = serial_to_name[serial]
            elif self.camera_names:
                key = self.camera_names[camera_idx]
            else:
                key = self._unique_device_key(base_key)
            camera_idx += 1

            self.devices_info[key] = {
                "serial": serial,
                "name": name,
                "index": i,
                "model_key": base_key,
            }
            self.depth_display_max_mm_by_name[key] = 1000 if base_key == "d405" else self.depth_display_max_mm
            if self.verbose:
                print(f"  - {key}: {name} (S/N: {serial})")

        if not self.devices_info:
            print("No target RealSense devices after filtering.")

    def _start_all(self):
        for name, info in self.devices_info.items():
            ok = self._start_one(name, info["serial"], info["model_key"])
            if not ok:
                print(f"Failed to start RealSense {name}")
        if self.pipelines:
            self.enabled = True
            if self.verbose:
                print(f"Initialized {len(self.pipelines)} RealSense device(s)")
        else:
            self.enabled = False

    def _start_one(self, dev_name, serial, model_key=None):
        stream_candidates = self._stream_candidates(model_key or dev_name)
        last_err = None

        for candidate in stream_candidates:
            pipe = None
            try:
                dw, dh, dfps, cw, ch, cfps = candidate
                if self.verbose:
                    print(
                        f"Trying {dev_name}: depth {dw}x{dh} {dfps}fps, "
                        f"color {cw}x{ch} {cfps}fps"
                    )

                cfg = rs.config()
                pipe = rs.pipeline()
                cfg.enable_device(serial)
                cfg.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, dfps)
                cfg.enable_stream(rs.stream.color, cw, ch, rs.format.any, cfps)

                cfg.resolve(pipe)
                profile = pipe.start(cfg)

                dev = profile.get_device()
                for sensor in dev.sensors:
                    try:
                        sensor.set_option(rs.option.frames_queue_size, 8)
                    except Exception:
                        pass

                self.aligners[dev_name] = rs.align(rs.stream.color if self.align_to_color else rs.stream.depth)

                warm_ok = False
                actual_cf = rs.format.any
                for _ in range(self.warmup_frames):
                    fs = self._wait_for_frames(pipe, self.wait_timeout_ms)
                    if fs:
                        d0 = fs.get_depth_frame()
                        c0 = fs.get_color_frame()
                        if d0 and c0:
                            warm_ok = True
                            actual_cf = c0.get_profile().format()
                            break
                if not warm_ok:
                    if self.verbose:
                        print(f"{dev_name}: warm-up did not get both streams; trying next profile")
                    try:
                        pipe.stop()
                    except Exception:
                        pass
                    self.aligners.pop(dev_name, None)
                    continue

                self.pipelines[dev_name] = pipe
                self.pipeline_profiles[dev_name] = profile
                self._fail_counts[dev_name] = 0

                worker = _RealSenseStreamWorker(
                    dev_name,
                    pipe,
                    self._capture_one,
                    verbose=self.verbose,
                )
                self.workers[dev_name] = worker
                worker.start()

                first_frame_timeout = max(1.0, self.wait_timeout_ms / 1000.0 * 2.0)
                if not self._wait_for_worker_frame(worker, first_frame_timeout):
                    if self.verbose:
                        print(f"{dev_name}: background reader did not cache a frame; trying next profile")
                    self._stop_worker(dev_name)
                    self.pipelines.pop(dev_name, None)
                    self.pipeline_profiles.pop(dev_name, None)
                    self.aligners.pop(dev_name, None)
                    try:
                        pipe.stop()
                    except Exception:
                        pass
                    continue

                if self.verbose:
                    cfmt = self._format_name(actual_cf)
                    print(
                        f"{dev_name} started @ depth {dw}x{dh} {dfps}fps, "
                        f"color {cw}x{ch} {cfps}fps ({cfmt})"
                    )
                return True
            except Exception as exc:
                last_err = exc
                if self.verbose:
                    print(f"{dev_name}: profile failed ({exc})")
                self._stop_worker(dev_name)
                self.pipelines.pop(dev_name, None)
                self.pipeline_profiles.pop(dev_name, None)
                if pipe is not None:
                    try:
                        pipe.stop()
                    except Exception:
                        pass
                self.aligners.pop(dev_name, None)

        print(f"Could not start RealSense {dev_name}: {last_err}")
        return False

    def _color_frame_to_bgr(self, color):
        color_np = np.asanyarray(color.get_data())
        fmt = color.get_profile().format()

        if fmt == rs.format.rgb8:
            return cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR)
        if fmt == rs.format.rgba8:
            return cv2.cvtColor(color_np, cv2.COLOR_RGBA2BGR)
        if fmt == rs.format.bgra8:
            return cv2.cvtColor(color_np, cv2.COLOR_BGRA2BGR)
        if fmt == rs.format.yuyv:
            return cv2.cvtColor(color_np, cv2.COLOR_YUV2BGR_YUY2)
        if fmt == rs.format.mjpeg:
            decoded = cv2.imdecode(color_np, cv2.IMREAD_COLOR)
            if decoded is None:
                raise RuntimeError("failed to decode MJPEG color frame")
            return decoded

        return color_np

    def _stream_candidates(self, dev_name):
        desired = (
            self.desired_w,
            self.desired_h,
            self.desired_fps,
            self.desired_w,
            self.desired_h,
            self.desired_fps,
        )

        base_name = dev_name.split("_", 1)[0]
        if base_name == "d405":
            candidates = [
                desired,
                (640, 480, 30, 640, 480, 30),
                (640, 480, 15, 640, 480, 15),
                (848, 480, 15, 1280, 720, 15),
                (848, 480, 10, 1280, 720, 10),
                (1280, 720, 15, 1280, 720, 15),
                (640, 360, 30, 640, 360, 30),
                (640, 360, 15, 640, 360, 15),
            ]
        else:
            candidates = [
                desired,
                (640, 480, 15, 640, 480, 15),
                (640, 480, 6, 640, 480, 6),
                (848, 480, 15, 848, 480, 15),
                (424, 240, 15, 424, 240, 15),
                (424, 240, 6, 424, 240, 6),
            ]

        return list(dict.fromkeys(candidates))

    @staticmethod
    def _format_name(fmt):
        names = {
            rs.format.bgr8: "BGR8",
            rs.format.rgb8: "RGB8",
            rs.format.rgba8: "RGBA8",
            rs.format.bgra8: "BGRA8",
            rs.format.yuyv: "YUYV",
            rs.format.mjpeg: "MJPEG",
            rs.format.any: "AUTO",
        }
        return names.get(fmt, str(fmt))

    def _restart_one(self, name):
        if name not in self.devices_info:
            return
        info = self.devices_info[name]
        serial = info["serial"]
        self._stop_worker(name)
        try:
            if name in self.pipelines:
                self.pipelines[name].stop()
        except Exception:
            pass
        time.sleep(0.2)
        self.pipelines.pop(name, None)
        self.pipeline_profiles.pop(name, None)
        self.aligners.pop(name, None)
        ok = self._start_one(name, serial, info["model_key"])
        if ok:
            self._fail_counts[name] = 0

    def _stop_worker(self, name):
        worker = self.workers.pop(name, None)
        if worker is None:
            return
        worker.stop()
        worker.join(timeout=max(1.0, self.wait_timeout_ms / 1000.0 + 0.2))

    @staticmethod
    def _wait_for_worker_frame(worker, timeout_seconds):
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if worker.has_latest:
                return True
            if not worker.is_alive():
                return False
            time.sleep(0.02)
        return worker.has_latest

    def _wait_for_frames(self, pipeline, timeout_ms):
        try:
            return pipeline.wait_for_frames(timeout_ms=timeout_ms)
        except Exception:
            return None

    def _unique_device_key(self, base_key):
        if base_key not in self.devices_info:
            return base_key
        index = 2
        while f"{base_key}_{index}" in self.devices_info:
            index += 1
        return f"{base_key}_{index}"

    @staticmethod
    def _device_key(name):
        lowered = (name or "").lower()
        if "d405" in lowered:
            return "d405"
        if "d435i" in lowered or "d430i" in lowered:
            return "d435i"
        if "d435" in lowered:
            return "d435"
        return lowered.replace(" ", "_") or "device"


def make_visualization(color, depth_colormap):
    if color.shape[:2] != depth_colormap.shape[:2]:
        depth_colormap = cv2.resize(depth_colormap, (color.shape[1], color.shape[0]))

    vis = np.hstack((color, depth_colormap))
    cv2.putText(vis, "Color", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(
        vis,
        "Depth",
        (color.shape[1] + 12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    return vis


def _normalize_optional_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Mapping):
        return [str(item) for item in value.keys()]
    return [str(item) for item in value]


def preview_from_collect_config(config_path=None):
    manager = RealSenseManager.from_collect_config(config_path)
    if not manager.is_enabled():
        return

    print("Press 'q' or Esc to exit.")
    try:
        while True:
            frames = manager.capture_frames()
            breakpoint()
            for name, pack in frames.items():
                vis = make_visualization(pack["color"], pack["depth_colormap"])
                cv2.imshow(f"RealSense {name}", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        manager.stop()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Preview RealSense cameras using configs/collect.yaml.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "collect.yaml"),
        help="Path to collect.yaml.",
    )
    args = parser.parse_args()
    preview_from_collect_config(args.config)


if __name__ == "__main__":
    main()
