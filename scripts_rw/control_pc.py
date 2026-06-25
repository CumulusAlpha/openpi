import os
import time
import numpy as np
from copy import deepcopy
import threading
from collections import deque
from pathlib import Path
import sys
from types import SimpleNamespace
import traceback

FILE = Path(__file__).resolve()
SCRIPT_DIR = FILE.parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (
    REPO_ROOT,
    REPO_ROOT / "src",
    REPO_ROOT / "packages" / "openpi-client" / "src",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from openpi_client import image_tools

from utils.button import CanButtonReader
from utils.realsense_cam import RealSenseManager


# ================= 配置区域 =================
HOST = "10.42.0.188"
PORT = 8080
TIMEOUT = 15000

FPS = 50
S_MIN = 30 
ACTION_SAFETY_CLIP = True
ACTION_LPF_ALPHA = 0.35
MAX_JOINT_STEP = 0.025
MAX_GRIPPER_STEP = 0.02

RIGHT_GRIPPER_BINARY_ACTION = True
RIGHT_GRIPPER_BINARY_DIM = 13
RIGHT_GRIPPER_BINARY_THRESHOLD = 0.5
RIGHT_GRIPPER_OPEN_WIDTH = 0.082
RIGHT_GRIPPER_CLOSED_WIDTH = 0.028
RIGHT_GRIPPER_DEBUG = True
RIGHT_GRIPPER_DEBUG_EVERY_N = 10
RIGHT_GRIPPER_DEBUG_EVERY_CLOSE_STEP = True
DEBUG_LOG_PATH = REPO_ROOT / "logs" / "control_pc_debug.log"

TASK = "Complete_chemical_reaction_experiment"

DELAY = 0 # 强制控制推理延迟
USE_RTC = False

LEFT_MODEL = "X5"
RIGHT_MODEL = "X5"
LEFT_INTERFACE = "can0"
RIGHT_INTERFACE = "can1"
BUTTON_INTERFACE = "can6"
BUTTON_CAN_ID = 0x721

# CanButtonReader reports zero-based indices. These are physical buttons 1, 2, 3.
BUTTON_HOME = 0
BUTTON_START_INFERENCE = 1
BUTTON_STOP_INFERENCE = 2

CAMERA_NAMES = ["left", "middle", "right"]
REALSENSE_SERIALS = {
    "left": "260322275038",
    "middle": "260322276842",
    "right": "260322272800",
}
REALSENSE_WIDTH = 640
REALSENSE_HEIGHT = 480
REALSENSE_FPS = 90
REALSENSE_WARMUP_FRAMES = 30
REALSENSE_WAIT_TIMEOUT_MS = 1000
REALSENSE_ALIGN_TO_COLOR = True
REALSENSE_VERBOSE = True

IMAGE_CAMERA_MAP = {
    "head": "middle",
    "left_wrist": "left",
    "right_wrist": "right",
}

# ===========================================

def _add_host_to_no_proxy(host: str):
    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        entries = [entry.strip() for entry in current.split(",") if entry.strip()]
        if host not in entries:
            entries.append(host)
            os.environ[key] = ",".join(entries)


def _debug_print(message: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


class DualArmCollector:
    """ARX bimanual controller used by the OpenPI websocket client."""

    def __init__(self):
        from utils.arx_controller import Arx5BimanualOperator

        self.args = SimpleNamespace(
            left_model=LEFT_MODEL,
            right_model=RIGHT_MODEL,
            left_interface=LEFT_INTERFACE,
            right_interface=RIGHT_INTERFACE,
        )
        self.robot_operator = Arx5BimanualOperator(self.args)
        self.robot_operator.set_command_active()
        self.realsense_manager = RealSenseManager(
            desired_width=REALSENSE_WIDTH,
            desired_height=REALSENSE_HEIGHT,
            desired_fps=REALSENSE_FPS,
            warmup_frames=REALSENSE_WARMUP_FRAMES,
            wait_timeout_ms=REALSENSE_WAIT_TIMEOUT_MS,
            device_serials=REALSENSE_SERIALS,
            camera_names=CAMERA_NAMES,
            align_to_color=REALSENSE_ALIGN_TO_COLOR,
            verbose=REALSENSE_VERBOSE,
        )
        self.last_commanded_action = None
        self._execute_count = 0
        self._last_right_gripper_decision = None
        self._last_commanded_right_gripper = None

    def get_observation(self) -> dict:
        qpos, qvel, effort, eef = self.robot_operator.read_arms()
        frames = self.realsense_manager.capture_frames()
        missing = [
            camera_name
            for camera_name in IMAGE_CAMERA_MAP.values()
            if camera_name not in frames
        ]
        if missing:
            raise RuntimeError(f"Missing RealSense frames for cameras: {missing}")

        return {
            "images": {
                image_key: frames[camera_name]["color"]
                for image_key, camera_name in IMAGE_CAMERA_MAP.items()
            },
            "left_qpos": qpos[:7].astype(np.float32),
            "right_qpos": qpos[7:14].astype(np.float32),
            "qpos": qpos.astype(np.float32),
            "qvel": qvel.astype(np.float32),
            "effort": effort.astype(np.float32),
            "eef": eef.astype(np.float32),
        }

    def execute_action(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape[0] != 14:
            raise ValueError(f"ARX action should have 14 values, got shape {action.shape}")
        raw_action = action.copy()
        raw_gripper_value = float(raw_action[RIGHT_GRIPPER_BINARY_DIM])
        raw_decision = "close" if raw_gripper_value > RIGHT_GRIPPER_BINARY_THRESHOLD else "open"
        previous_commanded_gripper = self._last_commanded_right_gripper

        action = self._map_binary_right_gripper_action(action)
        mapped_gripper = float(action[RIGHT_GRIPPER_BINARY_DIM])

        if self.last_commanded_action is None:
            qpos, _, _, _ = self.robot_operator.read_arms()
            self.last_commanded_action = qpos.astype(np.float64)
        action = ACTION_LPF_ALPHA * action + (1.0 - ACTION_LPF_ALPHA) * self.last_commanded_action
        lpf_gripper = float(action[RIGHT_GRIPPER_BINARY_DIM])
        lpf_gap = lpf_gripper - mapped_gripper
        gripper_delta_before_clip = None
        gripper_delta_after_clip = None
        gripper_clip_amount = 0.0
        gripper_was_clipped = False
        if ACTION_SAFETY_CLIP:
            qpos, _, _, _ = self.robot_operator.read_arms()
            step_limits = np.array([MAX_JOINT_STEP] * 6 + [MAX_GRIPPER_STEP] + [MAX_JOINT_STEP] * 6 + [MAX_GRIPPER_STEP])
            unclipped_delta = action - qpos
            delta = np.clip(unclipped_delta, -step_limits, step_limits)
            clipped_action = qpos + delta
            gripper_delta_before_clip = float(unclipped_delta[RIGHT_GRIPPER_BINARY_DIM])
            gripper_delta_after_clip = float(delta[RIGHT_GRIPPER_BINARY_DIM])
            gripper_clip_amount = gripper_delta_before_clip - gripper_delta_after_clip
            gripper_was_clipped = abs(gripper_clip_amount) > 1e-6
            if np.max(np.abs(action - clipped_action)) > 1e-6:
                print(
                    "[safety] action clipped "
                    f"raw_delta_max={np.max(np.abs(unclipped_delta)):.4f} "
                    f"sent_delta_max={np.max(np.abs(delta)):.4f}"
                )
            action = clipped_action
        else:
            qpos, _, _, _ = self.robot_operator.read_arms()
            gripper_delta_before_clip = float(action[RIGHT_GRIPPER_BINARY_DIM] - qpos[RIGHT_GRIPPER_BINARY_DIM])
            gripper_delta_after_clip = gripper_delta_before_clip

        self._execute_count += 1
        final_gripper = float(action[RIGHT_GRIPPER_BINARY_DIM])
        current_gripper = float(qpos[RIGHT_GRIPPER_BINARY_DIM])
        target_gap = final_gripper - mapped_gripper
        should_log = (
            RIGHT_GRIPPER_DEBUG
            and (
                self._execute_count % RIGHT_GRIPPER_DEBUG_EVERY_N == 0
                or (RIGHT_GRIPPER_DEBUG_EVERY_CLOSE_STEP and raw_decision == "close")
                or raw_decision != self._last_right_gripper_decision
            )
        )
        if should_log:
            _debug_print(
                "[gripper-exec] "
                f"step={self._execute_count} "
                f"raw_action13={raw_gripper_value:.5f} "
                f"threshold={RIGHT_GRIPPER_BINARY_THRESHOLD:.2f} "
                f"decision={raw_decision} "
                f"mapped_target={mapped_gripper:.5f} "
                f"current_qpos13={current_gripper:.5f} "
                f"prev_cmd13={previous_commanded_gripper if previous_commanded_gripper is not None else 'None'} "
                f"after_lpf={lpf_gripper:.5f} "
                f"lpf_gap_to_target={lpf_gap:+.5f} "
                f"delta_before_clip={gripper_delta_before_clip:+.5f} "
                f"delta_after_clip={gripper_delta_after_clip:+.5f} "
                f"clip_amount={gripper_clip_amount:+.5f} "
                f"gripper_clipped={gripper_was_clipped} "
                f"final_cmd13={final_gripper:.5f} "
                f"cmd_delta={final_gripper - current_gripper:+.5f} "
                f"final_gap_to_target={target_gap:+.5f}"
            )
            self._last_right_gripper_decision = raw_decision

        self.last_commanded_action = action.copy()
        self._last_commanded_right_gripper = final_gripper
        self.robot_operator.command_arms(action, command_delay=1.0 / FPS)

    def _map_binary_right_gripper_action(self, action):
        if not RIGHT_GRIPPER_BINARY_ACTION:
            return action

        action = action.copy()
        close_score = float(action[RIGHT_GRIPPER_BINARY_DIM])
        action[RIGHT_GRIPPER_BINARY_DIM] = (
            RIGHT_GRIPPER_CLOSED_WIDTH
            if close_score > RIGHT_GRIPPER_BINARY_THRESHOLD
            else RIGHT_GRIPPER_OPEN_WIDTH
        )
        return action

    def go_home(self):
        self.robot_operator.reset_to_home()
        self.robot_operator.set_command_active()
        self.last_commanded_action = None
        self._last_right_gripper_decision = None
        self._last_commanded_right_gripper = None

    def close(self):
        if self.realsense_manager is not None:
            self.realsense_manager.stop()
            self.realsense_manager = None
        if self.robot_operator is not None:
            self.robot_operator.set_teach_passive()
            self.robot_operator = None


def convert_obs_to_openpi(obs: dict) -> dict:
    """Convert DualArmCollector obs to the ALOHA policy server schema."""
    images = obs["images"]
    left_qpos = np.asarray(obs["left_qpos"], dtype=np.float32)
    right_qpos = np.asarray(obs["right_qpos"], dtype=np.float32)
    state = np.concatenate([left_qpos[:7], right_qpos[:7]]).astype(np.float32)

    return {
        "images": {
            "cam_high": _resize_image_chw(images["head"]),
            "cam_left_wrist": _resize_image_chw(images["left_wrist"]),
            "cam_right_wrist": _resize_image_chw(images["right_wrist"]),
        },
        "state": state,
        "prompt": TASK,
    }


def _resize_image_chw(img):
    img = image_tools.convert_to_uint8(img)
    img = image_tools.resize_with_pad(img, 224, 224)
    return np.asarray(img).transpose(2, 0, 1)
    

class AsyncInferenceManager:
    def __init__(self, client, controller: DualArmCollector, s_min: int):
        self.client = client
        self.controller = controller

        self.t = 0
        self.A_cur = None
        self.o_cur = None  # 当前观测
        self.chunk_prev = None
        self.chunk_cur = None

        self.s_min = s_min # 最小执行数量
        self.delay_queue = deque(maxlen=10)
        self.delay_queue.append(5)
        
        # 状态标志
        self.is_running = False
        self.is_inferencing = False
        self.inference_thread = None
        self.inference_trigger = threading.Event()
        self.action_lock = threading.Lock()

        # RTC数据记录
        self.rtc_data_list = []
        self.action_chunk_list = []


    def start(self):
        if self.is_running:
            print("推理已经在运行")
            return

        self.is_running = True
        self.is_inferencing = False
        self.t = 0
        self.A_cur = None
        self.o_cur = None
        self.chunk_prev = None
        self.chunk_cur = None
        self.delay_queue.clear()
        self.delay_queue.append(5)
        self.inference_trigger.clear()
        self.controller.last_commanded_action = None
        self.controller._last_commanded_right_gripper = None
        self.controller._last_right_gripper_decision = None
        if RIGHT_GRIPPER_DEBUG:
            DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            _debug_print(
                "[run-start] "
                f"task={TASK!r} "
                f"fps={FPS} "
                f"binary_action={RIGHT_GRIPPER_BINARY_ACTION} "
                f"dim={RIGHT_GRIPPER_BINARY_DIM} "
                f"threshold={RIGHT_GRIPPER_BINARY_THRESHOLD:.2f} "
                f"open_width={RIGHT_GRIPPER_OPEN_WIDTH:.5f} "
                f"closed_width={RIGHT_GRIPPER_CLOSED_WIDTH:.5f} "
                f"lpf_alpha={ACTION_LPF_ALPHA:.3f} "
                f"safety_clip={ACTION_SAFETY_CLIP} "
                f"max_gripper_step={MAX_GRIPPER_STEP:.5f}"
            )
        
        # 启动推理线程
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
        
        # 首次推理
        with self.action_lock:
            self.o_cur = self.controller.get_observation()
            self.is_inferencing = True
        self.inference_trigger.set()

    def stop(self):
        if not self.is_running and self.inference_thread is None:
            return

        self.is_running = False
        with self.action_lock:
            self.is_inferencing = False
        self.inference_trigger.set()
        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join()
        self.inference_thread = None
        self.inference_trigger.clear()
        with self.action_lock:
            self.t = 0
            self.A_cur = None
            self.o_cur = None
            self.chunk_prev = None
            self.chunk_cur = None
            self.is_inferencing = False
        self.controller.last_commanded_action = None
        self.controller._last_commanded_right_gripper = None
        if RIGHT_GRIPPER_DEBUG:
            _debug_print("[run-stop]")

    def _inference_loop(self):
        while self.is_running:
            self.inference_trigger.wait()
            if not self.is_running:
                break
            
            self.inference_trigger.clear()   
            with self.action_lock:
                s = self.t
                o = self.o_cur
                self.chunk_prev = deepcopy(self.A_cur) # 保存上一个chunk
                
            try:
                A_new = self._run_guided_inference(o, s)
            except Exception:
                traceback.print_exc()
                with self.action_lock:
                    self.is_inferencing = False
                continue

            if not self.is_running:
                with self.action_lock:
                    self.is_inferencing = False
                continue
            
            # 更新共享状态
            with self.action_lock:
                elapsed_steps = max(self.t - s, 0)
                self.A_cur = A_new
                self.chunk_cur = deepcopy(self.A_cur)
                self.t = min(elapsed_steps, max(len(A_new) - 2, 0))
                self.delay_queue.append(self.t)
                self.action_chunk_list.append({'chunk_prev': np.array(self.chunk_prev), 
                                               'chunk_cur': np.array(self.chunk_cur), 
                                               'start': s, 'persist': self.t})
                self.is_inferencing = False
            
            print(f"推理完成: 生成{len(A_new)}个动作, elapsed_steps={elapsed_steps}, resume_t={self.t}")

    def _run_guided_inference(self, obs_dict, s):
        # import pdb; pdb.set_trace()
        start = time.perf_counter()
        obs_openpi = convert_obs_to_openpi(obs_dict)
        
        if not USE_RTC or self.chunk_prev is None: # 第一次推理
            action_dict = self.client.infer(obs_openpi)
            print("不使用RTC")

        else:
            overlap = max(self.delay_queue)
            delay = np.array([overlap], dtype=np.int32)
            # 截断到 S_MIN:
            sliced_actions = np.array(self.chunk_prev)[self.s_min:]
            pad_len = 50 - len(sliced_actions)
            padded_actions = np.pad(
                sliced_actions, 
                pad_width=((0, pad_len), (0, 0)), 
                mode='constant', 
                constant_values=0.0
            )
            obs_openpi["actions"] = padded_actions
            action_dict = self.client.infer_rtc(
                obs_openpi,
                delay=delay,
            )
        
        actions_array = action_dict["actions"]
        if RIGHT_GRIPPER_DEBUG:
            action13 = np.asarray(actions_array, dtype=np.float32)[:, RIGHT_GRIPPER_BINARY_DIM]
            close_count = int(np.sum(action13 > RIGHT_GRIPPER_BINARY_THRESHOLD))
            preview = np.array2string(action13[:10], precision=4, separator=", ")
            current_right_gripper = float(obs_dict["right_qpos"][6])
            _debug_print(
                "[gripper-chunk] "
                f"dim={RIGHT_GRIPPER_BINARY_DIM} "
                f"current_right_qpos={current_right_gripper:.5f} "
                f"min={np.min(action13):.5f} "
                f"max={np.max(action13):.5f} "
                f"mean={np.mean(action13):.5f} "
                f"close_count={close_count}/{len(action13)} "
                f"first10={preview}"
            )
        new_actions = [actions_array[i] for i in range(len(actions_array))]
        end = time.perf_counter()
        # 强制延迟
        print(f"[client] inference_ms={(end-start)*1000.0:.3f}")
        if end - start < DELAY / 1000.0:
            time.sleep(DELAY / 1000.0 - (end - start))

        return new_actions

    def _run_inference_once(self):
        """首次推理"""
        obs_dict = self.controller.get_observation()
        self.A_cur = self._run_guided_inference(obs_dict, 0)

    def get_next_action(self):
        with self.action_lock:
            if not self.is_running:
                return None

            if self.A_cur is None:
                return None

            self.t += 1
            self.o_cur = self.controller.get_observation()
            if self.t >= self.s_min and not self.is_inferencing:
                self.is_inferencing = True
                self.inference_trigger.set()
            
            # 返回当前动作
            if self.t >= len(self.A_cur):
                print(f"动作不足 (t={self.t}, A_cur长度={len(self.A_cur) if self.A_cur else 0})")
                return None 
            action = self.A_cur[self.t - 1]
        
        return action


def handle_button_events(button_reader, async_manager, arx_controller):
    triggered = button_reader.poll_events()

    if BUTTON_HOME in triggered:
        print("[button 1] home")
        async_manager.stop()
        arx_controller.go_home()

    if BUTTON_START_INFERENCE in triggered:
        print("[button 2] start inference")
        async_manager.start()

    if BUTTON_STOP_INFERENCE in triggered:
        print("[button 3] stop inference")
        async_manager.stop()


def main():
    _add_host_to_no_proxy(HOST)

    try:
        from openpi_client import websocket_client_policy as _websocket_client_policy
    except ImportError as exc:
        raise ImportError("openpi_client.websocket_client_policy requires `websockets`. Run dependency sync/install first.") from exc

    arx_controller = None
    async_manager = None
    button_reader = None
    all_actions = []
    
    try:
        print("初始化 ARX 机械臂控制器...")
        arx_controller = DualArmCollector()
        time.sleep(2)
        print("✓ ARX initialized")

        button_reader = CanButtonReader(BUTTON_INTERFACE, BUTTON_CAN_ID)

        arx_controller.go_home()
        time.sleep(0.5)
        
        client = _websocket_client_policy.WebsocketClientPolicy(
            host=HOST, 
            port=PORT,
        )
        async_manager = AsyncInferenceManager(client, arx_controller, s_min=S_MIN)
        print("等待按钮输入: button 1=home, button 2=start inference, button 3=stop inference")
        
        while True:
            loop_start_t = time.perf_counter()
            handle_button_events(button_reader, async_manager, arx_controller)
            if not async_manager.is_running:
                time.sleep(0.02)
                continue

            action = async_manager.get_next_action()
            if action is None:
                print("等待动作生成中...")
                time.sleep(0.02)
                continue
            loop_request_t = time.perf_counter()
            print(f"[client] websocket_request_interval_ms={(loop_request_t-loop_start_t)*1000.0:.3f}")
            all_actions.append(action)
            arx_controller.execute_action(action)     
            loop_work_t = time.perf_counter()
            work_s = loop_work_t - loop_start_t
            sleep_s = max(1.0 / FPS - work_s, 0.0)
            time.sleep(sleep_s) # 更准确的控制频率
            loop_end_t = time.perf_counter()
            print(f"[client] execution_loop_interval_ms={(loop_end_t-loop_start_t)*1000.0:.3f}")

    except KeyboardInterrupt:
        print("\n停止运行")
    finally:
        if async_manager is not None:
            async_manager.stop()
        if button_reader is not None:
            button_reader.close()
        if arx_controller is not None:
            arx_controller.close()

if __name__ == "__main__":
    main()
