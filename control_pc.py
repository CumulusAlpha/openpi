import time
import numpy as np
from copy import deepcopy
import threading
from collections import deque
from pathlib import Path
import sys
from types import SimpleNamespace

from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

ACT_ROOT = Path("/home/ubuntu/ACT")
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))

from utils.arx_controller import Arx5BimanualOperator
from utils.realsense_cam import RealSenseManager


# ================= 配置区域 =================
HOST = "192.3.8.52" # 192.3.8.52 127.0.0.1 10.140.66.121
PORT = 8000
TIMEOUT = 15000

FPS = 30
S_MIN = 4 #30  # 3 30

TASK = "Complete_chemical_reaction_experiment"

DELAY = 0 # 强制控制推理延迟
USE_RTC = True

LEFT_MODEL = "X5"
RIGHT_MODEL = "X5"
LEFT_INTERFACE = "can0"
RIGHT_INTERFACE = "can1"

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


class DualArmCollector:
    """ARX bimanual controller used by the OpenPI websocket client."""

    def __init__(self):
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
        self.robot_operator.set_command_active()
        self.robot_operator.command_arms(action, command_delay=1.0 / FPS)

    def go_home(self):
        self.robot_operator.reset_to_home()
        self.robot_operator.set_command_active()

    def close(self):
        if self.realsense_manager is not None:
            self.realsense_manager.stop()
            self.realsense_manager = None
        if self.robot_operator is not None:
            self.robot_operator.set_teach_passive()
            self.robot_operator = None


def convert_obs_to_openpi(obs: dict) -> dict:
    """Convert DualArmCollector obs to the LeRobot 630 schema."""
    images = obs["images"]
    left_qpos = np.asarray(obs["left_qpos"], dtype=np.float32)
    right_qpos = np.asarray(obs["right_qpos"], dtype=np.float32)

    return {
        "images.rgb.head": _resize_image(images["head"]),
        "images.rgb.hand_left": _resize_image(images["left_wrist"]),
        "images.rgb.hand_right": _resize_image(images["right_wrist"]),
        "states.left_joint.position": left_qpos[:6],
        "states.left_gripper.position": left_qpos[6:7],
        "states.right_joint.position": right_qpos[:6],
        "states.right_gripper.position": right_qpos[6:7],
        "prompt": TASK,
    }


def _resize_image(img):
    img = image_tools.convert_to_uint8(img)
    return image_tools.resize_with_pad(img, 224, 224)
    

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
        self.is_running = True
        self.t = 0
        self.A_cur = None
        self.o_cur = None
        
        # 启动推理线程
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
        
        # 首次推理
        self._run_inference_once()

    def stop(self):
        self.is_running = False
        self.inference_trigger.set()
        if self.inference_thread:
            self.inference_thread.join()

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
                
            A_new = self._run_guided_inference(o, s)
            
            # 更新共享状态
            with self.action_lock:
                self.A_cur = A_new
                self.chunk_cur = deepcopy(self.A_cur)
                self.t = self.t - s
                self.delay_queue.append(self.t)
                self.action_chunk_list.append({'chunk_prev': np.array(self.chunk_prev), 
                                               'chunk_cur': np.array(self.chunk_cur), 
                                               'start': s, 'persist': self.t})
                self.is_inferencing = False
            
            print(f"推理完成: 生成{len(A_new)}个动作")

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
            obs_openpi["actions.left_joint.position"] = padded_actions[:, :6]
            obs_openpi["actions.left_gripper.position"] = padded_actions[:, 6:7]
            obs_openpi["actions.right_joint.position"] = padded_actions[:, 7:13]
            obs_openpi["actions.right_gripper.position"] = padded_actions[:, 13:14]
            action_dict = self.client.infer_rtc(
                obs_openpi,
                delay=delay,
            )
        
        actions_array = action_dict["actions"]
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
            self.t += 1
            self.o_cur = self.controller.get_observation()
            if self.A_cur is not None:
                if self.t >= self.s_min and not self.is_inferencing:
                    self.is_inferencing = True
                    self.inference_trigger.set()
            
            # 返回当前动作
            if self.A_cur is None or self.t >= len(self.A_cur):
                print(f"动作不足 (t={self.t}, A_cur长度={len(self.A_cur) if self.A_cur else 0})")
                return None 
            action = self.A_cur[self.t - 1]
        
        return action


def main():
    print("初始化 ARX 机械臂控制器...")
    arx_controller = DualArmCollector()
    time.sleep(2)
    print("✓ ARX initialized")

    arx_controller.go_home()
    time.sleep(0.5)
    
    client = _websocket_client_policy.WebsocketClientPolicy(
        host=HOST, 
        port=PORT,
    )
    async_manager = AsyncInferenceManager(client, arx_controller, s_min=S_MIN)
    all_actions = []
    
    try:
        async_manager.start()
        
        while True:
            loop_start_t = time.perf_counter()
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
        async_manager.stop()
        arx_controller.close()

if __name__ == "__main__":
    main()
