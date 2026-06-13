# -- coding: UTF-8
import os
import shutil
import subprocess
import sys
import html

from pathlib import Path

FILE = Path(__file__).resolve()
SCRIPT_DIR = FILE.parent
REPO_ROOT = SCRIPT_DIR.parent
ROOT = REPO_ROOT
for path in (
    REPO_ROOT,
    REPO_ROOT / "src",
    REPO_ROOT / "packages" / "openpi-client" / "src",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.chdir(str(REPO_ROOT))

from utils.runtime_log import setup_runtime_log

setup_runtime_log(ROOT, "collect.log")

import time
import cv2
import threading
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

import numpy as np

from utils.button import CanButtonReader
from utils.can import ensure_can_interfaces
from utils.config import config_to_namespace
from utils.realsense_cam import RealSenseManager
from utils.utils import precise_wait

np.set_printoptions(linewidth=200)

voice_lock = threading.Lock()
log_lock = threading.Lock()
log_lines = []
log_gui_handle = None
_spd_say = shutil.which("spd-say")
_voice_warning_shown = False

VALID_RECORD_MODES = {'Distance', 'Speed'}
DEFAULT_VISER_URDF_PATH = Path('/home/arx/arx5-sdk/models/X5.urdf')
SAVE_IMAGE_WIDTH = 640
SAVE_IMAGE_HEIGHT = 480
BUTTON_HOME = 0
BUTTON_START = 1
BUTTON_NEXT_SAVE = 2
BUTTON_CANCEL = 3


def _init_voice_engine():
    if pyttsx3 is None:
        return None
    try:
        engine = pyttsx3.init()
        engine.setProperty('voice', 'en')
        engine.setProperty('rate', 120)  # 设置语速
        return engine
    except Exception as exc:
        print(f"pyttsx3 init failed, fallback to spd-say: {exc}")
        return None


voice_engine = _init_voice_engine()


def _get_arg(args, names, default=None):
    for name in names:
        if hasattr(args, name):
            return getattr(args, name)
    return default


def _as_bool(value):
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _viser_requested(args):
    return (
        _as_bool(_get_arg(args, ("viser", "enable_viser", "viser_visualize"), False))
        or _get_arg(args, ("viser_urdf_path", "urdf_path"), None) is not None
    )


def log_message(line):
    global log_gui_handle
    line = str(line)
    print(line)
    with log_lock:
        log_lines.append(line)
        del log_lines[:-12]
        if log_gui_handle is not None:
            log_text = "\n".join(log_lines)
            log_gui_handle.content = (
                "<div style='font-size: 24px; line-height: 1.45; font-weight: 700; "
                "white-space: pre-wrap;'>"
                f"{html.escape(log_text)}"
                "</div>"
            )


def _warn_voice_backend_once(message):
    global _voice_warning_shown
    if not _voice_warning_shown:
        print(message)
        _voice_warning_shown = True


def _say_with_pyttsx3(engine, line):
    global voice_engine
    if engine is None:
        return False
    try:
        engine.say(line)
        engine.runAndWait()
        return True
    except Exception as exc:
        voice_engine = None
        _warn_voice_backend_once(f"pyttsx3 voice failed, fallback to spd-say: {exc}")
        return False


def _say_with_spd(line):
    if _spd_say is None:
        _warn_voice_backend_once("No voice backend available: install pyttsx3/espeak or speech-dispatcher.")
        return False
    try:
        subprocess.run(
            [_spd_say, "-w", "-l", "en", "-r", "-40", str(line)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return True
    except Exception as exc:
        _warn_voice_backend_once(f"spd-say voice failed: {exc}")
        return False


def voice_process(voice_engine, line):
    with voice_lock:
        if not _say_with_pyttsx3(voice_engine, line):
            _say_with_spd(line)
        log_message(line)

    return


class CollectRunner:
    def __init__(self, args):
        self.args = args
        self.viser_update = _as_bool(_get_arg(self.args, ("viser_update", "update_viser"), False))
        self.viser_enabled = _viser_requested(self.args) or self.viser_update
        self.viser_urdf_path = Path(_get_arg(self.args, ("viser_urdf_path", "urdf_path"), DEFAULT_VISER_URDF_PATH))
        self.viser = None
        self.viser_server = None
        self.viser_urdfs = []
        self.img_gui_handles = {}

        self.arm_init()
        self.button_reader = CanButtonReader(self.args.button_interface, self.args.button_can_id)

        self.realsense_manager = RealSenseManager(
            desired_width=self.args.realsense_width,
            desired_height=self.args.realsense_height,
            desired_fps=self.args.realsense_fps,
            warmup_frames=_get_arg(self.args, ("realsense_warmup_frames",), 30),
            wait_timeout_ms=_get_arg(self.args, ("realsense_wait_timeout_ms",), 1000),
            device_serials=_get_arg(self.args, ("realsense_serials",), None),
            camera_names=self.args.camera_names,
            align_to_color=_get_arg(self.args, ("realsense_align_to_color",), True),
            verbose=self.args.realsense_verbose,
        )
        
        if self.viser_enabled:
            self.viser_init()
        datasets_dir = args.datasets if sys.stdin.isatty() else Path.joinpath(ROOT, args.datasets)
        self.datasets_dir = str(datasets_dir)
        args.datasets = self.datasets_dir

    def arm_init(self):
        from utils.arx_controller import Arx5BimanualOperator

        ensure_can_interfaces([self.args.left_interface, self.args.right_interface], root=ROOT)
        self.robot_operator: Arx5BimanualOperator = Arx5BimanualOperator(self.args)
        voice_process(voice_engine, "open")
        self.robot_operator.open_grippers(
            duration=self.args.gripper_open_duration,
            settle=self.args.gripper_open_settle,
            target=self.args.gripper_open_target,
        )

    def viser_init(self):  
        global log_gui_handle
        try:
            import viser
            from viser.extras import ViserUrdf
        except ImportError as exc:
            raise ImportError("viser is required when viser visualization is enabled. Install `viser`.") from exc

        self.viser = viser.ViserServer()
        self.viser_server = self.viser
        self.viser_urdfs = []
        self.img_gui_handles = {}
        for root_node_name, position in (
            ("/left_x5", (0.0, 0.35, 0.0)),
            ("/right_x5", (0.0, -0.35, 0.0)),
        ):
            self.viser.scene.add_frame(root_node_name, position=position, show_axes=False)
            self.viser_urdfs.append(
                ViserUrdf(
                    self.viser,
                    self.viser_urdf_path,
                    root_node_name=root_node_name,
                )
            )

        with self.viser.gui.add_folder("📷 Cameras", expand_by_default=True):
            for cam in self.args.camera_names:
                init_img = np.zeros((240, 320, 3), dtype=np.uint8)
                self.img_gui_handles[cam] = self.viser.gui.add_image(
                    init_img,
                    label=f"{cam} image"
                )

        with self.viser.gui.add_folder("Messages", expand_by_default=True):
            log_text = "\n".join(log_lines) if log_lines else "Ready"
            log_gui_handle = self.viser.gui.add_markdown(
                "<div style='font-size: 24px; line-height: 1.45; font-weight: 700; "
                "white-space: pre-wrap;'>"
                f"{html.escape(log_text)}"
                "</div>"
            )

    def update_viser(self, qpos, frames):
        arm_qpos = (qpos[:6], qpos[7:13])
        for urdf, arm_cfg in zip(self.viser_urdfs, arm_qpos):
                urdf.update_cfg(np.asarray(arm_cfg, dtype=np.float32))
    
        for cam_name in self.args.camera_names:
            frame = frames.get(cam_name)
            if frame is not None:
                self.img_gui_handles[cam_name].image = cv2.cvtColor(frame["color"], cv2.COLOR_BGR2RGB)

    def wait_for_record_start(self, current_episode):
        log_message(f"Waiting to start episode {current_episode}")

        if self.args.key_collect:
            input("Enter any key to record:")
        else:
            while True:
                triggered = self.button_reader.poll_events()
                if BUTTON_HOME in triggered:
                    voice_process(voice_engine, "home")
                    self.robot_operator.reset_to_home()
                if BUTTON_START in triggered:
                    break
                time.sleep(0.02)

        voice_process(voice_engine, f"{current_episode % 100}")
        voice_process(voice_engine, "go")

    def get_obs(self):
        qpos, qvel, effort, eef = self.robot_operator.read_arms()
        frames = self.realsense_manager.capture_frames() if self.realsense_manager is not None else {}
        obs_dict = {
            'qpos': qpos,
            'qvel': qvel,
            'effort': effort,
            'eef': eef
        }
        if self.viser_update:
            self.update_viser(qpos, frames)

        for cam_name in self.args.camera_names:
            obs_dict[f'{cam_name}_image'] = frames[cam_name]['color']
            if self.args.use_depth_image:
                obs_dict[f'{cam_name}_depth_image'] = frames[cam_name]['depth_colormap']
        return obs_dict

    def collect_information(self):
        args = self.args
        observations = []
        actions = []
        actions_eef = []
        count = 0
        frame_interval = 1.0 / args.frame_rate
        home_after_save = False
        canceled = False

        while count < args.max_timesteps:
            start_time = time.monotonic()
            obs_dict = self.get_obs()
            if self.button_reader is not None:
                triggered = self.button_reader.poll_events()
                if BUTTON_HOME in triggered:
                    voice_process(voice_engine, "home")
                    self.robot_operator.reset_to_home()
                    continue
                if BUTTON_NEXT_SAVE in triggered:
                    voice_process(voice_engine, "next")
                    home_after_save = args.home_after_next
                    break
                if BUTTON_CANCEL in triggered:
                    voice_process(voice_engine, "cancel")
                    canceled = True
                    break
            action = obs_dict['qpos'].copy()
            action_eef = obs_dict['eef'].copy()

            observations.append(obs_dict)
            actions.append(action)
            actions_eef.append(action_eef)
            count += 1
            precise_wait(start_time + frame_interval)
        log_message(f"len(observations): {len(observations)}")
        log_message(f"len(actions)  : {len(actions)}")

        return observations, actions, actions_eef, home_after_save, canceled

    def _next_episode(self):
        max_episode = -1
        if os.path.exists(self.datasets_dir):
            for filename in os.listdir(self.datasets_dir):
                if filename.startswith('episode_') and filename.endswith('.hdf5'):
                    try:
                        episode_num = int(filename.split('_')[1].split('.')[0])
                        max_episode = max(max_episode, episode_num)
                    except ValueError:
                        continue

        if max_episode >= 0:
            return max_episode + 1
        if self.args.episode_idx == -1:
            return 0
        return self.args.episode_idx

    def run(self):
        args = self.args
        num_episodes = 1000 if args.episode_idx == -1 else 1
        current_episode = self._next_episode()

        try:
            saved_episodes = 0
            while saved_episodes < num_episodes:
                log_message(f'Episode {saved_episodes}')
                log_message(f"Start to record episode {current_episode}")
                self.wait_for_record_start(current_episode)
                observations, actions, actions_eef, home_after_save, canceled = self.collect_information()

                if canceled:
                    log_message(f"Canceled episode {current_episode}; not saved")
                    continue

                if not os.path.exists(self.datasets_dir):
                    os.makedirs(self.datasets_dir)

                dataset_path = os.path.join(self.datasets_dir, "episode_" + str(current_episode))
                save_thread = threading.Thread(
                    target=save_data,
                    args=(args, observations, actions, actions_eef, dataset_path,),
                )
                save_thread.start()
                if home_after_save:
                    save_thread.join()
                    voice_process(voice_engine, "home")
                    self.robot_operator.reset_to_home()

                current_episode += 1
                saved_episodes += 1
        finally:
            self.close()

    def close(self):
        if self.realsense_manager is not None:
            self.realsense_manager.stop()
            self.realsense_manager = None
        if self.button_reader is not None:
            self.button_reader.close()
            self.button_reader = None


def prepare_image_for_save(image):
    image = np.asarray(image)
    if image.shape[:2] != (SAVE_IMAGE_HEIGHT, SAVE_IMAGE_WIDTH):
        image = cv2.resize(image, (SAVE_IMAGE_WIDTH, SAVE_IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def create_and_write_hdf5(args, data_dict, dataset_path, data_size):
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required to save collected datasets. Run dependency sync/install first.") from exc

    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = False
        root.attrs['task'] = str(args.task)
        root.attrs['frame_rate'] = float(args.frame_rate)
        root.attrs['camera_names'] = np.array(args.camera_names, dtype=h5py.string_dtype(encoding='utf-8'))
        root.attrs['use_depth_image'] = bool(args.use_depth_image)
        root.attrs['use_base'] = bool(args.use_base)

        obs_dict = root.create_group('observations')
        for cam_name in args.camera_names:
            img_shape = (data_size, SAVE_IMAGE_HEIGHT, SAVE_IMAGE_WIDTH, 3)
            img_chunk = (1, SAVE_IMAGE_HEIGHT, SAVE_IMAGE_WIDTH, 3)
            if args.use_depth_image:
                depth_shape = (data_size, SAVE_IMAGE_HEIGHT, SAVE_IMAGE_WIDTH, 3)
                depth_chunk = (1, SAVE_IMAGE_HEIGHT, SAVE_IMAGE_WIDTH, 3)

            obs_dict.create_dataset(f'{cam_name}_image', img_shape, 'uint8', chunks=img_chunk)
            if args.use_depth_image:
                obs_dict.create_dataset(f'{cam_name}_depth_image', depth_shape, 'uint8', chunks=depth_chunk)

        # 创建观测和动作数据集
        state_dim = 14
        eef_dim = 14
        obs_specs = {'qpos': state_dim, 'eef': eef_dim, 'qvel': state_dim, 'effort': state_dim}
        act_specs = {'action': state_dim, 'action_eef': eef_dim}

        for name, dim in obs_specs.items():
            obs_dict.create_dataset(name, (data_size, dim), dtype=np.float32)
        for name, dim in act_specs.items():
            root.create_dataset(name, (data_size, dim), dtype=np.float32)

        for name, arr in data_dict.items():
            root[name][...] = arr


# 保存数据函数
def save_data(args, observations, actions, actions_eef, dataset_path):
    data_size = len(actions)

    # 数据字典
    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/observations/eef': [],
        '/action': [],
        '/action_eef': [],
    }

    # 初始化相机字典
    for cam_name in args.camera_names:
        data_dict[f'/observations/{cam_name}_image'] = []
        if args.use_depth_image:
            data_dict[f'/observations/{cam_name}_depth_image'] = []

    # 遍历并收集数据
    while actions:
        action = actions.pop(0)  # 动作  当前动作
        action_eef = actions_eef.pop(0)
        obs = observations.pop(0)  # 观察值

        # 填充数据
        data_dict['/observations/qpos'].append(obs['qpos'])
        data_dict['/observations/qvel'].append(obs['qvel'])
        data_dict['/observations/eef'].append(obs['eef'])
        data_dict['/observations/effort'].append(obs['effort'])
        data_dict['/action'].append(action)
        data_dict['/action_eef'].append(action_eef)

        # 相机数据
        for cam_name in args.camera_names:
            data_dict[f'/observations/{cam_name}_image'].append(
                prepare_image_for_save(obs[f'{cam_name}_image'])
            )
            if args.use_depth_image:
                data_dict[f'/observations/{cam_name}_depth_image'].append(
                    prepare_image_for_save(obs[f'{cam_name}_depth_image'])
                )

    t0 = time.time()
    create_and_write_hdf5(args, data_dict, dataset_path, data_size)

    voice_process(voice_engine, "Save")
    log_message(f"Saved in {time.time() - t0:.1f}s: {dataset_path}")

    return


def main(args):
    CollectRunner(args).run()

def hydra_entry():
    try:
        import hydra
    except ImportError as exc:
        raise ImportError("hydra-core is required to run collect.py as a CLI. Run dependency sync/install first.") from exc

    @hydra.main(version_base=None, config_path="configs", config_name="collect")
    def _entry(cfg):
        main(config_to_namespace(cfg))

    _entry()


if __name__ == '__main__':
    hydra_entry()
