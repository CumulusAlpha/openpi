# -- coding: UTF-8
import collections
import sys
import time
from pathlib import Path

import numpy as np


ARX5_SDK_ROOT = Path('/home/arx/arx5-sdk/python')
if str(ARX5_SDK_ROOT) not in sys.path:
    sys.path.append(str(ARX5_SDK_ROOT))

import arx5_interface as arx5


class Arx5BimanualOperator:
    def __init__(self, args):
        self.left = self._make_controller(args.left_model, args.left_interface)
        self.right = self._make_controller(args.right_model, args.right_interface)
        self.set_teach_passive()

    def _make_controller(self, model, interface):
        robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
        robot_config.gripper_open_readout = -3.4
        robot_config.gripper_width = 0.082

        controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
            "joint_controller", robot_config.joint_dof
        )
        controller_config.background_send_recv = True
        controller_config.over_current_cnt_max = 50

        return arx5.Arx5JointController(robot_config, controller_config, interface)

    def set_teach_passive(self):
        for controller in (self.left, self.right):
            gain = controller.get_gain()
            gain.kp()[:] = 0.0
            gain.kd()[:] *= 0.1
            gain.gripper_kp = 0.0
            gain.gripper_kd = 0.0
            controller.set_gain(gain)

    def open_grippers(self, duration=1.5, settle=0.2, target=None):
        controllers = (self.left, self.right)

        for controller in controllers:
            controller_config = controller.get_controller_config()
            gain = controller.get_gain()
            gain.gripper_kp = getattr(controller_config, "default_gripper_kp", 1.0) or 1.0
            gain.gripper_kd = getattr(controller_config, "default_gripper_kd", 0.05) or 0.05
            controller.set_gain(gain)

        t_end = time.monotonic() + duration
        while time.monotonic() < t_end:
            for controller in controllers:
                robot_config = controller.get_robot_config()
                target_gripper = robot_config.gripper_width if target is None else target
                state = controller.get_joint_state()
                joint_pos = state.pos().copy()
                zeros = np.zeros_like(joint_pos)
                cmd = arx5.JointState(joint_pos, zeros, zeros, target_gripper)
                cmd.timestamp = controller.get_timestamp() + 0.05
                controller.set_joint_cmd(cmd)
            time.sleep(0.02)

        if settle > 0:
            time.sleep(settle)
        self.set_teach_passive()

    def reset_to_home(self):
        self.left.reset_to_home()
        self.right.reset_to_home()
        self.set_teach_passive()

    def set_command_active(self):
        for controller in (self.left, self.right):
            controller_config = controller.get_controller_config()
            gain = controller.get_gain()
            gain.kp()[:] = controller_config.default_kp
            gain.kd()[:] = controller_config.default_kd
            gain.gripper_kp = min(getattr(controller_config, "default_gripper_kp", 1.0) or 1.0, 1.0)
            gain.gripper_kd = min(getattr(controller_config, "default_gripper_kd", 0.05) or 0.05, 0.05)
            controller.set_gain(gain)

    def command_arm(self, controller, arm_action, timestamp):
        joint_pos = np.asarray(arm_action[:6], dtype=np.float64)
        gripper_pos = float(arm_action[6])
        zeros = np.zeros_like(joint_pos)
        cmd = arx5.JointState(joint_pos, zeros, zeros, gripper_pos)
        cmd.timestamp = timestamp
        controller.set_joint_cmd(cmd)

    def command_arms(self, action, command_delay=0.1):
        left_action = np.asarray(action[:7], dtype=np.float64)
        right_action = np.asarray(action[7:14], dtype=np.float64)
        self.command_arm(self.left, left_action, self.left.get_timestamp() + command_delay)
        self.command_arm(self.right, right_action, self.right.get_timestamp() + command_delay)

    def read_arm(self, controller):
        state = controller.get_joint_state()
        qpos = np.concatenate((state.pos().copy(), [state.gripper_pos]), axis=0)
        qvel = np.concatenate((state.vel().copy(), [state.gripper_vel]), axis=0)
        effort = np.concatenate((state.torque().copy(), [state.gripper_torque]), axis=0)

        eef_state = controller.get_eef_state()
        eef = np.concatenate((eef_state.pose_6d().copy(), [state.gripper_pos]), axis=0)
        return qpos, qvel, effort, eef

    def read_arms(self, ts=-1):
        left_qpos, left_qvel, left_effort, left_eef = self.read_arm(self.left)
        right_qpos, right_qvel, right_effort, right_eef = self.read_arm(self.right)

        eef = np.concatenate((left_eef, right_eef), axis=0)
        qpos = np.concatenate((left_qpos, right_qpos), axis=0)
        qvel = np.concatenate((left_qvel, right_qvel), axis=0)
        effort = np.concatenate((left_effort, right_effort), axis=0)
        return qpos, qvel, effort, eef
