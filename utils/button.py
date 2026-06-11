# -- coding: UTF-8
import socket
import struct


CAN_FRAME_FORMAT = "=IB3x8s"
CAN_FRAME_SIZE = struct.calcsize(CAN_FRAME_FORMAT)
CAN_EFF_FLAG = 0x80000000
CAN_RTR_FLAG = 0x40000000
CAN_ERR_FLAG = 0x20000000
CAN_SFF_MASK = 0x000007FF
CAN_EFF_MASK = 0x1FFFFFFF
ARX_JOY_CAN_ID = 0x721


class CanButtonReader:
    def __init__(self, interface='can6', can_id=ARX_JOY_CAN_ID, n_buttons=8):
        self.interface = interface
        self.can_id = can_id
        self.n_buttons = n_buttons
        self.last_buttons = [0] * n_buttons
        self.sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self.sock.bind((interface,))
        self.sock.setblocking(False)

    def close(self):
        self.sock.close()

    def _parse_can_id(self, raw_can_id):
        if raw_can_id & CAN_EFF_FLAG:
            return raw_can_id & CAN_EFF_MASK
        return raw_can_id & CAN_SFF_MASK

    def _buttons_from_data(self, data):
        padded = data.ljust(self.n_buttons, b"\x00")
        return [1 if value else 0 for value in padded[:self.n_buttons]]

    def poll_events(self):
        events = {}

        while True:
            try:
                frame = self.sock.recv(CAN_FRAME_SIZE)
            except BlockingIOError:
                break

            raw_can_id, dlc, data = struct.unpack(CAN_FRAME_FORMAT, frame)
            if raw_can_id & (CAN_RTR_FLAG | CAN_ERR_FLAG):
                continue
            if self.can_id is not None and self._parse_can_id(raw_can_id) != self.can_id:
                continue

            buttons = self._buttons_from_data(data[:dlc])
            for i, value in enumerate(buttons):
                if self.last_buttons[i] == 0 and value == 1:
                    events[i] = buttons.copy()
            self.last_buttons = buttons

        return events
