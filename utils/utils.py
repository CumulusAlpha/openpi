# -- coding: UTF-8
import time


def precise_wait(target_time):
    """Sleep until a monotonic timestamp while keeping the final wait tight."""
    while True:
        remaining = target_time - time.monotonic()
        if remaining <= 0:
            return
        if remaining > 0.002:
            time.sleep(remaining - 0.001)
