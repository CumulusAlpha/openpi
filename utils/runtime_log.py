# -- coding: UTF-8
import logging
import sys
from pathlib import Path


def setup_runtime_log(root, filename):
    """Mirror stdout/stderr to a runtime log file under root."""
    log_path = Path(root).expanduser().resolve() / filename
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.__stdout__),
        ],
        force=True,
    )
    return log_path
