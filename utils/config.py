# -- coding: UTF-8
from types import SimpleNamespace


def config_to_namespace(cfg):
    """Convert Hydra/OmegaConf config objects into nested SimpleNamespace values."""
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        pass

    return _to_namespace(cfg)


def _to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_namespace(item) for item in value)
    return value
