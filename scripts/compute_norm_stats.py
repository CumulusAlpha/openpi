"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

from pathlib import Path

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


DEFAULT_OFFICIAL_NORM_STATS_DIR = Path(__file__).resolve().parents[1] / "assets/arx"
ARX_JOINT_INDICES = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def parse_indices(value: str | None) -> tuple[int, ...]:
    if value is None or not value.strip():
        return ARX_JOINT_INDICES

    indices = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start, end = part.split(":", 1)
            indices.extend(range(int(start), int(end)))
        else:
            indices.append(int(part))
    return tuple(dict.fromkeys(indices))


def replace_stats_indices(
    target: normalize.NormStats,
    source: normalize.NormStats,
    indices: tuple[int, ...],
    *,
    key: str,
) -> normalize.NormStats:
    arrays = {}
    for field in ("mean", "std", "q01", "q99"):
        target_arr = getattr(target, field)
        source_arr = getattr(source, field)
        if target_arr is None:
            arrays[field] = None
            continue
        if source_arr is None:
            raise ValueError(f"Official norm stats for {key!r} are missing {field!r}")

        target_arr = np.asarray(target_arr).copy()
        source_arr = np.asarray(source_arr)
        if max(indices, default=-1) >= target_arr.shape[-1]:
            raise ValueError(f"Joint index out of range for computed {key}.{field} shape {target_arr.shape}")
        if max(indices, default=-1) >= source_arr.shape[-1]:
            raise ValueError(f"Joint index out of range for official {key}.{field} shape {source_arr.shape}")
        target_arr[list(indices)] = source_arr[list(indices)]
        arrays[field] = target_arr

    return normalize.NormStats(**arrays)


def replace_joints_with_official_stats(
    norm_stats: dict[str, normalize.NormStats],
    official_norm_stats_dir: str | Path,
    indices: tuple[int, ...],
) -> dict[str, normalize.NormStats]:
    official_stats = normalize.load(Path(official_norm_stats_dir).expanduser())
    result = dict(norm_stats)
    for key in ("state", "actions"):
        if key not in result:
            continue
        if key not in official_stats:
            raise KeyError(f"Official norm stats do not contain {key!r}")
        result[key] = replace_stats_indices(result[key], official_stats[key], indices, key=key)
    return result


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str,
    repo_id: str | None = None,
    max_frames: int | None = None,
    official_joint_stats_dir: str | None = str(DEFAULT_OFFICIAL_NORM_STATS_DIR),
    official_joint_indices: str | None = "0:6,7:13",
):
    config = _config.with_repo_id(_config.get_config(config_name), repo_id)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
    if official_joint_stats_dir is not None:
        indices = parse_indices(official_joint_indices)
        print(f"Replacing joint norm stats with official stats from: {official_joint_stats_dir}")
        print(f"Official joint indices: {indices}")
        norm_stats = replace_joints_with_official_stats(norm_stats, official_joint_stats_dir, indices)

    if data_config.asset_id is None:
        raise ValueError("Data config must have an asset_id")
    output_path = config.assets_dirs / data_config.asset_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
