# VLA Fine-Tuning Notes

## Quick Commands

Collect HDF5 demos:

```bash
python scripts_rw/collect.py
```

Config: `scripts_rw/configs/collect.yaml`
Output: `<datasets>/episode_N.hdf5`, where `<datasets>` is the YAML `datasets:` field.
Current config value: `datasets/tube/hdf5`

Convert HDF5 to LeRobot:

```bash
python utils/convert_act_hdf5_to_lerobot.py
```

Config: `config/dataset/convert_act_hdf5_to_lerobot.yaml`
Input: `input_path`, currently `datasets/tube/hdf5`
Output: `output_path`, currently `datasets/tube/parquet`
Default behavior: non-destructive. If output already exists, conversion stops.
To intentionally regenerate output, run `python utils/convert_act_hdf5_to_lerobot.py overwrite=true`.

Compute norm stats:

```bash
python scripts/compute_norm_stats.py \
  --config-name pi0_arx_lora_chunk50_delta \
  --repo-id datasets/tube/parquet
```

Config: `src/openpi/training/config.py`, config name `pi0_arx_lora_chunk50_delta`
Input: `--repo-id`, here `datasets/tube/parquet`. If omitted, it falls back to `data.repo_id` in config.py.
Output: `config.assets_dirs / data_config.asset_id`. Exact save line: `scripts/compute_norm_stats.py:192`
Post-process: replaces joint dims `0:6,7:13` with official ARX stats; grippers and other dims stay computed from this dataset.

Train:

```bash
CUDA_VISIBLE_DEVICES=<GPU_IDS> XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py pi0_arx_lora_chunk50_delta \
  --data.repo-id=datasets/tube/parquet \
  --exp-name=tube_test \
  --overwrite \
  --keep-period=3000
```

Use `CUDA_VISIBLE_DEVICES=0` for GPU 0, or `CUDA_VISIBLE_DEVICES=0,1` for GPUs 0 and 1.
`--exp-name` names the run folder and wandb run. Checkpoints go to `checkpoints/pi0_arx_lora_chunk50_delta/tube_test/`.

Resume training:

```bash
CUDA_VISIBLE_DEVICES=<GPU_IDS> XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py pi0_arx_lora_chunk50_delta \
  --data.repo-id=datasets/tube/parquet \
  --exp-name=tube_test \
  --resume
```

Serve checkpoint:

```bash
python scripts/serve_policy.py --port=18000 policy:checkpoint \
  --policy.config=pi0_arx_lora_chunk50_delta \
  --policy.repo-id=datasets/tube/parquet \
  --policy.dir=checkpoints/pi0_arx_lora_chunk50_delta/tube_test/29999
```

This loads checkpoint `29999` and starts the websocket policy server.
Use port `18000` to match `scripts_rw/control_pc.py`.

Run robot inference client:

```bash
python scripts_rw/control_pc.py
```

Config: top constants in `scripts_rw/control_pc.py`
Checkpoint used: whatever checkpoint is being served by `scripts/serve_policy.py`.
`control_pc.py` does not load the checkpoint directly; it connects to the policy server.

## 1. Collect HDF5 Episodes

Collect robot demonstrations with:

```bash
python scripts_rw/collect.py
```

`collect.py` uses:

```text
scripts_rw/configs/collect.yaml
```

Collected episodes are saved as HDF5 files:

```text
<datasets>/episode_0.hdf5
<datasets>/episode_1.hdf5
...
```

`<datasets>` comes from the `datasets:` field in `scripts_rw/configs/collect.yaml`.
Current config value:

```yaml
datasets: datasets/tube/hdf5
```

If using the tube dataset folder layout, put or collect the raw HDF5 files under:

```text
datasets/tube/hdf5/
```

## 2. Convert HDF5 To LeRobot

Conversion config:

```text
config/dataset/convert_act_hdf5_to_lerobot.yaml
```

Important fields:

```yaml
input_path: datasets/tube/hdf5
repo_id: local/tube
output_path: datasets/tube/parquet
task: "Move a test tube from one tube rack to another tube rack."
```

`input_path` is the folder containing `episode_*.hdf5`.
`output_path` is the converted LeRobot dataset folder.
Relative paths resolve from the repo root.

Run conversion:

```bash
python utils/convert_act_hdf5_to_lerobot.py
```

Expected result:

```text
datasets/tube/parquet/
```

The converter is non-destructive by default:

```yaml
overwrite: False
```

If `datasets/tube/parquet` already exists, the command stops before writing. To intentionally regenerate it:

```bash
python utils/convert_act_hdf5_to_lerobot.py overwrite=true
```

Regeneration is guarded: the converter refuses to use the same folder for input and output, refuses nested
input/output paths, and writes to a temporary sibling folder before replacing the final output.

## 3. Update Training Config

Training configs are in:

```text
src/openpi/training/config.py
```

The current ARX LoRA config is:

```python
name="pi0_arx_lora_chunk50_delta"
```

The ARX config has defaults, but the dataset can be selected directly from the command line.
Use the converted LeRobot dataset path as `--data.repo-id`:

```bash
python scripts/train.py pi0_arx_lora_chunk50_delta \
  --data.repo-id=datasets/tube/parquet \
  --exp-name=tube_test \
  --overwrite
```

Current defaults in `config.py` are only fallbacks if no command override is passed:

```python
repo_id="datasets/apple/parquet"
base_config=DataConfig(prompt_from_task=True)
```

Do not pass `--data.default-prompt` for the tube dataset unless the converted dataset is missing task text.
The converter writes the instruction from `task:` in `config/dataset/convert_act_hdf5_to_lerobot.yaml`
into the LeRobot dataset, and `prompt_from_task=True` makes training use that task.

## 4. Compute Norm Stats

Run norm stats with the same repo id you will use for training:

```bash
python scripts/compute_norm_stats.py \
  --config-name pi0_arx_lora_chunk50_delta \
  --repo-id datasets/tube/parquet
```

This writes normalization assets used during training and inference.
The input dataset is not from `config/dataset/convert_act_hdf5_to_lerobot.yaml`.
It comes from `--repo-id`. If `--repo-id` is omitted, it falls back to the training config selected by
`--config-name`.

After computing stats from the dataset, the script replaces only ARX joint dimensions with official ARX stats:

```text
joint dims: 0,1,2,3,4,5,7,8,9,10,11,12
kept from dataset: left/right grippers at 6 and 13, plus any other dims
```

Default official source:

```text
assets/arx/
```

This file is tracked in the repo as `assets/arx/norm_stats.json`.

To disable this post-process and save fully dataset-computed stats:

```bash
python scripts/compute_norm_stats.py \
  --config-name pi0_arx_lora_chunk50_delta \
  --repo-id datasets/tube/parquet \
  --official-joint-stats-dir None
```

Exact code path:

```python
# scripts/compute_norm_stats.py
config = _config.with_repo_id(_config.get_config(config_name), repo_id)
data_config = config.data.create(config.assets_dirs, config.model)

# scripts/compute_norm_stats.py:183-188
norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
norm_stats = replace_joints_with_official_stats(norm_stats, official_joint_stats_dir, indices)

# src/openpi/training/config.py, current ARX fallback
repo_id="datasets/apple/parquet"
base_config=DataConfig(prompt_from_task=True)
```

For local paths, `data_config.repo_id` resolves to the absolute LeRobot dataset path used by LeRobot, while
`data_config.asset_id` remains a checkpoint-safe relative id such as `datasets/tube/parquet`.

The output path is also explicit in the script:

```python
# scripts/compute_norm_stats.py:190-192
output_path = config.assets_dirs / data_config.asset_id
print(f"Writing stats to: {output_path}")
normalize.save(output_path, norm_stats)
```

`config.assets_dirs` is:

```python
# src/openpi/training/config.py:537-540
return (pathlib.Path(self.assets_base_dir) / self.name).resolve()
```

So the output path is:

```text
assets/<config_name>/<data_config.asset_id>/
```

For example, if `data_config.asset_id` is `datasets/tube/parquet`, stats are written under:

```text
assets/pi0_arx_lora_chunk50_delta/datasets/tube/parquet/
```

It does not save into the dataset itself. For example, the apple LeRobot dataset is under `datasets/apple/parquet`,
but its OpenPI training norm stats are separate assets. In this repo, norm stats are treated as training/config assets,
then copied into checkpoints during checkpoint saving. That keeps the converted LeRobot dataset separate from
model-specific preprocessing stats.

## 5. Train

Basic JAX training command:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py pi0_arx_lora_chunk50_delta \
  --data.repo-id=datasets/tube/parquet \
  --exp-name=tube_test \
  --overwrite \
  --keep-period=3000
```

To select GPUs, prefix the command with `CUDA_VISIBLE_DEVICES=<GPU_IDS>`:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py pi0_arx_lora_chunk50_delta \
  --data.repo-id=datasets/tube/parquet \
  --exp-name=tube_test \
  --overwrite \
  --keep-period=3000
```

`--exp-name` names the training run. It affects:

```text
checkpoints/<config_name>/<exp_name>/
```

and the wandb run name/id. Use the same `--exp-name` with `--resume` to continue that run.

Use `--resume` instead of `--overwrite` to continue an existing run:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py pi0_arx_lora_chunk50_delta \
  --data.repo-id=datasets/tube/parquet \
  --exp-name=tube_test \
  --resume
```

Checkpoints are written to:

```text
checkpoints/pi0_arx_lora_chunk50_delta/tube_test/
```

For the current ARX config:

```python
save_interval=3000
num_train_steps=30000
```

So training attempts to save at:

```text
3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 29999
```

Orbax keeps only the latest checkpoint by default, plus checkpoints matching `keep_period`.
With default `keep_period=5000`, after a full run you may only see:

```text
15000
29999
```

Use `--keep-period=3000` if you want to preserve every scheduled checkpoint.

## 6. Serve For Robot Inference

After training, serve a checkpoint:

```bash
python scripts/serve_policy.py --port=18000 policy:checkpoint \
  --policy.config=pi0_arx_lora_chunk50_delta \
  --policy.repo-id=datasets/tube/parquet \
  --policy.dir=checkpoints/pi0_arx_lora_chunk50_delta/tube_test/29999
```

This loads the model from `--policy.dir` and runs a websocket policy server.
The robot client sends observations to this server and receives actions back.
Pass the same `--policy.repo-id` that was used for `--data.repo-id` during training so the policy config uses the
same dataset asset id when loading norm stats from the checkpoint.

Then run the robot control client:

```bash
python scripts_rw/control_pc.py
```

`control_pc.py` uses:

```python
HOST = "127.0.0.1"
PORT = 18000
```

It uses whichever checkpoint is currently loaded by the policy server. To switch checkpoints, restart `serve_policy.py` with a different `--policy.dir`.

Button controls in `control_pc.py`:

```text
button 1: home
button 2: start inference
button 3: stop inference
```
