# VLA 微调笔记

## 数据采集

采集前先激活 CAN：

```bash
bash utils/setup_can.sh
```

这个脚本会通过 `slcand` 创建并拉起机械臂和按钮使用的 CAN 接口：

- 左臂 - `/dev/arxcan1` -> `can0` 
- 右臂 - `/dev/arxcan3` -> `can1` 
- 按钮 - `/dev/arxcan6` -> `can6` 

运行结束后，脚本会打印当前链路状态。也可以手动检查：

```bash
ip -br link
```

采集 HDF5 演示数据：

```bash
python scripts_rw/collect.py
```

配置文件：`scripts_rw/configs/collect.yaml`
输出：`<datasets>/episode_N.hdf5`，其中 `<datasets>` 是 YAML 里的 `datasets:` 字段。
当前配置值：`datasets/tube/hdf5`

将 HDF5 转换为 LeRobot 格式：

```bash
python utils/convert_act_hdf5_to_lerobot.py
```

配置文件：`config/dataset/convert_act_hdf5_to_lerobot.yaml`
输入：`input_path`，当前为 `datasets/tube/hdf5`
输出：`output_path`，当前为 `datasets/tube/parquet`
默认行为：非破坏式。如果输出目录已经存在，转换会停止。
如果需要主动重新生成输出，运行 `python utils/convert_act_hdf5_to_lerobot.py overwrite=true`。

## 训练

计算归一化统计量：

```bash
python scripts/compute_norm_stats.py \
  --config-name pi0_arx_lora_chunk50_delta \
  --repo-id datasets/tube/parquet
```

配置：`src/openpi/training/config.py`，配置名为 `pi0_arx_lora_chunk50_delta`
输入：`--repo-id`，这里是 `datasets/tube/parquet`。如果省略，会回退到 `config.py` 里的 `data.repo_id`。
输出：`config.assets_dirs / data_config.asset_id`。具体保存位置对应 `scripts/compute_norm_stats.py:192`。
后处理：会用官方 ARX 统计量替换关节维度 `0:6,7:13`；夹爪和其他维度继续使用当前数据集计算得到的统计量。

开始训练：

```bash
CUDA_VISIBLE_DEVICES=<GPU_IDS> XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py pi0_arx_lora_chunk50_delta \
  --data.repo-id=datasets/tube/parquet \
  --exp-name=tube_test \
  --overwrite \
  --keep-period=3000
```

使用 `CUDA_VISIBLE_DEVICES=0` 表示使用 0 号 GPU，使用 `CUDA_VISIBLE_DEVICES=0,1` 表示使用 0 号和 1 号 GPU。
`--exp-name` 会命名运行目录和 wandb 运行名。Checkpoint 会保存到 `checkpoints/pi0_arx_lora_chunk50_delta/tube_test/`。

继续训练：

```bash
CUDA_VISIBLE_DEVICES=<GPU_IDS> XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py pi0_arx_lora_chunk50_delta \
  --data.repo-id=datasets/tube/parquet \
  --exp-name=tube_test \
  --resume
```

## 部署

先用网线连接笔记本电脑和台式机：

1. 用网线把笔记本电脑和台式机直接连接起来。
2. 在台式机上查看网卡和 IP：

```bash
ip addr show
```

3. 找到本地有线网卡，例如 `enp6s0`，查看它下面的 `inet` 地址，例如 `192.168.1.23/24`。
4. 在笔记本电脑上用 SSH 连接台式机：

```bash
ssh <台式机用户名>@<台式机IP> # 例如ssh ubuntu@192.168.1.23
```

如果有线网卡下面没有 `inet` 地址，说明直连网络可能还没有分配 IP，需要先给两台机器配置同一网段的静态 IP。

部署时也需要先激活 CAN：

```bash
bash utils/setup_can.sh
```

确认 CAN 已经拉起：

```bash
ip -br link
```
---
在台式机上修改serve_policy.py里面的ip地址为刚刚拿到的台式机ip，启动 checkpoint 服务：

```bash
python scripts/serve_policy.py --port=8080 policy:checkpoint \
  --policy.config=pi0_arx_lora_chunk50_delta \
  --policy.repo-id=datasets/tube/parquet \
  --policy.dir=checkpoints/pi0_arx_lora_chunk50_delta/tube_test/29999
```

这会加载 checkpoint `29999`，并启动 websocket policy server。
端口使用 `8080`，以匹配 `scripts_rw/control_pc.py`，这个地方的端口可以自己修改防止冲突。

---
在笔记本运行机器人控制客户端：

```bash
python scripts_rw/control_pc.py
```

使用的 checkpoint：由 `scripts/serve_policy.py` 当前提供服务的 checkpoint 决定。
`control_pc.py` 不会直接加载 checkpoint；它会连接到 policy server。
