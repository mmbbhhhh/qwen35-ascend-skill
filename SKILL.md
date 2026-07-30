# Qwen3.5-0.8B Ascend NPU FSDP2 Fine-tuning Skill


## 1. 简介

本 Skill 用于在华为昇腾（Ascend）NPU 平台上，基于 MindSpeed-MM 框架完成 Qwen3.5-0.8B 多模态模型全参数微调流程。

支持：

- Ascend NPU 环境配置；
- Qwen3.5-0.8B 模型权重加载；
- COCO2017 + LLaVA-Instruct-150K 数据准备；
- FSDP2 多卡全参数训练；
- checkpoint 保存；
- 实验流程复现。


本 Skill 主要用于验证 Qwen3.5-0.8B 在昇腾 NPU 上的完整训练链路：

```
环境部署
    ↓
模型加载
    ↓
数据准备
    ↓
FSDP2 分布式训练
    ↓
checkpoint 保存
    ↓
实验结果验证
```


---

# 2. 环境信息


## 硬件环境

| 项目 | 配置 |
|-|-|
| 加速设备 | Ascend NPU |
| NPU数量 | 2 |
| 并行方式 | FSDP2 |
| 启动方式 | torchrun |


## 软件环境

| 软件 | 版本 |
|-|-|
| Python | 3.11 |
| PyTorch | 2.7.1 |
| TorchNPU | 2.7.1 |
| CANN | 9.1.0-beta.3 |
| MindSpeed-MM | 26.1.0 |


---

# 3. 项目结构


```
qwen35-ascend-skill

├── README.md
│
├── configs
│   └── qwen3_5_0.8B_config.yaml
│
├── scripts
│   ├── install.sh
│   ├── download_model.sh
│   ├── prepare_data.sh
│   └── finetune_qwen3_5_0.8B.sh
│
├── docs
│   └── experiment_result.md
│
├── logs
│
└── checkpoints
```


---

# 4. 快速开始


## Step 1. 加载 CANN 环境


```bash
source /usr/local/Ascend/cann/set_env.sh
```


---

## Step 2. 安装 MindSpeed-MM 环境


执行：

```bash
bash scripts/install.sh
```


完成：

- MindSpeed-MM 安装；
- 训练依赖安装；
- Ascend NPU 运行环境配置。


---

## Step 3. 下载模型权重


执行：

```bash
bash scripts/download_model.sh
```


下载模型：

```
Qwen3.5-0.8B
```


默认路径：

```
ckpt/hf_path/Qwen3.5-0.8B
```


---

## Step 4. 准备训练数据


执行：

```bash
bash scripts/prepare_data.sh
```


数据来源：

```
COCO2017
+
LLaVA-Instruct-150K
```


数据转换后生成：

```
data/output_llava_coco_data.json
```


---

## Step 5. 启动训练


执行：

```bash
bash scripts/finetune_qwen3_5_0.8B.sh
```


训练方式：

```
FSDP2 Full Parameter Fine-tuning
```


启动方式：

```
torchrun
```


支持：

- 2 NPU 分布式训练；
- 参数分片；
- 梯度同步；
- checkpoint 保存。


---

# 5. 训练结果验证


训练完成后可以观察：


## Loss

训练过程中：

- loss 正常输出；
- learning rate 按策略变化；
- gradient norm 保持非零。


说明：

模型前向传播、反向传播以及参数更新流程正常。


---

## Checkpoint


训练过程中自动保存：

```
checkpoints/qwen3.5-0.8B-finetune/
```


示例：

```
iter_0000500
```


验证：

- FSDP2 训练状态保存成功；
- 多卡状态同步正常；
- checkpoint 可以生成。


---

# 6. 实验结果


本实验已完成：

✅ Qwen3.5-0.8B 模型加载

✅ Ascend NPU 运行验证

✅ COCO/LLaVA 多模态数据接入

✅ FSDP2 两卡训练

✅ 全参数微调

✅ Loss 与 Gradient 变化验证

✅ checkpoint 保存验证


详细实验过程及结果：

```
docs/experiment_result.md
```


---

# 7. 常见问题


## 7.1 NPU数量不匹配


问题：

```
Invalid device ID
```


原因：

启动进程数超过实际 NPU 数量。


解决：

修改：

```bash
NPUS_PER_NODE=2
```


---

## 7.2 Triton 算子依赖问题


问题：

```
ModuleNotFoundError: No module named triton
```


原因：

部分融合优化算子依赖 Triton。


解决：

关闭非必要优化算子：

```yaml
gdn_implementation: eager

causal_conv1d_implementation: eager
```


---

# 8. 复现流程总结


完整执行：


```bash
source /usr/local/Ascend/cann/set_env.sh

bash scripts/install.sh

bash scripts/download_model.sh

bash scripts/prepare_data.sh

bash scripts/finetune_qwen3_5_0.8B.sh
```


完成以上步骤即可复现：

```
Qwen3.5-0.8B
+
Ascend NPU
+
MindSpeed-MM
+
FSDP2 Full Parameter Fine-tuning
```


---

# 9. 输出产物


| 文件 | 说明 |
|-|-|
| configs/qwen3_5_0.8B_config.yaml | 训练配置 |
| scripts/*.sh | 自动化执行脚本 |
| logs/ | 训练日志 |
| checkpoints/ | FSDP2训练checkpoint |
| docs/experiment_result.md | 实验报告 |