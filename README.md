# Qwen3.5-0.8B Ascend NPU Fine-tuning Skill


## Overview

This skill provides an automated workflow for migrating and fine-tuning Qwen3.5-0.8B model on Ascend NPU.


## Features

- Environment check
- Model preparation
- FSDP2 full parameter fine-tuning
- Performance benchmark
- Training report generation


## Workflow


1. Check environment


bash scripts/check_env.sh



2. Prepare model

Place Qwen3.5-0.8B HF model under:


ckpt/hf_path/



3. Start training


bash scripts/train_qwen35_0.8B.sh



## Experiment Result

Hardware:
Ascend NPU 2 chip

Framework:
MindSpeed-MM FSDP2


Training completed:

20/20 iterations


Throughput:

0.175 samples/s

