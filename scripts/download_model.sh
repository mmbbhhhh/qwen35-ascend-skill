#!/bin/bash

set -e


echo "Download Qwen3.5-0.8B"


mkdir -p /workspace/MindSpeed-MM/ckpt/hf_path


modelscope download \
--model Qwen/Qwen3.5-0.8B \
--local_dir /workspace/MindSpeed-MM/ckpt/hf_path/Qwen3.5-0.8B


echo "Model downloaded"