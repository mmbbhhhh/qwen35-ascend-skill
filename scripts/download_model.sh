#!/bin/bash


set -e


mkdir -p ckpt/hf_path


echo "Download Qwen3.5-0.8B"


huggingface-cli download \
Qwen/Qwen3.5-0.8B \
--local-dir ckpt/hf_path/Qwen3.5-0.8B


echo "Model downloaded"
