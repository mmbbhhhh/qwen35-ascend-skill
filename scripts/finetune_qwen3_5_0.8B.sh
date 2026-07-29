#!/bin/bash

source /usr/local/Ascend/cann/set_env.sh

export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=2
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/workspace/MindSpeed-MM:$PYTHONPATH  #需要告知mindspeed在哪里
export HCCL_CONNECT_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=1800
export HCCL_ASYNC_ERROR_HANDLING=1
export HCCL_OP_EXPANSION_MODE=AIV       #关闭了hccl优化
cd /workspace/qwen35_ascend_skill


NPUS_PER_NODE=2
MASTER_ADDR=172.17.0.6     #需要修改的
MASTER_PORT=6000
NNODES=1
NODE_RANK=0


DISTRIBUTED_ARGS="
--nproc_per_node $NPUS_PER_NODE \
--nnodes $NNODES \
--node_rank $NODE_RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT
"


logfile=$(date +%Y%m%d)_$(date +%H%M%S)


config_path=/workspace/qwen35_ascend_skill/configs/qwen3_5_0.8B_config.yaml   #需要修改的告诉config在哪里


mkdir -p logs


torchrun $DISTRIBUTED_ARGS /workspace/MindSpeed-MM/mindspeed_mm/fsdp/train/trainer.py \
    ${config_path}     #需要修改的 用于告诉系统trainer.py在哪里