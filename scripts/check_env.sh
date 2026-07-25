#!/bin/bash

echo "===== Ascend Environment ====="

python3 - <<EOF
import torch
print("Torch:", torch.__version__)

try:
    import torch_npu
    print("torch_npu: OK")
except:
    print("torch_npu: FAILED")

print("NPU available:",
      torch.npu.is_available())
EOF


echo ""

echo "===== CANN ====="

echo $ASCEND_HOME_PATH
