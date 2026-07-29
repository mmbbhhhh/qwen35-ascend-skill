#!/bin/bash

set -e


echo "================================="
echo "Install Qwen3.5 Ascend Skill"
echo "================================="


echo "[1/4] Check python"

python3 --version


echo "[2/4] Install MindSpeed-MM"


cd /workspace


if [ ! -d "MindSpeed-MM" ]; then

    git clone https://github.com/Ascend/MindSpeed-MM.git

fi


cd MindSpeed-MM


git checkout 26.0.0


echo "[3/4] Install package"

pip install -e .


echo "[4/4] Finished"

echo "Environment ready"
