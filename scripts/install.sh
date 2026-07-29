#!/bin/bash

set -e

echo "================================="
echo "Install Qwen3.5 Ascend Skill"
echo "================================="


echo "[1/4] Check python"

python3 --version


echo "[2/4] Prepare MindSpeed-MM"

cd /workspace


if [ ! -d "MindSpeed-MM" ]; then

    git clone -b 26.0.0 https://github.com/Ascend/MindSpeed-MM.git

else

    echo "MindSpeed-MM exists"

fi


cd MindSpeed-MM


echo "[3/4] Install package"

python3 -m pip install -e .


echo "[4/4] Finished"

echo "Environment ready"