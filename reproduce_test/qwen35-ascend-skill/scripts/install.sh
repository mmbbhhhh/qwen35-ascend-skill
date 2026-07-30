#!/bin/bash

set -e


echo "================================="
echo "Install Qwen3.5 Ascend Skill"
echo "================================="


SKILL_DIR=$(cd $(dirname $0)/.. && pwd)


echo "[1/6] Check python"

python3 --version


echo "[2/6] Prepare MindSpeed-MM"

cd /workspace

if [ ! -d MindSpeed-MM ]; then
    git clone -b 26.1.0 https://github.com/Ascend/MindSpeed-MM.git
fi


cd MindSpeed-MM

git checkout 26.1.0


echo "[3/6] Install MindSpeed-MM"

python3 -m pip install -e .


echo "[4/6] Install Qwen3.5 transformers"

python3 -m pip install transformers==5.2.0

echo "[5/6] Install dependencies"

python3 -m pip install accelerate

python3 -m pip install modelscope==1.38.1

echo "[6/6] Install Triton Ascend"

pip install triton-ascend==3.2.1 \
--extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple


echo "Environment ready"