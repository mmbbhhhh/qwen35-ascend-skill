#!/bin/bash

set -e


echo "================================="
echo "Prepare Qwen3.5 dataset"
echo "================================="


SKILL_DIR=$(cd $(dirname $0)/.. && pwd)

cd ${SKILL_DIR}


echo "Dataset path:"
pwd



echo "[1/4] Install ModelScope"


pip install modelscope==1.38.1



echo "[2/4] Download COCO2017"


mkdir -p data/coco


if [ ! -f data/coco/train2017.zip ] && [ ! -d data/coco/train2017 ]; then

modelscope download \
--dataset PAI/COCO2017 \
train2017.zip \
--local_dir ./data/coco

else

echo "COCO exists"

fi



cd data/coco


if [ ! -d train2017 ]; then

python3 -m zipfile -e train2017.zip .

else

echo "COCO extracted"

fi


cd ${SKILL_DIR}



echo "[3/4] Download LLaVA"


mkdir -p data/llava


if [ ! -f data/llava/llava_instruct_150k.json ]; then


modelscope download \
--dataset AI-ModelScope/LLaVA-Instruct-150K \
llava_instruct_150k.json \
--local_dir ./data/llava


else

echo "LLaVA exists"

fi



echo "[4/4] Convert dataset"



if [ ! -f ${SKILL_DIR}/data/output_llava_coco_data.json ]; then

python3 \
/workspace/MindSpeed-MM/mindspeed_mm/fsdp/tools/data_tool/llava_instruct_2_mllm_demo_format.py \
\
--llava_json_path \
${SKILL_DIR}/data/llava/llava_instruct_150k.json \
\
--coco_path \
${SKILL_DIR}/data/coco \
\
--output_json_path \
${SKILL_DIR}/data/output_llava_coco_data.json


python3 - <<EOF
import json

path="${SKILL_DIR}/data/output_llava_coco_data.json"

with open(path,"r") as f:
    data=json.load(f)

for item in data:
    if "images" in item:
        item["images"]=[
            "data/coco/"+x.replace("./","")
            for x in item["images"]
        ]

with open(path,"w") as f:
    json.dump(data,f,ensure_ascii=False)

print("Fix image paths done")
EOF

else

echo "Converted dataset exists"

fi


echo "================================="
echo "Dataset preparation finished"
echo "================================="