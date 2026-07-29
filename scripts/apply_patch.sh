#!/bin/bash

set -e


echo "Apply MindSpeed-MM patch"


cd /workspace/MindSpeed-MM


patch -p1 < ../qwen35_ascend_skill/patches/train_engine.patch


echo "Patch applied successfully"
