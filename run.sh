#!/bin/bash

# 检查是否提供了所需的参数
if [ "$#" -lt 3 ]; then
    echo "用法: $0 <model_path> <tasks> <output_folder> [gpu_id]"
    echo "示例: $0 /data8/xjx/pretrained_models/e5-small-v2 \"longmemeval,locomo\" results_e5 0"
    exit 1
fi

# 解析参数
MODEL_PATH=$1
TASKS=$2
OUTPUT_FOLDER=$3
GPU_ID=${4:-0}  # 默认使用GPU 0，如果提供了第四个参数则使用它

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "正在运行评测："
echo "模型: $MODEL_PATH"
echo "任务: $TASKS"
echo "输出目录: $OUTPUT_FOLDER"
echo "使用GPU: $GPU_ID"

# 运行评测
python3 run.py \
    --model_path "$MODEL_PATH" \
    --tasks "$TASKS" \
    --output_folder "$OUTPUT_FOLDER"