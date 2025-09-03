#!/bin/bash

# 设置任务列表 - 所有模型使用相同的任务列表
TASKS="longmemeval,browsecomp-plus,toolret"

# 模型配置：每行包含 "模型路径 GPU_ID"
# 可以根据需要添加更多模型
declare -A MODEL_CONFIG=(
    ["/data8/xjx/pretrained_models/e5-small-v2"]="0"
    ["/data8/xjx/pretrained_models/e5-base-v2"]="1"
    # ["/data8/xjx/pretrained_models/model-3"]="2"
    # 添加更多模型...
)

# 存储所有后台进程的PID
pids=()

# 并行评测每个模型
for MODEL_PATH in "${!MODEL_CONFIG[@]}"; do
    GPU_ID="${MODEL_CONFIG[$MODEL_PATH]}"
    
    MODEL_NAME=$(basename "$MODEL_PATH")
    OUTPUT_DIR="results"
    
    echo "======================================="
    echo "启动模型评测: $MODEL_NAME (GPU: $GPU_ID)"
    echo "输出目录: $OUTPUT_DIR"
    echo "======================================="
    
    # 在后台运行评测脚本并记录PID
    (
        echo "$(date): 开始评测模型 $MODEL_NAME"
        bash run.sh "$MODEL_PATH" "$TASKS" "$OUTPUT_DIR" "$GPU_ID"
        echo "$(date): 模型 $MODEL_NAME 评测完成"
    ) > "logs/${MODEL_NAME}_eval.log" 2>&1 &
    
    # 保存进程ID
    pids+=($!)
    
    echo "模型 $MODEL_NAME 评测已在后台启动，日志文件: ${MODEL_NAME}_eval.log"
    echo ""
    
    # 可选：在启动下一个评测前短暂停顿，避免资源竞争
    sleep 1
done

echo "所有模型评测已启动，等待完成..."

# 等待所有后台进程完成
for pid in "${pids[@]}"; do
    wait $pid
    echo "进程 $pid 已完成"
done

echo "所有模型评测完成！"
echo "各模型的评测日志保存在各自的 *_eval.log 文件中"