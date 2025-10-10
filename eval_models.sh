#!/bin/bash
# 设置LOCAL_DATA_PREFIX
export LOCAL_DATA_PREFIX="/data8/xjx/local-data"

# 设置任务列表 - 所有模型使用相同的任务列表
# 需要传入任务类的元数据中的任务名才能被解析
TASKS="BrowseCompPlus,ToolRet,LoCoMo,LongMemEval,AITQA,CosQAPlus,FRAMES,FreshStack,LeetCode,MTRAG,MuSiQue"

# 模型配置：每行包含 "模型路径 GPU_ID BATCH_SIZE"
# 可以根据需要添加更多模型
declare -A MODEL_CONFIG=(
    ["/data8/xjx/pretrained_models/e5-small-v2"]="0:512"
    ["/data8/xjx/pretrained_models/e5-base-v2"]="1:256"
    ["/data8/xjx/pretrained_models/Qwen3-Embedding-0.6B"]="2:32"
    ["/data8/xjx/pretrained_models/Qwen3-Embedding-4B"]="3:16"
    ["/data8/xjx/pretrained_models/Qwen3-Embedding-8B"]="4:8"
    # ["/data8/xjx/pretrained_models/model-3"]="2"
    # 添加更多模型...
)

# 存储所有后台进程的PID
pids=()

# 并行评测每个模型
for MODEL_PATH in "${!MODEL_CONFIG[@]}"; do
    # 解析配置 "GPU_ID:BATCH_SIZE"
    IFS=':' read -r GPU_ID BATCH_SIZE <<< "${MODEL_CONFIG[$MODEL_PATH]}"
    
    MODEL_NAME=$(basename "$MODEL_PATH")
    OUTPUT_DIR="results"
    
    echo "======================================="
    echo "启动模型评测: $MODEL_NAME (GPU: $GPU_ID)"
    echo "输出目录: $OUTPUT_DIR"
    echo "======================================="
    
    # 在后台运行评测脚本并记录PID
    (
        echo "$(date): 开始评测模型 $MODEL_NAME"
        bash run.sh "$MODEL_PATH" "$TASKS" "$OUTPUT_DIR" "$BATCH_SIZE" "$GPU_ID"
        echo "$(date): 模型 $MODEL_NAME 评测完成"
    ) > "logs/${MODEL_NAME}_eval.log" 2>&1 &
    
    # 保存进程ID
    pids+=($!)
    
    echo "模型 $MODEL_NAME 评测已在后台启动，日志文件: ${MODEL_NAME}_eval.log"
    echo "配置: GPU=$GPU_ID, Batch Size=$BATCH_SIZE"
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