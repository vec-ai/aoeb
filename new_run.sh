export HF_HOME=/data8/xjx/.cache/huggingface
export CUDA_VISIBLE_DEVICES=3
export LOCAL_DATA_PREFIX="/data8/xjx/local-data"
python run.py \
    --model_path "/data8/xjx/pretrained_models/bge-m3" \
    --tasks "MVRBComposedScreenshotRetrievalKnowledgeRelation,MVRBComposedScreenshotRetrievalNewsToWiki,MVRBComposedScreenshotRetrievalProductDiscovery,MVRBComposedScreenshotRetrievalWikiToProduct" \
    --batch_size 32 \
    --output_dir "results/" \
    --precision "fp16" \
    --model_kwargs '{"trust_remote_code": true}' \
    --encode_kwargs '{"normalize_embeddings": true, "show_progress_bar": false}' \
    --is_multimodal True \
    > logs/bge-m3-visual.log 2>&1 &
    # --run_kwargs '{"verbosity": 2, "overwrite_results": true}'
    # --model_kwargs '{"trust_remote_code": true}' \
    # for jina-v4
    # --model_kwargs '{"trust_remote_code": true, "default_task": "retrieval"}' \
    # for bge-visual
    # --model_kwargs '{"model_name_bge": "/data8/xjx/pretrained_models/bge-m3", "model_weight": "/data8/xjx/pretrained_models/bge-visualized/Visualized_m3.pth"}' \
    # --precision "bf16" \
    # --tasks "MuSiQue,BrowseCompPlus,FreshStack,ToolRet,LoCoMo,LongMemEval,BRIGHT,AppsRetrieval,CodeFeedbackMT,CodeFeedbackST,CodeTransOceanContest,CodeTransOceanDL,CosQA,StackOverflowQA,SyntheticText2SQL,TempReasonL1,TempReasonL2Context,TempReasonL2Fact,TempReasonL2Pure,TempReasonL3Context,TempReasonL3Fact,TempReasonL3Pure" \
#     --tasks "AppsRetrieval,CodeFeedbackMT,CodeFeedbackST,CodeTransOceanContest,CodeTransOceanDL,\
# CosQA,StackOverflowQA,SyntheticText2SQL,TempReasonL1,TempReasonL2Context,TempReasonL2Fact,TempReasonL2Pure,\
# TempReasonL3Context,TempReasonL3Fact,TempReasonL3Pure" \
# --tasks "MuSiQue,ToolRet,LoCoMo,LongMemEval,BRIGHT,\
# AppsRetrieval,CodeFeedbackMT,CodeFeedbackST,CodeTransOceanContest,CodeTransOceanDL,\
# CosQA,StackOverflowQA,SyntheticText2SQL,TempReasonL1,TempReasonL2Context,TempReasonL2Fact,TempReasonL2Pure,\
# TempReasonL3Context,TempReasonL3Fact,TempReasonL3Pure,FreshStack" \