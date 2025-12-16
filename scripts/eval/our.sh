#!/bin/bash  bash ./scripts/eval/our.sh

# Devices
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

# Custom path
SPLIT="snei_eval"
SNEDIR="/home/ling/TinyLLaVA_Factory"
MODEL_PATH="/home/ling/TinyLLaVA_Factory/checkpoints/llava_factory/custom-finetune-TinyLLaVA-Phi-2-SigLIP-3.1B-lora"
MODEL_NAME="custom-finetune-TinyLLaVA-Phi-2-SigLIP-3.1B-lora"

# Inference loop
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m tinyllava.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --question-file $SNEDIR/${SPLIT}.jsonl \
        --image-folder $SNEDIR/snei_images \
        --answers-file $SNEDIR/answers/$SPLIT/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode phi &
done

wait

# Merge results
mkdir $SNEDIR/answers/$SPLIT/$MODEL_NAME
output_file=$SNEDIR/answers/$SPLIT/$MODEL_NAME/merge.jsonl
> "$output_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $SNEDIR/answers/$SPLIT/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done
