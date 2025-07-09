#!/bin/bash

# Make sure to give execute permission to this script
# chmod +x run_inference.sh

# Function to print memory status of all GPUs
print_gpu_memory() {
    echo "Current GPU memory usage:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
}

# Function to find an available GPU with at least 40 GB of free memory
find_available_gpu() {
    # Minimum required memory in MB (40 GB = 40960 MB)
    MIN_FREE_MEMORY=40960

    # Print all GPU memory status before selecting
    print_gpu_memory

    # Find a GPU with memory free >= 40GB
    AVAILABLE_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
                    awk -v min_mem=$MIN_FREE_MEMORY '$2 >= min_mem {print $1, $2}' | \
                    sort -n -k2 -r | \
                    head -n 1 | \
                    awk '{print $1}')
    
    if [ -z "$AVAILABLE_GPU" ]; then
        echo "No GPU with at least 40 GB of free memory found. Exiting."
        exit 1
    else
        echo "Using GPU $AVAILABLE_GPU with sufficient memory."
    fi
}

# Find an available GPU
find_available_gpu

# Set the GPU to use
export CUDA_VISIBLE_DEVICES=$AVAILABLE_GPU

# Parameters
MODEL_FAMILIY="llama"
MODEL_NAME="/home/jli265/workspace/Safety-Alignment-With-Explicit-Safety-Signal/model_checkpoints/fine-tuned-with-cls-meta-llama/Llama-2-7b-cls"
PROMPT_FILE="/home/jli265/workspace/llama-recipes/safety_evaluation/data/harmful-HEx-PHI/hex_prefill_qi_10.csv"
OUTPUT_FILE="safety_evaluation/question_outputs/hex_prefill_qi_10_output.json"
MAX_NEW_TOKENS=512
PROMPT_TEMPLATE_STYLE="base"
SEED=42
DO_SAMPLE=false
SPECIAL_ATTENTION=true
ADD_INITIAL_RESPONSE_TOKEN=true
INITIAL_TOKEN_WITH_MALICIOUS_PURPOSE=true
TOP_P=0.9
TEMPERATURE=0.7
TOP_K=100
REPETITION_PENALTY=1.0
LENGTH_PENALTY=1

# Run the Python script with the specified parameters
python safety_evaluation/question_inference_for_qi.py \
    --model_name "$MODEL_NAME" \
    --model_family "$MODEL_FAMILIY" \
    --special_attention $SPECIAL_ATTENTION \
    --prompt_file "$PROMPT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --max_new_tokens $MAX_NEW_TOKENS \
    --prompt_template_style "$PROMPT_TEMPLATE_STYLE" \
    --add_initial_response_token $ADD_INITIAL_RESPONSE_TOKEN \
    --initial_token_with_malicious_purpose $INITIAL_TOKEN_WITH_MALICIOUS_PURPOSE \
    --seed $SEED \
    --do_sample $DO_SAMPLE \
    --top_p $TOP_P \
    --temperature $TEMPERATURE \
    --top_k $TOP_K \
    --repetition_penalty $REPETITION_PENALTY \
    --length_penalty $LENGTH_PENALTY

echo "Inference completed. Results saved to $OUTPUT_FILE."
