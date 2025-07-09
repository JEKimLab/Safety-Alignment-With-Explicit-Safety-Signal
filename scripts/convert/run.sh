python src/llama_recipes/inference/checkpoint_converter_fsdp_hf.py \
    -fsdp_checkpoint_path "model_checkpoints/fine-tuned-with-cls-meta-llama/Llama-2-7b-hf" \
    -consolidated_model_path "model_checkpoints/fine-tuned-with-cls-meta-llama/Llama-2-7b-cls" \
    -HF_model_path_or_name "/home/jli265/workspace/llama-recipes/model_checkpoints/llama2-7b-with-cls-standard"