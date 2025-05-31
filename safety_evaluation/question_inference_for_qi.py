"""
Inference with customized input prompt
"""

import sys
sys.path.append('./')

import csv
import fire
import torch
import json
import os
from typing import List

from transformers import LlamaTokenizer, LlamaForCausalLM, MistralForCausalLM, AutoTokenizer

def question_read(text_file):
    dataset = []
    prefill_attacks = []
    with open(text_file, "r") as file:
        data = list(csv.reader(file, delimiter=","))
        for row in data:
            dataset.append(row[0])
            prefill_attacks.append(row[1])
    
    return dataset, prefill_attacks

def apply_prompt_template(
        prompt_template_style, 
        question_dataset,
        prefill_attacks,
        add_initial_response_token,
        initial_token_with_malicious_purpose,
        use_instruction=True):
    
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    prompt_template = (
        B_SYS + "Below is an instruction that describes a task. " +
        "Write a response that appropriately completes the request." + E_SYS +
        "### Instruction:\n{}\n\n### Response:\n"
    )

    if prompt_template_style == 'base':
        if use_instruction:
            # Apply the prompt format for each question in the dataset
            prompts = [
                B_INST + " " + prompt_template.format(question) + " " + E_INST
                for question in question_dataset
            ]

            if add_initial_response_token:
                if initial_token_with_malicious_purpose:
                    # Modify each prompt in the list with the malicious initial response token
                    for idx in range(len(prompts)):
                        prompts[idx] = prompts[idx] + f" {prefill_attacks[idx]}"
                else:
                    # Modify each prompt in the list with the non-malicious initial response token
                    prompts = [prompt + " Sorry, I can't fulfill" for prompt in prompts]

            return prompts
        else:
            return question_dataset
    else:
        # Define more templates if needed
        return question_dataset

def print_kwargs(**kwargs):
    print("#" * 30)
    print("#" * 30)
    for key, value in kwargs.items():
        print(f"{key} = {value}")
    print("#" * 30)
    print("#" * 30)

# Helper function to convert bash-style true/false strings to Python booleans
def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1"}:
        return True
    elif value.lower() in {"false", "0"}:
        return False
    else:
        raise ValueError(f"Invalid boolean value: {value}")


def main(
    model_name,
    model_family="llama",
    max_new_tokens=512,
    prompt_file=None,
    prompt_template_style='base',
    do_sample="false",
    sample_times=1,
    special_attention="false",
    add_initial_response_token="false",
    initial_token_with_malicious_purpose="false",
    no_strategic_attention="false",
    no_strategic_decoding="false",
    dynamic_reclassification_strategy="each", # each, first, periodic, annealing
    seed=42,
    top_p=0.9,
    temperature=0.7,
    top_k=100,
    output_file=None,
    use_cache=True,
    **kwargs
):
    # Convert string booleans to Python booleans
    do_sample = str_to_bool(do_sample)
    special_attention = str_to_bool(special_attention)
    add_initial_response_token = str_to_bool(add_initial_response_token)
    initial_token_with_malicious_purpose = str_to_bool(initial_token_with_malicious_purpose)
    no_strategic_attention=str_to_bool(no_strategic_attention)
    no_strategic_decoding=str_to_bool(no_strategic_decoding)

    print_kwargs(
        model_name=model_name,
        model_family=model_family,
        special_attention=special_attention,
        max_new_tokens=max_new_tokens,
        prompt_file=prompt_file,
        prompt_template_style=prompt_template_style,
        add_initial_response_token=add_initial_response_token,
        initial_token_with_malicious_purpose=initial_token_with_malicious_purpose,
        no_strategic_attention=no_strategic_attention,
        no_strategic_decoding=no_strategic_decoding,
        dynamic_reclassification_strategy=dynamic_reclassification_strategy,
        seed=seed,
        do_sample=do_sample,
        sample_times=sample_times,
        top_p=top_p,
        temperature=temperature,
        top_k=top_k,
        output_file=output_file,
        use_cache=use_cache,
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Load the model and tokenizer
    if model_family == "llama":
        print("Initialing LLama Models...")
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager")
    elif model_family == "mistral":
        print("Initialing Mistral Models...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = MistralForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager")

    model.model.mode = "generation"
    os.environ['special'] = "True" if special_attention else "False"
    os.environ['model-family'] = model_family
    if no_strategic_attention:
        os.environ['strategic_decoding_strategy'] = "False"
    if no_strategic_decoding:
        os.environ['strategic_attention_mechanism'] = "False"
    os.environ['dynamic_reclassification_strategy'] = dynamic_reclassification_strategy
    model = model.to(device)
    model.eval()

    # Read questions from file'
    question_dataset, prefill_attacks = question_read(prompt_file)

    # Apply the prompt template
    prompts = apply_prompt_template(
        prompt_template_style, 
        question_dataset,
        prefill_attacks, 
        add_initial_response_token, 
        initial_token_with_malicious_purpose, 
        use_instruction=True)

    # Initialize output
    out = []
    
    # [CLS] token setup (ensure tokenizer has a [CLS] token if needed)
    if tokenizer.cls_token is None:
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        model.resize_token_embeddings(len(tokenizer))
    cls_token_id = tokenizer.cls_token_id

    with torch.no_grad():
        for idx, prompt in enumerate(prompts):

            # Tokenize the prompt
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            
            # Calculate positions for the start and end of the input_text within the prompt
            input_text = question_dataset[idx]
            input_text_tokens = tokenizer.encode(input_text)
            start_idx_input = len(tokenizer.encode("[INST] " + prompts[0].split("### Instruction:")[0] + "### Instruction:\n")) + 1
            end_idx_input = start_idx_input + len(input_text_tokens) - 1
            positions = [start_idx_input, end_idx_input]

            # Add [CLS] token at the beginning of input_ids
            input_ids = torch.cat([torch.tensor([[cls_token_id]]), input_ids], dim=1)
            input_ids = input_ids.to(device)
            for _ in range(sample_times):
                # Generate output
                outputs = model.generate(
                    input_ids=input_ids,
                    positions=positions,
                    max_length=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    use_cache=use_cache,
                    num_return_sequences=1,
                )

                # Decode and collect the output text
                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Append result to output list
                out.append({'prompt': question_dataset[idx], 'answer': output_text})
                print('\n\n\n')
                print(f'>>> sample - {idx}')
                print('prompt = ', question_dataset[idx])
                print('answer = ', output_text)
                print('\n\n\n\n\n\n')
    
    # Save results to output file if specified
    if output_file is not None:
        with open(output_file, 'w') as f:
            for entry in out:
                f.write(json.dumps(entry))
                f.write("\n")

if __name__ == "__main__":
    fire.Fire(main)
