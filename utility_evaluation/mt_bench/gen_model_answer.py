"""
Generate answers for the mt-bench 80 questions.
"""

import sys
sys.path.append('./')
import os
import fire
import torch
from typing import Optional
import json
import time
import shortuuid
import copy
from transformers import LlamaTokenizer, LlamaForCausalLM, MistralForCausalLM, AutoTokenizer

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

prompt_template = (
    B_SYS + "Below is an instruction that describes a task. " +
    "Write a response that appropriately completes the request." + E_SYS +
    "### Instruction:\n{}\n\n### Response:\n"
)

def load_questions(question_file: str, begin: Optional[int] = None, end: Optional[int] = None):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    return questions

def apply_prompt_template(
        prompt_template_style, 
        question_dataset, 
        use_instruction=True):
    
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

            return prompts
        else:
            prompts = [prompt_template.format(question) for question in question_dataset]
            return question_dataset
    elif prompt_template_style == 'none':
        if use_instruction:
            # Apply the prompt format for each question in the dataset
            prompts = [
                B_INST + " " + question + " " + E_INST
                for question in question_dataset
            ]
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

# Sampling temperature configs for
temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
}

def main(
    model_name,
    model_family="llama",
    model_id: str=None,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens = 1024, #The maximum numbers of tokens to generate
    prompt_file: str='utility_evaluation/mt_bench/data/question.jsonl',
    prompt_template_style: str='base',
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=0.9, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_salesforce_content_safety: bool=True, # Enable safety check with Salesforce safety flan t5
    max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    output_file: str = None,
    **kwargs
):
    print_kwargs(
        model_name=model_name,
        model_family=model_family,
        model_id=model_id,
        peft_model=peft_model,
        quantization=quantization,
        max_new_tokens=max_new_tokens,
        prompt_file=prompt_file,
        prompt_template_style=prompt_template_style,
        seed=seed,
        do_sample=do_sample,
        min_length=min_length,
        use_cache=use_cache,
        top_p=top_p,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        enable_azure_content_safety=enable_azure_content_safety,
        enable_sensitive_topics=enable_sensitive_topics,
        enable_salesforce_content_safety=enable_salesforce_content_safety,
        max_padding_length=max_padding_length,
        use_fast_kernels=use_fast_kernels,
        output_file=output_file,
        **kwargs
    )
        
    if model_id is None:
        model_id = model_name.split("/")[-1]
    if output_file is None:
        output_file = f"utility_evaluation/mt_bench/data/model_answer/{model_id}.jsonl"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
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
    os.environ['special'] = "True"
    os.environ['model-family'] = model_family
    os.environ['strategic_decoding_strategy'] = "True"
    os.environ['strategic_attention_mechanism'] = "True"
    os.environ['dynamic_reclassification_strategy'] = "each"
    os.environ['consucultive_unsafe_tao'] = "3"
    os.environ['cls_logit_boundary'] = "0.5"
    os.environ['r1'] = "5" 
    os.environ['r2'] = "15"
    os.environ['r3'] = "10"
    model = model.to(device)

    model.eval()
    
    # Load questions
    question_file = load_questions(prompt_file)
    first_turn_question_dataset = [q['turns'][0] for q in question_file]
    second_turn_question_dataset = [q['turns'][1] for q in question_file]
    
    # Apply prompt template
    prompts = apply_prompt_template(
        prompt_template_style, 
        first_turn_question_dataset, 
        use_instruction=True)
    
    out = []

     # [CLS] token setup (ensure tokenizer has a [CLS] token if needed)
    if tokenizer.cls_token is None:
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        model.resize_token_embeddings(len(tokenizer))
    cls_token_id = tokenizer.cls_token_id

    with torch.no_grad():
        
        for idx, question in enumerate(question_file):
            # Sampling configuration (following mt-bench official code)
            if question["category"] in temperature_config:
                temperature = temperature_config[question["category"]]
            else:
                temperature = 0.7
            
            if temperature < 1e-4:
                do_sample = False
            else:
                do_sample = True

            # First turn
            input_ids = tokenizer(prompts[idx], return_tensors="pt").input_ids
            
            input_text = first_turn_question_dataset[idx]
            input_text_tokens = tokenizer.encode(input_text)
            if prompt_template_style == "base":
                start_idx_input = len(tokenizer.encode("[INST] " + prompts[0].split("### Instruction:")[0] + "### Instruction:\n")) + 1
            elif prompt_template_style == "none":
                start_idx_input = 4
            else:
                print(f"Not suppported {prompt_template_style}")
                exit()
            end_idx_input = start_idx_input + len(input_text_tokens) - 1
            positions = [start_idx_input, end_idx_input]
            input_token_length = input_ids.shape[1]
            
            # Add [CLS] token at the beginning of input_ids
            input_ids = torch.cat([torch.tensor([[cls_token_id]]), input_ids], dim=1)
            input_ids = input_ids.to(device)

            outputs = model.generate(
                input_ids = input_ids,
                positions=positions,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                # top_p=top_p,
                # top_k=top_k,
                use_cache=use_cache,
                num_return_sequences=1,
            )
            
            output_text_1 = tokenizer.decode(outputs[0][input_token_length:], skip_special_tokens=True)            
            print('\n\n\n')
            print('>>> sample - %d' % idx)
            print('prompt-1 = ', first_turn_question_dataset[idx])
            print('answer-1 = ', output_text_1)
            
            # Update dialog and tokens with assistant's response
            prompt_old = f"{first_turn_question_dataset[idx]} {output_text_1} "
            # Update dialog and tokens with the second prompt
            prompt2 = f"{B_INST} {prompt_template.format(prompt_old + second_turn_question_dataset[idx].strip())} {E_INST}"
            input_ids2 = tokenizer.encode(prompt2)
            start_idx_input2 =  len(tokenizer.encode("[INST] " + prompt_template.split("### Instruction:")[0] + "### Instruction:\n")) + 1
            end_idx_input2 = start_idx_input2 + len(tokenizer.encode(prompt_old + second_turn_question_dataset[idx].strip())) - 1
            positions2 = [start_idx_input2, end_idx_input2]
            print(positions2)

            # Second turn
            input_ids2= torch.tensor(input_ids2).long()
            input_ids2= input_ids2.unsqueeze(0)
            input_ids2 = torch.cat([torch.tensor([[cls_token_id]]), input_ids2], dim=1)
            input_ids2 = input_ids2.to(device)
            input_token_length = input_ids2.shape[1]
            
            outputs = model.generate(
                input_ids = input_ids2,
                positions=positions2,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                # top_p=top_p,
                # top_k=top_k,
                use_cache=use_cache,
                num_return_sequences=1,
            )
            
            output_text_2 = tokenizer.decode(outputs[0][input_token_length:], skip_special_tokens=True)
            print('\n-------------------------------\n')
            print('prompt-2 = ', second_turn_question_dataset[idx])
            print('answer-2 = ', output_text_2)

            
            out.append({
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": [{"index": 0,
                             "turns": [output_text_1, output_text_2]
                             }],
                "tstamp": time.time(),
            })
    
    
    if output_file is not None:
        with open(output_file, 'a') as f:
            for li in out:
                f.write(json.dumps(li) + "\n")

if __name__ == "__main__":
    fire.Fire(main)