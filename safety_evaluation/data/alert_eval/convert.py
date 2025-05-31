import json
import csv

def jsonl_to_csv(input_file, output_prefix):
    # Dictionary to store extracted prompts by attack_type
    attack_type_prompts = {}

    # Open the input JSONL file
    with open(input_file, 'r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            # Parse the JSON object
            data = json.loads(line)
            attack_type = data.get("attack_type")
            full_prompt = data.get("prompt", "")

            # Extract text between ### Instruction:\n and \n### Response:\n
            start_token = "### Instruction:\n"
            end_token = "\n### Response:\n"
            start_idx = full_prompt.find(start_token) + len(start_token)
            end_idx = full_prompt.find(end_token)
            if start_idx >= len(start_token) and end_idx != -1:
                extracted_prompt = full_prompt[start_idx:end_idx].strip()
            else:
                extracted_prompt = ""

            # Add the extracted prompt to the corresponding attack_type list
            if attack_type not in attack_type_prompts:
                attack_type_prompts[attack_type] = []
            attack_type_prompts[attack_type].append(extracted_prompt)

    # Write prompts to separate CSV files based on attack_type
    for attack_type, prompts in attack_type_prompts.items():
        output_file = f"{output_prefix}_{attack_type}.csv"
        with open(output_file, 'w', encoding='utf-8', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Write the header
            # writer.writerow(["prompt"])
            # Write each prompt as a row
            for prompt in prompts:
                writer.writerow([prompt])

    print(f"Conversion completed. CSV files have been saved with prefix '{output_prefix}'.")

# Example usage
input_file = "/home/jli265/workspace/llama-recipes/safety_evaluation/data/alert_eval/alert_adversarial.jsonl"  # Replace with your JSONL file
output_prefix = "alert"    # Prefix for output CSV files
jsonl_to_csv(input_file, output_prefix)
