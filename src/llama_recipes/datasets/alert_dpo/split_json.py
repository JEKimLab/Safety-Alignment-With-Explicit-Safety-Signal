import json
import sys
import os

def split_json(input_file):
    # Check if the input file has a .json extension
    if not input_file.endswith('.json'):
        print("Error: The input file must have a .json extension.")
        return

    # Generate output file names
    base_name = input_file.replace('.json', '')
    chosen_output_file = f"{base_name}_chosen.jsonl"
    rejected_output_file = f"{base_name}_rejected.jsonl"

    try:
        # Open the input file and load the JSON content
        with open(input_file, 'r') as f:
            data = json.load(f)

        # Check if the loaded data is a list (array)
        if not isinstance(data, list):
            print("Error: The JSON file must contain an array of objects.")
            return

        # Prepare lists to hold the chosen and rejected entries
        chosen_list = []
        rejected_list = []

        # Process each entry in the dataset
        for entry in data:
            common_fields = {
                "id": entry["id"],
                "prompt": entry["prompt"],
                "category": entry["category"]
            }
            
            # Add attack_type if it exists
            if "attack_type" in entry:
                common_fields["attack_type"] = entry["attack_type"]

            # Create chosen and rejected entries
            chosen_entry = {
                **common_fields,
                "response": entry["chosen"]
            }
            rejected_entry = {
                **common_fields,
                "response": entry["rejected"]
            }

            # Append to the respective lists
            chosen_list.append(chosen_entry)
            rejected_list.append(rejected_entry)

        # Write the chosen entries to a .jsonl file
        with open(chosen_output_file, 'w') as f:
            for obj in chosen_list:
                json_line = json.dumps(obj)
                f.write(json_line + '\n')

        # Write the rejected entries to a .jsonl file
        with open(rejected_output_file, 'w') as f:
            for obj in rejected_list:
                json_line = json.dumps(obj)
                f.write(json_line + '\n')

        print(f"Successfully split {input_file} into {chosen_output_file} and {rejected_output_file}")

    except FileNotFoundError:
        print(f"Error: The file {input_file} does not exist.")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON. Error: {e}")

if __name__ == "__main__":
    # Ensure that an input file is provided as an argument
    if len(sys.argv) != 2:
        print("Usage: python split_json.py <input_file.json>")
    else:
        input_file = sys.argv[1]
        split_json(input_file)
