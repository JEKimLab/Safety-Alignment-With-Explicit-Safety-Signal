import json
import sys
import os

def convert_jsonl_to_json(input_file):
    # Check if the input file has a .jsonl extension
    if not input_file.endswith('.jsonl'):
        print("Error: The input file must have a .jsonl extension.")
        return

    # Generate the output file name by replacing .jsonl with .json
    output_file = input_file.replace('.jsonl', '.json')

    # Create an empty list to hold all the JSON objects
    json_list = []

    try:
        # Open the input file and read each line
        with open(input_file, 'r') as f:
            for line in f:
                # Parse each line as a JSON object and append it to the list
                json_obj = json.loads(line)
                json_list.append(json_obj)

        # Write the list of JSON objects to a .json file
        with open(output_file, 'w') as f:
            json.dump(json_list, f, indent=4)  # The indent parameter makes the output more readable

        print(f"Successfully converted {input_file} to {output_file}")
    
    except FileNotFoundError:
        print(f"Error: The file {input_file} does not exist.")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON on line: {line}. Error: {e}")

if __name__ == "__main__":
    # Ensure that an input file is provided as an argument
    if len(sys.argv) != 2:
        print("Usage: python convert_jsonl_to_json.py <input_file.jsonl>")
    else:
        input_file = sys.argv[1]
        convert_jsonl_to_json(input_file)
