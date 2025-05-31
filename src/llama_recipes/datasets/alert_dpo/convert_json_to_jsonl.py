import json
import sys
import os

def convert_json_to_jsonl(input_file):
    # Check if the input file has a .json extension
    if not input_file.endswith('.json'):
        print("Error: The input file must have a .json extension.")
        return

    # Generate the output file name by replacing .json with .jsonl
    output_file = input_file.replace('.json', '.jsonl')

    try:
        # Open the input file and load the JSON content
        with open(input_file, 'r') as f:
            data = json.load(f)

        # Check if the loaded data is a list (array)
        if not isinstance(data, list):
            print("Error: The JSON file must contain an array of objects.")
            return

        # Open the output file and write each JSON object as a line
        with open(output_file, 'w') as f:
            for obj in data:
                json_line = json.dumps(obj)
                f.write(json_line + '\n')

        print(f"Successfully converted {input_file} to {output_file}")

    except FileNotFoundError:
        print(f"Error: The file {input_file} does not exist.")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON. Error: {e}")

if __name__ == "__main__":
    # Ensure that an input file is provided as an argument
    if len(sys.argv) != 2:
        print("Usage: python convert_json_to_jsonl.py <input_file.json>")
    else:
        input_file = sys.argv[1]
        convert_json_to_jsonl(input_file)
