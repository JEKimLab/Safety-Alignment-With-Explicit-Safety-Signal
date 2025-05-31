import json

# Keywords or phrases to identify harmful content
harmful_keywords = [
    "curse words", "cheat on my husband", "rob a convenience store", 
    "hurting myself", "get a friend to have sex", "be trusted", 
    "get a fake ID", "relationship without your parents knowing", 
    "physically flirt", "lazy college student", "married boyfriend", 
    "flip someone off", "bullet made from ice", "hire a hitman", 
    "Hitler writes a second book"
]

def contains_harmful_content(query):
    """
    Check if the query contains any harmful keywords or phrases.
    """
    return any(keyword in query for keyword in harmful_keywords)

def filter_harmful_data(input_file, output_file):
    """
    Filter out harmful queries from the input JSON file and save the filtered data to the output file.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Filter out items with harmful queries
    filtered_data = []
    for item in data:
        if 'conversations' in item and isinstance(item['conversations'], list):
            first_conversation = item['conversations'][0]
            if 'value' in first_conversation and not contains_harmful_content(first_conversation['value']):
                filtered_data.append(item)
            else:
                print(first_conversation['value'])
    
    # Save the filtered data to a new file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)
    
    # Print the original and filtered data counts
    print(f"Original number of samples: {len(data)}")
    print(f"Filtered number of samples: {len(filtered_data)}")

# File paths
input_file = 'lima.json'  # Replace with your input file path
output_file = 'lima_no_safety.json'  # The output file for the filtered data

# Run the filtering process
filter_harmful_data(input_file, output_file)
