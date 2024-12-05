import json

def process_json_to_binary(input_file, output_file, threshold=0.1):
    """
    Process a JSON file, convert predictions to binary based on a threshold, 
    and save the results to a text file with values separated by commas.

    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to save the output text file.
        threshold (float): Threshold for binary conversion (default is 0.1).
    """
    # Load the JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Convert predictions to binary format
    binary_data = []
    for item in data:
        binary_predictions = [1 if pred >= threshold else 0 for pred in item["predictions"]]
        binary_data.append({"idx": item["idx"], "binary_predictions": binary_predictions})
    
    # Save the binary results to a text file
    with open(output_file, 'w') as f:
        for item in binary_data:
            idx = item["idx"]
            predictions = ','.join(map(str, item["binary_predictions"]))
            f.write(f"{predictions}\n")
    
    print(f"Binary results saved to {output_file}")

# Example usage
input_file = '/home/ali.mekky/Documents/NLP/Project/Cross-Country-Dialectal-Arabic-Identification/SPML/checkpoints/test_predictions.json'  # Replace with your input JSON file path
output_file = 'output.txt'  # Replace with your desired output text file path
threshold = float(input("Enter threshold (default is 0.1): ") or 0.1)  # Input threshold or default to 0.1

process_json_to_binary(input_file, output_file, threshold)
