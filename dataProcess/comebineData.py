import json

def combine_files(file1_path, file2_path, output_path):
    # Load the first txt file with original data
    with open(file1_path, 'r') as f1:
        original_data = [line.split('. ', 1)[-1] for line in f1.readlines() if line.strip()]
        print(len(original_data))
    # Load the second txt file with extracted results
    with open(file2_path, 'r') as f2:
        results_data = [line.strip().replace('{', '').replace('}', '').replace('"', '').replace(':', '').replace(',', '').replace('Name', 'Name:').replace('Age', ', Age:').replace('Profession', ', Profession:').replace('Hobby', ', Hobby:') for line in f2]
        print(len(results_data))
    combined_data = []
    for i in range(len(original_data)):
        combined_data.append({
            "instruction":"""Extract Name,Age,Profession,and Hobby information from the following text description.""", 
            "input": original_data[i].strip(),
            "output": results_data[i]
        })

    # Save combined data to new txt file
    with open(output_path, 'w') as output_file:
        json.dump(combined_data, output_file, indent=4)

# Specify the file paths
file1_path = '/home/lina/upload/description.txt'
file2_path = '/home/lina/finetuneLLama/ExtractionResult.txt'
output_path = '/home/lina/finetuneLLama/llama3Bfinetune/CustomeData.json'

# Combine the files and save to the output path
combine_files(file1_path, file2_path, output_path)

print(f"Combined data has been saved to {output_path}")