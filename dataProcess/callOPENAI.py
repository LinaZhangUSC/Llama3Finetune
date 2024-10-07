import json
from textwrap import dedent
from openai import OpenAI
import traceback
#client = OpenAI(api_key='') 
MODEL = "gpt-4o-2024-08-06"
exraction_prompt = '''
You will extract the following information from the given text:
Name: The full name of the person mentioned,only include the person's name, without titles like Mr. or Dr.
Age: The person's age, if provided.
Profession: The person's job or occupation.
Hobby: Any mentioned activities or hobbies the person enjoys or is passionate about.
'''

def extraction_entity(text):
    response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system", 
            "content": dedent(exraction_prompt)
        },
        {
            "role": "user", 
            "content": text
        }
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "result_json",
            "schema": {
                "type": "object",
                "properties": {
                    "Name": {"type": "string"},
                    "Age": {"type": "string"},
                    "Profession": {"type": "string"},
                    "Hobby": {"type": "string"}
                },
                "required": ["Name", "Age","Profession","Hobby"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
    )

    return response.choices[0].message

# Testing with an example question
# text = "The kitchen hummed with activity as Sophia Chen, 28, expertly plated another farm-to-table creation. Her innovative approach to sustainable cuisine had earned her restaurant critical acclaim, a passion matched only by her dedication to zero-waste living and urban gardening."
# result = extraction_entity(text) 
# print(result.content)

def process_file(input_file, output_file):
    try:
        # Read the description.txt file
        with open(input_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()

        # Process each line and write results to the output file
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for line in lines:
                if line.strip():  # Only process non-empty lines
                    result = extraction_entity(line)
                    outfile.write(result.content + '\n')
                     
        print(f"Processing completed! Results saved in {output_file}")

    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

# Main program entry
if __name__ == "__main__":
    input_file = '/home/lina/upload/description.txt'  # Input file
    output_file = 'ExtractionResult.txt'  # Output file
    process_file(input_file, output_file)
