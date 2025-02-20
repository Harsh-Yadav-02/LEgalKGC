import csv
import subprocess
from tqdm import tqdm  # Import tqdm for the progress bar

# Load the entity types
def load_entity_types(entity_file):
    with open(entity_file, 'r') as f:
        return [line.strip() for line in f.readlines()]

# Load the few-shot examples
def load_few_shot_examples(examples_file):
    with open(examples_file, 'r') as f:
        return f.readlines()

# Read the document file
def load_document(document_file):
    with open(document_file, 'r') as f:
        return f.readlines()

# Construct a prompt for Ollama
def construct_prompt(few_shot_examples, entity_types, sentence):
    # Create a few-shot prompt with examples
    prompt = f"Given the following entity types: {', '.join(entity_types)}.\n"
    prompt += "Extract the entities and their types from the following sentence:\n"
    
    # Add few-shot examples
    prompt += "\n".join(few_shot_examples) + "\n"
    
    # Add the sentence to process
    prompt += f"Sentence: {sentence}\nEntities and Types:"
    
    return prompt

# Call Ollama locally via subprocess to extract entities
def extract_entities_from_ollama(prompt):
    try:
        # Run Ollama via subprocess and capture the output
        result = subprocess.run(['ollama', 'run', 'deepseek-r1:8b'], input=prompt, text=True, capture_output=True)
        
        if result.returncode == 0:
            # Return the output from Ollama
            return result.stdout.strip()
        else:
            print(f"Error: {result.stderr}")
            return None
    except Exception as e:
        print(f"Error running Ollama: {str(e)}")
        return None

# Process each sentence and extract entities
def process_document(document_file, few_shot_file, entity_file, output_csv_file):
    entity_types = load_entity_types(entity_file)
    few_shot_examples = load_few_shot_examples(few_shot_file)
    document_lines = load_document(document_file)
    
    # Prepare CSV output
    with open(output_csv_file, mode='w', newline='') as csvfile:
        fieldnames = ['Entity', 'Entity Type', 'Sentence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Use tqdm to show progress while processing each line of the document
        for line in tqdm(document_lines, desc="Processing sentences", unit="sentence"):
            line = line.strip()
            if line:  # Ignore empty lines
                # Construct prompt
                prompt = construct_prompt(few_shot_examples, entity_types, line)
                
                # Call Ollama to extract entities
                extracted_entities = extract_entities_from_ollama(prompt)
                
                # Parse the response (you can modify based on response format)
                if extracted_entities:
                    entities = extracted_entities.split('\n')
                    for entity in entities:
                        # Assuming the response has the format 'Entity: <entity_name>, Type: <entity_type>'
                        parts = entity.split(',')
                        if len(parts) == 2:
                            entity_name = parts[0].split(':')[-1].strip()
                            entity_type = parts[1].split(':')[-1].strip()
                            writer.writerow({'Entity': entity_name, 'Entity Type': entity_type, 'Sentence': line})

# Example Usage
if __name__ == "__main__":
    document_file = './datasets/processed.txt'
    few_shot_file = './datasets/few_shot_examples.txt'
    entity_file = './datasets/entity_types.txt'
    output_csv_file = './output/entities.csv'
    
    process_document(document_file, few_shot_file, entity_file, output_csv_file)
