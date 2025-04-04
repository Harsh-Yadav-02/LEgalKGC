import argparse
import csv
import json
import subprocess
import multiprocessing
import re
import os
from tqdm import tqdm

def load_entity_types(entity_file):
    """Load valid entity types and their definitions from a file.
    
    Each line in the file should be formatted as:
      EntityType: Definition text...
    """
    entity_dict = {}
    with open(entity_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(":", 1)  # Split at the first colon
                if len(parts) == 2:
                    entity, definition = parts
                    entity_dict[entity.strip()] = definition.strip()
                else:
                    entity_dict[line] = "No definition provided"
    return entity_dict

def load_few_shot_examples(examples_file):
    """Load few-shot examples from a file as a text block."""
    try:
        with open(examples_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print("Error reading few-shot file:", e)
        return ""
    
def chunk_entity_types(entity_types, chunk_size=10):
    """Yield smaller dictionaries of entity types of size chunk_size."""
    items = list(entity_types.items())
    for i in range(0, len(items), chunk_size):
        yield dict(items[i:i+chunk_size])

def load_document(document_file):
    """
    Load document sentences from a file (one sentence per line) and
    return a list of tuples: (sentence_number, sentence).
    """
    sentences = []
    with open(document_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if line:
                sentences.append((idx + 1, line))
    return sentences

def clean_output(output):
    """
    Remove any <think>...</think> tags and extra whitespace from the output.
    """
    cleaned = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL)
    return cleaned.strip()

def extract_json(text):
    """
    Extract a JSON array from the text using a regular expression.
    If found, returns the JSON substring; otherwise, returns the original text.
    """
    match = re.search(r'(\[.*\])', text, re.DOTALL)
    if match:
        return match.group(1)
    return text

def construct_prompt(few_shot_examples, entity_types, sentence, sentence_number):
    """
    Build a refined prompt for Deepseek relation extraction.
    
    The prompt includes:
      - (Optional) Few-shot examples for guidance.
      - A clear set of instructions for extracting legal relations.
      - A concise list of entity types and definitions.
      - The sentence number and sentence text.
    
    Note: The examples above are provided for format guidance only and should not be included in the final output.
    """
    prompt_parts = []
    if few_shot_examples:
        prompt_parts.append("Examples (for guidance only):\n" + few_shot_examples)
    
    prompt_parts.append("Instructions:")
    prompt_parts.append("You are a legal expert. Extract all legal relations from the sentence below using only the following entity types exactly as defined.")
    
    # List entity types and their definitions.
    entity_descriptions = "\n".join([f"- {etype}: {definition}" for etype, definition in entity_types.items()])
    prompt_parts.append("Entity Types and Definitions:")
    prompt_parts.append(entity_descriptions)
    
    prompt_parts.append("For each relation, return a JSON object with exactly these keys (all values must be strings):")
    prompt_parts.append("  - head: the first entity text")
    prompt_parts.append("  - head_type: one of the provided entity types")
    prompt_parts.append("  - relation: the phrase linking head and tail")
    prompt_parts.append("  - tail: the second entity text")
    prompt_parts.append("  - tail_type: one of the provided entity types")
    prompt_parts.append("  - relation_definition: a brief explanation of the legal connection")
    prompt_parts.append("  - sentence: the full sentence text")
    prompt_parts.append("  - sentence_number: the sentence's line number")
    prompt_parts.append("Output only a single valid JSON array. Do not include any extra text, markdown formatting, or explanations.")
    
    # Append the actual sentence details.
    prompt_parts.append(f"Sentence Number: {sentence_number}")
    prompt_parts.append(f"Sentence: {sentence}")
    
    return "\n\n".join(prompt_parts)

def call_deepseek(prompt):
    """
    Call Deepseek via Ollama by invoking a subprocess.
    If successful, returns the raw output string (expected to be a JSON array)
    from Deepseek; otherwise, returns None.
    """
    try:
        result = subprocess.run(
            ['ollama', 'run', 'deepseek-r1:14b'],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=None,
            encoding='utf-8'
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            cleaned = clean_output(output)
            json_text = extract_json(cleaned)
            return json_text
        else:
            print("Error calling Deepseek:", result.stderr)
            return None
    except Exception as e:
        print("Exception while calling Deepseek:", str(e))
        return None

def worker(input_queue, output_queue, few_shot_examples, entity_types, chunk_size, log_prompt=False, log_lock=None):
    while True:
        item = input_queue.get()
        if item is None:
            break  # Sentinel received; exit.
        sentence_number, sentence = item
        all_results = []
        # Process the sentence in chunks of entity types using the specified chunk_size.
        for entity_chunk in chunk_entity_types(entity_types, chunk_size):
            # Build a prompt using only this chunk.
            prompt = construct_prompt(few_shot_examples, entity_chunk, sentence, sentence_number)
            # If logging is enabled, save the prompt to a file.
            if log_prompt and log_lock is not None:
                with log_lock:
                    with open("sent_prompt.txt", "a", encoding="utf-8") as f:
                        f.write(f"DEBUG: Final prompt for sentence {sentence_number}:\n")
                        f.write(prompt + "\n")
                        f.write("=" * 60 + "\n")
            result = call_deepseek(prompt)
            if result:
                try:
                    parsed = json.loads(result)
                    if isinstance(parsed, list):
                        all_results.extend(parsed)
                except json.JSONDecodeError:
                    print("JSON decode error for sentence:", sentence)
                    print("Raw fetched output:", result)
        # Optionally, deduplicate results...
        unique_results = {json.dumps(obj, sort_keys=True): obj for obj in all_results}.values()
        for obj in unique_results:
            obj.setdefault('sentence', sentence)
            obj.setdefault('sentence_number', sentence_number)
            output_queue.put(obj)

def process_document(document_file, few_shot_file, entity_file,
                     output_json_file, output_csv_file, num_workers, log_prompt, chunk_size):
    """
    Orchestrates:
      - Loading input files.
      - Starting persistent worker processes.
      - Distributing (sentence_number, sentence) tuples for processing.
      - Collecting extracted relations.
      - Writing them to JSON and CSV.
    """
    entity_types = load_entity_types(entity_file)
    few_shot_examples = load_few_shot_examples(few_shot_file)
    sentence_tuples = load_document(document_file)

    manager = multiprocessing.Manager()
    input_queue = manager.Queue()
    output_queue = manager.Queue()
    # Create a lock for writing prompts to file.
    log_lock = multiprocessing.Lock()

    workers = []
    for _ in range(num_workers):
        p = multiprocessing.Process(target=worker,
                                    args=(input_queue, output_queue, few_shot_examples, entity_types, chunk_size, log_prompt, log_lock))
        p.start()
        workers.append(p)

    for item in sentence_tuples:
        input_queue.put(item)
    
    for _ in range(num_workers):
        input_queue.put(None)

    results = []
    num_sentences = len(sentence_tuples)
    pbar = tqdm(total=num_sentences, desc="Processing sentences")
    for _ in range(num_sentences):
        item = output_queue.get()  # blocking get for each expected sentence
        if item is not None:
            results.append(item)
        pbar.update(1)
    pbar.close()

    for p in workers:
        p.join()

    # Create directories if needed
    json_dir = os.path.dirname(output_json_file)
    if json_dir and not os.path.exists(json_dir):
        os.makedirs(json_dir)

    csv_dir = os.path.dirname(output_csv_file)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    with open(output_json_file, 'w', encoding='utf-8') as jf:
        json.dump(results, jf, indent=4)

    with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['head', 'head_type', 'relation', 'tail', 'tail_type', 'relation_definition', 'sentence', 'sentence_number']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow({
                'head': item.get('head', ''),
                'head_type': item.get('head_type', ''),
                'relation': item.get('relation', ''),
                'tail': item.get('tail', ''),
                'tail_type': item.get('tail_type', ''),
                'relation_definition': item.get('relation_definition', ''),
                'sentence': item.get('sentence', ''),
                'sentence_number': item.get('sentence_number', '')
            })

def main():
    parser = argparse.ArgumentParser(
        description="Extract legal relations from each sentence using Deepseek via Ollama with persistent worker processes. Outputs both JSON and CSV files."
    )
    parser.add_argument('--log_prompt', action='store_true', help="Save the prompt sent to the model in a text file named sent_prompt.txt.")
    parser.add_argument('--document_file', type=str, required=True,
                        help="Path to the input document file (one sentence per line).")
    parser.add_argument('--few_shot_file', type=str, required=True,
                        help="Path to the few-shot examples file (with examples following the new format).")
    parser.add_argument('--entity_file', type=str, required=True,
                        help="Path to the file containing valid entity types and their definitions (one per line, formatted as 'EntityType: Definition').")
    parser.add_argument('--output_json_file', type=str, default='output.json',
                        help="Path for the output JSON file.")
    parser.add_argument('--output_csv_file', type=str, default='output.csv',
                        help="Path for the output CSV file.")
    parser.add_argument('--num_workers', type=int, default=16,
                        help="Number of persistent worker processes.")
    parser.add_argument('--chunk_size', type=int, default=10,
                        help="Number of entity types per chunk when constructing the prompt.")
    args = parser.parse_args()

    process_document(args.document_file,
                     args.few_shot_file,
                     args.entity_file,
                     args.output_json_file,
                     args.output_csv_file,
                     args.num_workers,
                     args.log_prompt,
                     args.chunk_size)

if __name__ == '__main__':
    main()


##python run4.py --document_file ./datasets/processed_cr.txt --few_shot_file ./datasets/few_shot_examples_triple.txt --entity_file ./datasets/entity_types.txt --output_json_file .output/results.json --output_csv_file .output/results.csv --num_workers 16 --log_prompt --chunk_size 10
