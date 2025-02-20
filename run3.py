import argparse
import csv
import json
import multiprocessing
from tqdm import tqdm
from langchain_ollama import ChatOllama  # Ensure you have this package installed

def load_entity_types(entity_file):
    """Load valid entity types (one per line) from a file."""
    with open(entity_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def load_few_shot_examples(examples_file):
    """Load few-shot examples from a file as a text block."""
    try:
        with open(examples_file, 'r') as f:
            return f.read().strip()
    except Exception:
        return ""

def load_document(document_file):
    """
    Load document sentences from a file (one sentence per line) and
    return a list of tuples: (sentence_number, sentence).
    """
    sentences = []
    with open(document_file, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if line:
                sentences.append((idx + 1, line))
    return sentences

def construct_prompt(few_shot_examples, entity_types, sentence, sentence_number):
    """
    Build the prompt for Deepseek.
    
    The prompt includes:
      - The few-shot examples (if provided).
      - A list of valid entity types.
      - Clear instructions to extract all entities from the sentence that belong to one
        of the provided entity types.
      - For each extracted entity, return an object with the following keys:
            "entity", "entity_type", "sentence", "sentence_number"
      - If no entity is found, return an empty JSON array [].
      - Do not include any additional text, markdown formatting, or explanations.
    """
    prompt_parts = []
    if few_shot_examples:
        prompt_parts.append(few_shot_examples)
    prompt_parts.append(f"Entity Types: {', '.join(entity_types)}.")
    prompt_parts.append("Extract all entities from the sentence below that belong to one of the above entity types.")
    prompt_parts.append("For each entity, return an object with the following keys:")
    prompt_parts.append("  - entity (the extracted text)")
    prompt_parts.append("  - entity_type (one of the provided entity types)")
    prompt_parts.append("  - sentence (the full sentence text)")
    prompt_parts.append("  - sentence_number (the sentence's line number as provided)")
    prompt_parts.append("Return only a single valid JSON array containing these objects. If no entity is found, return an empty array [].")
    prompt_parts.append(f"Sentence Number: {sentence_number}")
    prompt_parts.append(f"Sentence: {sentence}")
    return "\n\n".join(prompt_parts)

def call_deepseek(prompt, llm):
    """
    Call Deepseek via the ChatOllama interface.
    If successful, returns the raw output string from the model.
    """
    try:
        result = llm.invoke(prompt)
        return result
    except Exception as e:
        print("Exception while calling deepseek via ChatOllama:", e)
        return None

def worker(input_queue, output_queue, few_shot_examples, entity_types):
    """
    Worker process that:
      - Creates its own ChatOllama instance.
      - Reads (sentence_number, sentence) tuples from the input_queue.
      - For each tuple, builds the prompt and calls Deepseek.
      - Parses the JSON output (a JSON array) and puts each extracted object into the output_queue.
      - Exits when it reads a sentinel value (None).
    """
    # Create a persistent ChatOllama instance in this worker.
    llm = ChatOllama(model="deepseek-r1:8b", temperature=0.0, format='json')
    
    while True:
        item = input_queue.get()
        if item is None:
            break  # Sentinel received; exit.
        sentence_number, sentence = item
        prompt = construct_prompt(few_shot_examples, entity_types, sentence, sentence_number)
        result = call_deepseek(prompt, llm)
        if result:
            try:
                parsed = json.loads(result)
                if isinstance(parsed, list):
                    for obj in parsed:
                        # Make sure the keys "sentence" and "sentence_number" are present.
                        obj.setdefault('sentence', sentence)
                        obj.setdefault('sentence_number', sentence_number)
                        output_queue.put(obj)
                else:
                    output_queue.put(None)
            except json.JSONDecodeError:
                print("JSON decode error for sentence:", sentence)
                output_queue.put(None)
        else:
            output_queue.put(None)

def process_document(document_file, few_shot_file, entity_file,
                     output_json_file, output_csv_file, num_workers):
    """
    Orchestrates:
      - Loading input files.
      - Starting persistent worker processes.
      - Distributing (sentence_number, sentence) tuples for processing.
      - Collecting extracted entities.
      - Writing them to JSON and CSV.
    """
    entity_types = load_entity_types(entity_file)
    few_shot_examples = load_few_shot_examples(few_shot_file)
    sentence_tuples = load_document(document_file)

    manager = multiprocessing.Manager()
    input_queue = manager.Queue()
    output_queue = manager.Queue()

    workers = []
    for _ in range(num_workers):
        p = multiprocessing.Process(target=worker,
                                    args=(input_queue, output_queue, few_shot_examples, entity_types))
        p.start()
        workers.append(p)

    # Enqueue each (sentence_number, sentence) tuple for processing.
    for item in sentence_tuples:
        input_queue.put(item)
    
    # Enqueue sentinel values to signal workers to exit.
    for _ in range(num_workers):
        input_queue.put(None)

    # Collect results.
    results = []
    num_sentences = len(sentence_tuples)
    pbar = tqdm(total=num_sentences, desc="Processing sentences")
    collected_sentences = 0
    while collected_sentences < num_sentences:
        try:
            item = output_queue.get(timeout=5)
            if item is not None:
                results.append(item)
        except Exception:
            collected_sentences += 1
            pbar.update(1)
    pbar.close()

    # Ensure all workers have terminated.
    for p in workers:
        p.join()

    # Write results to JSON.
    with open(output_json_file, 'w') as jf:
        json.dump(results, jf, indent=4)

    # Write results to CSV.
    with open(output_csv_file, 'w', newline='') as csvfile:
        fieldnames = ['entity', 'entity_type', 'sentence', 'sentence_number']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow({
                'entity': item.get('entity', ''),
                'entity_type': item.get('entity_type', ''),
                'sentence': item.get('sentence', ''),
                'sentence_number': item.get('sentence_number', '')
            })

def main():
    parser = argparse.ArgumentParser(
        description="Extract all legal entities (with their types) from each sentence using ChatOllama. "
                    "Outputs both JSON and CSV files."
    )
    parser.add_argument('--document_file', type=str, required=True,
                        help="Path to the input document file (one sentence per line).")
    parser.add_argument('--few_shot_file', type=str, required=True,
                        help="Path to the few-shot examples file.")
    parser.add_argument('--entity_file', type=str, required=True,
                        help="Path to the file containing valid entity types (one per line).")
    parser.add_argument('--output_json_file', type=str, default='output.json',
                        help="Path for the output JSON file.")
    parser.add_argument('--output_csv_file', type=str, default='output.csv',
                        help="Path for the output CSV file.")
    parser.add_argument('--num_workers', type=int, default=16,
                        help="Number of persistent worker processes.")
    args = parser.parse_args()

    process_document(args.document_file,
                     args.few_shot_file,
                     args.entity_file,
                     args.output_json_file,
                     args.output_csv_file,
                     args.num_workers)

if __name__ == '__main__':
    main()




# python run3.py \
#   --document_file ./datasets/processed.txt \
#   --few_shot_file ./datasets/few_shot_examples.txt \
#   --entity_file ./datasets/entity_types.txt \
#   --output_json_file results3.json \
#   --output_csv_file results3.csv \
#   --num_workers 16