import argparse
import csv
import json
import subprocess
import multiprocessing
import re
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
                    # If no definition is provided, you can give a default message
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

# def load_document(document_file):
#     """
#     Load document sentences from a file (one sentence per line) and
#     return a list of tuples: (sentence_number, sentence).
#     """
#     sentences = []
#     with open(document_file, 'r', encoding='utf-8') as f:
#         for idx, line in enumerate(f):
#             line = line.strip()
#             if line:
#                 sentences.append((idx + 1, line))
#     return sentences

def load_paragraphs(document_file):
    """
    Load paragraphs from a file. Paragraphs are assumed to be separated by one or more blank lines.
    Returns a list of tuples: (paragraph_number, paragraph_text).
    """
    with open(document_file, 'r', encoding='utf-8') as f:
        content = f.read()
    # Split on one or more blank lines
    paragraphs = [para.strip() for para in re.split(r'\n\s*\n', content) if para.strip()]
    return [(idx + 1, para) for idx, para in enumerate(paragraphs)]


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

# def construct_prompt(few_shot_examples, entity_types, sentence, sentence_number):
#     """
#     Build the prompt for Deepseek with entity definitions.
#     """
#     prompt_parts = []
#     if few_shot_examples:
#         prompt_parts.append(few_shot_examples)
    
#     # Build a descriptive list of entity types and definitions.
#     entity_descriptions = "\n".join(
#         [f"- {etype}: {definition}" for etype, definition in entity_types.items()]
#     )
#     prompt_parts.append("Entity Types and Definitions:\n" + entity_descriptions)
    
#     prompt_parts.append("Extract all entities from the sentence below that belong to one of the above entity types.")
#     prompt_parts.append("For each entity, return an object with the following keys:")
#     prompt_parts.append("  - entity (the extracted text)")
#     prompt_parts.append("  - entity_type (one of the provided entity types)")
#     prompt_parts.append("  - sentence (the full sentence text)")
#     prompt_parts.append("  - sentence_number (the sentence's line number as provided)")
#     prompt_parts.append("Return only a single valid JSON array containing these objects. If no entity is found, return an empty array [].")
#     prompt_parts.append(f"Sentence Number: {sentence_number}")
#     prompt_parts.append(f"Sentence: {sentence}")
#     return "\n\n".join(prompt_parts)

def construct_prompt(few_shot_examples, entity_types, paragraph, paragraph_number):
    prompt_parts = []
    if few_shot_examples:
        prompt_parts.append(few_shot_examples)
    
    entity_descriptions = "\n".join(
        [f"- {etype}: {definition}" for etype, definition in entity_types.items()]
    )
    prompt_parts.append("Entity Types and Definitions:\n" + entity_descriptions)
    
    prompt_parts.append("Extract all entities from the paragraph below that belong to one of the above entity types.")
    prompt_parts.append("For each entity, return an object with the following keys:")
    prompt_parts.append("  - entity (the extracted text)")
    prompt_parts.append("  - entity_type (one of the provided entity types)")
    prompt_parts.append("  - paragraph (the full paragraph text)")
    prompt_parts.append("  - paragraph_number (the paragraph's number as provided)")
    prompt_parts.append("Return only a single valid JSON array containing these objects. If no entity is found, return an empty array [].")
    prompt_parts.append(f"Paragraph Number: {paragraph_number}")
    prompt_parts.append(f"Paragraph: {paragraph}")
    return "\n\n".join(prompt_parts)



def call_deepseek(prompt):
    """
    Call Deepseek via Ollama by invoking a subprocess.
    If successful, returns the raw output string (expected to be a JSON array)
    from Deepseek; otherwise, returns None.
    """
    try:
        result = subprocess.run(
            ['ollama', 'run', 'deepseek-r1:8b'],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=None,         # No timeout
            encoding='utf-8'      # Use UTF-8 for decoding
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            # print(f"Fetched Output:\n{output}\n")
            cleaned = clean_output(output)
            json_text = extract_json(cleaned)
            return json_text
        else:
            print("Error calling Deepseek:", result.stderr)
            return None
    except Exception as e:
        print("Exception while calling Deepseek:", str(e))
        return None

# def worker(input_queue, output_queue, few_shot_examples, entity_types):
#     """
#     Worker process that:
#       - Reads (sentence_number, sentence) tuples from the input_queue.
#       - For each tuple, builds the prompt and calls Deepseek.
#       - Parses the JSON output (a JSON array) and puts each extracted object into the output_queue.
#       - Exits when it reads a sentinel value (None).
#     """
#     while True:
#         item = input_queue.get()
#         if item is None:
#             break  # Sentinel received; exit.
#         sentence_number, sentence = item
#         prompt = construct_prompt(few_shot_examples, entity_types, sentence, sentence_number)
#         result = call_deepseek(prompt)
#         if result:
#             try:
#                 parsed = json.loads(result)
#                 if isinstance(parsed, list):
#                     for obj in parsed:
#                         obj.setdefault('sentence', sentence)
#                         obj.setdefault('sentence_number', sentence_number)
#                         # Add the definition based on the extracted entity_type.
#                         etype = obj.get('entity_type', '')
#                         obj.setdefault('definition', entity_types.get(etype, "No definition provided"))
#                         output_queue.put(obj)
#                 else:
#                     output_queue.put(None)
#             except json.JSONDecodeError:
#                 print("JSON decode error for sentence:", sentence)
#                 print("Raw fetched output:", result)
#                 output_queue.put(None)
#         else:
#             output_queue.put(None)

def worker(input_queue, output_queue, few_shot_examples, entity_types):
    while True:
        item = input_queue.get()
        if item is None:
            break  # Sentinel received; exit.
        paragraph_number, paragraph = item
        prompt = construct_prompt(few_shot_examples, entity_types, paragraph, paragraph_number)
        result = call_deepseek(prompt)
        if result:
            try:
                parsed = json.loads(result)
                if isinstance(parsed, list):
                    for obj in parsed:
                        obj.setdefault('paragraph', paragraph)
                        obj.setdefault('paragraph_number', paragraph_number)
                        etype = obj.get('entity_type', '')
                        obj.setdefault('definition', entity_types.get(etype, "No definition provided"))
                        output_queue.put(obj)
                else:
                    output_queue.put(None)
            except json.JSONDecodeError:
                print("JSON decode error for paragraph:", paragraph)
                print("Raw fetched output:", result)
                output_queue.put(None)
        else:
            output_queue.put(None)



# def process_document(document_file, few_shot_file, entity_file,
#                      output_json_file, output_csv_file, num_workers):
#     """
#     Orchestrates:
#       - Loading input files.
#       - Starting persistent worker processes.
#       - Distributing (sentence_number, sentence) tuples for processing.
#       - Collecting extracted entities.
#       - Writing them to JSON and CSV.
#     """
#     entity_types = load_entity_types(entity_file)
#     few_shot_examples = load_few_shot_examples(few_shot_file)
#     sentence_tuples = load_paragraphs(document_file)

#     manager = multiprocessing.Manager()
#     input_queue = manager.Queue()
#     output_queue = manager.Queue()

#     workers = []
#     for _ in range(num_workers):
#         p = multiprocessing.Process(target=worker,
#                                     args=(input_queue, output_queue, few_shot_examples, entity_types))
#         p.start()
#         workers.append(p)

#     for item in sentence_tuples:
#         input_queue.put(item)
    
#     for _ in range(num_workers):
#         input_queue.put(None)

#     results = []
#     num_sentences = len(sentence_tuples)
#     pbar = tqdm(total=num_sentences, desc="Processing sentences")
#     collected_sentences = 0
#     while collected_sentences < num_sentences:
#         try:
#             item = output_queue.get(timeout=5)
#             if item is not None:
#                 results.append(item)
#         except Exception:
#             collected_sentences += 1
#             pbar.update(1)
#     pbar.close()

#     for p in workers:
#         p.join()

#     with open(output_json_file, 'w', encoding='utf-8') as jf:
#         json.dump(results, jf, indent=4)

#     with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
#         fieldnames = ['entity', 'entity_type', 'sentence', 'sentence_number']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         for item in results:
#             writer.writerow({
#                 'entity': item.get('entity', ''),
#                 'entity_type': item.get('entity_type', ''),
#                 'sentence': item.get('sentence', ''),
#                 'sentence_number': item.get('sentence_number', '')
#             })

def process_document(document_file, few_shot_file, entity_file,
                     output_json_file, output_csv_file, num_workers):
    entity_types = load_entity_types(entity_file)
    few_shot_examples = load_few_shot_examples(few_shot_file)
    # Use the new loader:
    paragraph_tuples = load_paragraphs(document_file)

    manager = multiprocessing.Manager()
    input_queue = manager.Queue()
    output_queue = manager.Queue()

    workers = []
    for _ in range(num_workers):
        p = multiprocessing.Process(target=worker,
                                    args=(input_queue, output_queue, few_shot_examples, entity_types))
        p.start()
        workers.append(p)

    for item in paragraph_tuples:
        input_queue.put(item)
    
    for _ in range(num_workers):
        input_queue.put(None)

    results = []
    num_paragraphs = len(paragraph_tuples)
    pbar = tqdm(total=num_paragraphs, desc="Processing paragraphs")
    collected_paragraphs = 0
    while collected_paragraphs < num_paragraphs:
        try:
            item = output_queue.get(timeout=5)
            if item is not None:
                results.append(item)
        except Exception:
            collected_paragraphs += 1
            pbar.update(1)
    pbar.close()

    for p in workers:
        p.join()

    with open(output_json_file, 'w', encoding='utf-8') as jf:
        json.dump(results, jf, indent=4)

    with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['entity', 'entity_type', 'paragraph', 'paragraph_number']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow({
                'entity': item.get('entity', ''),
                'entity_type': item.get('entity_type', ''),
                'paragraph': item.get('paragraph', ''),
                'paragraph_number': item.get('paragraph_number', '')
            })



def main():
    parser = argparse.ArgumentParser(
        description="Extract all legal entities (with their types) from each sentence using Deepseek via Ollama with persistent worker processes. Outputs both JSON and CSV files."
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
