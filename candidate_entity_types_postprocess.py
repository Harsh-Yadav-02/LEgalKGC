import csv
import os
from tqdm import tqdm
import argparse
import json

def load_relations_in_chunks(csv_file, chunk_size=1000):
    """Yield chunks of rows from the CSV file."""
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        chunk = []
        for row in reader:
            chunk.append(row)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

def build_entity_type_dict_from_chunks(csv_file, csv_chunk_size=1000):
    """
    Build a dictionary mapping each entity to its candidate entity types by processing the CSV in chunks.
    Each entity key maps to a list of candidate types.
    """
    entity_dict = {}
    for chunk in load_relations_in_chunks(csv_file, csv_chunk_size):
        for relation in tqdm(chunk, desc="Processing CSV chunk"):
            for key in ['head', 'tail']:
                entity = relation.get(key, '').strip()
                entity_type = relation.get(f"{key}_type", '').strip()
                if entity:
                    if entity not in entity_dict:
                        entity_dict[entity] = set()
                    if entity_type:
                        entity_dict[entity].add(entity_type)
    # Convert sets to lists for proper serialization.
    for entity in entity_dict:
        entity_dict[entity] = list(entity_dict[entity])
    return entity_dict

def save_original_entity_csv(entity_dict, filepath):
    """
    Saves the original entity dictionary to CSV.
    CSV will have columns: "entity" and "entity_types" (as a stringified list).
    """
    out_dir = os.path.dirname(filepath)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["entity", "entity_types"])
        writer.writeheader()
        for entity, types in entity_dict.items():
            writer.writerow({"entity": entity, "entity_types": str(types)})

def main():
    parser = argparse.ArgumentParser(
        description="Extract candidate entity types from the raw CSV."
    )
    parser.add_argument("--csv_file", type=str, required=True,
                        help="Path to the raw CSV file with columns: head, head_type, tail, tail_type.")
    parser.add_argument("--output", type=str, default="original_entities.csv",
                        help="Path to save the extracted entity dictionary CSV.")
    parser.add_argument("--csv_chunk_size", type=int, default=1000,
                        help="Number of CSV rows to process per chunk.")
    args = parser.parse_args()

    entity_dict = build_entity_type_dict_from_chunks(args.csv_file, args.csv_chunk_size)
    print("Extracted Entities:")
    print(json.dumps(entity_dict, indent=2))
    save_original_entity_csv(entity_dict, args.output)
    print(f"Original entity dictionary saved to {args.output}")

if __name__ == "__main__":
    main()
