import json
import csv

# Replace 'input.json' with the actual path to your JSON file
input_file = 'ontology.json'
output_file = 'legal_predicates.csv'

# Read JSON data from file
with open(input_file, 'r') as f:
    # If the file contains a JSON array
    data = json.load(f)

# Use a set to track unique (predicate, definition) pairs
unique_pairs = set()

for entry in data:
    predicate = entry.get("predicate_")
    definition = entry.get("definition")
    if predicate and definition:
        unique_pairs.add((predicate, definition))

# Write the unique pairs to a CSV file with two columns: predicate and definition
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # Write header
    writer.writerow(["predicate", "definition"])
    
    # Write rows
    for predicate, definition in unique_pairs:
        writer.writerow([predicate, definition])

print(f"CSV file '{output_file}' created successfully.")
