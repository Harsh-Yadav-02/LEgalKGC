import pandas as pd
import os

# Set pandas options to display all rows and columns without truncation
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Don't wrap lines in terminal
pd.set_option('display.max_colwidth', None)  # Don't truncate column content

def update_entity_types(mapping_file, results_file, output_file):
    # Read the CSV file with entity mappings
    mapping_df = pd.read_csv(mapping_file)
    # Create a dictionary with entity as key and final_type as value
    mapping_dict = pd.Series(mapping_df['final_type'].values, index=mapping_df['entity']).to_dict()
    
    # Read the results CSV file
    results_df = pd.read_csv(results_file)
    
    # Keep a copy of the original 'head_type' and 'tail_type' for comparison
    original_head_types = results_df['head_type'].copy()
    original_tail_types = results_df['tail_type'].copy()
    
    # Replace head_type using the mapping, if the head exists in mapping_dict;
    # otherwise, keep the original head_type
    results_df['head_type'] = results_df['head'].map(mapping_dict).fillna(results_df['head_type'])
    
    # Replace tail_type similarly using the tail column
    results_df['tail_type'] = results_df['tail'].map(mapping_dict).fillna(results_df['tail_type'])
    
    # Identify the rows where the head_type or tail_type changed
    changed_head_types = results_df[results_df['head_type'] != original_head_types]
    changed_tail_types = results_df[results_df['tail_type'] != original_tail_types]
    
    # Combine the changed head and tail entries
    changed_entities = pd.concat([changed_head_types[['head', 'head_type']], changed_tail_types[['tail', 'tail_type']]])
    
    # Remove duplicates (if an entity's type changed both in head and tail, only show it once)
    changed_entities = changed_entities.drop_duplicates()

    # Print the changes to the command line
    if not changed_entities.empty:
        print("The following entity types were changed:")
        print(changed_entities)
        print(f"\nTotal changes made: {len(changed_entities)}")
    else:
        print("No entity types were changed.")

    # Check if the output file exists
    if not os.path.exists(output_file):
        # File does not exist, so create it
        results_df.to_csv(output_file, index=False)
        print(f"{output_file} did not exist and has now been created.")
    else:
        # File exists, so update (overwrite) it
        results_df.to_csv(output_file, index=False)
        print(f"{output_file} already exists and has been updated.")

if __name__ == '__main__':
    # File names can be modified as per your file paths
    mapping_file = r".output\final_entities.csv"  # CSV file with two columns: entity, final_type
    results_file = r".output\results.csv"           # CSV file with head, head_type, tail, tail_type etc.
    output_file = r".output\results_updated.csv"    # Name for the updated output file
    update_entity_types(mapping_file, results_file, output_file)
