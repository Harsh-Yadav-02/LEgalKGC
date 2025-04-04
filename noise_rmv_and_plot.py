import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file (update the file name as needed)
df = pd.read_csv(r'.output\canonicalized_dataset.csv')

# Normalize the 'sentence' column: strip extra whitespace and convert to lowercase
df['sentence'] = df['sentence'].str.strip().str.lower()

# Read the exceptions from exceptions.txt (one sentence per line), cleaning each line
with open(r'datasets\exceptions.txt', 'r') as f:
    exceptions = [line.strip().lower() for line in f if line.strip()]

# Debug: Print which exceptions are actually found in the CSV
print("Checking exceptions in CSV:")
for exc in exceptions:
    matching = df[df['sentence'] == exc]
    if not matching.empty:
        print(f"Found: {repr(exc)} (Count: {len(matching)})")
    else:
        print(f"Not found: {repr(exc)}")

# Identify rows that match any of the exceptions
skipped_rows = df[df['sentence'].isin(exceptions)]

# Print the unique skipped sentences for debugging
print("\nSkipped sentences:")
for sentence in skipped_rows['sentence'].unique():
    print(repr(sentence))

# Filter out rows where the 'sentence' matches any exception
df_filtered = df[~df['sentence'].isin(exceptions)]

# Select only the desired columns and rename 'canonical_relation' to 'relation'
df_filtered = df_filtered[['head', 'head_type', 'canonical_relation', 'tail', 'tail_type']]
df_filtered = df_filtered.rename(columns={'canonical_relation': 'relation'})

# Save the filtered DataFrame to a new CSV file
df_filtered.to_csv(r'.output\filtered.csv', index=False)

# Count the occurrences of each entity type in both head_type and tail_type
head_counts = df_filtered['head_type'].value_counts()
tail_counts = df_filtered['tail_type'].value_counts()

# Combine the counts (summing frequencies for entity types that appear in both columns)
entity_counts = head_counts.add(tail_counts, fill_value=0)

# Plot the histogram as a bar chart
plt.figure(figsize=(10, 6))
entity_counts.sort_values(ascending=False).plot(kind='bar')
plt.xlabel('Entity Type')
plt.ylabel('Frequency')
plt.title('Frequency of Entity Types (Head and Tail)')
plt.tight_layout()
plt.show()
