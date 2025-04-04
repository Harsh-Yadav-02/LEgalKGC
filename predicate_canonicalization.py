import argparse
import csv
import json
import os
import re
import subprocess
import time
from collections import Counter
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# -------------------------------
# File addresses (adjust paths as needed)
# -------------------------------
CANONICAL_CSV = 'legal_predicates.csv'      # CSV with columns: predicate, definition
DATASET_CSV = '.output\\results_updated.csv'  # CSV with columns: head, head_type, relation, tail, tail_type, relation definition, sentence, sentence_number
OUTPUT_CSV = '.output\\canonicalized_dataset.csv'  # Output CSV file

# -------------------------------
# LLM call using DeepSeek via Ollama
# -------------------------------
def extract_json(text):
    """
    Use a regular expression to extract the first JSON object from the text.
    """
    pattern = r"(\{(?:.|\n)*\})"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None

def call_llm(prompt, timeout=300):
    """
    Call DeepSeek R1 14B via Ollama with the given prompt.
    Expects a strictly valid JSON response.
    """
    try:
        result = subprocess.run(
            ['ollama', 'run', 'deepseek-r1:14b'],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            encoding='utf-8'
        )
        print("---------- LLM Prompt ----------")
        print(prompt)
        print("---------- LLM Output ----------")
        print(result.stdout)
        print("---------- LLM Error (if any) ----------")
        print(result.stderr)
        output = result.stdout.strip()
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            extracted = extract_json(output)
            if extracted:
                try:
                    return json.loads(extracted)
                except json.JSONDecodeError:
                    return None
            else:
                return None
    except Exception as e:
        print("Exception while calling LLM:", str(e))
        return None

# -------------------------------
# Build LLM prompt for canonicalizing a relation
# -------------------------------
def build_prompt_for_relation(original_sentence, current_relation, candidate_list, relation_definition):
    """
    Build a prompt that instructs the LLM to choose the best canonical predicate.
    The candidate_list is a list of tuples: (predicate, definition, similarity_score).
    In addition to the candidates, instruct the LLM to output "unchanged" if none of the candidates fit.
    """
    candidates_formatted = [
        {"predicate": cand, "definition": defi, "score": score}
        for cand, defi, score in candidate_list
    ]
    prompt = f"""You are a legal expert. You are provided with:
1. An original sentence from a legal document.
2. The current relation (predicate) extracted from that sentence.
3. The relation definition as extracted from the dataset.
4. A candidate list of canonical predicates along with their definitions and similarity scores.

Your task is to analyze the context and decide which canonical predicate best represents the relation.
If none of the candidates are suitable, output "unchanged" to indicate that the predicate should remain as-is.
IMPORTANT: Output strictly valid JSON. Do not include any chain-of-thought or commentary.
The output must be a JSON object mapping the current relation (as provided below) to its chosen canonical predicate or the string "unchanged".

Original Sentence:
{original_sentence}

Current Relation:
{current_relation}

Relation Definition:
{relation_definition}

Candidate Canonical Predicates:
{json.dumps(candidates_formatted, indent=2)}

Revised Result:"""
    return prompt

# -------------------------------
# Compute top k candidates using cosine similarity
# -------------------------------
def get_top_k_candidates(relation_definition, canonical_embeddings, canonical_predicates, canonical_definitions, model, k=5):
    # Compute embedding for the relation definition from the dataset
    relation_definition = str(relation_definition)
    relation_embedding = model.encode([relation_definition], convert_to_tensor=True)
    # Compute cosine similarities
    cos_scores = cosine_similarity(relation_embedding.cpu().numpy(), canonical_embeddings.cpu().numpy())[0]
    
    # Get indices of top k scores
    top_k_indices = np.argsort(cos_scores)[-k:][::-1]
    
    # Create list of candidates as (predicate, definition, score)
    candidates = [(canonical_predicates[i], canonical_definitions[i], float(cos_scores[i])) for i in top_k_indices]
    return candidates

# -------------------------------
# Main canonicalization process
# -------------------------------
def main():
    start_time = time.time()
    
    # Load canonical predicates CSV (expects columns: 'predicate', 'definition')
    canonical_df = pd.read_csv(CANONICAL_CSV)
    canonical_definitions = canonical_df['definition'].tolist()
    canonical_predicates = canonical_df['predicate'].tolist()
    
    # Load dataset CSV (expects columns: 'head', 'head_type', 'relation', 'tail', 'tail_type', 'relation definition', 'sentence', 'sentence_number')
    dataset_df = pd.read_csv(DATASET_CSV)
    
    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    canonical_embeddings = model.encode(canonical_definitions, convert_to_tensor=True)
    
    canonicalized_predicates = []  # To store the final canonical predicate for each row
    unchanged_counter = 0         # Count how many times the predicate remains unchanged
    
    # Process each row in the dataset using tqdm for progress visualization
    for idx, row in tqdm(dataset_df.iterrows(), total=len(dataset_df), desc="Canonicalizing relations"):
        relation_definition = row['relation_definition']
        original_sentence = row['sentence']
        current_relation = row['relation']
        
        # Get top 5 candidate canonical predicates based on cosine similarity
        candidates = get_top_k_candidates(relation_definition, canonical_embeddings, canonical_predicates, canonical_definitions, model, k=5)
        
        # Print the original predicate and the candidate list
        print(f"\nRow {idx}:")
        print(f"Original Predicate: {current_relation}")
        print("Top 5 Candidate Canonical Predicates:")
        for cand, defi, score in candidates:
            print(f" - {cand} (Score: {score:.4f})")
        
        # Build the LLM prompt using the candidate list
        prompt = build_prompt_for_relation(original_sentence, current_relation, candidates, relation_definition)
        
        # Call the LLM to choose the best canonical predicate
        llm_response = call_llm(prompt)
        if (llm_response is None or not isinstance(llm_response, dict)
                or current_relation not in llm_response or not llm_response[current_relation]):
            print(f"LLM call failed for row {idx}. Using fallback candidate: {candidates[0][0]}")
            chosen_predicate = candidates[0][0]
        else:
            chosen_predicate = llm_response[current_relation]
        
        # If LLM returns "unchanged", keep the original predicate and increase counter.
        if isinstance(chosen_predicate, str) and chosen_predicate.lower() == "unchanged":
            print("LLM decision: 'unchanged' (no suitable candidate). Keeping original predicate.")
            chosen_predicate = current_relation
            unchanged_counter += 1
        else:
            print(f"LLM decision: Replace '{current_relation}' with canonical predicate '{chosen_predicate}'")
        
        canonicalized_predicates.append(chosen_predicate)
    
    # Add the canonical predicate as a new column in the dataset
    dataset_df['canonical_relation'] = canonicalized_predicates
    
    # Save the updated dataset to the output CSV
    dataset_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nCanonicalized dataset saved to {OUTPUT_CSV}")
    print(f"Total unchanged predicates (left as original): {unchanged_counter}")
    
    elapsed = time.time() - start_time
    print(f"Total processing time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canonicalize relations in dataset using canonical predicates and LLM refinement.")
    args = parser.parse_args()
    main()
