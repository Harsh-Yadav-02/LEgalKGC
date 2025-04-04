import spacy
import neuralcoref
import os
import argparse
import json

def process_and_get_metadata(text, nlp):
    """
    Process the given text with neuralcoref and collect metadata information.
    Returns a tuple of (resolved_text, metadata_info) as strings.
    The metadata now includes only the has_coref flag and the coreference clusters.
    """
    doc = nlp(text)
    lines = []

    # Document-level coreference info: has_coref and clusters
    lines.append("doc._.has_coref: " + str(doc._.has_coref))
    lines.append("")
    lines.append("doc._.coref_clusters:")
    for i, cluster in enumerate(doc._.coref_clusters, start=1):
        lines.append(f"  Cluster {i}:")
        lines.append("    Main mention: " + str(cluster.main))
        for mention in cluster.mentions:
            lines.append("    Mention: " + mention.text)
    lines.append("")

    # Token-level coreference info (without scores)
    lines.append("Token-level coreference info:")
    for token in doc:
        if token._.in_coref:
            lines.append(f"  Token '{token.text}':")
            for cluster in token._.coref_clusters:
                lines.append("    In cluster with main mention: " + str(cluster.main))
    lines.append("")

    # Span-level info for a sample span (last 3 tokens) without scores
    span = doc[-3:]
    lines.append("Span-level info for span: " + span.text)
    lines.append("  span._.is_coref: " + str(span._.is_coref))
    if span._.is_coref:
        lines.append("  span._.coref_cluster.main: " + str(span._.coref_cluster.main))
    lines.append("")
    lines.append("=" * 60)

    metadata_info = "\n".join(lines)
    return doc._.coref_resolved, metadata_info


def main(input_file, greedyness, max_dist, max_dist_match, blacklist, store_scores, conv_dict):
    # Load spaCy's English model and add neuralcoref to the pipeline
    nlp = spacy.load('en')  # Ensure you have the 'en' model installed

    # Parse conv_dict from JSON if provided as a string
    if isinstance(conv_dict, str):
        conv_dict = json.loads(conv_dict)
        
    neuralcoref.add_to_pipe(
        nlp,
        greedyness=greedyness,
        max_dist=max_dist,
        max_dist_match=max_dist_match,
        blacklist=blacklist,
        store_scores=store_scores,
        conv_dict=conv_dict
    )
    
    # Read the input text file
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Process the text and collect metadata
    resolved_text, metadata_info = process_and_get_metadata(text, nlp)
    
    # Build output filenames:
    # Resolved text file: original filename with "_cr" appended before the extension.
    # Metadata file: original filename with "_metadata" appended before the extension.
    base, ext = os.path.splitext(input_file)
    resolved_file = f"{base}_cr{ext}"
    metadata_file = f"{base}_metadata{ext}"

    # Write the resolved text to its file
    with open(resolved_file, 'w', encoding='utf-8') as f:
        f.write(resolved_text)
    
    # Write the metadata to its file
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write(metadata_info)
    
    print("Coreference-resolved text written to:", resolved_file)
    print("Coreference metadata written to:", metadata_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process coreference resolution on an input text file and save metadata (excluding original/resolved text)."
    )
    parser.add_argument("input_file", help="Path to the input text file (e.g., processed.txt)")
    parser.add_argument("--greedyness", type=float, default=0.5,
                        help="A number between 0 and 1 determining how greedy the model is about making coreference decisions (default: 0.5).")
    parser.add_argument("--max_dist", type=int, default=50,
                        help="How many mentions back to look when considering possible antecedents (default: 50).")
    parser.add_argument("--max_dist_match", type=int, default=500,
                        help="Distance to look back for a matching noun or proper noun (default: 500).")
    parser.add_argument("--blacklist", type=bool, default=True,
                        help="Should the system resolve coreferences for pronouns like 'i', 'me', 'my', 'you', 'your'? (default: True).")
    parser.add_argument("--store_scores", type=bool, default=True,
                        help="Should the system store the coreference scores? (default: True).")
    parser.add_argument("--conv_dict", type=str, default="{}",
                        help="Conversion dictionary in JSON format, e.g., '{\"Angela\": [\"woman\", \"girl\"]}'.")
    args = parser.parse_args()
    main(args.input_file, args.greedyness, args.max_dist, args.max_dist_match, args.blacklist, args.store_scores, args.conv_dict)


# python coreference_resolution.py datasets/processed.txt --greedyness 0.5 --max_dist 100  