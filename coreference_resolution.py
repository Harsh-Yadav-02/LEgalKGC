import spacy
import neuralcoref
import os
import argparse

def process_and_get_metadata(text, nlp):
    """
    Process the given text with neuralcoref and collect metadata information.
    Returns a tuple of (resolved_text, metadata_info) as strings.
    Note: The metadata does not include the original or resolved text.
    """
    doc = nlp(text)
    lines = []

    # Document-level coreference info
    lines.append("doc._.has_coref: " + str(doc._.has_coref))
    lines.append("")
    lines.append("doc._.coref_clusters:")
    for i, cluster in enumerate(doc._.coref_clusters, start=1):
        lines.append(f"  Cluster {i}:")
        lines.append("    Main mention: " + str(cluster.main))
        for mention in cluster.mentions:
            lines.append("    Mention: " + mention.text)
    lines.append("")
    lines.append("doc._.coref_scores:")
    lines.append(str(doc._.coref_scores))
    lines.append("")

    # Token-level coreference info
    lines.append("Token-level coreference info:")
    for token in doc:
        if token._.in_coref:
            lines.append(f"  Token '{token.text}':")
            for cluster in token._.coref_clusters:
                lines.append("    In cluster with main mention: " + str(cluster.main))
    lines.append("")

    # Span-level info for a sample span (last 3 tokens)
    span = doc[-3:]
    lines.append("Span-level info for span: " + span.text)
    lines.append("  span._.is_coref: " + str(span._.is_coref))
    if span._.is_coref:
        lines.append("  span._.coref_cluster.main: " + str(span._.coref_cluster.main))
        lines.append("  span._.coref_scores: " + str(span._.coref_scores))
    lines.append("")
    lines.append("=" * 60)

    metadata_info = "\n".join(lines)
    return doc._.coref_resolved, metadata_info

def main(input_file):
    # Load spaCy's English model and add neuralcoref to the pipeline
    nlp = spacy.load('en')  # Ensure you have the 'en' model installed
    neuralcoref.add_to_pipe(nlp)
    
    # Read the input text file
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Process the text and collect metadata (without original/resolved text in metadata)
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
    args = parser.parse_args()
    main(args.input_file)


# python coreference_resolution.py datasets\processed.txt 