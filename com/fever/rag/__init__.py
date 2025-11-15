# ============================================================================
# MAIN EXECUTION
# ============================================================================

WIKI_DIR = "wiki-pages"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000

# Multiple embedding models to experiment with
EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",  # Fast, 384 dim
    # "sentence-transformers/all-mpnet-base-v2",        # Better quality, 768 dim
    # "sentence-transformers/multi-qa-MiniLM-L6-cos-v1" # Optimized for Q&A
]

# Chunking parameters
CHUNKING_CONFIGS = {
    'sentence': {},
    'fixed_char': {'size': 500, 'overlap': 50},
    'fixed_token': {'size': 128, 'overlap': 20},
    'edu': {'combine_n': 3}  # Will test 3, 4, 5 later
}

# Processing
BATCH_SIZE = 100
MAX_FILES = None  # Set to number for testing, None for all


def main():
    """Main execution pipeline for all embedding models."""
    print("=" * 70)
    print("HIERARCHICAL CHROMADB VECTOR DATABASE BUILDER FOR FEVER")
    print("=" * 70)
    print(f"\nWill process {len(EMBEDDING_MODELS)} embedding model(s)")
    print(f"With {len(CHUNKING_CONFIGS)} chunking method(s) each")
    print(f"Total collections to create: {len(EMBEDDING_MODELS) * len(CHUNKING_CONFIGS)}")

    # Process each embedding model
    for embedding_model_name in EMBEDDING_MODELS:
        print("\n" + "=" * 70)
        print(f"STARTING: {embedding_model_name}")
        print("=" * 70)

        # Initialize components for this model
        embedding_model, nlp, client = initialize_components(embedding_model_name)

        # Create collections for this model
        collections = create_collections_for_model(
            client,
            embedding_model_name,
            reset=True  # Set to False to keep existing data
        )

        # Process Wikipedia files
        edu_segmenter = None  # TODO: Replace with actual segmenter
        process_wikipedia_for_model(
            embedding_model_name=embedding_model_name,
            embedding_model=embedding_model,
            nlp=nlp,
            collections=collections,
            edu_segmenter=edu_segmenter
        )

    # Final summary
    print("\n" + "=" * 70)
    print("ALL EMBEDDINGS PROCESSED SUCCESSFULLY!")
    print("=" * 70)

    # List all collections
    print("\nFinal Collection Summary:")
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    all_collections = client.list_collections()

    for collection in sorted(all_collections, key=lambda x: x.name):
        print(f"  {collection.name:35s}: {collection.count():,} documents")

    print(f"\nTotal collections created: {len(all_collections)}")
    print("\nYou can now use these collections for retrieval experiments!")
    print("Example collection naming: minilm_sentence_chunks, mpnet_fixed_char_chunks, etc.")


if __name__ == "__main__":
    main()