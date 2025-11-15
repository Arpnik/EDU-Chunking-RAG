import argparse
from com.fever.rag.chunker.fixed_char_chunker import FixedCharChunker
from com.fever.rag.chunker.fixed_token_chunker import FixedTokenChunker
from com.fever.rag.chunker.sentence_chunker import SentenceChunker
from com.fever.rag.evidence.vector_db_builder import VectorDBBuilder

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evidence Vector DB Builder"
    )
    # Model configuration
    parser.add_argument("--wiki_dir", type=str, default="../../../../dataset/wiki-pages/wiki-pages",
                        help="Directory containing Wikipedia pages")
    parser.add_argument("--chroma_host", type=str, default = "localhost")
    parser.add_argument("--chroma_port", type=int, default = 8000)
    parser.add_argument("--batch_size", type=int, default = 100,
                        help="Number of documents to process in each batch")
    parser.add_argument("--max_files", type=int, default = 5,
                        help="Maximum number of wiki files to process (for testing). Set to None to process all files.")
    parser.add_argument("--reset_chroma", type=bool, default=True,
                        help="To delete all previous entries if existing with same name is found")
    parser.add_argument("--embedding_models", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Comma seperated names of embedding model as mentioned in HuggingFace")

    return parser.parse_args()

def main():
    args = parse_args()
    builder = VectorDBBuilder(
        wiki_dir = args.wiki_dir,
        chroma_host = args.chroma_host,
        chroma_port = args.chroma_port,
        batch_size = args.batch_size,
        max_files= args.max_files
    )

    # Add embedding models
    builder.add_embedding_model(args.embedding_models)
    # builder.add_embedding_model("sentence-transformers/all-mpnet-base-v2")

    # Add chunkers
    builder.add_chunker(SentenceChunker())
    builder.add_chunker(FixedCharChunker(size=500, overlap=50))
    builder.add_chunker(FixedTokenChunker(size=128, overlap=20))
    # builder.add_chunker(EDUChunker(combine_n=3))

    #TODO: Optionally add more EDU variations

    # builder.add_chunker(EDUChunker(combine_n=4))
    # builder.add_chunker(EDUChunker(combine_n=5))

    # Build all databases
    builder.build(reset=args.reset_chroma)

if __name__ == "__main__":
    main()