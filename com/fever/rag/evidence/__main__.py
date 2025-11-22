import argparse
from com.fever.rag.chunker.fixed_char_chunker import FixedCharChunker
from com.fever.rag.chunker.fixed_token_chunker import FixedTokenChunker
from com.fever.rag.chunker.sentence_chunker import SentenceChunker
from com.fever.rag.evidence.vector_db_builder import VectorDBBuilder
from com.fever.rag.utils.data_helper import VectorDBConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evidence Vector DB Builder with Qdrant"
    )
    # Data configuration
    parser.add_argument("--wiki_dir", type=str,
                        default="../../../../dataset/reduced_fever_data/wiki",
                        help="Directory containing Wikipedia pages")

    # Qdrant configuration
    parser.add_argument("--qdrant_host", type=str, default="localhost",
                        help="Qdrant server host")
    parser.add_argument("--qdrant_port", type=int, default=6334,
                        help="Qdrant server port (6333 for HTTP, 6334 for gRPC)")
    parser.add_argument("--use_grpc", type=bool, default=True,
                        help="Use gRPC protocol for faster communication")

    # Performance configuration
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="Number of documents to batch before inserting")
    parser.add_argument("--encode_batch_size", type=int, default=128,
                        help="Batch size for embedding generation (higher = faster but more memory)")

    # Processing configuration
    parser.add_argument("--max_files", type=int, default=200,
                        help="Maximum number of wiki files to process (for testing). Set to None for all files.")
    parser.add_argument("--reset", type=bool, default=True,
                        help="Delete existing collections with same name")

    # Model configuration
    parser.add_argument("--embedding_models", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Comma-separated names of embedding models from HuggingFace")

    return parser.parse_args()


def main():
    args = parse_args()

    print("Starting Qdrant Vector DB Builder...")
    print(f"Using {'gRPC' if args.use_grpc else 'HTTP'} protocol")

    # Initialize builder with Qdrant configuration
    db_config = VectorDBConfig(host=args.qdrant_host, port = args.qdrant_port, use_grpc=True)

    builder = VectorDBBuilder(
        wiki_dir=args.wiki_dir,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        batch_size=args.batch_size,
        encode_batch_size=args.encode_batch_size,
        max_files=args.max_files,
        use_grpc=args.use_grpc,
        db_config=db_config
    )

    # Add embedding models
    embedding_models = [m.strip() for m in args.embedding_models.split(',')]
    for model in embedding_models:
        builder.add_embedding_model(model)

    # Add chunkers (enable as needed)
    builder.add_chunker(SentenceChunker())
    # builder.add_chunker(FixedCharChunker(size=500, overlap=50))
    # builder.add_chunker(FixedTokenChunker(size=128, overlap=20))

    # Build all databases
    builder.build(reset=args.reset)

    print("\n✓ Database building complete!")


if __name__ == "__main__":
    main()