from typing import Optional, List, Tuple, Dict
import chromadb
from chromadb import Settings
from com.fever.rag.chunker.base_chunker import BaseChunker
from com.fever.rag.utils.text_cleaner import TextCleaner
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
from tqdm import tqdm
import torch


class VectorDBBuilder:
    """Main class for building vector databases with multiple configurations."""

    def __init__(
            self,
            wiki_dir: str = "wiki-pages",
            chroma_host: str = "localhost",
            chroma_port: int = 8000,
            batch_size: int = 100,
            max_files: Optional[int] = None
    ):
        """
        Initialize the Vector DB Builder.

        Args:
            wiki_dir: Directory containing Wikipedia JSONL files
            chroma_host: ChromaDB host
            chroma_port: ChromaDB port
            batch_size: Number of chunks to batch before inserting
            max_files: Limit number of files to process (None = all)
        """
        self.wiki_dir = wiki_dir
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port
        self.batch_size = batch_size
        self.max_files = max_files
        self.embedding_models: List[str] = []
        self.chunkers: List[BaseChunker] = []
        self.nlp = None
        self.device = self._get_device()

    def _get_device(self) -> str:
        """Automatically detect best available device."""
        if torch.cuda.is_available():
            device = "cuda"
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = "mps"
            print("Using Apple MPS (Metal Performance Shaders)")
        else:
            device = "cpu"
            print("Using CPU")
        return device

    def add_embedding_model(self, model_name: str):
        """Add an embedding model to process."""
        self.embedding_models.append(model_name)
        return self

    def add_chunker(self, chunker: BaseChunker):
        """Add a chunking strategy."""
        self.chunkers.append(chunker)
        return self

    def _get_collection_name(self, embedding_model: str, chunker: BaseChunker) -> str:
        """Generate collection name from model and chunker."""
        # Shorten embedding model name
        model_short = embedding_model.split('/')[-1].split('-')[0].lower()
        if 'minilm' in embedding_model.lower():
            model_short = 'minilm'
        elif 'mpnet' in embedding_model.lower():
            model_short = 'mpnet'
        elif 'multi-qa' in embedding_model.lower():
            model_short = 'multiqa'

        return f"{model_short}_{chunker.name}_chunks"

    def _connect_to_chroma(self) -> chromadb.HttpClient:
        """Connect to ChromaDB."""
        client = chromadb.HttpClient(
            host=self.chroma_host,
            port=self.chroma_port,
            settings=Settings(anonymized_telemetry=False)
        )
        client.heartbeat()
        return client

    def _parse_article_lines(self, lines_str: str) -> List[str]:
        """Parse article lines into clean sentences."""
        if not lines_str:
            return []

        sentences = []
        for line in lines_str.strip().split('\n'):
            if not line.strip():
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                sentence = TextCleaner.clean(parts[1])
                if sentence:
                    sentences.append(sentence)
        return sentences

    def _process_article(
            self,
            article: Dict,
            chunker: BaseChunker,
            embedding_model: SentenceTransformer
    ) -> List[Tuple[str, Dict]]:
        """Process one article with a specific chunker."""
        article_id = article['id']
        sentences = self._parse_article_lines(article.get('lines', ''))
        full_text = TextCleaner.clean(article.get('text', ''))

        if not sentences or not full_text:
            return []

        # Get chunks using the chunker
        try:
            chunks = chunker.chunk(
                text=full_text,
                sentences=sentences,
                tokenizer=embedding_model.tokenizer if hasattr(embedding_model, 'tokenizer') else None
            )
        except Exception as e:
            return []

        # Create chunk-metadata tuples
        results = [
            (chunk, chunker.get_metadata(article_id, i, chunk))
            for i, chunk in enumerate(chunks)
        ]

        return results

    def _batch_insert(
            self,
            collection,
            chunks_batch: List[Tuple[str, Dict]],
            embedding_model: SentenceTransformer
    ):
        """Insert a batch of chunks into ChromaDB."""
        if not chunks_batch:
            return

        texts = [chunk[0] for chunk in chunks_batch]
        metadatas = [chunk[1] for chunk in chunks_batch]

        # Generate embeddings
        embeddings = embedding_model.encode(
            texts,
            show_progress_bar=False,
            device=self.device
        )

        # Generate unique IDs
        ids = [f"{meta['article_id']}_chunk_{meta['chunk_index']}" for meta in metadatas]

        # Insert
        collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

    def _count_lines_in_file(self, file_path: Path) -> int:
        """Count number of lines in a file efficiently."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)

    def _process_files_for_config(
            self,
            embedding_model_name: str,
            embedding_model: SentenceTransformer,
            chunker: BaseChunker,
            collection,
            wiki_files: List[Path]
    ):
        """Process Wikipedia files for one embedding model + chunker combination."""
        batch = []
        total_articles = 0
        total_chunks = 0
        cleaning_issues = 0

        # Outer progress bar for files
        for file_path in tqdm(wiki_files, desc="    Files", position=0, leave=True):
            # Count lines for inner progress bar
            num_lines = self._count_lines_in_file(file_path)

            with open(file_path, 'r', encoding='utf-8') as f:
                # Inner progress bar for articles within file
                for line in tqdm(f, total=num_lines, desc=f"      {file_path.name}", position=1, leave=False):
                    try:
                        article = json.loads(line.strip())

                        total_articles += 1

                        # Process article
                        chunks = self._process_article(article, chunker, embedding_model)

                        # Track cleaning issues
                        cleaning_issues += sum(1 for _, meta in chunks if not meta.get('cleaned', True))

                        batch.extend(chunks)
                        total_chunks += len(chunks)

                        # Insert batch if full
                        if len(batch) >= self.batch_size:
                            self._batch_insert(collection, batch, embedding_model)
                            batch = []

                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        continue

        # Insert remaining
        if batch:
            self._batch_insert(collection, batch, embedding_model)

        return total_articles, total_chunks, cleaning_issues

    def build(self, reset: bool = True):
        """
        Build all vector databases.

        Args:
            reset: Whether to delete existing collections before building
        """
        print("=" * 70)
        print("HIERARCHICAL CHROMADB VECTOR DATABASE BUILDER")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Wiki directory: {self.wiki_dir}")
        print(f"  ChromaDB: {self.chroma_host}:{self.chroma_port}")
        print(f"  Embedding models: {len(self.embedding_models)}")
        print(f"  Chunking methods: {len(self.chunkers)}")
        print(f"  Total collections: {len(self.embedding_models) * len(self.chunkers)}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Max files: {self.max_files or 'All'}")
        print(f"  Device: {self.device}")

        # Get wiki files
        try:
            wiki_path = Path(self.wiki_dir)
            if not wiki_path.exists() or not wiki_path.is_dir():
                raise ValueError(f"Wiki directory does not exist: {self.wiki_dir}")
            wiki_files = sorted(wiki_path.glob("wiki-*.jsonl"))
            if self.max_files:
                wiki_files = wiki_files[:self.max_files]
            print(f"\nWill process {len(wiki_files)} wiki files")
        except Exception as e:
            raise ValueError(f"Error accessing wiki directory: {e}")

        # Process each embedding model
        for embedding_model_name in self.embedding_models:
            print("\n" + "=" * 70)
            print(f"PROCESSING: {embedding_model_name}")
            print("=" * 70)

            # Load embedding model
            print(f"  Loading embedding model...")
            embedding_model = SentenceTransformer(embedding_model_name, device=self.device)

            # Connect to ChromaDB
            print(f"  Connecting to ChromaDB...")
            client = self._connect_to_chroma()

            # Process each chunker
            for chunker in self.chunkers:
                collection_name = self._get_collection_name(embedding_model_name, chunker)

                print(f"\n  [{chunker.name}] Creating collection: {collection_name}")

                # Reset if requested
                if reset:
                    try:
                        client.delete_collection(name=collection_name)
                        print(f"    Deleted existing collection")
                    except:
                        pass

                # Create collection
                collection = client.get_or_create_collection(
                    name=collection_name,
                    metadata={
                        "embedding_model": embedding_model_name,
                        "chunking_method": chunker.name,
                        **chunker.config
                    }
                )

                # Process files
                total_articles, total_chunks, cleaning_issues = self._process_files_for_config(
                    embedding_model_name,
                    embedding_model,
                    chunker,
                    collection,
                    wiki_files
                )

                print(f"    ✓ Complete: {total_chunks:,} chunks from {total_articles:,} articles")
                print(f"    Cleaning issues: {cleaning_issues:,}")
                print(f"    Final count: {collection.count():,} documents")

        # Final summary
        print("\n" + "=" * 70)
        print("BUILD COMPLETE!")
        print("=" * 70)

        client = self._connect_to_chroma()
        all_collections = client.list_collections()

        print(f"\nAll Collections ({len(all_collections)}):")
        for collection in sorted(all_collections, key=lambda x: x.name):
            print(f"  {collection.name:40s}: {collection.count():,} documents")