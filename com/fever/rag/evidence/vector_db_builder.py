from typing import Optional, List, Tuple, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, OptimizersConfigDiff,
    SparseVectorParams, SparseIndexParams, SparseVector
)
from com.fever.rag.chunker.base_chunker import BaseChunker
from com.fever.rag.utils.data_helper import get_device, VectorDBConfig
from com.fever.rag.utils.text_cleaner import TextCleaner
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
from tqdm import tqdm
import time
import math
from collections import Counter


class SimpleBM25Encoder:
    """Lightweight BM25 encoder for creating sparse vectors."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vocab = {}
        self.idf = {}
        self.doc_count = 0
        self.total_doc_len = 0
        self.doc_freqs = {}

    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenization."""
        return [t.lower() for t in text.split() if len(t) >= 2 and len(t) <= 20]

    def update_statistics(self, texts: List[str]):
        """Update BM25 statistics with new batch of documents."""
        for text in texts:
            tokens = self._tokenize(text)
            self.doc_count += 1
            self.total_doc_len += len(tokens)

            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

        # Recalculate IDF values
        self.idf.clear()
        for token, doc_freq in self.doc_freqs.items():
            idf = math.log((self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
            self.idf[token] = idf

    @property
    def avg_doc_len(self):
        return self.total_doc_len / self.doc_count if self.doc_count > 0 else 1.0

    def encode(self, text: str) -> SparseVector:
        """Encode text into BM25 sparse vector."""
        tokens = self._tokenize(text)
        doc_len = len(tokens)
        term_freqs = Counter(tokens)

        indices = []
        values = []

        for token, tf in term_freqs.items():
            if token in self.vocab:
                idx = self.vocab[token]
                idf = self.idf.get(token, 0.0)

                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                score = idf * (numerator / denominator)

                indices.append(idx)
                values.append(score)

        return SparseVector(indices=indices, values=values)

    def save(self, filepath: str):
        """Save BM25 encoder to file."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str):
        """Load BM25 encoder from file."""
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class VectorDBBuilder:
    """Main class for building vector databases with Qdrant (with BM25 hybrid support)."""

    def __init__(
            self,
            wiki_dir: str = "wiki",
            batch_size: int = 100,
            max_files: Optional[int] = None,
            encode_batch_size: int = 128,
            db_config: VectorDBConfig = None,
            shared_client: Optional[QdrantClient] = None,
            use_hybrid: bool = True
    ):
        """
        Initialize the Vector DB Builder with Qdrant.

        Args:
            wiki_dir: Directory containing Wikipedia JSONL files
            batch_size: Number of chunks to batch before inserting
            max_files: Limit number of files to process (None = all)
            encode_batch_size: Batch size for embedding generation
            db_config: Vector database configuration
            shared_client: Optional shared Qdrant client
            use_hybrid: Whether to create BM25 sparse vectors for hybrid retrieval
        """
        self.wiki_dir = wiki_dir
        self.db_config = db_config
        self.batch_size = batch_size
        self.max_files = max_files
        self.encode_batch_size = encode_batch_size
        self.embedding_models: List[str] = []
        self.chunkers: List[BaseChunker] = []

        self.device = get_device()
        print(f"Using device: {self.device}")

        self.shared_client = shared_client
        self.use_hybrid = use_hybrid
        self.bm25_encoders: Dict[str, SimpleBM25Encoder] = {}

        # Performance tracking
        self.timing_stats = {
            'embed_time': 0.0,
            'sparse_time': 0.0,
            'insert_time': 0.0,
            'process_time': 0.0,
            'total_batches': 0,
            'insert_times': [],
            'collection_sizes': []
        }

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
        model_short = embedding_model.split('/')[-1].split('-')[0].lower()
        if 'minilm' in embedding_model.lower():
            model_short = 'minilm'
        elif 'mpnet' in embedding_model.lower():
            model_short = 'mpnet'
        elif 'multi-qa' in embedding_model.lower():
            model_short = 'multiqa'

        return f"{model_short}_{chunker.name}_chunks"

    def _process_article(
            self,
            article: Dict,
            chunker: BaseChunker,
            embedding_model: SentenceTransformer
    ) -> List[Tuple[str, Dict]]:
        """Process one article with a specific chunker."""
        article_id = article['id']
        full_text = TextCleaner.clean(article.get('text', ''))

        if not full_text:
            return []

        annotated_lines = article.get('lines', '')

        try:
            chunks_with_ids = chunker.chunk(
                cleaned_text=full_text,
                annotated_lines=annotated_lines,
                tokenizer=embedding_model.tokenizer if hasattr(embedding_model, 'tokenizer') else None
            )
        except Exception as e:
            print(f"ERROR processing article {article_id}: {e}")
            return []

        results = [
            (chunk_text, chunker.get_metadata(article_id, i, chunk_text, sentence_ids=sentence_ids))
            for i, (chunk_text, sentence_ids) in enumerate(chunks_with_ids)
        ]

        return results

    def _batch_insert(
            self,
            client: QdrantClient,
            collection_name: str,
            chunks_batch: List[Tuple[str, Dict]],
            embedding_model: SentenceTransformer,
            start_id: int,
            embedding_model_name: str = "",
            chunker_name: str = "",
            bm25_encoder: Optional[SimpleBM25Encoder] = None
    ) -> int:
        """Insert a batch of chunks into Qdrant with both dense and sparse vectors."""
        if not chunks_batch:
            return start_id

        batch_size = len(chunks_batch)
        texts = [chunk[0] for chunk in chunks_batch]
        metadatas = [chunk[1] for chunk in chunks_batch]

        # Update BM25 statistics with this batch (if hybrid mode)
        if self.use_hybrid and bm25_encoder is not None:
            bm25_encoder.update_statistics(texts)

        # Time dense embedding generation
        t_embed = time.time()
        embeddings = embedding_model.encode(
            texts,
            show_progress_bar=False,
            device=self.device,
            batch_size=self.encode_batch_size,
            convert_to_numpy=True,
        )
        embed_duration = time.time() - t_embed
        self.timing_stats['embed_time'] += embed_duration

        # Time sparse vector generation
        sparse_vectors = None
        if self.use_hybrid and bm25_encoder is not None:
            t_sparse = time.time()
            sparse_vectors = [bm25_encoder.encode(text) for text in texts]
            sparse_duration = time.time() - t_sparse
            self.timing_stats['sparse_time'] += sparse_duration

        # Prepare points for Qdrant
        points = []
        for i, (embedding, metadata) in enumerate(zip(embeddings, metadatas)):
            # Ensure embedding is a list
            if hasattr(embedding, 'tolist'):
                dense_vec = embedding.tolist()
            else:
                dense_vec = list(embedding)

            # Build vector dict with BOTH dense and sparse vectors
            if self.use_hybrid and sparse_vectors:
                vector_dict = {
                    "dense": dense_vec,
                    "sparse": sparse_vectors[i]  # ← CRITICAL: Store sparse vector!
                }
            else:
                vector_dict = {"dense": dense_vec}

            point = PointStruct(
                id=start_id + i,
                vector=vector_dict,
                payload={
                    **metadata,
                    "text": texts[i],
                    "embedding_model": embedding_model_name,
                    "chunking_method": chunker_name
                }
            )
            points.append(point)

        # Time database insert
        t_insert = time.time()
        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=False
        )
        insert_duration = time.time() - t_insert

        # Track performance
        self.timing_stats['insert_time'] += insert_duration
        self.timing_stats['insert_times'].append(insert_duration)
        self.timing_stats['total_batches'] += 1

        # Get collection size (sampling to avoid overhead)
        if self.timing_stats['total_batches'] % 10 == 0:
            info = client.get_collection(collection_name)
            self.timing_stats['collection_sizes'].append(info.points_count)

        # Log slow inserts
        if insert_duration > 1.0:
            print(f"\n      [SLOW INSERT] Batch {self.timing_stats['total_batches']}: "
                  f"{insert_duration:.2f}s for {batch_size} chunks")

        return start_id + batch_size

    def _count_lines_in_file(self, file_path: Path) -> int:
        """Count number of lines in a file efficiently."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)

    def _process_files_for_config(
            self,
            embedding_model: SentenceTransformer,
            embedding_model_name: str,
            chunker: BaseChunker,
            client: QdrantClient,
            collection_name: str,
            wiki_files: List[Path]
    ):
        """Process Wikipedia files for one embedding model + chunker combination."""
        batch = []
        total_articles = 0
        total_chunks = 0
        cleaning_issues = 0
        current_id = 0

        # Reset timing stats
        self.timing_stats = {
            'embed_time': 0.0,
            'sparse_time': 0.0,
            'insert_time': 0.0,
            'process_time': 0.0,
            'total_batches': 0,
            'insert_times': [],
            'collection_sizes': []
        }

        # Initialize BM25 encoder for this collection (if hybrid mode)
        bm25_encoder = None
        if self.use_hybrid:
            bm25_encoder = SimpleBM25Encoder()
            print(f"    Initialized BM25 encoder for {collection_name}")

        t_start_all = time.time()

        for file_path in tqdm(wiki_files, desc="    Files", position=0, leave=True):
            num_lines = self._count_lines_in_file(file_path)

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, total=num_lines, desc=f"      {file_path.name}", position=1, leave=False):
                    try:
                        t_proc = time.time()
                        article = json.loads(line.strip())
                        total_articles += 1

                        chunks = self._process_article(article, chunker, embedding_model)
                        cleaning_issues += sum(1 for _, meta in chunks if not meta.get('cleaned', True))

                        batch.extend(chunks)
                        total_chunks += len(chunks)

                        self.timing_stats['process_time'] += time.time() - t_proc

                        if len(batch) >= self.batch_size:
                            current_id = self._batch_insert(
                                client, collection_name, batch, embedding_model, current_id,
                                embedding_model_name, chunker.name, bm25_encoder
                            )
                            batch = []

                    except json.JSONDecodeError:
                        continue
                    except Exception:
                        continue

        # Insert remaining
        if batch:
            current_id = self._batch_insert(
                client, collection_name, batch, embedding_model, current_id,
                embedding_model_name, chunker.name, bm25_encoder
            )

        # Save BM25 encoder for retrieval
        if self.use_hybrid and bm25_encoder is not None:
            bm25_path = f"bm25_{collection_name}.pkl"
            bm25_encoder.save(bm25_path)
            print(f"\n    ✓ BM25 encoder saved to {bm25_path}")
            print(f"    Vocabulary size: {len(bm25_encoder.vocab):,} tokens")
            print(f"    Total documents: {bm25_encoder.doc_count:,}")
            self.bm25_encoders[collection_name] = bm25_encoder

        # Wait for all async operations to complete
        print(f"\n    Waiting for final inserts to complete...")
        time.sleep(2)

        total_time = time.time() - t_start_all

        # Analyze performance
        self._print_performance_analysis(total_time, total_chunks)

        return total_articles, total_chunks, cleaning_issues

    def _print_performance_analysis(self, total_time: float, total_chunks: int):
        """Print detailed performance analysis."""
        print(f"\n    Performance Breakdown:")
        print(f"      Total time: {total_time:.2f}s")
        print(f"      Dense Embedding ({self.device}): {self.timing_stats['embed_time']:.2f}s "
              f"({self.timing_stats['embed_time'] / total_time * 100:.1f}%)")

        if self.use_hybrid:
            print(f"      Sparse BM25 (CPU): {self.timing_stats['sparse_time']:.2f}s "
                  f"({self.timing_stats['sparse_time'] / total_time * 100:.1f}%)")

        print(f"      DB Insert: {self.timing_stats['insert_time']:.2f}s "
              f"({self.timing_stats['insert_time'] / total_time * 100:.1f}%)")
        print(f"      Processing: {self.timing_stats['process_time']:.2f}s "
              f"({self.timing_stats['process_time'] / total_time * 100:.1f}%)")

        overhead = total_time - self.timing_stats['embed_time'] - \
                   self.timing_stats.get('sparse_time', 0) - \
                   self.timing_stats['insert_time'] - self.timing_stats['process_time']
        print(f"      Overhead: {overhead:.2f}s ({overhead / total_time * 100:.1f}%)")

        print(f"\n    Batch Statistics:")
        print(f"      Total batches: {self.timing_stats['total_batches']}")
        print(f"      Avg chunks/batch: {total_chunks / max(self.timing_stats['total_batches'], 1):.1f}")

        if self.timing_stats['insert_times']:
            avg_insert = sum(self.timing_stats['insert_times']) / len(self.timing_stats['insert_times'])
            print(f"      Avg insert time/batch: {avg_insert:.3f}s")

    def build(self, reset: bool = True):
        """Build all vector databases."""
        print("=" * 70)
        print("QDRANT VECTOR DATABASE BUILDER" + (" (HYBRID MODE - BM25)" if self.use_hybrid else ""))
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Wiki directory: {self.wiki_dir}")
        print(f"  Qdrant: {self.db_config.host}:{self.db_config.port}")
        print(f"  Protocol: {'gRPC' if self.db_config.use_grpc else 'HTTP'}")
        print(f"  Embedding models: {self.embedding_models}")
        print(f"  Chunking methods: {self.chunkers}")
        print(f"  Total collections: {len(self.embedding_models) * len(self.chunkers)}")
        print(f"  Document batch size: {self.batch_size}")
        print(f"  Encoding batch size: {self.encode_batch_size}")
        print(f"  Max files: {self.max_files or 'All'}")
        print(f"  Device: {self.device}")
        print(f"  Hybrid retrieval: {'ENABLED (BM25 sparse vectors)' if self.use_hybrid else 'DISABLED'}")

        try:
            wiki_path = Path(self.wiki_dir)
            if not wiki_path.exists() or not wiki_path.is_dir():
                raise ValueError(f"Wiki directory does not exist: {self.wiki_dir}")
            wiki_files = sorted(wiki_path.glob("*.jsonl"))
            if self.max_files:
                wiki_files = wiki_files[:self.max_files]
            print(f"\nWill process {len(wiki_files)} wiki files")
        except Exception as e:
            raise ValueError(f"Error accessing wiki directory: {e}")

        for embedding_model_name in self.embedding_models:
            print("\n" + "=" * 70)
            print(f"PROCESSING: {embedding_model_name}")
            print("=" * 70)

            print(f"  Loading embedding model...")
            embedding_model = SentenceTransformer(embedding_model_name, device=self.device)
            vector_size = embedding_model.get_sentence_embedding_dimension()

            print(f"  ✓ Model loaded on device: {self.device}")

            print(f"  Connecting to Qdrant...")
            if self.shared_client is not None:
                client = self.shared_client
            else:
                client = self.db_config.connect_to_qdrant()

            for chunker in self.chunkers:
                collection_name = self._get_collection_name(embedding_model_name, chunker)

                print(f"\n  [{chunker.name}] Creating collection: {collection_name}")

                if reset:
                    try:
                        client.delete_collection(collection_name=collection_name)
                        print(f"    Deleted existing collection")
                    except:
                        pass

                # Create collection with sparse vector support
                if self.use_hybrid:
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config={
                            "dense": VectorParams(
                                size=vector_size,
                                distance=Distance.COSINE
                            )
                        },
                        sparse_vectors_config={
                            "sparse": SparseVectorParams(
                                index=SparseIndexParams(
                                    on_disk=False
                                )
                            )
                        },
                        optimizers_config=OptimizersConfigDiff(
                            indexing_threshold=20000,
                            memmap_threshold=50000
                        )
                    )
                    print(f"    ✓ Collection created with HYBRID support (dense + sparse BM25 vectors)")
                else:
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=vector_size,
                            distance=Distance.COSINE
                        ),
                        optimizers_config=OptimizersConfigDiff(
                            indexing_threshold=20000,
                            memmap_threshold=50000
                        )
                    )
                    print(f"    ✓ Collection created (dense vectors only)")

                print(f"    Collection metadata: model={embedding_model_name}, chunker={chunker.name}")

                total_articles, total_chunks, cleaning_issues = self._process_files_for_config(
                    embedding_model,
                    embedding_model_name,
                    chunker,
                    client,
                    collection_name,
                    wiki_files
                )

                # Get final count
                collection_info = client.get_collection(collection_name)

                print(f"\n    ✓ Complete: {total_chunks:,} chunks from {total_articles:,} articles")
                print(f"    Cleaning issues: {cleaning_issues:,}")
                print(f"    Final count: {collection_info.points_count:,} documents")

                print(f"\n  Collecting statistics for {chunker.name}...")
                if chunker.stats is not None:
                    chunker.stats.print_stats()
                    stats_filename = f"statistics_{chunker.name}_{embedding_model_name.split('/')[-1]}.json"
                    chunker.stats.save_to_file(stats_filename)
                if hasattr(chunker, "boundary_count") and chunker.boundary_count is not None:
                    print("total boundaries found: ", chunker.boundary_count)

        print("\n" + "=" * 70)
        print("BUILD COMPLETE!")
        print("=" * 70)

        client = self.db_config.connect_to_qdrant()
        all_collections = client.get_collections().collections

        print(f"\nAll Collections ({len(all_collections)}):")
        for collection in sorted(all_collections, key=lambda x: x.name):
            info = client.get_collection(collection.name)
            print(f"  {collection.name:40s}: {info.points_count:,} documents")