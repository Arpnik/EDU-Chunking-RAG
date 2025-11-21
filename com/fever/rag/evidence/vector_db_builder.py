from typing import Optional, List, Tuple, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, OptimizersConfigDiff
from com.fever.rag.chunker.base_chunker import BaseChunker
from com.fever.rag.utils.DataHelper import get_device, VectorDBConfig
from com.fever.rag.utils.text_cleaner import TextCleaner
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
from tqdm import tqdm
import time


class VectorDBBuilder:
    """Main class for building vector databases with Qdrant."""

    def __init__(
            self,
            wiki_dir: str = "wiki",
            batch_size: int = 100,
            max_files: Optional[int] = None,
            encode_batch_size: int = 128,
            use_grpc: bool = True,
            db_config: VectorDBConfig = None
    ):
        """
        Initialize the Vector DB Builder with Qdrant.

        Args:
            wiki_dir: Directory containing Wikipedia JSONL files
            qdrant_host: Qdrant host
            qdrant_port: Qdrant port (6333 for HTTP, 6334 for gRPC)
            batch_size: Number of chunks to batch before inserting
            max_files: Limit number of files to process (None = all)
            encode_batch_size: Batch size for embedding generation
            use_grpc: Use gRPC for faster communication (recommended)
        """
        self.wiki_dir = wiki_dir
        self.db_config = db_config
        self.batch_size = batch_size
        self.max_files = max_files
        self.encode_batch_size = encode_batch_size
        self.use_grpc = use_grpc
        self.embedding_models: List[str] = []
        self.chunkers: List[BaseChunker] = []
        self.device = get_device()

        # Performance tracking
        self.timing_stats = {
            'embed_time': 0.0,
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

        try:
            chunks_with_ids = chunker.chunk(cleaned_text=full_text, annotated_lines=article.get('lines', ''),
                                            tokenizer=embedding_model.tokenizer if hasattr(embedding_model, 'tokenizer') else None)
        except Exception as e:
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
            chunker_name: str = ""
    ) -> int:
        """Insert a batch of chunks into Qdrant with performance tracking."""
        if not chunks_batch:
            return start_id

        batch_size = len(chunks_batch)
        texts = [chunk[0] for chunk in chunks_batch]
        metadatas = [chunk[1] for chunk in chunks_batch]

        # Time embedding generation
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

        # Prepare points for Qdrant
        points = []
        for i, (embedding, metadata) in enumerate(zip(embeddings, metadatas)):
            point = PointStruct(
                id=start_id + i,
                vector=embedding.tolist(),
                payload={
                    **metadata,
                    "text": texts[i],  # Store the actual text in payload
                    "embedding_model": embedding_model_name,  # Store model info
                    "chunking_method": chunker_name  # Store chunker info
                }
            )
            points.append(point)

        # Time database insert
        t_insert = time.time()
        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=False  # Async insert for better performance
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
            'insert_time': 0.0,
            'process_time': 0.0,
            'total_batches': 0,
            'insert_times': [],
            'collection_sizes': []
        }

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
                                embedding_model_name, chunker.name
                            )
                            batch = []

                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        continue

        # Insert remaining
        if batch:
            current_id = self._batch_insert(
                client, collection_name, batch, embedding_model, current_id,
                embedding_model_name, chunker.name
            )

        # Wait for all async operations to complete
        print(f"\n    Waiting for final inserts to complete...")
        time.sleep(2)  # Give time for async operations

        total_time = time.time() - t_start_all

        # Analyze performance
        self._print_performance_analysis(total_time, total_chunks)

        return total_articles, total_chunks, cleaning_issues

    def _print_performance_analysis(self, total_time: float, total_chunks: int):
        """Print detailed performance analysis."""
        print(f"\n    Performance Breakdown:")
        print(f"      Total time: {total_time:.2f}s")
        print(f"      Embedding: {self.timing_stats['embed_time']:.2f}s "
              f"({self.timing_stats['embed_time'] / total_time * 100:.1f}%)")
        print(f"      DB Insert: {self.timing_stats['insert_time']:.2f}s "
              f"({self.timing_stats['insert_time'] / total_time * 100:.1f}%)")
        print(f"      Processing: {self.timing_stats['process_time']:.2f}s "
              f"({self.timing_stats['process_time'] / total_time * 100:.1f}%)")

        overhead = total_time - self.timing_stats['embed_time'] - \
                   self.timing_stats['insert_time'] - self.timing_stats['process_time']
        print(f"      Overhead: {overhead:.2f}s ({overhead / total_time * 100:.1f}%)")

        print(f"\n    Batch Statistics:")
        print(f"      Total batches: {self.timing_stats['total_batches']}")
        print(f"      Avg chunks/batch: {total_chunks / max(self.timing_stats['total_batches'], 1):.1f}")

        if self.timing_stats['insert_times']:
            avg_insert = sum(self.timing_stats['insert_times']) / len(self.timing_stats['insert_times'])
            print(f"      Avg insert time/batch: {avg_insert:.3f}s")

            # Analyze insert time degradation
            if len(self.timing_stats['insert_times']) >= 10:
                first_10_avg = sum(self.timing_stats['insert_times'][:10]) / 10
                last_10_avg = sum(self.timing_stats['insert_times'][-10:]) / 10
                slowdown_pct = ((last_10_avg - first_10_avg) / first_10_avg * 100) if first_10_avg > 0 else 0

                print(f"\n    Insert Performance Degradation:")
                print(f"      First 10 batches avg: {first_10_avg:.3f}s")
                print(f"      Last 10 batches avg: {last_10_avg:.3f}s")
                print(f"      Slowdown: {slowdown_pct:+.1f}%")

                if slowdown_pct > 50:
                    print(f"      ⚠️  WARNING: Significant insert slowdown detected!")
                elif slowdown_pct < 10:
                    print(f"      ✓ Excellent: Minimal performance degradation")

    def build(self, reset: bool = True):
        """Build all vector databases."""
        print("=" * 70)
        print("QDRANT VECTOR DATABASE BUILDER")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Wiki directory: {self.wiki_dir}")
        print(f"  Qdrant: {self.db_config.host}:{self.db_config.port}")
        print(f"  Protocol: {'gRPC' if self.use_grpc else 'HTTP'}")
        print(f"  Embedding models: {self.embedding_models}")
        print(f"  Chunking methods: {self.chunkers}")
        print(f"  Total collections: {len(self.embedding_models) * len(self.chunkers)}")
        print(f"  Document batch size: {self.batch_size}")
        print(f"  Encoding batch size: {self.encode_batch_size}")
        print(f"  Max files: {self.max_files or 'All'}")
        print(f"  Device: {self.device}")

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

            print(f"  Connecting to Qdrant...")
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

                # Create collection with optimized settings
                # Note: Collection metadata stored in payload schema, not collection level
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    ),
                    optimizers_config=OptimizersConfigDiff(
                        indexing_threshold=20000,  # Start indexing after 20k points
                        memmap_threshold=50000  # Use memory-mapped storage for large collections
                    )
                )

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

        print("\n" + "=" * 70)
        print("BUILD COMPLETE!")
        print("=" * 70)

        client = self.db_config.connect_to_qdrant()
        all_collections = client.get_collections().collections

        print(f"\nAll Collections ({len(all_collections)}):")
        for collection in sorted(all_collections, key=lambda x: x.name):
            info = client.get_collection(collection.name)
            print(f"  {collection.name:40s}: {info.points_count:,} documents")