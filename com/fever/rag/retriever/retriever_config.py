from typing import Optional, List, Dict
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, QueryRequest
from sentence_transformers import SentenceTransformer
import time
from com.fever.rag.utils.data_helper import VectorDBConfig, RetrievalConfig, RetrievalResult, RetrievalStrategy, \
    get_device


class SimpleBM25Encoder:
    """Lightweight BM25 encoder for query encoding."""

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

    @property
    def avg_doc_len(self):
        return self.total_doc_len / self.doc_count if self.doc_count > 0 else 1.0

    def encode(self, text: str):
        """Encode text into BM25 sparse vector."""
        from collections import Counter
        from qdrant_client.models import SparseVector
        import math

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

    @staticmethod
    def load(filepath: str):
        """Load BM25 encoder from file."""
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class VectorDBRetriever:
    """Retrieves chunks from Qdrant vector database with hybrid search support."""

    def __init__(
            self,
            db_config: VectorDBConfig,
            shared_client: Optional[QdrantClient] = None,
            use_hybrid: bool = True
    ):
        """
        Initialize the retriever with hybrid search support.

        Args:
            db_config: Vector database configuration
            shared_client: Optional shared Qdrant client
            use_hybrid: Whether to use hybrid retrieval (dense + BM25 sparse)
        """
        self.db_config = db_config
        self.device = get_device()
        self._model_cache: Dict[str, SentenceTransformer] = {}
        self._bm25_cache: Dict[str, SimpleBM25Encoder] = {}
        self.shared_client = shared_client
        self.use_hybrid = use_hybrid

    def _get_embedding_model(self, model_name: str) -> SentenceTransformer:
        """Get or load an embedding model (with caching)."""
        if model_name not in self._model_cache:
            print(f"Loading embedding model: {model_name}")
            self._model_cache[model_name] = SentenceTransformer(model_name, device=self.device)
        return self._model_cache[model_name]

    def _get_bm25_encoder(self, collection_name: str) -> Optional[SimpleBM25Encoder]:
        """Load BM25 encoder for a collection (with caching)."""
        if collection_name not in self._bm25_cache:
            bm25_path = f"bm25_{collection_name}.pkl"

            # Try both current directory and parent directories
            search_paths = [
                Path(bm25_path),
                Path('.') / bm25_path,
                Path('..') / bm25_path,
            ]

            loaded = False
            for path in search_paths:
                if path.exists():
                    try:
                        self._bm25_cache[collection_name] = SimpleBM25Encoder.load(str(path))
                        print(f"✓ Loaded BM25 encoder from {path}")
                        print(f"  Vocabulary size: {len(self._bm25_cache[collection_name].vocab):,} tokens")
                        loaded = True
                        break
                    except Exception as e:
                        print(f"⚠️  Failed to load BM25 from {path}: {e}")

            if not loaded:
                print(f"⚠️  BM25 encoder not found for collection: {collection_name}")
                print(f"   Searched paths: {[str(p) for p in search_paths]}")
                print(f"   Falling back to dense-only retrieval for this collection")
                return None

        return self._bm25_cache.get(collection_name)

    def retrieve(
            self,
            claim: str,
            collection_name: str,
            embedding_model_name: str,
            config: RetrievalConfig,
            claim_id: Optional[int] = None
    ) -> RetrievalResult:
        """
        Retrieve chunks for a single claim using hybrid search.

        Args:
            claim: The claim text to retrieve evidence for
            collection_name: Name of the Qdrant collection
            embedding_model_name: Name of the embedding model to use
            config: Retrieval configuration
            claim_id: Optional claim ID for tracking

        Returns:
            RetrievalResult containing all retrieved chunks
        """
        if self.shared_client is not None:
            client = self.shared_client
        else:
            client = self.db_config.connect_to_qdrant()

        embedding_model = self._get_embedding_model(embedding_model_name)

        t_start = time.time()

        # Generate dense embedding
        claim_embedding = embedding_model.encode(
            claim,
            show_progress_bar=False,
            device=self.device,
            convert_to_numpy=True
        )

        # Try to get BM25 encoder for hybrid search
        bm25_encoder = None
        if self.use_hybrid:
            bm25_encoder = self._get_bm25_encoder(collection_name)

        # Perform retrieval
        if self.use_hybrid and bm25_encoder is not None:
            # HYBRID RETRIEVAL (dense + sparse BM25)
            results = self._hybrid_search(
                client=client,
                collection_name=collection_name,
                claim=claim,
                dense_vector=claim_embedding.tolist(),
                bm25_encoder=bm25_encoder,
                config=config
            )
        else:
            # DENSE-ONLY RETRIEVAL
            results = self._dense_only_search(
                client=client,
                collection_name=collection_name,
                dense_vector=claim_embedding.tolist(),
                config=config
            )

        retrieval_time = time.time() - t_start

        return RetrievalResult(
            claim=claim,
            claim_id=claim_id,
            collection_name=collection_name,
            embedding_model_name=embedding_model_name,
            retrieval_config=config,
            chunks=results,
            retrieval_time=retrieval_time
        )

    def _dense_only_search(
            self,
            client: QdrantClient,
            collection_name: str,
            dense_vector: List[float],
            config: RetrievalConfig
    ) -> List:
        """
        Perform dense-only search.

        Args:
            client: Qdrant client
            collection_name: Collection to search
            dense_vector: Dense embedding vector
            config: Retrieval configuration

        Returns:
            List of search results
        """
        if config.strategy == RetrievalStrategy.TOP_K:
            results = client.search(
                collection_name=collection_name,
                query_vector=("dense", dense_vector),
                limit=config.k
            )
        else:  # THRESHOLD strategy
            results = client.search(
                collection_name=collection_name,
                query_vector=("dense", dense_vector),
                limit=100,
                score_threshold=config.threshold
            )

        return results

    def _hybrid_search(
            self,
            client: QdrantClient,
            collection_name: str,
            claim: str,
            dense_vector: List[float],
            bm25_encoder: SimpleBM25Encoder,
            config: RetrievalConfig
    ) -> List:
        """
        Perform hybrid search using dense + BM25 sparse vectors with RRF fusion.

        Args:
            client: Qdrant client
            collection_name: Collection to search
            claim: Query text
            dense_vector: Dense embedding vector
            bm25_encoder: BM25 encoder for sparse vectors
            config: Retrieval configuration

        Returns:
            List of fused search results
        """
        # Determine retrieval limit
        limit = config.k if config.strategy == RetrievalStrategy.TOP_K else 100

        # Generate sparse vector from claim using BM25 encoder
        sparse_vector = bm25_encoder.encode(claim)

        try:
            # Try using Qdrant's native query API with prefetch + fusion
            from qdrant_client.models import FusionQuery

            response = client.query_points(
                collection_name=collection_name,
                prefetch=[
                    # Prefetch from dense vectors
                    Prefetch(
                        query=dense_vector,
                        using="dense",
                        limit=limit * 3  # Get more candidates for better fusion
                    ),
                    # Prefetch from sparse vectors
                    Prefetch(
                        query=sparse_vector,
                        using="sparse",
                        limit=limit * 3
                    )
                ],
                query=FusionQuery(fusion="rrf"),  # Reciprocal Rank Fusion
                limit=limit
            )
            results = response.points

        except (ImportError, AttributeError, Exception) as e:
            print(f"⚠️  Native fusion API failed ({e}), using manual RRF fusion...")

            try:
                # Alternative: Manual RRF fusion
                # Get results from both searches
                dense_results = client.search(
                    collection_name=collection_name,
                    query_vector=("dense", dense_vector),
                    limit=limit * 2
                )

                # Search using sparse BM25 vector
                sparse_results = client.search(
                    collection_name=collection_name,
                    query_vector=("sparse", sparse_vector),
                    limit=limit * 2
                )

                # Manual RRF fusion
                results = self._manual_rrf_fusion(dense_results, sparse_results, limit)

            except Exception as e2:
                print(f"⚠️  Sparse search also failed ({e2}), falling back to dense-only")
                results = self._dense_only_search(
                    client=client,
                    collection_name=collection_name,
                    dense_vector=dense_vector,
                    config=config
                )

        # Apply threshold filter if using THRESHOLD strategy
        if config.strategy == RetrievalStrategy.THRESHOLD:
            results = [r for r in results if r.score >= config.threshold]

        return results

    def _manual_rrf_fusion(self, dense_results: List, sparse_results: List, limit: int, k: int = 60) -> List:
        """
        Manually perform Reciprocal Rank Fusion on two result sets.

        RRF formula: score(doc) = sum over all rankings of (1 / (k + rank))
        where k=60 is a constant and rank is the position in each ranking (1-indexed).

        Args:
            dense_results: Results from dense vector search
            sparse_results: Results from sparse BM25 search
            limit: Number of results to return
            k: RRF constant (default: 60)

        Returns:
            Fused results sorted by RRF score
        """
        scores = {}

        # Add dense scores (rank starts at 1)
        for rank, result in enumerate(dense_results, start=1):
            point_id = result.id
            scores[point_id] = scores.get(point_id, 0) + 1 / (k + rank)

        # Add sparse scores (rank starts at 1)
        for rank, result in enumerate(sparse_results, start=1):
            point_id = result.id
            scores[point_id] = scores.get(point_id, 0) + 1 / (k + rank)

        # Get all unique results (preserve result objects)
        all_results = {r.id: r for r in dense_results}
        all_results.update({r.id: r for r in sparse_results})

        # Sort by RRF score (descending)
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:limit]

        # Return results with RRF scores
        fused_results = []
        for point_id in sorted_ids:
            result = all_results[point_id]
            # Replace original score with RRF score
            result.score = scores[point_id]
            fused_results.append(result)

        return fused_results

    def set_hybrid_mode(self, enabled: bool):
        """
        Enable or disable hybrid retrieval.

        Args:
            enabled: True to enable hybrid (dense + BM25), False for dense-only
        """
        self.use_hybrid = enabled
        mode = "HYBRID (dense + BM25 sparse)" if enabled else "DENSE-ONLY"
        print(f"Retrieval mode set to: {mode}")

    def clear_cache(self):
        """Clear model and BM25 encoder caches to free memory."""
        self._model_cache.clear()
        self._bm25_cache.clear()
        print("✓ Cleared model and BM25 encoder caches")

    def get_cache_info(self) -> Dict:
        """Get information about cached models."""
        return {
            'embedding_models_cached': list(self._model_cache.keys()),
            'bm25_encoders_cached': list(self._bm25_cache.keys()),
            'hybrid_mode': 'ENABLED (dense + BM25)' if self.use_hybrid else 'DISABLED'
        }