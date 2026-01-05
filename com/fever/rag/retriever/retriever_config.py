from typing import Optional, List, Dict, Set
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import time
from com.fever.rag.utils.data_helper import VectorDBConfig, RetrievalConfig, RetrievalResult, RetrievalStrategy, \
    get_device


class VectorDBRetriever:
    """Retrieves chunks from Qdrant vector database with optional hybrid search."""

    def __init__(
            self,
            db_config: VectorDBConfig,
            shared_client: Optional[QdrantClient] = None,
            use_hybrid: bool = False  # NEW: Enable hybrid search (disabled by default for backward compatibility)
    ):
        """
        Initialize the retriever.

        Args:
            db_config: Vector database configuration
            shared_client: Optional shared Qdrant client
            use_hybrid: Whether to use hybrid retrieval (dense + BM25 sparse)
        """
        self.db_config = db_config
        self.device = get_device()
        self._model_cache: Dict[str, SentenceTransformer] = {}
        self._bm25_cache: Dict = {}  # NEW: Cache for BM25 encoders
        self.shared_client = shared_client
        self.use_hybrid = use_hybrid  # NEW

    def _get_embedding_model(self, model_name: str) -> SentenceTransformer:
        """Get or load an embedding model (with caching)."""
        if model_name not in self._model_cache:
            print(f"Loading embedding model: {model_name}")
            self._model_cache[model_name] = SentenceTransformer(model_name, device=self.device)
        return self._model_cache[model_name]

    def _get_bm25_encoder(self, collection_name: str):
        """Load BM25 encoder for a collection if available (with caching)."""
        if collection_name not in self._bm25_cache:
            bm25_path = f"bm25_{collection_name}.pkl"

            search_paths = [Path(bm25_path), Path('.') / bm25_path, Path('..') / bm25_path]

            for path in search_paths:
                if path.exists():
                    try:
                        import pickle
                        with open(path, 'rb') as f:
                            self._bm25_cache[collection_name] = pickle.load(f)
                        print(f"✓ Loaded BM25 encoder from {path}")
                        return self._bm25_cache[collection_name]
                    except Exception as e:
                        print(f"⚠️  Failed to load BM25 from {path}: {e}")

            # BM25 not found - will use dense-only
            self._bm25_cache[collection_name] = None

        return self._bm25_cache.get(collection_name)

    def _hybrid_search(self, client, collection_name, claim_embedding, claim_text, bm25_encoder, limit):
        """Perform hybrid search with manual RRF fusion."""
        try:
            # Dense search
            dense_results = client.search(
                collection_name=collection_name,
                query_vector=("dense", claim_embedding),
                limit=limit * 2
            )

            # Sparse search
            sparse_vector = bm25_encoder.encode(claim_text)
            sparse_results = client.search(
                collection_name=collection_name,
                query_vector=("sparse", sparse_vector),
                limit=limit * 2
            )

            # Manual RRF fusion (k=60)
            scores = {}
            for rank, result in enumerate(dense_results, start=1):
                scores[result.id] = scores.get(result.id, 0) + 1 / (60 + rank)

            for rank, result in enumerate(sparse_results, start=1):
                scores[result.id] = scores.get(result.id, 0) + 1 / (60 + rank)

            # Get all unique results
            all_results = {r.id: r for r in dense_results}
            all_results.update({r.id: r for r in sparse_results})

            # Sort by RRF score
            sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:limit]

            # Return results with RRF scores
            fused_results = []
            for point_id in sorted_ids:
                result = all_results[point_id]
                result.score = scores[point_id]
                fused_results.append(result)

            return fused_results

        except Exception as e:
            print(f"⚠️  Hybrid search failed: {e}, falling back to dense-only")
            return None

    def retrieve(
            self,
            claim: str,
            collection_name: str,
            embedding_model_name: str,
            config: RetrievalConfig,
            claim_id: Optional[int] = None
    ) -> RetrievalResult:
        """
        Retrieve chunks for a single claim.

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

        # Embed the claim
        t_start = time.time()
        claim_embedding = embedding_model.encode(
            claim,
            show_progress_bar=False,
            device=self.device,
            convert_to_numpy=True
        ).tolist()

        # Determine retrieval limit
        limit = config.k if config.strategy == RetrievalStrategy.TOP_K else 100

        # Try hybrid search if enabled
        results = None
        if self.use_hybrid:
            bm25_encoder = self._get_bm25_encoder(collection_name)
            if bm25_encoder is not None:
                results = self._hybrid_search(
                    client, collection_name, claim_embedding,
                    claim, bm25_encoder, limit
                )

        # Fallback to dense-only search
        if results is None:
            if config.strategy == RetrievalStrategy.TOP_K:
                results = client.search(
                    collection_name=collection_name,
                    query_vector=claim_embedding,
                    limit=config.k
                )
            else:  # THRESHOLD strategy
                results = client.search(
                    collection_name=collection_name,
                    query_vector=claim_embedding,
                    limit=100,
                    score_threshold=config.threshold
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