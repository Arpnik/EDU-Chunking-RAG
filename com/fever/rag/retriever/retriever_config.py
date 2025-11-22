from typing import Optional, List, Dict, Set
from sentence_transformers import SentenceTransformer
import time
from com.fever.rag.utils.data_helper import VectorDBConfig, RetrievalConfig, RetrievalResult, RetrievalStrategy, \
     get_device


class VectorDBRetriever:
    """Retrieves chunks from Qdrant vector database."""

    def __init__(self, db_config: VectorDBConfig):
        """
        Initialize the retriever.

        Args:
            db_config: Vector database configuration
            device: Device for embeddings ('cuda', 'mps', 'cpu', or None for auto)
        """
        self.db_config = db_config
        self.device = get_device()
        self._model_cache: Dict[str, SentenceTransformer] = {}

    def _get_embedding_model(self, model_name: str) -> SentenceTransformer:
        """Get or load an embedding model (with caching)."""
        if model_name not in self._model_cache:
            print(f"Loading embedding model: {model_name}")
            self._model_cache[model_name] = SentenceTransformer(model_name, device=self.device)
        return self._model_cache[model_name]

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
        client = self.db_config.connect_to_qdrant()
        embedding_model = self._get_embedding_model(embedding_model_name)

        # Embed the claim
        t_start = time.time()
        claim_embedding = embedding_model.encode(
            claim,
            show_progress_bar=False,
            device=self.device,
            convert_to_numpy=True
        )

        # Retrieve based on strategy
        if config.strategy == RetrievalStrategy.TOP_K:
            results = client.search(
                collection_name=collection_name,
                query_vector=claim_embedding.tolist(),
                limit=config.k
            )
        else:  # THRESHOLD strategy
            results = client.search(
                collection_name=collection_name,
                query_vector=claim_embedding.tolist(),
                limit=100,  # Retrieve more than we might need
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
