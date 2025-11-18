from typing import List, Dict, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchParams
from sentence_transformers import SentenceTransformer
import torch
from dataclasses import dataclass
import json


@dataclass
class RetrievedEvidence:
    """Container for retrieved evidence."""
    text: str
    article_id: str
    chunk_id: int
    score: float
    metadata: Dict

    def __repr__(self):
        return f"Evidence(article={self.article_id}, chunk={self.chunk_id}, score={self.score:.4f})"

    def to_dict(self):
        """Convert to dictionary."""
        return {
            'text': self.text,
            'article_id': self.article_id,
            'chunk_id': self.chunk_id,
            'score': self.score,
            'metadata': self.metadata
        }


class QdrantRetriever:
    """retriever for querying Qdrant vector database with claims."""

    def __init__(
            self,
            collection_name: str,
            embedding_model_name: str,
            qdrant_host: str = "localhost",
            qdrant_port: int = 6334,
            use_grpc: bool = True,
            device: Optional[str] = None
    ):
        """
        Initialize the Qdrant retriever.

        Args:
            collection_name: Name of the Qdrant collection to query
            embedding_model_name: Name of the embedding model (must match what was used to build the DB)
            qdrant_host: Qdrant host
            qdrant_port: Qdrant port (6334 for gRPC, 6333 for HTTP)
            use_grpc: Use gRPC for faster communication
            device: Device to use for embeddings (cuda/mps/cpu), auto-detected if None
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.use_grpc = use_grpc

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Initializing Qdrant retriever...")
        print(f"  Collection: {collection_name}")
        print(f"  Model: {embedding_model_name}")
        print(f"  Device: {self.device}")

        # Connect to Qdrant
        self.client = self._connect_to_qdrant()

        # Load embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name, device=self.device)

        # Verify collection exists
        self._verify_collection()

        print(f"✓ retriever ready!")

    def _connect_to_qdrant(self) -> QdrantClient:
        """Connect to Qdrant."""
        if self.use_grpc:
            client = QdrantClient(
                host=self.qdrant_host,
                port=self.qdrant_port,
                prefer_grpc=True
            )
        else:
            client = QdrantClient(
                host=self.qdrant_host,
                port=self.qdrant_port
            )
        return client

    def _verify_collection(self):
        """Verify the collection exists and get info."""
        try:
            info = self.client.get_collection(self.collection_name)
            print(f"  Collection size: {info.points_count:,} documents")
        except Exception as e:
            raise ValueError(f"Collection '{self.collection_name}' not found: {e}")

    def retrieve(
            self,
            claim: str,
            top_k: int = 5,
            score_threshold: Optional[float] = None,
            filter_article_ids: Optional[List[str]] = None,
            return_full_metadata: bool = False
    ) -> List[RetrievedEvidence]:
        """
        Retrieve relevant evidence for a claim.

        Args:
            claim: The claim to verify
            top_k: Number of top results to return
            score_threshold: Minimum similarity score (0-1), None for no threshold
            filter_article_ids: Optional list of article IDs to restrict search to
            return_full_metadata: If True, include all metadata in results

        Returns:
            List of RetrievedEvidence objects, sorted by relevance score
        """
        # Encode the claim
        query_vector = self.embedding_model.encode(
            claim,
            show_progress_bar=False,
            device=self.device,
            convert_to_numpy=True
        )

        # Build filter if article IDs specified
        query_filter = None
        if filter_article_ids:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="article_id",
                        match=MatchValue(any=filter_article_ids)
                    )
                ]
            )

        # Search in Qdrant
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
            score_threshold=score_threshold,
            with_payload=True,
            search_params=SearchParams(
                exact=False,  # Use HNSW index for faster search
                hnsw_ef=128  # Higher ef = more accurate but slower
            )
        )

        # Convert to RetrievedEvidence objects
        results = []
        for hit in search_result:
            evidence = RetrievedEvidence(
                text=hit.payload.get('text', ''),
                article_id=hit.payload.get('article_id', ''),
                chunk_id=hit.payload.get('chunk_id', 0),
                score=hit.score,
                metadata=hit.payload if return_full_metadata else {
                    k: v for k, v in hit.payload.items()
                    if k not in ['text', 'article_id', 'chunk_id']
                }
            )
            results.append(evidence)

        return results

    def retrieve_batch(
            self,
            claims: List[str],
            top_k: int = 5,
            score_threshold: Optional[float] = None,
            show_progress: bool = True
    ) -> List[List[RetrievedEvidence]]:
        """
        Retrieve evidence for multiple claims in batch.

        Args:
            claims: List of claims to verify
            top_k: Number of top results per claim
            score_threshold: Minimum similarity score
            show_progress: Show progress bar

        Returns:
            List of lists of RetrievedEvidence objects
        """
        # Encode all claims at once (more efficient)
        query_vectors = self.embedding_model.encode(
            claims,
            show_progress_bar=show_progress,
            device=self.device,
            convert_to_numpy=True,
            batch_size=32
        )

        results = []
        for query_vector in query_vectors:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True
            )

            claim_results = [
                RetrievedEvidence(
                    text=hit.payload.get('text', ''),
                    article_id=hit.payload.get('article_id', ''),
                    chunk_id=hit.payload.get('chunk_id', 0),
                    score=hit.score,
                    metadata={k: v for k, v in hit.payload.items()
                              if k not in ['text', 'article_id', 'chunk_id']}
                )
                for hit in search_result
            ]
            results.append(claim_results)

        return results

    def retrieve_by_article(
            self,
            claim: str,
            article_id: str,
            top_k: int = 5
    ) -> List[RetrievedEvidence]:
        """
        Retrieve evidence from a specific article.

        Args:
            claim: The claim to verify
            article_id: Specific article ID to search in
            top_k: Number of top results

        Returns:
            List of RetrievedEvidence objects from the specified article
        """
        return self.retrieve(
            claim=claim,
            top_k=top_k,
            filter_article_ids=[article_id]
        )

    def get_article_chunks(
            self,
            article_id: str,
            limit: int = 100
    ) -> List[RetrievedEvidence]:
        """
        Get all chunks for a specific article (no semantic search).

        Args:
            article_id: Article ID to retrieve
            limit: Maximum number of chunks to return

        Returns:
            List of all chunks from the article
        """
        # Use scroll to get all points matching the filter
        results = []
        offset = None

        while True:
            response = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="article_id",
                            match=MatchValue(value=article_id)
                        )
                    ]
                ),
                limit=min(100, limit - len(results)),
                offset=offset,
                with_payload=True
            )

            points, offset = response

            for point in points:
                evidence = RetrievedEvidence(
                    text=point.payload.get('text', ''),
                    article_id=point.payload.get('article_id', ''),
                    chunk_id=point.payload.get('chunk_id', 0),
                    score=0.0,  # No score for direct retrieval
                    metadata={k: v for k, v in point.payload.items()
                              if k not in ['text', 'article_id', 'chunk_id']}
                )
                results.append(evidence)

            if offset is None or len(results) >= limit:
                break

        return results

    def print_results(
            self,
            claim: str,
            results: List[RetrievedEvidence],
            max_text_length: int = 200
    ):
        """
        Pretty print retrieval results.

        Args:
            claim: The original claim
            results: List of retrieved evidence
            max_text_length: Maximum length of text to display
        """
        print("=" * 80)
        print(f"CLAIM: {claim}")
        print("=" * 80)
        print(f"\nFound {len(results)} relevant evidence:\n")

        for i, evidence in enumerate(results, 1):
            text_display = evidence.text[:max_text_length]
            if len(evidence.text) > max_text_length:
                text_display += "..."

            print(f"{i}. [{evidence.article_id}] (Score: {evidence.score:.4f})")
            print(f"   {text_display}")
            print()

    def save_results(
            self,
            results: List[Tuple[str, List[RetrievedEvidence]]],
            output_file: str
    ):
        """
        Save retrieval results to JSON file.

        Args:
            results: List of (claim, evidence_list) tuples
            output_file: Path to output JSON file
        """
        output = []
        for claim, evidence_list in results:
            output.append({
                'claim': claim,
                'evidence': [e.to_dict() for e in evidence_list]
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved {len(results)} results to {output_file}")


# Example usage
if __name__ == "__main__":
    # Initialize retriever
    retriever = QdrantRetriever(
        collection_name="minilm_sentence_chunks",  # Adjust to your collection name
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        qdrant_host="localhost",
        qdrant_port=6334,
        use_grpc=True
    )

    # Example 1: Single claim retrieval
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Single Claim Retrieval")
    print("=" * 80)

    claim = "The Eiffel Tower is located in Paris, France."
    results = retriever.retrieve(
        claim=claim,
        top_k=5,
        score_threshold=0.3  # Only return results with similarity > 0.3
    )

    retriever.print_results(claim, results)

    # Example 2: Batch retrieval
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Batch Retrieval")
    print("=" * 80)

    claims = [
        "Python is a programming language.",
        "The Earth orbits around the Sun.",
        "Water boils at 100 degrees Celsius."
    ]

    batch_results = retriever.retrieve_batch(
        claims=claims,
        top_k=3,
        show_progress=True
    )

    for claim, results in zip(claims, batch_results):
        print(f"\nClaim: {claim}")
        print(f"Found {len(results)} evidence pieces")
        for i, evidence in enumerate(results[:2], 1):  # Show top 2
            print(f"  {i}. [{evidence.article_id}] Score: {evidence.score:.4f}")


    # Example 5: Save results to file
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Save Results")
    print("=" * 80)

    results_to_save = [
        (claim, retriever.retrieve(claim, top_k=5))
        for claim in claims
    ]

    retriever.save_results(results_to_save, "retrieval_results.json")