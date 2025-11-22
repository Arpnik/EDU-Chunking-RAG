from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Set, Dict

import numpy as np
from qdrant_client.grpc import ScoredPoint
from sympy.printing.pytorch import torch
from qdrant_client import QdrantClient


class RetrievalStrategy(Enum):
    """Retrieval strategy types."""
    TOP_K = "top_k"
    THRESHOLD = "threshold"


@dataclass
class VectorDBConfig:
    """Configuration for vector database connection."""
    host: str = "localhost"
    port: int = 6333
    use_grpc: bool = True

    @property
    def actual_port(self) -> int:
        """Get actual port based on gRPC setting."""
        return 6334 if self.use_grpc else self.port

    def connect_to_qdrant(self) -> QdrantClient:
        """Connect to Qdrant."""
        if self.use_grpc:
            return QdrantClient(
                host=self.host,
                port=self.actual_port,
                prefer_grpc=True
            )
        else:
            return QdrantClient(
                host=self.host,
                port=self.port
            )



@dataclass
class RetrievalConfig:
    """Configuration for retrieval approach."""
    strategy: RetrievalStrategy
    k: Optional[int] = None  # For TOP_K strategy
    threshold: Optional[float] = None  # For THRESHOLD strategy

    def __post_init__(self):
        if self.strategy == RetrievalStrategy.TOP_K and self.k is None:
            raise ValueError("k must be specified for TOP_K strategy")
        if self.strategy == RetrievalStrategy.THRESHOLD and self.threshold is None:
            raise ValueError("threshold must be specified for THRESHOLD strategy")

    @property
    def name(self) -> str:
        """Get descriptive name for this config."""
        if self.strategy == RetrievalStrategy.TOP_K:
            return f"top_{self.k}"
        else:
            return f"threshold_{self.threshold:.2f}"


@dataclass
class RetrievalResult:
    """Result from a single retrieval operation."""
    claim: str
    claim_id: Optional[int]
    collection_name: str
    embedding_model_name: str
    retrieval_config: RetrievalConfig
    chunks: List[ScoredPoint]
    retrieval_time: float

    @property
    def retrieved_article_ids(self) -> Set[str]:
        """Extract unique article IDs from retrieved chunks."""
        article_ids = set()
        for chunk in self.chunks:
            if 'article_id' in chunk.payload:
                article_ids.add(chunk.payload['article_id'])
        return article_ids

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    @property
    def avg_score(self) -> float:
        return np.mean([chunk.score for chunk in self.chunks]) if self.chunks else 0.0

    @property
    def max_score(self) -> float:
        return max([chunk.score for chunk in self.chunks]) if self.chunks else 0.0


@dataclass
class ClassificationMetrics:
    """Store classification metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    support: int
    per_class_metrics: Dict[str, Dict[str, float]]

    def __str__(self):
        return (f"Accuracy: {self.accuracy:.3f}\n"
                f"Precision: {self.precision:.3f}\n"
                f"Recall: {self.recall:.3f}\n"
                f"F1: {self.f1:.3f}\n"
                f"Support: {self.support}")

def get_device() -> str:
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


@dataclass
class EvaluationMetrics:
    """Stores evaluation metrics for retrieval."""
    precision_at_k: Dict[int, float]  # k -> precision
    recall_at_k: Dict[int, float]  # k -> recall
    accuracy_at_k: Dict[int, float]  # k -> accuracy (at least one correct in top-k)
    mean_reciprocal_rank: float
    total_claims: int
    total_relevant_docs: int
    avg_retrieval_time: float

