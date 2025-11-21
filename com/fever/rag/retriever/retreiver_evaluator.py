import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from tqdm import tqdm

from com.fever.rag.chunker.base_chunker import BaseChunker
from com.fever.rag.chunker.edu_chunker_with_linear_head import EDUChunkerWithLinearHead
from com.fever.rag.chunker.fixed_char_chunker import FixedCharChunker
from com.fever.rag.chunker.sentence_chunker import SentenceChunker
from com.fever.rag.evidence.vector_db_builder import VectorDBBuilder
from com.fever.rag.retriever.retriever_config import VectorDBRetriever
from com.fever.rag.utils.DataHelper import VectorDBConfig, EvaluationMetrics, RetrievalConfig, RetrievalStrategy


class RetrieverEvaluator:
    """Evaluates retriever performance using FEVER claims."""

    def __init__(
            self,
            claim_file_path: str,
            embedding_model_name: str,
            chunker: BaseChunker,
            db_config: VectorDBConfig,
            wiki_dir: str = "wiki",
            output_file: str = "retrieval_evaluation_results.jsonl",
            k_values: List[int] = None,
            batch_size: int = 100,
            max_files: Optional[int] = None,
            overlap: Optional[int] = None
    ):
        """
        Initialize the retriever evaluator.

        Args:
            claim_file_path: Path to FEVER claims JSONL file
            embedding_model_name: Name of embedding model to use
            chunker: Chunking strategy to use
            db_config: Vector database configuration
            wiki_dir: Directory containing Wikipedia JSONL files
            output_file: File to append evaluation results
            k_values: List of k values to evaluate (default: [1, 3, 5, 10, 20])
            batch_size: Batch size for vector DB building
            max_files: Maximum number of wiki files to process
        """
        self.claim_file_path = Path(claim_file_path)
        self.embedding_model_name = embedding_model_name
        self.chunker = chunker
        self.db_config = db_config
        self.wiki_dir = wiki_dir
        self.output_file = Path(output_file)
        self.k_values = k_values or [1, 3, 5, 10, 20]
        self.batch_size = batch_size
        self.max_files = max_files
        self.overlap = overlap

        # Initialize components
        self.builder = VectorDBBuilder(
            wiki_dir=wiki_dir,
            batch_size=batch_size,
            max_files=max_files,
            db_config=db_config
        )
        self.retriever = VectorDBRetriever(db_config=db_config)

        # Collection name
        self.collection_name = self._get_collection_name()

    def _get_collection_name(self) -> str:
        """Generate collection name from model and chunker."""
        model_short = self.embedding_model_name.split('/')[-1].split('-')[0].lower()
        if 'minilm' in self.embedding_model_name.lower():
            model_short = 'minilm'
        elif 'mpnet' in self.embedding_model_name.lower():
            model_short = 'mpnet'
        elif 'multi-qa' in self.embedding_model_name.lower():
            model_short = 'multiqa'

        return f"{model_short}_{self.chunker.name}_chunks"

    def _load_claims(self) -> List[Dict]:
        """Load claims from FEVER JSONL file."""
        claims = []
        print(f"\nLoading claims from: {self.claim_file_path}")

        with open(self.claim_file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading claims"):
                try:
                    claim_data = json.loads(line.strip())
                    claims.append(claim_data)
                except json.JSONDecodeError:
                    continue

        print(f"Loaded {len(claims)} claims")
        return claims

    def _extract_evidence_articles(self, claim_data: Dict) -> Set[str]:
        """
        Extract article IDs from FEVER evidence.

        FEVER format: evidence is a list of [wiki_url, sentence_id] pairs
        """
        evidence_articles = set()

        if 'evidence' in claim_data:
            for evidence_set in claim_data['evidence']:
                for evidence_item in evidence_set:
                    if len(evidence_item) >= 2:
                        # evidence_item is typically [None, None, wiki_url, sentence_id]
                        # or [wiki_url, sentence_id]
                        article_id = evidence_item[2] if len(evidence_item) == 4 else evidence_item[0]
                        if article_id:
                            evidence_articles.add(article_id)

        return evidence_articles

    def calculate_metrics(
            self,
            retrieved_articles: List[str],
            relevant_articles: Set[str],
            k_values: List[int]
    ) -> Dict:
        """
        Calculate precision@k, recall@k, and accuracy@k.

        Args:
            retrieved_articles: Ordered list of retrieved article IDs
            relevant_articles: Set of ground truth article IDs
            k_values: List of k values to evaluate

        Returns:
            Dictionary containing metrics for each k
        """
        metrics = {}

        for k in k_values:
            top_k = retrieved_articles[:k]
            top_k_set = set(top_k)

            # True positives: relevant articles in top-k
            tp = len(top_k_set.intersection(relevant_articles))

            # Precision@k: proportion of retrieved that are relevant
            precision = tp / k if k > 0 else 0.0

            # Recall@k: proportion of relevant that are retrieved
            recall = tp / len(relevant_articles) if len(relevant_articles) > 0 else 0.0

            # Accuracy@k: 1 if at least one relevant doc in top-k, else 0
            accuracy = 1.0 if tp > 0 else 0.0

            metrics[k] = {
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'true_positives': tp
            }

        return metrics

    def _calculate_mrr(self, retrieved_articles: List[str], relevant_articles: Set[str]) -> float:
        """Calculate Mean Reciprocal Rank."""
        for rank, article_id in enumerate(retrieved_articles, start=1):
            if article_id in relevant_articles:
                return 1.0 / rank
        return 0.0

    def build_vector_db(self, reset: bool = True):
        """Build the vector database."""
        print("\n" + "=" * 70)
        print("BUILDING VECTOR DATABASE")
        print("=" * 70)

        self.builder.add_embedding_model(self.embedding_model_name)
        self.builder.add_chunker(self.chunker)
        self.builder.build(reset=reset)

    def evaluate(self, retrieval_config: RetrievalConfig) -> EvaluationMetrics:
        """
        Evaluate retriever on the claims dataset.

        Args:
            retrieval_config: Configuration for retrieval

        Returns:
            EvaluationMetrics with aggregated results
        """
        print("\n" + "=" * 70)
        print("EVALUATING RETRIEVER")
        print("=" * 70)
        print(f"Collection: {self.collection_name}")
        print(f"Embedding Model: {self.embedding_model_name}")
        print(f"Chunker: {self.chunker.name}")
        print(f"Retrieval Strategy: {retrieval_config.strategy.value}")
        print(f"K values: {self.k_values}")

        # Load claims
        claims = self._load_claims()

        # Aggregate metrics
        total_metrics = {k: {'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0} for k in self.k_values}
        total_mrr = 0.0
        total_retrieval_time = 0.0
        total_relevant_docs = 0
        evaluated_claims = 0

        print("\nEvaluating claims...")
        for claim_data in tqdm(claims, desc="Processing claims"):
            claim_text = claim_data.get('claim', '')
            claim_id = claim_data.get('id')

            if not claim_text:
                continue

            # Get ground truth evidence
            relevant_articles = self._extract_evidence_articles(claim_data)
            if not relevant_articles:
                continue

            total_relevant_docs += len(relevant_articles)
            evaluated_claims += 1

            # Retrieve chunks
            result = self.retriever.retrieve(
                claim=claim_text,
                collection_name=self.collection_name,
                embedding_model_name=self.embedding_model_name,
                config=retrieval_config,
                claim_id=claim_id
            )

            total_retrieval_time += result.retrieval_time

            # Extract article IDs from retrieved chunks
            retrieved_articles = []
            for chunk in result.chunks:
                article_id = chunk.payload.get('article_id')
                if article_id and article_id not in retrieved_articles:
                    retrieved_articles.append(article_id)

            # Calculate metrics for this claim
            claim_metrics = self.calculate_metrics(
                retrieved_articles,
                relevant_articles,
                self.k_values
            )

            # Calculate MRR
            mrr = self._calculate_mrr(retrieved_articles, relevant_articles)
            total_mrr += mrr

            # Aggregate metrics
            for k in self.k_values:
                total_metrics[k]['precision'] += claim_metrics[k]['precision']
                total_metrics[k]['recall'] += claim_metrics[k]['recall']
                total_metrics[k]['accuracy'] += claim_metrics[k]['accuracy']

        # Average metrics
        avg_metrics = EvaluationMetrics(
            precision_at_k={k: total_metrics[k]['precision'] / evaluated_claims
                            for k in self.k_values},
            recall_at_k={k: total_metrics[k]['recall'] / evaluated_claims
                         for k in self.k_values},
            accuracy_at_k={k: total_metrics[k]['accuracy'] / evaluated_claims
                           for k in self.k_values},
            mean_reciprocal_rank=total_mrr / evaluated_claims,
            total_claims=evaluated_claims,
            total_relevant_docs=total_relevant_docs,
            avg_retrieval_time=total_retrieval_time / evaluated_claims
        )

        return avg_metrics

    @staticmethod
    def print_metrics(metrics: EvaluationMetrics):
        """Print evaluation metrics to console."""
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        print(f"\nDataset Statistics:")
        print(f"  Total claims evaluated: {metrics.total_claims}")
        print(f"  Total relevant documents: {metrics.total_relevant_docs}")
        print(f"  Avg relevant docs per claim: {metrics.total_relevant_docs / metrics.total_claims:.2f}")
        print(f"  Avg retrieval time: {metrics.avg_retrieval_time * 1000:.2f}ms")
        print(f"  Mean Reciprocal Rank: {metrics.mean_reciprocal_rank:.4f}")

        print(f"\nMetrics by K:")
        print(f"{'K':>5} {'Precision':>12} {'Recall':>12} {'Accuracy':>12}")
        print("-" * 45)
        for k in sorted(metrics.precision_at_k.keys()):
            print(f"{k:>5} {metrics.precision_at_k[k]:>12.4f} "
                  f"{metrics.recall_at_k[k]:>12.4f} {metrics.accuracy_at_k[k]:>12.4f}")

    def save_results(self, metrics: EvaluationMetrics, retrieval_config: RetrievalConfig):
        """Save evaluation results to output file (append mode)."""
        result = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'embedding_model': self.embedding_model_name,
            'chunker': self.chunker.name,
            'collection_name': self.collection_name,
            'retrieval_strategy': retrieval_config.strategy.value,
            'retrieval_k': retrieval_config.k if retrieval_config.strategy == RetrievalStrategy.TOP_K else None,
            'retrieval_threshold': retrieval_config.threshold if retrieval_config.strategy == RetrievalStrategy.THRESHOLD else None,
            'total_claims': metrics.total_claims,
            'total_relevant_docs': metrics.total_relevant_docs,
            'avg_retrieval_time_ms': metrics.avg_retrieval_time * 1000,
            'mean_reciprocal_rank': metrics.mean_reciprocal_rank,
            'precision_at_k': metrics.precision_at_k,
            'recall_at_k': metrics.recall_at_k,
            'accuracy_at_k': metrics.accuracy_at_k,
            'overlap': self.overlap if self.overlap else 0
        }

        # Append to file
        with open(self.output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result) + '\n')

        print(f"\n✓ Results saved to: {self.output_file}")

    def run(self, build_db: bool = True, retrieval_config: RetrievalConfig = None):
        """
        Run the complete evaluation pipeline.

        Args:
            build_db: Whether to build the vector database first
            retrieval_config: Retrieval configuration (default: top-20)
        """
        if retrieval_config is None:
            retrieval_config = RetrievalConfig(
                strategy=RetrievalStrategy.TOP_K,
                k=max(self.k_values)
            )

        # Build database if requested
        if build_db:
            self.build_vector_db(reset=True)

        # Evaluate
        metrics = self.evaluate(retrieval_config)

        # Print results
        self.print_metrics(metrics)

        # Save results
        self.save_results(metrics, retrieval_config)

        return metrics


if __name__ == "__main__":
    # Configure
    db_config = VectorDBConfig(
        host="localhost",
        port=6333,
        use_grpc=True
    )

    OVERLAP = [0]
    for overlap in OVERLAP:
        chunker = EDUChunkerWithLinearHead(model_path = "../../../../edu_segmenter_linear/best_model", overlap=overlap)
        evaluator = RetrieverEvaluator(
            claim_file_path="../../../../dataset/reduced_fever_data/paper_dev.jsonl",
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            chunker=chunker,
            db_config=db_config,
            wiki_dir="../../../../dataset/reduced_fever_data/wiki",
            output_file="../../../retrieval_evaluation_results.jsonl",
            k_values=[1, 3, 5, 10, 20],
            max_files=None,
            overlap=overlap
        )

        # Run evaluation
        retrieval_config = RetrievalConfig(
            strategy=RetrievalStrategy.TOP_K,
            k=20
        )
        evaluator.run(build_db=True, retrieval_config=retrieval_config)

    # chunker = SentenceChunker()
    # evaluator = RetrieverEvaluator(
    #     claim_file_path="../../../../dataset/reduced_fever_data/paper_dev.jsonl",
    #     embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    #     chunker=chunker,
    #     db_config=db_config,
    #     wiki_dir="../../../../dataset/reduced_fever_data/wiki",
    #     output_file="../../../retrieval_evaluation_results.jsonl",
    #     k_values=[1, 3, 5, 10, 20],
    #     max_files=None)
    #
    # # Run evaluation
    # retrieval_config = RetrievalConfig(
    #     strategy=RetrievalStrategy.TOP_K,
    #     k=20
    # )
    # evaluator.run(build_db=True, retrieval_config=retrieval_config)