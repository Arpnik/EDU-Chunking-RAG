import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Set

from qdrant_client import QdrantClient
from tqdm import tqdm
from com.fever.rag.chunker.base_chunker import BaseChunker
from com.fever.rag.evidence.vector_db_builder import VectorDBBuilder, DatasetConfig, DatasetFormat
from com.fever.rag.retriever.retriever_config import VectorDBRetriever
from com.fever.rag.utils.chunker_helper import get_chunker, CHUNKER_ARGS, ChunkerType
from com.fever.rag.utils.data_helper import VectorDBConfig, EvaluationMetrics, RetrievalConfig, RetrievalStrategy, \
    get_collection_name


class RetrieverEvaluator:
    """Evaluates retriever performance on FEVER or SQuAD datasets."""

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
            overlap: Optional[int] = None,
            shared_client: Optional[QdrantClient] = None,
            dataset_type: str = "fever",
    ):
        """
        Initialize the retriever evaluator.

        Args:
            claim_file_path: Path to claims/questions file (JSONL for FEVER, JSON for SQuAD)
            embedding_model_name: Name of embedding model to use
            chunker: Chunking strategy to use
            db_config: Vector database configuration
            wiki_dir: Directory containing evidence files
            output_file: File to append evaluation results
            k_values: List of k values to evaluate (default: [1, 3, 5, 10, 20])
            batch_size: Batch size for vector DB building
            max_files: Maximum number of files to process
            overlap: Overlap parameter for chunking
            shared_client: Optional shared Qdrant client
            dataset_type: Either 'fever' or 'squad'
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
        self.shared_client = shared_client
        self.dataset_type = dataset_type.lower()

        # Create appropriate dataset config for VectorDBBuilder
        if self.dataset_type == "squad":
            dataset_config = DatasetConfig.squad_config()
        else:
            dataset_config = DatasetConfig.fever_config()

        # Initialize components
        self.builder = VectorDBBuilder(
            wiki_dir=wiki_dir,
            batch_size=batch_size,
            max_files=max_files,
            db_config=db_config,
            shared_client=shared_client,
            dataset_config=dataset_config
        )
        self.retriever = VectorDBRetriever(
            db_config=db_config,
            shared_client=shared_client,
            use_hybrid=False
        )

        # Collection name
        self.collection_name = get_collection_name(self.embedding_model_name, self.chunker)

    def _load_claims(self) -> List[Dict]:
        """
        Load claims/questions from file.

        FEVER: JSONL format (one claim per line)
        SQuAD: Nested JSON format (needs conversion to flat structure)
        """
        claims = []
        dataset_name = "SQuAD questions" if self.dataset_type == "squad" else "FEVER claims"
        print(f"\nLoading {dataset_name} from: {self.claim_file_path}")

        if self.dataset_type == "fever":
            # FEVER: Load JSONL
            with open(self.claim_file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Loading"):
                    try:
                        claim_data = json.loads(line.strip())
                        claims.append(claim_data)
                    except json.JSONDecodeError:
                        continue

        else:  # squad
            # SQuAD: Load nested JSON and flatten
            with open(self.claim_file_path, 'r', encoding='utf-8') as f:
                squad_data = json.load(f)

            print("Flattening SQuAD structure...")
            for article in tqdm(squad_data.get('data', []), desc="Loading"):
                title = article.get('title', 'unknown')

                for para_idx, paragraph in enumerate(article.get('paragraphs', [])):
                    context = paragraph.get('context', '')
                    context_id = f"{title}_para_{para_idx}"

                    for qa in paragraph.get('qas', []):
                        question_id = qa.get('id')
                        question = qa.get('question', '')
                        is_impossible = qa.get('is_impossible', False)

                        # Extract gold answers
                        gold_answers = []
                        if not is_impossible:
                            answers = qa.get('answers', [])
                            gold_answers = [ans.get('text', '') for ans in answers]

                        # Create flattened claim structure
                        claim_data = {
                            'question_id': question_id,
                            'question': question,
                            'gold_context_id': context_id,
                            'gold_answers': gold_answers,
                            'is_impossible': is_impossible,
                            'title': title
                        }
                        claims.append(claim_data)

        print(f"Loaded {len(claims)} {dataset_name}")
        return claims

    def _extract_evidence_articles(self, claim_data: Dict) -> Set[str]:
        """
        Extract article/context IDs from evidence.

        FEVER format: evidence is [[None, None, wiki_url, sentence_id], ...]
        SQuAD format: gold_context_id field contains the context ID
        """
        evidence_articles = set()

        if self.dataset_type == "fever":
            # FEVER: Extract from evidence field
            if 'evidence' in claim_data:
                for evidence_set in claim_data['evidence']:
                    for evidence_item in evidence_set:
                        if len(evidence_item) >= 2:
                            article_id = evidence_item[2] if len(evidence_item) == 4 else evidence_item[0]
                            if article_id:
                                evidence_articles.add(article_id)

        else:  # squad
            # SQuAD: gold_context_id contains the single context ID
            if 'gold_context_id' in claim_data:
                evidence_articles.add(claim_data['gold_context_id'])

        return evidence_articles

    def _check_answer_in_chunks(self, chunks: List, gold_answers: List[str]) -> bool:
        """
        Check if any gold answer appears in retrieved chunks (for SQuAD).

        Args:
            chunks: Retrieved chunks
            gold_answers: List of acceptable answer texts

        Returns:
            True if any answer found in chunks
        """
        if not gold_answers:
            return False

        for chunk in chunks:
            chunk_text = chunk.payload.get('text', '').lower()
            if any(ans.lower() in chunk_text for ans in gold_answers if ans):
                return True
        return False

    def calculate_metrics(
            self,
            retrieved_articles: List[str],
            relevant_articles: Set[str],
            k_values: List[int],
            retrieved_chunks: List = None,
            gold_answers: List[str] = None
    ) -> Dict:
        """
        Calculate precision@k, recall@k, and accuracy@k.

        Args:
            retrieved_articles: Ordered list of retrieved article IDs
            relevant_articles: Set of ground truth article IDs
            k_values: List of k values to evaluate
            retrieved_chunks: Retrieved chunks (for SQuAD answer checking)
            gold_answers: Gold answers (for SQuAD)

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

            # For SQuAD: Also check if answer text appears in chunks
            answer_accuracy = 0.0
            if self.dataset_type == "squad" and retrieved_chunks and gold_answers:
                top_k_chunks = retrieved_chunks[:k]
                answer_found = self._check_answer_in_chunks(top_k_chunks, gold_answers)
                answer_accuracy = 1.0 if answer_found else 0.0

            metrics[k] = {
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'answer_accuracy': answer_accuracy if self.dataset_type == "squad" else accuracy,
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
        print(f"BUILDING VECTOR DATABASE ({self.dataset_type.upper()})")
        print("=" * 70)

        self.builder.add_embedding_model(self.embedding_model_name)
        self.builder.add_chunker(self.chunker)
        self.builder.build(reset=reset)

    def evaluate(self, retrieval_config: RetrievalConfig) -> EvaluationMetrics:
        """
        Evaluate retriever on the dataset.

        Args:
            retrieval_config: Configuration for retrieval

        Returns:
            EvaluationMetrics with aggregated results
        """
        print("\n" + "=" * 70)
        print(f"EVALUATING RETRIEVER ({self.dataset_type.upper()})")
        print("=" * 70)
        print(f"Collection: {self.collection_name}")
        print(f"Embedding Model: {self.embedding_model_name}")
        print(f"Chunker: {self.chunker.name}")
        print(f"Retrieval Strategy: {retrieval_config.strategy.value}")
        print(f"K values: {self.k_values}")

        # Load claims/questions
        claims = self._load_claims()

        # Aggregate metrics
        total_metrics = {
            k: {
                'precision': 0.0,
                'recall': 0.0,
                'accuracy': 0.0,
                'answer_accuracy': 0.0
            } for k in self.k_values
        }
        total_mrr = 0.0
        total_retrieval_time = 0.0
        total_relevant_docs = 0
        evaluated_claims = 0
        skipped_claims = 0

        print("\nEvaluating...")
        for claim_data in tqdm(claims, desc="Processing"):
            # Get query text (different field names for FEVER vs SQuAD)
            if self.dataset_type == "fever":
                claim_text = claim_data.get('claim', '')
                claim_id = claim_data.get('id')
                gold_answers = None
                is_impossible = False
            else:  # squad
                claim_text = claim_data.get('question', '')
                claim_id = claim_data.get('question_id')
                gold_answers = claim_data.get('gold_answers', [])
                is_impossible = claim_data.get('is_impossible', False)

            if not claim_text:
                skipped_claims += 1
                continue

            # Get ground truth evidence
            relevant_articles = self._extract_evidence_articles(claim_data)
            if not relevant_articles:
                skipped_claims += 1
                continue

            # Skip unanswerable questions in SQuAD
            if is_impossible:
                skipped_claims += 1
                continue

            total_relevant_docs += len(relevant_articles)
            evaluated_claims += 1

            # Retrieve chunks
            try:
                result = self.retriever.retrieve(
                    claim=claim_text,
                    collection_name=self.collection_name,
                    embedding_model_name=self.embedding_model_name,
                    config=retrieval_config,
                    claim_id=claim_id
                )
            except Exception as e:
                print(f"\nError retrieving for claim {claim_id}: {e}")
                skipped_claims += 1
                evaluated_claims -= 1
                continue

            total_retrieval_time += result.retrieval_time

            # Extract article IDs from retrieved chunks
            retrieved_articles = []
            for chunk in result.chunks:
                article_id = chunk.payload.get('article_id')
                if article_id and article_id not in retrieved_articles:
                    retrieved_articles.append(article_id)

            # Debug: Print first few claims to check ID matching
            if evaluated_claims < 3 and self.dataset_type == "squad":
                print(f"\n[DEBUG] Question {claim_id}:")
                print(f"  Gold context: {relevant_articles}")
                print(f"  Retrieved (top 3): {retrieved_articles[:3]}")
                if result.chunks:
                    print(f"  First chunk article_id: {result.chunks[0].payload.get('article_id')}")
                    print(f"  First chunk text preview: {result.chunks[0].payload.get('text', '')[:100]}...")

            # Calculate metrics for this claim
            claim_metrics = self.calculate_metrics(
                retrieved_articles,
                relevant_articles,
                self.k_values,
                retrieved_chunks=result.chunks,
                gold_answers=gold_answers
            )

            # Calculate MRR
            mrr = self._calculate_mrr(retrieved_articles, relevant_articles)
            total_mrr += mrr

            # Aggregate metrics
            for k in self.k_values:
                total_metrics[k]['precision'] += claim_metrics[k]['precision']
                total_metrics[k]['recall'] += claim_metrics[k]['recall']
                total_metrics[k]['accuracy'] += claim_metrics[k]['accuracy']
                if self.dataset_type == "squad":
                    total_metrics[k]['answer_accuracy'] += claim_metrics[k]['answer_accuracy']

        # Check if we have any evaluated claims
        if evaluated_claims == 0:
            print("\n⚠️  WARNING: No claims were successfully evaluated!")
            print(f"   Total claims in file: {len(claims)}")
            print(f"   Skipped claims: {skipped_claims}")
            print("\nPossible issues:")
            print("  - No valid ground truth evidence found")
            print("  - All questions marked as impossible (SQuAD)")
            print("  - Database collection doesn't exist or is empty")
            raise ValueError("No claims were successfully evaluated. Check your data and configuration.")

        # Average metrics
        avg_metrics = EvaluationMetrics(
            precision_at_k={k: total_metrics[k]['precision'] / evaluated_claims
                            for k in self.k_values},
            recall_at_k={k: total_metrics[k]['recall'] / evaluated_claims
                         for k in self.k_values},
            accuracy_at_k={k: total_metrics[k][
                                  'answer_accuracy' if self.dataset_type == "squad" else 'accuracy'] / evaluated_claims
                           for k in self.k_values},
            mean_reciprocal_rank=total_mrr / evaluated_claims,
            total_claims=evaluated_claims,
            total_relevant_docs=total_relevant_docs,
            avg_retrieval_time=total_retrieval_time / evaluated_claims
        )

        print(f"\n✓ Evaluated {evaluated_claims} claims (skipped {skipped_claims})")

        return avg_metrics

    @staticmethod
    def print_metrics(metrics: EvaluationMetrics):
        """Print evaluation metrics to console."""
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        print(f"\nDataset Statistics:")
        print(f"  Total evaluated: {metrics.total_claims}")
        print(f"  Total relevant documents: {metrics.total_relevant_docs}")
        print(f"  Avg relevant docs per query: {metrics.total_relevant_docs / metrics.total_claims:.2f}")
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
        chunker_config = {}
        if hasattr(self.chunker, 'chunk_size'):
            chunker_config['chunk_size'] = self.chunker.chunk_size
        if hasattr(self.chunker, 'max_tokens'):
            chunker_config['max_tokens'] = self.chunker.max_tokens
        if hasattr(self.chunker, 'overlap'):
            chunker_config['overlap'] = self.chunker.overlap
        if hasattr(self.chunker, 'model_path'):
            chunker_config['model_path'] = str(self.chunker.model_path)
        if hasattr(self.chunker, 'dataset_type'):
            chunker_config['dataset_type'] = self.chunker.dataset_type

        result = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_type': self.dataset_type,
            'embedding_model': self.embedding_model_name,
            'chunker': self.chunker.name,
            'chunker_config': chunker_config,
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluation of different chunking strategies for retrieval"
    )

    # Dataset config
    parser.add_argument('--dataset', type=str, default='fever',
                        choices=['fever', 'squad'],
                        help="Dataset type: 'fever' or 'squad'")
    parser.add_argument('--squad_evidence', type=str, default='train-v2.0.json',
                        help='Path to SQuAD evidence file (e.g., train-v2.0.json)')
    parser.add_argument('--squad_questions', type=str, default='dev-v2.0.json',
                        help='Path to SQuAD questions file (e.g., dev-v2.0.json)')

    # DB config
    parser.add_argument("--qdrant_host", type=str, default="localhost",
                        help="URL for Qdrant vector database")
    parser.add_argument("--qdrant_port", type=int, default=6333,
                        help="port for Qdrant vector database")
    parser.add_argument("--qdrant_in_memory", type=bool, default=False,
                        help="use qdrant in memory or not")

    # Chunker config
    parser.add_argument("--embedding_model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="embedding model name as in huggingface")
    parser.add_argument("--chunking_overlap", type=int, default=2,
                        help="overlap for chunking strategy (0,1,2,3...)")
    parser.add_argument("--chunk_size", type=int, default=500,
                        help="fixed character size to be included in chunk if fixed char chunker")
    parser.add_argument("--max_tokens", type=int, default=128,
                        help="token size if fixed token chunker")

    parser.add_argument("--k_retrieval", type=int,
                        nargs="+", default=[1, 3, 5, 10, 20],
                        help="Retrieving k-value (1,3,5,10,20)")
    parser.add_argument("--wiki_dir", type=str, default="../../../../dataset/reduced_fever_data/wiki")
    parser.add_argument("--output_file", type=str, default="../../../retrieval_evaluation_results.jsonl")
    parser.add_argument("--model_path", type=str, default="../../../../edu_segmenter_linear/best_model")
    parser.add_argument("--claim_file_path", type=str, default="../../../../dataset/reduced_fever_data/paper_dev.jsonl")
    parser.add_argument(
        "--chunker_type", type=lambda s: ChunkerType(s), choices=list(ChunkerType), default=ChunkerType.CUSTOM_EDU,
    )
    parser.add_argument("--long_sentence_threshold_for_custom_edu", type=int, default=60,
                        help="Long sentence threshold after which it would be broken into edu boundaries" )
    # Retrieval config
    parser.add_argument("--retrieval_strategy", type=lambda s: RetrievalStrategy(s),
                        choices=list(RetrievalStrategy),
                        default=RetrievalStrategy.TOP_K)
    parser.add_argument("--top_k", type=int, default=5, help="k value for top-k retrieval")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for retrieval")

    return parser.parse_args()


if __name__ == "__main__":
    # Configure
    args = parse_args()

    # Set paths based on dataset type
    if args.dataset == "squad":
        # For SQuAD: evidence file is the training data, questions are the dev/test set
        args.wiki_dir = args.squad_evidence

        # Update claim file to questions file if using defaults
        if args.claim_file_path == "../../../../dataset/reduced_fever_data/paper_dev.jsonl":
            args.claim_file_path = args.squad_questions

        if args.output_file == "../../../retrieval_evaluation_results.jsonl":
            args.output_file = "../../../retrieval_evaluation_results_squad.jsonl"

    db_config = VectorDBConfig(
        host=args.qdrant_host,
        port=args.qdrant_port,
        use_memory=args.qdrant_in_memory
    )

    shared_client = db_config.connect_to_qdrant() if args.qdrant_in_memory else None

    # Define chunker with dataset type
    chunker_type = args.chunker_type
    required_keys = CHUNKER_ARGS[chunker_type]
    chunker_kwargs = {
        key: getattr(args, key)
        for key in required_keys
        if getattr(args, key) is not None
    }
    chunker_kwargs['dataset_type'] = args.dataset
    print(f"Chunker config: {chunker_kwargs}")
    chunker = get_chunker(chunker_type, **chunker_kwargs)

    # Initialize evaluator
    evaluator = RetrieverEvaluator(
        claim_file_path=args.claim_file_path,
        embedding_model_name=args.embedding_model_name,
        chunker=chunker,
        db_config=db_config,
        wiki_dir=args.wiki_dir,
        output_file=args.output_file,
        k_values=args.k_retrieval,
        max_files=None,
        overlap=args.chunking_overlap,
        shared_client=shared_client,
        dataset_type=args.dataset,
    )

    retrieval_config = RetrievalConfig(
        strategy=args.retrieval_strategy,
        k=args.top_k if args.retrieval_strategy == RetrievalStrategy.TOP_K else None,
        threshold=args.threshold if args.retrieval_strategy == RetrievalStrategy.THRESHOLD else None
    )

    evaluator.run(build_db=True, retrieval_config=retrieval_config)