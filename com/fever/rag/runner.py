"""
Integrated RAG Pipeline Runner
Combines retrieval evaluation with zero-shot/few-shot classification.
Supports both FEVER and SQuAD datasets.
"""
import argparse
from pathlib import Path
from typing import Optional

from com.fever.rag.chunker.base_chunker import BaseChunker
from com.fever.rag.generator.zero_shot_prompting import FEVERClassifier
from com.fever.rag.retriever.retriever_config import VectorDBRetriever
from com.fever.rag.retriever.retriever_evaluator import RetrieverEvaluator
from com.fever.rag.utils.chunker_helper import get_chunker, CHUNKER_ARGS, ChunkerType
from com.fever.rag.utils.data_helper import (
    VectorDBConfig,
    RetrievalConfig,
    RetrievalStrategy,
    ClassificationMetrics, get_collection_name
)

class IntegratedRAGPipeline:
    """
    Integrated pipeline that:
    1. Builds vector DB with specified chunking strategy
    2. Evaluates retrieval performance
    3. Runs zero-shot/few-shot generation using the same retrieval config

    Supports both FEVER (classification) and SQuAD (answer generation).
    """

    def __init__(
            self,
            # Database config
            db_config: VectorDBConfig,

            # Chunking config
            chunker: BaseChunker,
            embedding_model_name: str,

            # Data paths
            wiki_dir: str,
            claim_file_path: str,
            examples_file: Optional[str] = None,

            # Retrieval config
            retrieval_config: RetrievalConfig = None,

            # Classification/Generation config
            llm_model_name: str = "gemma2:2b",
            temperature: float = 0.0,
            few_shot_examples: int = 0,
            max_evidence_chunks: int = 5,

            # Evaluation config
            k_values: list = None,
            max_claims: Optional[int] = None,

            # Output
            output_dir: str = "results",
            overlap: Optional[int] = None,

            # Dataset type
            dataset_type: str = "fever",
    ):
        """
        Initialize the integrated RAG pipeline.

        Args:
            db_config: Vector database configuration
            chunker: Chunking strategy
            embedding_model_name: Embedding model for retrieval
            wiki_dir: Directory containing evidence files (JSONL for FEVER, JSON for SQuAD)
            claim_file_path: Path to queries file (JSONL for FEVER, JSON for SQuAD)
            examples_file: Path to training examples for few-shot
            retrieval_config: Retrieval configuration
            llm_model_name: LLM for generation
            temperature: Sampling temperature
            few_shot_examples: Number of examples per class (FEVER) or total (SQuAD)
            max_evidence_chunks: Max evidence chunks to include
            k_values: K values for retrieval evaluation
            max_claims: Maximum queries to evaluate
            output_dir: Directory for output files
            overlap: Overlap for chunking
            dataset_type: Either "fever" or "squad"
        """
        self.db_config = db_config
        self.chunker = chunker
        self.embedding_model_name = embedding_model_name
        self.wiki_dir = wiki_dir
        self.claim_file_path = claim_file_path
        self.examples_file = examples_file
        self.llm_model_name = llm_model_name
        self.temperature = temperature
        self.few_shot_examples = few_shot_examples
        self.max_evidence_chunks = max_evidence_chunks
        self.k_values = k_values or [1, 3, 5, 10, 20]
        self.max_claims = max_claims
        self.output_dir = Path(output_dir)
        self.overlap = overlap
        self.dataset_type = dataset_type.lower()

        if self.dataset_type not in ["fever", "squad"]:
            raise ValueError(f"dataset_type must be 'fever' or 'squad', got '{dataset_type}'")

        # Set default retrieval config
        self.retrieval_config = retrieval_config or RetrievalConfig(
            strategy=RetrievalStrategy.TOP_K,
            k=5
        )

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Shared Qdrant client for in-memory mode
        self.shared_client = db_config.connect_to_qdrant() if db_config.use_memory else None

        # Initialize components
        self.retriever_evaluator = None
        self.classifier = None
        self.collection_name = get_collection_name(self.embedding_model_name, self.chunker)

    def build_vector_db(self):
        """Step 1: Build the vector database."""
        print("\n" + "=" * 80)
        print(f"STEP 1: BUILDING VECTOR DATABASE ({self.dataset_type.upper()})")
        print("=" * 80)

        self.retriever_evaluator = RetrieverEvaluator(
            claim_file_path=self.claim_file_path,
            embedding_model_name=self.embedding_model_name,
            chunker=self.chunker,
            db_config=self.db_config,
            wiki_dir=self.wiki_dir,
            output_file=str(self.output_dir / f"retrieval_evaluation_{self.dataset_type}.jsonl"),
            k_values=self.k_values,
            overlap=self.overlap,
            shared_client=self.shared_client,
            dataset_type=self.dataset_type,
        )

        self.collection_name = self.retriever_evaluator.collection_name
        print(f"Collection name: {self.collection_name}")

        # Build the database
        self.retriever_evaluator.build_vector_db(reset=True)

        print("✓ Vector database built successfully")

    def evaluate_retrieval(self):
        """Step 2: Evaluate retrieval performance."""
        print("\n" + "=" * 80)
        print(f"STEP 2: EVALUATING RETRIEVAL PERFORMANCE ({self.dataset_type.upper()})")
        print("=" * 80)

        if self.retriever_evaluator is None:
            raise RuntimeError("Must run build_vector_db first")

        # Evaluate retrieval
        retrieval_metrics = self.retriever_evaluator.evaluate(self.retrieval_config)

        # Print metrics
        RetrieverEvaluator.print_metrics(retrieval_metrics)

        # Save metrics
        self.retriever_evaluator.save_results(retrieval_metrics, self.retrieval_config)

        print("✓ Retrieval evaluation completed")
        return retrieval_metrics

    def run_generation(self):
        """Step 3: Run generation (classification for FEVER, answer generation for SQuAD)."""
        task_name = "CLASSIFICATION" if self.dataset_type == "fever" else "ANSWER GENERATION"
        print("\n" + "=" * 80)
        print(f"STEP 3: {task_name} WITH RAG ({self.dataset_type.upper()})")
        print("=" * 80)

        if self.collection_name is None:
            raise RuntimeError("Must run build_vector_db first")

        # Initialize retriever for generation
        retriever = VectorDBRetriever(
            db_config=self.db_config,
            shared_client=self.shared_client
        )

        # Initialize classifier/generator
        self.classifier = FEVERClassifier(
            model_name=self.llm_model_name,
            few_shot_examples=self.few_shot_examples,
            examples_file=self.examples_file,
            temperature=self.temperature,
            retriever=retriever,
            retrieval_config=self.retrieval_config,
            collection_name=self.collection_name,
            embedding_model_name=self.embedding_model_name,
            max_evidence_chunks=self.max_evidence_chunks,
            dataset_type=self.dataset_type,
        )

        # Run evaluation
        generation_mode = "few-shot" if self.few_shot_examples > 0 else "zero-shot"
        output_file = self.output_dir / f"{generation_mode}_{self.dataset_type}_results.json"

        generation_metrics = self.classifier.evaluate(
            file_path=self.claim_file_path,
            max_claims=self.max_claims,
            output_file=str(output_file)
        )

        print(f"✓ {task_name.capitalize()} completed")
        return generation_metrics

    def run_full_pipeline(self):
        """Run the complete pipeline: build DB -> evaluate retrieval -> generate."""
        print("\n" + "=" * 80)
        print(f"INTEGRATED RAG PIPELINE ({self.dataset_type.upper()})")
        print("=" * 80)
        print(f"Dataset: {self.dataset_type.upper()}")
        print(f"Embedding Model: {self.embedding_model_name}")
        print(f"Chunker: {self.chunker.name}")
        print(f"LLM: {self.llm_model_name}")
        print(f"Retrieval Strategy: {self.retrieval_config.strategy.value}")
        print(f"Generation Mode: {'Few-shot' if self.few_shot_examples > 0 else 'Zero-shot'}")
        print("=" * 80)

        # Step 1: Build vector DB
        self.build_vector_db()

        # Step 2: Evaluate retrieval
        retrieval_metrics = self.evaluate_retrieval()

        # Step 3: Run generation
        generation_metrics = self.run_generation()

        # Print final summary
        self._print_final_summary(retrieval_metrics, generation_metrics)

        return {
            'retrieval_metrics': retrieval_metrics,
            'generation_metrics': generation_metrics
        }

    def _print_final_summary(self, retrieval_metrics, generation_metrics):
        """Print final summary of the pipeline."""
        print("\n" + "=" * 80)
        print("PIPELINE SUMMARY")
        print("=" * 80)

        print("\n📊 Retrieval Performance:")
        print(f"  Mean Reciprocal Rank: {retrieval_metrics.mean_reciprocal_rank:.4f}")
        print(f"  Recall@5: {retrieval_metrics.recall_at_k.get(5, 0):.4f}")
        print(f"  Precision@5: {retrieval_metrics.precision_at_k.get(5, 0):.4f}")

        if self.dataset_type == "fever":
            # FEVER: Classification metrics
            print("\n🎯 Classification Performance:")
            print(f"  Accuracy: {generation_metrics.accuracy:.4f}")
            print(f"  F1 Score: {generation_metrics.f1:.4f}")
            print(f"  Precision: {generation_metrics.precision:.4f}")
            print(f"  Recall: {generation_metrics.recall:.4f}")
        else:
            # SQuAD: Answer generation metrics
            print("\n🎯 Answer Generation Performance:")
            print(f"  Exact Match: {generation_metrics.get('exact_match', 0):.4f}")
            print(f"  F1 Score: {generation_metrics.get('f1', 0):.4f}")
            print(f"  ROUGE-L: {generation_metrics.get('rouge_l', 0):.4f}")
            print(f"  BERTScore: {generation_metrics.get('bert_score', 0):.4f}")

        print("\n📁 Results saved to:")
        print(f"  {self.output_dir.absolute()}")
        print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Integrated RAG Pipeline: Retrieval Evaluation + Generation (FEVER/SQuAD)"
    )

    # Dataset selection
    parser.add_argument("--dataset_type", type=str, default="fever",
                        choices=["fever", "squad"],
                        help="Dataset type: 'fever' or 'squad'")

    # Database config
    parser.add_argument("--qdrant_host", type=str, default="localhost")
    parser.add_argument("--qdrant_port", type=int, default=6333)
    parser.add_argument("--qdrant_in_memory", action="store_true")

    # Chunker config
    parser.add_argument("--embedding_model_name", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--chunking_overlap", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--chunker_type", type=lambda s: ChunkerType(s),
                        choices=list(ChunkerType), default=ChunkerType.SENTENCE)
    parser.add_argument("--long_sentence_threshold_for_custom_edu", type=int, default=50)

    # Data paths - FEVER defaults
    parser.add_argument("--wiki_dir", type=str,
                        default="../../../../dataset/reduced_fever_data/wiki",
                        help="Evidence directory (FEVER: wiki JSONL, SQuAD: train JSON)")
    parser.add_argument("--claim_file_path", type=str,
                        default="../../../../dataset/reduced_fever_data/paper_dev.jsonl",
                        help="Query file (FEVER: claims JSONL, SQuAD: questions JSON)")
    parser.add_argument("--examples_file", type=str,
                        default="../../../../dataset/reduced_fever_data/train.jsonl",
                        help="Few-shot examples file")
    parser.add_argument("--model_path", type=str,
                        default="../../../../edu_segmenter_linear/best_model")

    # SQuAD-specific paths (override defaults when --dataset_type=squad)
    parser.add_argument("--squad_evidence", type=str,
                        default="../../../../dataset/squad/train-v2.0.json",
                        help="SQuAD evidence file (train-v2.0.json)")
    parser.add_argument("--squad_questions", type=str,
                        default="../../../../dataset/squad/dev-v2.0.json",
                        help="SQuAD questions file (dev-v2.0.json)")

    # Retrieval config
    parser.add_argument("--retrieval_strategy", type=lambda s: RetrievalStrategy(s),
                        choices=list(RetrievalStrategy), default=RetrievalStrategy.TOP_K)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--k_retrieval", type=int, nargs="+", default=[1, 3, 5, 10, 20])

    # Generation config
    parser.add_argument("--llm_name", type=str, default="gemma2:2b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--few_shot_examples", type=int, default=0,
                        help="Number of examples per class (FEVER) or total (SQuAD)")
    parser.add_argument("--max_evidence_chunks", type=int, default=5)
    parser.add_argument("--max_claims", type=int, default=None)

    # Output
    parser.add_argument("--output_dir", type=str, default="results")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Override paths if using SQuAD
    if args.dataset_type == "squad":
        args.wiki_dir = args.squad_evidence
        args.claim_file_path = args.squad_questions
        # Use SQuAD train data for few-shot examples
        if args.few_shot_examples > 0 and args.examples_file == "../../../../dataset/reduced_fever_data/train.jsonl":
            args.examples_file = args.squad_evidence
        # Update output directory
        if args.output_dir == "results":
            args.output_dir = "results_squad"

    # Setup database config
    db_config = VectorDBConfig(
        host=args.qdrant_host,
        port=args.qdrant_port,
        use_memory=args.qdrant_in_memory
    )

    # Setup chunker with dataset type
    chunker_type = args.chunker_type
    required_keys = CHUNKER_ARGS[chunker_type]
    chunker_kwargs = {
        key: getattr(args, key)
        for key in required_keys
        if getattr(args, key) is not None
    }
    # Add dataset_type to chunker config
    chunker_kwargs['dataset_type'] = args.dataset_type
    chunker = get_chunker(chunker_type, **chunker_kwargs)

    # Setup retrieval config
    retrieval_config = RetrievalConfig(
        strategy=args.retrieval_strategy,
        k=args.top_k if args.retrieval_strategy == RetrievalStrategy.TOP_K else None,
        threshold=args.threshold if args.retrieval_strategy == RetrievalStrategy.THRESHOLD else None
    )

    # Initialize pipeline
    pipeline = IntegratedRAGPipeline(
        db_config=db_config,
        chunker=chunker,
        embedding_model_name=args.embedding_model_name,
        wiki_dir=args.wiki_dir,
        claim_file_path=args.claim_file_path,
        examples_file=args.examples_file if args.few_shot_examples > 0 else None,
        retrieval_config=retrieval_config,
        llm_model_name=args.llm_name,
        temperature=args.temperature,
        few_shot_examples=args.few_shot_examples,
        max_evidence_chunks=args.max_evidence_chunks,
        k_values=args.k_retrieval,
        max_claims=args.max_claims,
        output_dir=args.output_dir,
        overlap=args.chunking_overlap,
        dataset_type=args.dataset_type,
    )

    # Run the full pipeline
    results = pipeline.run_full_pipeline()
    print("\nDone!")