"""
Simplified FEVER claim classifier for testing zero-shot and few-shot prompting.
Using Ollama Python library instead of REST API calls.
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import ollama
from com.fever.rag.retriever.retriever_config import VectorDBRetriever
from com.fever.rag.utils.data_helper import ClassificationMetrics, RetrievalConfig


class FEVERClassifier:
    """
    Simple classifier for FEVER claims supporting zero-shot and few-shot prompting.

    Usage:
        classifier = FEVERClassifier(
            model_name="gemma2:2b",
            few_shot_examples=5
        )
        metrics = classifier.evaluate("data/fever/dev.jsonl", max_claims=100)
    """

    LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

    def __init__(
        self,
        model_name: str = "gemma2:2b",
        few_shot_examples: int = 0,
        examples_file: Optional[str] = None,
        temperature: float = 0.0,
        retriever: VectorDBRetriever = None,
        retrieval_config: Optional[RetrievalConfig] = None,
        collection_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        max_evidence_chunks: int = 5
    ):
        """
        Initialize the classifier.

        Args:
            model_name: Name of the LLM to use
            few_shot_examples: Number of examples to include in prompt (0 for zero-shot)
            examples_file: Path to JSONL file with examples for few-shot
            temperature: Sampling temperature for the model
            retriever: VectorDBRetriever instance for evidence retrieval
            retrieval_config: Configuration for retrieval (strategy, k, threshold)
            collection_name: Name of the Qdrant collection
            embedding_model_name: Name of the embedding model for retrieval
            max_evidence_chunks: Maximum number of evidence chunks to include in prompt
        """
        self.model_name = model_name
        self.few_shot_examples = few_shot_examples
        self.temperature = temperature

        # Retrieval components
        self.retriever = retriever
        self.retrieval_config = retrieval_config
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.max_evidence_chunks = max_evidence_chunks

        # Validate retrieval setup
        if retriever is not None:
            if retrieval_config is None or collection_name is None or embedding_model_name is None:
                raise ValueError(
                    "If retriever is provided, retrieval_config, collection_name, "
                    "and embedding_model_name must also be provided"
                )

        self.examples = []
        if few_shot_examples > 0:
            if examples_file is None:
                raise ValueError("examples_file required for few-shot prompting")
            self.examples = self.load_examples(examples_file, few_shot_examples)

    def load_examples(self, file_path: str, n: int) -> List[Dict]:
        """
        Load n examples per class from JSONL file for balanced few-shot learning.

        Args:
            file_path: Path to JSONL file with labeled examples
            n: Number of examples to load per class

        Returns:
            List of examples with n examples from each class (SUPPORTS, REFUTES, NOT ENOUGH INFO)
        """
        examples_by_class = {label: [] for label in self.LABELS}

        # Read all lines and group by label
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                label = data.get('label')

                # Only add if we haven't reached n for this class
                if label in examples_by_class and len(examples_by_class[label]) < n:
                    examples_by_class[label].append(data)

                # Stop if we have n examples for all classes
                if all(len(examples) >= n for examples in examples_by_class.values()):
                    break

        # Combine all examples in order: SUPPORTS, REFUTES, NOT ENOUGH INFO
        examples = []
        for label in self.LABELS:
            examples.extend(examples_by_class[label])

        # Print distribution for verification
        print(f"Loaded {len(examples)} examples:")
        for label in self.LABELS:
            count = len(examples_by_class[label])
            print(f"  {label}: {count} examples")

        return examples

    def retrieve_evidence(self, claim: str) -> str:
        """
        Retrieve evidence chunks for the claim using the VectorDBRetriever.

        Args:
            claim: The claim text to retrieve evidence for

        Returns:
            Formatted string containing evidence chunks
        """
        if not self.retriever:
            return "No evidence available."

        try:
            # Use the retrieve method from VectorDBRetriever
            result = self.retriever.retrieve(
                claim=claim,
                collection_name=self.collection_name,
                embedding_model_name=self.embedding_model_name,
                config=self.retrieval_config
            )

            # Format evidence chunks
            if not result.chunks:
                return "No evidence found."

            evidence_texts = []
            for i, chunk in enumerate(result.chunks[:self.max_evidence_chunks], 1):
                article_id = chunk.payload.get('article_id', 'Unknown')
                text = chunk.payload.get('text', '')
                score = chunk.score

                evidence_texts.append(
                    f"[Evidence {i}] (Source: {article_id}, Relevance: {score:.3f})\n{text}"
                )

            return "\n\n".join(evidence_texts)

        except Exception as e:
            print(f"Warning: Evidence retrieval failed: {e}")
            return "Evidence retrieval failed."

    def build_prompt(self, claim: str) -> str:
        """Build prompt for classification."""
        prompt = "Classify the following claim into one of these categories:\n"
        prompt += "- SUPPORTS: The claim is supported by evidence\n"
        prompt += "- REFUTES: The claim is refuted by evidence\n"
        prompt += "- NOT ENOUGH INFO: There is not enough information to verify\n\n"

        # Add few-shot examples
        if self.examples:
            prompt += "Examples:\n\n"
            for ex in self.examples:
                prompt += f"Claim: {ex['claim']}\n"
                prompt += f"Label: {ex['label']}\n\n"

        # Retrieve and add evidence if retriever is available
        if self.retriever:
            evidence = self.retrieve_evidence(claim)
            prompt += f"Claim: {claim}\n\n"
            prompt += f"Evidence:\n{evidence}\n\n"
        else:
            prompt += f"Claim: {claim}\n\n"

        prompt += "Label:"

        return prompt

    def call_model(self, prompt: str) -> str:
        """
        Call the LLM model via Ollama Python library.
        """
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': self.temperature
                }
            )
            return response['response']
        except Exception as e:
            raise Exception(f"Ollama API call failed: {str(e)}")

    def _parse_prediction(self, response: str) -> str:
        """Parse model response to extract label."""
        response = response.strip().upper()

        # Try exact match first
        for label in self.LABELS:
            if label in response:
                return label

        # Default to NOT ENOUGH INFO if unclear
        return "NOT ENOUGH INFO"

    def predict(self, claim: str) -> str:
        """Predict label for a single claim."""
        prompt = self.build_prompt(claim)
        response = self.call_model(prompt)
        return self._parse_prediction(response)

    def evaluate(
        self,
        jsonl_path: str,
        max_claims: Optional[int] = None,
        output_file: Optional[str] = None
    ) -> ClassificationMetrics:
        """
        Evaluate classifier on FEVER dataset.

        Args:
            jsonl_path: Path to JSONL file with claims
            max_claims: Maximum number of claims to evaluate (None for all)
            output_file: Optional path to save predictions

        Returns:
            ClassificationMetrics with evaluation results
        """
        true_labels = []
        pred_labels = []
        results = []

        print(f"Evaluating on {jsonl_path}")
        print(f"Mode: {'Few-shot' if self.few_shot_examples > 0 else 'Zero-shot'}")
        if self.few_shot_examples > 0:
            print(f"Examples: {self.few_shot_examples}")
        print()

        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if max_claims and i >= max_claims:
                    break

                data = json.loads(line)
                claim = data['claim']
                true_label = data['label']

                # Predict
                pred_label = self.predict(claim)

                true_labels.append(true_label)
                pred_labels.append(pred_label)

                # Store result
                results.append({
                    'claim': claim,
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'correct': true_label == pred_label
                })

                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1} claims...")

        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, pred_labels, average='weighted', zero_division=0
        )

        # Per-class metrics
        report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
        per_class = {label: report[label] for label in self.LABELS if label in report}

        metrics = ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            support=len(true_labels),
            per_class_metrics=per_class
        )

        # Save results if requested
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump({
                    'config': {
                        'model': self.model_name,
                        'few_shot_examples': self.few_shot_examples,
                        'temperature': self.temperature
                    },
                    'metrics': {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'support': len(true_labels)
                    },
                    'per_class': per_class,
                    'predictions': results
                }, f, indent=2)

        return metrics
