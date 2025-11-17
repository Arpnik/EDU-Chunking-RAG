"""
Simplified FEVER claim classifier for testing zero-shot and few-shot prompting.
"""
import json
from pathlib import Path
from typing import List, Dict, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import requests

from com.fever.rag.retriever.retriever_config import VectorDBRetriever
from com.fever.rag.utils.DataHelper import ClassificationMetrics


class FEVERClassifier:
    """
    Simple classifier for FEVER claims supporting zero-shot and few-shot prompting.

    Usage:
        classifier = FEVERClassifier(
            model_name="gpt-4",
            few_shot_examples=5
        )
        metrics = classifier.evaluate("data/fever/dev.jsonl", max_claims=100)
    """

    LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

    def __init__(
        self,
        model_name: str = "gemma2:2b",
        model_path: str = "http://localhost:11434/api/generate",
        few_shot_examples: int = 0,
        examples_file: Optional[str] = None,
        temperature: float = 0.0,
        retriever: VectorDBRetriever = None
    ):
        """
        Initialize the classifier.

        Args:
            model_name: Name of the LLM to use
            few_shot_examples: Number of examples to include in prompt (0 for zero-shot)
            examples_file: Path to JSONL file with examples for few-shot
            temperature: Sampling temperature for the model
        """
        self.model_name = model_name
        self.few_shot_examples = few_shot_examples
        self.temperature = temperature
        self.model_path = model_path
        self.retriever = retriever
        self.examples = []

        if few_shot_examples > 0:
            if examples_file is None:
                raise ValueError("examples_file required for few-shot prompting")
            self.examples = self._load_examples(examples_file, few_shot_examples)

    def _load_examples(self, file_path: str, n: int) -> List[Dict]:
        """Load n examples from JSONL file."""
        examples = []
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                examples.append(json.loads(line))
        return examples

    def _build_prompt(self, claim: str) -> str:
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

        evidence = self.retriever(claim)
        # Add query claim
        prompt += f"Claim: {claim}\n"
        prompt += f"Evidence: {evidence}\n"
        prompt += "Label:"

        return prompt

    def _call_model(self, prompt: str) -> str:
        """
        Call the LLM model via Ollama API.
        """
        response = requests.post(
            self.model_path,
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "temperature": self.temperature
            }
        )

        if response.status_code == 200:
            return response.json()['response']
        else:
            raise Exception(f"API call failed with status {response.status_code}: {response.text}")

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
        prompt = self._build_prompt(claim)
        response = self._call_model(prompt)
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


# Example usage
if __name__ == "__main__":
    # Zero-shot evaluation
    print("Zero-shot Classification")
    print("=" * 70)
    classifier_zero = FEVERClassifier(
        model_name="gemma2:2b",
        model_path="http://localhost:11434/api/generate",
        few_shot_examples=0
    )
    metrics_zero = classifier_zero.evaluate(
        "../../../../dataset/reduced_fever_data/paper_dev.jsonl",
        max_claims=100,
        output_file="results/zero_shot.json"
    )
    print(metrics_zero)
    print()
    #
    # # Few-shot evaluation
    # print("Few-shot Classification (5 examples)")
    # print("=" * 70)
    # classifier_few = FEVERClassifier(
    #     model_name="gemma2:2b",
    #     model_path="http://localhost:11434/api/generate",
    #     few_shot_examples=5,
    #     examples_file="data/fever/train.jsonl"
    # )
    # metrics_few = classifier_few.evaluate(
    #     "data/fever/dev.jsonl",
    #     max_claims=100,
    #     output_file="results/few_shot.json"
    # )
    # print(metrics_few)