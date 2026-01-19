"""
Simplified FEVER claim classifier for testing zero-shot and few-shot prompting.
Using Ollama Python library instead of REST API calls.
Supports both FEVER and SQuAD datasets.

For SQuAD: Generates answers and evaluates using Exact Match, F1, ROUGE-L, and BERTScore.
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re
from collections import Counter

import os
import time
import subprocess
import urllib.request

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import ollama
from com.fever.rag.retriever.retriever_config import VectorDBRetriever
from com.fever.rag.utils.data_helper import ClassificationMetrics, RetrievalConfig

# For SQuAD metrics
try:
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score
    SQUAD_METRICS_AVAILABLE = True
except ImportError:
    SQUAD_METRICS_AVAILABLE = False
    print("⚠️ Warning: rouge-score and/or bert-score not installed. Install with:")
    print("   pip install rouge-score bert-score")


def ensure_ollama_running() -> bool:
    """Check if Ollama is running, restart if needed (for Colab runtime)."""
    try:
        # Quick health check
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2)
        return True
    except Exception:
        print("Ollama server not responding, restarting...")

    # Kill any zombie processes (no error if none exist)
    subprocess.run(["pkill", "-f", "ollama"], check=False)
    time.sleep(2)

    # Restart Ollama server in the background
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setpgrp,  # detach from Python process
    )

    # Wait for it to come up
    for _ in range(20):
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1)
            print("  Ollama restarted successfully")
            return True
        except Exception:
            time.sleep(1)

    print("  Failed to restart Ollama")
    return False


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison (lowercase, remove articles, punctuation, extra whitespace)."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        import string
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, ground_truths: List[str]) -> float:
    """Compute exact match score (1.0 if prediction matches any ground truth)."""
    normalized_pred = normalize_answer(prediction)
    for gt in ground_truths:
        if normalized_pred == normalize_answer(gt):
            return 1.0
    return 0.0


def compute_f1(prediction: str, ground_truths: List[str]) -> float:
    """Compute F1 score (max F1 across all ground truths)."""
    def _f1_score(pred_tokens, gt_tokens):
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    normalized_pred = normalize_answer(prediction)
    pred_tokens = normalized_pred.split()

    f1_scores = []
    for gt in ground_truths:
        normalized_gt = normalize_answer(gt)
        gt_tokens = normalized_gt.split()
        f1_scores.append(_f1_score(pred_tokens, gt_tokens))

    return max(f1_scores) if f1_scores else 0.0


class FEVERClassifier:
    """
    Classifier/Generator for FEVER claims and SQuAD questions.

    - FEVER: Classification task (SUPPORTS/REFUTES/NOT ENOUGH INFO)
    - SQuAD: Answer generation task with EM, F1, ROUGE-L, BERTScore metrics

    Usage:
        # FEVER mode
        classifier = FEVERClassifier(
            model_name="gemma2:2b",
            few_shot_examples=5,
            dataset_type="fever"
        )
        metrics = classifier.evaluate("data/fever/dev.jsonl", max_claims=100)

        # SQuAD mode
        classifier = FEVERClassifier(
            model_name="gemma2:2b",
            few_shot_examples=5,
            dataset_type="squad"
        )
        metrics = classifier.evaluate("data/squad/dev-v2.0.json", max_claims=100)
    """

    FEVER_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

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
        max_evidence_chunks: int = 5,
        dataset_type: str = "fever"
    ):
        """
        Initialize the classifier/generator.

        Args:
            model_name: Name of the LLM to use
            few_shot_examples: Number of examples to include in prompt (0 for zero-shot)
            examples_file: Path to file with examples for few-shot
            temperature: Sampling temperature for the model
            retriever: VectorDBRetriever instance for evidence retrieval
            retrieval_config: Configuration for retrieval (strategy, k, threshold)
            collection_name: Name of the Qdrant collection
            embedding_model_name: Name of the embedding model for retrieval
            max_evidence_chunks: Maximum number of evidence chunks to include in prompt
            dataset_type: Either "fever" or "squad"
        """
        self.model_name = model_name
        self.few_shot_examples = few_shot_examples
        self.temperature = temperature
        self.dataset_type = dataset_type.lower()

        if self.dataset_type not in ["fever", "squad"]:
            raise ValueError(f"dataset_type must be 'fever' or 'squad', got '{dataset_type}'")

        # Set labels based on dataset type (only for FEVER)
        if self.dataset_type == "fever":
            self.LABELS = self.FEVER_LABELS

        # Initialize ROUGE scorer for SQuAD
        if self.dataset_type == "squad" and SQUAD_METRICS_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

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

        print(f"Initialized for {self.dataset_type.upper()} dataset")
        if self.dataset_type == "fever":
            print(f"Labels: {self.LABELS}")
        else:
            print("Task: Answer Generation (EM, F1, ROUGE-L, BERTScore)")

    def load_examples(self, file_path: str, n: int) -> List[Dict]:
        """Load examples for few-shot learning."""
        if self.dataset_type == "fever":
            return self._load_fever_examples(file_path, n)
        else:
            return self._load_squad_examples(file_path, n)

    def _load_fever_examples(self, file_path: str, n: int) -> List[Dict]:
        """Load FEVER examples (classification)."""
        examples_by_class = {label: [] for label in self.LABELS}

        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                label = data.get('label')

                if label in examples_by_class and len(examples_by_class[label]) < n:
                    examples_by_class[label].append(data)

                if all(len(examples) >= n for examples in examples_by_class.values()):
                    break

        examples = []
        for label in self.LABELS:
            examples.extend(examples_by_class[label])

        print(f"Loaded {len(examples)} FEVER examples:")
        for label in self.LABELS:
            count = len(examples_by_class[label])
            print(f"  {label}: {count} examples")

        return examples

    def _load_squad_examples(self, file_path: str, n: int) -> List[Dict]:
        """Load SQuAD examples (QA pairs)."""
        examples = []

        with open(file_path, 'r') as f:
            squad_data = json.load(f)

        for article in squad_data.get('data', []):
            for paragraph in article.get('paragraphs', []):
                context = paragraph.get('context', '')
                for qa in paragraph.get('qas', []):
                    if len(examples) >= n:
                        break

                    # Only use answerable questions for few-shot examples
                    if not qa.get('is_impossible', False) and qa.get('answers'):
                        examples.append({
                            'question': qa.get('question', ''),
                            'context': context,
                            'answer': qa['answers'][0]['text']
                        })
                if len(examples) >= n:
                    break
            if len(examples) >= n:
                break

        print(f"Loaded {len(examples)} SQuAD examples")
        return examples

    def retrieve_evidence(self, query: str) -> str:
        """Retrieve evidence chunks using VectorDBRetriever."""
        if not self.retriever:
            return "No evidence available."

        try:
            result = self.retriever.retrieve(
                claim=query,
                collection_name=self.collection_name,
                embedding_model_name=self.embedding_model_name,
                config=self.retrieval_config
            )

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

    def build_prompt(self, input_text: str, context: Optional[str] = None) -> str:
        """Build prompt based on dataset type."""
        if self.dataset_type == "fever":
            return self._build_fever_prompt(input_text)
        else:
            return self._build_squad_prompt(input_text, context)

    def _build_fever_prompt(self, claim: str) -> str:
        """Build FEVER classification prompt."""
        prompt = "Classify the following claim into one of these categories:\n"
        prompt += "- SUPPORTS: The claim is supported by evidence\n"
        prompt += "- REFUTES: The claim is refuted by evidence\n"
        prompt += "- NOT ENOUGH INFO: There is not enough information to verify\n\n"

        if self.examples:
            prompt += "Examples:\n\n"
            for ex in self.examples:
                prompt += f"Claim: {ex['claim']}\n"
                prompt += f"Label: {ex['label']}\n\n"

        if self.retriever:
            evidence = self.retrieve_evidence(claim)
            prompt += f"Claim: {claim}\n\n"
            prompt += f"Evidence:\n{evidence}\n\n"
        else:
            prompt += f"Claim: {claim}\n\n"

        prompt += "Label:"
        return prompt

    def _build_squad_prompt(self, question: str, context: Optional[str] = None) -> str:
        """Build SQuAD answer generation prompt."""
        prompt = "Answer the following question based on the given context. "
        prompt += "Provide a short, direct answer extracted from the context. "
        prompt += "If the answer cannot be found in the context, respond with 'unanswerable'.\n\n"

        if self.examples:
            prompt += "Examples:\n\n"
            for ex in self.examples:
                prompt += f"Context: {ex['context'][:200]}...\n"
                prompt += f"Question: {ex['question']}\n"
                prompt += f"Answer: {ex['answer']}\n\n"

        if context:
            prompt += f"Context: {context}\n"
            prompt += f"Question: {question}\n"
        elif self.retriever:
            evidence = self.retrieve_evidence(question)
            prompt += f"Context:\n{evidence}\n\n"
            prompt += f"Question: {question}\n"
        else:
            prompt += f"Question: {question}\n"

        prompt += "Answer:"
        return prompt

    def call_model(self, prompt: str) -> str:
        """Call LLM via Ollama with health checks."""
        ensure_ollama_running()

        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": self.temperature,
                    "keep_alive": "1h",
                },
            )
            return response["response"]
        except Exception as e:
            print(f"Ollama call failed: {e}. Trying to restart server...")
            if ensure_ollama_running():
                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        "temperature": self.temperature,
                        "keep_alive": "1h",
                    },
                )
                return response["response"]

            raise Exception(f"Ollama API call failed after restart attempt: {str(e)}")

    def _parse_fever_prediction(self, response: str) -> str:
        """Parse FEVER classification response."""
        response = response.strip().upper()

        for label in self.LABELS:
            if label in response:
                return label

        return "NOT ENOUGH INFO"

    def _parse_squad_answer(self, response: str) -> str:
        """Parse SQuAD answer from response."""
        # Clean up the response
        answer = response.strip()

        # Remove common prefixes
        prefixes = ["answer:", "the answer is:", "the answer is", "answer is"]
        for prefix in prefixes:
            if answer.lower().startswith(prefix):
                answer = answer[len(prefix):].strip()

        # Take only the first sentence/line for conciseness
        answer = answer.split('\n')[0].strip()

        return answer

    def predict_fever(self, claim: str) -> str:
        """Predict FEVER label."""
        prompt = self.build_prompt(claim)
        response = self.call_model(prompt)
        return self._parse_fever_prediction(response)

    def predict_squad(self, question: str, context: str) -> str:
        """Generate SQuAD answer."""
        prompt = self.build_prompt(question, context)
        response = self.call_model(prompt)
        return self._parse_squad_answer(response)

    def evaluate(
        self,
        file_path: str,
        max_claims: Optional[int] = None,
        output_file: Optional[str] = None
    ) -> Dict:
        """Evaluate on dataset."""
        if self.dataset_type == "fever":
            return self._evaluate_fever(file_path, max_claims, output_file)
        else:
            return self._evaluate_squad(file_path, max_claims, output_file)

    def _evaluate_fever(self, jsonl_path: str, max_claims: Optional[int], output_file: Optional[str]) -> ClassificationMetrics:
        """Evaluate FEVER classification."""
        true_labels = []
        pred_labels = []
        results = []

        print(f"Evaluating on {jsonl_path}")
        print(f"Dataset: FEVER")
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

                pred_label = self.predict_fever(claim)

                true_labels.append(true_label)
                pred_labels.append(pred_label)

                results.append({
                    'claim': claim,
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'correct': true_label == pred_label
                })

                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1} claims...")

        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, pred_labels, average='weighted', zero_division=0
        )

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

        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        print(f"\n🎯 Overall Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  Support:   {len(true_labels)}")

        self.print_confusion_matrix(true_labels, pred_labels)

        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump({
                    'config': {
                        'model': self.model_name,
                        'few_shot_examples': self.few_shot_examples,
                        'temperature': self.temperature,
                        'dataset_type': self.dataset_type
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

    def _evaluate_squad(self, json_path: str, max_claims: Optional[int], output_file: Optional[str]) -> Dict:
        """Evaluate SQuAD answer generation."""
        results = []
        exact_matches = []
        f1_scores = []
        rouge_scores = []

        predictions_for_bert = []
        references_for_bert = []

        print(f"Evaluating on {json_path}")
        print(f"Dataset: SQuAD")
        print(f"Mode: {'Few-shot' if self.few_shot_examples > 0 else 'Zero-shot'}")
        if self.few_shot_examples > 0:
            print(f"Examples: {self.few_shot_examples}")
        print()

        with open(json_path, 'r') as f:
            squad_data = json.load(f)

        count = 0
        for article in squad_data.get('data', []):
            title = article.get('title', '')
            for paragraph in article.get('paragraphs', []):
                context = paragraph.get('context', '')
                for qa in paragraph.get('qas', []):
                    if max_claims and count >= max_claims:
                        break

                    question = qa.get('question', '')
                    qa_id = qa.get('id', '')
                    is_impossible = qa.get('is_impossible', False)

                    # Get ground truth answers
                    ground_truths = [ans['text'] for ans in qa.get('answers', [])]

                    # Generate prediction
                    predicted_answer = self.predict_squad(question, context)

                    # Compute metrics
                    em = compute_exact_match(predicted_answer, ground_truths) if ground_truths else 0.0
                    f1 = compute_f1(predicted_answer, ground_truths) if ground_truths else 0.0

                    exact_matches.append(em)
                    f1_scores.append(f1)

                    # ROUGE-L (use first ground truth as reference)
                    if SQUAD_METRICS_AVAILABLE and ground_truths:
                        rouge_result = self.rouge_scorer.score(ground_truths[0], predicted_answer)
                        rouge_scores.append(rouge_result['rougeL'].fmeasure)

                    # Collect for BERTScore
                    if ground_truths:
                        predictions_for_bert.append(predicted_answer)
                        references_for_bert.append(ground_truths[0])

                    results.append({
                        'id': qa_id,
                        'title': title,
                        'question': question,
                        'context': context[:100] + '...',
                        'ground_truth': ground_truths,
                        'predicted_answer': predicted_answer,
                        'is_impossible': is_impossible,
                        'exact_match': em,
                        'f1': f1
                    })

                    count += 1
                    if count % 10 == 0:
                        print(f"Processed {count} questions...")

                if max_claims and count >= max_claims:
                    break
            if max_claims and count >= max_claims:
                break

        # Calculate aggregate metrics
        avg_em = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

        # Calculate BERTScore
        avg_bertscore = 0.0
        if SQUAD_METRICS_AVAILABLE and predictions_for_bert:
            print("\nCalculating BERTScore (this may take a moment)...")
            P, R, F1 = bert_score(predictions_for_bert, references_for_bert, lang='en', verbose=False)
            avg_bertscore = F1.mean().item()

        print("\n" + "=" * 70)
        print("EVALUATION RESULTS (SQuAD)")
        print("=" * 70)
        print(f"\n🎯 Answer Generation Metrics:")
        print(f"  Exact Match (EM):  {avg_em:.4f}")
        print(f"  F1 Score:          {avg_f1:.4f}")
        print(f"  ROUGE-L:           {avg_rouge:.4f}")
        print(f"  BERTScore:         {avg_bertscore:.4f}")
        print(f"  Total Questions:   {len(results)}")
        print("=" * 70 + "\n")

        metrics = {
            'exact_match': avg_em,
            'f1': avg_f1,
            'rouge_l': avg_rouge,
            'bert_score': avg_bertscore,
            'total': len(results)
        }

        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump({
                    'config': {
                        'model': self.model_name,
                        'few_shot_examples': self.few_shot_examples,
                        'temperature': self.temperature,
                        'dataset_type': self.dataset_type
                    },
                    'metrics': metrics,
                    'predictions': results
                }, f, indent=2)

        return metrics

    def print_confusion_matrix(self, y_true: List[str], y_pred: List[str]):
        """Print confusion matrix (FEVER only)."""
        labels = self.LABELS
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        print("\n" + "=" * 70)
        print("📊 CONFUSION MATRIX")
        print("=" * 70)

        header = "Actual \\ Predicted".ljust(20)
        for label in labels:
            header += label[:12].center(15)
        print(header)
        print("-" * 70)

        for i, label in enumerate(labels):
            row = label[:18].ljust(20)
            for j in range(len(labels)):
                row += str(cm[i][j]).center(15)
            print(row)

        print("-" * 70)

        print("\n📈 PER-CLASS METRICS (from Confusion Matrix)")
        print("-" * 70)
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 70)

        for i, label in enumerate(labels):
            tp = cm[i][i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            support = cm[i, :].sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"{label:<20} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<10}")

        print("=" * 70 + "\n")