"""
EDU-based chunker with sliding window support for long sentences.
"""

from typing import List, Dict, Tuple
from pathlib import Path
import torch
import transformers
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from peft import AutoPeftModelForTokenClassification
from com.fever.rag.chunker.base_chunker import BaseChunker
from com.fever.rag.models.BERTWithMLPClassifier import BERTWithMLPClassifier
from com.fever.rag.utils.chunker_stats import ChunkerStatistics
from com.fever.rag.utils.data_helper import get_device


class CustomEDUChunker(BaseChunker):
    """
    EDU-based chunker with sliding window support for long sentences.

    Key improvements:
    - Handles sentences longer than 512 tokens using sliding windows
    - Aggregates predictions across windows using max pooling
    - Maintains comprehensive statistics tracking
    """

    def __init__(
        self,
        model_path: str,
        edus_per_chunk: int = 3,
        overlap: int = 1,
        max_length: int = 512,
        window_stride: int = 256,
        aggregation_method: str = 'max',
        **kwargs
    ):
        """
        Initialize the EDU chunker with sliding window support.

        Args:
            model_path: Path to the trained EDU segmentation model
            edus_per_chunk: Number of consecutive EDUs to combine per chunk
            overlap: Number of EDUs to overlap between chunks
            max_length: Maximum sequence length for BERT (default: 512)
            window_stride: Stride for sliding window (default: 256)
            aggregation_method: How to combine predictions ('max', 'avg', 'vote')
        """
        print("overlap:", overlap)
        print("edus_per_chunk:", edus_per_chunk)
        print("model_path:", model_path)
        print(f"max_length: {max_length}, window_stride: {window_stride}")
        print(f"aggregation_method: {aggregation_method}")

        super().__init__('edu_linear_head', model_path=model_path)
        self.model_path = Path(model_path)
        self.device = get_device()
        self.edus_per_chunk = max(1, edus_per_chunk)
        self.overlap = max(0, overlap)
        self.max_length = max_length
        self.window_stride = window_stride
        self.aggregation_method = aggregation_method
        self.boundary_count = 0

        # Initialize statistics tracker
        self.stats = ChunkerStatistics('custom_edu_chunker')

        # Auto-detect model type from config
        config_path = self.model_path / "chunker_config.json"
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                self.model_config = json.load(f)
            model_type = self.model_config.get("model_type", "linear_head")
        else:
            print("⚠️ No chunker_config.json found, assuming linear_head model")
            model_type = "linear_head"
            self.model_config = {"model_type": "linear_head"}

        print(f"Loading EDU model from: {self.model_path}")
        print(f"Detected model type: {model_type}")

        transformers.logging.set_verbosity_error()
        base_model_name = "bert-base-uncased"

        print(f"Loading tokenizer from base model: {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Load model based on type
        if model_type == "mlp_classifier":
            mlp_dims = self.model_config.get("mlp_hidden_dims", [256, 128])
            mlp_dropout = self.model_config.get("mlp_dropout", 0.3)

            # Determine base model name
            # Priority: chunker_config.json > adapter_config.json > default
            base_model_name = "bert-base-uncased"  # Default fallback

            # Check chunker_config.json first
            if "base_model_name" in self.model_config and self.model_config["base_model_name"]:
                base_model_name = self.model_config["base_model_name"]
                print(f"✓ Using base model from chunker_config.json: {base_model_name}")
            else:
                # Try adapter_config.json
                peft_config_path = self.model_path / "adapter_config.json"
                if peft_config_path.exists():
                    with open(peft_config_path, 'r') as f:
                        peft_config = json.load(f)
                    candidate = peft_config.get("base_model_name_or_path")

                    # Validate: not None/null and not empty string
                    if candidate and str(candidate).strip():
                        base_model_name = candidate
                        print(f"✓ Using base model from adapter_config.json: {base_model_name}")
                    else:
                        print(f"⚠️ base_model_name_or_path is null/empty in adapter_config.json")
                        print(f"   Using default: {base_model_name}")
                else:
                    print(f"⚠️ adapter_config.json not found, using default: {base_model_name}")

            print(f"Initializing BERTWithMLPClassifier with base model: {base_model_name}")
            base_model = BERTWithMLPClassifier(
                model_name=base_model_name,
                num_labels=2,
                mlp_hidden_dims=mlp_dims,
                mlp_dropout=mlp_dropout
            )

            from peft import PeftModel
            print(f"Loading PEFT adapters from: {self.model_path}")
            self.model = PeftModel.from_pretrained(
                base_model, str(self.model_path), local_files_only=True
            )
        else:
            self.model = AutoPeftModelForTokenClassification.from_pretrained(
                str(self.model_path), local_files_only=True
            )

        transformers.logging.set_verbosity_warning()

        self.model.to(self.device)
        self.model.eval()

        print(f"✓ EDU model loaded on {self.device}")
        print(f"✓ EDUs per chunk: {self.edus_per_chunk}")
        print(f"✓ Chunk overlap: {self.overlap} EDU(s)")
        print(f"✓ Sliding window: max_length={max_length}, stride={window_stride}")

        self.data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True
        )

    def predict_edu_boundaries_for_line(self, line_text: str) -> List[int]:
        """
        Predict EDU boundaries using sliding window for long sentences.

        Strategy:
        1. Tokenize the full sentence
        2. If <= max_length: process normally
        3. If > max_length: use sliding windows with stride
        4. Aggregate predictions across windows

        Returns:
            List of predictions (0 or 1) for each token
        """
        if not line_text or not line_text.strip():
            return []

        # Tokenize full sentence
        encoding = self.tokenizer(
            line_text,
            return_tensors="pt",
            truncation=False,
            padding=False,
            add_special_tokens=False
        )

        input_ids = encoding['input_ids'].squeeze()
        if input_ids.dim() == 0:
            # Single token - just process normally
            return self._predict_single_window(line_text)
        seq_length = len(input_ids)

        # If short enough, process normally
        if seq_length <= self.max_length - 2:  # Account for [CLS] and [SEP]
            return self._predict_single_window(line_text)

        # Use sliding window for long sequences
        return self._predict_with_sliding_window(input_ids, line_text)

    def _predict_with_sliding_window(
            self,
            input_ids: torch.Tensor,
            line_text: str
    ) -> List[int]:
        """
        Process long sequence using sliding windows.

        Args:
            input_ids: Full tokenized sequence (without special tokens)
            line_text: Original text for offset mapping

        Returns:
            Aggregated predictions for each token
        """
        seq_length = len(input_ids)
        # Effective window size (excluding special tokens)
        window_size = self.max_length - 2

        # Store predictions for each token position
        # Each position gets a list of predictions from different windows
        token_predictions = [[] for _ in range(seq_length)]

        # Slide window across sequence
        start = 0
        window_count = 0

        while start < seq_length:
            end = min(start + window_size, seq_length)
            window_ids = input_ids[start:end]

            # Add special tokens
            window_with_special = torch.cat([
                torch.tensor([self.tokenizer.cls_token_id]),
                window_ids,
                torch.tensor([self.tokenizer.sep_token_id])
            ]).unsqueeze(0).to(self.device)

            attention_mask = torch.ones_like(window_with_special).to(self.device)

            # Get predictions for this window
            with torch.no_grad():
                outputs = self.model(
                    input_ids=window_with_special,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=2).squeeze(0)  # Only remove batch dim!

            # Handle different prediction dimensions
            if predictions.dim() == 0:
                # Single scalar prediction (very rare edge case)
                window_preds = [int(predictions.item())]
            else:
                # Extract predictions (skip [CLS] and [SEP])
                pred_slice = predictions[1:-1]

                # Check if the slice resulted in a 0-d tensor
                if pred_slice.dim() == 0:
                    window_preds = [int(pred_slice.item())]
                elif pred_slice.numel() == 0:
                    # Empty tensor after slicing
                    window_preds = []
                else:
                    # Normal case: convert to list
                    window_preds = pred_slice.cpu().numpy().tolist()
                    # Ensure it's a list even if single element
                    if not isinstance(window_preds, list):
                        window_preds = [int(window_preds)]
                    else:
                        # Ensure all elements are ints
                        window_preds = [int(p) for p in window_preds]

            # Store predictions for corresponding token positions
            for i, pred in enumerate(window_preds):
                if start + i < seq_length:  # Safety check
                    token_predictions[start + i].append(int(pred))

            window_count += 1

            # Move window
            if end >= seq_length:
                break
            start += self.window_stride

        # Aggregate predictions across windows
        final_predictions = self._aggregate_predictions(token_predictions)

        print(f"  → Processed {window_count} windows for {seq_length} tokens")

        return final_predictions

    def _predict_single_window(self, text: str) -> List[int]:
        """Process a single window (short sentence)."""
        # Handle empty or whitespace-only text
        if not text or not text.strip():
            return []

        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)  # Shape: [batch, seq]

        # Remove batch dimension - use squeeze(0) to only remove first dim
        predictions = predictions.squeeze(0)  # Shape: [seq_len]

        # Handle 0-d tensor case (single prediction)
        if predictions.dim() == 0:
            return [int(predictions.item())]

        # Convert to numpy first, then to list for safety
        pred_array = predictions.cpu().numpy()
        pred_list = pred_array.tolist()

        # Ensure it's actually a list
        if not isinstance(pred_list, list):
            pred_list = [int(pred_list)]

        # Remove [CLS] and [SEP] tokens if we have enough predictions
        if len(pred_list) > 2:
            result = pred_list[1:-1]
            # Ensure all elements are Python ints, not numpy types
            return [int(x) for x in result]
        elif len(pred_list) == 2:
            # Just [CLS] and [SEP], no real content
            return []
        elif len(pred_list) == 1:
            # Single token (shouldn't happen but handle it)
            return [int(pred_list[0])]

        return []

    def _aggregate_predictions(
        self,
        token_predictions: List[List[int]]
    ) -> List[int]:
        """
        Aggregate predictions from multiple windows.

        Args:
            token_predictions: List where each element is predictions for that token
                              from different windows

        Returns:
            Single prediction per token
        """
        final_preds = []

        for preds in token_predictions:
            if not preds:
                final_preds.append(0)
                continue

            if self.aggregation_method == 'max':
                # Max pooling: predict boundary if ANY window predicts it
                final_preds.append(max(preds))

            elif self.aggregation_method == 'avg':
                # Average: predict boundary if average > 0.5
                avg = sum(preds) / len(preds)
                final_preds.append(1 if avg > 0.5 else 0)

            elif self.aggregation_method == 'vote':
                # Majority vote
                ones = sum(preds)
                zeros = len(preds) - ones
                final_preds.append(1 if ones > zeros else 0)
            else:
                # Default to max
                final_preds.append(max(preds))

        return final_preds

    def split_line_into_edus(
            self,
            line_text: str,
            predictions: List[int]
    ) -> List[str]:
        """
        Split a line into EDUs based on predictions.

        Returns:
            List of EDU strings from this sentence
        """
        if not predictions or not line_text.strip():
            return [line_text] if line_text.strip() else []

        encoding = self.tokenizer(
            line_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False
        )

        offsets = encoding['offset_mapping']

        # Handle 0-d tensor case - convert to list first
        if isinstance(offsets, torch.Tensor):
            if offsets.dim() == 0:
                # Single offset, shouldn't happen but handle it
                return [line_text]
            offsets = offsets.tolist()

        # Now offsets should be a list
        if len(offsets) != len(predictions):
            print(f"⚠️ Length mismatch: {len(offsets)} offsets vs {len(predictions)} predictions")
            return [line_text]

        # Find EDU boundaries (where prediction == 1)
        edu_starts = [0]
        for i, pred in enumerate(predictions):
            if pred == 1 and i > 0:
                edu_starts.append(offsets[i][0])

        edu_starts.append(len(line_text))

        # Extract EDUs
        edus = []
        for i in range(len(edu_starts) - 1):
            start_char = edu_starts[i]
            end_char = edu_starts[i + 1]
            edu_text = line_text[start_char:end_char].strip()
            if edu_text:
                edus.append(edu_text)

        return edus if edus else [line_text]

    def process_lines_to_edus(
        self,
        lines: List[Tuple[int, str]]
    ) -> List[Tuple[str, int]]:
        """
        Process all lines and extract EDUs with their sentence IDs.
        Records statistics for each sentence and EDU.

        Returns:
            List of (edu_text, sentence_id) tuples
        """
        all_edus = []

        for sent_id, line_text in lines:
            if not line_text or not line_text.strip():
                continue

            if len(line_text.strip()) == 1:
                # print(f"⚠️ Skipping single-character line at sentence {sent_id}: '{line_text.strip()}'")
                continue
            # Predict EDU boundaries (now handles long sentences)
            predictions = self.predict_edu_boundaries_for_line(line_text)

            if not predictions:
                all_edus.append((line_text.strip(), sent_id))
                self.stats.record_sentence(line_text.strip(), edu_count=1)
                self.stats.record_edu(line_text.strip())
                continue

            # Split line into EDUs
            edus = self.split_line_into_edus(line_text, predictions)

            # Count EDU boundaries
            safe_predictions = [int(p) if isinstance(p, torch.Tensor) else p for p in predictions]
            self.boundary_count += sum(safe_predictions)

            # Record sentence statistics
            self.stats.record_sentence(line_text, edu_count=len(edus))

            # Add all EDUs and record each
            for edu in edus:
                all_edus.append((edu, sent_id))
                self.stats.record_edu(edu)

        return all_edus

    def create_chunks_with_overlap(
        self,
        edus_with_ids: List[Tuple[str, int]]
    ) -> List[Tuple[str, List[int], int]]:
        """
        Combine consecutive EDUs into chunks with overlap.

        Returns:
            List of (chunk_text, sentence_ids, edu_count) tuples
        """
        if not edus_with_ids:
            return []

        chunks = []

        if self.overlap == 0:
            # No overlap: simple chunking
            i = 0
            while i < len(edus_with_ids):
                window_edus = edus_with_ids[i:i + self.edus_per_chunk]
                chunk_text = " ".join([edu_text for edu_text, _ in window_edus])
                sentence_ids = sorted(set([sent_id for _, sent_id in window_edus]))
                chunks.append((chunk_text, sentence_ids, len(window_edus)))
                i += self.edus_per_chunk
        else:
            # With overlap: sliding window chunking
            step_size = self.edus_per_chunk - self.overlap
            if step_size <= 0:
                step_size = 1

            i = 0
            while i < len(edus_with_ids):
                window_edus = edus_with_ids[i:i + self.edus_per_chunk]

                if not window_edus:
                    break

                chunk_text = " ".join([edu_text for edu_text, _ in window_edus])
                sentence_ids = sorted(set([sent_id for _, sent_id in window_edus]))
                chunks.append((chunk_text, sentence_ids, len(window_edus)))

                i += step_size

        return chunks

    def chunk(
            self,
            cleaned_text: str,
            annotated_lines: str,
            **kwargs
    ) -> List[Tuple[str, List[int]]]:
        """
        Chunk text using sentences as the primary unit.
        Only split abnormally long sentences using EDU boundaries.
        Records comprehensive statistics during processing.

        Returns:
            List of (chunk_text, sentence_ids) tuples
        """
        lines = self.parse_annotated_lines(annotated_lines)

        if not lines:
            return []

        self.stats.record_article()

        # Token length threshold for "long" sentences
        LONG_SENTENCE_THRESHOLD = 60  # tokens

        chunks = []

        for sent_id, line_text in enumerate(lines):
            if not line_text or not line_text.strip():
                continue

            # Skip single-character lines
            if len(line_text.strip()) == 1:
                continue

            # Calculate token count (approximate)
            token_count = len(line_text.split())

            # For normal-length sentences, keep as-is
            if token_count < LONG_SENTENCE_THRESHOLD:
                chunks.append((line_text.strip(), [sent_id]))

                # Record statistics
                self.stats.record_sentence(line_text, edu_count=1)
                self.stats.record_edu(line_text)
                self.stats.record_chunk(line_text.strip(), [sent_id], edu_count=1)
            else:
                # For long sentences, use EDU segmentation
                predictions = self.predict_edu_boundaries_for_line(line_text)

                if not predictions:
                    # If prediction fails, keep sentence intact
                    chunks.append((line_text.strip(), [sent_id]))
                    self.stats.record_sentence(line_text, edu_count=1)
                    self.stats.record_edu(line_text)
                    self.stats.record_chunk(line_text.strip(), [sent_id], edu_count=1)
                    continue

                # Split into EDUs
                edus = self.split_line_into_edus(line_text, predictions)

                # Count boundaries for this long sentence
                safe_predictions = [int(p) if isinstance(p, torch.Tensor) else p for p in predictions]
                self.boundary_count += sum(safe_predictions)

                # Record sentence statistics
                self.stats.record_sentence(line_text, edu_count=len(edus))

                # Each EDU becomes a separate chunk (all with same sentence_id)
                for edu in edus:
                    chunks.append((edu, [sent_id]))
                    self.stats.record_edu(edu)
                    self.stats.record_chunk(edu, [sent_id], edu_count=1)

        return chunks

    def get_metadata(
        self,
        article_id: str,
        chunk_index: int,
        chunk_text: str,
        sentence_ids: List[int] = None
    ) -> Dict:
        """Generate metadata for an EDU chunk."""
        sentence_ids = sentence_ids or []

        return {
            'article_id': article_id,
            'chunk_index': chunk_index,
            'sentence_ids': sentence_ids,
            'chunk_type': 'edu_linear_head',
            'chunk_size': len(chunk_text),
            'token_count': len(chunk_text.split()),
            'num_sentences': len(sentence_ids),
            'edus_per_chunk': self.edus_per_chunk,
            'overlap': self.overlap,
            'max_length': self.max_length,
            'window_stride': self.window_stride,
            'aggregation_method': self.aggregation_method,
            'model_path': str(self.model_path),
            'cleaned': bool(chunk_text.strip())
        }