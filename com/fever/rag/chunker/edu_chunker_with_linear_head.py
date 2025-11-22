"""
EDU-based chunker using a fine-tuned BERT model with linear head for token classification.

This chunker processes individual lines (sentences) from FEVER dataset and creates chunks
based on predicted EDU boundaries with configurable overlap between chunks.
"""

from typing import List, Dict, Tuple
from pathlib import Path
import torch
import transformers
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from peft import AutoPeftModelForTokenClassification
from com.fever.rag.chunker.base_chunker import BaseChunker
from com.fever.rag.utils.data_helper import get_device


class EDUChunkerWithLinearHead(BaseChunker):
    """
    EDU-based chunker using a fine-tuned token classification model.

    The model predicts labels for each token:
    - 0 = EDU Continue (token is within current EDU)
    - 1 = EDU Start (token begins a new EDU)

    Processes sentences line-by-line and combines EDUs into chunks with overlap control.
    """

    def __init__(self, model_path: str, overlap: int = 0):
        """
        Initialize the EDU chunker.

        Args:
            model_path: Path to the trained PEFT model directory (contains adapter + tokenizer)
            overlap: Number of overlapping sentences between consecutive chunks (default: 0)
                    - 0: No overlap, chunks are completely separate
                    - 1: Adjacent chunks share 1 sentence
                    - 2+: Adjacent chunks share multiple sentences
        """
        super().__init__('edu_linear_head', model_path=model_path)

        self.model_path = Path(model_path)
        self.device = get_device()
        self.overlap = max(0, overlap)  # Ensure non-negative

        # Load PEFT model and tokenizer
        print(f"Loading EDU PEFT model from: {self.model_path}")

        transformers.logging.set_verbosity_error()
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoPeftModelForTokenClassification.from_pretrained(str(self.model_path))
        transformers.logging.set_verbosity_warning()

        self.model.to(self.device)
        self.model.eval()
        print(f"✓ EDU PEFT model loaded on {self.device}")
        print(f"✓ Chunk overlap: {self.overlap} sentence(s)")

        # Data collator for batch processing
        self.data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True
        )

    def predict_edu_boundaries_for_line(self, line_text: str) -> List[int]:
        """
        Predict EDU boundaries for a single line (sentence).

        Args:
            line_text: Text of a single sentence

        Returns:
            List of predictions (0 or 1) for each token in the sentence
        """
        if not line_text or not line_text.strip():
            return []

        # Tokenize the line
        encoding = self.tokenizer(
            line_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2).squeeze()

        # Extract predictions (skip [CLS] and [SEP] tokens)
        if predictions.dim() == 0:  # Single token case
            return [predictions.item()]

        pred_list = predictions.cpu().numpy().tolist()

        # Remove special tokens ([CLS] at start, [SEP] at end)
        if len(pred_list) > 2:
            return pred_list[1:-1]

        return []

    def split_line_into_edus(
        self,
        line_text: str,
        predictions: List[int]
    ) -> List[str]:
        """
        Split a line into EDUs based on predictions.

        Args:
            line_text: Original sentence text
            predictions: List of boundary predictions (0=continue, 1=start)

        Returns:
            List of EDU strings from this sentence
        """
        if not predictions or not line_text.strip():
            return [line_text] if line_text.strip() else []

        # Tokenize to get word-level alignment
        encoding = self.tokenizer(
            line_text,
            return_offsets_mapping=True,
            add_special_tokens=False
        )

        offsets = encoding['offset_mapping']

        # Align predictions with character offsets
        if len(offsets) != len(predictions):
            # Fallback: return whole line as single EDU
            return [line_text]

        # Find EDU boundaries (where prediction == 1)
        edu_starts = [0]  # First EDU always starts at position 0
        for i, pred in enumerate(predictions):
            if pred == 1 and i > 0:  # New EDU starts here
                edu_starts.append(offsets[i][0])  # Character position

        # Add end position
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

        Args:
            lines: List of (sentence_id, sentence_text) tuples

        Returns:
            List of (edu_text, sentence_id) tuples
        """
        all_edus = []

        for sent_id, line_text in lines:
            if not line_text or not line_text.strip():
                continue

            # Predict EDU boundaries for this line
            predictions = self.predict_edu_boundaries_for_line(line_text)

            if not predictions:
                # No predictions, treat whole line as single EDU
                all_edus.append((line_text.strip(), sent_id))
                continue

            # Split line into EDUs
            edus = self.split_line_into_edus(line_text, predictions)

            # Add all EDUs from this sentence
            for edu in edus:
                all_edus.append((edu, sent_id))

        return all_edus

    def create_chunks_with_overlap(
        self,
        edus_with_ids: List[Tuple[str, int]]
    ) -> List[Tuple[str, List[int]]]:
        """
        Combine EDUs into chunks with configurable sentence-level overlap.

        Args:
            edus_with_ids: List of (edu_text, sentence_id) tuples

        Returns:
            List of (chunk_text, sentence_ids) tuples
        """
        if not edus_with_ids:
            return []

        # Group EDUs by sentence ID
        sentence_edus = {}  # {sentence_id: [edu_texts]}
        for edu_text, sent_id in edus_with_ids:
            if sent_id not in sentence_edus:
                sentence_edus[sent_id] = []
            sentence_edus[sent_id].append(edu_text)

        # Get ordered sentence IDs
        sentence_ids = sorted(sentence_edus.keys())

        if self.overlap == 0:
            # No overlap: each sentence becomes its own chunk
            chunks = []
            for sent_id in sentence_ids:
                chunk_text = " ".join(sentence_edus[sent_id])
                chunks.append((chunk_text, [sent_id]))
            return chunks

        # With overlap: sliding window over sentences
        chunks = []
        i = 0

        while i < len(sentence_ids):
            # Determine window size (current sentence + overlap)
            window_size = 1 + self.overlap
            window_sent_ids = sentence_ids[i:i + window_size]

            # Collect all EDUs in this window
            chunk_edus = []
            for sent_id in window_sent_ids:
                chunk_edus.extend(sentence_edus[sent_id])

            # Create chunk
            chunk_text = " ".join(chunk_edus)
            chunks.append((chunk_text, window_sent_ids))

            # Move to next window (stride = 1 to create overlap)
            i += 1

        return chunks

    def merge_duplicate_chunks(
        self,
        chunks: List[Tuple[str, List[int]]]
    ) -> List[Tuple[str, List[int]]]:
        """
        Merge chunks that share more than 'overlap' sentences.

        This handles edge cases where chunks accidentally share too many sentences
        due to EDU boundary predictions.

        Args:
            chunks: List of (chunk_text, sentence_ids) tuples

        Returns:
            Deduplicated list of chunks
        """
        if not chunks or self.overlap == 0:
            return chunks

        merged_chunks = []
        i = 0

        while i < len(chunks):
            current_text, current_ids = chunks[i]

            # Check if next chunk overlaps too much
            if i + 1 < len(chunks):
                next_text, next_ids = chunks[i + 1]

                # Count overlapping sentences
                overlap_count = len(set(current_ids) & set(next_ids))

                # If overlap exceeds configured limit, merge
                if overlap_count > self.overlap:
                    # Merge the two chunks
                    merged_ids = sorted(set(current_ids) | set(next_ids))
                    merged_text = current_text + " " + next_text

                    merged_chunks.append((merged_text, merged_ids))
                    i += 2  # Skip next chunk since we merged it
                    continue

            # No merge needed
            merged_chunks.append((current_text, current_ids))
            i += 1

        return merged_chunks

    def chunk(
        self,
        cleaned_text: str,
        annotated_lines: str,
        **kwargs
    ) -> List[Tuple[str, List[int]]]:
        """
        Chunk text into EDUs using the trained model, processing line by line.

        Args:
            cleaned_text: Full article text (unused, kept for interface compatibility)
            annotated_lines: Tab-separated lines from FEVER dataset:
                           Format: "sentence_id\tsentence_text"
            **kwargs: Additional arguments (unused)

        Returns:
            List of (chunk_text, sentence_ids) tuples
        """
        # Parse annotated lines into (id, text) tuples
        lines = self.parse_annotated_lines(annotated_lines)

        if not lines:
            return []

        lines_with_number = [(i, line) for i, line in enumerate(lines)]
        # Step 1: Process each line and extract EDUs
        edus_with_ids = self.process_lines_to_edus(lines_with_number)

        # Step 2: Create chunks with overlap
        chunks = self.create_chunks_with_overlap(edus_with_ids)

        # Step 3: Merge chunks with excessive overlap
        merged_chunks = self.merge_duplicate_chunks(chunks)

        return merged_chunks

    def get_metadata(
        self,
        article_id: str,
        chunk_index: int,
        chunk_text: str,
        sentence_ids: List[int] = None
    ) -> Dict:
        """
        Generate metadata for an EDU chunk.

        Args:
            article_id: Wikipedia article ID from FEVER dataset
            chunk_index: Index of this chunk within the article
            chunk_text: The actual text content of the chunk
            sentence_ids: List of sentence IDs contained in this chunk

        Returns:
            Dictionary containing metadata for retrieval evaluation
        """
        sentence_ids = sentence_ids or []

        return {
            'article_id': article_id,
            'chunk_index': chunk_index,
            'sentence_ids': sentence_ids,
            'chunk_type': 'edu_linear_head',
            'chunk_size': len(chunk_text),
            'token_count': len(chunk_text.split()),
            'num_sentences': len(sentence_ids),
            'overlap': self.overlap,
            'model_path': str(self.model_path),
            'cleaned': bool(chunk_text.strip())
        }