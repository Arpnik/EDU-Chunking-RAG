"""
EDU-based chunker using a fine-tuned BERT model with linear head for token classification.

This chunker uses a trained model to predict EDU (Elementary Discourse Unit) boundaries
at the token level and creates chunks based on these predicted boundaries.
"""

from typing import List, Dict, Tuple, Optional
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification
)
from com.fever.rag.chunker.base_chunker import BaseChunker
from com.fever.rag.utils.DataHelper import _get_device


class EDUChunkerWithLinearHead(BaseChunker):
    """
    EDU-based chunker using a fine-tuned token classification model.

    The model predicts labels for each token:
    - 0 = EDU Continue (token is within current EDU)
    - 1 = EDU Start (token begins a new EDU)

    When label 1 is predicted, a new chunk starts.
    """

    def __init__(self, model_path: str):
        """
        Initialize the EDU chunker.

        Args:
            model_path: Path to the trained model directory (contains model + tokenizer)
            device: Device to run model on ('cuda', 'mps', 'cpu', or None for auto)
        """
        super().__init__('edu_linear_head', model_path=model_path)

        self.model_path = Path(model_path)
        self.device = _get_device()

        # Load tokenizer and model
        print(f"Loading EDU model from: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForTokenClassification.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ EDU model loaded on {self.device}")

        # Data collator for batch processing
        self.data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True
        )

    def predict_edu_boundaries_with_sliding_window(
            self,
            text: str,
            max_length: int = 512,
            stride: int = 256  # 50% overlap
    ) -> List[int]:
        """
        Process long texts in overlapping windows.
        """
        # Tokenize full text first
        full_encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False
        )

        tokens = full_encoding['input_ids']
        offsets = full_encoding['offset_mapping']

        all_predictions = []

        # Process in windows
        for start_idx in range(0, len(tokens), stride):
            end_idx = min(start_idx + max_length - 2, len(tokens))  # -2 for [CLS]/[SEP]

            window_tokens = tokens[start_idx:end_idx]

            # Add special tokens
            window_input = torch.tensor([[
                self.tokenizer.cls_token_id,
                *window_tokens,
                self.tokenizer.sep_token_id
            ]]).to(self.device)

            # Predict for this window
            with torch.no_grad():
                outputs = self.model(window_input)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=2).squeeze()

            # Extract predictions (skip [CLS] and [SEP])
            window_preds = predictions[1:-1].cpu().numpy().tolist()

            # For overlapping regions, keep first occurrence
            if start_idx == 0:
                all_predictions.extend(window_preds)
            else:
                # Skip overlapped tokens
                all_predictions.extend(window_preds[stride:])

            if end_idx >= len(tokens):
                break

        return all_predictions[:len(tokens)]  # Ensure same length as input

    def _align_tokens_to_text(
            self,
            text: str,
            predictions: List[int]
    ) -> List[Tuple[str, int, int, int]]:
        """
        Align token predictions back to character positions in original text.

        Args:
            text: Original text
            predictions: EDU boundary predictions for each token

        Returns:
            List of (token_text, start_char, end_char, prediction) tuples
        """
        # Tokenize with offset mapping to get character positions
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True
        )

        offset_mapping = encoding['offset_mapping']
        tokens = encoding.tokens()

        # Align predictions with offsets (skip special tokens)
        aligned_tokens = []
        pred_idx = 0

        for i, (token, (start, end)) in enumerate(zip(tokens, offset_mapping)):
            # Skip special tokens ([CLS], [SEP], [PAD])
            if start == end == 0:
                continue

            if pred_idx < len(predictions):
                prediction = predictions[pred_idx]
                token_text = text[start:end]
                aligned_tokens.append((token_text, start, end, prediction))
                pred_idx += 1

        return aligned_tokens

    def _create_edu_chunks(
            self,
            text: str,
            aligned_tokens: List[Tuple[str, int, int, int]],
            sentences: List[str]
    ) -> List[Tuple[str, List[int]]]:
        """
        Create EDU chunks based on predicted boundaries and track sentence IDs.

        Args:
            text: Original text
            aligned_tokens: List of (token_text, start_char, end_char, prediction)
            sentences: Pre-parsed sentences from article

        Returns:
            List of (chunk_text, sentence_ids) tuples
        """
        if not aligned_tokens:
            return []

        # Build sentence position map
        sentence_positions = []
        current_pos = 0
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            start = text.find(sentence, current_pos)
            if start == -1:
                continue
            end = start + len(sentence)
            sentence_positions.append((i, start, end))
            current_pos = end

        # Create chunks based on EDU boundaries (prediction = 1)
        chunks = []
        current_chunk_start = aligned_tokens[0][1]  # Start of first token
        current_chunk_end = aligned_tokens[0][2]  # End of first token

        for i, (token_text, start, end, prediction) in enumerate(aligned_tokens):
            if prediction == 1 and i > 0:  # New EDU starts (but not at first token)
                # Finalize previous chunk
                chunk_text = text[current_chunk_start:current_chunk_end].strip()

                if chunk_text:
                    # Find which sentences overlap with this chunk
                    chunk_sentence_ids = []
                    for sent_id, sent_start, sent_end in sentence_positions:
                        if sent_start < current_chunk_end and sent_end > current_chunk_start:
                            chunk_sentence_ids.append(sent_id)

                    chunks.append((chunk_text, chunk_sentence_ids))

                # Start new chunk
                current_chunk_start = start
                current_chunk_end = end
            else:
                # Continue current chunk
                current_chunk_end = end

        # Add final chunk
        chunk_text = text[current_chunk_start:current_chunk_end].strip()
        if chunk_text:
            chunk_sentence_ids = []
            for sent_id, sent_start, sent_end in sentence_positions:
                if sent_start < current_chunk_end and sent_end > current_chunk_start:
                    chunk_sentence_ids.append(sent_id)
            chunks.append((chunk_text, chunk_sentence_ids))

        return chunks

    def chunk(self, cleaned_text: str, annotated_lines: str, **kwargs) -> List[Tuple[str, List[int]]]:
        """
        Chunk text into EDUs using the trained model.

        Args:
            cleaned_text: Full article text
            **kwargs: Additional arguments (unused)

        Returns:
            List of (chunk_text, sentence_ids) tuples
        """
        if not cleaned_text or not cleaned_text.strip():
            return []

        # Step 1: Predict EDU boundaries for each token
        predictions = self.predict_edu_boundaries_with_sliding_window(cleaned_text)

        # Step 2: Align token predictions back to character positions
        aligned_tokens = self._align_tokens_to_text(cleaned_text, predictions)

        # Step 3: Create chunks based on boundaries and track sentence IDs
        sentences = self.parse_annotated_lines(annotated_lines)
        chunks_with_ids = self._create_edu_chunks(cleaned_text, aligned_tokens, sentences)

        return chunks_with_ids

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
            'model_path': str(self.model_path),
            'cleaned': bool(chunk_text.strip())
        }
