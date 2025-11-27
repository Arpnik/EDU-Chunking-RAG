"""
EDU-based chunker with complete statistics tracking.
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
    EDU-based chunker using a fine-tuned token classification model.

    Tracks comprehensive statistics about EDU segmentation and chunking.
    """

    def __init__(self, model_path: str, overlap: int = 0, **kwargs):
        """Initialize the EDU chunker with statistics tracking."""
        super().__init__('edu_linear_head', model_path=model_path)
        self.model_path = Path(model_path)
        self.device = get_device()
        self.overlap = max(0, overlap)
        self.boundary_count =0
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
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))

        # Load model based on type
        if model_type == "mlp_classifier":
            mlp_dims = self.model_config.get("mlp_hidden_dims", [256, 128])
            mlp_dropout = self.model_config.get("mlp_dropout", 0.3)

            peft_config_path = self.model_path / "adapter_config.json"
            with open(peft_config_path, 'r') as f:
                peft_config = json.load(f)
            base_model_name = peft_config.get("base_model_name_or_path")
            if base_model_name is None:
                print("⚠️ base_model_name_or_path missing in adapter_config.json — defaulting to bert-base-uncased")
                base_model_name = "bert-base-uncased"

            base_model = BERTWithMLPClassifier(
                model_name=base_model_name,
                num_labels=2,
                mlp_hidden_dims=mlp_dims,
                mlp_dropout=mlp_dropout
            )

            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base_model, str(self.model_path))
        else:
            from peft import AutoPeftModelForTokenClassification
            self.model = AutoPeftModelForTokenClassification.from_pretrained(str(self.model_path))

        transformers.logging.set_verbosity_warning()

        self.model.to(self.device)
        self.model.eval()

        print(f"✓ EDU model loaded on {self.device}")
        print(f"✓ Chunk overlap: {self.overlap} sentence(s)")

        self.data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True
        )

    def predict_edu_boundaries_for_line(self, line_text: str) -> List[int]:
        """
        Predict EDU boundaries for a single line (sentence).

        Returns:
            List of predictions (0 or 1) for each token in the sentence
        """
        if not line_text or not line_text.strip():
            return []

        encoding = self.tokenizer(
            line_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2).squeeze()

        if predictions.dim() == 0:
            return [predictions.item()]

        pred_list = predictions.cpu().numpy().tolist()

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

        Returns:
            List of EDU strings from this sentence
        """
        if not predictions or not line_text.strip():
            return [line_text] if line_text.strip() else []

        encoding = self.tokenizer(
            line_text,
            return_offsets_mapping=True,
            add_special_tokens=False
        )

        offsets = encoding['offset_mapping']

        if len(offsets) != len(predictions):
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

            # Predict EDU boundaries for this line
            predictions = self.predict_edu_boundaries_for_line(line_text)

            if not predictions:
                # No predictions, treat whole line as single EDU
                all_edus.append((line_text.strip(), sent_id))

                # Record statistics
                self.stats.record_sentence(line_text.strip(), edu_count=1)
                self.stats.record_edu(line_text.strip())
                continue

            # Split line into EDUs
            edus = self.split_line_into_edus(line_text, predictions)

            # Count EDU boundaries (number of 1s in predictions = boundaries detected)
            self.boundary_count += sum(predictions)

            # Record sentence statistics
            self.stats.record_sentence(line_text, edu_count=len(edus))

            # Add all EDUs from this sentence and record each
            for edu in edus:
                all_edus.append((edu, sent_id))
                self.stats.record_edu(edu)

        return all_edus

    def create_chunks_with_overlap(
        self,
        edus_with_ids: List[Tuple[str, int]]
    ) -> List[Tuple[str, List[int]]]:
        """
        Combine EDUs into chunks with configurable sentence-level overlap.

        Returns:
            List of (chunk_text, sentence_ids) tuples
        """
        if not edus_with_ids:
            return []

        # Group EDUs by sentence ID
        sentence_edus = {}
        for edu_text, sent_id in edus_with_ids:
            if sent_id not in sentence_edus:
                sentence_edus[sent_id] = []
            sentence_edus[sent_id].append(edu_text)

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
            window_size = 1 + self.overlap
            window_sent_ids = sentence_ids[i:i + window_size]

            chunk_edus = []
            for sent_id in window_sent_ids:
                chunk_edus.extend(sentence_edus[sent_id])

            chunk_text = " ".join(chunk_edus)
            chunks.append((chunk_text, window_sent_ids))

            i += 1

        return chunks

    def merge_duplicate_chunks(
        self,
        chunks: List[Tuple[str, List[int]]]
    ) -> List[Tuple[str, List[int]]]:
        """Merge chunks that share more than 'overlap' sentences."""
        if not chunks or self.overlap == 0:
            return chunks

        merged_chunks = []
        i = 0

        while i < len(chunks):
            current_text, current_ids = chunks[i]

            if i + 1 < len(chunks):
                next_text, next_ids = chunks[i + 1]
                overlap_count = len(set(current_ids) & set(next_ids))

                if overlap_count > self.overlap:
                    merged_ids = sorted(set(current_ids) | set(next_ids))
                    merged_text = current_text + " " + next_text
                    merged_chunks.append((merged_text, merged_ids))
                    i += 2
                    continue

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
        Chunk text into EDUs using the trained model.
        Records comprehensive statistics during processing.

        Returns:
            List of (chunk_text, sentence_ids) tuples
        """
        lines = self.parse_annotated_lines(annotated_lines)

        if not lines:
            return []

        # Record article
        self.stats.record_article()

        lines_with_number = [(i, line) for i, line in enumerate(lines)]

        # Step 1: Process each line and extract EDUs (statistics recorded here)
        edus_with_ids = self.process_lines_to_edus(lines_with_number)

        # Step 2: Create chunks with overlap
        chunks = self.create_chunks_with_overlap(edus_with_ids)

        # Step 3: Merge chunks with excessive overlap
        merged_chunks = self.merge_duplicate_chunks(chunks)

        # Step 4: Record chunk statistics
        for chunk_text, sentence_ids in merged_chunks:
            # Count EDUs in this chunk by checking which EDUs belong to these sentences
            edu_count = sum(1 for edu_text, sent_id in edus_with_ids
                          if sent_id in sentence_ids)
            self.stats.record_chunk(chunk_text, sentence_ids, edu_count=edu_count)

        return merged_chunks

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
            'overlap': self.overlap,
            'model_path': str(self.model_path),
            'cleaned': bool(chunk_text.strip())
        }