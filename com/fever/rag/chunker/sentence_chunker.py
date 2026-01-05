from typing import List, Dict, Tuple, Optional
from com.fever.rag.chunker.base_chunker import BaseChunker
from com.fever.rag.utils.chunker_stats import ChunkerStatistics


class SentenceChunker(BaseChunker):
    """Each sentence is a chunk. Works with both FEVER and SQuAD datasets."""

    def __init__(self, dataset_type: str = "fever", **kwargs):
        """
        Initialize sentence chunker.

        Args:
            dataset_type: Either "fever" or "squad"
                - "fever": Uses annotated_lines with sentence IDs
                - "squad": Uses simple sentence splitting on cleaned_text
        """
        super().__init__('sentence')
        self.stats = ChunkerStatistics('sentence_chunker')
        self.dataset_type = dataset_type.lower()

        if self.dataset_type not in ["fever", "squad"]:
            raise ValueError(f"dataset_type must be 'fever' or 'squad', got '{dataset_type}'")

    def chunk(self, cleaned_text: str, annotated_lines: str, **kwargs) -> List[Tuple[str, List[int]]]:
        """
        Chunk text into sentences.

        Args:
            cleaned_text: Cleaned text (used for SQuAD)
            annotated_lines: Annotated lines with sentence IDs (used for FEVER)

        Returns:
            List of (chunk_text, [sentence_id]) tuples
        """
        self.stats.record_article()

        if self.dataset_type == "fever":
            # FEVER: Use annotated lines with sentence IDs
            sentences = self.parse_annotated_lines(annotated_lines)
        else:
            # SQuAD: Simple sentence splitting
            sentences = self._split_into_sentences(cleaned_text)

        chunks = []
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                chunks.append((sentence, [i]))
                # Record statistics
                self.stats.record_sentence(sentence, edu_count=1)
                self.stats.record_edu(sentence)
                self.stats.record_chunk(sentence, [i], edu_count=1)

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Simple sentence splitting for SQuAD (no sentence IDs available).

        Args:
            text: Raw text to split

        Returns:
            List of sentences
        """
        # Simple rule-based sentence splitting
        import re

        # Split on . ! ? followed by space and capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

        return [s.strip() for s in sentences if s.strip()]

    def get_metadata(
            self,
            article_id: str,
            chunk_index: int,
            chunk_text: str,
            sentence_ids: List[int] = None,
            start_char: Optional[int] = None,  # NEW: For SQuAD position tracking
            end_char: Optional[int] = None  # NEW: For SQuAD position tracking
    ) -> Dict:
        """
        Generate metadata for a sentence chunk.

        Args:
            article_id: Article/context ID
            chunk_index: Index of this chunk
            chunk_text: The chunk text
            sentence_ids: List of sentence IDs (FEVER) or indices (SQuAD)
            start_char: Starting character position in original text (SQuAD only)
            end_char: Ending character position in original text (SQuAD only)
        """
        sentence_ids = sentence_ids or [chunk_index]

        metadata = {
            'article_id': article_id,
            'chunk_index': chunk_index,
            'sentence_ids': sentence_ids,
            'chunk_type': 'sentence',
            'chunk_size': len(chunk_text),
            'token_count': len(chunk_text.split()),
            'cleaned': bool(chunk_text.strip()),
            'dataset_type': self.dataset_type
        }

        # Add character positions for SQuAD (needed for Recall@K calculation)
        if self.dataset_type == "squad" and start_char is not None:
            metadata['start_char'] = start_char
            metadata['end_char'] = end_char if end_char is not None else start_char + len(chunk_text)

        return metadata