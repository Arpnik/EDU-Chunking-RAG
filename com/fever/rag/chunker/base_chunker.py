from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod

from com.fever.rag.utils.text_cleaner import TextCleaner


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.config = kwargs

    @abstractmethod
    def chunk(self, cleaned_text: str, annotated_lines: str, **kwargs) -> List[str]:
        """
        Chunk the text into smaller pieces.

        Args:
            cleaned_text: Full article text
            sentences: Pre-parsed sentences from the article
            **kwargs: Additional arguments (e.g., tokenizer, segmenter)

        Returns:
            List of text chunks
        """
        raise NotImplementedError(
            f"Chunker '{self.name}' has not implemented the 'chunk()' method."
        )

    @abstractmethod
    def get_metadata(self, article_id: str, chunk_index: int, chunk_text: str, sentence_ids: List[int] = None) -> Dict:
        """Generate metadata for a chunk."""
        raise NotImplementedError(
            f"Chunker '{self.name}' has not implemented the 'get_metadata()' method."
        )

    @staticmethod
    def parse_annotated_lines(annotated_lines: str) -> List[str]:
        """Parse article lines into clean sentences."""
        if not annotated_lines:
            return []

        sentences = []
        for line in annotated_lines.strip().split('\n'):
            if not line and not line.strip():
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                sentence = TextCleaner.clean(parts[1])
                if sentence:
                    sentences.append(sentence)
        return sentences
