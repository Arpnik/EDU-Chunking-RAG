from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod

class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.config = kwargs

    @abstractmethod
    def chunk(self, text: str, sentences: List[str], **kwargs) -> List[str]:
        """
        Chunk the text into smaller pieces.

        Args:
            text: Full article text
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
