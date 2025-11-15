from typing import List, Dict
from com.fever.rag.chunker.base_chunker import BaseChunker

class FixedCharChunker(BaseChunker):
    """Fixed character size chunks with overlap."""

    def __init__(self, size: int = 500, overlap: int = 50):
        super().__init__('fixed_char', size=size, overlap=overlap)
        self.size = size
        self.overlap = overlap

    def chunk(self, text: str, sentences: List[str], **kwargs) -> List[str]:
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.size
            chunk = text[start:end]

            if chunk.strip():
                chunks.append(chunk.strip())

            start += (self.size - self.overlap)

        return chunks

    def get_metadata(self, article_id: str, chunk_index: int, chunk_text: str) -> Dict:
        """
        Generate metadata for a fixed character chunk.

        Args:
            article_id: Wikipedia article ID from FEVER dataset
            chunk_index: Index of this chunk within the article
            chunk_text: The actual text content of the chunk

        Returns:
            Dictionary containing metadata for retrieval evaluation
        """
        return {
            'article_id': article_id,
            'chunk_index': chunk_index,
            'chunk_text': chunk_text,
            'chunk_type': 'fixed_char',
            'chunk_size': len(chunk_text),
            'token_count': len(chunk_text.split()),
            'target_size': self.size,
            'overlap': self.overlap,
            'cleaned': bool(chunk_text.strip())
        }