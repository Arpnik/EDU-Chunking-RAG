from typing import List, Dict
from com.fever.rag.chunker.base_chunker import BaseChunker


class FixedTokenChunker(BaseChunker):
    """Fixed token size chunks with overlap."""

    def __init__(self, size: int = 128, overlap: int = 20):
        super().__init__('fixed_token', size=size, overlap=overlap)
        self.size = size
        self.overlap = overlap

    def chunk(self, text: str, sentences: List[str], **kwargs) -> List[str]:
        tokenizer = kwargs.get('tokenizer')
        if not tokenizer:
            raise ValueError("FixedTokenChunker requires 'tokenizer' in kwargs")

        tokens = tokenizer.encode(text)
        chunks = []
        start = 0

        while start < len(tokens):
            end = start + self.size
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens)

            if chunk_text.strip():
                chunks.append(chunk_text.strip())

            start += (self.size - self.overlap)

        return chunks

    def get_metadata(self, article_id: str, chunk_index: int, chunk_text: str) -> Dict:
        """
        Generate metadata for a fixed token chunk.

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
            'chunk_type': 'fixed_token',
            'chunk_size': len(chunk_text),
            'token_count': len(chunk_text.split()),
            'target_token_size': self.size,
            'overlap_tokens': self.overlap,
            'cleaned': bool(chunk_text.strip())
        }