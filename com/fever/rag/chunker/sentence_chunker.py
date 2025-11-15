from typing import List, Dict

from com.fever.rag.chunker.base_chunker import BaseChunker


class SentenceChunker(BaseChunker):
    """Each sentence is a chunk."""

    def __init__(self):
        super().__init__('sentence')

    def chunk(self, text: str, sentences: List[str], **kwargs) -> List[str]:
        return [s for s in sentences if s.strip()]

    def get_metadata(self, article_id: str, chunk_index: int, chunk_text: str) -> Dict:
        """
        Generate metadata for a sentence chunk.

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
            'chunk_type': 'sentence',
            'chunk_size': len(chunk_text),
            'token_count': len(chunk_text.split()),
            'cleaned': bool(chunk_text.strip())
        }