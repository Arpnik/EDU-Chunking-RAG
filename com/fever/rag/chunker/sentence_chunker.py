from typing import List, Dict, Tuple

from com.fever.rag.chunker.base_chunker import BaseChunker


class SentenceChunker(BaseChunker):
    """Each sentence is a chunk."""

    def __init__(self):
        super().__init__('sentence')

    def chunk(self, text: str, sentences: List[str], **kwargs) -> List[Tuple[str, List[int]]]:
        """
        Returns: List of (chunk_text, [sentence_id]) tuples
        """
        return [(s, [i]) for i, s in enumerate(sentences) if s.strip()]

    def get_metadata(self, article_id: str, chunk_index: int, chunk_text: str,
                     sentence_ids: List[int] = None) -> Dict:
        """Generate metadata for a sentence chunk."""
        sentence_ids = sentence_ids or [chunk_index]

        return {
            'article_id': article_id,
            'chunk_index': chunk_index,
            'sentence_ids': sentence_ids,
            'chunk_type': 'sentence',
            'chunk_size': len(chunk_text),
            'token_count': len(chunk_text.split()),
            'cleaned': bool(chunk_text.strip())
        }