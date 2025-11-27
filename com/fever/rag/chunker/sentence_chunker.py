from typing import List, Dict, Tuple
from com.fever.rag.chunker.base_chunker import BaseChunker
from com.fever.rag.utils.chunker_stats import ChunkerStatistics
from com.fever.rag.utils.text_cleaner import TextCleaner


class SentenceChunker(BaseChunker):
    """Each sentence is a chunk."""

    def __init__(self,  **kwargs):
        super().__init__('sentence')
        self.stats = ChunkerStatistics('sentence_chunker')


    def chunk(self, cleaned_text: str,annotated_lines: str, **kwargs) -> List[Tuple[str, List[int]]]:
        """
        Returns: List of (chunk_text, [sentence_id]) tuples
        """
        sentences = self.parse_annotated_lines(annotated_lines)
        self.stats.record_article()
        chunks = []
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                chunks.append((sentence, [i]))
                # Record statistics
                # For sentence chunker, assume 1 EDU per sentence
                self.stats.record_sentence(sentence, edu_count=1)
                self.stats.record_edu(sentence)  # Treat whole sentence as one EDU
                self.stats.record_chunk(sentence, [i], edu_count=1)

        return chunks

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