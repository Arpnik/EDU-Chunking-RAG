from typing import List, Dict, Tuple
from com.fever.rag.chunker.base_chunker import BaseChunker
from com.fever.rag.utils.text_cleaner import TextCleaner


class SentenceChunker(BaseChunker):
    """Each sentence is a chunk."""

    def __init__(self,  **kwargs):
        super().__init__('sentence')

    @staticmethod
    def parse_annotated_lines_with_ids(annotated_lines: str) -> List[Tuple[int, str]]:
        """
        Parse article lines into (sentence_id, text) tuples.

        Args:
            annotated_lines: Tab-separated format "sentence_id\tsentence_text"

        Returns:
            List of (original_sentence_id, cleaned_text) tuples
        """
        if not annotated_lines:
            return []

        sentences_with_ids = []
        for line in annotated_lines.strip().split('\n'):
            if not line or not line.strip():
                continue

            parts = line.split('\t')
            if len(parts) >= 2:
                try:
                    # Parse the sentence ID from FEVER format
                    sentence_id = int(parts[0])
                    sentence_text = TextCleaner.clean(parts[1])

                    if sentence_text:  # Only include non-empty sentences
                        sentences_with_ids.append((sentence_id, sentence_text))

                except ValueError:
                    # Skip lines where sentence_id isn't a valid integer
                    continue

        return sentences_with_ids

    def chunk(self, cleaned_text: str,annotated_lines: str, **kwargs) -> List[Tuple[str, List[int]]]:
        """
        Returns: List of (chunk_text, [sentence_id]) tuples
        """
        sentences_with_ids = self.parse_annotated_lines_with_ids(annotated_lines)

        # Return format: (text, [original_sentence_id])
        return [(text, [sent_id]) for sent_id, text in sentences_with_ids if text.strip()]

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