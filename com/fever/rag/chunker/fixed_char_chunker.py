from typing import List, Dict, Tuple
from com.fever.rag.chunker.base_chunker import BaseChunker

class FixedCharChunker(BaseChunker):
    """Fixed character size chunks with overlap - tracks sentence boundaries during chunking."""

    def __init__(self, size: int = 500, overlap: int = 50,  **kwargs):
        super().__init__('fixed_char', size=size, overlap=overlap)
        self.size = size
        self.overlap = overlap

    def chunk(self, cleaned_text: str, annotated_lines: str, **kwargs) -> List[Tuple[str, List[int]]]:
        """
        Returns: List of (chunk_text, [sentence_ids]) tuples

        Efficiently tracks which sentences are in each chunk by building
        a position map during chunking.
        """
        # Build sentence position map
        sentences = self.parse_annotated_lines(annotated_lines)
        sentence_positions = []
        current_pos = 0

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            # Find sentence in full text
            start = cleaned_text.find(sentence, current_pos)
            if start == -1:
                continue
            end = start + len(sentence)
            sentence_positions.append((i, start, end))
            current_pos = end

        # Create chunks and track sentence IDs
        chunks_with_ids = []
        start = 0
        text_len = len(cleaned_text)

        while start < text_len:
            end = start + self.size
            chunk = cleaned_text[start:end].strip()

            if chunk:
                # Find which sentences overlap with this chunk
                chunk_sentence_ids = []
                for sent_id, sent_start, sent_end in sentence_positions:
                    # Check if sentence overlaps with chunk [start, end)
                    if sent_start < end and sent_end > start:
                        chunk_sentence_ids.append(sent_id)

                chunks_with_ids.append((chunk, chunk_sentence_ids))

            start += (self.size - self.overlap)

        return chunks_with_ids

    def get_metadata(self, article_id: str, chunk_index: int, chunk_text: str,
                     sentence_ids: List[int] = None) -> Dict:
        """Generate metadata for a fixed character chunk."""
        sentence_ids = sentence_ids or []

        return {
            'article_id': article_id,
            'chunk_index': chunk_index,
            'sentence_ids': sentence_ids,
            'chunk_type': 'fixed_char',
            'chunk_size': len(chunk_text),
            'token_count': len(chunk_text.split()),
            'target_size': self.size,
            'overlap': self.overlap,
            'cleaned': bool(chunk_text.strip())
        }
