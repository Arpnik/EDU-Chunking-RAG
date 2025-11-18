from typing import List, Dict, Tuple
from com.fever.rag.chunker.base_chunker import BaseChunker


class FixedTokenChunker(BaseChunker):
    """Fixed token size chunks with overlap - tracks sentence boundaries during chunking."""

    def __init__(self, size: int = 128, overlap: int = 20):
        super().__init__('fixed_token', size=size, overlap=overlap)
        self.size = size
        self.overlap = overlap

    def chunk(self, text: str, sentences: List[str], **kwargs) -> List[Tuple[str, List[int]]]:
        """
        Returns: List of (chunk_text, [sentence_ids]) tuples

        Efficiently tracks which sentences are in each chunk by mapping
        tokens back to sentence positions.
        """
        tokenizer = kwargs.get('tokenizer')
        if not tokenizer:
            raise ValueError("FixedTokenChunker requires 'tokenizer' in kwargs")

        # Build sentence position map (character positions)
        sentence_positions = []
        current_pos = 0

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            start = text.find(sentence, current_pos)
            if start == -1:
                continue
            end = start + len(sentence)
            sentence_positions.append((i, start, end))
            current_pos = end

        # Tokenize full text
        tokens = tokenizer.encode(text)
        chunks_with_ids = []
        start = 0

        while start < len(tokens):
            end = start + self.size
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens).strip()

            if chunk_text:
                # Find character positions of this chunk in original text
                # (approximate by finding the chunk text)
                chunk_start = text.find(chunk_text)
                if chunk_start != -1:
                    chunk_end = chunk_start + len(chunk_text)

                    # Find which sentences overlap
                    chunk_sentence_ids = []
                    for sent_id, sent_start, sent_end in sentence_positions:
                        if sent_start < chunk_end and sent_end > chunk_start:
                            chunk_sentence_ids.append(sent_id)

                    chunks_with_ids.append((chunk_text, chunk_sentence_ids))
                else:
                    # Fallback: couldn't find chunk in text
                    chunks_with_ids.append((chunk_text, []))

            start += (self.size - self.overlap)

        return chunks_with_ids

    def get_metadata(self, article_id: str, chunk_index: int, chunk_text: str,
                     sentence_ids: List[int] = None) -> Dict:
        """Generate metadata for a fixed token chunk."""
        sentence_ids = sentence_ids or []

        return {
            'article_id': article_id,
            'chunk_index': chunk_index,
            'sentence_ids': sentence_ids,
            'chunk_type': 'fixed_token',
            'chunk_size': len(chunk_text),
            'token_count': len(chunk_text.split()),
            'target_token_size': self.size,
            'overlap_tokens': self.overlap,
            'cleaned': bool(chunk_text.strip())
        }
