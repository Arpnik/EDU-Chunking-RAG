from typing import List
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