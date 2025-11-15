from typing import List
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
