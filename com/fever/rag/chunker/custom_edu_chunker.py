from typing import List

from com.fever.rag.chunker.base_chunker import BaseChunker

#TODO : Implement actual EDU segmentation logic or integrate with an existing library.
class EDUChunker(BaseChunker):
    """EDU-based chunking (combines N consecutive EDUs/sentences)."""

    def __init__(self, combine_n: int = 3, segmenter=None):
        super().__init__('edu', combine_n=combine_n)
        self.combine_n = combine_n
        self.segmenter = segmenter

    def chunk(self, text: str, sentences: List[str], **kwargs) -> List[str]:
        if self.segmenter is None:
            # Placeholder: combine N consecutive sentences
            chunks = []
            for i in range(0, len(sentences), self.combine_n):
                combined = ' '.join(sentences[i:i + self.combine_n])
                if combined.strip():
                    chunks.append(combined.strip())
            return chunks
        else:
            # Use actual EDU segmenter
            segmenter = kwargs.get('edu_segmenter', self.segmenter)
            edus = segmenter.segment(text)

            chunks = []
            for i in range(0, len(edus), self.combine_n):
                combined = ' '.join(edus[i:i + self.combine_n])
                if combined.strip():
                    chunks.append(combined.strip())
            return chunks
