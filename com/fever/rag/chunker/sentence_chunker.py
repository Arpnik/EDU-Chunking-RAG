from com.fever.rag.chunker.base_chunker import BaseChunker


class SentenceChunker(BaseChunker):
    """Each sentence is a chunk."""

    def __init__(self):
        super().__init__('sentence')

    def chunk(self, text: str, sentences: List[str], **kwargs) -> List[str]:
        return [s for s in sentences if s.strip()]
