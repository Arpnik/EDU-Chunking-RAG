from enum import Enum

from com.fever.rag.chunker.custom_edu_chunker import CustomEDUChunker
from com.fever.rag.chunker.fixed_char_chunker import FixedCharChunker
from com.fever.rag.chunker.fixed_token_chunker import FixedTokenChunker
from com.fever.rag.chunker.sentence_chunker import SentenceChunker

class ChunkerType(Enum):
    FIXED_CHAR = "fixed_char"
    FIXED_TOKEN = "fixed_token"
    SENTENCE = "sentence"
    CUSTOM_EDU = "custom_edu"

def get_chunker(chunker_type: ChunkerType, **kwargs):
    """Factory to get chunker based on type."""
    if chunker_type == ChunkerType.FIXED_CHAR:
        return FixedCharChunker(overlap=kwargs["chunking_overlap"],size=kwargs["chunk_size"], **kwargs)
    elif chunker_type == ChunkerType.FIXED_TOKEN:
        return FixedTokenChunker(size=kwargs["max_tokens"], overlap=kwargs["chunking_overlap"], **kwargs)
    elif chunker_type == ChunkerType.SENTENCE:
        return SentenceChunker(**kwargs)
    elif chunker_type == ChunkerType.CUSTOM_EDU:
        return CustomEDUChunker(overlap=kwargs["chunking_overlap"], **kwargs)
    else:
        raise ValueError(f"Unsupported chunker type: {chunker_type}")

CHUNKER_ARGS = {
    ChunkerType.FIXED_CHAR: ["chunk_size","chunking_overlap"],
    ChunkerType.FIXED_TOKEN: ["max_tokens","chunking_overlap"],
    ChunkerType.SENTENCE: [],
    ChunkerType.CUSTOM_EDU: ["model_path", "chunking_overlap"],
}

