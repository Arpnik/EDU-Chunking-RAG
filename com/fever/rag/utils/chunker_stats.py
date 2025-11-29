"""
Add this class to collect statistics during the chunking process.
Insert this code in a new file: chunker_statistics.py
"""

import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
import json


class ChunkerStatistics:
    """Collects statistics about chunking behavior."""

    def __init__(self, chunker_name: str):
        self.chunker_name = chunker_name

        # Chunk-level statistics
        self.chunk_char_lengths = []
        self.chunk_token_lengths = []
        self.edus_per_chunk = []  # Only for EDU chunker
        self.sentences_per_chunk = []

        # EDU-level statistics (only for EDU chunker)
        self.edu_char_lengths = []
        self.edu_token_lengths = []
        self.edus_per_sentence = []

        # Sentence-level statistics
        self.sentence_char_lengths = []
        self.sentence_token_lengths = []

        self.total_chunks = 0
        self.total_articles = 0

    def record_chunk(self, chunk_text: str, sentence_ids: List[int],
                     edu_count: int = None):
        """Record statistics for a single chunk."""
        self.total_chunks += 1

        # Chunk statistics
        char_len = len(chunk_text)
        token_len = len(chunk_text.split())

        self.chunk_char_lengths.append(char_len)
        self.chunk_token_lengths.append(token_len)
        self.sentences_per_chunk.append(len(sentence_ids))

        # EDU count (only for EDU chunker)
        if edu_count is not None:
            self.edus_per_chunk.append(edu_count)

    def record_edu(self, edu_text: str):
        """Record statistics for a single EDU."""
        self.edu_char_lengths.append(len(edu_text))
        self.edu_token_lengths.append(len(edu_text.split()))

    def record_sentence(self, sentence_text: str, edu_count: int = 1):
        """Record statistics for a single sentence."""
        self.sentence_char_lengths.append(len(sentence_text))
        self.sentence_token_lengths.append(len(sentence_text.split()))
        if edu_count > 0:
            self.edus_per_sentence.append(edu_count)

    def record_article(self):
        """Increment article counter."""
        self.total_articles += 1

    def get_stats(self) -> Dict:
        """Calculate and return all statistics."""

        def safe_stats(data):
            if not data:
                return {
                    'mean': 0, 'std': 0, 'min': 0, 'max': 0,
                    'median': 0, 'count': 0
                }
            arr = np.array(data)
            return {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'median': float(np.median(arr)),
                'count': len(data)
            }

        stats = {
            'chunker_name': self.chunker_name,
            'total_articles': self.total_articles,
            'total_chunks': self.total_chunks,

            # Chunk statistics
            'chunk_characters': safe_stats(self.chunk_char_lengths),
            'chunk_tokens': safe_stats(self.chunk_token_lengths),
            'sentences_per_chunk': safe_stats(self.sentences_per_chunk),

            # Sentence statistics
            'sentence_characters': safe_stats(self.sentence_char_lengths),
            'sentence_tokens': safe_stats(self.sentence_token_lengths),
        }

        # EDU-specific statistics
        if self.edus_per_chunk:
            stats['edus_per_chunk'] = safe_stats(self.edus_per_chunk)

        if self.edu_char_lengths:
            stats['edu_characters'] = safe_stats(self.edu_char_lengths)
            stats['edu_tokens'] = safe_stats(self.edu_token_lengths)

        if self.edus_per_sentence:
            stats['edus_per_sentence'] = safe_stats(self.edus_per_sentence)
            stats['sentences_with_multiple_edus'] = {
                'count': sum(1 for x in self.edus_per_sentence if x > 1),
                'percentage': 100 * sum(1 for x in self.edus_per_sentence if x > 1) / len(
                    self.edus_per_sentence) if self.edus_per_sentence else 0
            }

        return stats

    def print_stats(self):
        """Print formatted statistics."""
        stats = self.get_stats()

        print("\n" + "=" * 70)
        print(f"STATISTICS FOR: {stats['chunker_name']}")
        print("=" * 70)

        print(f"\nOverall:")
        print(f"  Total articles: {stats['total_articles']:,}")
        print(f"  Total chunks: {stats['total_chunks']:,}")

        print(f"\nChunk Statistics:")
        print(f"  Characters per chunk:")
        print(f"    Mean: {stats['chunk_characters']['mean']:.1f}")
        print(f"    Std:  {stats['chunk_characters']['std']:.1f}")
        print(f"    Min:  {stats['chunk_characters']['min']:.0f}")
        print(f"    Max:  {stats['chunk_characters']['max']:.0f}")
        print(f"    Median: {stats['chunk_characters']['median']:.1f}")

        print(f"  Tokens per chunk:")
        print(f"    Mean: {stats['chunk_tokens']['mean']:.1f}")
        print(f"    Std:  {stats['chunk_tokens']['std']:.1f}")
        print(f"    Min:  {stats['chunk_tokens']['min']:.0f}")
        print(f"    Max:  {stats['chunk_tokens']['max']:.0f}")
        print(f"    Median: {stats['chunk_tokens']['median']:.1f}")

        print(f"  Sentences per chunk:")
        print(f"    Mean: {stats['sentences_per_chunk']['mean']:.2f}")
        print(f"    Std:  {stats['sentences_per_chunk']['std']:.2f}")
        print(f"    Min:  {stats['sentences_per_chunk']['min']:.0f}")
        print(f"    Max:  {stats['sentences_per_chunk']['max']:.0f}")
        print(f"    Median: {stats['sentences_per_chunk']['median']:.1f}")

        if 'edus_per_chunk' in stats:
            print(f"  EDUs per chunk:")
            print(f"    Mean: {stats['edus_per_chunk']['mean']:.2f}")
            print(f"    Std:  {stats['edus_per_chunk']['std']:.2f}")
            print(f"    Min:  {stats['edus_per_chunk']['min']:.0f}")
            print(f"    Max:  {stats['edus_per_chunk']['max']:.0f}")
            print(f"    Median: {stats['edus_per_chunk']['median']:.1f}")

        print(f"\nSentence Statistics:")
        print(f"  Characters per sentence:")
        print(f"    Mean: {stats['sentence_characters']['mean']:.1f}")
        print(f"    Std:  {stats['sentence_characters']['std']:.1f}")

        print(f"  Tokens per sentence:")
        print(f"    Mean: {stats['sentence_tokens']['mean']:.1f}")
        print(f"    Std:  {stats['sentence_tokens']['std']:.1f}")

        if 'edu_characters' in stats:
            print(f"\nEDU Statistics:")
            print(f"  Characters per EDU:")
            print(f"    Mean: {stats['edu_characters']['mean']:.1f}")
            print(f"    Std:  {stats['edu_characters']['std']:.1f}")

            print(f"  Tokens per EDU:")
            print(f"    Mean: {stats['edu_tokens']['mean']:.1f}")
            print(f"    Std:  {stats['edu_tokens']['std']:.1f}")

            print(f"  EDUs per sentence:")
            print(f"    Mean: {stats['edus_per_sentence']['mean']:.2f}")
            print(f"    Std:  {stats['edus_per_sentence']['std']:.2f}")

            if 'sentences_with_multiple_edus' in stats:
                print(f"  Sentences with multiple EDUs:")
                print(f"    Count: {stats['sentences_with_multiple_edus']['count']:,}")
                print(f"    Percentage: {stats['sentences_with_multiple_edus']['percentage']:.1f}%")

        print("=" * 70)

    def save_to_file(self, filepath: str):
        """Save statistics to JSON file."""
        stats = self.get_stats()
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nStatistics saved to: {filepath}")