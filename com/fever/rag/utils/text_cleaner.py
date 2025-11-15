import re
import html

class TextCleaner:
    """Handles text cleaning and validation."""

    @staticmethod
    def clean(text: str) -> str:
        """Clean formatting artifacts from Wikipedia text."""
        # Bracket replacements
        bracket_map = {
            '-LRB-': '(', '-RRB-': ')',
            '-LSB-': '[', '-RSB-': ']',
            '-LCB-': '{', '-RCB-': '}',
        }

        for old, new in bracket_map.items():
            text = text.replace(old, new)

        # Handle HTML entities
        text = html.unescape(text)

        # Handle special dash patterns
        text = re.sub(r'-([A-Z])-(?!\w)', r'\1', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    @staticmethod
    def validate(text: str) -> bool:
        """Check if text still has problematic patterns."""
        problematic = ['-LRB-', '-RRB-', '-LSB-', '-RSB-', '&nbsp;', '&amp;']
        return not any(pattern in text for pattern in problematic)

