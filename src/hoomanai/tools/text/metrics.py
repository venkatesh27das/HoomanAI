from typing import Dict, Any

class TextMetrics:
    """Utility for calculating various text metrics."""
    
    @staticmethod
    def calculate_metrics(text: str) -> Dict[str, Any]:
        """Calculate various metrics for the text."""
        return {
            'length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(text.split('.')),
            'average_word_length': sum(len(word) for word in text.split()) / len(text.split()) if text else 0
        }