from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class SplitConfig:
    max_chunk_size: int = 1000
    overlap: int = 100
    split_by: str = 'sentence'  # 'sentence' or 'character'
    min_chunk_size: int = 100

class TextSplitter:
    """Utility for splitting text into manageable chunks."""
    
    def __init__(self, config: SplitConfig = None):
        self.config = config or SplitConfig()
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks based on configuration."""
        if self.config.split_by == 'sentence':
            return self._split_by_sentence(text)
        return self._split_by_character(text)
    
    def _split_by_sentence(self, text: str) -> List[str]:
        """Split text into chunks by sentence boundaries."""
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # Check if adding this sentence would exceed max chunk size
            if current_size + sentence_size > self.config.max_chunk_size and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                
                # Keep last sentence for overlap if configured
                if self.config.overlap > 0 and current_chunk:
                    current_chunk = current_chunk[-1:]
                    current_size = len(current_chunk[0])
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add remaining chunk
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
            
        return chunks
    
    def _split_by_character(self, text: str) -> List[str]:
        """Split text into chunks by character count."""
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end of the chunk
            end = start + self.config.max_chunk_size
            
            # Adjust end to not split words
            if end < len(text):
                while end > start and not text[end].isspace():
                    end -= 1
                if end == start:  # No space found
                    end = start + self.config.max_chunk_size
            
            chunks.append(text[start:end])
            
            # Calculate next start position considering overlap
            start = end - self.config.overlap if self.config.overlap > 0 else end
        
        return chunks