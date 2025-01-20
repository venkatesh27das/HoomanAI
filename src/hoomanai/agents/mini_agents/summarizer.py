from typing import Dict, Any, List
from hoomanai.core.mini_agent import MiniAgent
from hoomanai.core.types import Task
from hoomanai.tools.llm.client import LLMClient
from hoomanai.tools.text.splitter import TextSplitter, SplitConfig
from hoomanai.tools.text.metrics import TextMetrics

class SummarizerAgent(MiniAgent):
    """Mini agent for generating text summaries."""
    
    def __init__(self, llm_config=None):
        super().__init__(name="Summarizer")
        self.llm_client = LLMClient(config=llm_config)
        self.text_splitter = TextSplitter(
            config=SplitConfig(
                max_chunk_size=2000,
                overlap=100
            )
        )
        self.metrics = TextMetrics()

    def execute(self, task: Task) -> Dict[str, Any]:
        """
        Execute summarization task.
        
        Expected task context:
        - text: Text to summarize
        - style: Summary style (default: concise)
        - max_length: Maximum length of summary (optional)
        """
        text = task.context.get('text', '')
        style = task.context.get('style', 'concise')
        max_length = task.context.get('max_length')
        
        # Calculate input metrics
        input_metrics = self.metrics.calculate_metrics(text)
        
        try:
            # Split text if necessary
            if input_metrics['length'] > 2000:
                chunks = self.text_splitter.split_text(text)
                summary = self._process_long_text(chunks, style, max_length)
            else:
                summary = self._generate_summary(text, style, max_length)
            
            # Calculate output metrics
            output_metrics = self.metrics.calculate_metrics(summary)
            
            return {
                'summary': summary,
                'metrics': {
                    'input': input_metrics,
                    'output': output_metrics,
                    'compression_ratio': output_metrics['length'] / input_metrics['length']
                },
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'summary': '',
                'metrics': input_metrics,
                'status': 'failed',
                'error': str(e)
            }

    def _process_long_text(self, chunks: List[str], style: str, max_length: int = None) -> str:
        """Process long text by summarizing chunks and combining."""
        # Summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            summary = self._generate_summary(chunk, style)
            chunk_summaries.append(summary)
        
        # Combine chunk summaries
        combined_text = " ".join(chunk_summaries)
        
        # Generate final summary
        return self._generate_summary(combined_text, style, max_length)

    def _generate_summary(self, text: str, style: str, max_length: int = None) -> str:
        """Generate summary for a piece of text."""
        prompt = self._create_summary_prompt(text, style, max_length)
        
        response = self.llm_client.complete(
            prompt=prompt,
            system_message="""You are an expert summarizer. Create clear, accurate 
            summaries while preserving key information and maintaining the original 
            text's meaning."""
        )
        
        return response.content.strip()
    
    def _create_summary_prompt(self, text: str, style: str, max_length: int = None) -> str:
        """Create prompt for summary generation."""
        length_constraint = f"Maximum length: {max_length} characters." if max_length else ""
        
        style_instructions = {
            'concise': 'Create a brief summary focusing on the most essential points.',
            'detailed': 'Provide a comprehensive summary including main points and key supporting details.',
            'bullet': 'Present the main points in a bulleted list format.',
            'explained': 'Create a summary that explains complex concepts in simple terms.'
        }.get(style, 'Create a clear and concise summary.')
        
        return f"""
        {style_instructions}
        
        {length_constraint}
        
        Text to summarize:
        {text}
        
        Requirements:
        - Preserve the main ideas and key information
        - Maintain factual accuracy
        - Use clear, concise language
        - Keep the original tone
        - Ensure logical flow
        """