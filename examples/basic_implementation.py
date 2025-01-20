import sys
sys.path.append("/Users/vpdas/Documents/codes/genai/AgenticFramework/git/hoomanai/src")


from hoomanai.core.master_agent import MasterAgent, UserContext
from hoomanai.tools.llm.config import LLMConfig
from hoomanai.memory.redis_storage import RedisStorage
from hoomanai.agents.mini_agents.summarizer import SummarizerAgent
from hoomanai.agents.mini_agents.qna import QnAAgent
from typing import Dict, Any

class SummaryAgent(SummarizerAgent):
    """Extended summarizer agent with capabilities description"""
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": "summarizer",
            "description": "Generates concise, accurate summaries of text content",
            "input_types": {
                "text": "str",
                "style": "str",
                "max_length": "int"
            },
            "output_types": {
                "summary": "str",
                "metrics": "dict"
            },
            "supported_styles": ["concise", "detailed", "bullet", "explained"],
            "max_input_length": 10000,
            "languages": ["en"]
        }

class ContextQnAAgent(QnAAgent):
    """Extended QnA agent with capabilities description"""
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": "qna",
            "description": "Answers questions about the text to extract specific information",
            "input_types": {
                "text": "str",
                "question": "str"
            },
            "output_types": {
                "answer": "str",
                "confidence": "float"
            },
            "supported_question_types": ["factual", "analytical", "descriptive"],
            "max_input_length": 5000,
            "languages": ["en"]
        }

def main():
    # Initialize Redis storage for conversation memory
    redis_storage = RedisStorage(
        host="localhost",
        port=6379,
        db=0,
        prefix="demo:"
    )
    
    # Configure LLM
    llm_config = LLMConfig(
        base_url = "http://localhost:1234/v1",
        model_name = "llama-3.2-3b-instruct"
    )
    
    # Initialize Master Agent
    master = MasterAgent(
        conversation_memory=redis_storage,
        llm_config=llm_config
    )
    
    # Register available agents
    summarizer = SummaryAgent(llm_config=llm_config)
    qna_agent = ContextQnAAgent(llm_config=llm_config)
    
    master.register_mini_agent("summarizer", summarizer)
    master.register_mini_agent("qna", qna_agent)
    
    # Sample text for summarization
    sample_text = """
    Artificial Intelligence (AI) is revolutionizing industries across the globe. Machine learning, 
    a subset of AI, enables systems to learn from experience without explicit programming. Deep 
    learning, a specialized form of machine learning, uses neural networks with many layers to 
    analyze various factors of data. Natural Language Processing (NLP) is another crucial area 
    of AI that focuses on the interaction between computers and human language. These technologies 
    are being applied in various fields, from healthcare and finance to autonomous vehicles and 
    personal assistants. However, the rapid advancement of AI also raises important ethical 
    considerations regarding privacy, bias, and the future of human work. Researchers and 
    organizations are working to address these challenges while continuing to push the boundaries 
    of what AI can achieve.
    """
    
    # Create user context for comprehensive analysis
    user_context = UserContext(
        query="Analyze this text about AI and provide a comprehensive summary. Also extract key points about AI applications and ethical concerns.",
        additional_context={
            "text": sample_text,
            "requirements": {
                "summary_style": "detailed",
                "max_length": 150,
                "key_points_needed": True,
                "ethical_analysis_needed": True
            }
        }
    )
    
    try:
        # Create execution plan
        print("\nCreating execution plan...")
        plan = master.create_execution_plan(user_context)
        
        # Print plan details
        print("\nExecution Plan:")
        for i, task in enumerate(plan.tasks):
            print(f"\nTask {i+1}:")
            print(f"Description: {task.description}")
            print(f"Agent: {task.agent_type.value} - {task.agent_name}")
            print(f"Dependencies: {[str(dep) for dep in task.dependencies]}")
        
        # Execute plan
        print("\nExecuting plan...")
        results = master.execute_plan(plan)
        
        # Print results
        print("\nResults:")
        print("=" * 80)
        
        if results["execution_summary"]["failed_tasks"] == 0:
            print("\nAnalysis completed successfully!")
            
            # Access individual task results
            for task_id, task_result in results["final_result"].items():
                task = next((t for t in plan.tasks if str(t.id) == task_id), None)
                if task:
                    print(f"\n{task.description}:")
                    if task.agent_name == "summarizer":
                        print(f"Summary: {task_result['summary']}")
                        print(f"Compression ratio: {task_result['metrics']['compression_ratio']:.2%}")
                    elif task.agent_name == "qna":
                        print(f"Answer: {task_result['answer']}")
                        print(f"Confidence: {task_result['confidence']:.2%}")
        else:
            print("\nSome tasks failed during execution:")
            print(f"Completed tasks: {results['execution_summary']['completed_tasks']}")
            print(f"Failed tasks: {results['execution_summary']['failed_tasks']}")
            
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
    
    finally:
        # Clean up
        redis_storage.clear()

if __name__ == "__main__":
    main()