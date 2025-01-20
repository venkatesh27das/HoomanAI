import sys
sys.path.append("/Users/vpdas/Documents/codes/genai/AgenticFramework/git/backuo_hooman/src")


from hoomanai.core import MasterAgent
from hoomanai.agents.mini_agents import SummarizerAgent, QnAAgent

# Create instances
master = MasterAgent()
summarizer = SummarizerAgent()
qna = QnAAgent()