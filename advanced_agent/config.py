"""
config.py - Configuration and Settings
"""

import os
from dotenv import load_dotenv
load_dotenv()


GROQ_API_KEY = os.getenv("OPENAI_API_KEY", "") 
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")


GROQ_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"

MAX_TOKENS = 200000
MAX_TOOL_CALLS = 15
MAX_EXECUTION_TIME = 70  
MAX_REFLECTIONS = 2

RATE_LIMITS = {
    "tavily_search": 10,
    "get_weather": 30,
    "convert_currency": 50,
    "get_wikipedia_summary": 20,
    "get_world_time": 100
}


BLOCKED_PATTERNS = [
    r'ignore (previous|above|all) instructions',
    r'you are now',
    r'disregard',
    r'new instructions',
    r'system prompt',
    r'forget (everything|all)',
]

REFLECTION_PROMPT = """Review this response for quality:

{response}

Check:
1. Accuracy - Is information correct?
2. Completeness - Does it fully answer the question?
3. Safety - Is it appropriate and harmless?

Reply 'APPROVED' if good, or explain specific issues to fix."""

PLANNER_PROMPT = """Break down this task into clear, sequential steps:

Task: {query}

Provide a numbered list of steps needed to complete this task.
Each step should be actionable and specific.
If the task is simple (single question, one action needed), just say "SIMPLE_TASK"."""

COMPLEX_QUERY_WORDS = 20 