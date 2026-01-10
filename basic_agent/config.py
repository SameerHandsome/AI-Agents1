"""
basic_agent/config.py - Basic Agent Configuration
NO advanced features, just essentials
"""

import os
from dotenv import load_dotenv

load_dotenv()


GROQ_API_KEY = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")


GROQ_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"