"""
basic_agent/graph.py - Basic Agent Graph
Simple ReAct: LLM → Tools → Response (with Memory)
NO advanced features
"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
import sqlite3
from datetime import datetime

from config import *
from tools import ALL_TOOLS



class ChatState(TypedDict):
    """Basic state - just messages"""
    messages: Annotated[list[BaseMessage], add_messages]



class MemoryManager:
    """Simple memory with SQLite storage"""
    
    def __init__(self, db_path: str = "chat_history.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add message to history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )
        conn.commit()
        conn.close()
    
    def get_recent_messages(self, session_id: str, limit: int = 5):
        """Get recent messages"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """SELECT role, content FROM chat_history 
               WHERE session_id = ? 
               ORDER BY id DESC LIMIT ?""",
            (session_id, limit)
        )
        rows = cursor.fetchall()
        conn.close()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
    
    def get_context(self, session_id: str) -> str:
        """Get conversation context"""
        recent = self.get_recent_messages(session_id, limit=5)
        
        if recent:
            recent_text = "\n".join([f"{m['role']}: {m['content']}" for m in recent])
            return f"Recent conversation:\n{recent_text}"
        
        return ""



def create_agent():
    """
    Create basic agent with:
    ✅ Memory
    ✅ Tool calling

    """
    
    # Initialize LLM
    llm = ChatOpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        model=GROQ_MODEL,
        temperature=0
    )
    
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    
    memory = MemoryManager()
    
    def chat_node(state: ChatState):
        """Just invoke LLM, nothing else"""
        messages = state['messages']
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    graph = StateGraph(ChatState)
    
    graph.add_node("chat", chat_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    
    graph.add_edge(START, "chat")
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")
    
    return graph.compile(), memory



if __name__ == "__main__":
    agent, mem = create_agent()
    print("✅ Basic Agent created successfully")
    print("📊 Features: Memory, Tool Calling")
    print("❌ No advanced features")