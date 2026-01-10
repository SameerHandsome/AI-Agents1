"""
graph.py - LangGraph Agent with All Features
"""

from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
import os
from datetime import datetime
from collections import defaultdict
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor

from config import *
from tools import ALL_TOOLS


class ChatState(TypedDict):
    """Agent state with tracking"""
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    tool_calls_count: int
    reflection_count: int
    start_time: float

class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self):
        self.tracker = defaultdict(lambda: defaultdict(list))
    
    def check(self, session_id: str, tool_name: str) -> bool:
        """Check if rate limit is exceeded"""
        now = datetime.now().timestamp()
        hour_ago = now - 3600
        
        self.tracker[session_id][tool_name] = [
            t for t in self.tracker[session_id][tool_name] 
            if t > hour_ago
        ]
        
        limit = RATE_LIMITS.get(tool_name, 100)
        if len(self.tracker[session_id][tool_name]) >= limit:
            return False
        
        self.tracker[session_id][tool_name].append(now)
        return True


rate_limiter = RateLimiter()


def check_guardrails(message: str) -> tuple[bool, str]:
    """Check input against security patterns"""
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, message, re.IGNORECASE):
            return False, "Blocked: Potential prompt injection detected"
    return True, ""



class MemoryManager:
    """Manages chat history with SQLite storage"""
    
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
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_summaries (
                session_id TEXT PRIMARY KEY,
                summary TEXT,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
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
    
    def summarize_history(self, session_id: str, llm) -> str:
        """Generate conversation summary"""
        messages = self.get_recent_messages(session_id, limit=20)
        
        if not messages:
            return ""
        
        chat_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        summary_prompt = f"""Summarize this conversation in 3-4 sentences, focusing on key topics and context:

{chat_text}

Summary:"""
        
        try:
            response = llm.invoke([HumanMessage(content=summary_prompt)])
            summary = response.content
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """INSERT OR REPLACE INTO session_summaries (session_id, summary, last_updated) 
                   VALUES (?, ?, CURRENT_TIMESTAMP)""",
                (session_id, summary)
            )
            conn.commit()
            conn.close()
            
            return summary
        except:
            return ""
    
    def get_context(self, session_id: str) -> str:
        """Get conversation context"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT summary FROM session_summaries WHERE session_id = ?",
            (session_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        context_parts = []
        
        if result:
            context_parts.append(f"Previous conversation summary:\n{result[0]}\n")
        
        recent = self.get_recent_messages(session_id, limit=5)
        if recent:
            recent_text = "\n".join([f"{m['role']}: {m['content']}" for m in recent])
            context_parts.append(f"Recent messages:\n{recent_text}")
        
        return "\n".join(context_parts)


def create_agent():
    """Create the agent graph"""
    
    llm = ChatOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    model=GROQ_MODEL,
    temperature=0
    )
    
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    
    memory = MemoryManager()
    

    def route_complexity(state: ChatState) -> Literal["planner", "chat_node"]:
        """Route complex queries through planner"""
        query = state['messages'][-1].content
        word_count = len(query.split())
        has_multiple_questions = query.count('?') > 1
        
        if word_count > COMPLEX_QUERY_WORDS or has_multiple_questions:
            return "planner"
        
        return "chat_node"
    
    def planner_node(state: ChatState):
        """Create execution plan for complex queries"""
        query = state['messages'][-1].content
        
        plan_prompt = PLANNER_PROMPT.format(query=query)
        plan_response = llm.invoke([HumanMessage(content=plan_prompt)])
        
        if "SIMPLE_TASK" in plan_response.content:
            return state
        
        plan_msg = SystemMessage(
            content=f" Execution Plan:\n{plan_response.content}\n\nFollow this plan step by step."
        )
        
        return {"messages": [plan_msg]}
    

    def chat_node(state: ChatState):
        """Main LLM node with budget checks and guardrails"""
        
        if state.get('tool_calls_count', 0) >= MAX_TOOL_CALLS:
            return {
                "messages": [AIMessage(content="Budget exceeded: Maximum tool calls reached")]
            }
        
        elapsed = datetime.now().timestamp() - state.get('start_time', datetime.now().timestamp())
        if elapsed > MAX_EXECUTION_TIME:
            return {
                "messages": [AIMessage(content="Budget exceeded: Maximum execution time reached")]
            }
        
        last_msg = state['messages'][-1]
        if isinstance(last_msg, HumanMessage):
            is_safe, reason = check_guardrails(last_msg.content)
            if not is_safe:
                return {"messages": [AIMessage(content=reason)]}
        
        response = llm_with_tools.invoke(state['messages'])
        
        new_count = state.get('tool_calls_count', 0)
        if hasattr(response, 'tool_calls') and response.tool_calls:
            new_count += len(response.tool_calls)
        
        return {
            "messages": [response],
            "tool_calls_count": new_count
        }
    
    
    def parallel_tool_node(state: ChatState):
        """Execute multiple tools in parallel with rate limiting"""
        last_msg = state['messages'][-1]
        
        if not hasattr(last_msg, 'tool_calls') or not last_msg.tool_calls:
            return {"messages": []}
        
        session_id = state.get('session_id', 'default')
        
        for tool_call in last_msg.tool_calls:
            if not rate_limiter.check(session_id, tool_call['name']):
                return {
                    "messages": [ToolMessage(
                        content=f"⏱️ Rate limit exceeded for {tool_call['name']}. Try again later.",
                        tool_call_id=tool_call['id']
                    )]
                }
        
        def execute_tool(tool_call):
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            
            for tool in ALL_TOOLS:
                if tool.name == tool_name:
                    result = tool.invoke(tool_args)
                    return ToolMessage(content=result, tool_call_id=tool_call['id'])
            
            return ToolMessage(
                content=f"Tool {tool_name} not found",
                tool_call_id=tool_call['id']
            )
        
        # Run concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(execute_tool, last_msg.tool_calls))
        
        return {"messages": results}
    
    
    def reflection_node(state: ChatState):
        """Self-critique and regenerate if needed"""
        
        reflection_count = state.get('reflection_count', 0)
        
        # Max reflections to avoid loops
        if reflection_count >= MAX_REFLECTIONS:
            return state
        
        # Get last AI response
        messages = state['messages']
        last_response = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not hasattr(msg, 'tool_calls'):
                last_response = msg.content
                break
        
        if not last_response:
            return state
        
        # Reflect
        reflection_prompt = REFLECTION_PROMPT.format(response=last_response)
        reflection = llm.invoke([HumanMessage(content=reflection_prompt)])
        
        # If approved, done
        if "APPROVED" in reflection.content.upper():
            return state
        
        # Regenerate with feedback
        improvement_msg = HumanMessage(
            content=f"🔄 Previous response needs improvement:\n{reflection.content}\n\nProvide a better response."
        )
        
        new_messages = messages[:-1] + [improvement_msg]
        
        return {
            "messages": new_messages,
            "reflection_count": reflection_count + 1
        }
    
    # ========================================================================
    # ROUTING LOGIC
    # ========================================================================
    
    def after_chat_routing(state: ChatState) -> Literal["tools", "reflection", END]:
        """Route after chat node"""
        last_msg = state['messages'][-1]
        
        # If tool calls, go to tools
        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            return "tools"
        
        # If budget exceeded or error, end
        if last_msg.content.startswith("⚠️"):
            return END
        
        # Otherwise reflect
        return "reflection"
    
    def after_reflection_routing(state: ChatState) -> Literal["chat_node", END]:
        """Route after reflection"""
        # If needs regeneration
        if state['messages'][-1].content.startswith("🔄"):
            return "chat_node"
        
        return END

    graph = StateGraph(ChatState)
    
    graph.add_node("planner", planner_node)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", parallel_tool_node)
    graph.add_node("reflection", reflection_node)
    
    graph.add_conditional_edges(START, route_complexity)
    graph.add_edge("planner", "chat_node")
    graph.add_conditional_edges("chat_node", after_chat_routing)
    graph.add_edge("tools", "chat_node")
    graph.add_conditional_edges("reflection", after_reflection_routing)
    
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)
    
    return compiled, memory




if __name__ == "__main__":
    agent, mem = create_agent()
    print("Agent created successfully")
    print(f"Features: Retry, Budget, Guardrails, Parallel Tools, Reflection, Validation, Planning")