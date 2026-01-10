"""
main.py - FastAPI Application
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
import sqlite3

from graph import create_agent
from config import *

app = FastAPI(
    title="LangGraph Agent API",
    description="AI Agent with retry, budget limits, guardrails, parallel tools, reflection, validation, and planning",
    version="2.0.0"
)
agent_graph = None
memory_manager = None


@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    global agent_graph, memory_manager
    
    print("🚀 Starting LangGraph Agent...")
    agent_graph, memory_manager = create_agent()
    print("✅ Agent initialized successfully")
    print(f"📊 Features enabled:")
    print(f"   - ✅ Retry Logic (3 attempts)")
    print(f"   - ✅ Budget Limits (max {MAX_TOOL_CALLS} tools, {MAX_EXECUTION_TIME}s)")
    print(f"   - ✅ Guardrails (prompt injection protection)")
    print(f"   - ✅ Parallel Tool Execution")
    print(f"   - ✅ Self-Reflection (max {MAX_REFLECTIONS} iterations)")
    print(f"   - ✅ Output Validation")
    print(f"   - ✅ Strategic Planning")


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str
    metadata: dict


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    features: List[str]
    database_status: str
    rate_limits: dict


class HistoryResponse(BaseModel):
    session_id: str
    messages: List[dict]
    count: int


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "LangGraph Agent API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    
    Returns system status and available features
    """
    try:
        conn = sqlite3.connect("chat_history.db")
        conn.close()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return HealthResponse(
        status="healthy" if agent_graph else "initializing",
        timestamp=datetime.utcnow().isoformat(),
        features=[
            "retry_logic",
            "budget_limits",
            "guardrails",
            "parallel_tools",
            "reflection",
            "output_validation",
            "strategic_planning",
            "rate_limiting",
            "conversation_memory"
        ],
        database_status=db_status,
        rate_limits=RATE_LIMITS
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Main chat endpoint with full agent features
    
    Features:
    - Automatic retry on failures
    - Budget limits for safety
    - Guardrails against prompt injection
    - Parallel tool execution
    - Self-reflection for quality
    - Output validation
    - Strategic planning for complex queries
    - Rate limiting per tool
    - Conversation memory with summarization
    
    Args:
        request: ChatRequest with message and session_id
        
    Returns:
        ChatResponse with agent response and metadata
    """
    if not agent_graph or not memory_manager:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        context = memory_manager.get_context(request.session_id)
        
        messages = []
        if context:
            messages.append(SystemMessage(content=f"Context:\n{context}"))
        messages.append(HumanMessage(content=request.message))
        
        initial_state = {
            "messages": messages,
            "session_id": request.session_id,
            "start_time": datetime.now().timestamp(),
            "tool_calls_count": 0,
            "reflection_count": 0
        }
        
        config = {"configurable": {"thread_id": request.session_id}}
        result = agent_graph.invoke(initial_state, config)
        
        response_content = result['messages'][-1].content
        
        memory_manager.add_message(request.session_id, "user", request.message)
        memory_manager.add_message(request.session_id, "assistant", response_content)
        
        metadata = {
            "tool_calls_used": result.get('tool_calls_count', 0),
            "reflections": result.get('reflection_count', 0),
            "budget_status": {
                "tools_remaining": MAX_TOOL_CALLS - result.get('tool_calls_count', 0),
                "reflections_remaining": MAX_REFLECTIONS - result.get('reflection_count', 0)
            }
        }
        
        return ChatResponse(
            response=response_content,
            session_id=request.session_id,
            timestamp=datetime.utcnow().isoformat(),
            metadata=metadata
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.get("/history/{session_id}", response_model=HistoryResponse, tags=["History"])
async def get_history(session_id: str, limit: int = 10):
    """
    Get chat history for a session
    
    Args:
        session_id: Session identifier
        limit: Maximum number of messages to return (default: 10)
        
    Returns:
        HistoryResponse with message history
    """
    if not memory_manager:
        raise HTTPException(status_code=503, detail="Memory manager not initialized")
    
    try:
        messages = memory_manager.get_recent_messages(session_id, limit)
        return HistoryResponse(
            session_id=session_id,
            messages=messages,
            count=len(messages)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@app.delete("/history/{session_id}", tags=["History"])
async def clear_history(session_id: str):
    """
    Clear chat history for a session
    
    Args:
        session_id: Session identifier
        
    Returns:
        Success message
    """
    try:
        conn = sqlite3.connect("chat_history.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM session_summaries WHERE session_id = ?", (session_id,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        return {
            "status": "success",
            "message": f"History cleared for session {session_id}",
            "deleted_messages": deleted
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")


@app.get("/stats", tags=["Stats"])
async def get_stats():
    """
    Get system statistics
    
    Returns:
        System stats including total sessions, messages, etc.
    """
    try:
        conn = sqlite3.connect("chat_history.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM chat_history")
        total_messages = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT session_id) FROM chat_history")
        total_sessions = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM chat_history 
            WHERE timestamp > datetime('now', '-1 day')
        """)
        recent_messages = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_messages": total_messages,
            "total_sessions": total_sessions,
            "recent_messages_24h": recent_messages,
            "features_active": 9,
            "rate_limits": RATE_LIMITS
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    print("="*60)
    print("🤖 LangGraph Agent API")
    print("="*60)
    print("🚀 Starting server...")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)