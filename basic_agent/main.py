"""
basic_agent/main.py - Basic Agent FastAPI
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
import sqlite3
from contextlib import asynccontextmanager

from graph import create_agent
from config import *


agent_graph = None
memory_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler"""
    global agent_graph, memory_manager
    
    print("🚀 Starting Basic Agent...")
    agent_graph, memory_manager = create_agent()
    print(" Basic Agent initialized")
    print(" Features: Memory, Tool Calling")
    print(" No advanced features (retry, budget, guardrails, etc.)")
    
    yield
    
    print("Shutting down...")



app = FastAPI(
    title="Basic Agent API",
    description="Simple AI Agent with memory and tool calling",
    version="1.0.0",
    lifespan=lifespan
)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    features: List[str]
    database_status: str


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Basic Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
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
            "conversation_memory",
            "tool_calling"
        ],
        database_status=db_status
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Basic chat endpoint with memory
    
    Features:
    - Conversation memory
    - Tool calling
    
    No advanced features (retry, validation, etc.)
    """
    if not agent_graph or not memory_manager:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        context = memory_manager.get_context(request.session_id)
        
        messages = []
        if context:
            messages.append(SystemMessage(content=context))
        messages.append(HumanMessage(content=request.message))
        
        result = agent_graph.invoke({"messages": messages})
        
        response_content = result['messages'][-1].content
        
        memory_manager.add_message(request.session_id, "user", request.message)
        memory_manager.add_message(request.session_id, "assistant", response_content)
        
        return ChatResponse(
            response=response_content,
            session_id=request.session_id,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.get("/history/{session_id}", tags=["History"])
async def get_history(session_id: str, limit: int = 10):
    """Get chat history for a session"""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="Memory manager not initialized")
    
    try:
        messages = memory_manager.get_recent_messages(session_id, limit)
        return {
            "session_id": session_id,
            "messages": messages,
            "count": len(messages)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@app.delete("/history/{session_id}", tags=["History"])
async def clear_history(session_id: str):
    """Clear chat history for a session"""
    try:
        conn = sqlite3.connect("chat_history.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
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


if __name__ == "__main__":
    import uvicorn
    
    print("="*60)
    print("🤖 Basic Agent API")
    print("="*60)
    print()
    print("📝 Make sure to set environment variables:")
    print("   export OPENAI_API_KEY='your-groq-key'")
    print("   export TAVILY_API_KEY='your-tavily-key'")
    print()
    print("🚀 Starting server on port 8001...")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8001)