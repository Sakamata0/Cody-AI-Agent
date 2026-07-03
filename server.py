"""
Cody AI Agent — FastAPI REST API Server.

Run with: uvicorn server:app --reload --port 8000
"""

import sys
import os
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.messages import HumanMessage, AIMessage

from src.agent import create_agent, StepLoggingHandler, ALL_TOOLS
from src.llm import MODEL_ID, REGION

# --- App Setup ---
app = FastAPI(
    title="Cody AI Agent API",
    description="REST API for Cody — SMARTOVATE's autonomous AI agent powered by AWS Bedrock.",
    version="1.0.0",
)

# CORS — allow frontend to connect from any origin (restrict in production).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-memory session storage ---
sessions: dict[str, list] = {}


# --- Request/Response Models ---
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    steps: list
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    region: str
    tools: int


class ToolInfo(BaseModel):
    name: str
    description: str


# --- Endpoints ---
@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model=MODEL_ID,
        region=REGION,
        tools=len(ALL_TOOLS),
    )


@app.get("/tools", response_model=list[ToolInfo])
def list_tools():
    """List all available tools."""
    return [
        ToolInfo(name=tool.name, description=tool.description[:100])
        for tool in ALL_TOOLS
    ]


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Send a message to Cody and get a response.

    If session_id is provided, conversation history is maintained.
    If not, a new session is created.
    """
    # Manage session.
    session_id = request.session_id or f"session_{int(time.time())}"
    if session_id not in sessions:
        sessions[session_id] = []

    chat_history = sessions[session_id]

    # Run agent.
    executor = create_agent()
    logger = StepLoggingHandler()

    start = time.time()
    try:
        result = executor.invoke(
            {"input": request.message, "chat_history": chat_history},
            config={"callbacks": [logger]},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    latency = (time.time() - start) * 1000

    # Update session history.
    sessions[session_id].append(HumanMessage(content=request.message))
    sessions[session_id].append(AIMessage(content=result["output"]))

    return ChatResponse(
        response=result["output"],
        session_id=session_id,
        steps=logger.steps,
        latency_ms=round(latency, 0),
    )


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Clear a conversation session."""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"Session {session_id} cleared."}
    raise HTTPException(status_code=404, detail="Session not found.")


@app.get("/sessions")
def list_sessions():
    """List active sessions."""
    return {
        "sessions": [
            {"id": sid, "messages": len(msgs)}
            for sid, msgs in sessions.items()
        ]
    }
