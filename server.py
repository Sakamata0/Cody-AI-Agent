"""
Cody AI Agent — FastAPI REST API Server.

Run with: uvicorn server:app --reload --port 8000
"""

import sys
import os
import time
import json
import asyncio
from typing import Optional
from queue import Queue
from threading import Thread

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.middleware import get_current_user
from src.auth import router as auth_router
from src.settings import router as settings_router
from src.migration import router as migration_router
from src.usage import router as usage_router, increment_usage

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler

from src.agent import create_agent, StepLoggingHandler, ALL_TOOLS
from src.llm import MODEL_ID, REGION
from src.storage import (
    list_conversations, load_conversation, save_conversation,
    delete_conversation, new_conversation_id, generate_title,
)

# --- App Setup ---
app = FastAPI(
    title="Cody AI Agent API",
    description="REST API for Cody — SMARTOVATE's autonomous AI agent powered by AWS Bedrock.",
    version="1.0.0",
)

# CORS — restrict origins to known frontends.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://16.192.58.80:3000",
        "https://cody.formaa.studio",
        os.environ.get("FRONTEND_ORIGIN", "http://localhost:3000"),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Register Routers ---
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(settings_router, prefix="/settings", tags=["Settings"])
app.include_router(usage_router, prefix="/usage", tags=["Usage"])
app.include_router(migration_router, prefix="/admin", tags=["Administration"])


# --- Global exception handler for CORS on 500 errors ---
from fastapi.responses import JSONResponse
from starlette.requests import Request


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions so CORS headers are still applied."""
    print(f"[ERROR] Unhandled exception on {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
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
def chat(request: ChatRequest, user_id: str = Depends(get_current_user)):
    """
    Send a message to Cody and get a response.

    - session_id: identifies the conversation (for multi-turn memory)
    """
    # Check usage limit
    if not increment_usage(user_id):
        raise HTTPException(
            status_code=429,
            detail="Weekly message limit reached. Contact support to extend your limit.",
        )

    session_id = request.session_id or new_conversation_id()

    # Use user-scoped session key for in-memory storage.
    session_key = f"{user_id}:{session_id}"

    # Load or create session.
    if session_key not in sessions:
        sessions[session_key] = []

    chat_history = sessions[session_key]

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

    # Update in-memory session.
    sessions[session_key].append(HumanMessage(content=request.message))
    sessions[session_key].append(AIMessage(content=result["output"]))

    # Persist to S3.
    messages_to_save = []
    for msg in sessions[session_key]:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        messages_to_save.append({"role": role, "content": msg.content})

    # Add steps to the last assistant message
    if messages_to_save and messages_to_save[-1]["role"] == "assistant":
        messages_to_save[-1]["steps"] = logger.steps
        messages_to_save[-1]["latency_ms"] = round(latency, 0)

    title = generate_title(messages_to_save[0]["content"]) if len(messages_to_save) == 2 else None
    if title:
        save_conversation(user_id, session_id, title, messages_to_save)
    else:
        # Load existing title.
        existing = load_conversation(user_id, session_id)
        t = existing["title"] if existing else "Chat"
        save_conversation(user_id, session_id, t, messages_to_save)

    return ChatResponse(
        response=result["output"],
        session_id=session_id,
        steps=logger.steps,
        latency_ms=round(latency, 0),
    )


class StreamingStepHandler(BaseCallbackHandler):
    """Callback handler that pushes events to a queue for SSE streaming."""

    def __init__(self, queue: Queue):
        self.queue = queue
        self.steps = []
        self.step_count = 0

    def on_agent_action(self, action, **kwargs):
        self.step_count += 1
        step = {
            "step": self.step_count,
            "type": "action",
            "tool": action.tool,
            "input": action.tool_input if isinstance(action.tool_input, str) else str(action.tool_input),
        }
        self.steps.append(step)
        self.queue.put({"event": "step", "data": step})

    def on_tool_end(self, output, **kwargs):
        output_str = str(output)
        step = {
            "step": self.step_count,
            "type": "observation",
            "output": output_str,
        }
        self.steps.append(step)
        self.queue.put({"event": "observation", "data": step})

    def on_agent_finish(self, finish, **kwargs):
        pass


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, user_id: str = Depends(get_current_user)):
    """
    Stream agent responses via Server-Sent Events.
    Sends step-by-step progress, then the final answer.
    """
    # Check usage limit
    if not increment_usage(user_id):
        raise HTTPException(
            status_code=429,
            detail="Weekly message limit reached. Contact support to extend your limit.",
        )

    session_id = request.session_id or new_conversation_id()

    # Use user-scoped session key for in-memory storage.
    session_key = f"{user_id}:{session_id}"

    if session_key not in sessions:
        sessions[session_key] = []

    chat_history = sessions[session_key]

    async def event_generator():
        queue = Queue()
        handler = StreamingStepHandler(queue)

        # Send session_id immediately
        yield f"data: {json.dumps({'event': 'session', 'session_id': session_id})}\n\n"

        # Run agent in a thread so we can stream events
        result_holder = {}

        def run_agent():
            executor = create_agent()
            start = time.time()
            try:
                result = executor.invoke(
                    {"input": request.message, "chat_history": chat_history},
                    config={"callbacks": [handler]},
                )
                result_holder["output"] = result["output"]
                result_holder["latency_ms"] = (time.time() - start) * 1000
            except Exception as e:
                result_holder["output"] = f"Error: {str(e)}"
                result_holder["latency_ms"] = (time.time() - start) * 1000
            finally:
                queue.put(None)  # Signal done

        thread = Thread(target=run_agent, daemon=True)
        thread.start()

        # Stream events from the queue
        while True:
            await asyncio.sleep(0.05)
            while not queue.empty():
                item = queue.get()
                if item is None:
                    # Agent finished — send final response
                    output = result_holder.get("output", "")
                    latency = result_holder.get("latency_ms", 0)

                    # Update session
                    sessions[session_key].append(HumanMessage(content=request.message))
                    sessions[session_key].append(AIMessage(content=output))

                    # Persist to S3
                    messages_to_save = []
                    for msg in sessions[session_key]:
                        role = "user" if isinstance(msg, HumanMessage) else "assistant"
                        messages_to_save.append({"role": role, "content": msg.content})

                    # Add steps to the last assistant message
                    if messages_to_save and messages_to_save[-1]["role"] == "assistant":
                        messages_to_save[-1]["steps"] = handler.steps
                        messages_to_save[-1]["latency_ms"] = round(latency, 0)

                    title = generate_title(messages_to_save[0]["content"]) if len(messages_to_save) == 2 else None
                    if title:
                        save_conversation(user_id, session_id, title, messages_to_save)
                    else:
                        existing = load_conversation(user_id, session_id)
                        t = existing["title"] if existing else "Chat"
                        save_conversation(user_id, session_id, t, messages_to_save)

                    # Send final answer
                    yield f"data: {json.dumps({'event': 'done', 'response': output, 'steps': handler.steps, 'latency_ms': round(latency, 0)})}\n\n"
                    return
                else:
                    yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/conversations")
def get_conversations(user_id: str = Depends(get_current_user)):
    """List all conversations for the authenticated user."""
    return list_conversations(user_id)


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str, user_id: str = Depends(get_current_user)):
    """Load a specific conversation. Returns 403 if it doesn't belong to the user."""
    data = load_conversation(user_id, conversation_id)
    if not data:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return data


@app.delete("/conversations/{conversation_id}")
def remove_conversation(conversation_id: str, user_id: str = Depends(get_current_user)):
    """Delete a conversation. Returns 403 if it doesn't belong to the user."""
    # Verify the conversation belongs to this user before deleting.
    data = load_conversation(user_id, conversation_id)
    if not data:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    deleted = delete_conversation(user_id, conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"message": "Deleted."}


class RenameRequest(BaseModel):
    title: str


@app.patch("/conversations/{conversation_id}")
def rename_conversation(conversation_id: str, request: RenameRequest, user_id: str = Depends(get_current_user)):
    """Rename a conversation. Returns 403 if it doesn't belong to the user."""
    data = load_conversation(user_id, conversation_id)
    if not data:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    save_conversation(user_id, conversation_id, request.title, data["messages"])
    return {"message": "Renamed.", "title": request.title}


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
