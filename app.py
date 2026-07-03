"""
Cody AI Agent — Streamlit Chat Interface.

Run with: streamlit run app.py
"""

import sys
import os
import time
import base64
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from src.llm import chat_model, MODEL_ID, REGION
from src.agent import create_agent, StepLoggingHandler, ALL_TOOLS
from src.storage import (
    list_conversations, load_conversation, save_conversation,
    delete_conversation, new_conversation_id, generate_title,
)

# --- Page Config ---
st.set_page_config(
    page_title="Cody AI Agent",
    page_icon="🤖",
    layout="wide",
)

# --- Custom CSS for chat layout ---
st.markdown("""
<style>
/* User messages: right-aligned, no avatar */
.stChatMessage[data-testid="stChatMessage"]:has(.stMarkdown) {
}
div[data-testid="stChatMessage"][data-testid*="user"] {
    flex-direction: row-reverse;
}
/* Hide user avatar */
[data-testid="stChatMessageAvatarUser"],
[data-testid="chatAvatarIcon-user"],
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) > div:first-child {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
}
/* Blue theme for input focus outlines */
[data-testid="stChatInput"] > div {
    border-color: #3B82F6 !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 1px #3B82F6 !important;
}
/* Remove double border on bottom container */
[data-testid="stBottom"] > div {
    border: none !important;
    box-shadow: none !important;
}
/* Text inputs */
input:focus, textarea:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 1px #3B82F6 !important;
}
/* Buttons */
/* Chat submit button - match blue theme */
[data-testid="stChatInput"] button {
    background-color: #3B82F6 !important;
    border-color: #3B82F6 !important;
}
[data-testid="stChatInput"] button:hover {
    background-color: #2563EB !important;
    border-color: #2563EB !important;
}
</style>
""", unsafe_allow_html=True)

# --- Session State Init ---
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = new_conversation_id()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "title" not in st.session_state:
    st.session_state.title = "New Chat"


# --- Helper Functions ---
def start_new_chat():
    st.session_state.conversation_id = new_conversation_id()
    st.session_state.messages = []
    st.session_state.title = "New Chat"


def load_chat(conversation_id: str):
    data = load_conversation(conversation_id)
    if data:
        st.session_state.conversation_id = data["id"]
        st.session_state.messages = data["messages"]
        st.session_state.title = data["title"]


def save_current_chat():
    if st.session_state.messages:
        save_conversation(
            st.session_state.conversation_id,
            st.session_state.title,
            st.session_state.messages,
        )


def format_latency(ms: float) -> str:
    """Format milliseconds into human-readable time."""
    seconds = int(ms / 1000)
    if seconds < 1:
        return f"{ms:.0f}ms"
    minutes = seconds // 60
    secs = seconds % 60
    if minutes == 0:
        return f"{secs}s"
    hours = minutes // 60
    mins = minutes % 60
    if hours == 0:
        return f"{mins}m {secs}s"
    return f"{hours}h {mins}m {secs}s"


def get_greeting() -> str:
    """Generate a time-based greeting."""
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    elif hour < 18:
        return "Good afternoon"
    else:
        return "Good evening"


def get_logo_base64() -> str:
    """Load logo as base64 for inline HTML."""
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "cody.png")
    with open(logo_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# --- Sidebar: Conversation History ---
with st.sidebar:
    # Logo + name side by side (no model info)
    logo_b64 = get_logo_base64()
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
            <img src="data:image/png;base64,{logo_b64}" width="28" />
            <span style="font-size: 1.3rem; font-weight: bold;">Cody</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    if st.button("➕ New Chat", use_container_width=True):
        start_new_chat()
        st.rerun()

    st.markdown("### Conversations")
    conversations = list_conversations()

    if not conversations:
        st.caption("No saved conversations yet.")
    else:
        for conv in conversations:
            col1, col2 = st.columns([5, 1])
            with col1:
                display_title = conv['title'][:20] + "..." if len(conv['title']) > 20 else conv['title']
                if st.button(
                    f"💬 {display_title}",
                    key=f"load_{conv['id']}",
                    use_container_width=True,
                ):
                    load_chat(conv["id"])
                    st.rerun()
            with col2:
                with st.popover("", use_container_width=True):
                    # If currently renaming this conversation, show input.
                    if st.session_state.get("renaming_id") == conv["id"]:
                        new_name = st.text_input(
                            "New name",
                            value=conv["title"],
                            key=f"rename_input_{conv['id']}",
                            label_visibility="collapsed",
                        )
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("Save", key=f"save_{conv['id']}", use_container_width=True):
                                data = load_conversation(conv["id"])
                                if data:
                                    save_conversation(conv["id"], new_name, data["messages"])
                                    if st.session_state.conversation_id == conv["id"]:
                                        st.session_state.title = new_name
                                st.session_state.pop("renaming_id", None)
                                st.rerun()
                        with c2:
                            if st.button("Cancel", key=f"cancel_{conv['id']}", use_container_width=True):
                                st.session_state.pop("renaming_id", None)
                                st.rerun()
                    else:
                        # Show options.
                        if st.button("✏️ Rename", key=f"rename_btn_{conv['id']}", use_container_width=True):
                            st.session_state.renaming_id = conv["id"]
                            st.rerun()
                        if st.button("🗑️ Delete", key=f"del_{conv['id']}", use_container_width=True):
                            delete_conversation(conv["id"])
                            if st.session_state.conversation_id == conv["id"]:
                                start_new_chat()
                            st.rerun()

    st.divider()
    # Agent Info button — pinned to bottom with CSS
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='flex-grow: 1;'></div>", unsafe_allow_html=True)
    if st.button("⚙️ Agent Info", use_container_width=True):
        st.session_state.show_agent_info = True
        st.rerun()


# --- Agent Info Dialog ---
@st.dialog("Agent Info")
def show_agent_info_dialog():
    logo_b64 = get_logo_base64()
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
            <img src="data:image/png;base64,{logo_b64}" width="35" />
            <span style="font-size: 1.3rem; font-weight: bold;">Cody</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"**Model:** `{MODEL_ID}`")
    st.markdown(f"**Region:** `{REGION}`")
    st.divider()
    st.markdown(f"**Tools available:** {len(ALL_TOOLS)}")
    for tool in ALL_TOOLS:
        st.markdown(f"- `{tool.name}` — {tool.description[:60]}...")
    st.divider()
    st.markdown(f"**Messages in session:** {len(st.session_state.messages)}")
    st.markdown("**Built by:** SMARTOVATE LTD")
    st.markdown("**Developed by:** Skander Boughnimi")

if st.session_state.get("show_agent_info"):
    st.session_state.pop("show_agent_info")
    show_agent_info_dialog()


# --- Main Chat Area ---

# If no messages yet, show the welcome/greeting screen (like Claude).
if not st.session_state.messages:
    # Push content to vertical center.
    st.markdown("<div style='height: 30vh;'></div>", unsafe_allow_html=True)

    greeting = get_greeting()
    logo_b64 = get_logo_base64()

    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; align-items: center; gap: 15px; margin-bottom: 30px;">
            <img src="data:image/png;base64,{logo_b64}" width="45" />
            <span style="font-size: 2rem; font-weight: 600;">{greeting}, Skander</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Centered narrower input area style.
    st.markdown(
        """
        <style>
        .stChatInput {
            max-width: 600px;
            margin: 0 auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

else:
    # Display message history.
    for msg in st.session_state.messages:
        role = msg["role"]
        if role == "assistant":
            with st.chat_message("assistant", avatar="assets/cody.png"):
                st.markdown(msg["content"])
                if msg.get("steps"):
                    tool_steps = [s for s in msg["steps"] if s["type"] == "action"]
                    if tool_steps:
                        with st.expander("🧠 Reasoning Steps", expanded=False):
                            for step in msg["steps"]:
                                if step["type"] == "action":
                                    st.markdown(f"**Action:** `{step['tool']}({step['input']})`")
                                elif step["type"] == "observation":
                                    obs = step["output"][:300]
                                    st.markdown(f"**Observation:** {obs}")
                if msg.get("latency_ms"):
                    tokens_text = ""
                    if msg.get("tokens"):
                        tokens_text = f" • Tokens: {msg['tokens']}"
                    st.markdown(
                        f"<span style='color: #888; font-size: 0.8rem;'>Elapsed time: {format_latency(msg['latency_ms'])}{tokens_text}</span>",
                        unsafe_allow_html=True,
                    )
        else:
            with st.chat_message("user"):
                st.markdown(msg["content"])

# Chat input — different placeholder depending on state.
placeholder = "How can I help you today?" if not st.session_state.messages else "Write a message..."
if prompt := st.chat_input(placeholder):
    # Add user message to state first.
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate title from first message.
    if st.session_state.title == "New Chat":
        st.session_state.title = generate_title(prompt)

    # Set a flag to process the message after rerun (so greeting disappears).
    st.session_state.pending_prompt = prompt
    st.rerun()

# Process pending prompt (after rerun cleared the greeting).
if st.session_state.get("pending_prompt"):
    prompt = st.session_state.pop("pending_prompt")

    with st.chat_message("assistant", avatar="assets/cody.png"):
        # Use st.status for real-time step display.
        status_container = st.status("Cody is thinking...", expanded=False)
        with status_container as status:
            chat_history = []
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    chat_history.append(AIMessage(content=msg["content"]))

            executor = create_agent()

            # Custom callback that writes to the status widget in real time.
            class StreamlitStepLogger(StepLoggingHandler):
                def __init__(self):
                    super().__init__()
                    self.step_placeholders = []

                def on_agent_action(self, action, **kwargs):
                    super().on_agent_action(action, **kwargs)
                    tool_labels = {
                        "web_search": "🔍 Searching the web...",
                        "sql_query": "🗄️ Querying database...",
                        "weather_tool": "🌤️ Checking weather...",
                        "exchange_rate_tool": "💱 Converting currency...",
                        "code_executor_tool": "💻 Executing code...",
                        "disaster_predictor_tool": "🌪️ Analyzing disaster risk...",
                    }
                    label = tool_labels.get(action.tool, f"⚙️ Using {action.tool}...")
                    status.update(label=label, expanded=True)
                    # Mark previous step as done.
                    if self.step_placeholders:
                        prev = self.step_placeholders[-1]
                        prev_text = prev["text"]
                        prev["placeholder"].markdown(f"✅ {prev_text}")
                    # Add new step as in-progress.
                    placeholder = st.empty()
                    placeholder.markdown(f"⏳ {label}")
                    self.step_placeholders.append({"placeholder": placeholder, "text": label})

                def on_tool_end(self, output, **kwargs):
                    super().on_tool_end(output, **kwargs)
                    # Mark current step as done.
                    if self.step_placeholders:
                        current = self.step_placeholders[-1]
                        done_label = current["text"].replace("...", "")
                        current["placeholder"].markdown(f"✅ {done_label} ✓")

            ui_logger = StreamlitStepLogger()

            start = time.time()
            result = executor.invoke(
                {"input": prompt, "chat_history": chat_history},
                config={"callbacks": [ui_logger]},
            )
            latency = (time.time() - start) * 1000

            # If tools were used, collapse status. Otherwise hide it.
            tool_steps = [s for s in ui_logger.steps if s["type"] == "action"]
            if tool_steps:
                status.update(label="Done", state="complete", expanded=False)
            else:
                status.update(label="", state="complete", expanded=False)

        # Hide status entirely if no tools were used.
        if not tool_steps:
            status_container.empty()

        # Show final answer.
        st.markdown(result["output"])
        if tool_steps:
            st.caption(f"⏱️ {latency:.0f}ms | {ui_logger.step_count} steps")

    # Extract token usage from the result metadata if available.
    tokens_used = 0
    try:
        # LangChain sometimes includes usage in the response metadata.
        if hasattr(result, 'get') and result.get('__metadata'):
            tokens_used = result['__metadata'].get('total_tokens', 0)
    except (AttributeError, KeyError, TypeError):
        pass

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["output"],
        "steps": ui_logger.steps,
        "latency_ms": latency,
        "tokens": tokens_used if tokens_used else None,
    })

    save_current_chat()
    st.rerun()
