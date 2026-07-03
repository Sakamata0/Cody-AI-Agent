# 🤖 Cody — Autonomous AI Agent

**SMARTOVATE LTD** | Built by Skander Boughnimi  
Powered by AWS Bedrock + LangChain

---

## Overview

Cody is an autonomous AI agent that can reason, plan, and execute multi-step tasks using a ReAct (Reasoning and Acting) approach. It combines AWS Bedrock's Claude Haiku LLM with custom tools to answer questions, search the web, query databases, convert currencies, check weather, execute code, and predict natural disasters.

## Features

- 🧠 **ReAct Reasoning** — Multi-step planning with tool chaining
- 🔍 **Web Search** — Real-time information via Tavily API
- 🗄️ **Database Queries** — Natural language to SQL (SQLite)
- 💱 **Currency Conversion** — 160+ currencies via ExchangeRate-API
- 🌤️ **Weather** — Current conditions via Open-Meteo
- 💻 **Code Execution** — Sandboxed Python execution
- 🌪️ **Disaster Prediction** — ML-based forecasting (tornados, earthquakes, hurricanes)
- 💬 **Conversational Memory** — Context retention across messages with auto-summarization
- 🖥️ **Streamlit UI** — Chat interface with conversation history
- 🌐 **REST API** — FastAPI server for integration with any frontend

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | AWS Bedrock (Claude Haiku 4.5) |
| Orchestration | LangChain / AgentExecutor |
| ML Models | Prophet (time-series forecasting) |
| API Server | FastAPI + Uvicorn |
| UI | Streamlit |
| Database | SQLite |
| Language | Python 3.11+ |
| Containerization | Docker + Docker Compose |

## Project Structure

```
Cody AI Agent/
├── app.py                  # Streamlit chat interface
├── server.py               # FastAPI REST API
├── src/
│   ├── agent.py            # Core ReAct agent with memory
│   ├── llm.py              # Bedrock LLM client configuration
│   ├── storage.py          # Conversation persistence (JSON)
│   ├── rate_limiter.py     # Rate limiting for Bedrock API
│   ├── token_tracker.py    # Token usage and cost tracking
│   ├── token_cache.py      # Response caching (LRU)
│   └── tools/
│       ├── web_search.py       # Tavily web search
│       ├── database.py         # SQL database queries
│       ├── code_executor.py    # Sandboxed Python execution
│       ├── disaster_predictor.py  # ML disaster prediction
│       └── api/
│           ├── weather.py      # Open-Meteo weather
│           └── exchange.py     # Currency exchange rates
├── data/
│   ├── seed_database.py    # Generate sample company database
│   └── disasters/
│       ├── train_tornado_model.py
│       ├── train_earthquake_model.py
│       ├── train_hurricane_model.py
│       └── models/             # Trained Prophet models
├── scripts/                # Test scripts
├── assets/                 # Logo and static files
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Setup

### Prerequisites

- Python 3.11+
- AWS account with Bedrock access (Claude Haiku enabled)
- AWS credentials configured (`~/.aws/credentials` or environment variables)
- Tavily API key (free tier: 1000 searches/month)

### Installation

```bash
# Clone the repository
git clone https://github.com/Sakamata0/Cody-AI-Agent.git
cd Cody-AI-Agent

# Run setup (creates venv, installs dependencies)
setup.bat

# Or manually:
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env with your API keys

# Seed the database
python data/seed_database.py

# Train disaster prediction models
python data/disasters/train_tornado_model.py
python data/disasters/train_earthquake_model.py
python data/disasters/train_hurricane_model.py
```

### Environment Variables

```env
AWS_REGION=eu-north-1
BEDROCK_MODEL_ID=eu.anthropic.claude-haiku-4-5-20251001-v1:0
TAVILY_API_KEY=your-tavily-api-key
```

## Usage

### Streamlit UI

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### REST API

```bash
uvicorn server:app --reload --port 8000
# API docs at http://localhost:8000/docs
```

#### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat` | Send a message to Cody |
| GET | `/health` | Health check |
| GET | `/tools` | List available tools |
| GET | `/sessions` | List active sessions |
| DELETE | `/sessions/{id}` | Clear a session |

#### Example Request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the tornado risk in Oklahoma?", "session_id": "demo"}'
```

### Docker

```bash
docker-compose up --build
# API: http://localhost:8000
# UI: http://localhost:8501
```

## Disaster Prediction Models

| Disaster | Region | Data Source | Records | Period |
|----------|--------|-------------|---------|--------|
| Tornados | USA (10 states) | NOAA SPC | 70,022 | 1950-2023 |
| Earthquakes | Japan | USGS | 17,022 | 1950-2024 |
| Hurricanes | Atlantic | NOAA HURDAT2 | 1,090 | 1851-2023 |

Models are trained using Facebook's Prophet library for time-series forecasting with seasonal decomposition.

## Bug Fixes Implemented

- ♻️ **Infinite Loop Prevention** — max_iterations + duplicate call detection + 60s timeout
- 🛡️ **Tool Input Validation** — Pydantic schemas prevent LLM parameter hallucination
- 📉 **Context Overflow** — Auto-summarization when history exceeds 40 messages
- ⏱️ **Throttling Protection** — Adaptive retry with exponential backoff

## License

Internal project — SMARTOVATE LTD © 2026
