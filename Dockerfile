FROM python:3.13-slim

WORKDIR /app

# Install system dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code.
COPY . .

# Seed the company database.
RUN python data/seed_database.py

# Expose ports: 8000 for API, 8501 for Streamlit.
EXPOSE 8000 8501

# Default: run the FastAPI server.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
