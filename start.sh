#!/bin/bash

set -e

echo "Starting Ollama..."

ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!

echo "Waiting for Ollama..."

until curl -s http://127.0.0.1:11434/api/tags > /dev/null; do
    sleep 2
done

echo "Ollama ready."

if ! ollama list | grep -q "qwen3:4b-instruct"; then
    echo "Pulling qwen3:4b-instruct..."
    ollama pull qwen3:4b-instruct
fi

echo "Starting FastAPI..."

uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port 8000 &

API_PID=$!

echo "Waiting for FastAPI..."

until curl -s http://127.0.0.1:8000/health > /dev/null; do
    sleep 2
done

echo "FastAPI ready."

echo "Starting Streamlit..."

exec streamlit run app.py \
    --server.address=0.0.0.0 \
    --server.port=7860 \
    --server.headless=true