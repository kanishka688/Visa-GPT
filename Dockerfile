FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV KKGPT_LLM_MODEL=qwen3:4b-instruct
ENV KKGPT_OLLAMA_URL=http://127.0.0.1:11434/api/chat
ENV KKGPT_RETRIEVAL_TOP_K=5
ENV KKGPT_MAX_EVIDENCE_SOURCES=5
ENV KKGPT_API_BASE_URL=http://127.0.0.1:8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        libgomp1 \
        zstd \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data ./data
COPY app.py .
COPY start.sh .
COPY .env.example ./.env.example

RUN chmod +x /app/start.sh

EXPOSE 7860
EXPOSE 8000
EXPOSE 11434

CMD ["/app/start.sh"]