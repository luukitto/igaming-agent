FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py *.md *.json *.html ./
RUN python seed.py
# Without OPENROUTER_API_KEY, Ollama runs on the host (models are big; no need to bake them into the image)
ENV OLLAMA_URL=http://host.docker.internal:11434
EXPOSE 8000
# Cloud hosts like Railway pick the port and pass it in $PORT
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
