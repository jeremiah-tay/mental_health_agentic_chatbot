# AgentCore Runtime requires linux/arm64 (AWS Graviton).
# CodeBuild builds this image during `agentcore deploy` when build=Container.
FROM --platform=linux/arm64 python:3.13-slim-trixie

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY requirements-agentcore.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-agentcore.txt

# Application code
COPY agentcore_main.py .
COPY agents/ agents/
COPY tools/ tools/
COPY backend/ backend/
COPY config/ config/
COPY conversation_history/ conversation_history/
COPY riskclassifier_v2/crisis_response.py riskclassifier_v2/

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# BedrockAgentCoreApp serves POST /invocations and GET /ping on port 8080
EXPOSE 8080

# Numeric USER avoids AgentCore overlay mount issues with named users
USER 1000

CMD ["python", "agentcore_main.py"]
