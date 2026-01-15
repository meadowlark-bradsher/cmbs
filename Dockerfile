# CMBS Agent Docker Image for ITBench
#
# This Dockerfile builds the CMBS agent for ITBench leaderboard submission.
#
# Build:
#   docker build -t cmbs-agent:latest .
#
# Run locally (for testing):
#   docker run --rm -it --network host \
#       -v /path/to/scenario_data.json:/tmp/agent/scenario_data.json \
#       cmbs-agent:latest
#
# Note: Requires Ollama server accessible at localhost:11434 (use --network host)
# or configure OLLAMA_HOST environment variable.

FROM icr.io/agent-bench/ciso-agent-harness-base:0.0.3 AS base

# Use bash
RUN ln -sf /bin/bash /bin/sh

# Install system dependencies
RUN apt update -y && apt install -y \
    curl \
    gnupg2 \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Install kubectl
RUN curl -LO "https://dl.k8s.io/release/v1.31.0/bin/linux/$(dpkg --print-architecture)/kubectl" && \
    chmod +x ./kubectl && \
    mv ./kubectl /usr/local/bin/kubectl

# Copy CMBS source
COPY cmbs /etc/cmbs/cmbs
COPY agent-harness.yaml /etc/cmbs/agent-harness.yaml

# Install Python dependencies
WORKDIR /etc/cmbs
RUN pip install --no-cache-dir \
    ollama \
    pydantic

# Create agent working directory
RUN mkdir -p /tmp/agent

# Copy entrypoint from agent-bench (provided by base image)
# The base image handles reading agent-harness.yaml and executing the run command

WORKDIR /etc/agent-benchmark

ENTRYPOINT ["/etc/entrypoint.sh"]
