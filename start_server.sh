#!/bin/bash
# Qwen3-Omni Server Starter for RTX 4090
# Copy this to your desktop and run: ./start_server.sh

set -e

VENV_DIR="$HOME/vllm-omni"
MODEL="cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit"
PORT=8901

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    echo "Installing vLLM (this may take a few minutes)..."
    pip install -U vllm openai
else
    source "$VENV_DIR/bin/activate"
fi

echo ""
echo "Starting Qwen3-Omni server on port $PORT..."
echo "First run will download ~18GB model."
echo "Press Ctrl+C to stop."
echo ""

vllm serve "$MODEL" \
    --quantization awq \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --port "$PORT" \
    --host 0.0.0.0 \
    --allowed-local-media-path / \
    --gpu-memory-utilization 0.95
