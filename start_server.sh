#!/bin/bash
# Qwen3-VL Server Starter for RTX 4090 (Linux/WSL2)
#
# On Windows: Run this inside WSL2 Ubuntu
#   1. Install WSL2: wsl --install -d Ubuntu-24.04
#   2. Open Ubuntu terminal and run: ./start_server.sh

VENV_DIR="$HOME/vllm-omni"
MODEL="Qwen/Qwen3-VL-8B-Thinking-FP8"
PORT=8901

# Create venv if needed
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    echo "Installing vLLM (this may take a few minutes)..."
    pip install -U pip
    pip install vllm openai
else
    source "$VENV_DIR/bin/activate"
fi

echo ""
echo "Starting Qwen3-VL server on port $PORT..."
echo "First run will download ~9GB model."
echo "Press Ctrl+C to stop."
echo ""

vllm serve "$MODEL" \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --port "$PORT" \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.90
