#!/usr/bin/env python3
"""
Qwen3-VL Video Inference Client

Usage:
    python infer_omni.py video.mp4 "What happens in this video?"
    python infer_omni.py video.mp4  # Uses default gaming clip prompt
    python infer_omni.py video.mp4 --server 192.168.1.100:8901
"""

import argparse
import base64
import re
import sys
from pathlib import Path

from openai import OpenAI

DEFAULT_PROMPT = """Count the kills made by the player (first-person perspective). Only count kills the player gets, not other players' kills shown in the kill feed.

Output ONLY a single number (0, 1, 2, 3, etc). Nothing else."""


def strip_thinking(text: str) -> str:
    """Remove thinking blocks from model output."""
    # Handle both <think>...</think> and just </think> (when template adds opening)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'^.*?</think>', '', text, flags=re.DOTALL)  # Remove everything before </think>
    return text.strip()


def analyze_clip(video_path: Path, prompt: str, server: str) -> str:
    """Send video to Qwen3-VL server for analysis."""
    client = OpenAI(
        api_key="not-used",
        base_url=f"http://{server}/v1"
    )

    print(f"Loading video: {video_path}", file=sys.stderr)
    with open(video_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode()

    print(f"Sending to server ({server})...", file=sys.stderr)
    response = client.chat.completions.create(
        model="Qwen/Qwen3-VL-4B-Thinking",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}}
            ]
        }],
        max_tokens=2048  # Room for thinking + response
    )

    return strip_thinking(response.choices[0].message.content)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze video clips with Qwen3-VL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video", type=Path, help="Path to video file")
    parser.add_argument(
        "prompt",
        nargs="?",
        default=DEFAULT_PROMPT,
        help="Question or prompt about the video",
    )
    parser.add_argument(
        "--server",
        default="localhost:8901",
        help="Server address (default: localhost:8901)",
    )

    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    result = analyze_clip(args.video, args.prompt, args.server)
    print(result)


if __name__ == "__main__":
    main()
