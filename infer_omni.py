#!/usr/bin/env python3
"""
Qwen3-Omni Video+Audio Inference Client

Usage:
    python infer_omni.py video.mp4 "What happens in this video?"
    python infer_omni.py video.mp4  # Uses default gaming clip prompt
    python infer_omni.py video.mp4 --server 192.168.1.100:8901
"""

import argparse
import base64
import sys
from pathlib import Path

from openai import OpenAI

DEFAULT_PROMPT = """Analyze this gaming clip. Identify:
1. Kill events (check kill log - username on LEFT of arrow means they got the kill)
2. Funny moments from audio/reactions
3. Skill level of plays (whiffs vs clutches)

Provide a brief summary and categorize as one of: kills_X (X=count), funny, skilled, or other."""


def analyze_clip(video_path: Path, prompt: str, server: str) -> str:
    """Send video to Qwen3-Omni server for analysis."""
    client = OpenAI(
        api_key="not-used",
        base_url=f"http://{server}/v1"
    )

    print(f"Loading video: {video_path}", file=sys.stderr)
    with open(video_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode()

    print(f"Sending to server ({server})...", file=sys.stderr)
    response = client.chat.completions.create(
        model="cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}}
            ]
        }],
        max_tokens=1024
    )

    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(
        description="Analyze video clips with Qwen3-Omni (audio+video)",
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
