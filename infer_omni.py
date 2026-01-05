#!/usr/bin/env python3
"""
Qwen3-VL Video Inference Client

Usage:
    python infer_omni.py video.mp4 "What happens in this video?"
    python infer_omni.py video.mp4  # Uses default gaming clip prompt
    python infer_omni.py video.mp4 --server 192.168.1.100:8901
    python infer_omni.py video.mp4 --fps 4  # Higher frame rate
"""

import argparse
import base64
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from openai import OpenAI

DEFAULT_PROMPT = """Count the kills made by the player in this first-person gaming clip.

Think carefully through the ENTIRE video frame by frame:
1. Watch for each enemy death caused by the player
2. Look for hit markers, elimination notifications, and kill confirmations on screen
3. Check the kill feed (usually top-right) to confirm kills attributed to the player
4. Count each distinct kill - don't miss rapid successive kills

After thorough analysis, output ONLY a single number (0, 1, 2, 3, etc). Nothing else."""


def strip_thinking(text: str) -> str:
    """Remove thinking blocks from model output."""
    # Handle both <think>...</think> and just </think> (when template adds opening)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'^.*?</think>', '', text, flags=re.DOTALL)  # Remove everything before </think>
    return text.strip()


def preprocess_video(video_path: Path, max_height: int = 1080, fps: float = 3.0) -> bytes:
    """Resize video to max_height and resample to target fps using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        tmp_path = tmp.name

    cmd = [
        'ffmpeg', '-y', '-i', str(video_path),
        '-vf', f'scale=-2:{max_height}:flags=lanczos,fps={fps}',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-an',  # Remove audio
        tmp_path
    ]

    print(f"Preprocessing: {max_height}p @ {fps} fps...", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"Warning: ffmpeg failed, using original video", file=sys.stderr)
        with open(video_path, 'rb') as f:
            return f.read()

    with open(tmp_path, 'rb') as f:
        data = f.read()
    os.unlink(tmp_path)
    return data


def analyze_clip(video_path: Path, prompt: str, server: str, fps: float) -> str:
    """Send video to Qwen3-VL server for analysis."""
    client = OpenAI(
        api_key="not-used",
        base_url=f"http://{server}/v1"
    )

    print(f"Loading video: {video_path}", file=sys.stderr)
    video_data = preprocess_video(video_path, max_height=1080, fps=fps)
    video_b64 = base64.b64encode(video_data).decode()

    print(f"Sending to server ({server})...", file=sys.stderr)
    response = client.chat.completions.create(
        model="Qwen/Qwen3-VL-8B-Thinking-FP8",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}}
            ]
        }],
        max_tokens=4096  # Room for thorough thinking + response
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
    parser.add_argument(
        "--fps",
        type=float,
        default=3.0,
        help="Frames per second to sample (default: 3.0)",
    )

    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    result = analyze_clip(args.video, args.prompt, args.server, args.fps)
    print(result)


if __name__ == "__main__":
    main()
