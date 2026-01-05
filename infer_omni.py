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

DEFAULT_PROMPT = """Count the kills made by the player in this first-person Valorant clip.

You are watching from the player's perspective (first-person view). Count kills that THIS player makes.

How to identify a player kill:
1. The player shoots/attacks an enemy and they die
2. A red "X" hit marker or elimination indicator appears on screen
3. "+X" score popup or "ELIMINATED" text appears
4. In the kill feed (top-right): the PLAYER'S name appears on the LEFT side of the weapon icon
   - Format: [Killer] [weapon icon] [Victim]
   - Left of weapon = killer, Right of weapon = victim

DO NOT count:
- Teammate kills (kills made by other players on your team)
- Deaths (when the player dies)
- The round score at top (e.g., 9-8) - this is rounds won, not kills

After careful analysis, output ONLY a single number (0, 1, 2, 3, etc). Nothing else."""


def strip_thinking(text: str) -> str:
    """Remove thinking blocks from model output."""
    # Handle <think>...</think>
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Handle when template auto-adds <think> (output only has </think>)
    if '</think>' in text:
        text = text.split('</think>')[-1]
    return text.strip()


def preprocess_video(video_path: Path, max_height: int = 720, fps: float = 6.0, debug: bool = False) -> bytes:
    """Resize video to max_height and resample to target fps using ffmpeg."""
    orig_size = video_path.stat().st_size
    print(f"Original video: {orig_size / 1024 / 1024:.2f} MB", file=sys.stderr)

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

    print(f"Processed video: {len(data) / 1024 / 1024:.2f} MB", file=sys.stderr)

    if debug:
        debug_path = video_path.with_suffix('.debug.mp4')
        with open(debug_path, 'wb') as f:
            f.write(data)
        print(f"Debug video saved: {debug_path}", file=sys.stderr)

    return data


def analyze_clip(video_path: Path, prompt: str, server: str, fps: float,
                 debug: bool = False, save_thinking: Path = None) -> str:
    """Send video to Qwen3-VL server for analysis."""
    client = OpenAI(
        api_key="not-used",
        base_url=f"http://{server}/v1"
    )

    print(f"Loading video: {video_path}", file=sys.stderr)
    video_data = preprocess_video(video_path, max_height=720, fps=fps, debug=debug)
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
        max_tokens=16384,  # Thinking needs lots of room
        temperature=1.0,
        top_p=0.95,
        extra_body={"top_k": 20}
    )

    raw_response = response.choices[0].message.content

    if save_thinking:
        save_thinking.write_text(raw_response, encoding='utf-8')
        print(f"Thinking saved to: {save_thinking}", file=sys.stderr)

    return strip_thinking(raw_response)


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
        default=6.0,
        help="Frames per second to sample (default: 6.0)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save preprocessed video for inspection",
    )
    parser.add_argument(
        "--save-thinking",
        type=Path,
        metavar="FILE",
        help="Save full response with thinking to file",
    )

    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    result = analyze_clip(args.video, args.prompt, args.server, args.fps,
                          args.debug, args.save_thinking)
    print(result)


if __name__ == "__main__":
    main()
