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

DEFAULT_PROMPT = """Count the player's kills in this first-person Valorant clip.

=== KILL FEED BASICS ===
Location: Top-right corner of screen
Format: [KILLER NAME] [weapon/ability icon] [VICTIM NAME]
- Name on LEFT of icon = the killer (who got the kill)
- Name on RIGHT of icon = the victim (who died)

=== CRITICAL: FRAME PERSISTENCE vs STACKED ENTRIES ===

FRAME PERSISTENCE (same kill shown multiple times):
- A kill feed entry stays visible for several seconds before fading
- If you see "PlayerA killed VictimX" in frame 5, 6, 7, 8... that's still ONE kill
- The entry is just persisting on screen across multiple frames
- DO NOT count the same entry multiple times

STACKED ENTRIES (multiple kills at once):
- When kills happen rapidly, multiple entries stack VERTICALLY in the same frame
- If ONE frame shows 3 rows stacked like:
    PlayerA killed Victim1
    PlayerA killed Victim2
    PlayerA killed Victim3
- That's THREE separate kills - count each row as one kill
- Each row has a DIFFERENT victim name

=== HOW TO COUNT CORRECTLY ===

Step 1 - IDENTIFY YOUR NAME:
- You are the first-person view (the one holding the gun)
- When YOU shoot and kill someone, watch the kill feed
- A NEW entry appears with YOUR name on the LEFT
- That name (e.g., "Me" or your username) is your player name

Step 2 - COUNT UNIQUE VICTIMS:
- Find all kill feed entries where YOUR name is on the left
- Each DIFFERENT victim name = one kill
- In Valorant, you cannot kill the same person twice per round
- So count unique victim names killed by you

Step 3 - CHECK FOR STACKED RAPID KILLS:
- Look for frames where multiple entries are stacked
- If you got a triple kill, you'll see 3 entries stacked with YOUR name on the left
- Count each stacked entry as a separate kill

=== OUTPUT FORMAT ===
PLAYER NAME: [your name from kill feed]
KILLS:
1. [first victim name]
2. [second victim name]
3. [third victim name]
...
TOTAL: [number]"""


def strip_thinking(text: str) -> str:
    """Remove thinking blocks from model output."""
    original = text

    # Method 1: Handle proper <think>...</think> tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # Method 2: Handle missing opening tag (model outputs thinking then </think>)
    if '</think>' in text:
        text = text.split('</think>')[-1]

    # Method 3: Handle other thinking markers
    # Sometimes models use different formats
    text = re.sub(r'<\|think\|>.*?<\|/think\|>', '', text, flags=re.DOTALL)

    # Method 4: If output starts with common reasoning phrases and has a clear answer at end,
    # try to extract just the answer (look for the structured output format)
    stripped = text.strip()

    # If we still have a very long response that looks like reasoning,
    # try to find the structured output at the end
    if len(stripped) > 500 and 'PLAYER NAME:' in stripped:
        # Find the last occurrence of the output format
        idx = stripped.rfind('PLAYER NAME:')
        if idx != -1:
            stripped = stripped[idx:]

    # If output is just a number (simple case), return it
    if stripped.isdigit():
        return stripped

    # Clean up any remaining whitespace
    return stripped.strip()


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
