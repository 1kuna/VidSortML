#!/usr/bin/env python3
"""
Qwen3-VL Video Inference Client

Usage:
    python infer_omni.py video.mp4 --crop             # Crop to kill feed (recommended)
    python infer_omni.py video.mp4                    # Full frame analysis
    python infer_omni.py video.mp4 "What happens?"   # Custom prompt
    python infer_omni.py video.mp4 --crop --debug    # Save cropped video for inspection
    python infer_omni.py video.mp4 --scale 1.0       # Full resolution
    python infer_omni.py video.mp4 --server 192.168.1.100:8901
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

SYSTEM_PROMPT = """You are analyzing a Valorant kill feed to count the POV player's kills.

This video shows ONLY the kill feed region (top-right corner of gameplay).

KILL FEED FORMAT:
- Each row: [Agent portrait] KILLER [weapon icon] VICTIM [Agent portrait]
- LEFT name = killer, RIGHT name = victim
- Green background on killer side, red/orange on victim side
- Rows stack vertically (newest on top)
- Rows persist for a few seconds before fading

CRITICAL: POV PLAYER KILLS HAVE A YELLOW HIGHLIGHT
- When YOU get a kill, the row has a bright YELLOW outline/glow on the LEFT side
- This yellow highlight is the definitive marker for YOUR kills
- Count rows with this yellow highlight

YOUR METHOD:
1) Scan every frame. Look for rows with YELLOW highlight on the left.
2) These are YOUR kills. Read the victim name (right side) for each.
3) Count unique victim names (same name in multiple frames = 1 kill).

OUTPUT FORMAT (nothing else):
PLAYER NAME: <name from highlighted rows>
KILLS:
1. <victim>
2. <victim>
...
TOTAL: <number>"""

DEFAULT_PROMPT = """Count the POV player's kills from this kill feed. Output only the final result."""


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


def preprocess_video(video_path: Path, fps: float = 5.0, scale: float = 0.8,
                     crop_killfeed: bool = False, debug: bool = False) -> bytes:
    """Preprocess video: optionally crop to kill feed, scale, and resample fps.

    Args:
        video_path: Path to input video
        fps: Target frames per second
        scale: Scale factor (0.8 = 80% of original resolution)
        crop_killfeed: If True, crop to top-right region where kill feed appears
        debug: If True, save processed video for inspection
    """
    orig_size = video_path.stat().st_size
    print(f"Original video: {orig_size / 1024 / 1024:.2f} MB", file=sys.stderr)

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        tmp_path = tmp.name

    # Build filter chain
    filters = []

    if crop_killfeed:
        # Crop to top-right 40% width, top 45% height (where kill feed lives)
        # crop=out_w:out_h:x:y - x/y are expressions based on input dimensions
        filters.append('crop=iw*0.40:ih*0.45:iw*0.60:0')
        print(f"Cropping to kill feed region (top-right 40%x45%)", file=sys.stderr)

    # Scale to target size, ensuring even dimensions for libx264
    # -2 means "round to nearest even number"
    filters.append(f'scale=trunc(iw*{scale}/2)*2:trunc(ih*{scale}/2)*2:flags=lanczos')

    # Resample framerate
    filters.append(f'fps={fps}')

    filter_str = ','.join(filters)

    cmd = [
        'ffmpeg', '-y', '-i', str(video_path),
        '-vf', filter_str,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-an',  # Remove audio
        tmp_path
    ]

    crop_str = " + kill feed crop" if crop_killfeed else ""
    print(f"Preprocessing: {scale*100:.0f}% scale @ {fps} fps{crop_str}...", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"Warning: ffmpeg failed: {result.stderr.decode()}", file=sys.stderr)
        print(f"Using original video", file=sys.stderr)
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


def send_request(client, video_b64: str, prompt: str, system_prompt: str = None) -> str:
    """Send a single request to the model and return raw response."""
    messages = []

    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}}
        ]
    })

    response = client.chat.completions.create(
        model="Qwen/Qwen3-VL-8B-Thinking-FP8",
        messages=messages,
        max_tokens=16384,  # Thinking needs lots of room
        temperature=1.0,
        top_p=0.95,
        extra_body={"top_k": 20}
    )
    return response.choices[0].message.content


def analyze_clip(video_path: Path, prompt: str, server: str, fps: float,
                 scale: float = 0.8, crop_killfeed: bool = False,
                 debug: bool = False, save_thinking: Path = None,
                 system_prompt: str = None) -> str:
    """Send video to Qwen3-VL server for analysis."""
    client = OpenAI(
        api_key="not-used",
        base_url=f"http://{server}/v1"
    )

    print(f"Loading video: {video_path}", file=sys.stderr)
    video_data = preprocess_video(video_path, fps=fps, scale=scale,
                                   crop_killfeed=crop_killfeed, debug=debug)
    video_b64 = base64.b64encode(video_data).decode()

    print(f"Sending to server ({server})...", file=sys.stderr)
    raw_response = send_request(client, video_b64, prompt, system_prompt)

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
        default=3.0,
        help="Frames per second to sample (default: 3.0)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.6,
        help="Scale factor for resolution (default: 0.6 = 60%%)",
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Crop to kill feed region (top-right corner)",
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

    # Use system prompt only with default kill-counting prompt
    system_prompt = SYSTEM_PROMPT if args.prompt == DEFAULT_PROMPT else None

    result = analyze_clip(args.video, args.prompt, args.server, args.fps,
                          args.scale, args.crop, args.debug, args.save_thinking,
                          system_prompt)
    print(result)


if __name__ == "__main__":
    main()
