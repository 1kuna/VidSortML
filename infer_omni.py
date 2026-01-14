#!/usr/bin/env python3
"""
Qwen3-VL Video Inference Client

Two-pass processing for Valorant clip sorting:

Default (full pipeline):
    python infer_omni.py video.mp4              # Categorize then count kills

Individual passes:
    python infer_omni.py video.mp4 --mode categorize   # Pass 1 only
    python infer_omni.py video.mp4 --mode killcount    # Pass 2 only

Custom prompt:
    python infer_omni.py video.mp4 "What happens?"     # Override with custom prompt
"""

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from openai import OpenAI

# =============================================================================
# PASS 1: CATEGORIZATION (cheap, full frame, 1fps, 480p)
# =============================================================================

CATEGORIZE_SYSTEM = """You are categorizing video clips to determine their content type.

Analyze this clip and determine:

1. IS THIS VALORANT?
   - Valorant HUD elements (health, abilities, minimap, kill feed)
   - Valorant agents (distinctive character designs)
   - Valorant maps (Bind, Haven, Ascent, etc.)
   - NOT Valorant: other games, desktop, main menu without gameplay

2. IF VALORANT, IS THERE A BUY PHASE?
   Buy phase indicators:
   - Buy menu overlay (weapon purchase screen)
   - Players in spawn with barriers
   - "BUY PHASE" text or countdown timer
   - Round start countdown

3. CONTENT TYPE:
   - "single-round": Valorant gameplay, NO buy phase visible
   - "multi-round": Valorant gameplay WITH buy phase visible (spans rounds)
   - "not-gameplay": Main menu, lobby, agent select, or non-Valorant content
   - "unknown": Can't determine

OUTPUT (JSON only):
{"category": "single-round|multi-round|not-gameplay|unknown", "reason": "<brief explanation>"}
"""

CATEGORIZE_PROMPT = "Categorize this video clip. Output JSON only."

# =============================================================================
# PASS 2: KILL COUNTING (expensive, cropped, 3fps, 900p)
# =============================================================================

KILLCOUNT_SYSTEM = """You are counting Valorant kills from a kill feed video.

This video shows ONLY the kill feed region (top-right corner).

BASICS:
- YOUR kills have a YELLOW highlight on the left side of the row
- Same victim in multiple frames = 1 kill (entries persist ~3 seconds)
- Count only yellow-highlighted entries

RESURRECTION ABILITIES (same player CAN die twice in ONE round):
- Phoenix ult: He "dies" during ult, respawns, can be killed again
- Sage: Can resurrect one dead teammate
- Clove: Can self-resurrect after getting a kill
Because of these, same victim twice does NOT mean different rounds.
Max kills per round: ~8 (5 base + Phoenix + Sage rez + Clove rez)

ROMAN NUMERALS (if visible next to yellow entries):
- III=3, IV=4, V=5 (ace), VI=6, VII=7, VIII=8
- Numerals indicate MINIMUM kills (you have AT LEAST that many)
- If you see IV but also see a 5th kill without a numeral, count 5
- Only use numeral as final count if clip starts mid-round (missed early kills)

MULTI-ROUND DETECTION:
Clips may span multiple rounds. RELIABLE boundary signals:
1. Roman numeral RESET (saw IV, then later I or II) → definite new round
2. More than 8 kills total → definitely multi-round
3. ACE (V) + gap + kills on NEW victims → likely new round

WHICH ROUND TO COUNT (if multiple detected):
- Count the round that occupies the MAJORITY of the clip duration
- If you see 1 kill, then a gap, then 3+ kills → count the 3+ (focused round)
- If you see 4 kills, then a gap, then 1 kill → count the 4 (focused round)
- Tiebreaker: prefer the later round

IF NO CLEAR BOUNDARY: Assume single round, count ALL yellow-highlighted kills you see.

IMPORTANT: Count what you SEE. If you see 5 unique victims with yellow highlight, report 5 kills.
Roman numerals only help when clip is cut off at start (e.g., first frame shows III = 3 kills happened before clip).

OUTPUT FORMAT:
KILLS: <number>
VICTIMS: <comma-separated list of victim names>
"""

# Prompt template - duration will be injected
KILLCOUNT_PROMPT_TEMPLATE = "This clip is {duration:.1f}s. Count kills for the focused round."

# Legacy prompt for backward compatibility
DEFAULT_PROMPT = """Count the POV player's kills from this kill feed. Output only the final result."""

# Legacy system prompt (kept for backward compat with custom prompts)
LEGACY_SYSTEM_PROMPT = """You are analyzing a Valorant kill feed to count the POV player's kills.

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


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Warning: ffprobe failed, using default duration", file=sys.stderr)
        return 30.0  # Default fallback
    return float(result.stdout.strip())


def preprocess_video(video_path: Path, fps: float = 3.0, max_height: int = 900,
                     crop_killfeed: bool = False, debug: bool = False) -> bytes:
    """Preprocess video: optionally crop to kill feed, scale, and resample fps.

    Args:
        video_path: Path to input video
        fps: Target frames per second
        max_height: Scale to this height (maintains aspect ratio)
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

    # Scale to max_height, maintaining aspect ratio, ensuring even dimensions
    # -2 means "calculate width to maintain aspect ratio, round to even"
    filters.append(f'scale=-2:{max_height}:flags=lanczos')

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
    print(f"Preprocessing: {max_height}p @ {fps} fps{crop_str}...", file=sys.stderr)
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
                 max_height: int = 900, crop_killfeed: bool = False,
                 debug: bool = False, save_thinking: Path = None,
                 system_prompt: str = None) -> str:
    """Send video to Qwen3-VL server for analysis."""
    client = OpenAI(
        api_key="not-used",
        base_url=f"http://{server}/v1"
    )

    print(f"Loading video: {video_path}", file=sys.stderr)
    video_data = preprocess_video(video_path, fps=fps, max_height=max_height,
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
        default=None,
        help="Custom prompt (overrides --mode)",
    )
    parser.add_argument(
        "--mode",
        choices=["categorize", "killcount", "full"],
        default="full",
        help="Processing mode: categorize (Pass 1), killcount (Pass 2), or full (both passes, default)",
    )
    parser.add_argument(
        "--server",
        default="localhost:8901",
        help="Server address (default: localhost:8901)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Frames per second (default: 1.0 for categorize, 3.0 for killcount)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Scale height in pixels (default: 480 for categorize, 900 for killcount)",
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Crop to kill feed region (auto-enabled for killcount mode)",
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

    # Get video duration for kill counting prompt
    duration = get_video_duration(args.video)

    # Determine settings based on mode
    if args.prompt:
        # Custom prompt mode (legacy behavior)
        system_prompt = LEGACY_SYSTEM_PROMPT if args.mode == "killcount" else None
        prompt = args.prompt
        fps = args.fps or 3.0
        height = args.height or 900
        crop = args.crop
        result = analyze_clip(args.video, prompt, args.server, fps,
                              height, crop, args.debug, args.save_thinking,
                              system_prompt)
        print(result)

    elif args.mode == "categorize":
        # Pass 1 only: Categorization (cheap, full frame)
        system_prompt = CATEGORIZE_SYSTEM
        prompt = CATEGORIZE_PROMPT
        fps = args.fps or 1.0
        height = args.height or 480
        crop = False
        print(f"Mode: CATEGORIZE (Pass 1) - {height}p @ {fps}fps, full frame", file=sys.stderr)
        result = analyze_clip(args.video, prompt, args.server, fps,
                              height, crop, args.debug, args.save_thinking,
                              system_prompt)
        print(result)

    elif args.mode == "killcount":
        # Pass 2 only: Kill Counting (expensive, cropped)
        system_prompt = KILLCOUNT_SYSTEM
        prompt = KILLCOUNT_PROMPT_TEMPLATE.format(duration=duration)
        fps = args.fps or 3.0
        height = args.height or 900
        crop = True
        print(f"Mode: KILLCOUNT (Pass 2) - {height}p @ {fps}fps, cropped, {duration:.1f}s clip", file=sys.stderr)
        result = analyze_clip(args.video, prompt, args.server, fps,
                              height, crop, args.debug, args.save_thinking,
                              system_prompt)
        print(result)

    else:
        # Full mode: Run both passes
        print(f"Mode: FULL (Pass 1 + Pass 2)", file=sys.stderr)

        # === PASS 1: Categorization ===
        print(f"\n{'='*50}", file=sys.stderr)
        print(f"PASS 1: CATEGORIZATION (480p @ 1fps, full frame)", file=sys.stderr)
        print(f"{'='*50}", file=sys.stderr)

        cat_result = analyze_clip(
            args.video,
            CATEGORIZE_PROMPT,
            args.server,
            fps=1.0,
            max_height=480,
            crop_killfeed=False,
            debug=args.debug,
            save_thinking=None,  # Don't save thinking for categorization
            system_prompt=CATEGORIZE_SYSTEM
        )

        print(f"\nCategorization result:", file=sys.stderr)
        print(cat_result, file=sys.stderr)

        # Parse category from JSON response
        category = "unknown"
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^}]+\}', cat_result)
            if json_match:
                cat_json = json.loads(json_match.group())
                category = cat_json.get("category", "unknown")
        except (json.JSONDecodeError, AttributeError):
            # If JSON parsing fails, look for category keywords
            cat_lower = cat_result.lower()
            if "single-round" in cat_lower:
                category = "single-round"
            elif "multi-round" in cat_lower:
                category = "multi-round"
            elif "not-gameplay" in cat_lower:
                category = "not-gameplay"

        print(f"\nDetected category: {category}", file=sys.stderr)

        # === PASS 2: Kill Counting (only for gameplay) ===
        if category in ["single-round", "multi-round"]:
            print(f"\n{'='*50}", file=sys.stderr)
            print(f"PASS 2: KILL COUNTING (900p @ 3fps, cropped)", file=sys.stderr)
            print(f"{'='*50}", file=sys.stderr)

            kill_result = analyze_clip(
                args.video,
                KILLCOUNT_PROMPT_TEMPLATE.format(duration=duration),
                args.server,
                fps=3.0,
                max_height=900,
                crop_killfeed=True,
                debug=args.debug,
                save_thinking=args.save_thinking,
                system_prompt=KILLCOUNT_SYSTEM
            )

            # Output combined result
            print(f"\n{'='*50}")
            print(f"RESULT")
            print(f"{'='*50}")
            print(f"CATEGORY: {category}")
            print(kill_result)

        else:
            # Not gameplay - skip kill counting
            print(f"\nSkipping kill counting (not Valorant gameplay)", file=sys.stderr)
            print(f"\n{'='*50}")
            print(f"RESULT")
            print(f"{'='*50}")
            print(f"CATEGORY: {category}")
            print(f"KILLS: N/A")


if __name__ == "__main__":
    main()
