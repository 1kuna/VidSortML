#!/usr/bin/env python3
"""
Simple CLI for Qwen3-VL video inference.

Usage:
    python infer.py video.mp4 "What happens in this video?"
    python infer.py video.mp4  # Uses default prompt
    python infer.py video.mp4 "Describe this" --fps 1.0 --max-tokens 256
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


def load_model():
    """Load model and processor. Cached after first download."""
    model_id = "Qwen/Qwen3-VL-8B-Instruct"

    print("Loading model...", file=sys.stderr)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    print("Model loaded.", file=sys.stderr)

    return model, processor


def infer(video_path: Path, prompt: str, fps: float, max_tokens: int):
    """Run inference on a video file."""
    model, processor = load_model()

    video_path_str = str(video_path.resolve())
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path_str, "fps": fps},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    print(f"Processing video at {fps} fps...", file=sys.stderr)

    # Qwen3-VL correct pattern: single-step processing
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    print(f"Inputs ready. Device: {model.device}", file=sys.stderr)
    print("Generating response...", file=sys.stderr)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        repetition_penalty=1.1,  # Reduce repetitive output
    )

    # Decode (skip input tokens)
    output_ids = [
        out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    return response


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL video inference CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video", type=Path, help="Path to video file")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Describe what happens in this video.",
        help="Question or prompt about the video",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames per second to sample (default: 2.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max response tokens (default: 512)",
    )

    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    response = infer(args.video, args.prompt, args.fps, args.max_tokens)
    print(response)


if __name__ == "__main__":
    main()
