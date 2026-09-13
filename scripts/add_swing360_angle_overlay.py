"""Burn the running max |pivot angle| and turn count into a swing360 video.

Same spirit as add_swing_angle_overlay.py (green flash on new records), but
the metric is the unwrapped rigid-arm pivot angle: peak-to-peak span is
meaningless once the seat is looping (it pins at 360 after the first turn).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--x", type=int, default=30)
    parser.add_argument("--y", type=int, default=26)
    parser.add_argument("--font-size", type=int, default=72)
    parser.add_argument("--green-hold-s", type=float, default=0.24)
    args = parser.parse_args()

    samples = json.loads(args.metrics.read_text())["samples"]
    times = np.asarray([s["time_s"] for s in samples])
    running_max_deg = np.degrees(
        np.maximum.accumulate(np.abs([s["angle_rad"] for s in samples]))
    )

    reader = imageio.get_reader(args.input)
    fps = float(reader.get_meta_data()["fps"])
    font = _load_font(args.font_size)
    small = _load_font(max(20, args.font_size // 2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        args.output,
        fps=fps,
        codec="libx264",
        quality=9,
        pixelformat="yuv420p",
        ffmpeg_params=["-profile:v", "high", "-movflags", "+faststart", "-an"],
    )

    last_displayed = 0
    green_until_s = -math.inf
    for frame_count, frame in enumerate(reader, start=1):
        frame_time_s = (frame_count - 1) / fps
        idx = int(np.clip(np.searchsorted(times, frame_time_s, side="right") - 1, 0, len(times) - 1))
        displayed = round(float(running_max_deg[idx]))
        if displayed > last_displayed and frame_count > 1:
            green_until_s = frame_time_s + args.green_hold_s
        last_displayed = max(last_displayed, displayed)
        turns = int(math.radians(last_displayed) // (2.0 * math.pi))

        image = Image.fromarray(np.asarray(frame)).convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        color = (0, 174, 66, 255) if frame_time_s <= green_until_s else (255, 255, 255, 255)
        draw.text((args.x, args.y), f"max {last_displayed}\u00b0", font=font, fill=color)
        draw.text(
            (args.x + 6, args.y + args.font_size + 12),
            f"{turns} turn{'s' if turns != 1 else ''}",
            font=small,
            fill=color,
        )
        writer.append_data(np.asarray(Image.alpha_composite(image, overlay).convert("RGB")))
    writer.close()
    print(f"wrote {args.output} (final max {last_displayed} deg)")


if __name__ == "__main__":
    main()
