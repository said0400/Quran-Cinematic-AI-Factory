"""Local semantic vector scenes for Quran educational videos.

Gemini supplies only a JSON plan. Pillow draws every shape and FFmpeg encodes the frames;
no image-generation API, stock footage, or external visual asset is used.
"""
from __future__ import annotations

import hashlib
import math
import random
import subprocess
from pathlib import Path
from typing import Iterator

from PIL import Image, ImageDraw, ImageFont

PALETTE = {
    "background": (5, 13, 25), "grid": (26, 56, 82), "grid_major": (38, 87, 112),
    "cyan": (69, 221, 222), "blue": (76, 143, 255), "gold": (236, 191, 72),
    "white": (226, 242, 247), "muted": (112, 157, 178),
}


def _seed(theme: str) -> int:
    return int.from_bytes(hashlib.sha256(theme.encode("utf-8")).digest()[:8], "big")


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in ("/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
                 "/usr/share/fonts/truetype/noto/NotoNaskhArabicUI-Regular.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _glow_circle(draw: ImageDraw.ImageDraw, cx: float, cy: float, radius: float,
                 color: tuple[int, int, int], rings: int = 4) -> None:
    for ring in range(rings, 0, -1):
        factor = (rings - ring + 1) / (rings + 1)
        shade = tuple(int(channel * (0.18 + 0.62 * factor)) for channel in color)
        draw.ellipse((cx - radius - ring * 7, cy - radius - ring * 7,
                      cx + radius + ring * 7, cy + radius + ring * 7),
                     outline=shade, width=max(1, ring))
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=color, width=3)


def _scene_at(plan: dict[str, object] | None, t: float, duration: float) -> tuple[dict[str, object], float]:
    scenes = [x for x in (plan or {}).get("scenes", []) if isinstance(x, dict)]
    if not scenes:
        return {"type": "generic_explainer", "elements": []}, t / max(duration, 0.001)
    total = sum(float(x.get("duration_seconds", 1)) for x in scenes)
    cursor = 0.0
    for index, scene in enumerate(scenes):
        length = duration * float(scene.get("duration_seconds", 1)) / max(total, 0.001)
        if t <= cursor + length or index == len(scenes) - 1:
            return scene, max(0.0, min(1.0, (t - cursor) / max(length, 0.001)))
        cursor += length
    return scenes[-1], 1.0


def _draw_scene(draw: ImageDraw.ImageDraw, width: int, height: int,
                scene: dict[str, object], progress: float) -> None:
    """Render a safe, semantic vector scene selected by Gemini's validated plan."""
    scene_type = str(scene.get("type", "generic_explainer"))
    cx, cy = width * 0.50, height * 0.42
    if scene_type == "solar_lunar_year_comparison":
        draw.ellipse((width * .12, height * .24, width * .42, height * .54), outline=PALETTE["gold"], width=4)
        draw.ellipse((width * .58, height * .24, width * .88, height * .54), outline=PALETTE["cyan"], width=4)
        draw.ellipse((width * .25, height * .37, width * .29, height * .41), fill=PALETTE["gold"])
        mx = width * (.64 + .16 * math.cos(progress * math.tau))
        my = height * (.39 + .10 * math.sin(progress * math.tau))
        draw.ellipse((mx - 16, my - 16, mx + 16, my + 16), fill=PALETTE["cyan"])
        draw.line((width * .12, height * .62, width * (.12 + .76 * progress), height * .62), fill=PALETTE["gold"], width=5)
    elif scene_type in {"timeline_counter", "generic_explainer"}:
        y = height * .55
        draw.line((width * .10, y, width * .90, y), fill=PALETTE["cyan"], width=6)
        for index in range(7):
            x = width * (.10 + .80 * index / 6)
            draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=PALETTE["gold"] if index / 6 <= progress else PALETTE["muted"])
        draw.text((width * .12, y + 28), f"{int(300 + 9 * progress)}", font=_font(max(32, width // 18)), fill=PALETTE["gold"])
    elif scene_type == "obstacle_path":
        points = [(width * .10, height * .68), (width * .28, height * .56),
                  (width * .45, height * .66), (width * .65, height * .50), (width * .90, height * .58)]
        count = max(2, int(2 + len(points) * progress))
        draw.line(points[:count], fill=PALETTE["cyan"], width=8, joint="curve")
        for x, y in points[1:-1]:
            draw.rectangle((x - 22, y - 22, x + 22, y + 22), outline=PALETTE["gold"], width=4)
        px, py = points[min(len(points) - 1, int(progress * (len(points) - 1)))]
        draw.ellipse((px - 18, py - 18, px + 18, py + 18), fill=PALETTE["gold"])
    elif scene_type == "growth_and_provision":
        ground = height * .68
        stem = height * (.68 - .25 * progress)
        draw.line((width * .15, ground, width * .85, ground), fill=PALETTE["muted"], width=4)
        draw.line((cx, ground, cx, stem), fill=PALETTE["cyan"], width=8)
        draw.ellipse((cx - 70 * progress, stem - 42, cx - 4, stem + 20), fill=PALETTE["cyan"])
        draw.ellipse((cx + 4, stem - 42, cx + 70 * progress, stem + 20), fill=PALETTE["gold"])
    elif scene_type == "message_verification":
        for x in (width * .20, width * .62):
            draw.rounded_rectangle((x, height * .32, x + width * .18, height * .62), radius=20, outline=PALETTE["cyan"], width=5)
        draw.line((width * .38, height * .47, width * (.47 + .10 * progress), height * .47), fill=PALETTE["gold"], width=6)
        draw.polygon([(width * .56, height * .47), (width * .51, height * .43), (width * .51, height * .51)], fill=PALETTE["gold"])
        draw.ellipse((width * .69, height * .42, width * .75, height * .48), outline=PALETTE["gold"], width=5)
    elif scene_type == "storm_to_calm":
        for index in range(12):
            x = width * (.12 + index * .07)
            y = height * (.32 + ((index * 23) % 80) / 1000)
            draw.line((x, y, x - 12, y + 34), fill=PALETTE["blue"], width=3)
        radius = 28 + 55 * progress
        draw.ellipse((width * .68 - radius, height * .35 - radius,
                      width * .68 + radius, height * .35 + radius), outline=PALETTE["gold"], width=5)
    elif scene_type in {"home_relationship", "room_diagram"}:
        roof = [(width * .25, height * .42), (width * .50, height * .25), (width * .75, height * .42)]
        draw.polygon(roof, outline=PALETTE["gold"])
        draw.rectangle((width * .30, height * .42, width * .70, height * .70), outline=PALETTE["cyan"], width=5)
        draw.rectangle((width * .46, height * .56, width * .54, height * .70), fill=PALETTE["gold"])


def _draw_frame(width: int, height: int, t: float, duration: float, theme: str,
                particles: list[tuple[float, float, float, tuple[int, int, int]]],
                plan: dict[str, object] | None) -> Image.Image:
    image = Image.new("RGB", (width, height), PALETTE["background"])
    draw = ImageDraw.Draw(image)
    for y in range(0, height, 8):
        factor = y / max(1, height - 1)
        draw.rectangle((0, y, width, min(height, y + 8)),
                       fill=(int(5 + 7 * factor), int(13 + 13 * factor), int(25 + 22 * factor)))
    grid = max(48, width // 14)
    major = grid * 4
    ox, oy = int(t * 18 % grid), int(t * 11 % grid)
    for x in range(-grid + ox, width + grid, grid):
        draw.line((x, 0, x, height), fill=PALETTE["grid"], width=1)
    for y in range(-grid + oy, height + grid, grid):
        draw.line((0, y, width, y), fill=PALETTE["grid"], width=1)
    for x in range(-major + ox, width + major, major):
        draw.line((x, 0, x, height), fill=PALETTE["grid_major"], width=2)
    for y in range(-major + oy, height + major, major):
        draw.line((0, y, width, y), fill=PALETTE["grid_major"], width=2)
    cx, cy = width * .50, height * .43
    orbit_a, orbit_b = width * .30, height * .16
    for scale in (.52, .73, 1.0, 1.28):
        draw.ellipse((cx - orbit_a * scale, cy - orbit_b * scale,
                      cx + orbit_a * scale, cy + orbit_b * scale), outline=PALETTE["muted"], width=2)
    _glow_circle(draw, cx, cy, 18 + 7 * math.sin(t * 2.1), PALETTE["gold"], rings=5)
    _glow_circle(draw, cx + orbit_a * math.cos(t * .95), cy + orbit_b * math.sin(t * .95), 12, PALETTE["cyan"], rings=3)
    _glow_circle(draw, cx + orbit_a * .72 * math.cos(-t * .62 + 1.8), cy + orbit_b * .72 * math.sin(-t * .62 + 1.8), 8, PALETTE["blue"], rings=3)
    for x0, y0, size, color in particles:
        x = (x0 * width + math.sin(t * .4 + y0 * 9) * 18) % width
        y = (y0 * height - t * (12 + size * 2)) % height
        radius = max(1, size * (.7 + .3 * math.sin(t * 1.3 + x0 * 8)))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    scene, progress = _scene_at(plan, t, duration)
    _draw_scene(draw, width, height, scene, progress)
    equations = ("300 × 365", "309 × 354", "Δt = 11", "∑  •  ∞", "دقّة  •  حساب") if "18:25" in theme or "الكهف" in theme else ("x(t)", "Δt", "n²", "∑  •  ∞", "πr²")
    small = _font(max(24, width // 33))
    big = _font(max(52, width // 15))
    for index, text in enumerate(equations):
        phase = (t * (.12 + index * .015) + index * 1.7) % 1.0
        x = int(width * (.08 + (index % 3) * .36) + math.sin(t * .3 + index) * 22)
        y = int(height * (.10 + ((index * .19 + phase * .12) % .72)))
        draw.text((x, y), text, font=small if index else big,
                  fill=PALETTE["gold"] if index == 0 else PALETTE["white"],
                  stroke_width=1, stroke_fill=PALETTE["background"])
    draw.text((width * .06, height * .88), "رسم دلالي إجرائي", font=small,
              fill=PALETTE["muted"], stroke_width=1, stroke_fill=PALETTE["background"])
    progress_total = max(0.0, min(1.0, t / max(duration, .001)))
    draw.rectangle((width * .06, height * .94, width * .94, height * .944), fill=PALETTE["grid"])
    draw.rectangle((width * .06, height * .94, width * (.06 + .88 * progress_total), height * .944), fill=PALETTE["gold"])
    return image


def frames(width: int, height: int, fps: int, duration: float, theme: str,
           plan: dict[str, object] | None = None) -> Iterator[bytes]:
    rng = random.Random(_seed(theme))
    particles = [(rng.random(), rng.random(), rng.uniform(1, 4),
                  rng.choice((PALETTE["cyan"], PALETTE["blue"], PALETTE["gold"]))) for _ in range(80)]
    for number in range(int(math.ceil(duration * fps))):
        yield _draw_frame(width, height, number / fps, duration, theme, particles, plan).tobytes()


def render_background(destination: Path, width: int, height: int, fps: int, duration: float,
                      theme: str, render_scale: float = .5,
                      plan: dict[str, object] | None = None) -> dict[str, object]:
    """Render a deterministic semantic vector background with Pillow and FFmpeg."""
    if (width <= 0 or height <= 0 or width % 2 or height % 2 or fps <= 0 or duration <= 0
            or not .25 <= render_scale <= 1.0):
        raise ValueError("أبعاد أو معدل إطارات أو مدة الخلفية غير صالحة.")
    rw, rh = max(2, int(width * render_scale) // 2 * 2), max(2, int(height * render_scale) // 2 * 2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{rw}x{rh}", "-r", str(fps), "-i", "-", "-an",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-vf",
               f"scale={width}:{height}:flags=lanczos", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               str(destination)]
    process = None
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        assert process.stdin is not None
        for frame in frames(rw, rh, fps, duration, theme, plan):
            process.stdin.write(frame)
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
    except (BrokenPipeError, OSError) as exc:
        if process is not None:
            process.kill()
        raise RuntimeError(f"تعذر بث الإطارات إلى FFmpeg: {exc}") from exc
    if return_code != 0:
        raise RuntimeError(f"فشل إنشاء الخلفية الإجرائية: {stderr[-500:]}")
    if not destination.is_file() or destination.stat().st_size < 1024:
        raise RuntimeError("لم ينتج مولّد الخلفية ملفًا صالحًا.")
    return {"kind": "procedural_semantic_scenes", "width": width, "height": height,
            "render_width": rw, "render_height": rh, "render_scale": render_scale,
            "fps": fps, "duration_seconds": round(duration, 3),
            "theme_hash": hashlib.sha256(theme.encode("utf-8")).hexdigest()[:16],
            "visual_type": str((plan or {}).get("visual_type", "generic_explainer")),
            "external_assets": False}
