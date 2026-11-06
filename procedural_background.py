"""Script-synchronised semantic vector scenes for Quran educational videos.

Gemini returns only a JSON plan: an opening scene (hook), 3-6 scenes tied to words of the spoken
message (`text_span`), and a closing scene (call to action). `build_timeline` converts every
`text_span` into a real start time using the word timings of the voice-over, so a scene changes
exactly when its sentence is spoken. Pillow draws a library of composable elements (optionally
replaced by operator-supplied Lottie / SVG files) and FFmpeg encodes the frames.
No image-generation API, stock footage or download is used.

Layout note: the commentary card covers the middle of the frame, so the sky lives at the top
(SKY_Y) and every ground element stands on GROUND_Y, in the lower band that stays visible.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import subprocess
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import urlparse

from PIL import Image, ImageDraw

GROUND_Y = 0.84
SKY_Y = 0.15
APPEAR_SECONDS = 0.9
XFADE_SECONDS = 0.4
MIN_SCENE_SECONDS = 1.6
ASSET_ROOT = Path(__file__).resolve().parent / "assets" / "visuals"

# name -> (sky top, horizon, ground, ink used for silhouettes)
MOODS: dict[str, tuple[tuple[int, int, int], ...]] = {
    "night": ((8, 14, 32), (28, 44, 86), (10, 14, 24), (236, 238, 245)),
    "dawn": ((38, 44, 92), (236, 156, 104), (24, 22, 34), (248, 240, 232)),
    "day": ((70, 140, 205), (190, 225, 245), (34, 64, 52), (28, 38, 58)),
    "storm": ((22, 28, 40), (66, 78, 96), (14, 16, 22), (226, 230, 238)),
    "calm": ((10, 28, 52), (38, 92, 120), (10, 22, 34), (230, 244, 246)),
    "warm": ((60, 32, 36), (224, 150, 86), (30, 18, 18), (252, 240, 226)),
}
POSITIONS = {"far_left": 0.14, "left": 0.30, "center": 0.50, "right": 0.70, "far_right": 0.86}
SIZES = {"small": 0.8, "medium": 1.0, "large": 1.3}
ANIMATIONS = {"none", "fade_in", "rise", "grow", "walk", "pulse", "float", "fall"}
GOLD, LEAF, WATER = (236, 191, 72), (96, 196, 120), (150, 190, 255)


def _c(rgb: tuple[int, ...], a: float = 1.0) -> tuple[int, int, int, int]:
    return (rgb[0], rgb[1], rgb[2], max(0, min(255, int(255 * a))))


def _ease(x: float) -> float:
    x = 0.0 if x < 0 else 1.0 if x > 1 else x
    return x * x * (3 - 2 * x)


def _mix(a: tuple[int, ...], b: tuple[int, ...], k: float) -> tuple[int, int, int]:
    return tuple(int(a[i] + (b[i] - a[i]) * k) for i in range(3))  # type: ignore[return-value]


def _ellipse(cx: float, cy: float, rx: float, ry: float, rot: float = 0.0, n: int = 24) -> list[tuple[float, float]]:
    cs, sn = math.cos(rot), math.sin(rot)
    pts = []
    for k in range(n):
        ang = k * math.tau / n
        px, py = rx * math.cos(ang), ry * math.sin(ang)
        pts.append((cx + px * cs - py * sn, cy + px * sn + py * cs))
    return pts


class _Ctx:
    """Everything a drawer needs for one frame of one scene."""
    def __init__(self, img: Image.Image, mood: str, t: float, assets: "VisualAssets | None"):
        self.img, self.d = img, ImageDraw.Draw(img, "RGBA")
        self.W, self.H = img.size
        self.top, self.horizon, self.ground, self.ink = MOODS[mood]
        self.mood, self.t, self.assets = mood, t, assets


# ------------------------------------------------------------------ element drawers
# Signature: drawer(ctx, x, y, u, st). Ground elements: y = feet line. Sky elements: y = centre.
# u = 1% of the frame height times the size factor; st = animation state (alpha, grow, step, p, draw).
def d_person(c: _Ctx, x, y, u, st):
    u *= st["grow"]; ink = _c(c.ink, st["alpha"]); w = max(2, int(1.5 * u)); sw = math.sin(st["step"]) * 2.2 * u
    c.d.line((x, y - 8 * u, x - sw, y), fill=ink, width=w)
    c.d.line((x, y - 8 * u, x + sw, y), fill=ink, width=w)
    c.d.line((x, y - 8 * u, x, y - 13 * u), fill=ink, width=max(2, int(1.9 * u)))
    c.d.line((x, y - 12.5 * u, x - 2 * u + sw * .4, y - 9 * u), fill=ink, width=w)
    c.d.line((x, y - 12.5 * u, x + 2 * u - sw * .4, y - 9 * u), fill=ink, width=w)
    c.d.ellipse((x - 1.8 * u, y - 17 * u, x + 1.8 * u, y - 13.4 * u), fill=ink)


def d_sitting_person(c: _Ctx, x, y, u, st):
    u *= st["grow"]; ink = _c(c.ink, st["alpha"]); w = max(2, int(1.6 * u))
    c.d.line((x - u, y, x + 4 * u, y - 3 * u, x + 4 * u, y), fill=ink, width=w)
    c.d.line((x - u, y, x + .5 * u, y - 6.5 * u), fill=ink, width=int(2.8 * u))
    c.d.line((x + .5 * u, y - 5 * u, x + 3.8 * u, y - 4.2 * u), fill=ink, width=w)
    c.d.ellipse((x + .4 * u, y - 10.2 * u, x + 3.8 * u, y - 6.8 * u), fill=ink)


def d_house(c: _Ctx, x, y, u, st):
    u *= st["grow"]; a = st["alpha"]; glow = .75 + .25 * math.sin(c.t * 2)
    c.d.rectangle((x - 6 * u, y - 7 * u, x + 6 * u, y), fill=_c(_mix(c.ground, c.ink, .35), a), outline=_c(c.ink, a), width=max(1, int(.4 * u)))
    c.d.polygon([(x - 7.5 * u, y - 7 * u), (x, y - 13 * u), (x + 7.5 * u, y - 7 * u)], fill=_c(_mix(c.ground, c.ink, .6), a))
    c.d.rectangle((x - 4.5 * u, y - 5 * u, x - 1.5 * u, y - 2 * u), fill=_c(GOLD, a * glow))
    c.d.rectangle((x + 1.5 * u, y - 5.5 * u, x + 4 * u, y), fill=_c(c.ink, a * .55))


def d_tree(c: _Ctx, x, y, u, st):
    u *= st["grow"]; a = st["alpha"]
    c.d.line((x, y, x, y - 6.5 * u), fill=_c((120, 84, 56), a), width=max(2, int(1.5 * u)))
    for dx, dy, r in ((0, -9, 4.6), (-3.2, -7, 3.3), (3.2, -7, 3.3), (0, -12, 3.2)):
        c.d.ellipse((x + (dx - r) * u, y + (dy - r) * u, x + (dx + r) * u, y + (dy + r) * u), fill=_c(LEAF, a * .9))


def d_seed(c: _Ctx, x, y, u, st):
    a, p = st["alpha"], st["p"]
    c.d.ellipse((x - 1.6 * u, y - 1.6 * u, x + 1.6 * u, y), fill=_c((150, 110, 70), a))
    h = 8 * u * p
    if h > 1:
        c.d.line((x, y - 1.4 * u, x, y - 1.4 * u - h), fill=_c(LEAF, a), width=max(2, int(.7 * u)))
    if p > .45:
        r = (p - .45) / .55
        for sgn in (-1, 1):
            c.d.polygon(_ellipse(x + sgn * 1.9 * u * r, y - 1.4 * u - h + .6 * u, 2 * u * r + .1, .9 * u * r + .1, sgn * -.5), fill=_c(LEAF, a))


def d_mountain(c: _Ctx, x, y, u, st):
    u *= st["grow"]; a = st["alpha"]; col = _c(_mix(c.ground, c.ink, .28), a)
    c.d.polygon([(x - 13 * u, y), (x - 2 * u, y - 13 * u), (x + 7 * u, y)], fill=col)
    c.d.polygon([(x - 2 * u, y), (x + 8 * u, y - 9 * u), (x + 16 * u, y)], fill=_c(_mix(c.ground, c.ink, .4), a))
    c.d.polygon([(x - 4.2 * u, y - 10.2 * u), (x - 2 * u, y - 13 * u), (x + .2 * u, y - 10.2 * u)], fill=_c(c.ink, a * .85))


def d_path(c: _Ctx, x, y, u, st):
    n, prog = 28, st["draw"]; left, right = [], []
    for i in range(int(n * prog) + 1):
        f = i / n
        cx = x + math.sin(f * 5) * 3 * u * (1 - f * .5); yy = y - f * 16 * u; hw = (4.5 * (1 - f) + .4) * u
        left.append((cx - hw, yy)); right.append((cx + hw, yy))
    if len(left) > 1:
        c.d.polygon(left + right[::-1], fill=_c(c.ink, st["alpha"] * .32))


def d_wave(c: _Ctx, x, y, u, st):
    for k in range(3):
        pts = [(x + (i - 12) * u, y - k * 1.8 * u + math.sin(i * .55 + c.t * 2 + k) * u) for i in range(25)]
        c.d.line(pts, fill=_c(WATER, st["alpha"] * (.9 - k * .2)), width=max(2, int(.5 * u)))


def d_sun(c: _Ctx, x, y, u, st):
    u *= st["grow"]; a = st["alpha"]
    for i in (4, 3, 2, 1):
        r = 4.2 * u + i * 1.9 * u
        c.d.ellipse((x - r, y - r, x + r, y + r), fill=_c(GOLD, a * .07))
    for k in range(10):
        ang = c.t * .3 + k * math.tau / 10
        c.d.line((x + math.cos(ang) * 6 * u, y + math.sin(ang) * 6 * u, x + math.cos(ang) * 8.2 * u, y + math.sin(ang) * 8.2 * u),
                 fill=_c(GOLD, a * .8), width=max(2, int(.5 * u)))
    c.d.ellipse((x - 4.2 * u, y - 4.2 * u, x + 4.2 * u, y + 4.2 * u), fill=_c(GOLD, a))


def d_moon(c: _Ctx, x, y, u, st):
    u *= st["grow"]; a = st["alpha"]
    bg = c.img.getpixel((max(0, min(c.W - 1, int(x))), max(0, min(c.H - 1, int(y)))))
    for i in (3, 2, 1):
        r = 3.6 * u + i * 1.5 * u
        c.d.ellipse((x - r, y - r, x + r, y + r), fill=_c((240, 236, 200), a * .06))
    c.d.ellipse((x - 3.6 * u, y - 3.6 * u, x + 3.6 * u, y + 3.6 * u), fill=_c((240, 236, 200), a))
    c.d.ellipse((x - 1.2 * u, y - 4.1 * u, x + 5 * u, y + 2.9 * u), fill=(*bg[:3], int(255 * min(1, a * 1.5))))


def d_stars(c: _Ctx, x, y, u, st):
    rng = random.Random(7)
    for i in range(16):
        sx, sy = c.W * rng.uniform(.06, .94), c.H * rng.uniform(.03, .27); r = rng.uniform(.18, .45) * u
        tw = .55 + .45 * math.sin(c.t * 1.8 + i * 1.7)
        c.d.ellipse((sx - r, sy - r, sx + r, sy + r), fill=_c((255, 250, 225), st["alpha"] * tw))


def d_cloud(c: _Ctx, x, y, u, st):
    u *= st["grow"]; x += math.sin(c.t * .35) * 1.6 * u
    col = _c((92, 100, 116) if c.mood == "storm" else (232, 238, 246), st["alpha"] * .88)
    for dx, dy, rx, ry in ((0, 0, 6, 2.6), (-4, .8, 4, 2), (4.2, .8, 4.2, 2.1), (-1.2, -1.6, 3.2, 2.4), (2, -1.4, 3, 2.2)):
        c.d.ellipse((x + (dx - rx) * u, y + (dy - ry) * u, x + (dx + rx) * u, y + (dy + ry) * u), fill=col)


def d_rain(c: _Ctx, x, y, u, st):
    for i in range(30):
        rx = x + (((i * 37) % 100) / 100 - .5) * .45 * c.W
        ry = c.H * (.20 + .62 * ((c.t * 1.3 + i * .173) % 1.0))
        c.d.line((rx, ry, rx - .6 * u, ry + 2.2 * u), fill=_c(WATER, st["alpha"] * .6), width=2)


def d_light_rays(c: _Ctx, x, y, u, st):
    for k in range(-3, 4):
        sway = .7 + .3 * math.sin(c.t * 1.2 + k)
        c.d.polygon([(x + k * .6 * u, 0), (x + k * .6 * u + 1.2 * u, 0),
                     (x + k * .13 * c.W + .045 * c.W, c.H * GROUND_Y), (x + k * .13 * c.W - .045 * c.W, c.H * GROUND_Y)],
                    fill=_c(GOLD, st["alpha"] * .11 * sway))


def d_bird(c: _Ctx, x, y, u, st):
    x += math.sin(c.t * .5) * 3 * u; flap = math.sin(c.t * 6) * 1.3 * u
    c.d.line((x - 3 * u, y + flap, x, y, x + 3 * u, y + flap), fill=_c(c.ink, st["alpha"]), width=max(2, int(.5 * u)))


def d_hands(c: _Ctx, x, y, u, st):
    u *= st["grow"]; a = st["alpha"]
    glow = .6 + .4 * math.sin(c.t * 2)
    for i in (3, 2, 1):
        c.d.ellipse((x - i * 3.2 * u, y - 9 * u - i * 2.4 * u, x + i * 3.2 * u, y - 9 * u + i * 2.4 * u), fill=_c(GOLD, a * .06 * glow))
    c.d.polygon(_ellipse(x - 2.4 * u, y - 4.5 * u, 1.7 * u, 4.2 * u, .55), fill=_c(c.ink, a))
    c.d.polygon(_ellipse(x + 2.4 * u, y - 4.5 * u, 1.7 * u, 4.2 * u, -.55), fill=_c(c.ink, a))


def d_heart(c: _Ctx, x, y, u, st):
    u *= st["grow"]; k = .36 * u
    pts = [(x + 16 * math.sin(t) ** 3 * k, y - 6 * u - (13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)) * k)
           for t in (i * math.tau / 40 for i in range(40))]
    c.d.polygon(pts, fill=_c((232, 84, 104), st["alpha"] * .92))


def d_coin(c: _Ctx, x, y, u, st):
    u *= st["grow"]; a = st["alpha"]; cy = y - 2.8 * u
    c.d.ellipse((x - 2.6 * u, cy - 2.6 * u, x + 2.6 * u, cy + 2.6 * u), fill=_c(GOLD, a), outline=_c((160, 120, 30), a), width=max(1, int(.3 * u)))
    c.d.ellipse((x - 1.6 * u, cy - 1.6 * u, x + 1.6 * u, cy + 1.6 * u), outline=_c((160, 120, 30), a), width=max(1, int(.25 * u)))


def d_scale(c: _Ctx, x, y, u, st):
    u *= st["grow"]; a = st["alpha"]; ink = _c(c.ink, a); tilt = math.sin(c.t * 1.1) * .13 * st["p"]; w = max(2, int(.55 * u))
    top = y - 12 * u; c.d.line((x, y, x, top), fill=ink, width=w)
    lx, ly = x - 6.5 * u, top + math.sin(tilt) * -6.5 * u; rx, ry = x + 6.5 * u, top + math.sin(tilt) * 6.5 * u
    c.d.line((lx, ly, rx, ry), fill=ink, width=w)
    for px, py in ((lx, ly), (rx, ry)):
        c.d.line((px, py, px - 2.2 * u, py + 4 * u), fill=ink, width=1); c.d.line((px, py, px + 2.2 * u, py + 4 * u), fill=ink, width=1)
        c.d.pieslice((px - 2.4 * u, py + 2 * u, px + 2.4 * u, py + 6 * u), 0, 180, fill=_c(GOLD, a * .9))


def d_phone(c: _Ctx, x, y, u, st):
    u *= st["grow"]; a = st["alpha"]; p = st["p"]
    c.d.rounded_rectangle((x - 3.4 * u, y - 13 * u, x + 3.4 * u, y), radius=int(.9 * u), fill=_c((14, 18, 28), a), outline=_c(c.ink, a), width=max(1, int(.35 * u)))
    c.d.rounded_rectangle((x - 2.8 * u, y - 11.8 * u, x + 2.8 * u, y - 1.3 * u), radius=int(.5 * u), fill=_c(_mix(c.horizon, (255, 255, 255), .2), a * .35))
    for i, (bx, by, bw) in enumerate(((-2.2, -10.5, 3.6), (-1.2, -7.5, 3.4), (-2.2, -4.5, 2.6))):
        if p > i * .3:
            c.d.rounded_rectangle((x + bx * u, y + by * u, x + (bx + bw) * u, y + (by + 2) * u), radius=int(.6 * u), fill=_c(GOLD if i == 1 else c.ink, a * .85))


def d_book(c: _Ctx, x, y, u, st):
    u *= st["grow"]; a = st["alpha"]; pulse = .6 + .4 * math.sin(c.t * 2)
    for i in (3, 2, 1):
        c.d.ellipse((x - 4 * i * u, y - 7 * u - 2 * i * u, x + 4 * i * u, y - 7 * u + 2 * i * u), fill=_c(GOLD, a * .05 * pulse))
    c.d.polygon([(x, y - 1 * u), (x - 7 * u, y - 2.5 * u), (x - 7 * u, y - 8 * u), (x, y - 6.5 * u)], fill=_c((242, 238, 222), a))
    c.d.polygon([(x, y - 1 * u), (x + 7 * u, y - 2.5 * u), (x + 7 * u, y - 8 * u), (x, y - 6.5 * u)], fill=_c((230, 226, 208), a))
    c.d.line((x, y - 1 * u, x, y - 6.5 * u), fill=_c((120, 100, 70), a), width=max(1, int(.3 * u)))


def d_door(c: _Ctx, x, y, u, st):
    u *= st["grow"]; a = st["alpha"]; glow = .5 + .25 * math.sin(c.t * 1.6) * st["p"]
    c.d.polygon([(x - 4 * u, y), (x + 4 * u, y), (x + 9 * u, y + (c.H - y) * .0 + 1.5 * u), (x - 9 * u, y + 1.5 * u)], fill=_c(GOLD, a * .18))
    c.d.rectangle((x - 4 * u, y - 12 * u, x + 4 * u, y), fill=_c(GOLD, a * glow), outline=_c(c.ink, a), width=max(2, int(.45 * u)))


def d_clock(c: _Ctx, x, y, u, st):
    u *= st["grow"]; a = st["alpha"]; cy = y - 6 * u; ink = _c(c.ink, a)
    c.d.ellipse((x - 5 * u, cy - 5 * u, x + 5 * u, cy + 5 * u), outline=ink, width=max(2, int(.6 * u)), fill=_c(c.ground, a * .5))
    for rate, ln in ((1.6, 4), (.13, 2.6)):
        ang = c.t * rate - math.pi / 2
        c.d.line((x, cy, x + math.cos(ang) * ln * u, cy + math.sin(ang) * ln * u), fill=ink, width=max(2, int(.5 * u)))


def d_check(c: _Ctx, x, y, u, st):
    u *= st["grow"]; a = st["alpha"]; cy = y - 5 * u
    c.d.ellipse((x - 4.6 * u, cy - 4.6 * u, x + 4.6 * u, cy + 4.6 * u), fill=_c(LEAF, a * .25), outline=_c(LEAF, a), width=max(2, int(.5 * u)))
    c.d.line((x - 2 * u, cy, x - .5 * u, cy + 1.7 * u, x + 2.4 * u, cy - 1.8 * u), fill=_c(LEAF, a), width=max(2, int(.8 * u)))


def d_arrow(c: _Ctx, x, y, u, st):
    a = st["alpha"]; end = x - 6 * u + 11 * u * st["draw"]; cy = y - 4 * u; col = _c(GOLD, a)
    c.d.line((x - 6 * u, cy, end, cy), fill=col, width=max(2, int(.8 * u)))
    c.d.polygon([(end + 1.8 * u, cy), (end - .6 * u, cy - 1.7 * u), (end - .6 * u, cy + 1.7 * u)], fill=col)


DRAWERS: dict[str, Callable[..., None]] = {
    "person": d_person, "sitting_person": d_sitting_person, "house": d_house, "tree": d_tree, "seed": d_seed,
    "mountain": d_mountain, "path": d_path, "wave": d_wave, "sun": d_sun, "moon": d_moon, "stars": d_stars,
    "cloud": d_cloud, "rain": d_rain, "light_rays": d_light_rays, "bird": d_bird, "hands": d_hands,
    "heart": d_heart, "coin": d_coin, "scale": d_scale, "phone": d_phone, "book": d_book, "door": d_door,
    "clock": d_clock, "check": d_check, "arrow": d_arrow,
}
ELEMENT_TYPES = frozenset(DRAWERS)
SKY_TYPES = frozenset({"sun", "moon", "stars", "cloud", "rain", "light_rays", "bird"})
# Back-to-front order (lower first); everything else is drawn last.
LAYER_RANK = {"stars": 0, "sun": 0, "moon": 0, "cloud": 1, "light_rays": 1, "mountain": 1, "path": 2, "wave": 2, "rain": 3}
ASSET_PIXELS = 18   # an asset is drawn in a square of ASSET_PIXELS * u


def layer_of(element_type: str) -> str:
    return "sky" if element_type in SKY_TYPES else "ground"


# ------------------------------------------------------------------ optional Lottie / SVG assets
class _LottieAsset:
    def __init__(self, path: Path):
        # Validate in Python first: rlottie accepts some garbage files without raising, and
        # calling into the native library with a null animation is unsafe.
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ValueError("ملف Lottie ليس JSON صالحًا.") from None
        layers = data.get("layers") if isinstance(data, dict) else None
        if (not isinstance(layers, list) or not layers
                or not all(isinstance(data.get(key), (int, float)) and data[key] > 0 for key in ("fr", "w", "h"))
                or not isinstance(data.get("op"), (int, float)) or data["op"] <= data.get("ip", 0)):
            raise ValueError("ملف Lottie ينقصه layers أو fr أو w أو h أو op.")
        from rlottie_python import LottieAnimation
        self.anim = LottieAnimation.from_file(str(path))
        self.frames = max(1, int(self.anim.lottie_animation_get_totalframe()))
        self.fps = float(self.anim.lottie_animation_get_framerate() or 24)
        self.frame(0.0, 16)                    # fail now (and fall back) rather than in the middle of a render

    def frame(self, t: float, size: int) -> Image.Image:
        number = int(t * self.fps) % self.frames
        return self.anim.render_pillow_frame(frame_num=number, width=size, height=size).convert("RGBA")


class _SvgAsset:
    def __init__(self, path: Path):
        import cairosvg
        self.path, self.cairosvg, self._cache = path, cairosvg, {}

    def frame(self, t: float, size: int) -> Image.Image:
        if size not in self._cache:
            png = self.cairosvg.svg2png(url=str(self.path), output_width=size, output_height=size)
            self._cache[size] = Image.open(io.BytesIO(png)).convert("RGBA")
        return self._cache[size]


class VisualAssets:
    """Operator-supplied animated assets: assets/visuals/lottie/<type>.json or svg/<type>.svg.

    Used only when VISUAL_ASSETS_LICENSE_CONFIRMED=true and VISUAL_ASSETS_LICENSE_URL is HTTPS;
    otherwise (or when a file/library is missing) the built-in Pillow drawer is used. Every asset
    used is listed in `used` so the production report keeps a licence trail.
    """
    def __init__(self, root: Path | None = None):
        self.root = Path(root or os.getenv("VISUAL_ASSET_DIR", "") or ASSET_ROOT)
        confirmed = os.getenv("VISUAL_ASSETS_LICENSE_CONFIRMED", "").strip().lower() in {"1", "true", "yes", "on"}
        self.license_url = os.getenv("VISUAL_ASSETS_LICENSE_URL", "").strip()
        self.enabled = confirmed and urlparse(self.license_url).scheme == "https" and self.root.is_dir()
        self.used: dict[str, dict[str, str]] = {}
        self.notes: list[str] = []
        self._cache: dict[str, Any] = {}

    def get(self, element_type: str) -> Any:
        if not self.enabled:
            return None
        if element_type in self._cache:
            return self._cache[element_type]
        asset = None
        for fmt, suffix, loader in (("lottie", ".json", _LottieAsset), ("svg", ".svg", _SvgAsset)):
            path = self.root / fmt / f"{element_type}{suffix}"
            if path.is_file():
                try:
                    asset = loader(path)
                    self.used[element_type] = {"type": element_type, "format": fmt, "file": path.name,
                                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                               "license_url": self.license_url}
                    break
                except Exception as exc:       # missing library, corrupt file: fall back to the built-in drawer
                    self.notes.append(f"{fmt}/{path.name}: {type(exc).__name__}")
        self._cache[element_type] = asset
        return asset


# ------------------------------------------------------------------ scene rendering
def _anim_state(animation: str, local_t: float, t: float) -> dict[str, float]:
    p = _ease(local_t / APPEAR_SECONDS)
    st = {"alpha": 1.0, "dx": 0.0, "dy": 0.0, "grow": 1.0, "step": 0.0, "p": 1.0, "draw": _ease(local_t / 2.0)}
    if animation == "none":
        st["draw"] = 1.0
        return st
    st["p"] = p
    if animation == "fade_in":
        st["alpha"] = p
    elif animation == "rise":
        st.update(alpha=p, dy=(1 - p) * .05)
    elif animation == "fall":
        st.update(alpha=min(1.0, p * 2), dy=-(1 - p) * .12)
    elif animation == "grow":
        st.update(alpha=min(1.0, p * 2), grow=.15 + .85 * p)
    elif animation == "pulse":
        st.update(alpha=p, grow=1 + .05 * math.sin(t * 3.2))
    elif animation == "float":
        st.update(alpha=p, dy=.008 * math.sin(t * 1.7))
    elif animation == "walk":
        w = _ease(local_t / 3.0)
        st.update(alpha=min(1.0, local_t / .5), dx=-(1 - w) * .22, step=(t * 7 if w < 1 else 0.0), p=1.0)
    return st


@lru_cache(maxsize=24)
def _gradient(mood: str, w: int, h: int) -> Image.Image:
    top, horizon, ground, _ = MOODS[mood]
    column = Image.new("RGB", (1, h))
    column.putdata([_mix(top, horizon, min(1.0, (y / (h * GROUND_Y)) ** 1.2)) for y in range(h)])
    image = column.resize((w, h))
    ImageDraw.Draw(image).rectangle((0, int(h * GROUND_Y), w, h), fill=ground)
    ImageDraw.Draw(image).line((0, int(h * GROUND_Y), w, int(h * GROUND_Y)), fill=_mix(ground, horizon, .35), width=2)
    return image


def _draw_element(c: _Ctx, element: dict[str, Any], local_t: float) -> None:
    kind = element["type"]
    st = _anim_state(element.get("animation", "fade_in"), local_t, c.t)
    u = c.H * .01 * SIZES.get(element.get("size", "medium"), 1.0)
    x = c.W * (POSITIONS.get(element.get("position", "center"), .5) + st["dx"])
    sky = kind in SKY_TYPES
    y = c.H * (SKY_Y if sky else GROUND_Y) + st["dy"] * c.H
    asset = c.assets.get(kind) if c.assets else None
    if asset is not None and st["alpha"] > 0:
        size = max(8, int(u * ASSET_PIXELS))
        try:
            frame = asset.frame(c.t, size)
            mask = frame.getchannel("A").point(lambda v, k=st["alpha"]: int(v * k))
            c.img.paste(frame.convert("RGB"), (int(x - size / 2), int(y - (size / 2 if sky else size))), mask)
            return
        except Exception:                          # a broken asset must never break the video
            c.assets._cache[kind] = None
            c.assets.used.pop(kind, None)
            c.assets.notes.append(f"{kind}: render failed, built-in drawer used")
    if st["alpha"] > 0:
        DRAWERS[kind](c, x, y, u, st)


def render_scene_image(width: int, height: int, scene: dict[str, Any], local_t: float, t: float,
                       assets: VisualAssets | None = None) -> Image.Image:
    image = _gradient(scene["mood"], width, height).copy()
    ctx = _Ctx(image, scene["mood"], t, assets)
    rng = random.Random(11)
    for _ in range(14):                                   # a few drifting dust motes: depth, not content
        px, py, pr = rng.random() * width, rng.random() * height, rng.uniform(.8, 1.8)
        py = (py - t * 7) % height
        ctx.d.ellipse((px - pr, py - pr, px + pr, py + pr), fill=_c(ctx.ink, .16))
    for element in sorted(scene["elements"], key=lambda e: LAYER_RANK.get(e["type"], 5)):
        _draw_element(ctx, element, local_t)
    return image


# ------------------------------------------------------------------ AI image scenes (Ken Burns)
KEN_BURNS_ZOOM = 0.14
CALM_DARKEN = 0.22
VIGNETTE_STRENGTH = 0.5


class _ImageStore:
    """Loads each scene image once; a missing or corrupt file just means 'use the procedural scene'."""
    def __init__(self) -> None:
        self._cache: dict[str, Image.Image | None] = {}

    def get(self, path: str) -> Image.Image | None:
        if path not in self._cache:
            try:
                with Image.open(path) as handle:
                    self._cache[path] = handle.convert("RGB")
            except Exception:
                self._cache[path] = None
        return self._cache[path]


@lru_cache(maxsize=8)
def _vignette_mask(width: int, height: int) -> Image.Image:
    mask = Image.radial_gradient("L").resize((width, height), Image.Resampling.BILINEAR)
    return mask.point(lambda v: int(max(0, v - 90) / 165 * 255 * VIGNETTE_STRENGTH))


def render_image_frame(width: int, height: int, source: Image.Image, progress: float, variant: int,
                       calm: bool = False) -> Image.Image:
    """Slow zoom + drift over `source` (cover-fit to width x height); direction alternates by `variant`."""
    p = _ease(progress)
    zoom = 1.0 + KEN_BURNS_ZOOM * (p if variant % 2 == 0 else 1.0 - p)
    sw, sh = source.size
    aspect = width / height
    if sw / sh >= aspect:
        view_h = sh / zoom
        view_w = view_h * aspect
    else:
        view_w = sw / zoom
        view_h = view_w / aspect
    slack_x, slack_y = (sw - view_w) / 2, (sh - view_h) / 2
    drift = (progress - .5) * 2
    cx = sw / 2 + (-1, 1, 1, -1)[variant % 4] * drift * slack_x * .7
    cy = sh / 2 + (1, -1)[(variant // 2) % 2] * drift * slack_y * .5
    left = min(max(cx - view_w / 2, 0.0), sw - view_w)
    top = min(max(cy - view_h / 2, 0.0), sh - view_h)
    frame = source.resize((width, height), Image.Resampling.BICUBIC, box=(left, top, left + view_w, top + view_h))
    black = Image.new("RGB", (width, height), (0, 0, 0))
    frame = Image.composite(black, frame, _vignette_mask(width, height))
    if calm:
        frame = Image.blend(frame, black, CALM_DARKEN)
    return frame


def _render_entry(width: int, height: int, index: int, entry: dict[str, Any], local: float, t: float,
                  assets: VisualAssets | None, store: _ImageStore) -> Image.Image:
    path = entry.get("image")
    if path:
        source = store.get(path)
        if source is not None:
            length = max(entry["end"] - entry["start"], .001)
            return render_image_frame(width, height, source, max(0.0, min(1.0, local / length)), index,
                                      bool(entry.get("calm")))
    return render_scene_image(width, height, entry["scene"], local, t, assets)


def frames(width: int, height: int, fps: int, duration: float, timeline: list[dict[str, Any]],
           assets: VisualAssets | None = None) -> Iterator[bytes]:
    starts = [entry["start"] for entry in timeline]
    store = _ImageStore()
    for number in range(int(math.ceil(duration * fps))):
        t = number / fps
        index = max(0, bisect_right(starts, t) - 1)
        entry = timeline[index]
        local = t - entry["start"]
        image = _render_entry(width, height, index, entry, local, t, assets, store)
        if index > 0 and local < XFADE_SECONDS:           # soft cross-fade; the old scene keeps moving
            previous = timeline[index - 1]
            old = _render_entry(width, height, index - 1, previous, t - previous["start"], t, assets, store)
            image = Image.blend(old, image, _ease(local / XFADE_SECONDS))
        yield image.tobytes()


# ------------------------------------------------------------------ script -> timeline
MAX_SHOT_SECONDS = 3.2        # a picture is held about 2.5-3.2 s; a longer shot is cut into several crops
SPLIT_THRESHOLD = 1.35
RECITATION_SHOT_SECONDS = 4.0
OPENING_MIN_SHOT_SECONDS = 1.2      # the hook may cut faster than the rest to grab attention
DEFAULT_ELEMENTS = ({"type": "light_rays", "position": "center", "animation": "fade_in", "size": "medium"},)


def _segment_shots(shots: list[dict[str, Any]], timings: list[tuple[float, float, int]] | None, start: float,
                   end: float, key_prefix: str, min_len: float = MIN_SCENE_SECONDS) -> list[dict[str, Any]]:
    """Cut one spoken segment into shots.

    With `timings` (the message) each shot opens at the word it is bound to; without them (the hook
    and the call to action, whose wording is not known to the planner) the shots share the segment
    evenly. The first shot always opens the segment; a shot that would start too close to the
    previous one is merged into it.
    """
    kept: list[tuple[float, dict[str, Any], int]] = []
    for position, shot in enumerate(shots):
        if position == 0:
            shot_start = start
        elif timings is None:
            shot_start = start + (end - start) * position / len(shots)
        else:
            index = shot.get("word_start")
            if not isinstance(index, int) or not 0 <= index < len(timings):
                continue
            shot_start = start + timings[index][0]
        if kept and shot_start - kept[-1][0] < min_len:
            continue
        kept.append((shot_start, shot, position))
    if len(kept) > 1 and end - kept[-1][0] < min_len:
        kept.pop()
    if not kept:
        raise ValueError("لا توجد لقطات صالحة لهذا المقطع.")
    entries = []
    for number, (shot_start, shot, position) in enumerate(kept):
        shot_end = kept[number + 1][0] if number + 1 < len(kept) else end
        scene = {**shot, "elements": list(shot.get("elements") or DEFAULT_ELEMENTS)}
        entries.append({"start": shot_start, "end": shot_end, "scene": scene, "image_key": f"{key_prefix}-{position + 1}"})
    return entries


def _split_long(entries: list[dict[str, Any]], name: str, limit: float = MAX_SHOT_SECONDS,
                threshold: float = SPLIT_THRESHOLD) -> list[dict[str, Any]]:
    """A shot longer than ~4.3 s becomes several equal parts, each with its own camera move on the same picture."""
    result: list[dict[str, Any]] = []
    for entry in entries:
        length = entry["end"] - entry["start"]
        parts = 1 if length <= limit * threshold else math.ceil(length / limit)
        for part in range(parts):
            result.append({**entry, "name": f"{name}-{len(result) + 1}",
                           "start": round(entry["start"] + length * part / parts, 3),
                           "end": round(entry["start"] + length * (part + 1) / parts, 3),
                           "image": entry.get("image"), "calm": entry.get("calm", False)})
    return result


def shot_index(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every shot of the plan by its image key: opening-N, scene-N, closing-N."""
    index: dict[str, dict[str, Any]] = {}
    for number, shot in enumerate((plan.get("opening") or {}).get("shots", []), start=1):
        index[f"opening-{number}"] = shot
    for number, shot in enumerate(plan.get("scenes", []), start=1):
        index[f"scene-{number}"] = shot
    for number, shot in enumerate((plan.get("closing") or {}).get("shots", []), start=1):
        index[f"closing-{number}"] = shot
    return index


def timeline_image_keys(timeline: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(entry["image_key"] for entry in timeline))


def pick_image_keys(keys: list[str], limit: int) -> list[str]:
    """Which shots get a generated picture when the budget is smaller than the shot count.

    The opening and closing shots always come first; the message shots are sampled evenly.
    """
    keys = list(dict.fromkeys(keys))
    if len(keys) <= limit:
        return keys
    fixed = [key for key in keys if not key.startswith("scene-")][:max(0, limit)]
    middle = [key for key in keys if key.startswith("scene-")]
    room = max(0, limit - len(fixed))
    chosen: set[str] = set()
    if room and middle:
        step = len(middle) / room
        chosen = {middle[min(len(middle) - 1, int(i * step))] for i in range(room)}
    keep = set(fixed) | chosen
    return [key for key in keys if key in keep]


def apply_images(timeline: list[dict[str, Any]], images: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach generated pictures; a shot without one reuses the nearest picture instead of switching style."""
    for entry in timeline:
        found = (images.get(entry["image_key"]) or {}).get("path")
        entry["image"], entry["reused_image"] = found, False
    last = None
    for entry in timeline:
        if entry["image"]:
            last = entry["image"]
        elif last:
            entry["image"], entry["reused_image"] = last, True
    following = None
    for entry in reversed(timeline):
        if entry["image"] and not entry["reused_image"]:
            following = entry["image"]
        elif not entry["image"] and following:
            entry["image"], entry["reused_image"] = following, True
    return timeline


def build_timeline(plan: dict[str, Any], message_timings: list[tuple[float, float, int]],
                   intervals: dict[str, tuple[float, float]], duration: float,
                   images: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Turn the validated plan into absolute, gap-free shots: a new picture about every 2.5-3 seconds.

    The message shots open at the words they are bound to (`message_timings` are RELATIVE to the
    message audio, as returned by video_engine.align_speech_words); the hook and call-to-action shots
    share their segment evenly. `intervals` are the absolute (start, end) of each audio segment exactly
    as build_video lays them out. During the recitation the opening pictures come back one after
    another under a dark veil.
    """
    for key in ("hook", "recitation", "seal", "message", "cta"):
        if key not in intervals:
            raise ValueError(f"فترة الصوت {key} مفقودة.")
    hook = _split_long(_segment_shots(plan["opening"]["shots"], None, *intervals["hook"], "opening",
                                      min_len=OPENING_MIN_SHOT_SECONDS), "hook")
    message = _split_long(_segment_shots(plan["scenes"], message_timings, *intervals["message"], "scene"), "message")
    cta_end = max(duration, intervals["cta"][1])
    cta = _split_long(_segment_shots(plan["closing"]["shots"], None, intervals["cta"][0], cta_end, "closing"), "cta")
    rec_start, rec_end = intervals["recitation"][0], intervals["seal"][1]
    opening_keys = list(dict.fromkeys(entry["image_key"] for entry in hook))
    mood_of = {entry["image_key"]: entry["scene"]["mood"] for entry in hook}
    parts = max(1, int(round((rec_end - rec_start) / RECITATION_SHOT_SECONDS)))
    recitation = []
    for part in range(parts):
        key = opening_keys[part % len(opening_keys)]
        recitation.append({"name": f"recitation-{part + 1}",
                           "start": round(rec_start + (rec_end - rec_start) * part / parts, 3),
                           "end": round(rec_start + (rec_end - rec_start) * (part + 1) / parts, 3),
                           "scene": {"mood": mood_of[key], "elements": list(DEFAULT_ELEMENTS)},
                           "image_key": key, "image": None, "calm": True})
    entries = [*hook, *recitation, *message, *cta]
    if images:
        apply_images(entries, images)
    return entries


MIN_PIECE_SECONDS = 0.8     # a picture is never cut into a piece shorter than this (verse edges excepted)


def build_script_timeline(plan: dict[str, Any], segments: list[dict[str, Any]], duration: float,
                          images: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Gap-free shots for a script that alternates spoken parts and recited verses.

    `segments` are the audio pieces in order, each with absolute `start`/`end`:
      {"kind": "speech", "timings": [(start, end, index)...]}   timings relative to that piece, one per word
      {"kind": "verse"}
    The first segment is the hook: its shots (plan["opening"]) share it evenly. After the hook, the shots in
    plan["scenes"] open at the words they are bound to; word indexes run over the spoken words of all later
    speech segments in order. A recited verse keeps the current picture under a dark veil (cut into several
    crops if it is long), so the picture changes only where the narrator says something new.
    """
    if not segments or segments[0].get("kind") != "speech" or segments[0].get("role") != "hook":
        raise ValueError("يجب أن يبدأ التسلسل بمقطع الهوك المنطوق.")
    verse_spans = [(seg["start"], seg["end"]) for seg in segments[1:] if seg["kind"] == "verse"]
    if not verse_spans:
        raise ValueError("لا توجد آية في التسلسل.")
    hook = segments[0]
    hook_entries = _split_long(_segment_shots(plan["opening"]["shots"], None, hook["start"], hook["end"], "opening",
                                              min_len=OPENING_MIN_SHOT_SECONDS), "hook")
    body_start, end_time = hook["end"], max(duration, segments[-1]["end"])
    word_times = [seg["start"] + timing[0] for seg in segments[1:] if seg["kind"] == "speech" for timing in seg["timings"]]
    starts: list[tuple[float, dict[str, Any], int]] = []
    for position, shot in enumerate(plan["scenes"]):
        if position == 0:
            shot_time = body_start
        else:
            index = shot.get("word_start")
            if not isinstance(index, int) or not 0 <= index < len(word_times):
                continue
            shot_time = word_times[index]
        if starts and shot_time - starts[-1][0] < MIN_SCENE_SECONDS:
            continue
        starts.append((shot_time, shot, position))
    if not starts:
        raise ValueError("لا توجد لقطات صالحة بعد الهوك.")
    mandatory = {body_start, end_time} | {edge for span in verse_spans for edge in span}
    ordered = sorted({cut for cut in ({time for time, _, _ in starts} | mandatory) if body_start <= cut <= end_time})
    cuts = [ordered[0]]
    for cut in ordered[1:]:
        if cut - cuts[-1] < MIN_PIECE_SECONDS:
            if cut in mandatory and cuts[-1] not in mandatory:
                cuts[-1] = cut                      # a verse edge wins over a shot cut that sits right next to it
            elif cut in mandatory:
                cuts.append(cut)
            continue
        cuts.append(cut)
    body_entries: list[dict[str, Any]] = []
    for left, right in zip(cuts, cuts[1:]):
        shot_time, shot, position = [item for item in starts if item[0] <= left + 1e-6][-1]
        calm = any(a - 1e-6 <= left and right <= b + 1e-6 for a, b in verse_spans)
        scene = ({"mood": shot["mood"], "elements": list(DEFAULT_ELEMENTS)} if calm
                 else {**shot, "elements": list(shot.get("elements") or DEFAULT_ELEMENTS)})
        piece = {"start": round(left, 3), "end": round(right, 3), "scene": scene, "image_key": f"scene-{position + 1}",
                 "image": None, "calm": calm}
        body_entries.extend(_split_long([piece], "calm" if calm else "scene",
                                        RECITATION_SHOT_SECONDS if calm else MAX_SHOT_SECONDS,
                                        1.5 if calm else SPLIT_THRESHOLD))
    for number, entry in enumerate(body_entries, start=1):
        entry["name"] = f"{'recitation' if entry['calm'] else 'scene'}-{number}"
    entries = [*hook_entries, *body_entries]
    if images:
        apply_images(entries, images)
    return entries


# ------------------------------------------------------------------ encoding
def render_background(destination: Path, width: int, height: int, fps: int, duration: float,
                      timeline: list[dict[str, Any]], theme: str = "", render_scale: float = .5,
                      assets: VisualAssets | None = None) -> dict[str, Any]:
    if (width <= 0 or height <= 0 or width % 2 or height % 2 or fps <= 0 or duration <= 0
            or not .25 <= render_scale <= 1.0 or not timeline):
        raise ValueError("أبعاد أو معدل إطارات أو مدة أو جدول مشاهد الخلفية غير صالح.")
    if assets is None:
        assets = VisualAssets()
    rw, rh = max(2, int(width * render_scale) // 2 * 2), max(2, int(height * render_scale) // 2 * 2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{rw}x{rh}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "20", "-vf", f"scale={width}:{height}:flags=lanczos", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(destination)]
    process = None
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        assert process.stdin is not None
        for frame in frames(rw, rh, fps, duration, timeline, assets):
            process.stdin.write(frame)
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if process.stderr:
            process.stderr.close()
    except (BrokenPipeError, OSError) as exc:
        if process is not None:
            process.kill()
        raise RuntimeError(f"تعذر بث الإطارات إلى FFmpeg: {exc}") from exc
    if return_code != 0:
        raise RuntimeError(f"فشل إنشاء الخلفية الإجرائية: {stderr[-500:]}")
    if not destination.is_file() or destination.stat().st_size < 1024:
        raise RuntimeError("لم ينتج مولّد الخلفية ملفًا صالحًا.")
    return {"kind": "script_synced_scenes", "width": width, "height": height,
            "render_width": rw, "render_height": rh, "render_scale": render_scale, "fps": fps,
            "duration_seconds": round(duration, 3),
            "theme_hash": hashlib.sha256(theme.encode("utf-8")).hexdigest()[:16],
            "image_scenes": sum(1 for e in timeline if e.get("image")),
            "scenes": [{"name": e["name"], "start": e["start"], "end": e["end"], "mood": e["scene"]["mood"],
                        "image": bool(e.get("image")),
                        "elements": [x["type"] for x in e["scene"]["elements"]]} for e in timeline],
            "operator_supplied_assets": list(assets.used.values()), "asset_notes": assets.notes,
            "external_assets": False}


# ------------------------------------------------------------------ tests
if __name__ == "__main__":
    import tempfile
    import unittest

    def element(kind, position="center", animation="fade_in", size="medium"):
        return {"type": kind, "position": position, "animation": animation, "size": size}

    def shot(mood, *els, start=None, prompt=""):
        out = {"mood": mood, "elements": list(els) or [element("light_rays")], "image_prompt": prompt}
        if start is not None:
            out["word_start"] = start
        return out

    def sample_plan():
        return {"opening": {"shots": [shot("night", element("sitting_person"), element("moon", "right"), start=0),
                                      shot("storm", element("cloud"), element("rain"), start=6)]},
                "scenes": [shot("storm", element("cloud"), element("rain"), element("person", "left", "walk"), start=0),
                           shot("dawn", element("sun", "right"), element("path", "center"), start=10),
                           shot("day", element("seed", "center", "grow"), element("tree", "right", "grow"), start=22),
                           shot("warm", element("heart"), element("hands", "left"), start=30)],
                "closing": {"shots": [shot("warm", element("heart"), element("hands", "left"), start=0)]}}

    def timings(count, step=.4):
        return [(i * step, i * step + step - .05, i) for i in range(count)]

    INTERVALS = {"hook": (0.0, 6.0), "recitation": (6.0, 18.0), "seal": (18.0, 20.0), "message": (20.0, 44.0), "cta": (44.0, 50.0)}

    def line_for(plan=None, **kwargs):
        return build_timeline(plan or sample_plan(), timings(60), INTERVALS, 50.0, **kwargs)

    class BackgroundTests(unittest.TestCase):
        def test_every_element_type_renders_with_every_animation(self):
            for kind in sorted(ELEMENT_TYPES):
                for animation in sorted(ANIMATIONS):
                    scene = {"mood": "dawn", "elements": [element(kind, "center", animation, "large")]}
                    for local in (0.1, 1.0, 5.0):
                        image = render_scene_image(180, 320, scene, local, local + 3.0)
                        self.assertEqual(image.size, (180, 320), (kind, animation))

        def test_every_mood_renders_and_ground_elements_sit_below_the_text_cards(self):
            for mood in MOODS:
                render_scene_image(180, 320, {"mood": mood, "elements": [element("house")]}, 2.0, 2.0)
            a = render_scene_image(180, 320, {"mood": "night", "elements": []}, 2.0, 2.0)
            b = render_scene_image(180, 320, {"mood": "night", "elements": [element("house")]}, 2.0, 2.0)
            changed = [y for y in range(320) for x in range(0, 180, 3) if a.getpixel((x, y)) != b.getpixel((x, y))]
            self.assertTrue(changed and min(changed) > 320 * .60)

        def test_timeline_cuts_exactly_when_each_phrase_is_spoken_and_is_gap_free(self):
            line = line_for()
            message = [e for e in line if e["name"].startswith("message")]
            self.assertEqual(timeline_image_keys(message), ["scene-1", "scene-2", "scene-3", "scene-4"])
            starts = {e["image_key"]: e["start"] for e in reversed(message)}
            for key, expected in (("scene-1", 20.0), ("scene-2", 20.0 + 10 * .4), ("scene-3", 20.0 + 22 * .4), ("scene-4", 20.0 + 30 * .4)):
                self.assertAlmostEqual(starts[key], expected, places=3)
            hook = [e for e in line if e["name"].startswith("hook")]
            self.assertEqual(timeline_image_keys(hook), ["opening-1", "opening-2"])
            self.assertAlmostEqual([e for e in hook if e["image_key"] == "opening-2"][0]["start"], 3.0, places=3)   # even split of 6 s
            self.assertEqual((line[0]["start"], line[-1]["end"]), (0.0, 50.0))
            for left, right in zip(line, line[1:]):
                self.assertAlmostEqual(left["end"], right["start"], places=3)

        def test_no_picture_is_held_much_longer_than_three_seconds_outside_the_recitation(self):
            for entry in line_for():
                limit = RECITATION_SHOT_SECONDS * 1.5 if entry["calm"] else MAX_SHOT_SECONDS * SPLIT_THRESHOLD
                self.assertLessEqual(entry["end"] - entry["start"], limit + 0.01, entry["name"])
            names = [e["name"] for e in line_for() if e["name"].startswith("message")]
            self.assertGreaterEqual(len(names), 8)                     # ~24 s of message => a new picture every ~2.5-3 s

        def test_recitation_cycles_the_opening_pictures_under_a_veil(self):
            rec = [e for e in line_for() if e["name"].startswith("recitation")]
            self.assertEqual(len(rec), 4)
            self.assertEqual([e["image_key"] for e in rec], ["opening-1", "opening-2", "opening-1", "opening-2"])
            self.assertTrue(all(e["calm"] for e in rec))

        def test_timeline_merges_shots_that_are_too_close_and_keeps_fast_hook_cuts(self):
            plan = sample_plan(); plan["scenes"][1]["word_start"] = 2          # 0.8 s after the first: merged away
            self.assertEqual(timeline_image_keys([e for e in line_for(plan) if e["name"].startswith("message")]),
                             ["scene-1", "scene-3", "scene-4"])
            short = dict(INTERVALS, hook=(0.0, 2.0), recitation=(2.0, 18.0))      # a 2 s hook: 1.0 s per shot is too short
            fast = build_timeline(sample_plan(), timings(60), short, 50.0)
            self.assertEqual(timeline_image_keys([e for e in fast if e["name"].startswith("hook")]), ["opening-1"])
            ok = dict(INTERVALS, hook=(0.0, 2.6), recitation=(2.6, 18.0))         # 1.3 s per shot is allowed for the hook
            self.assertEqual(timeline_image_keys([e for e in build_timeline(sample_plan(), timings(60), ok, 50.0)
                                                  if e["name"].startswith("hook")]), ["opening-1", "opening-2"])
            self.assertEqual(fast[-1]["end"], 50.0)
            with self.assertRaises(ValueError):
                build_timeline(sample_plan(), timings(60), {"hook": (0, 1)}, 10.0)

        def test_shot_index_pick_image_keys_and_apply_images(self):
            plan = sample_plan()
            self.assertEqual(sorted(shot_index(plan)), ["closing-1", "opening-1", "opening-2", "scene-1", "scene-2", "scene-3", "scene-4"])
            keys = ["opening-1", "opening-2", *[f"scene-{i}" for i in range(1, 13)], "closing-1"]
            self.assertEqual(pick_image_keys(keys, 99), keys)
            picked = pick_image_keys(keys, 8)
            self.assertEqual(len(picked), 8)
            self.assertTrue({"opening-1", "opening-2", "closing-1"} <= set(picked))
            self.assertEqual(picked, [k for k in keys if k in picked])           # order preserved
            self.assertEqual(len(pick_image_keys(keys, 2)), 2)
            line = line_for()
            apply_images(line, {"opening-1": {"path": "/x/o1.jpg"}, "scene-2": {"path": "/x/s2.jpg"}})
            by_key = {}
            for e in line:
                by_key.setdefault(e["image_key"], e)
            self.assertEqual(by_key["opening-1"]["image"], "/x/o1.jpg")
            self.assertFalse(by_key["opening-1"]["reused_image"])
            self.assertEqual(by_key["opening-2"]["image"], "/x/o1.jpg")           # reuses the nearest earlier picture
            self.assertTrue(by_key["opening-2"]["reused_image"])
            self.assertEqual(by_key["scene-2"]["image"], "/x/s2.jpg")
            self.assertEqual(by_key["closing-1"]["image"], "/x/s2.jpg")
            apply_images(line, {})
            self.assertTrue(all(e["image"] is None for e in line))
            only_late = line_for()
            apply_images(only_late, {"scene-3": {"path": "/x/s3.jpg"}})
            self.assertTrue(all(e["image"] == "/x/s3.jpg" for e in only_late))      # earlier shots borrow the first real picture

        def test_frames_stream_has_exact_size_and_count(self):
            got = list(frames(90, 160, 2, 50.0, line_for()))
            self.assertEqual(len(got), 100)
            self.assertTrue(all(len(f) == 90 * 160 * 3 for f in got))

        def test_assets_are_ignored_without_license_confirmation(self):
            with tempfile.TemporaryDirectory() as temp:
                (Path(temp) / "svg").mkdir()
                (Path(temp) / "svg" / "sun.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"/>')
                from unittest.mock import patch
                with patch.dict(os.environ, {"VISUAL_ASSETS_LICENSE_CONFIRMED": "", "VISUAL_ASSETS_LICENSE_URL": ""}):
                    self.assertIsNone(VisualAssets(Path(temp)).get("sun"))
                with patch.dict(os.environ, {"VISUAL_ASSETS_LICENSE_CONFIRMED": "true", "VISUAL_ASSETS_LICENSE_URL": "http://insecure"}):
                    self.assertFalse(VisualAssets(Path(temp)).enabled)

        def test_missing_asset_library_falls_back_to_the_builtin_drawer(self):
            with tempfile.TemporaryDirectory() as temp:
                (Path(temp) / "lottie").mkdir()
                (Path(temp) / "lottie" / "sun.json").write_text("{}")
                from unittest.mock import patch
                with patch.dict(os.environ, {"VISUAL_ASSETS_LICENSE_CONFIRMED": "true", "VISUAL_ASSETS_LICENSE_URL": "https://example.org/l"}):
                    assets = VisualAssets(Path(temp))
                    self.assertTrue(assets.enabled)
                    asset = assets.get("sun")                          # corrupt/unsupported file -> None, never an exception
                    image = render_scene_image(120, 200, {"mood": "day", "elements": [element("sun", "center")]}, 1.0, 1.0, assets)
                self.assertIsNone(asset)
                self.assertEqual(image.size, (120, 200))

        def test_render_background_encodes_a_valid_video(self):
            with tempfile.TemporaryDirectory() as temp:
                line = line_for()
                out = Path(temp) / "bg.mp4"
                info = render_background(out, 180, 320, 4, 50.0, line, theme="t", render_scale=.5)
                self.assertTrue(out.stat().st_size > 1024)
                self.assertEqual(len(info["scenes"]), len(line))
                self.assertFalse(info["external_assets"])

        # ----- AI image scenes -----
        @staticmethod
        def _photo(path, size=(512, 512)):
            from PIL import ImageChops
            gradient = Image.linear_gradient("L").resize(size)
            noise = Image.effect_noise(size, 40)
            Image.merge("RGB", (ImageChops.add(gradient, noise, scale=2), gradient, noise)).save(path, "JPEG")
            return str(path)

        def test_ken_burns_moves_and_stays_inside_the_frame(self):
            with tempfile.TemporaryDirectory() as temp:
                source = Image.open(self._photo(Path(temp) / "a.jpg", (512, 512)))
                for variant in range(4):
                    start = render_image_frame(90, 160, source, 0.0, variant)
                    end = render_image_frame(90, 160, source, 1.0, variant)
                    self.assertEqual((start.size, end.size), ((90, 160), (90, 160)))
                    self.assertNotEqual(start.tobytes(), end.tobytes())             # the camera really moves
                wide = Image.new("RGB", (900, 300), (90, 120, 200))
                self.assertEqual(render_image_frame(90, 160, wide, .5, 1).size, (90, 160))
                plain = render_image_frame(90, 160, source, .5, 0)
                dark = render_image_frame(90, 160, source, .5, 0, calm=True)
                from PIL import ImageStat
                self.assertLess(ImageStat.Stat(dark.convert("L")).mean[0], ImageStat.Stat(plain.convert("L")).mean[0])

        def test_image_scenes_render_and_a_broken_image_falls_back_to_the_procedural_scene(self):
            with tempfile.TemporaryDirectory() as temp:
                images = {"opening-1": {"path": self._photo(Path(temp) / "o.jpg")},
                          "scene-1": {"path": str(Path(temp) / "missing.jpg")}}
                line = line_for(images=images)
                got = list(frames(90, 160, 2, 12.0, line))
                self.assertEqual(len(got), 24)
                self.assertTrue(all(len(f) == 90 * 160 * 3 for f in got))
                out = Path(temp) / "bg.mp4"
                info = render_background(out, 180, 320, 4, 50.0, line, theme="t", render_scale=.5)
                self.assertEqual(info["image_scenes"], len(line))               # every shot carries a (possibly reused) path
                self.assertTrue(out.stat().st_size > 1024)

        # ----- script timeline: spoken parts and verses in any order -----
        @staticmethod
        def script_segments():
            def speech(start, end, count, role="body"):
                step = (end - start) / count
                return {"kind": "speech", "role": role, "start": start, "end": end,
                        "timings": [(i * step, i * step + step * .9, i) for i in range(count)]}
            return [speech(0.0, 5.0, 14, "hook"), speech(5.0, 7.0, 3), {"kind": "verse", "start": 7.0, "end": 15.0},
                    speech(15.0, 40.0, 50), {"kind": "verse", "start": 40.0, "end": 45.0}, speech(45.0, 50.0, 8, "closing")]

        @staticmethod
        def script_plan():
            plan = sample_plan()
            plan.pop("closing", None)
            plan["scenes"] = [shot("dawn", element("sun", "right"), start=0), shot("storm", element("cloud"), start=3),
                              shot("day", element("seed", "center", "grow"), start=13), shot("warm", element("heart"), start=23),
                              shot("dawn", element("path"), start=33), shot("night", element("moon", "right"), start=43),
                              shot("warm", element("hands"), start=53)]
            return plan

        def script_line(self, plan=None, **kwargs):
            return build_script_timeline(plan or self.script_plan(), self.script_segments(), 50.0, **kwargs)

        def test_script_timeline_is_gap_free_and_cuts_where_the_narrator_starts_a_new_phrase(self):
            line = self.script_line()
            self.assertEqual((line[0]["start"], line[-1]["end"]), (0.0, 50.0))
            for left, right in zip(line, line[1:]):
                self.assertAlmostEqual(left["end"], right["start"], places=3)
            starts = {round(e["start"], 1) for e in line}
            for expected in (5.0, 15.0, 20.0, 25.0, 30.0, 35.0, 45.0):                  # phrase starts bound to words
                self.assertIn(expected, starts)
            self.assertEqual(timeline_image_keys([e for e in line if e["name"].startswith("hook")]), ["opening-1", "opening-2"])

        def test_a_recited_verse_keeps_the_current_picture_under_a_veil_and_is_cut_into_crops(self):
            line = self.script_line()
            verse1 = [e for e in line if 7.0 <= e["start"] < 15.0]
            verse2 = [e for e in line if 40.0 <= e["start"] < 45.0]
            self.assertTrue(all(e["calm"] and e["name"].startswith("recitation") for e in verse1 + verse2))
            self.assertEqual({e["image_key"] for e in verse1}, {verse1[0]["image_key"]})
            self.assertEqual(verse1[0]["image_key"], "scene-1")                           # shot 1 is still active at 7 s
            self.assertEqual(verse2[0]["image_key"], "scene-6")
            self.assertGreaterEqual(len(verse1), 2)                                       # 8 s of recitation: several crops
            self.assertTrue(all(e["end"] - e["start"] <= RECITATION_SHOT_SECONDS * 1.5 + .01 for e in verse1))
            spoken = [e for e in line if not e["calm"] and e["start"] >= 5.0]
            self.assertTrue(all(e["end"] - e["start"] <= MAX_SHOT_SECONDS * SPLIT_THRESHOLD + .01 for e in spoken))

        def test_a_shot_cut_right_next_to_a_verse_edge_is_dropped_but_the_verse_edge_stays(self):
            plan = self.script_plan()
            plan["scenes"][2] = shot("day", element("seed"), start=4)                      # speech 2 word 1 = 15.5 s, 0.5 s after the verse
            line = self.script_line(plan)
            self.assertNotIn("scene-3", {e["image_key"] for e in line})
            self.assertIn(15.0, {e["start"] for e in line})
            self.assertTrue(all(e["end"] - e["start"] >= MIN_PIECE_SECONDS - 0.01 for e in line))

        def test_script_timeline_rejects_a_sequence_without_hook_or_verse(self):
            segments = self.script_segments()
            with self.assertRaises(ValueError):
                build_script_timeline(self.script_plan(), segments[1:], 50.0)
            with self.assertRaises(ValueError):
                build_script_timeline(self.script_plan(), [segments[0], segments[1], segments[3]], 50.0)
            with self.assertRaises(ValueError):
                build_script_timeline(self.script_plan(), [], 50.0)

        def test_script_timeline_images_pick_budget_and_reuse(self):
            line = self.script_line()
            keys = timeline_image_keys(line)
            self.assertEqual(keys[:2], ["opening-1", "opening-2"])
            self.assertEqual(len(pick_image_keys(keys, 5)), 5)
            apply_images(line, {"opening-1": {"path": "/x/o.jpg"}, "scene-1": {"path": "/x/s1.jpg"}})
            self.assertTrue(all(e["image"] for e in line))
            self.assertEqual([e["image"] for e in line if e["image_key"] == "scene-1"][0], "/x/s1.jpg")

        def test_script_timeline_renders_into_a_video(self):
            with tempfile.TemporaryDirectory() as temp:
                line = self.script_line(images={"scene-1": {"path": self._photo(Path(temp) / "s1.jpg")}})
                out = Path(temp) / "bg.mp4"
                info = render_background(out, 180, 320, 4, 50.0, line, theme="t", render_scale=.5)
                self.assertTrue(out.stat().st_size > 1024)
                self.assertEqual(len(info["scenes"]), len(line))

    unittest.main()
