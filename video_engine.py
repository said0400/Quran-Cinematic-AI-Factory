"""Vertical video engine. Quran text is rendered as supplied; no audio is sped up or cut.

Rendering model
---------------
* The background is one or several stock clips merged by `compose_background` (cover-crop each
  clip to the vertical frame, slow it down, concatenate), then processed by FFmpeg again in
  `build_video` (slow time-based zoom, light dark filter).
* All text is drawn here with Pillow/RAQM as RGBA frames and piped to FFmpeg as raw video, so the
  number of FFmpeg inputs does not grow with the number of words.
* The Quran verse sits on an ornamental card and is highlighted word by word using the source
  timestamps. Commentary texts (hook, message, call to action) stay fully visible on a plain card
  with the current spoken word highlighted; a long text is split into consecutive cards.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

from PIL import Image, ImageDraw, ImageFont, features


class VideoError(RuntimeError):
    """Video generation or validation failed."""


# Arabic is shaped and ordered by Pillow's RAQM layout engine (HarfBuzz + FriBiDi) directly from
# the logical (canonical) text. Pre-shaping with arabic-reshaper/python-bidi must NOT be combined
# with RAQM: the text would be processed twice, giving reversed order and unjoined letters.
TEXT_DIRECTION = "rtl"
TEXT_LANGUAGE = "ar"

WHITE = (255, 255, 255, 255)
GOLD = (212, 175, 55, 255)
GOLD_SOFT = (150, 122, 38, 255)
LABEL_COLOR = (255, 218, 74, 255)
VERSE_CARD_FILL = (9, 26, 34, 240)
PLAIN_CARD_FILL = (0, 0, 0, 158)
VERSE_HIGHLIGHT = (120, 240, 160, 255)   # light green: stands out from the gold frame
PLAIN_HIGHLIGHT = (255, 218, 74, 255)    # yellow
FADE_SECONDS = 0.20
HIGHLIGHT_LINGER_SECONDS = 0.40          # keep the last word lit through short pauses between words
FADE_STEPS = 8                           # alpha is quantised so identical frames can be reused
MAX_DIM_STRENGTH = 0.80
ORNATE_OPEN_BRACKET = "\ufd3f"
ORNATE_CLOSE_BRACKET = "\ufd3e"

MIN_BACKGROUND_CLIPS = 1
MAX_BACKGROUND_CLIPS = 5
DEFAULT_BACKGROUND_SPEED = 0.75          # 0.75 = 25% slower than the original footage


def run_checked(command: list[str], timeout: int = 900) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise VideoError(f"الأداة المطلوبة غير مثبتة: {command[0]}") from exc
    except subprocess.TimeoutExpired:
        raise VideoError(f"انتهت مهلة الأمر {command[0]}.") from None
    if result.returncode != 0:
        stderr = (result.stderr or "").strip().splitlines()
        detail = stderr[-1][-300:] if stderr else "لا توجد تفاصيل"
        raise VideoError(f"فشل الأمر {command[0]} برمز {result.returncode}: {detail}")
    return result


def probe(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size < 1024:
        raise VideoError(f"الملف غير موجود أو صغير جدًا: {path.name}")
    result = run_checked(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], timeout=60)
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise VideoError("تعذر تحليل بيانات ffprobe.") from None
    return data


def media_duration(path: Path) -> float:
    result = run_checked(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)], timeout=30)
    try:
        duration = float(result.stdout.strip())
    except (TypeError, ValueError):
        raise VideoError(f"تعذر قراءة مدة {path.name}.") from None
    if not math.isfinite(duration) or duration <= 0:
        raise VideoError(f"مدة صوت غير صالحة: {path.name}.")
    return duration


def discover_font() -> str:
    if not features.check("raqm"):
        raise VideoError("بناء Pillow الحالي لا يدعم RAQM؛ لا يمكن ضمان ترتيب وتشكيل العربية بأمان.")
    custom_font = os.getenv("ARABIC_FONT", "").strip()
    if custom_font:
        if not Path(custom_font).is_file():
            raise VideoError("مسار ARABIC_FONT المخصص غير موجود.")
        license_url = os.getenv("ARABIC_FONT_LICENSE_URL", "").strip()
        if os.getenv("ARABIC_FONT_LICENSE_CONFIRMED", "").strip().lower() not in {"1", "true", "yes", "on"} or urlparse(license_url).scheme != "https":
            raise VideoError("الخط المخصص يتطلب ARABIC_FONT_LICENSE_CONFIRMED=true ورابط ترخيص HTTPS.")
        return custom_font
    candidates = [
        "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoNaskhArabicUI-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoNaskhArabic-Regular.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    fc_match = shutil.which("fc-match")
    if fc_match:
        result = subprocess.run([fc_match, "-f", "%{file}", "Noto Naskh Arabic"], capture_output=True, text=True, check=False)
        if result.returncode == 0 and Path(result.stdout).is_file() and "/noto/" in result.stdout.lower():
            return result.stdout
    raise VideoError("لم يُعثر على خط عربي قابل للقراءة. ثبّت Noto Naskh Arabic أو اضبط ARABIC_FONT.")


def _font_license_metadata(font_path: str) -> dict[str, str]:
    custom_font = os.getenv("ARABIC_FONT", "").strip()
    if custom_font and Path(custom_font).resolve() == Path(font_path).resolve():
        license_url = os.getenv("ARABIC_FONT_LICENSE_URL", "").strip()
        if os.getenv("ARABIC_FONT_LICENSE_CONFIRMED", "").strip().lower() not in {"1", "true", "yes", "on"} or urlparse(license_url).scheme != "https":
            raise VideoError("لا يمكن تسجيل الخط المخصص دون ترخيص HTTPS مؤكد.")
        return {"name": "Operator-supplied custom Arabic font", "license": "Operator-confirmed", "license_url": license_url}
    return {"name": "Noto Arabic font package", "license": "SIL Open Font License 1.1", "license_url": "https://scripts.sil.org/OFL"}


def _visual(text: str) -> str:
    # Deliberately the identity: RAQM receives the canonical logical text and performs shaping
    # and bidirectional ordering itself (see TEXT_DIRECTION above). Kept as a single choke point
    # so no caller can accidentally reintroduce a second shaping pass.
    return text


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> float:
    """Advance width of Arabic text as laid out by RAQM (includes side bearings, so it matches draw.text)."""
    return draw.textlength(_visual(text), font=font, direction=TEXT_DIRECTION, language=TEXT_LANGUAGE)


def _text_center_offset(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> float:
    """Vertical distance from the drawing origin to the optical middle of the text."""
    box = draw.textbbox((0, 0), _visual(text), font=font, direction=TEXT_DIRECTION, language=TEXT_LANGUAGE)
    return (box[1] + box[3]) / 2


def _draw_text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font: ImageFont.FreeTypeFont,
               fill: tuple[int, int, int, int], stroke: bool = True, stroke_width: int = 1) -> None:
    kwargs: dict[str, Any] = {"font": font, "fill": fill, "direction": TEXT_DIRECTION, "language": TEXT_LANGUAGE}
    if stroke:
        kwargs.update(stroke_width=stroke_width, stroke_fill=(0, 0, 0, 255))
    draw.text(xy, _visual(text), **kwargs)

def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size=size, layout_engine=ImageFont.Layout.RAQM)
    except (OSError, AttributeError):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError as exc:
            raise VideoError("تعذر تحميل الخط العربي المحدد.") from exc


def _arabic_indic(number: int) -> str:
    return str(int(number)).translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))


def _wrap_words(words: list[str], font: ImageFont.FreeTypeFont, max_width: int, draw: ImageDraw.ImageDraw,
                gap: int) -> list[list[int]]:
    lines: list[list[int]] = []
    current: list[int] = []
    width = 0.0
    for index, word in enumerate(words):
        word_width = _text_width(draw, word, font)
        if current and width + gap + word_width > max_width:
            lines.append(current)
            current, width = [], 0.0
        current.append(index)
        width += word_width + (gap if len(current) > 1 else 0)
    if current:
        lines.append(current)
    return lines


def _star_points(cx: float, cy: float, outer: float, inner: float, points: int = 8,
                 rotation: float = -math.pi / 2) -> list[tuple[float, float]]:
    result = []
    for k in range(points * 2):
        angle = rotation + k * math.pi / points
        radius = outer if k % 2 == 0 else inner
        result.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return result


def quran_words(text: str) -> list[str]:
    """Split Quran text into display/timing words.

    Whitespace-separated tokens without any letter (waqf signs, the rosette, the verse-end sign)
    are not words and have no timing: they are attached to the preceding word (or to the next one
    at the very start), so word counts match the source timestamps. Kept identical to
    `services.quran_words` (a unit test in main.py compares them).
    """
    words: list[str] = []
    pending: list[str] = []
    for token in (text or "").split():
        if not any(unicodedata.category(char).startswith("L") for char in token):
            if words:
                words[-1] += " " + token
            else:
                pending.append(token)
            continue
        if pending:
            token = " ".join([*pending, token])
            pending = []
        words.append(token)
    return words


SENTENCE_END_CHARS = "،,.؟?!؛:"
PAGE_MAX_WORDS = 24      # a card holds at most 7 lines; 24 words fit even at the smallest font size
PAGE_MIN_WORDS = 10


def _best_cut(words: list[str], low: int, high: int) -> int:
    """Exclusive end index in [low, high] that follows sentence punctuation, else `high`."""
    for index in range(high, low - 1, -1):
        if words[index - 1][-1:] in SENTENCE_END_CHARS:
            return index
    return high


def paginate_text(text: str, max_words: int = PAGE_MAX_WORDS, min_words: int = PAGE_MIN_WORDS) -> list[list[str]]:
    """Split commentary into card-sized pages, preferring to break after sentence punctuation."""
    words = text.split()
    if not words:
        raise VideoError("نص المشهد فارغ.")
    pages: list[list[str]] = []
    start = 0
    while len(words) - start > max_words:
        end = _best_cut(words, start + min_words, start + max_words)
        pages.append(words[start:end])
        start = end
    pages.append(words[start:])
    if len(pages) > 1 and len(pages[-1]) < min_words // 2:
        # Never leave a tiny last card: share the last two pages evenly.
        combined = pages[-2] + pages[-1]
        middle = len(combined) // 2
        cut = _best_cut(combined, max(1, middle - 3), min(len(combined) - 1, middle + 3))
        pages[-2:] = [combined[:cut], combined[cut:]]
    return pages


# --------------------------------------------------------------------------------------------
# Layouts: static card + word positions. A frame is the card plus the words with one highlighted.
# --------------------------------------------------------------------------------------------
@dataclass
class TextLayout:
    base: Image.Image                           # full-size RGBA: card, ornaments and labels, no words
    font: ImageFont.FreeTypeFont
    placements: list[tuple[float, float, str]]  # (x, y, word); list index == word index
    highlight: tuple[int, int, int, int]
    word_count: int
    stroke: int = 1                             # outline width around every word (captions use a thicker one)


def draw_layout(layout: TextLayout, current_word: int | None = None) -> Image.Image:
    """Return the card with every word visible and `current_word` (if any) highlighted."""
    if current_word is not None and not 0 <= current_word < layout.word_count:
        raise VideoError("نص الآية أو فهرس الكلمة المتلوة غير صالح.")
    image = layout.base.copy()
    draw = ImageDraw.Draw(image)
    for index, (x, y, word) in enumerate(layout.placements):
        color = layout.highlight if index == current_word else WHITE
        _draw_text(draw, (x, y), word, layout.font, color, stroke_width=layout.stroke)
    return image

def _place_lines(words: list[str], lines: list[list[int]], font: ImageFont.FreeTypeFont, gap: int,
                 draw: ImageDraw.ImageDraw, width: int, start_y: float,
                 line_height: float) -> list[tuple[float, float, str]]:
    positions: dict[int, tuple[float, float, str]] = {}
    for row, indexes in enumerate(lines):
        widths = [_text_width(draw, words[index], font) for index in indexes]
        total = sum(widths) + gap * max(0, len(widths) - 1)
        cursor = (width + total) / 2
        y = start_y + row * line_height
        # Logical Arabic order starts at the right; each word is drawn as an independent unit.
        for index, word_width in zip(indexes, widths):
            cursor -= word_width
            positions[index] = (cursor, y, words[index])
            cursor -= gap
    return [positions[index] for index in range(len(words))]


def build_text_layout(text: str, label: str = "", width: int = 1080, height: int = 1920,
                      font_path: str | None = None) -> TextLayout:
    """Plain dark card for commentary text; the spoken word is highlighted in yellow."""
    if not isinstance(text, str) or not text.strip() or "�" in text:
        raise VideoError("نص المشهد فارغ أو يحوي رمزًا تالفًا.")
    font_path = font_path or discover_font()
    s = width / 1080.0
    probe_draw = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    words = text.split()
    chosen = None
    for size in range(74, 41, -2):
        pixel_size = max(8, int(round(size * s)))
        font = _font(font_path, pixel_size)
        gap = max(4, pixel_size // 3)
        lines = _wrap_words(words, font, int(width * 0.84), probe_draw, gap)
        if len(lines) <= 7:
            chosen = (font, lines, pixel_size, gap)
            break
    if chosen is None:
        raise VideoError("النص أطول من مساحة الشاشة؛ رُفض بدل إسقاط كلمات.")
    font, lines, pixel_size, gap = chosen
    line_height = int(pixel_size * 1.55)
    label_block = int(75 * s) if label else 0
    block_height = len(lines) * line_height + label_block
    top = int(height * 0.48 - block_height / 2)
    side = int(width * 0.06)
    base = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(base)
    draw.rounded_rectangle((side, top - int(46 * s), width - side, top + block_height + int(50 * s)),
                           radius=max(4, int(34 * s)), fill=PLAIN_CARD_FILL)
    if label:
        label_font = _font(font_path, max(8, int(42 * s)))
        label_width = _text_width(draw, label, label_font)
        _draw_text(draw, ((width - label_width) / 2, top + int(8 * s)), label, label_font, LABEL_COLOR, stroke=False)
    start_y = top + (int(78 * s) if label else int(20 * s))
    placements = _place_lines(words, lines, font, gap, draw, width, start_y, line_height)
    return TextLayout(base, font, placements, PLAIN_HIGHLIGHT, len(words))


def build_verse_layout(text: str, chapter_name: str = "", verse_number: int | None = None,
                       width: int = 1080, height: int = 1920, font_path: str | None = None) -> TextLayout:
    """Ornamental Quran card: gold frame, corner stars, surah banner, verse-number medallion, ornate brackets."""
    if not isinstance(text, str) or not text.strip() or "�" in text:
        raise VideoError("نص الآية فارغ أو يحوي رمزًا تالفًا.")
    font_path = font_path or discover_font()
    s = width / 1080.0
    probe_draw = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    words = quran_words(text)
    chosen = None
    for size in range(66, 39, -2):
        pixel_size = max(8, int(round(size * s)))
        font = _font(font_path, pixel_size)
        gap = max(4, pixel_size // 3)
        lines = _wrap_words(words, font, int(width * 0.70), probe_draw, gap)
        if len(lines) <= 8:
            chosen = (font, lines, pixel_size, gap)
            break
    if chosen is None:
        raise VideoError("تعذر عرض الآية كاملة بخط مقروء ضمن مساحة الشاشة.")
    font, lines, pixel_size, gap = chosen
    line_height = int(pixel_size * 1.70)
    block_height = len(lines) * line_height
    pad_top, pad_bottom = int(78 * s), int(82 * s)
    card_height = pad_top + block_height + pad_bottom
    card_left, card_right = int(width * 0.05), int(width * 0.95)
    card_top = int(height * 0.40 - card_height / 2)
    card_bottom = card_top + card_height
    line_w = max(2, int(5 * s))
    thin_w = max(1, int(2 * s))

    base = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(base)
    draw.rounded_rectangle((card_left, card_top, card_right, card_bottom), radius=max(6, int(40 * s)),
                           fill=VERSE_CARD_FILL, outline=GOLD, width=line_w)
    inset = int(18 * s)
    inner = (card_left + inset, card_top + inset, card_right - inset, card_bottom - inset)
    draw.rounded_rectangle(inner, radius=max(4, int(28 * s)), outline=GOLD_SOFT, width=thin_w)
    # Corner stars sit on the corners of the inner frame.
    for cx, cy in ((inner[0], inner[1]), (inner[2], inner[1]), (inner[0], inner[3]), (inner[2], inner[3])):
        draw.polygon(_star_points(cx, cy, 22 * s, 9 * s), fill=GOLD, outline=VERSE_CARD_FILL)
        draw.ellipse((cx - 4 * s, cy - 4 * s, cx + 4 * s, cy + 4 * s), fill=VERSE_CARD_FILL)

    center_x = (card_left + card_right) / 2
    if chapter_name.strip():
        banner_font = _font(font_path, max(8, int(40 * s)))
        banner_text = f"سورة {chapter_name.strip()}"
        banner_text_width = _text_width(draw, banner_text, banner_font)
        half_w, half_h = banner_text_width / 2 + 52 * s, 32 * s
        cy = card_top
        banner = [(center_x - half_w, cy), (center_x - half_w + half_h, cy - half_h),
                  (center_x + half_w - half_h, cy - half_h), (center_x + half_w, cy),
                  (center_x + half_w - half_h, cy + half_h), (center_x - half_w + half_h, cy + half_h)]
        draw.polygon(banner, fill=VERSE_CARD_FILL, outline=GOLD, width=line_w)
        offset = _text_center_offset(draw, banner_text, banner_font)
        _draw_text(draw, (center_x - banner_text_width / 2, cy - offset), banner_text, banner_font, GOLD, stroke=False)
    if verse_number is not None:
        number_font = _font(font_path, max(8, int(36 * s)))
        number_text = _arabic_indic(verse_number)
        number_width = _text_width(draw, number_text, number_font)
        radius = max(40 * s, number_width / 2 + 16 * s)
        draw.polygon(_star_points(center_x, card_bottom, radius, radius * 0.84, points=12),
                     fill=VERSE_CARD_FILL, outline=GOLD, width=line_w)
        offset = _text_center_offset(draw, number_text, number_font)
        _draw_text(draw, (center_x - number_width / 2, card_bottom - offset), number_text, number_font, GOLD, stroke=False)

    # Ornate brackets flank the verse, vertically centred on the text block.
    bracket_font = _font(font_path, max(10, int(150 * s)))
    block_top = card_top + pad_top
    middle = block_top + block_height / 2
    bracket_offset = _text_center_offset(draw, ORNATE_OPEN_BRACKET, bracket_font)
    open_width = _text_width(draw, ORNATE_OPEN_BRACKET, bracket_font)
    _draw_text(draw, (card_right - inset - 14 * s - open_width, middle - bracket_offset),
               ORNATE_OPEN_BRACKET, bracket_font, GOLD, stroke=False)
    _draw_text(draw, (card_left + inset + 14 * s, middle - bracket_offset),
               ORNATE_CLOSE_BRACKET, bracket_font, GOLD, stroke=False)

    placements = _place_lines(words, lines, font, gap, draw, width, block_top, line_height)
    return TextLayout(base, font, placements, VERSE_HIGHLIGHT, len(words))


def build_dim_layer(width: int, height: int, strength: float) -> Image.Image:
    """Light black filter for the whole video: slightly stronger around the text band than at the edges."""
    if not isinstance(strength, (int, float)) or isinstance(strength, bool) or not 0 <= strength <= MAX_DIM_STRENGTH:
        raise VideoError(f"شدة الفلتر الداكن يجب أن تكون بين 0 و{MAX_DIM_STRENGTH}.")
    column = Image.new("L", (1, height))
    values = []
    for y in range(height):
        bump = math.exp(-(((y / height) - 0.42) / 0.22) ** 2)
        values.append(int(round(255 * min(0.95, strength * (0.80 + 0.45 * bump)))))
    column.putdata(values)
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    layer.putalpha(column.resize((width, height)))
    return layer


# --------------------------------------------------------------------------------------------
# Word timing for spoken commentary
# --------------------------------------------------------------------------------------------
def _word_weight(word: str) -> float:
    letters = sum(1 for ch in word if unicodedata.category(ch).startswith("L"))
    weight = float(max(1, letters))
    if word and word[-1] in "،,.؟?!؛:":
        weight += 1.5  # speakers pause after punctuation
    return weight


def estimate_word_timings(words: list[str], duration: float, lead: float = 0.08,
                          tail: float = 0.12) -> list[tuple[float, float, int]]:
    """Distribute `duration` over the words by length. An ESTIMATE, used only for commentary speech."""
    if not words or duration <= 0:
        raise VideoError("لا يمكن تقدير توقيت الكلمات بلا نص أو مدة.")
    span = max(0.1, duration - lead - tail)
    weights = [_word_weight(word) for word in words]
    total = sum(weights)
    result = []
    cursor = lead
    for index, weight in enumerate(weights):
        length = span * weight / total
        result.append((cursor, cursor + length, index))
        cursor += length
    return result


def align_speech_words(text: str, boundaries: list[dict[str, Any]] | None,
                       duration: float) -> tuple[list[tuple[float, float, int]], str]:
    """Return ([(start_s, end_s, word_index)], source) for spoken commentary.

    Real TTS word boundaries are used only when there is exactly one boundary per word and the
    timings are ordered and inside the audio; otherwise the timings are an estimate. The source is
    reported ("tts_word_boundary" or "estimated") so the production report stays honest.
    """
    words = text.split()
    if not words:
        raise VideoError("نص الكلام المنطوق فارغ.")
    if boundaries and len(boundaries) == len(words):
        rows: list[tuple[float, float, int]] = []
        valid = True
        previous_start = -1.0
        for index, item in enumerate(boundaries):
            try:
                start = float(item["start_ms"]) / 1000.0
                end = float(item["end_ms"]) / 1000.0
            except (KeyError, TypeError, ValueError):
                valid = False
                break
            if start < 0 or end <= start or start < previous_start or start > duration + 0.35:
                valid = False
                break
            previous_start = start
            rows.append((start, min(end, duration), index))
        if valid and rows:
            return rows, "tts_word_boundary"
    return estimate_word_timings(words, duration), "estimated"


# --------------------------------------------------------------------------------------------
# Timeline and frames
# --------------------------------------------------------------------------------------------
@dataclass
class Scene:
    name: str
    start: float
    end: float
    layout: TextLayout
    words: list[tuple[float, float, int]]   # absolute (start_s, end_s, word_index), sorted by start
    fade_in: float = FADE_SECONDS
    fade_out: float = FADE_SECONDS


def scene_word_at(scene: Scene, t: float) -> int | None:
    current: tuple[float, int] | None = None
    for start, end, index in scene.words:
        if start <= t:
            current = (end, index)
        else:
            break
    if current is not None and t <= current[0] + HIGHLIGHT_LINGER_SECONDS:
        return current[1]
    return None


def scene_alpha(scene: Scene, t: float) -> float:
    value = 1.0
    if scene.fade_in > 0:
        value = min(value, (t - scene.start) / scene.fade_in)
    if scene.fade_out > 0:
        value = min(value, (scene.end - t) / scene.fade_out)
    value = max(0.0, min(1.0, value))
    return round(value * FADE_STEPS) / FADE_STEPS


def _with_alpha(image: Image.Image, alpha: float) -> Image.Image:
    result = image.copy()
    result.putalpha(image.getchannel("A").point(lambda value, k=alpha: int(value * k)))
    return result


def render_frames(scenes: list[Scene], total_frames: int, fps: int, width: int, height: int) -> Iterator[bytes]:
    """Yield one raw RGBA frame per output frame; identical consecutive states reuse the same bytes."""
    blank = bytes(width * height * 4)
    last_key: tuple[Any, ...] | None = None
    last_bytes = blank
    cached_state: tuple[int, int | None] | None = None
    cached_image: Image.Image | None = None
    for n in range(total_frames):
        t = n / fps
        scene = next((item for item in scenes if item.start <= t < item.end), None)
        if scene is None:
            key: tuple[Any, ...] | None = None
        else:
            alpha = scene_alpha(scene, t)
            key = None if alpha <= 0 else (id(scene), scene_word_at(scene, t), alpha)
        if key != last_key:
            if key is None or scene is None:
                last_bytes = blank
            else:
                state = (key[0], key[1])
                if state != cached_state or cached_image is None:
                    cached_image = draw_layout(scene.layout, key[1])
                    cached_state = state
                image = cached_image if key[2] >= 1.0 else _with_alpha(cached_image, key[2])
                last_bytes = image.tobytes()
            last_key = key
        yield last_bytes


def _encode_with_overlay_pipe(command: list[str], frames: Iterator[bytes], timeout: int) -> None:
    """Run FFmpeg, feeding the overlay frames to its stdin. Any failure becomes a VideoError."""
    with tempfile.TemporaryFile() as stderr_file:
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr_file)
        except FileNotFoundError:
            raise VideoError("الأداة المطلوبة غير مثبتة: ffmpeg") from None
        watchdog = threading.Timer(timeout, process.kill)
        watchdog.start()
        pipe_closed_early = False
        try:
            assert process.stdin is not None
            for chunk in frames:
                process.stdin.write(chunk)
        except (BrokenPipeError, OSError):
            pipe_closed_early = True
        except BaseException:
            process.kill()
            raise
        finally:
            try:
                if process.stdin is not None:
                    process.stdin.close()
            except OSError:
                pass
            return_code = process.wait()
            watchdog.cancel()
        if return_code != 0 or pipe_closed_early:
            stderr_file.seek(0)
            lines = stderr_file.read().decode("utf-8", errors="replace").strip().splitlines()
            detail = lines[-1][-300:] if lines else "لا توجد تفاصيل"
            reason = "انتهت المهلة" if return_code == -9 else f"برمز {return_code}"
            raise VideoError(f"فشل ffmpeg {reason}: {detail}")


def _audio_concat(audio_files: list[Path], destination: Path) -> float:
    if len(audio_files) != 5:
        raise VideoError("يجب تمرير خمسة مقاطع صوتية منفصلة: هوك، تلاوة، العبارة، رسالة، دعوة المشاركة.")
    for file in audio_files:
        if not file.is_file() or file.stat().st_size < 512:
            raise VideoError(f"مقطع صوت غير صالح: {file.name}")
    inputs = []
    filters = []
    for idx, file in enumerate(audio_files):
        inputs.extend(["-i", str(file)])
        filters.append(f"[{idx}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a{idx}]")
    concat_inputs = "".join(f"[a{idx}]" for idx in range(len(audio_files)))
    filters.append(f"{concat_inputs}concat=n={len(audio_files)}:v=0:a=1[aout]")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs,
               "-filter_complex", ";".join(filters), "-map", "[aout]", "-c:a", "pcm_s16le", str(destination)]
    run_checked(command, timeout=900)
    if not destination.is_file():
        raise VideoError("فشل إنشاء مسار الصوت المركب.")
    return media_duration(destination)


def _probe_audio_seconds(path: Path) -> float:
    return media_duration(path)


# --------------------------------------------------------------------------------------------
# Multi-clip background
# --------------------------------------------------------------------------------------------
def compose_background(clips: list[Path], destination: Path, width: int = 1080, height: int = 1920,
                       fps: int = 30, speed: float = DEFAULT_BACKGROUND_SPEED,
                       target_duration: float | None = None,
                       source_seconds_each: float = 20.0) -> dict[str, Any]:
    """Merge 1..5 stock clips into one vertical background slowed to `speed` (0.75 = 25% slower).

    Every clip is cover-cropped to width x height, slowed by stretching its timestamps (frames are
    repeated by the fps filter, never interpolated) and concatenated in the given order. Audio of the
    stock footage is dropped. When `target_duration` (seconds of OUTPUT) is given, each clip
    contributes an equal share of it; otherwise at most `source_seconds_each` seconds of every source
    clip are used. If the result is shorter than the target, `build_video` loops it, which is
    reported through `will_loop`.
    """
    if not isinstance(clips, list) or not clips:
        raise VideoError("لا توجد مقاطع خلفية للدمج.")
    if len(clips) > MAX_BACKGROUND_CLIPS:
        raise VideoError(f"عدد مقاطع الخلفية يجب ألا يتجاوز {MAX_BACKGROUND_CLIPS}.")
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not 0.25 <= speed <= 1.0:
        raise VideoError("سرعة الخلفية يجب أن تكون بين 0.25 و1.0.")
    if width <= 0 or height <= 0 or width % 2 or height % 2 or fps <= 0:
        raise VideoError("أبعاد أو معدل إطارات الخلفية غير صالح.")
    if target_duration is not None:
        if target_duration <= 0:
            raise VideoError("المدة المطلوبة للخلفية يجب أن تكون موجبة.")
        # Output seconds = source seconds / speed, so each clip needs target*speed/n source seconds.
        source_seconds_each = math.ceil(target_duration * speed / len(clips)) + 1
    if source_seconds_each <= 0:
        raise VideoError("المدة المأخوذة من كل مقطع يجب أن تكون موجبة.")
    for clip in clips:
        data = probe(Path(clip))
        if not any(s.get("codec_type") == "video" for s in data.get("streams", [])):
            raise VideoError(f"الملف لا يحتوي مسار فيديو: {Path(clip).name}")
    inputs: list[str] = []
    filters: list[str] = []
    for index, clip in enumerate(clips):
        inputs.extend(["-i", str(clip)])
        filters.append(
            f"[{index}:v]trim=duration={source_seconds_each:.3f},setpts=PTS-STARTPTS,"
            f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1,"
            f"setpts=PTS/{speed:.6f},fps={fps},format=yuv420p[v{index}]"
        )
    filters.append("".join(f"[v{index}]" for index in range(len(clips))) + f"concat=n={len(clips)}:v=1:a=0[vout]")
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_checked(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs,
                 "-filter_complex", ";".join(filters), "-map", "[vout]", "-an",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                 str(destination)], timeout=1200)
    if not destination.is_file():
        raise VideoError("لم ينتج FFmpeg ملف الخلفية المدموج.")
    duration = media_duration(destination)
    return {
        "clips": len(clips),
        "speed": float(speed),
        "source_seconds_per_clip": float(source_seconds_each),
        "duration_seconds": round(duration, 3),
        "target_duration_seconds": None if target_duration is None else round(float(target_duration), 3),
        "will_loop": bool(target_duration is not None and duration + 0.25 < float(target_duration)),
    }


def commentary_scenes(name: str, label: str, text: str, start: float, end: float,
                      timings: list[tuple[float, float, int]], width: int, height: int, font_path: str,
                      first_fade_in: float = FADE_SECONDS, last_fade_out: float = FADE_SECONDS) -> list[Scene]:
    """Scenes for one spoken text: one card, or consecutive cards when the text is long.

    `timings` are absolute (start_s, end_s, word_index) rows over `text.split()`. Each card starts
    when its first word is spoken and lasts until the next card begins, so the screen is never blank
    between cards.
    """
    pages = paginate_text(text)
    scenes: list[Scene] = []
    first = 0
    for number, page in enumerate(pages):
        last_page = number == len(pages) - 1
        following = first + len(page)
        page_start = start if number == 0 else timings[first][0]
        page_end = end if last_page else timings[following][0]
        page_end = max(page_end, page_start + 0.05)
        page_timings = [(a, b, index - first) for a, b, index in timings if first <= index < following]
        scenes.append(Scene(name if len(pages) == 1 else f"{name}-{number + 1}", page_start, page_end,
                            build_text_layout(" ".join(page), label, width, height, font_path), page_timings,
                            fade_in=first_fade_in if number == 0 else 0.12,
                            fade_out=last_fade_out if last_page else 0.0))
        first = following
    return scenes


def build_video(background: Path, audio_files: dict[str, Path], recitation_segments: list[dict[str, int]],
                verse_text: str, content: dict[str, Any], output_path: Path,
                width: int = 1080, height: int = 1920, fps: int = 30,
                target_min_duration: float = 45, target_max_duration: float = 60,
                tts_word_boundaries: dict[str, list[dict[str, Any]]] | None = None,
                chapter_name: str = "", verse_number: int | None = None,
                dim_strength: float = 0.30) -> dict[str, Any]:
    """Build exactly ordered scenes without cropping or speeding up the recitation."""
    required = ["hook", "message", "call_to_action"]
    if not all(isinstance(content.get(key), str) and content[key].strip() for key in required):
        raise VideoError("نصوص الفيديو المطلوبة ناقصة.")
    if "صدق الله العظيم" in " ".join(content[key] for key in required):
        raise VideoError("عبارة «صدق الله العظيم» يجب أن تكون مقطعًا مستقلًا بعد التلاوة فقط.")
    keys = ["hook", "recitation", "seal", "message", "cta"]
    if any(key not in audio_files for key in keys):
        raise VideoError("مقاطع الصوت الخمسة المطلوبة غير مكتملة.")
    if not recitation_segments:
        raise VideoError("توقيتات الكلمات الحقيقية غير متاحة.")
    dim_layer = build_dim_layer(width, height, float(dim_strength))  # validates the strength early
    bg_probe = probe(background)
    if not any(s.get("codec_type") == "video" for s in bg_probe.get("streams", [])):
        raise VideoError("ملف الخلفية لا يحتوي مسار فيديو.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work = output_path.parent / ".video_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    font_path = discover_font()
    combined_audio = work / "combined.wav"
    audio_order = [audio_files[key] for key in keys]
    total_audio = _audio_concat(audio_order, combined_audio)
    durations = {key: _probe_audio_seconds(audio_files[key]) for key in keys}
    # Derive the recitation duration from the exact source file; timing metadata is only accepted
    # after services.validate_word_segments checked word count/order/bounds.
    rec_duration = durations["recitation"]
    cursor = 0.0
    stage_intervals: dict[str, tuple[float, float]] = {}
    for key in keys:
        d = durations[key]
        stage_intervals[key] = (cursor, cursor + d)
        cursor += d
    if cursor > total_audio + 0.35:
        raise VideoError("مدة المقاطع لا تتفق مع الصوت المركب.")
    # Pad only the tail, never the recitation, to reach a 45s target; long recitations remain whole.
    video_duration = max(total_audio, float(target_min_duration))
    if total_audio > float(target_max_duration):
        # Longer output is permitted only because the complete recitation must not be truncated.
        video_duration = total_audio

    boundaries = tts_word_boundaries or {}
    timing_sources: dict[str, str] = {}
    scenes: list[Scene] = []

    commentary_pages: dict[str, int] = {}

    # Hook: the first scene appears immediately (no fade-in) so the opening frame is never empty.
    hook_start, hook_end = stage_intervals["hook"]
    hook_timings, timing_sources["hook"] = align_speech_words(content["hook"], boundaries.get("hook"), durations["hook"])
    hook_scenes = commentary_scenes("hook", "", content["hook"], hook_start, hook_end,
                                    [(hook_start + a, hook_start + b, i) for a, b, i in hook_timings],
                                    width, height, font_path, first_fade_in=0.0)
    commentary_pages["hook"] = len(hook_scenes)
    scenes.extend(hook_scenes)

    # Quran verse: ornamental card, words highlighted with the real source timestamps only.
    rec_start, rec_end = stage_intervals["recitation"]
    verse_words = quran_words(verse_text)
    recitation_timings: list[tuple[float, float, int]] = []
    for segment in recitation_segments:
        word_index = int(segment["word_index"])
        if not 0 <= word_index < len(verse_words):
            raise VideoError("فهرس توقيت كلمة خارج نص الآية.")
        start = rec_start + int(segment["start_ms"]) / 1000.0
        end = rec_start + int(segment["end_ms"]) / 1000.0
        if start < rec_start or end > rec_end + 0.3 or end <= start:
            raise VideoError("توقيت الكلمة لا يقع داخل مقطع التلاوة.")
        recitation_timings.append((start, end, word_index))
    recitation_timings.sort(key=lambda row: row[0])
    scenes.append(Scene("recitation", rec_start, rec_end,
                        build_verse_layout(verse_text, chapter_name, verse_number, width, height, font_path),
                        recitation_timings))

    # The closing phrase is short and always fully visible, without a moving highlight.
    seal_start, seal_end = stage_intervals["seal"]
    scenes.append(Scene("seal", seal_start, seal_end,
                        build_text_layout("صدق الله العظيم", "", width, height, font_path), []))

    for key, label, text_key in (("message", "رسالة لك", "message"),
                                 ("cta", "صدقة جارية", "call_to_action")):
        start, end = stage_intervals[key]
        timings, timing_sources[key] = align_speech_words(content[text_key], boundaries.get(key), durations[key])
        last_fade_out = FADE_SECONDS
        if key == "cta":
            if video_duration > total_audio:
                # Keep the final call-to-action on screen over any deliberate minimum-length padding.
                end = video_duration
            last_fade_out = 0.0
        key_scenes = commentary_scenes(key, label, content[text_key], start, end,
                                       [(start + a, start + b, i) for a, b, i in timings],
                                       width, height, font_path, last_fade_out=last_fade_out)
        commentary_pages[key] = len(key_scenes)
        scenes.extend(key_scenes)

    total_frames = int(math.ceil(video_duration * fps))
    zoom = f"(1+0.08*min(t/{max(video_duration, 1.0):.3f},1))"
    dim_path = work / "dim.png"
    use_dim = float(dim_strength) > 0
    if use_dim:
        dim_layer.save(dim_path, "PNG")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-stream_loop", "-1", "-i", str(background), "-i", str(combined_audio),
               "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{width}x{height}", "-framerate", str(fps), "-i", "pipe:0"]
    if use_dim:
        command.extend(["-loop", "1", "-framerate", str(fps), "-i", str(dim_path)])
    filters = [
        f"[0:v]scale=w='trunc({width}*{zoom}/2)*2':h='trunc({height}*{zoom}/2)*2':eval=frame:"
        f"force_original_aspect_ratio=increase,crop={width}:{height},setsar=1,"
        "eq=contrast=1.03:saturation=1.06,format=yuv420p[bg]"
    ]
    current = "bg"
    if use_dim:
        filters.append(f"[{current}][3:v]overlay=0:0[dimmed]")
        current = "dimmed"
    filters.append(f"[{current}][2:v]overlay=0:0:eof_action=pass,format=yuv420p[vout]")
    filters.append(f"[1:a]apad=whole_dur={video_duration:.6f},atrim=duration={video_duration:.6f},asetpts=PTS-STARTPTS[aout]")
    command.extend(["-filter_complex", ";".join(filters), "-map", "[vout]", "-map", "[aout]",
                    "-t", f"{video_duration:.6f}", "-r", str(fps), "-c:v", "libx264", "-preset", "medium",
                    "-crf", "19", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                    "-ar", "48000", "-movflags", "+faststart", str(output_path)])
    _encode_with_overlay_pipe(command, render_frames(scenes, total_frames, fps, width, height), timeout=3000)
    if not output_path.is_file():
        raise VideoError("لم ينتج FFmpeg ملف الفيديو النهائي.")
    validation = validate_video(output_path, width, height, fps, expected_duration=video_duration)
    validation.update({"audio_segments": {key: {"start": round(a, 3), "end": round(b, 3)} for key, (a, b) in stage_intervals.items()},
                       "recitation_duration_seconds": round(rec_duration, 3),
                       "target_duration_met": video_duration >= target_min_duration,
                       "padding_seconds": round(max(0, video_duration - total_audio), 3),
                       "dim_strength": float(dim_strength),
                       "tts_word_timing": dict(timing_sources),
                       "commentary_pages": commentary_pages,
                       "font_source": {"path": font_path, **_font_license_metadata(font_path)}})
    return validation


def validate_video(path: Path, width: int = 1080, height: int = 1920, fps: int = 30,
                   expected_duration: float | None = None) -> dict[str, Any]:
    data = probe(path)
    streams = data.get("streams", [])
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
    if not video or not audio:
        raise VideoError("الفيديو النهائي يجب أن يحتوي مساري صورة وصوت.")
    if video.get("codec_name") != "h264" or int(video.get("width", 0)) != width or int(video.get("height", 0)) != height:
        raise VideoError("ترميز/أبعاد الفيديو لا تطابق مواصفات H.264 والعمودي.")
    rate = video.get("avg_frame_rate") or video.get("r_frame_rate")
    try:
        numerator, denominator = (int(x) for x in rate.split("/", 1))
        actual_fps = numerator / denominator
    except (ValueError, AttributeError, ZeroDivisionError):
        raise VideoError("معدل إطارات الفيديو غير صالح.") from None
    if abs(actual_fps - fps) > 0.05:
        raise VideoError(f"معدل الإطارات النهائي {actual_fps:g} لا يساوي {fps}.")
    if audio.get("codec_name") != "aac":
        raise VideoError("ترميز الصوت النهائي يجب أن يكون AAC.")
    video_duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0)
    audio_duration = float(audio.get("duration") or data.get("format", {}).get("duration") or 0)
    if video_duration <= 0 or audio_duration <= 0 or abs(video_duration - audio_duration) > 0.30:
        raise VideoError("مدة الصوت والصورة غير صالحة أو غير متزامنة.")
    if expected_duration is not None and abs(video_duration - expected_duration) > 0.35:
        raise VideoError("مدة الملف النهائي لا تطابق مخطط المونتاج.")
    # Fail on a sustained fully black output (short fades/dark nature footage are tolerated).
    black = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path), "-vf", "blackdetect=d=3:pix_th=0.04",
                            "-an", "-f", "null", "-"], capture_output=True, text=True, check=False, timeout=180)
    if black.returncode != 0:
        raise VideoError("تعذر إكمال فحص الشاشات السوداء.")
    if re.search(r"black_duration:([3-9]\d*(?:\.\d+)?|[3-9](?:\.\d+)?|[1-9]\d+\.\d+)", black.stderr or ""):
        raise VideoError("اكتُشفت شاشة سوداء ممتدة في الفيديو.")
    return {
        "valid": True,
        "path": str(path),
        "sha256_ready": True,
        "video_codec": video.get("codec_name"),
        "audio_codec": audio.get("codec_name"),
        "width": width,
        "height": height,
        "fps": round(actual_fps, 3),
        "duration_seconds": round(video_duration, 3),
        "audio_duration_seconds": round(audio_duration, 3),
        "file_size_bytes": path.stat().st_size,
    }


# Tests are embedded to keep the requested source-file count at five.
if __name__ == "__main__":
    import unittest
    from unittest.mock import patch

    class VideoEngineTests(unittest.TestCase):
        def test_text_layout_rejects_empty_text(self):
            with self.assertRaises(VideoError):
                build_text_layout("")
            with self.assertRaises(VideoError):
                build_verse_layout("   ")

        def test_verse_highlight_index_must_be_valid(self):
            layout = build_verse_layout("بسم الله", width=540, height=960)
            with self.assertRaises(VideoError):
                draw_layout(layout, 2)
            with self.assertRaises(VideoError):
                draw_layout(layout, -1)

        def test_highlight_changes_only_the_current_word_and_keeps_the_card(self):
            layout = build_verse_layout("بسم الله الرحمن", chapter_name="الفاتحة", verse_number=1, width=540, height=960)
            plain = draw_layout(layout, None)
            lit = draw_layout(layout, 1)
            self.assertEqual(plain.size, lit.size)
            plain_bytes, lit_bytes = plain.tobytes(), lit.tobytes()
            changed = sum(1 for i in range(0, len(plain_bytes), 4) if plain_bytes[i:i + 4] != lit_bytes[i:i + 4])
            total_ink = sum(1 for i in range(3, len(plain_bytes), 4) if plain_bytes[i] > 0)
            self.assertGreater(changed, 0)
            self.assertLess(changed, total_ink * 0.25)

        def test_custom_font_requires_explicit_license(self):
            with patch.dict(os.environ, {"ARABIC_FONT": ""}):
                font_path = discover_font()
            with patch.dict(os.environ, {"ARABIC_FONT": font_path, "ARABIC_FONT_LICENSE_CONFIRMED": "false",
                                         "ARABIC_FONT_LICENSE_URL": "https://example.org/license"}):
                with self.assertRaises(VideoError):
                    discover_font()
            with patch.dict(os.environ, {"ARABIC_FONT": font_path, "ARABIC_FONT_LICENSE_CONFIRMED": "true",
                                         "ARABIC_FONT_LICENSE_URL": "https://example.org/license"}):
                self.assertEqual(discover_font(), font_path)
                self.assertEqual(_font_license_metadata(font_path)["license"], "Operator-confirmed")

        def test_duplicate_seal_in_generated_text_is_rejected(self):
            with tempfile.TemporaryDirectory() as temp:
                dummy = Path(temp) / "dummy"
                dummy.write_bytes(b"x" * 1000)
                audio = {key: dummy for key in ["hook", "recitation", "seal", "message", "cta"]}
                with self.assertRaises(VideoError):
                    build_video(dummy, audio, [{"word_index": 0, "start_ms": 0, "end_ms": 500}],
                                "بسم الله", {"hook": "سؤال", "message": "صدق الله العظيم", "call_to_action": "شارك"},
                                Path(temp) / "final_output.mp4")

        def test_final_mp4_probe_checks_video_and_audio_streams(self):
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "test.mp4"
                run_checked(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                             "-f", "lavfi", "-i", "color=c=white:s=1080x1920:r=30:d=1.5",
                             "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "1.5",
                             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)])
                result = validate_video(path, expected_duration=1.5)
                self.assertEqual(result["video_codec"], "h264")
                self.assertEqual(result["audio_codec"], "aac")

        # ----- Arabic rendering regressions (double shaping / reversed, unjoined letters) -----
        def test_text_is_passed_to_raqm_unmodified(self):
            # Pre-shaped or pre-reversed text would be shaped/reordered a second time by RAQM.
            sample = "فكرة الآية"
            self.assertEqual(_visual(sample), sample)

        def test_rendering_uses_rtl_direction_and_arabic_language(self):
            calls = []
            real_text = ImageDraw.ImageDraw.text
            def spy(self, xy, text, *args, **kwargs):
                calls.append((text, kwargs.get("direction"), kwargs.get("language")))
                return real_text(self, xy, text, *args, **kwargs)
            with patch.object(ImageDraw.ImageDraw, "text", spy):
                plain = build_text_layout("ما هو الإيمان بالغيب؟", "فكرة الآية", width=540, height=960)
                verse = build_verse_layout("بسم الله الرحمن", "الفاتحة", 1, width=540, height=960)
                draw_layout(plain, 0)
                draw_layout(verse, 1)
            self.assertTrue(calls)
            allowed_ornate = {ORNATE_OPEN_BRACKET, ORNATE_CLOSE_BRACKET}
            for text, direction, language in calls:
                self.assertEqual(direction, "rtl")
                self.assertEqual(language, "ar")
                # Canonical logical text only: never Arabic presentation forms (the ornate brackets are punctuation).
                self.assertFalse(any(("\ufb50" <= ch <= "\ufdff" or "\ufe70" <= ch <= "\ufeff") and ch not in allowed_ornate
                                     for ch in text), text)

        def test_joined_word_is_narrower_than_its_isolated_letters(self):
            # Proof that contextual shaping is applied: a connected word is not a row of isolated glyphs.
            font = _font(discover_font(), 60)
            draw = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
            word = "بببب"
            joined = _text_width(draw, word, font)
            isolated = 4 * _text_width(draw, "ب", font)
            self.assertLess(joined, isolated)

        # ----- Dark filter -----
        def test_dim_layer_is_light_black_and_strongest_near_the_text_band(self):
            layer = build_dim_layer(100, 400, 0.30)
            alpha = layer.getchannel("A")
            top, middle = alpha.getpixel((50, 2)), alpha.getpixel((50, int(400 * 0.42)))
            self.assertGreater(middle, top)
            self.assertLessEqual(alpha.getextrema()[1], int(255 * 0.30 * 1.25) + 2)
            self.assertEqual(layer.getpixel((50, 200))[:3], (0, 0, 0))

        def test_dim_strength_must_be_in_range(self):
            for bad in (-0.1, 0.9, True, "0.3"):
                with self.assertRaises(VideoError):
                    build_dim_layer(10, 10, bad)  # type: ignore[arg-type]
            self.assertEqual(build_dim_layer(10, 10, 0).getchannel("A").getextrema()[1], 0)

        # ----- Commentary word timing -----
        def test_estimated_timings_cover_duration_in_order_and_weight_by_length(self):
            words = ["في", "الاستقامة", "خير"]
            rows = estimate_word_timings(words, 3.0)
            self.assertEqual([r[2] for r in rows], [0, 1, 2])
            self.assertTrue(all(a < b for a, b, _ in rows))
            self.assertTrue(all(rows[i][1] <= rows[i + 1][0] + 1e-9 for i in range(2)))
            self.assertGreater(rows[1][1] - rows[1][0], rows[0][1] - rows[0][0])
            self.assertLessEqual(rows[-1][1], 3.0)

        def test_real_boundaries_are_used_only_when_count_and_bounds_match(self):
            text = "كلمة اولى وثانية"
            good = [{"start_ms": 100, "end_ms": 500}, {"start_ms": 600, "end_ms": 900}, {"start_ms": 1000, "end_ms": 1400}]
            rows, source = align_speech_words(text, good, 1.6)
            self.assertEqual(source, "tts_word_boundary")
            self.assertEqual(rows[1], (0.6, 0.9, 1))
            _, source = align_speech_words(text, good[:2], 1.6)
            self.assertEqual(source, "estimated")
            _, source = align_speech_words(text, None, 1.6)
            self.assertEqual(source, "estimated")
            unordered = [good[1], good[0], good[2]]
            _, source = align_speech_words(text, unordered, 1.6)
            self.assertEqual(source, "estimated")
            beyond = [{"start_ms": 100, "end_ms": 500}, {"start_ms": 600, "end_ms": 900}, {"start_ms": 9000, "end_ms": 9400}]
            _, source = align_speech_words(text, beyond, 1.6)
            self.assertEqual(source, "estimated")

        # ----- Timeline -----
        def test_timeline_highlight_fade_and_blank_gaps(self):
            layout = build_text_layout("كلمة اولى ثانية", width=270, height=480)
            scene = Scene("x", 1.0, 3.0, layout, [(1.0, 1.5, 0), (1.5, 2.0, 1), (2.0, 2.5, 2)], fade_in=0.0)
            self.assertIsNone(scene_word_at(scene, 0.9))
            self.assertEqual(scene_word_at(scene, 1.2), 0)
            self.assertEqual(scene_word_at(scene, 1.6), 1)
            self.assertEqual(scene_word_at(scene, 2.7), 2)       # lingers just after the last word
            self.assertIsNone(scene_word_at(scene, 2.95))
            self.assertEqual(scene_alpha(scene, 1.0), 1.0)         # no fade-in
            self.assertLess(scene_alpha(scene, 2.95), 1.0)         # fading out
            frames = list(render_frames([scene], 40, 10, 270, 480))  # 4 seconds at 10 fps
            self.assertEqual(len(frames), 40)
            blank = bytes(270 * 480 * 4)
            self.assertEqual(frames[0], blank)                      # before the scene
            self.assertNotEqual(frames[15], blank)                  # inside the scene
            self.assertEqual(frames[39], blank)                     # after the scene
            self.assertIs(frames[10], frames[11])                   # same word, reused bytes

        # ----- Multi-clip slowed background -----
        @staticmethod
        def _make_clip(path: Path, size: str, seconds: float, color: str, rate: int = 10) -> None:
            run_checked(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                         "-i", f"color=c={color}:s={size}:r={rate}:d={seconds}",
                         "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", str(seconds),
                         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)])

        def test_compose_background_slows_each_clip_and_concatenates(self):
            with tempfile.TemporaryDirectory() as temp:
                temp_path = Path(temp)
                clips = []
                for index, (size, color) in enumerate((("320x568", "red"), ("640x360", "green"), ("360x640", "blue"))):
                    clip = temp_path / f"c{index}.mp4"
                    self._make_clip(clip, size, 2.0, color)
                    clips.append(clip)
                output = temp_path / "background.mp4"
                info = compose_background(clips, output, 270, 480, 10, speed=0.75, source_seconds_each=2.0)
                # 3 clips x 2s of source at 0.75x speed = 8s of output.
                self.assertAlmostEqual(info["duration_seconds"], 8.0, delta=0.4)
                self.assertEqual(info["clips"], 3)
                self.assertEqual(info["speed"], 0.75)
                streams = probe(output)["streams"]
                video = next(s for s in streams if s["codec_type"] == "video")
                self.assertEqual((int(video["width"]), int(video["height"])), (270, 480))
                self.assertFalse(any(s["codec_type"] == "audio" for s in streams))

        def test_compose_background_target_duration_sets_per_clip_share(self):
            with tempfile.TemporaryDirectory() as temp:
                temp_path = Path(temp)
                clips = []
                for index in range(2):
                    clip = temp_path / f"c{index}.mp4"
                    self._make_clip(clip, "320x568", 12.0, "gray")
                    clips.append(clip)
                info = compose_background(clips, temp_path / "bg.mp4", 270, 480, 10, speed=0.75, target_duration=8.0)
                # ceil(8 * 0.75 / 2) + 1 = 4 source seconds per clip -> about 4 / 0.75 = 5.3s each.
                self.assertEqual(info["source_seconds_per_clip"], 4.0)
                self.assertGreaterEqual(info["duration_seconds"], 8.0)
                self.assertFalse(info["will_loop"])

        def test_compose_background_reports_when_the_result_must_loop(self):
            with tempfile.TemporaryDirectory() as temp:
                temp_path = Path(temp)
                clip = temp_path / "short.mp4"
                self._make_clip(clip, "320x568", 2.0, "gray")
                info = compose_background([clip], temp_path / "bg.mp4", 270, 480, 10, speed=0.75, target_duration=30.0)
                self.assertTrue(info["will_loop"])

        def test_compose_background_rejects_bad_arguments(self):
            with tempfile.TemporaryDirectory() as temp:
                temp_path = Path(temp)
                clip = temp_path / "c.mp4"
                self._make_clip(clip, "320x568", 1.0, "gray")
                destination = temp_path / "bg.mp4"
                with self.assertRaises(VideoError):
                    compose_background([], destination)
                for bad_speed in (0.1, 1.5, True):
                    with self.assertRaises(VideoError):
                        compose_background([clip], destination, 270, 480, 10, speed=bad_speed)  # type: ignore[arg-type]
                with self.assertRaises(VideoError):
                    compose_background([clip] * (MAX_BACKGROUND_CLIPS + 1), destination, 270, 480, 10)
                with self.assertRaises(VideoError):
                    compose_background([clip], destination, 271, 480, 10)
                with self.assertRaises(VideoError):
                    compose_background([clip], destination, 270, 480, 10, target_duration=-1)

        # ----- Quran words, pagination, multi-card commentary -----
        def test_quran_words_attach_waqf_signs_and_match_the_word_count_of_the_timestamps(self):
            self.assertEqual(quran_words("بسم الله الرحمن"), ["بسم", "الله", "الرحمن"])
            text = "فَٱتَّقُوا۟ ٱللَّهَ ۗ وَمَن يُوقَ ۚ فَأُو۟لَٰٓئِكَ"
            self.assertEqual(len(text.split()), 7)
            self.assertEqual(len(quran_words(text)), 5)
            self.assertEqual(quran_words("۞ بسم الله"), ["۞ بسم", "الله"])

        def test_verse_layout_counts_real_words_and_fits_a_long_verse(self):
            text = "بسم الله \u06d7 الرحمن الرحيم \u06da " + " ".join(["الحمد", "لله", "رب", "العالمين"] * 7)
            layout = build_verse_layout(text, "الفاتحة", 1, width=540, height=960)
            self.assertEqual(layout.word_count, len(quran_words(text)))
            self.assertLess(layout.word_count, len(text.split()))
            self.assertEqual(draw_layout(layout, layout.word_count - 1).size, (540, 960))
            with self.assertRaises(VideoError):
                draw_layout(layout, layout.word_count)

        def test_paginate_text_keeps_every_word_in_order_and_prefers_sentence_breaks(self):
            sentences = ["كلمة " * 11 + "تامة.", "عبارة " * 11 + "ثانية،", "جملة " * 11 + "ثالثة.", "ختام " * 5 + "قوي."]
            text = " ".join(sentences)
            pages = paginate_text(text)
            self.assertEqual([word for page in pages for word in page], text.split())
            self.assertTrue(all(len(page) <= PAGE_MAX_WORDS for page in pages))
            self.assertTrue(all(len(page) >= PAGE_MIN_WORDS // 2 for page in pages))
            self.assertTrue(pages[0][-1].endswith((".", "،")))
            self.assertEqual(paginate_text("كلمتان فقط"), [["كلمتان", "فقط"]])
            with self.assertRaises(VideoError):
                paginate_text("   ")
            tail = " ".join(["كلمة"] * 25 + ["أ", "ب"])           # would leave a 2-word last page
            tail_pages = paginate_text(tail)
            self.assertEqual([w for page in tail_pages for w in page], tail.split())
            self.assertGreaterEqual(len(tail_pages[-1]), PAGE_MIN_WORDS // 2)

        def test_commentary_scenes_split_long_text_into_gapless_cards_with_page_relative_timings(self):
            words = [("كلمة%d." % i if i % 12 == 11 else "كلمة%d" % i) for i in range(50)]
            text = " ".join(words)
            timings = [(10.0 + i * 0.4, 10.0 + i * 0.4 + 0.35, i) for i in range(50)]
            scenes = commentary_scenes("message", "رسالة لك", text, 10.0, 30.5, timings, 270, 480, discover_font())
            self.assertGreaterEqual(len(scenes), 3)
            self.assertEqual(scenes[0].start, 10.0)
            self.assertEqual(scenes[-1].end, 30.5)
            for left, right in zip(scenes, scenes[1:]):
                self.assertAlmostEqual(left.end, right.start, places=6)     # no blank gap between cards
                self.assertEqual(left.fade_out, 0.0)
            self.assertEqual(scenes[0].fade_in, FADE_SECONDS)
            self.assertEqual(scenes[-1].fade_out, FADE_SECONDS)
            self.assertEqual(sum(scene.layout.word_count for scene in scenes), 50)
            for scene in scenes:
                indexes = [index for _, _, index in scene.words]
                self.assertEqual(indexes, list(range(scene.layout.word_count)))
            single = commentary_scenes("hook", "", "جملة قصيرة قوية", 0.0, 3.0, [(0.1, 0.8, 0), (0.9, 1.5, 1), (1.6, 2.4, 2)],
                                       270, 480, discover_font(), first_fade_in=0.0)
            self.assertEqual((len(single), single[0].name, single[0].fade_in), (1, "hook", 0.0))

        # ----- Whole pipeline at a small size -----
        def test_full_build_uses_card_dim_and_reports_timing_sources(self):
            with tempfile.TemporaryDirectory() as temp:
                temp_path = Path(temp)
                run_checked(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                             "-i", "color=c=0x303030:s=360x640:r=10:d=2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                             str(temp_path / "bg.mp4")])
                lengths = {"hook": 1.0, "recitation": 2.0, "seal": 1.0, "message": 3.0, "cta": 1.0}
                audio = {}
                for name, seconds in lengths.items():
                    audio[name] = temp_path / f"{name}.wav"
                    run_checked(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                                 "-i", f"sine=frequency=330:duration={seconds}", "-ar", "48000", str(audio[name])])
                segments = [{"word_index": i, "start_ms": 100 + i * 450, "end_ms": 500 + i * 450} for i in range(4)]
                content = {"hook": "تعبك الذي لا يراه أحد يسمعه الله",
                           "message": "هذه الآية تقول لك إن الصبر ليس ضعفًا. ابدأ اليوم بخطوة صغيرة واحدة.",
                           "call_to_action": "شارك هذا المقطع واجعله صدقة جارية"}
                hook_boundaries = [{"start_ms": 50 + i * 110, "end_ms": 140 + i * 110}
                                   for i in range(len("تعبك الذي لا يراه أحد يسمعه الله".split()))]
                output = temp_path / "final_output.mp4"
                result = build_video(temp_path / "bg.mp4", audio, segments, "بسم الله الرحمن الرحيم", content, output,
                                     width=270, height=480, fps=10, target_min_duration=8, target_max_duration=60,
                                     tts_word_boundaries={"hook": hook_boundaries}, chapter_name="الفاتحة",
                                     verse_number=1, dim_strength=0.30)
                self.assertTrue(result["valid"])
                self.assertEqual(result["tts_word_timing"]["hook"], "tts_word_boundary")
                self.assertEqual(result["tts_word_timing"]["message"], "estimated")
                self.assertEqual(result["commentary_pages"], {"hook": 1, "message": 1, "cta": 1})
                self.assertEqual(result["dim_strength"], 0.30)

                def frame_at(second: float) -> Image.Image:
                    frame = temp_path / f"frame-{second}.png"
                    run_checked(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(second),
                                 "-i", str(output), "-frames:v", "1", str(frame)])
                    with Image.open(frame) as image:
                        return image.convert("RGB").copy()
                def gold_pixels(image: Image.Image) -> int:
                    data = image.tobytes()
                    return sum(1 for i in range(0, len(data), 3)
                               if data[i] > 150 and 110 < data[i + 1] < 210 and data[i + 2] < 130)
                recitation_frame = frame_at(2.0)
                # The recitation card shows gold ornaments; a hook frame does not (solid background).
                self.assertGreater(gold_pixels(recitation_frame), 40)
                self.assertLess(gold_pixels(frame_at(0.4)), gold_pixels(recitation_frame) / 2)
                # The dark filter darkens the plain background (0x30 = 48) at the very top of the frame.
                top_pixel = recitation_frame.getpixel((135, 2))
                self.assertLess(top_pixel[0], 44)

        def test_full_build_accepts_a_composed_multi_clip_background(self):
            with tempfile.TemporaryDirectory() as temp:
                temp_path = Path(temp)
                clips = []
                for index, color in enumerate(("0x203040", "0x304020")):
                    clip = temp_path / f"c{index}.mp4"
                    self._make_clip(clip, "360x640", 3.0, color)
                    clips.append(clip)
                background = temp_path / "background.mp4"
                compose_background(clips, background, 270, 480, 10, speed=0.75, target_duration=8.0)
                lengths = {"hook": 1.0, "recitation": 2.0, "seal": 1.0, "message": 3.0, "cta": 1.0}
                audio = {}
                for name, seconds in lengths.items():
                    audio[name] = temp_path / f"{name}.wav"
                    run_checked(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                                 "-i", f"sine=frequency=330:duration={seconds}", "-ar", "48000", str(audio[name])])
                segments = [{"word_index": i, "start_ms": 100 + i * 450, "end_ms": 500 + i * 450} for i in range(4)]
                content = {"hook": "تعبك الذي لا يراه أحد يسمعه الله",
                           "message": "هذه الآية تقول لك إن الصبر ليس ضعفًا. ابدأ اليوم بخطوة صغيرة واحدة.",
                           "call_to_action": "شارك هذا المقطع واجعله صدقة جارية"}
                result = build_video(background, audio, segments, "بسم الله الرحمن الرحيم", content,
                                     temp_path / "final_output.mp4", width=270, height=480, fps=10,
                                     target_min_duration=8, target_max_duration=60)
                self.assertTrue(result["valid"])

        def test_encoder_failure_is_reported_as_video_error(self):
            frames = iter([bytes(16)] * 3)
            with self.assertRaises(VideoError):
                _encode_with_overlay_pipe(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "does-not-exist.mp4",
                                           "-f", "null", "-"], frames, timeout=30)

    unittest.main()
