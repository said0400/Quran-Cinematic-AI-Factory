"""Vertical video engine. Quran text is rendered as supplied; no audio is sped up or cut."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PIL import Image, ImageDraw, ImageFont, features


class VideoError(RuntimeError):
    """Video generation or validation failed."""


# Arabic is shaped and ordered by Pillow's RAQM layout engine (HarfBuzz + FriBiDi) directly from
# the logical (canonical) text. Pre-shaping with arabic-reshaper/python-bidi must NOT be combined
# with RAQM: the text would be processed twice, giving reversed order and unjoined letters.
TEXT_DIRECTION = "rtl"
TEXT_LANGUAGE = "ar"


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


def _draw_text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font: ImageFont.FreeTypeFont,
               fill: tuple[int, int, int, int], stroke: bool = True) -> None:
    kwargs: dict[str, Any] = {"font": font, "fill": fill, "direction": TEXT_DIRECTION, "language": TEXT_LANGUAGE}
    if stroke:
        kwargs.update(stroke_width=1, stroke_fill=(0, 0, 0, 255))
    draw.text(xy, _visual(text), **kwargs)


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size=size, layout_engine=ImageFont.Layout.RAQM)
    except (OSError, AttributeError):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError as exc:
            raise VideoError("تعذر تحميل الخط العربي المحدد.") from exc


def _wrap_words(words: list[str], font: ImageFont.FreeTypeFont, max_width: int, draw: ImageDraw.ImageDraw,
                gap: int) -> list[list[int]]:
    lines: list[list[int]] = []
    current: list[int] = []
    width = 0
    for index, word in enumerate(words):
        word_width = _text_width(draw, word, font)
        if current and width + gap + word_width > max_width:
            lines.append(current)
            current, width = [], 0
        current.append(index)
        width += word_width + (gap if len(current) > 1 else 0)
    if current:
        lines.append(current)
    return lines


def render_verse_overlay(text: str, current_word: int | None, destination: Path,
                          width: int = 1080, height: int = 1920, font_path: str | None = None,
                          draw_background: bool = True, highlight_only: bool = False) -> None:
    words = text.split()
    if not words or (current_word is not None and not 0 <= current_word < len(words)):
        raise VideoError("نص الآية أو فهرس الكلمة المتلوة غير صالح.")
    font_path = font_path or discover_font()
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    # Adapt font size and lines to retain all words at a readable size.
    selected = None
    for size in range(66, 39, -2):
        font = _font(font_path, size)
        lines = _wrap_words(words, font, int(width * 0.86), draw, max(12, size // 3))
        if len(lines) <= 5:
            selected = (font, lines, size)
            break
    if selected is None:
        raise VideoError("تعذر عرض الآية كاملة بخط مقروء ضمن مساحة الشاشة.")
    font, lines, size = selected
    gap = max(12, size // 3)
    line_height = int(size * 1.65)
    content_height = len(lines) * line_height
    top = int(height * 0.36 - content_height / 2)
    left, right = int(width * 0.055), int(width * 0.945)
    if draw_background:
        draw.rounded_rectangle((left, top - 44, right, top + content_height + 44), radius=34, fill=(0, 0, 0, 155))
    for row, indexes in enumerate(lines):
        widths = [_text_width(draw, words[index], font) for index in indexes]
        total = sum(widths) + gap * max(0, len(widths) - 1)
        cursor = (width + total) / 2
        y = top + row * line_height
        # Logical Arabic order starts at the right; render each word as an independent unit.
        for index, word_width in zip(indexes, widths):
            if highlight_only and index != current_word:
                cursor -= word_width + gap
                continue
            cursor -= word_width
            color = (255, 218, 74, 255) if index == current_word else (255, 255, 255, 255)
            _draw_text(draw, (cursor, y), words[index], font, color)
            cursor -= gap
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, "PNG", optimize=True)


def render_text_overlay(text: str, destination: Path, label: str = "", width: int = 1080,
                        height: int = 1920, font_path: str | None = None) -> None:
    if not isinstance(text, str) or not text.strip() or "�" in text:
        raise VideoError("نص المشهد فارغ أو يحوي رمزًا تالفًا.")
    font_path = font_path or discover_font()
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    label_font = _font(font_path, 42)
    # Find a size that keeps all text on screen; explicit line wrapping preserves all words.
    font = None
    lines = []
    size = 68
    for try_size in range(74, 41, -2):
        candidate = _font(font_path, try_size)
        logical = text.split()
        candidate_lines: list[str] = []
        current: list[str] = []
        for word in logical:
            attempt = " ".join(current + [word])
            if current and _text_width(draw, attempt, candidate) > int(width * 0.84):
                candidate_lines.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            candidate_lines.append(" ".join(current))
        if len(candidate_lines) <= 7:
            font, lines, size = candidate, candidate_lines, try_size
            break
    if font is None:
        raise VideoError("النص أطول من مساحة الشاشة؛ رُفض بدل إسقاط كلمات.")
    label_y = int(height * 0.34)
    body_y = label_y + (76 if label else 0)
    line_height = int(size * 1.55)
    block_height = len(lines) * line_height + (75 if label else 0)
    top = int(height * 0.48 - block_height / 2)
    draw.rounded_rectangle((int(width * .06), top - 46, int(width * .94), top + block_height + 50),
                           radius=34, fill=(0, 0, 0, 158))
    if label:
        label_width = _text_width(draw, label, label_font)
        _draw_text(draw, ((width - label_width) / 2, top + 8), label, label_font, (255, 218, 74, 255), stroke=False)
    start_y = top + (78 if label else 20)
    for idx, line in enumerate(lines):
        line_width = _text_width(draw, line, font)
        _draw_text(draw, ((width - line_width) / 2, start_y + idx * line_height), line, font, (255, 255, 255, 255))
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, "PNG", optimize=True)


def _audio_concat(audio_files: list[Path], destination: Path) -> float:
    if len(audio_files) != 6:
        raise VideoError("يجب تمرير ستة مقاطع صوتية منفصلة: هوك، تلاوة، العبارة، تفسير، فائدة، خاتمة.")
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


def build_video(background: Path, audio_files: dict[str, Path], recitation_segments: list[dict[str, int]],
                verse_text: str, content: dict[str, Any], output_path: Path,
                width: int = 1080, height: int = 1920, fps: int = 30,
                target_min_duration: float = 45, target_max_duration: float = 60) -> dict[str, Any]:
    """Build exactly ordered scenes without cropping or speeding up the recitation."""
    required = ["hook", "explanation", "practical_benefit", "call_to_action"]
    if not all(isinstance(content.get(key), str) and content[key].strip() for key in required):
        raise VideoError("نصوص الفيديو المطلوبة ناقصة.")
    if "صدق الله العظيم" in " ".join(content[key] for key in required):
        raise VideoError("عبارة «صدق الله العظيم» يجب أن تكون مقطعًا مستقلًا بعد التلاوة فقط.")
    keys = ["hook", "recitation", "seal", "explanation", "benefit", "cta"]
    if any(key not in audio_files for key in keys):
        raise VideoError("مقاطع الصوت الستة المطلوبة غير مكتملة.")
    if not recitation_segments:
        raise VideoError("توقيتات الكلمات الحقيقية غير متاحة.")
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
    overlays: list[tuple[Path, float, float]] = []
    hook_start, hook_end = stage_intervals["hook"]
    hook_png = work / "hook.png"
    render_text_overlay(content["hook"], hook_png, "فكرة الآية", width, height, font_path)
    overlays.append((hook_png, hook_start, hook_end))
    rec_start, rec_end = stage_intervals["recitation"]
    base_verse = work / "verse-base.png"
    render_verse_overlay(verse_text, None, base_verse, width, height, font_path)
    overlays.append((base_verse, rec_start, rec_end))
    words = verse_text.split()
    for index, segment in enumerate(recitation_segments):
        word_index = int(segment["word_index"])
        if not 0 <= word_index < len(words):
            raise VideoError("فهرس توقيت كلمة خارج نص الآية.")
        start = rec_start + int(segment["start_ms"]) / 1000.0
        end = rec_start + int(segment["end_ms"]) / 1000.0
        if start < rec_start or end > rec_end + 0.3 or end <= start:
            raise VideoError("توقيت الكلمة لا يقع داخل مقطع التلاوة.")
        frame = work / f"word-{index:04d}.png"
        render_verse_overlay(verse_text, word_index, frame, width, height, font_path,
                             draw_background=False, highlight_only=True)
        overlays.append((frame, start, end))
    seal_start, seal_end = stage_intervals["seal"]
    seal_png = work / "seal.png"
    render_text_overlay("صدق الله العظيم", seal_png, "", width, height, font_path)
    overlays.append((seal_png, seal_start, seal_end))
    for key, label in (("explanation", "المعنى"), ("benefit", "فائدة عملية"), ("cta", "شارك الفائدة")):
        start, end = stage_intervals[key]
        if key == "cta" and video_duration > total_audio:
            # Keep the final call-to-action on screen over any deliberate minimum-length padding.
            end = video_duration
        text_key = {"explanation": "explanation", "benefit": "practical_benefit", "cta": "call_to_action"}[key]
        frame = work / f"{key}.png"
        render_text_overlay(content[text_key], frame, label, width, height, font_path)
        overlays.append((frame, start, end))
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-stream_loop", "-1", "-i", str(background), "-i", str(combined_audio)]
    for frame, start, end in overlays:
        frame_duration = max(0.08, end - start + 0.08)
        command.extend(["-loop", "1", "-framerate", str(fps), "-t", f"{frame_duration:.6f}", "-i", str(frame)])
    filters = [
        f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},"
        f"zoompan=z='min(zoom+0.00015,1.08)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={width}x{height}:fps={fps},"
        "eq=contrast=1.03:saturation=1.06,format=yuv420p[base]"
    ]
    current = "base"
    for index, (_, start, end) in enumerate(overlays):
        # PNG inputs are placed at their real timeline offset; no inferred word timings are used.
        nxt = f"v{index}"
        input_index = 2 + index
        filters.append(f"[{input_index}:v]format=rgba,setpts=PTS-STARTPTS+{start:.6f}/TB[ov{index}]")
        filters.append(f"[{current}][ov{index}]overlay=0:0:eof_action=pass:enable='between(t,{start:.6f},{end:.6f})'[{nxt}]")
        current = nxt
    filters.append(f"[1:a]apad=whole_dur={video_duration:.6f},atrim=duration={video_duration:.6f},asetpts=PTS-STARTPTS[aout]")
    command.extend(["-filter_complex", ";".join(filters), "-map", f"[{current}]", "-map", "[aout]",
                    "-t", f"{video_duration:.6f}", "-r", str(fps), "-c:v", "libx264", "-preset", "medium",
                    "-crf", "19", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                    "-ar", "48000", "-movflags", "+faststart", str(output_path)])
    run_checked(command, timeout=1800)
    if not output_path.is_file():
        raise VideoError("لم ينتج FFmpeg ملف الفيديو النهائي.")
    validation = validate_video(output_path, width, height, fps, expected_duration=video_duration)
    validation.update({"audio_segments": {key: {"start": round(a, 3), "end": round(b, 3)} for key, (a, b) in stage_intervals.items()},
                       "recitation_duration_seconds": round(rec_duration, 3),
                       "target_duration_met": video_duration >= target_min_duration,
                       "padding_seconds": round(max(0, video_duration - total_audio), 3),
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
    import tempfile
    import unittest
    from unittest.mock import patch

    class VideoEngineTests(unittest.TestCase):
        def test_text_overlay_rejects_empty_text(self):
            with tempfile.TemporaryDirectory() as temp:
                with self.assertRaises(VideoError):
                    render_text_overlay("", Path(temp) / "empty.png")

        def test_verse_highlight_index_must_be_valid(self):
            with tempfile.TemporaryDirectory() as temp:
                with self.assertRaises(VideoError):
                    render_verse_overlay("بسم الله", 2, Path(temp) / "bad.png")

        def test_word_highlight_layer_is_transparent_and_does_not_repeat_card_background(self):
            with tempfile.TemporaryDirectory() as temp:
                base = Path(temp) / "base.png"
                word = Path(temp) / "word.png"
                render_verse_overlay("بسم الله", None, base, width=540, height=960)
                render_verse_overlay("بسم الله", 0, word, width=540, height=960,
                                     draw_background=False, highlight_only=True)
                with Image.open(base) as base_image, Image.open(word) as word_image:
                    base_alpha = base_image.getchannel("A").histogram()
                    word_alpha = word_image.getchannel("A").histogram()
                    self.assertGreater(sum(base_alpha[1:]), sum(word_alpha[1:]))

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
                audio = {key: dummy for key in ["hook", "recitation", "seal", "explanation", "benefit", "cta"]}
                with self.assertRaises(VideoError):
                    build_video(dummy, audio, [{"word_index": 0, "start_ms": 0, "end_ms": 500}],
                                "بسم الله", {"hook": "سؤال", "explanation": "صدق الله العظيم", "practical_benefit": "فائدة", "call_to_action": "شارك"},
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
            with tempfile.TemporaryDirectory() as temp, patch.object(ImageDraw.ImageDraw, "text", spy):
                render_text_overlay("ما هو الإيمان بالغيب؟", Path(temp) / "t.png", "فكرة الآية", width=540, height=960)
                render_verse_overlay("بسم الله الرحمن", None, Path(temp) / "v.png", width=540, height=960)
            self.assertTrue(calls)
            for text, direction, language in calls:
                self.assertEqual(direction, "rtl")
                self.assertEqual(language, "ar")
                # Canonical logical text only: never presentation forms (U+FB50-U+FDFF, U+FE70-U+FEFF).
                self.assertFalse(any("\ufb50" <= ch <= "\ufdff" or "\ufe70" <= ch <= "\ufeff" for ch in text), text)

        def test_joined_word_is_narrower_than_its_isolated_letters(self):
            # Proof that contextual shaping is applied: a connected word is not a row of isolated glyphs.
            font = _font(discover_font(), 60)
            draw = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
            word = "بببب"
            joined = _text_width(draw, word, font)
            isolated = 4 * _text_width(draw, "ب", font)
            self.assertLess(joined, isolated)

    unittest.main()
