"""External services for Quran Cinematic AI Factory.

All network access is isolated here (scene images live in scene_images.py). Rights are never
inferred from API access; operators must configure and record the applicable source-specific permissions.
"""
from __future__ import annotations

import asyncio
import hashlib
import html
import io
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from procedural_background import ANIMATIONS, ELEMENT_TYPES, MOODS, POSITIONS, SIZES, SKY_TYPES
from scene_images import sanitize_image_prompt

QF_API = "https://apis.quran.foundation/content/api/v4"
QF_PRELIVE_API = "https://apis-prelive.quran.foundation/content/api/v4"
QF_PRODUCTION_AUTH = "https://oauth2.quran.foundation"
QF_PRELIVE_AUTH = "https://prelive-oauth2.quran.foundation"
QF_DOCS = "https://api-docs.quran.foundation/docs/content_apis_versioned/4.0.0/"
QF_TERMS = "https://api-docs.quran.foundation/legal/developer-terms/"
YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
YOUTUBE_READ_SCOPE = "https://www.googleapis.com/auth/youtube.readonly"
YOUTUBE_UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
GITHUB_API = "https://api.github.com"
# Make.com webhooks live under *.make.com (and legacy *.integromat.com).
MAKE_ALLOWED_SUFFIXES = (".make.com", ".integromat.com")

# Gemini: delays (seconds) between attempts. len(delays)+1 attempts in total.
GEMINI_RETRY_DELAYS = (2.0, 6.0, 15.0, 30.0)
GEMINI_REQUEST_TIMEOUT_MS = 90_000
# Exceptions without an HTTP code (network layer) are retried when their class
# hierarchy contains one of these names (httpx / aiohttp / builtins).
GEMINI_TRANSIENT_EXCEPTION_NAMES = {
    "TransportError", "TimeoutException", "ConnectionError", "TimeoutError",
    "ServerDisconnectedError", "ClientConnectionError",
}

# Gemini text-to-speech (commentary voice only; the recitation always stays source audio).
# The TTS rate limits are tighter than text generation, hence the longer back-off.
GEMINI_TTS_RETRY_DELAYS = (3.0, 10.0, 30.0, 60.0)
DEFAULT_GEMINI_TTS_MODEL = "gemini-3.8-flash-tts"
DEFAULT_GEMINI_TTS_VOICE = "Algieba"
GEMINI_TTS_TERMS = "https://ai.google.dev/gemini-api/terms"
# Plausible speaking pace, used to reject truncated or runaway generations.
TTS_MIN_SECONDS_PER_WORD = 0.12
TTS_MAX_SECONDS_PER_WORD = 3.0
TTS_MIN_SECONDS = 0.4
TTS_MIN_PEAK = 500                      # of 32767; below this the audio is effectively silent
TTS_PROVIDERS = ("edge", "gemini", "auto")

# Arabic names of the 114 surahs (index = chapter number - 1), shown on the verse card.
SURAH_NAMES_AR = (
    'الفاتحة', 'البقرة', 'آل عمران', 'النساء', 'المائدة', 'الأنعام',
    'الأعراف', 'الأنفال', 'التوبة', 'يونس', 'هود', 'يوسف',
    'الرعد', 'إبراهيم', 'الحجر', 'النحل', 'الإسراء', 'الكهف',
    'مريم', 'طه', 'الأنبياء', 'الحج', 'المؤمنون', 'النور',
    'الفرقان', 'الشعراء', 'النمل', 'القصص', 'العنكبوت', 'الروم',
    'لقمان', 'السجدة', 'الأحزاب', 'سبأ', 'فاطر', 'يس',
    'الصافات', 'ص', 'الزمر', 'غافر', 'فصلت', 'الشورى',
    'الزخرف', 'الدخان', 'الجاثية', 'الأحقاف', 'محمد', 'الفتح',
    'الحجرات', 'ق', 'الذاريات', 'الطور', 'النجم', 'القمر',
    'الرحمن', 'الواقعة', 'الحديد', 'المجادلة', 'الحشر', 'الممتحنة',
    'الصف', 'الجمعة', 'المنافقون', 'التغابن', 'الطلاق', 'التحريم',
    'الملك', 'القلم', 'الحاقة', 'المعارج', 'نوح', 'الجن',
    'المزمل', 'المدثر', 'القيامة', 'الإنسان', 'المرسلات', 'النبأ',
    'النازعات', 'عبس', 'التكوير', 'الانفطار', 'المطففين', 'الانشقاق',
    'البروج', 'الطارق', 'الأعلى', 'الغاشية', 'الفجر', 'البلد',
    'الشمس', 'الليل', 'الضحى', 'الشرح', 'التين', 'العلق',
    'القدر', 'البينة', 'الزلزلة', 'العاديات', 'القارعة', 'التكاثر',
    'العصر', 'الهمزة', 'الفيل', 'قريش', 'الماعون', 'الكوثر',
    'الكافرون', 'النصر', 'المسد', 'الإخلاص', 'الفلق', 'الناس',
)

# edge-tts: delays between synthesis attempts for one text segment.
TTS_RETRY_DELAYS = (2.0, 5.0)
# Known misspellings of Microsoft voice names, corrected before the availability check.
TTS_VOICE_ALIASES = {"ar-EG-ShakerNeural": "ar-EG-ShakirNeural"}


class ServiceError(RuntimeError):
    """A sanitized, user-safe service or validation failure."""
    def __init__(self, message: str, platform_state: dict[str, Any] | None = None):
        super().__init__(message)
        self.platform_state = platform_state or {}


class _GeminiTransient(Exception):
    """Internal marker: a Gemini failure that is worth retrying."""


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ServiceError(f"المتغير المطلوب {name} غير مضبوط.")
    return value


def _license_allowlist_contains(name: str, resource_id: int, license_url: str) -> bool:
    """Allowlist entries use `resource_id|https://license-url`, separated by semicolons."""
    entries = os.getenv(name, "").split(";")
    expected = (str(resource_id), license_url.rstrip("/"))
    for entry in entries:
        if "|" not in entry:
            continue
        allowed_id, allowed_url = (part.strip() for part in entry.split("|", 1))
        if (allowed_id, allowed_url.rstrip("/")) == expected:
            return True
    return False


def _is_mishary_reciter(name: str) -> bool:
    normalized = normalize_arabic(name).lower()
    latin_name = re.sub(r"[^a-z]", "", normalized)
    return (("mishari" in latin_name or "mishary" in latin_name)
            and ("afasy" in latin_name or "afasi" in latin_name or "afassy" in latin_name)) or (
                "مشاري" in normalized and "عفاسي" in normalized
            )


def verse_key(chapter: int, verse: int) -> str:
    if not 1 <= int(chapter) <= 114 or int(verse) < 1:
        raise ServiceError("رقم السورة أو الآية خارج النطاق.")
    return f"{int(chapter)}:{int(verse)}"


def normalize_arabic(value: str) -> str:
    """Comparison-only normalization; never used to rewrite displayed Quran text."""
    value = unicodedata.normalize("NFKC", value or "")
    value = value.replace("ـ", "")
    value = "".join(ch for ch in value if not unicodedata.category(ch).startswith("M"))
    value = value.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي"}))
    return re.sub(r"\s+", " ", value).strip()


def quran_words(text: str) -> list[str]:
    """Split Quran text into display/timing words.

    Whitespace-separated tokens that contain no letter (waqf signs such as the small high
    ligatures, the rosette, the verse-end sign) are not words: Quran Foundation gives no timing
    for them. They are attached to the preceding word (or the next one at the very start) so the
    word count matches the word-level timestamps and the sign is highlighted with its word.
    Kept identical to `video_engine.quran_words` (a unit test in main.py compares them).
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


def validate_word_segments(segments: Any, text_uthmani: str, audio_duration_ms: int,
                           end_tolerance_ms: int = 300) -> list[dict[str, int]]:
    """Validate Quran Foundation [word_index,start_ms,end_ms] timing rows.

    Only contiguous 0-based or 1-based word indexes are accepted. No timing is
    generated or interpolated by this function.
    """
    words = quran_words(text_uthmani)
    if not isinstance(segments, list) or not words:
        raise ServiceError("توقيتات الكلمات غير متاحة أو نص الآية فارغ؛ رُفض المرشح.")
    if len(segments) != len(words):
        raise ServiceError(f"عدد التوقيتات ({len(segments)}) لا يطابق عدد كلمات الآية ({len(words)}).")
    parsed: list[tuple[int, int, int]] = []
    for row in segments:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ServiceError("صيغة توقيت الكلمة لا تطابق [word_index,start_ms,end_ms].")
        try:
            idx, start, end = (int(row[0]), int(row[1]), int(row[2]))
        except (TypeError, ValueError):
            raise ServiceError("توقيت الكلمة يحتوي قيمة غير رقمية.") from None
        if start < 0 or end <= start or end > audio_duration_ms + end_tolerance_ms:
            raise ServiceError(
                f"توقيت كلمة خارج مدة الصوت أو بترتيب غير صالح: index={idx}, start={start}ms, "
                f"end={end}ms, audio_duration={audio_duration_ms}ms."
            )
        parsed.append((idx, start, end))
    indexes = [row[0] for row in parsed]
    if indexes == list(range(len(words))):
        offset = 0
    elif indexes == list(range(1, len(words) + 1)):
        offset = 1
    else:
        raise ServiceError("فهارس توقيت الكلمات ليست متصلة أو متوافقة مع الآية.")
    previous_start = -1
    result = []
    for idx, start, end in parsed:
        if start < previous_start:
            raise ServiceError("توقيتات الكلمات ليست مرتبة.")
        previous_start = start
        result.append({"word_index": idx - offset, "start_ms": start, "end_ms": end})
    if result[0]["start_ms"] > 1200:
        raise ServiceError("التوقيتات لا تغطي بداية التلاوة بما يكفي للتحقق من المزامنة.")
    if audio_duration_ms - result[-1]["end_ms"] > max(2500, int(audio_duration_ms * 0.18)):
        raise ServiceError("التوقيتات لا تغطي نهاية التلاوة؛ رُفض المرشح.")
    return result


@dataclass
class QuranVerse:
    chapter: int
    verse: int
    key: str
    text_uthmani: str
    verse_id: int | None
    chapter_name: str = ""


@dataclass
class QuranRecitation:
    reciter_id: int
    reciter_name: str
    audio_url: str
    segments: list[dict[str, int]]
    source_url: str
    duration_hint: int | None = None


@dataclass
class TafsirSource:
    text: str
    source_name: str
    resource_id: int
    source_url: str
    license_url: str
    license_verified_by_operator: bool


def _plain_text(markup: str) -> str:
    class StripHTML(__import__("html.parser", fromlist=["HTMLParser"]).HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.parts: list[str] = []
        def handle_data(self, data: str) -> None:
            self.parts.append(data)
    parser = StripHTML()
    parser.feed(markup or "")
    return html.unescape(" ".join(parser.parts)).strip()


class QuranFoundationClient:
    """Server-side Content API client with cached OAuth2 client-credentials tokens."""
    def __init__(self, session: requests.Session | None = None, base_url: str | None = None,
                 auth_base_url: str | None = None):
        environment = os.getenv("QF_ENV", "production").strip().lower()
        if environment not in {"production", "prelive"}:
            raise ServiceError("QF_ENV يجب أن يكون production أو prelive.")
        default_api = QF_PRELIVE_API if environment == "prelive" else QF_API
        default_auth = QF_PRELIVE_AUTH if environment == "prelive" else QF_PRODUCTION_AUTH
        self.base_url = (base_url or os.getenv("QF_API_BASE_URL", default_api)).rstrip("/")
        self.auth_base_url = (auth_base_url or os.getenv("QF_AUTH_BASE_URL", default_auth)).rstrip("/")
        self.session = session or requests.Session()
        self.client_id = _required_env("QF_CLIENT_ID")
        self.client_secret = _required_env("QF_CLIENT_SECRET")
        self.headers = {"x-client-id": self.client_id, "Accept": "application/json"}
        self._access_token_expires_at = 0.0

    def _request_access_token(self) -> None:
        try:
            response = self.session.post(
                f"{self.auth_base_url}/oauth2/token",
                auth=(self.client_id, self.client_secret),
                data={"grant_type": "client_credentials", "scope": "content"},
                headers={"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
                timeout=(15, 45),
            )
        except requests.RequestException as exc:
            raise ServiceError(f"تعذر طلب رمز Quran Foundation ({type(exc).__name__}).") from None
        if response.status_code != 200:
            raise ServiceError(f"خدمة مصادقة Quran Foundation أعادت HTTP {response.status_code}.")
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError):
            raise ServiceError("استجابة مصادقة Quran Foundation ليست JSON صالحًا.") from None
        token = payload.get("access_token") if isinstance(payload, dict) else None
        try:
            lifetime = float(payload.get("expires_in", 0)) if isinstance(payload, dict) else 0
        except (TypeError, ValueError):
            lifetime = 0
        if not isinstance(token, str) or not token.strip() or lifetime <= 0:
            raise ServiceError("استجابة مصادقة Quran Foundation تفتقر إلى access_token أو expires_in صالح.")
        self.headers["x-auth-token"] = token
        refresh_margin = min(120.0, max(10.0, lifetime * 0.1))
        self._access_token_expires_at = time.monotonic() + max(1.0, lifetime - refresh_margin)

    def _ensure_access_token(self, force: bool = False) -> None:
        if not force and self.headers.get("x-auth-token") and time.monotonic() < self._access_token_expires_at:
            return
        self._request_access_token()

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        self._ensure_access_token()
        try:
            response = self.session.get(url, headers=self.headers, params=params or {}, timeout=(15, 45))
            if response.status_code == 401:
                self._ensure_access_token(force=True)
                response = self.session.get(url, headers=self.headers, params=params or {}, timeout=(15, 45))
        except requests.RequestException as exc:
            raise ServiceError(f"تعذر الاتصال بـQuran Foundation ({type(exc).__name__}).") from None
        if response.status_code != 200:
            raise ServiceError(f"Quran Foundation أعاد HTTP {response.status_code}.")
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError):
            raise ServiceError("استجابة Quran Foundation ليست JSON صالحًا.") from None
        if not isinstance(payload, dict):
            raise ServiceError("بنية استجابة Quran Foundation غير صالحة.")
        return payload

    def get_verse(self, chapter: int, verse: int) -> QuranVerse:
        key = verse_key(chapter, verse)
        payload = self._get_json(f"verses/by_key/{key}", {
            "fields": "text_uthmani,verse_key,verse_number,chapter_id,id",
        })
        item = payload.get("verse")
        if not isinstance(item, dict):
            raise ServiceError("استجابة الآية لا تحتوي الحقل الموثق verse.")
        if item.get("verse_key") != key or item.get("verse_number") != int(verse) or item.get("chapter_id") != int(chapter):
            raise ServiceError("مفتاح الآية أو بياناتها لا تطابق الطلب اليدوي.")
        text = item.get("text_uthmani")
        if not isinstance(text, str) or not text.strip() or "�" in text:
            raise ServiceError("النص العثماني مفقود أو تالف.")
        verse_id = item.get("id")
        return QuranVerse(int(chapter), int(verse), key, text.strip(),
                          int(verse_id) if verse_id is not None else None,
                          SURAH_NAMES_AR[int(chapter) - 1])

    def get_recitation(self, chapter: int, verse: int, audio_dir: Path) -> tuple[QuranRecitation, Path]:
        key = verse_key(chapter, verse)
        reciter_id = int(_required_env("CHAPTER_RECITER_ID"))
        if not env_bool("RECITATION_LICENSE_CONFIRMED"):
            raise ServiceError("رفض استخدام التلاوة: RECITATION_LICENSE_CONFIRMED=true مطلوب بعد التحقق من حق إعادة النشر.")
        license_url = _required_env("RECITATION_LICENSE_URL")
        if urlparse(license_url).scheme != "https":
            raise ServiceError("RECITATION_LICENSE_URL يجب أن يكون HTTPS.")
        if not _license_allowlist_contains("RECITATION_ALLOWLIST", reciter_id, license_url):
            raise ServiceError("معرّف التلاوة ورابط ترخيصه غير موجودين في RECITATION_ALLOWLIST؛ رُفض الاستخدام.")

        catalog = self._get_json("resources/chapter_reciters", {"language": "en"})
        reciters = catalog.get("reciters")
        if not isinstance(reciters, list):
            raise ServiceError("كتالوج قراء السور لا يحتوي الحقل الموثق reciters.")
        selected = next((item for item in reciters if isinstance(item, dict) and item.get("id") == reciter_id), None)
        reciter_name = selected.get("name", "").strip() if isinstance(selected, dict) and isinstance(selected.get("name"), str) else ""
        if not reciter_name:
            raise ServiceError(f"CHAPTER_RECITER_ID={reciter_id} غير موجود في كتالوج قراء السور؛ لا تستخدم RECITATION_ID من كتالوج مختلف.")
        if env_bool("RECITATION_REQUIRE_MISHARY", True) and not _is_mishary_reciter(reciter_name):
            raise ServiceError("المعرّف المحدد لا يطابق مشاري العفاسي؛ اختر قارئ سورة مرخصًا أو عطّل RECITATION_REQUIRE_MISHARY بعد التحقق.")

        source_url = f"{QF_API}/chapter_recitations/{reciter_id}/{chapter}?segments=true"
        payload = self._get_json(f"chapter_recitations/{reciter_id}/{chapter}", {"segments": "true"})
        audio_file = payload.get("audio_file")
        if not isinstance(audio_file, dict):
            raise ServiceError("استجابة chapter_recitations لا تحتوي audio_file.")
        audio_url = audio_file.get("audio_url")
        timestamps = audio_file.get("timestamps")
        if not isinstance(audio_url, str) or not audio_url or not isinstance(timestamps, list):
            raise ServiceError("مصدر التلاوة لا يوفر audio_url وtimestamps؛ لا يمكن مزامنة الكلمات بأمان.")
        verse_timing = next((row for row in timestamps if isinstance(row, dict) and row.get("verse_key") == key), None)
        if not isinstance(verse_timing, dict):
            raise ServiceError(f"لا توجد توقيتات موثقة للآية {key} في صوت السورة.")
        try:
            verse_start = int(verse_timing["timestamp_from"])
            verse_end = int(verse_timing["timestamp_to"])
        except (KeyError, TypeError, ValueError):
            raise ServiceError(f"حدود توقيت الآية {key} غير صالحة في استجابة المصدر.") from None
        if verse_start < 0 or verse_end <= verse_start:
            raise ServiceError(f"حدود توقيت الآية {key} غير مرتبة.")
        raw_segments = verse_timing.get("segments")
        if not isinstance(raw_segments, list):
            raise ServiceError(f"المصدر لا يوفر قائمة توقيتات كلمات للآية {key}.")
        absolute_segments: list[list[int]] = []
        for row in raw_segments:
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                raise ServiceError(f"المصدر لم يقدّم توقيت كلمة ثلاثيًا للآية {key}؛ اختر قارئ سورة يوفر word-level timestamps.")
            try:
                word_index, start_ms, end_ms = (int(value) for value in row)
            except (TypeError, ValueError):
                raise ServiceError(f"المصدر أعاد توقيت كلمة غير رقمي للآية {key}.") from None
            if start_ms < 0 or end_ms <= start_ms:
                raise ServiceError(f"المصدر أعاد نطاق توقيت غير صالح لكلمة في الآية {key}.")
            # Keep the exact API word bounds; the clip origin is derived from their union
            # with the verse interval so small source/encoder boundary differences cannot
            # make a valid first or last word fall outside the cropped MP3.
            absolute_segments.append([word_index, start_ms, end_ms])
        if not absolute_segments:
            raise ServiceError(f"قائمة توقيتات الكلمات للآية {key} فارغة.")
        clip_start_ms = max(0, min(verse_start, min(row[1] for row in absolute_segments)) - 200)
        clip_end_ms = max(verse_end, max(row[2] for row in absolute_segments)) + 200
        segments = [[index, start_ms - clip_start_ms, end_ms - clip_start_ms]
                    for index, start_ms, end_ms in absolute_segments]

        full_url = audio_url if audio_url.startswith("https://") else urljoin("https://verses.quran.foundation/", audio_url)
        parsed = urlparse(full_url)
        allowed_hosts = {"verses.quran.foundation", "download.quranicaudio.com"}
        if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
            raise ServiceError("رابط صوت السورة ليس HTTPS على مضيف موثق ومسموح.")
        audio_dir.mkdir(parents=True, exist_ok=True)
        chapter_audio = audio_dir / f"chapter-recitation-{reciter_id}-{chapter}.mp3"
        destination = audio_dir / f"recitation-{chapter}-{verse}.mp3"
        try:
            response = self.session.get(full_url, headers={"User-Agent": "Quran-Cinematic-AI-Factory/1.0"}, timeout=(20, 120), stream=True)
            response.raise_for_status()
            final_host = urlparse(getattr(response, "url", full_url)).hostname
            content_type = (response.headers.get("Content-Type", "") or "").split(";", 1)[0].lower()
            if final_host not in allowed_hosts:
                raise ServiceError("تنزيل التلاوة أعاد تحويلًا إلى مضيف غير مصرح.")
            if content_type and not (content_type.startswith("audio/") or content_type == "application/octet-stream"):
                raise ServiceError("استجابة CDN ليست نوع ملف صوتيًا.")
            total = 0
            with chapter_audio.open("wb") as out:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        total += len(chunk)
                        if total > 160 * 1024 * 1024:
                            raise ServiceError("ملف السورة يتجاوز حد التنزيل الآمن البالغ 160 MiB.")
                        out.write(chunk)
        except ServiceError:
            chapter_audio.unlink(missing_ok=True)
            raise
        except requests.RequestException as exc:
            chapter_audio.unlink(missing_ok=True)
            raise ServiceError(f"تعذر تنزيل التلاوة ({type(exc).__name__}).") from None
        if not chapter_audio.is_file() or chapter_audio.stat().st_size < 4096:
            chapter_audio.unlink(missing_ok=True)
            raise ServiceError("ملف صوت السورة فارغ أو غير مكتمل.")
        try:
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(chapter_audio), "-ss", f"{clip_start_ms / 1000:.3f}",
                "-t", f"{(clip_end_ms - clip_start_ms) / 1000:.3f}", "-vn",
                "-c:a", "libmp3lame", "-q:a", "2", str(destination),
            ], check=True, capture_output=True, text=True, timeout=180)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            destination.unlink(missing_ok=True)
            chapter_audio.unlink(missing_ok=True)
            raise ServiceError("تعذر اقتطاع الآية من صوت السورة؛ تحقق من تثبيت FFmpeg وصلاحية ملف الصوت.") from None
        if not destination.is_file() or destination.stat().st_size < 4096:
            destination.unlink(missing_ok=True)
            raise ServiceError("ملف التلاوة المقتطع فارغ أو غير مكتمل.")
        chapter_audio.unlink(missing_ok=True)
        return QuranRecitation(reciter_id, reciter_name, full_url, segments, source_url,
                               clip_end_ms - clip_start_ms), destination

    def get_tafsir(self, key: str) -> TafsirSource:
        resource_id = int(_required_env("TAFSIR_ID"))
        if not env_bool("TAFSIR_COMMERCIAL_LICENSE_VERIFIED"):
            raise ServiceError("رفض استخدام التفسير: أكد الترخيص التجاري وإعادة النشر عبر TAFSIR_COMMERCIAL_LICENSE_VERIFIED=true.")
        license_url = _required_env("TAFSIR_LICENSE_URL")
        if urlparse(license_url).scheme != "https":
            raise ServiceError("TAFSIR_LICENSE_URL يجب أن يكون HTTPS.")
        if not _license_allowlist_contains("TAFSIR_ALLOWLIST", resource_id, license_url):
            raise ServiceError("معرّف التفسير ورابط ترخيصه غير موجودين في TAFSIR_ALLOWLIST؛ رُفض الاستخدام.")
        source_url = f"{QF_API}/tafsirs/{resource_id}/by_ayah/{key}"
        payload = self._get_json(f"tafsirs/{resource_id}/by_ayah/{key}", {
            "fields": "verse_key,resource_name,id,language_name,language_id",
        })
        item = payload.get("tafsir")
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise ServiceError("استجابة التفسير لا تطابق بنية tafsir الموثقة.")
        if item.get("resource_id") != resource_id:
            raise ServiceError("معرّف مصدر التفسير في الاستجابة لا يطابق المورد المطلوب.")
        source_name = item.get("resource_name")
        if not isinstance(source_name, str) or not source_name.strip():
            raise ServiceError("اسم مصدر التفسير غير متوفر.")
        language_name = item.get("language_name")
        if not isinstance(language_name, str) or language_name.strip().lower() != "arabic":
            detected_language = language_name.strip() if isinstance(language_name, str) and language_name.strip() else "غير محددة"
            hint = " مورد التفسير الميسر العربي في Quran.com هو TAFSIR_ID=16." if resource_id == 169 else ""
            raise ServiceError(
                f"TAFSIR_ID={resource_id} ({source_name}) لغته {detected_language} وليس تفسيرًا عربيًا؛ "
                f"اختر موردًا عربيًا وحدّث TAFSIR_ALLOWLIST إلى معرّفه بعد التحقق من الترخيص.{hint}"
            )
        verse_ids = item.get("verses")
        if not isinstance(verse_ids, dict) or key not in verse_ids:
            raise ServiceError("الاستجابة لا تثبت أن نص التفسير يخص مفتاح الآية المطلوب.")
        text = _plain_text(item["text"])
        if len(text) < 30:
            raise ServiceError("نص التفسير قصير أو فارغ ولا يكفي للتلخيص.")
        arabic_letters = sum(1 for char in text if "\u0600" <= char <= "\u06ff" and char.isalpha())
        if arabic_letters < 15 or arabic_letters / max(1, len(text)) < 0.10:
            raise ServiceError("نص التفسير لا يبدو عربيًا بما يكفي؛ رُفض بدل تلخيص لغة غير مطلوبة.")
        return TafsirSource(text, source_name.strip(), resource_id, source_url, license_url, True)


HOOK_SCORE_KEYS = (
    "start_strength", "curiosity", "emotional_impact", "relevance", "clarity",
    "brevity", "uniqueness", "non_misleading", "general_audience",
)
HOOK_MIN_WORDS, HOOK_MAX_WORDS = 3, 18
MESSAGE_MIN_WORDS_FLOOR = 20            # hard floor; the prompt asks for a longer text
CTA_MIN_WORDS, CTA_MAX_WORDS = 6, 34
TITLE_MAX_WORDS, TITLE_MAX_CHARS = 16, 95
DESCRIPTION_MIN_CHARS, DESCRIPTION_MAX_CHARS = 300, 1400
KEYWORDS_MIN, KEYWORDS_MAX = 6, 15
HASHTAGS_MIN, HASHTAGS_MAX = 6, 12
CONTENT_ATTEMPTS = 3                    # regenerations when the content fails validation or review
SHARE_VERB_PATTERN = re.compile(r"شارك|انشر|أرسل|ارسل|مرّر|مرر")
# The published text must not name sources (reciter, tafsir, stock site, AI vendors).
FORBIDDEN_SOURCE_PATTERN = re.compile(
    r"pexels|quran\.foundation|gemini|جيميناي|الميسر|مشاري|العفاسي|ابن كثير|الطبري|السعدي|القرطبي|البغوي",
    re.I)
URL_PATTERN = re.compile(r"(?:https?://|www\.)", re.I)
SEAL_PHRASE = "صدق الله العظيم"

_ELEMENT_SCHEMA = {"type": "OBJECT", "properties": {
    "type": {"type": "STRING"}, "position": {"type": "STRING"},
    "animation": {"type": "STRING"}, "size": {"type": "STRING"},
}, "required": ["type", "position", "animation"]}
_SCENE_SCHEMA = {"type": "OBJECT", "properties": {
    "mood": {"type": "STRING"}, "text_span": {"type": "STRING"}, "image_prompt": {"type": "STRING"},
    "elements": {"type": "ARRAY", "items": _ELEMENT_SCHEMA},
}, "required": ["mood", "elements", "image_prompt"]}

GENERATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "hook_candidates": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "text": {"type": "STRING"},
                    **{key: {"type": "INTEGER"} for key in HOOK_SCORE_KEYS},
                },
                "required": ["text", *HOOK_SCORE_KEYS]
            }
        },
        "message": {"type": "STRING"},
        "call_to_action": {"type": "STRING"},
        "title": {"type": "STRING"},
        "description": {"type": "STRING"},
        "keywords": {"type": "ARRAY", "items": {"type": "STRING"}},
        "hashtags": {"type": "ARRAY", "items": {"type": "STRING"}},
        "visual_plan": {
            "type": "OBJECT",
            "properties": {
                "opening": _SCENE_SCHEMA,
                "scenes": {"type": "ARRAY", "items": _SCENE_SCHEMA},
                "closing": _SCENE_SCHEMA,
            },
            "required": ["opening", "scenes", "closing"]
        },
    },
    "required": [
        "hook_candidates", "message", "call_to_action",
        "title", "description", "keywords", "hashtags", "visual_plan"
    ],
}
REVIEW_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "grounded_in_source": {"type": "BOOLEAN"},
        "no_unsupported_claims": {"type": "BOOLEAN"},
        "no_contradiction": {"type": "BOOLEAN"},
        "needs_review": {"type": "BOOLEAN"},
        "reason": {"type": "STRING"},
    },
    "required": ["grounded_in_source", "no_unsupported_claims", "no_contradiction", "needs_review", "reason"],
}


class ContentValidationError(ServiceError):
    """The generated copy broke a content rule; regenerating it is worthwhile."""


def _normalize_hashtags(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        raise ContentValidationError("حقل الهاشتاجات ليس قائمة.")
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            raise ContentValidationError("أحد الهاشتاجات ليس نصًا.")
        tag = re.sub(r"\s+", "_", item.strip().lstrip("#").strip())
        tag = re.sub(r"[^\w\u0600-\u06FF]", "", tag).strip("_")
        if not 2 <= len(tag) <= 40:
            continue
        signature = normalize_arabic(tag).lower()
        if signature in seen:
            continue
        seen.add(signature)
        result.append("#" + tag)
    if len(result) < HASHTAGS_MIN:
        raise ContentValidationError(f"عدد الهاشتاجات الصالحة أقل من {HASHTAGS_MIN}.")
    return result[:HASHTAGS_MAX]


def _normalize_keywords(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        raise ContentValidationError("حقل الكلمات المفتاحية ليس قائمة.")
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            raise ContentValidationError("إحدى الكلمات المفتاحية ليست نصًا.")
        keyword = re.sub(r"\s+", " ", item.strip().lstrip("#").strip())
        if not 2 <= len(keyword) <= 40 or URL_PATTERN.search(keyword):
            continue
        signature = normalize_arabic(keyword).lower()
        if signature in seen:
            continue
        seen.add(signature)
        result.append(keyword)
    if len(result) < KEYWORDS_MIN:
        raise ContentValidationError(f"عدد الكلمات المفتاحية الصالحة أقل من {KEYWORDS_MIN}.")
    return result[:KEYWORDS_MAX]


def _norm_tokens(text: str) -> list[tuple[int, str]]:
    """(index in text.split(), normalised word); punctuation-only tokens are skipped."""
    out = []
    for index, word in enumerate((text or "").split()):
        token = re.sub(r"[^\w]", "", normalize_arabic(word))
        if token:
            out.append((index, token))
    return out


def _clean_scene(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("mood") not in MOODS:
        raise ContentValidationError(f"مزاج المشهد {label} غير مسموح أو مفقود.")
    elements = raw.get("elements")
    if not isinstance(elements, list) or not 1 <= len(elements) <= 6:
        raise ContentValidationError(f"المشهد {label} يجب أن يحتوي من عنصر إلى ستة عناصر.")
    clean: list[dict[str, str]] = []
    taken: set[tuple[str, str]] = set()
    for item in elements:
        if (not isinstance(item, dict) or item.get("type") not in ELEMENT_TYPES
                or item.get("position") not in POSITIONS or item.get("animation") not in ANIMATIONS):
            raise ContentValidationError("عنصر بصري غير مسموح (النوع أو الموضع أو الحركة).")
        slot = (item["type"], "") if item["type"] in SKY_TYPES else ("ground", item["position"])
        if slot in taken:
            continue                                   # two things in one spot: keep the first
        taken.add(slot)
        size = item.get("size") if item.get("size") in SIZES else "medium"
        clean.append({"type": item["type"], "position": item["position"], "animation": item["animation"], "size": size})
    # An unusable prompt only costs this scene its picture (it is drawn procedurally); it never fails the video.
    return {"mood": raw["mood"], "elements": clean, "image_prompt": sanitize_image_prompt(raw.get("image_prompt"))}


def validate_visual_plan(raw: Any, message: str) -> dict[str, Any]:
    """Validate Gemini's scene plan and bind each scene to a word of the spoken message."""
    if not isinstance(raw, dict):
        raise ContentValidationError("الخطة البصرية ليست كائنًا.")
    opening, closing = _clean_scene(raw.get("opening"), "opening"), _clean_scene(raw.get("closing"), "closing")
    items = raw.get("scenes")
    if not isinstance(items, list) or not 3 <= len(items) <= 6:
        raise ContentValidationError("مشاهد الرسالة يجب أن تكون من ثلاثة إلى ستة.")
    words = _norm_tokens(message)
    cursor, scenes = 0, []
    for number, item in enumerate(items, start=1):
        scene = _clean_scene(item, str(number))
        span = item.get("text_span") if isinstance(item.get("text_span"), str) else ""
        tokens = [token for _, token in _norm_tokens(span)][:3]
        if not tokens:
            raise ContentValidationError(f"المشهد {number} بلا text_span.")
        found = next((i for i in range(cursor, len(words) - len(tokens) + 1)
                      if [token for _, token in words[i:i + len(tokens)]] == tokens), None)
        if found is None:
            raise ContentValidationError(f"عبارة المشهد {number} ليست في الرسالة بالترتيب.")
        scene["word_start"], scene["text_span"] = words[found][0], span.strip()[:160]
        cursor = found + 1
        scenes.append(scene)
    return {"opening": opening, "scenes": scenes, "closing": closing}


def validate_generated_content(data: Any, max_message_words: int = 100) -> dict[str, Any]:
    """Validate the model output and return the normalized copy.

    Every rule failure raises `ContentValidationError`, which `GeminiContentService` answers by
    regenerating (a few times) instead of failing the whole candidate.
    """
    if not isinstance(data, dict):
        raise ContentValidationError("مخرجات النموذج ليست كائن JSON.")
    candidates = data.get("hook_candidates")
    if not isinstance(candidates, list) or len(candidates) != 5:
        raise ContentValidationError("يجب إنشاء خمس صيغ مستقلة للهوك بالضبط.")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("text"), str):
            raise ContentValidationError("بنية أحد مرشحي الهوك غير صالحة.")
        text = candidate["text"].strip()
        word_count = len(text.split())
        if not HOOK_MIN_WORDS <= word_count <= HOOK_MAX_WORDS or SEAL_PHRASE in text or URL_PATTERN.search(text):
            raise ContentValidationError("مرشح الهوك قصير جدًا أو طويل أو يكرر عبارة ما بعد التلاوة.")
        signature = normalize_arabic(text)
        if signature in seen:
            raise ContentValidationError("مرشحو الهوك مكرّرون.")
        seen.add(signature)
        scores = {}
        for key in HOOK_SCORE_KEYS:
            score = candidate.get(key)
            if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
                raise ContentValidationError(f"تقييم الهوك {key} خارج المجال 1..5.")
            scores[key] = score
        if scores["non_misleading"] < 4 or scores["relevance"] < 4:
            raise ContentValidationError("أحد مرشحي الهوك لا يحقق الحد الأدنى للارتباط وعدم التضليل.")
        normalized.append({"text": text, "scores": scores, "total_score": sum(scores.values())})

    def text_field(name: str, min_words: int, max_words: int, max_chars: int) -> str:
        value = data.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ContentValidationError(f"حقل {name} فارغ أو غير صالح.")
        value = value.strip()
        words = len(value.split())
        if not min_words <= words <= max_words or len(value) > max_chars:
            raise ContentValidationError(f"طول حقل {name} خارج الحدود ({words} كلمة).")
        if SEAL_PHRASE in value:
            raise ContentValidationError("يُمنع تكرار عبارة صدق الله العظيم داخل النص المولّد.")
        if URL_PATTERN.search(value):
            raise ContentValidationError("روابط أو نصوص خارجية غير موثوقة مرفوضة في مخرجات Gemini.")
        return value

    message = text_field("message", MESSAGE_MIN_WORDS_FLOOR, int(max_message_words), 1600)
    cta = text_field("call_to_action", CTA_MIN_WORDS, CTA_MAX_WORDS, 400)
    if not SHARE_VERB_PATTERN.search(cta):
        raise ContentValidationError("دعوة المشاركة لا تحتوي فعل مشاركة واضحًا.")
    title = text_field("title", 3, TITLE_MAX_WORDS, TITLE_MAX_CHARS)
    if "#" in title:
        raise ContentValidationError("العنوان يجب ألا يحتوي هاشتاجات.")
    description = text_field("description", 30, 400, DESCRIPTION_MAX_CHARS)
    if len(description) < DESCRIPTION_MIN_CHARS:
        raise ContentValidationError("الوصف أقصر من الحد الأدنى.")
    keywords = _normalize_keywords(data.get("keywords"))
    hashtags = _normalize_hashtags(data.get("hashtags"))
    visual_plan = validate_visual_plan(data.get("visual_plan"), message)
    published_text = " ".join([title, description, *keywords, *hashtags])
    if FORBIDDEN_SOURCE_PATTERN.search(published_text):
        raise ContentValidationError("النص المنشور يذكر مصدرًا أو اسمًا ممنوعًا.")
    best = max(normalized, key=lambda item: (item["total_score"], item["scores"]["non_misleading"], item["scores"]["relevance"]))
    return {"hook": best["text"], "hook_candidates": normalized, "message": message,
            "call_to_action": cta, "title": title, "description": description,
            "keywords": keywords, "hashtags": hashtags, "visual_plan": visual_plan}


def _gemini_exception_info(exc: BaseException) -> tuple[bool, int | None, str]:
    """Classify a Gemini SDK/network exception as (retryable, http_code, status_text).

    google-genai API errors expose `.code` (HTTP status) and `.status` (e.g. UNAVAILABLE).
    Network-layer errors have no code and are recognised by class name.
    """
    code = getattr(exc, "code", None)
    if isinstance(code, bool) or not isinstance(code, int):
        code = None
    status = getattr(exc, "status", None)
    status_text = status if isinstance(status, str) else ""
    if code is not None:
        return (code in {408, 425, 429} or 500 <= code <= 599), code, status_text
    names = {cls.__name__ for cls in type(exc).__mro__}
    return bool(names & GEMINI_TRANSIENT_EXCEPTION_NAMES), None, status_text


def _describe_gemini_exception(exc: BaseException, api_key: str, code: int | None, status: str) -> str:
    """Diagnostic text for logs: class, HTTP code, status and a redacted, truncated message."""
    message = getattr(exc, "message", None)
    if not isinstance(message, str) or not message.strip():
        message = str(exc)
    message = re.sub(r"\s+", " ", message).strip()
    if api_key:
        message = message.replace(api_key, "***")
    message = re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "***", message)
    parts = [type(exc).__name__]
    if code is not None:
        parts.append(f"code={code}")
    if status:
        parts.append(f"status={status}")
    detail = ", ".join(parts)
    if message:
        detail += f": {message[:200]}"
    return detail


def _gemini_finish_reason(response: Any) -> str:
    try:
        reason = response.candidates[0].finish_reason
        return str(getattr(reason, "name", reason))
    except Exception:
        return "unknown"


class GeminiContentService:
    # Class attributes so tests (and callers) can override them per instance.
    retry_delays = GEMINI_RETRY_DELAYS
    _sleep = staticmethod(time.sleep)

    def __init__(self, model: str | None = None):
        self.api_key = _required_env("GEMINI_API_KEY")
        self.model = (model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")).strip()
        if not self.model:
            raise ServiceError("GEMINI_MODEL لا يمكن أن يكون فارغًا.")

    def _generate_json_once(self, prompt: str, schema: dict[str, Any], max_tokens: int) -> dict[str, Any]:
        """One Gemini call. Raises _GeminiTransient for retryable failures, ServiceError otherwise."""
        try:
            from google import genai
            from google.genai import types
        except Exception as exc:
            raise ServiceError(f"تعذر استيراد مكتبة Gemini ({type(exc).__name__}).") from None
        try:
            client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=GEMINI_REQUEST_TIMEOUT_MS),
            )
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.25,
                    max_output_tokens=max_tokens,
                ),
            )
        except Exception as exc:
            retryable, code, status = _gemini_exception_info(exc)
            detail = _describe_gemini_exception(exc, self.api_key, code, status)
            if retryable:
                raise _GeminiTransient(detail) from None
            raise ServiceError(f"فشل استدعاء Gemini دون قابلية لإعادة المحاولة ({detail}).") from None
        try:
            text = getattr(response, "text", None)
        except Exception:
            text = None
        if not isinstance(text, str) or not text.strip():
            raise _GeminiTransient(f"استجابة فارغة (finish_reason={_gemini_finish_reason(response)})")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            raise _GeminiTransient(
                f"استجابة ليست JSON صالحًا (finish_reason={_gemini_finish_reason(response)})"
            ) from None
        if not isinstance(data, dict):
            raise _GeminiTransient("استجابة Gemini لا تطابق الكائن المتوقع.")
        return data

    def _generate_json(self, prompt: str, schema: dict[str, Any], max_tokens: int) -> dict[str, Any]:
        """Call Gemini with exponential-style backoff on transient failures (5xx, 429, timeouts, empty output)."""
        attempts = len(self.retry_delays) + 1
        for attempt in range(1, attempts + 1):
            try:
                return self._generate_json_once(prompt, schema, max_tokens)
            except _GeminiTransient as exc:
                detail = str(exc)
                print(f"Gemini المحاولة {attempt}/{attempts} فشلت (مؤقت): {detail}", file=sys.stderr)
                if attempt == attempts:
                    raise ServiceError(f"فشل استدعاء Gemini بعد {attempts} محاولات: {detail}") from None
                self._sleep(self.retry_delays[attempt - 1])
        raise ServiceError("فشل استدعاء Gemini.")  # pragma: no cover - loop always returns or raises

    @staticmethod
    def _generation_prompt(verse: QuranVerse, tafsir: TafsirSource, topic: str | None,
                           message_words: tuple[int, int]) -> str:
        moods = ", ".join(sorted(MOODS))
        kinds = ", ".join(sorted(ELEMENT_TYPES))
        places = ", ".join(POSITIONS)
        moves = ", ".join(sorted(ANIMATIONS))
        low, high = message_words
        topic_line = f"الموضوع المقترح لزاوية الطرح: {topic}\n" if topic else ""
        return (
            "أنت كاتب محتوى عربي محترف متخصص في الفيديوهات القصيرة المؤثرة، ولست مصدرًا شرعيًا. "
            "المعنى الديني يؤخذ حصرًا من نص التفسير الموثق المرفق: لا تضف أحكامًا ولا أسباب نزول ولا أحاديث "
            "ولا قصصًا ولا أرقامًا ولا معلومات غير موجودة فيه، ولا تنسب للآية وعدًا شخصيًا غيبيًا ولا تضمن للمشاهد "
            "حل مشكلته أو أجرًا معينًا. اكتب بعربية فصحى بسيطة نابضة، بجمل قصيرة وإيقاع مؤثر، وخاطب المشاهد بضمير المخاطب.\n\n"
            "المطلوب (وفق المخطط حصرًا):\n"
            "1) hook_candidates: خمس صيغ مختلفة تمامًا لهوك افتتاحي يُقال قبل الآية، كل واحدة من "
            f"{HOOK_MIN_WORDS + 3} إلى 16 كلمة. يضرب الهوك إحساسًا حقيقيًا يعيشه الناس اليوم (التعب الصامت، ضيق الرزق، "
            "الخوف من المستقبل، ظلم الناس، انتظار الفرج، ضغط الدراسة والامتحان، الوحدة، فقد الأحبة، الإرهاق من الدنيا) "
            "ويرتبط بمعنى الآية، ويخلق فضولًا لا يُحل إلا بسماع الآية. تجنب المقدمات المستهلكة مثل «هل تعلم»، "
            "وتجنب الأسئلة العامة الباردة، ولا تقتبس نص الآية، ولا توهم بأن الآية رسالة شخصية له. "
            "قيّم كل صيغة بأعداد صحيحة من 1 إلى 5 للمعايير: " + ", ".join(HOOK_SCORE_KEYS) + ".\n"
            f"2) message: نص واحد متماسك يُقال بعد الآية ({low} إلى {high} كلمة) يدمج معنى الآية من التفسير والفائدة العملية "
            "في رسالة قوية تلمس قلب كل مشاهد مهما كانت ظروفه: ابدأ بكشف المعنى ببساطة، ثم خاطب موقفًا يوميًا واقعيًا "
            "يعيشه الناس، ثم أعطه خطوة عملية يفعلها اليوم، واختم بجملة قوية تبقى في ذهنه. بلا قوائم، ولا تكرر الهوك، "
            "ولا تقتبس الآية، ولا تكتب «صدق الله العظيم».\n"
            f"3) call_to_action: دعوة مشاركة عاطفية قوية ومحترمة من {CTA_MIN_WORDS + 2} إلى 28 كلمة تدفع المشاهد لمشاركة المقطع الآن، "
            "وتربط المشاركة بالصدقة الجارية والخير الذي يصل لمن يحب، كوالديه أو أحبّته أو من فقدهم أو من يحتاج هذه الكلمات "
            "(مثال على الروح فقط، لا تنسخه: «شاركه الآن، فلعله يكون صدقة جارية لك ولوالديك»). استخدم فعل «شارك» أو «انشر» أو «أرسل». "
            "نوّع الصياغة، ولا تجزم بأجر معين ولا تذكر حديثًا ولا حكمًا شرعيًا.\n"
            f"4) title: عنوان واحد قوي جدًا في جملة واحدة (3 إلى {TITLE_MAX_WORDS} كلمة، وأقل من {TITLE_MAX_CHARS} حرفًا) "
            "يجمع الفضول والإحساس، بلا رقم آية ولا هاشتاجات ولا علامات اقتباس.\n"
            f"5) description: وصف طويل بلا حشو ولا تحيات ({DESCRIPTION_MIN_CHARS + 200} إلى 1100 حرف) في فقرات قصيرة، "
            "يبدأ بجملة قوية، ويوضح ما سيجده المشاهد من معنى وفائدة، ويضمّن الكلمات المفتاحية طبيعيًا، وينتهي بدعوة للمشاركة. "
            "لا روابط، ولا تذكر أي مصدر أو قارئ أو موقع أو تفسير أو أداة أو ذكاءً اصطناعيًا.\n"
            f"6) keywords: من {KEYWORDS_MIN + 4} إلى {KEYWORDS_MAX} كلمة أو عبارة بحث عربية (كلمتان إلى أربع) يبحث بها الناس فعلًا عن هذا الموضوع.\n"
            f"7) hashtags: من {HASHTAGS_MIN + 2} إلى {HASHTAGS_MAX} هاشتاجًا عربيًا (ويجوز اثنان بالإنجليزية) يبدأ كل منها بعلامة # دون مسافات (استخدم _ بين الكلمات).\n\n"
            "8) visual_plan: خطة رسوم لمحرك محلي (نص فقط؛ لا صور ولا API) تجعل الصورة تحكي ما يُقال بالضبط:\n"
            "- opening: مشهد يرافق الهوك وحده ويجسّد الشعور أو المشكلة.\n"
            "- scenes: من 3 إلى 6 مشاهد تتبع message بالترتيب. لكل مشهد text_span = أول 3 إلى 6 كلمات من الجملة "
            "التي يبدأ عندها المشهد، منسوخة حرفيًا من message كما كُتبت، ليظهر المشهد لحظة نطقها. "
            "يتبدل المشهد عند تحول المعنى (الشعور، معنى الآية، الأمل، الخطوة العملية).\n"
            "- closing: مشهد دعوة المشاركة (قلب، دعاء، نور، باب).\n"
            f"لكل مشهد: mood من [{moods}]، وعناصر elements من 2 إلى 4: type من [{kinds}]، "
            f"position من [{places}]، animation من [{moves}]، وsize اختياري small أو medium أو large. "
            "اجعل العناصر تجسّد المعنى حرفيًا: تعب وابتلاء = sitting_person مع cloud وrain؛ فرج وأمل = sun أو light_rays أو door؛ "
            "رزق ونمو = seed ثم tree أو coin؛ ثبات وطريق = path مع person بحركة walk؛ دعاء = hands مع stars؛ "
            "أسرة وبر = house مع heart؛ شائعات وهاتف = phone؛ حساب وميزان = scale أو clock؛ علم وقرآن = book. "
            "وأضف لكل مشهد (opening وscenes وclosing) الحقل image_prompt: وصف بصري بالإنجليزية فقط "
            "(من 20 إلى 60 كلمة، وأقل من 400 حرف) للمشهد نفسه، يطابق ما يُقال في تلك اللحظة: المكان والإضاءة "
            "والألوان والأشياء الواضحة والحالة الشعورية. استعمل رموزًا بصرية (طريق، نافذة، مطر، بحر، بذرة، ضوء، باب، "
            "يدان مرفوعتان، بيت من بعيد) وأشخاصًا كظلال بعيدة فقط. ممنوع ذكر الأنبياء والملائكة والصحابة والكعبة "
            "والمصحف والوجوه والأشخاص القريبين والكتابة والشعارات والأعلام والروابط. لا تذكر الأسلوب الفني فهو يُضاف تلقائيًا.\n"
            "لا تجسّد نبيًا ولا صحابيًا ولا ملاكًا، ولا تضع أرقامًا أو نصوصًا داخل الرسم، ولا تضف حقائق ليست في التفسير.\n\n"
            "أعد JSON وفق المخطط فقط.\n\n"
            f"{topic_line}"
            f"الآية الأصلية بالرسم العثماني (للسياق فقط، لا تعِد كتابتها أو تعديلها): {verse.text_uthmani}\n"
            f"نص التفسير الموثق (المصدر الوحيد للمعنى الديني): {tafsir.text}"
        )

    def create_and_review(self, verse: QuranVerse, tafsir: TafsirSource, topic: str | None = None,
                          message_words: tuple[int, int] = (40, 70)) -> dict[str, Any]:
        """Generate the video copy, validate it and have it reviewed against the tafsir.

        A copy that breaks a rule or is rejected by the review is regenerated up to CONTENT_ATTEMPTS
        times; API failures are not retried here (`_generate_json` handles transient ones).
        """
        prompt = self._generation_prompt(verse, tafsir, topic, message_words)
        # Output budgets are generous because "thinking" models may spend part of the budget
        # on reasoning tokens, which would otherwise truncate the JSON.
        for attempt in range(1, CONTENT_ATTEMPTS + 1):
            try:
                generated = validate_generated_content(
                    self._generate_json(prompt, GENERATION_SCHEMA, 8192),
                    max_message_words=message_words[1] + 20)
                review = self._review(verse, tafsir, generated)
                generated["review"] = review
                generated["model"] = self.model
                return generated
            except ContentValidationError as exc:
                print(f"محتوى Gemini المحاولة {attempt}/{CONTENT_ATTEMPTS} مرفوض: {str(exc)[:300]}", file=sys.stderr)
                if attempt == CONTENT_ATTEMPTS:
                    raise
        raise ServiceError("فشل إنشاء المحتوى.")  # pragma: no cover - loop always returns or raises

    def _review(self, verse: QuranVerse, tafsir: TafsirSource, generated: dict[str, Any]) -> dict[str, Any]:
        reviewed = {key: generated[key] for key in ("hook", "message", "call_to_action", "title", "description")}
        review_prompt = (
            "راجع النص المقترح بمقارنة ادعاءاته الدينية حصريًا بالنص التفسيري المرفق وبالآية الأصلية. أنت لست مفتيًا. "
            "اللغة العاطفية والتشويقية في الهوك والعنوان ودعوة المشاركة أسلوبٌ مقبول، فلا ترفض النص بسببه. "
            "ارفض النص أو اجعل needs_review=true فقط إذا: ورد ادعاء ديني أو تفسيري غير موجود بوضوح في المصدر، "
            "أو يناقضه، أو ضُمن أجر أو نتيجة معينة، أو نُسبت للآية رسالة شخصية غيبية، أو ذُكر حديث أو حكم شرعي. "
            "أعد JSON فقط حسب المخطط.\n"
            f"الآية العثمانية: {verse.text_uthmani}\nالتفسير الأصلي: {tafsir.text}\n"
            f"النص المراد مراجعته: {json.dumps(reviewed, ensure_ascii=False)}"
        )
        review = self._generate_json(review_prompt, REVIEW_SCHEMA, 2048)
        for key in ("grounded_in_source", "no_unsupported_claims", "no_contradiction", "needs_review"):
            if not isinstance(review.get(key), bool):
                raise ServiceError("استجابة مراجعة Gemini تحتوي حقولًا منطقية غير صالحة.")
        if not isinstance(review.get("reason"), str):
            raise ServiceError("استجابة المراجعة تفتقر إلى سبب نصي.")
        if review["needs_review"] or not all(review[k] for k in ("grounded_in_source", "no_unsupported_claims", "no_contradiction")):
            raise ContentValidationError("رفضت مراجعة المحتوى أو طلبت مراجعة بشرية: " + review["reason"][:240])
        return review


def _validate_tts_texts(texts: dict[str, str]) -> None:
    """Shared by every TTS provider: the closing phrase is its own segment and appears exactly once."""
    forbidden = "صدق الله العظيم"
    if texts.get("seal") != forbidden or sum(value.count(forbidden) for value in texts.values() if isinstance(value, str)) != 1:
        raise ServiceError("يجب إنشاء «صدق الله العظيم» مرة واحدة بالضبط في المقطع المستقل seal.")
    for key, value in texts.items():
        if not isinstance(value, str) or not value.strip() or (key != "seal" and forbidden in value):
            raise ServiceError(f"نص TTS {key} فارغ أو يكرر العبارة المحجوزة.")


class TTSService:
    """Speech for commentary only (edge-tts). Quran recitation must always remain source audio."""
    retry_delays = TTS_RETRY_DELAYS
    provider = "edge-tts"
    _sleep = staticmethod(time.sleep)

    def __init__(self, voice: str | None = None):
        requested = (voice or os.getenv("TTS_VOICE", "ar-EG-ShakirNeural")).strip()
        if not requested:
            raise ServiceError("TTS_VOICE فارغ.")
        corrected = TTS_VOICE_ALIASES.get(requested)
        if corrected:
            print(f"تنبيه: اسم الصوت {requested} غير صحيح، سيُستخدم {corrected}. حدّث المتغير TTS_VOICE.",
                  file=sys.stderr)
            requested = corrected
        self.voice = requested
        if not env_bool("TTS_TERMS_CONFIRMED"):
            raise ServiceError("أكد حقوق الاستخدام التجاري لمزوّد TTS عبر TTS_TERMS_CONFIRMED=true.")
        self.terms_url = _required_env("TTS_LICENSE_URL")
        if urlparse(self.terms_url).scheme != "https":
            raise ServiceError("TTS_LICENSE_URL يجب أن يكون HTTPS.")
        self._voice_checked = False
        # Raw per-word timings reported by the TTS service, by segment name (may be empty).
        self.word_boundaries: dict[str, list[dict[str, Any]]] = {}
        self._last_boundaries: list[dict[str, Any]] = []

    @staticmethod
    def _make_communicate(edge_tts_module: Any, text: str, voice: str) -> Any:
        try:
            return edge_tts_module.Communicate(text=text, voice=voice, rate="+0%", pitch="+0Hz",
                                               boundary="WordBoundary")
        except TypeError:
            # Older edge-tts releases have no `boundary` option and emit word boundaries by default.
            return edge_tts_module.Communicate(text=text, voice=voice, rate="+0%", pitch="+0Hz")

    async def _save(self, text: str, path: Path) -> None:
        import edge_tts
        if not getattr(self, "_voice_checked", False):
            voices = await edge_tts.list_voices()
            available = {item.get("ShortName") for item in voices}
            if self.voice not in available:
                arabic = sorted(name for name in available if isinstance(name, str) and name.startswith("ar-"))
                listing = "، ".join(arabic) if arabic else "لا توجد أصوات عربية في القائمة"
                raise ServiceError(
                    f"صوت TTS_VOICE غير متاح حاليًا: {self.voice}. الأصوات العربية المتاحة: {listing}."
                )
            self._voice_checked = True
        communicate = self._make_communicate(edge_tts, text, self.voice)
        boundaries: list[dict[str, Any]] = []
        with Path(path).open("wb") as handle:
            async for chunk in communicate.stream():
                kind = chunk.get("type")
                if kind == "audio":
                    handle.write(chunk["data"])
                elif kind == "WordBoundary":
                    # edge-tts reports offset and duration in 100-nanosecond ticks.
                    start_ms = int(round(float(chunk["offset"]) / 10_000.0))
                    end_ms = start_ms + int(round(float(chunk["duration"]) / 10_000.0))
                    boundaries.append({"text": str(chunk.get("text", "")), "start_ms": start_ms, "end_ms": end_ms})
        self._last_boundaries = boundaries

    def synthesize(self, texts: dict[str, str], output_dir: Path) -> dict[str, Path]:
        _validate_tts_texts(texts)
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        self.word_boundaries = {}
        attempts = len(self.retry_delays) + 1
        for name, text in texts.items():
            path = output_dir / f"tts-{name}.mp3"
            for attempt in range(1, attempts + 1):
                try:
                    self._last_boundaries = []
                    asyncio.run(self._save(text, path))
                    self.word_boundaries[name] = list(getattr(self, "_last_boundaries", None) or [])
                    break
                except ServiceError:
                    path.unlink(missing_ok=True)
                    raise
                except Exception as exc:
                    path.unlink(missing_ok=True)
                    if attempt == attempts:
                        raise ServiceError(
                            f"فشل توليد TTS للمقطع {name} ({type(exc).__name__}) بعد {attempts} محاولات."
                        ) from None
                    print(f"TTS المقطع {name}: المحاولة {attempt}/{attempts} فشلت ({type(exc).__name__}).",
                          file=sys.stderr)
                    self._sleep(self.retry_delays[attempt - 1])
            if not path.is_file() or path.stat().st_size < 512:
                path.unlink(missing_ok=True)
                raise ServiceError(f"ملف TTS للمقطع {name} غير صالح.")
            results[name] = path
        return results


def _gemini_tts_key() -> str:
    """Gemini TTS may use its own key (GEMINI_API_KEY1) so its quota is separate from text generation."""
    for name in ("GEMINI_API_KEY1", "GEMINI_API_KEY"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    raise ServiceError("مفتاح Gemini TTS غير مضبوط: اضبط GEMINI_API_KEY1 أو GEMINI_API_KEY.")


def _parse_pcm_mime(mime_type: str) -> dict[str, int]:
    """Bits per sample and sample rate from a mime type such as `audio/L16;codec=pcm;rate=24000`."""
    bits_per_sample, rate = 16, 24000
    for part in (mime_type or "").split(";"):
        part = part.strip()
        if part.lower().startswith("rate="):
            try:
                rate = int(part.split("=", 1)[1])
            except (ValueError, IndexError):
                pass
        elif part.lower().startswith("audio/l"):
            try:
                bits_per_sample = int(part.lower().split("l", 1)[1])
            except (ValueError, IndexError):
                pass
    return {"bits_per_sample": bits_per_sample, "rate": rate}


def _pcm_to_wav(data: bytes, mime_type: str) -> bytes:
    """Wrap raw little-endian PCM (what Gemini TTS returns) into a WAV container; WAV passes through."""
    if data[:4] == b"RIFF":
        return data
    if not (mime_type or "").lower().startswith("audio/"):
        raise ServiceError(f"نوع ملف صوت Gemini TTS غير متوقع: {mime_type or 'غير محدد'}.")
    params = _parse_pcm_mime(mime_type)
    if params["bits_per_sample"] != 16 or not 8000 <= params["rate"] <= 96000:
        raise ServiceError(f"صيغة PCM غير مدعومة: {mime_type}.")
    data = data[: len(data) // 2 * 2]
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(params["rate"])
        writer.writeframes(data)
    return buffer.getvalue()


def _wav_stats(wav: bytes) -> tuple[float, int]:
    """(duration in seconds, peak absolute 16-bit sample) of a WAV file."""
    try:
        with wave.open(io.BytesIO(wav), "rb") as reader:
            frames, rate = reader.getnframes(), reader.getframerate()
            width = reader.getsampwidth()
            raw = reader.readframes(frames)
    except (wave.Error, EOFError):
        raise _GeminiTransient("ملف WAV الناتج تالف.") from None
    if frames <= 0 or rate <= 0:
        raise _GeminiTransient("ملف WAV الناتج فارغ.")
    peak = 32767
    if width == 2:
        samples = array("h")
        samples.frombytes(raw[: len(raw) // 2 * 2])
        if sys.byteorder == "big":
            samples.byteswap()
        peak = max(map(abs, samples), default=0)
    return frames / rate, peak


class GeminiTTSService:
    """Commentary speech from the Gemini API (same interface as `TTSService`).

    The API returns raw PCM without word timings, so `word_boundaries` stays empty and the video
    engine falls back to its length-weighted estimate (reported as "estimated" in the checks).
    Each segment is validated for plausible duration and non-silence, and retried on failure.
    """
    retry_delays = GEMINI_TTS_RETRY_DELAYS
    _sleep = staticmethod(time.sleep)

    def __init__(self, voice: str | None = None, model: str | None = None):
        self.api_key = _gemini_tts_key()
        self.model = (model or os.getenv("GEMINI_TTS_MODEL", "").strip() or DEFAULT_GEMINI_TTS_MODEL).strip()
        self.voice = (voice or os.getenv("GEMINI_TTS_VOICE", "").strip() or DEFAULT_GEMINI_TTS_VOICE).strip()
        if not re.fullmatch(r"[a-z0-9][a-z0-9.\-]{2,60}", self.model):
            raise ServiceError("GEMINI_TTS_MODEL غير صالح.")
        if not re.fullmatch(r"[A-Za-z]{3,30}", self.voice):
            raise ServiceError("GEMINI_TTS_VOICE غير صالح؛ استخدم اسم صوت جاهز مثل Algieba.")
        if not env_bool("TTS_TERMS_CONFIRMED"):
            raise ServiceError("أكد حقوق الاستخدام التجاري لمزوّد TTS عبر TTS_TERMS_CONFIRMED=true.")
        self.terms_url = os.getenv("GEMINI_TTS_TERMS_URL", "").strip() or GEMINI_TTS_TERMS
        if urlparse(self.terms_url).scheme != "https":
            raise ServiceError("GEMINI_TTS_TERMS_URL يجب أن يكون HTTPS.")
        self.provider = f"gemini-tts ({self.model})"
        self.word_boundaries: dict[str, list[dict[str, Any]]] = {}

    def _prompt(self, text: str) -> str:
        """Gemini 2.5 TTS follows a director's-note prompt; later models speak the text verbatim."""
        if self.model.startswith("gemini-2.5"):
            style = os.getenv("GEMINI_TTS_STYLE", "").strip() or (
                "Calm, respectful and clear. Pace: natural and unhurried. Language: Modern Standard Arabic.")
            return (
                "Read the following transcript aloud exactly as written, word for word, without adding, "
                "removing, translating or explaining anything.\n\n"
                "# Audio Profile\nA calm, respectful Arabic narrator.\n\n"
                f"# Director's note\n{style}\n\n## Transcript:\n{text}"
            )
        return text

    @staticmethod
    def _extract_audio(response: Any) -> tuple[bytes, str]:
        try:
            parts = response.candidates[0].content.parts or []
        except Exception:
            parts = []
        data = bytearray()
        mime = ""
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                data.extend(inline.data)
                mime = getattr(inline, "mime_type", "") or mime
        return bytes(data), mime

    def _request_audio(self, text: str) -> tuple[bytes, str]:
        """One Gemini TTS call. Raises _GeminiTransient for retryable failures, ServiceError otherwise."""
        try:
            from google import genai
            from google.genai import types
        except Exception as exc:
            raise ServiceError(f"تعذر استيراد مكتبة Gemini ({type(exc).__name__}).") from None
        try:
            client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=GEMINI_REQUEST_TIMEOUT_MS),
            )
            response = client.models.generate_content(
                model=self.model,
                contents=self._prompt(text),
                config=types.GenerateContentConfig(
                    temperature=1.0,
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice)
                        )
                    ),
                ),
            )
        except Exception as exc:
            retryable, code, status = _gemini_exception_info(exc)
            detail = _describe_gemini_exception(exc, self.api_key, code, status)
            if retryable:
                raise _GeminiTransient(detail) from None
            raise ServiceError(f"فشل استدعاء Gemini TTS دون قابلية لإعادة المحاولة ({detail}).") from None
        data, mime = self._extract_audio(response)
        if not data:
            raise _GeminiTransient(f"استجابة TTS بلا صوت (finish_reason={_gemini_finish_reason(response)})")
        return data, mime

    @staticmethod
    def _check_plausible(name: str, text: str, seconds: float, peak: int) -> None:
        words = max(1, len(text.split()))
        pace = seconds / words
        if seconds < TTS_MIN_SECONDS or pace < TTS_MIN_SECONDS_PER_WORD:
            raise _GeminiTransient(f"المقطع {name} أقصر من المعقول ({seconds:.2f}s لـ{words} كلمات)")
        if pace > TTS_MAX_SECONDS_PER_WORD:
            raise _GeminiTransient(f"المقطع {name} أطول من المعقول ({seconds:.2f}s لـ{words} كلمات)")
        if peak < TTS_MIN_PEAK:
            raise _GeminiTransient(f"المقطع {name} صامت تقريبًا (peak={peak})")

    def synthesize(self, texts: dict[str, str], output_dir: Path) -> dict[str, Path]:
        _validate_tts_texts(texts)
        output_dir.mkdir(parents=True, exist_ok=True)
        results: dict[str, Path] = {}
        self.word_boundaries = {}
        attempts = len(self.retry_delays) + 1
        for name, text in texts.items():
            path = output_dir / f"tts-{name}.wav"
            for attempt in range(1, attempts + 1):
                try:
                    data, mime = self._request_audio(text)
                    wav = _pcm_to_wav(data, mime)
                    seconds, peak = _wav_stats(wav)
                    self._check_plausible(name, text, seconds, peak)
                    path.write_bytes(wav)
                    self.word_boundaries[name] = []
                    break
                except _GeminiTransient as exc:
                    print(f"Gemini TTS المقطع {name}: المحاولة {attempt}/{attempts} فشلت (مؤقت): {exc}",
                          file=sys.stderr)
                    if attempt == attempts:
                        raise ServiceError(
                            f"فشل توليد Gemini TTS للمقطع {name} بعد {attempts} محاولات: {exc}") from None
                    self._sleep(self.retry_delays[attempt - 1])
            if not path.is_file() or path.stat().st_size < 512:
                path.unlink(missing_ok=True)
                raise ServiceError(f"ملف TTS للمقطع {name} غير صالح.")
            results[name] = path
        return results


class FallbackTTSService:
    """Try the primary provider for every segment; on failure redo ALL segments with the secondary.

    Switching per segment would mix two different voices inside one video, so the fallback is all-or-nothing.
    `voice`, `terms_url`, `provider` and `word_boundaries` always describe the provider that produced the audio.
    """
    def __init__(self, primary: Any, secondary: Any):
        self.primary, self.secondary = primary, secondary
        self._active = primary
        self.word_boundaries: dict[str, list[dict[str, Any]]] = {}

    @property
    def voice(self) -> str:
        return self._active.voice

    @property
    def terms_url(self) -> str:
        return self._active.terms_url

    @property
    def provider(self) -> str:
        return getattr(self._active, "provider", "tts")

    def synthesize(self, texts: dict[str, str], output_dir: Path) -> dict[str, Path]:
        try:
            files = self.primary.synthesize(texts, output_dir)
            self._active = self.primary
        except ServiceError as exc:
            print(f"تنبيه: فشل مزوّد TTS الأساسي ({getattr(self.primary, 'provider', 'primary')}): "
                  f"{str(exc)[:300]} — سيُستخدم المزوّد الاحتياطي.", file=sys.stderr)
            files = self.secondary.synthesize(texts, output_dir)
            self._active = self.secondary
        self.word_boundaries = self._active.word_boundaries
        return files


def make_tts_service(edge_voice: str | None = None) -> Any:
    """Pick the commentary voice provider from TTS_PROVIDER: edge (default), gemini, or auto.

    `auto` uses Gemini first and falls back to edge-tts if Gemini is unconfigured or fails.
    """
    provider = os.getenv("TTS_PROVIDER", "").strip().lower() or "edge"
    if provider not in TTS_PROVIDERS:
        raise ServiceError("TTS_PROVIDER يجب أن يكون edge أو gemini أو auto.")
    if provider == "edge":
        return TTSService(edge_voice)
    if provider == "gemini":
        return GeminiTTSService()
    try:
        primary = GeminiTTSService()
    except ServiceError as exc:
        print(f"تنبيه: تعذر تهيئة Gemini TTS ({str(exc)[:200]}) — سيُستخدم edge-tts.", file=sys.stderr)
        return TTSService(edge_voice)
    return FallbackTTSService(primary, TTSService(edge_voice))


def _youtube_client() -> Any:
    client_id = _required_env("YOUTUBE_CLIENT_ID")
    client_secret = _required_env("YOUTUBE_CLIENT_SECRET")
    refresh_token = _required_env("YOUTUBE_REFRESH_TOKEN")
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        credentials = Credentials(
            token=None, refresh_token=refresh_token, token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id, client_secret=client_secret,
            scopes=[YOUTUBE_UPLOAD_SCOPE, YOUTUBE_READ_SCOPE],
        )
        credentials.refresh(Request())
        return build("youtube", "v3", credentials=credentials, cache_discovery=False)
    except Exception as exc:
        raise ServiceError(f"تعذر تهيئة YouTube API OAuth ({type(exc).__name__}).") from None


class YouTubePublisher:
    def __init__(self, privacy_status: str | None = None):
        self.privacy_status = (privacy_status or os.getenv("YOUTUBE_PRIVACY_STATUS", "private")).strip()
        if self.privacy_status not in {"public", "unlisted", "private"}:
            raise ServiceError("YOUTUBE_PRIVACY_STATUS غير صالح.")

    def _verify(self, youtube: Any, video_id: str, timeout_seconds: int = 180) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        latest = {}
        while time.monotonic() < deadline:
            try:
                response = youtube.videos().list(part="status,processingDetails", id=video_id).execute()
            except Exception as exc:
                raise ServiceError(f"تعذر التحقق من حالة YouTube ({type(exc).__name__}).") from None
            items = response.get("items", []) if isinstance(response, dict) else []
            if not items:
                raise ServiceError("لم يؤكد YouTube وجود الفيديو المرفوع.")
            latest = items[0]
            status = latest.get("status", {})
            processing = latest.get("processingDetails", {}).get("processingStatus")
            upload = status.get("uploadStatus")
            if upload in {"failed", "rejected", "deleted"} or processing == "failed":
                raise ServiceError(f"فشل YouTube في معالجة الفيديو: uploadStatus={upload}, processing={processing}.")
            if processing == "succeeded" and upload in {"processed", "uploaded"}:
                if status.get("privacyStatus") != self.privacy_status:
                    raise ServiceError("اكتملت المعالجة لكن خصوصية الفيديو في YouTube لا تطابق الإعداد المطلوب.")
                return {"status": "published", "video_id": video_id, "privacy": status.get("privacyStatus"), "processing": processing}
            time.sleep(5)
        return {"status": "pending", "video_id": video_id, "reason": "YouTube processing is not yet confirmed"}

    def publish_or_verify(self, video_path: Path, title: str, description: str,
                          previous: dict[str, Any] | None = None) -> dict[str, Any]:
        previous = previous or {}
        if previous.get("status") == "published" and previous.get("video_id"):
            return dict(previous)
        if previous.get("status") == "upload_uncertain" and not previous.get("video_id"):
            raise ServiceError("حالة رفع YouTube السابقة غير محسومة ولا يوجد معرّف؛ لن يعاد الرفع منعًا للتكرار.",
                               platform_state=previous)
        youtube = _youtube_client()
        if previous.get("video_id"):
            video_id = str(previous["video_id"])
            try:
                return self._verify(youtube, video_id)
            except ServiceError as exc:
                raise ServiceError(str(exc), platform_state={"status": "uploaded_unconfirmed", "video_id": video_id}) from None
        if not video_path.is_file():
            raise ServiceError("ملف الفيديو غير موجود لرفعه إلى YouTube.")
        try:
            from googleapiclient.http import MediaFileUpload
            request = youtube.videos().insert(
                part="snippet,status",
                body={"snippet": {"title": title[:100], "description": description[:5000], "categoryId": "27"},
                      "status": {"privacyStatus": self.privacy_status, "selfDeclaredMadeForKids": False}},
                media_body=MediaFileUpload(str(video_path), mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True),
            )
            response = None
            while response is None:
                _, response = request.next_chunk()
            video_id = response.get("id") if isinstance(response, dict) else None
            if not video_id:
                raise ServiceError("لم يُرجع YouTube معرّف الفيديو بعد الرفع.")
        except ServiceError:
            raise ServiceError("لم يؤكد YouTube معرّف الفيديو بعد الرفع؛ تحقق يدويًا قبل إعادة المحاولة.",
                               platform_state={"status": "upload_uncertain"}) from None
        except Exception as exc:
            raise ServiceError(f"فشل رفع الفيديو إلى YouTube ({type(exc).__name__})؛ النتيجة غير محسومة.",
                               platform_state={"status": "upload_uncertain"}) from None
        try:
            return self._verify(youtube, str(video_id))
        except ServiceError as exc:
            raise ServiceError(str(exc), platform_state={"status": "uploaded_unconfirmed", "video_id": str(video_id)}) from None


class FacebookReelsPublisher:
    def __init__(self, session: requests.Session | None = None):
        if not env_bool("META_REELS_DISCLOSURE_CONFIRMED"):
            raise ServiceError("يلزم تهيئة الإفصاح وخيارات التحكم التي يطلبها دليل Meta Reels قبل النشر.")
        self.token = _required_env("META_PAGE_ACCESS_TOKEN")
        self.page_id = _required_env("META_PAGE_ID")
        self.version = os.getenv("META_GRAPH_VERSION", "v26.0").strip()
        if not re.fullmatch(r"v\d+\.0", self.version):
            raise ServiceError("META_GRAPH_VERSION يجب أن يكون بالشكل vNN.0.")
        self.session = session or requests.Session()
        self.root = f"https://graph.facebook.com/{self.version}"

    def _api(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self.session.request(method, url, timeout=(20, 60), **kwargs)
        except requests.RequestException as exc:
            raise ServiceError(f"تعذر الاتصال بـMeta Graph API ({type(exc).__name__}).") from None
        if response.status_code < 200 or response.status_code >= 300:
            raise ServiceError(f"Meta Graph API أعاد HTTP {response.status_code}.")
        try:
            payload = response.json()
        except ValueError:
            raise ServiceError("استجابة Meta ليست JSON صالحًا.") from None
        if not isinstance(payload, dict):
            raise ServiceError("بنية استجابة Meta غير صالحة.")
        if payload.get("error"):
            error = payload["error"] if isinstance(payload["error"], dict) else {}
            raise ServiceError(f"رفض Meta الطلب (code={error.get('code', 'unknown')}).")
        return payload

    def _verify(self, video_id: str, timeout_seconds: int = 180) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            status_data = self._api("GET", f"{self.root}/{video_id}",
                                    params={"fields": "status", "access_token": self.token})
            status = status_data.get("status", {})
            if not isinstance(status, dict):
                raise ServiceError("حقل status من Meta غير صالح.")
            phases = [status.get("uploading_phase", {}), status.get("processing_phase", {}), status.get("publishing_phase", {})]
            phase_values = [phase.get("status") for phase in phases if isinstance(phase, dict)]
            bad = {str(value).lower() for value in phase_values} & {"error", "failed", "expired"}
            if bad or str(status.get("video_status", "")).lower() in {"error", "failed"}:
                raise ServiceError("أبلغت Meta عن فشل معالجة أو نشر Reel.")
            listed = self._api("GET", f"{self.root}/{self.page_id}/video_reels",
                               params={"fields": "id", "limit": 100, "access_token": self.token})
            data = listed.get("data", [])
            visible = isinstance(data, list) and any(isinstance(row, dict) and str(row.get("id")) == video_id for row in data)
            completed = all(isinstance(phase, dict) and str(phase.get("status", "")).lower() == "complete" for phase in phases)
            if visible and completed:
                return {"status": "published", "video_id": video_id, "phase_status": phase_values}
            time.sleep(5)
        return {"status": "pending", "video_id": video_id, "reason": "Meta processing/public listing is not yet confirmed"}

    def publish_or_verify(self, video_path: Path, title: str, description: str,
                          previous: dict[str, Any] | None = None) -> dict[str, Any]:
        previous = previous or {}
        if previous.get("status") == "published" and previous.get("video_id"):
            return dict(previous)
        if previous.get("video_id"):
            return self._verify(str(previous["video_id"]))
        if previous.get("status") in {"upload_session_uncertain", "upload_uncertain"}:
            raise ServiceError("جلسة رفع Meta السابقة غير محسومة ولا يوجد معرّف؛ لن يبدأ رفع جديد منعًا للتكرار.",
                               platform_state=previous)
        if not video_path.is_file():
            raise ServiceError("ملف الفيديو غير موجود لرفعه إلى Facebook Reels.")
        try:
            start = self._api("POST", f"{self.root}/{self.page_id}/video_reels",
                              data={"upload_phase": "start", "access_token": self.token})
        except ServiceError as exc:
            raise ServiceError(str(exc), platform_state={"status": "upload_session_uncertain"}) from None
        video_id, upload_url = start.get("video_id"), start.get("upload_url")
        if not video_id or not isinstance(upload_url, str) or not upload_url.startswith("https://"):
            raise ServiceError("لم تُرجع Meta video_id وupload_url صالحين لبدء الرفع.",
                               platform_state={"status": "upload_session_uncertain", **({"video_id": str(video_id)} if video_id else {})})
        if urlparse(upload_url).hostname != "rupload.facebook.com":
            raise ServiceError("مضيف رفع Reel لا يطابق مضيف Meta الرسمي.",
                               platform_state={"status": "upload_session_created", "video_id": str(video_id)})
        size = video_path.stat().st_size
        try:
            with video_path.open("rb") as stream:
                upload_response = self.session.post(
                    upload_url,
                    headers={"Authorization": f"OAuth {self.token}", "offset": "0", "file_size": str(size),
                             "Content-Type": "application/octet-stream"},
                    data=stream, timeout=(30, 600),
                )
            if upload_response.status_code < 200 or upload_response.status_code >= 300:
                raise ServiceError(f"رفع ملف Reel إلى Meta أعاد HTTP {upload_response.status_code}.",
                                   platform_state={"status": "uploading_unknown", "video_id": str(video_id)})
        except ServiceError as exc:
            if exc.platform_state:
                raise
            raise ServiceError(str(exc), platform_state={"status": "uploading_unknown", "video_id": str(video_id)}) from None
        except requests.RequestException as exc:
            # The created ID is not discarded; however, the caller must persist an uncertain state
            # rather than creating a second upload on retry.
            raise ServiceError(f"تعذر رفع ملف Reel إلى Meta ({type(exc).__name__}); تحقق من سجل المنصة قبل إعادة المحاولة.",
                               platform_state={"status": "uploading_unknown", "video_id": str(video_id)}) from None
        try:
            finish = self._api("POST", f"{self.root}/{self.page_id}/video_reels", data={
                "video_id": str(video_id), "upload_phase": "finish", "video_state": "PUBLISHED",
                "title": title[:255], "description": description[:5000], "access_token": self.token,
            })
        except ServiceError as exc:
            raise ServiceError(str(exc), platform_state={"status": "uploaded_unconfirmed", "video_id": str(video_id)}) from None
        if finish.get("success") is not True:
            raise ServiceError("لم تؤكد Meta نجاح مرحلة finish والنشر.",
                               platform_state={"status": "uploaded_unconfirmed", "video_id": str(video_id)})
        try:
            return self._verify(str(video_id))
        except ServiceError as exc:
            raise ServiceError(str(exc), platform_state={"status": "uploaded_unconfirmed", "video_id": str(video_id)}) from None


class MakeWebhookPublisher:
    """Uploads the MP4 to a public GitHub Release and notifies a Make.com webhook.

    Make fans out to Facebook / YouTube / Instagram / TikTok. A 2xx answer proves delivery to Make,
    not publication on each platform (that is Make's job and is visible in its execution history).
    The release tag derives from the file hash, so retries reuse the same release and the webhook
    carries a stable `idempotency_key` that the Make scenario can use to ignore duplicates.
    """
    _sleep = staticmethod(time.sleep)

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()
        self.webhook = _required_env("MAKE_WEBHOOK_URL")
        parsed = urlparse(self.webhook)
        host = parsed.hostname or ""
        if parsed.scheme != "https" or not any(host.endswith(suffix) for suffix in MAKE_ALLOWED_SUFFIXES):
            raise ServiceError("MAKE_WEBHOOK_URL يجب أن يكون رابط HTTPS على نطاق make.com.")
        self.api_key = os.getenv("MAKE_WEBHOOK_API_KEY", "").strip()
        self.token = _required_env("GITHUB_TOKEN")
        self.repo = _required_env("GITHUB_REPOSITORY")
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", self.repo):
            raise ServiceError("GITHUB_REPOSITORY غير صالح.")
        self.platforms = [part.strip() for part in os.getenv(
            "MAKE_PLATFORMS", "facebook,youtube,instagram,tiktok").split(",") if part.strip()]
        self.retry_delay = 5.0

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _gh(self, method: str, url: str, **kwargs: Any) -> Any:
        headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                   "X-GitHub-Api-Version": "2022-11-28", **kwargs.pop("headers", {})}
        try:
            return self.session.request(method, url, headers=headers, timeout=(20, 600), **kwargs)
        except requests.RequestException as exc:
            raise ServiceError(f"تعذر الاتصال بـGitHub ({type(exc).__name__}).") from None

    @staticmethod
    def _uploaded_asset_url(release: dict[str, Any], asset_name: str) -> str | None:
        for asset in release.get("assets", []) or []:
            if isinstance(asset, dict) and asset.get("name") == asset_name and asset.get("state") == "uploaded":
                url = asset.get("browser_download_url")
                if isinstance(url, str) and url.startswith("https://"):
                    return url
        return None

    def _ensure_asset(self, video_path: Path, tag: str, title: str) -> str:
        """Idempotent: the same tag/asset is reused when it already exists."""
        asset_name = f"{tag}.mp4"
        base = f"{GITHUB_API}/repos/{self.repo}/releases"
        response = self._gh("GET", f"{base}/tags/{tag}")
        if response.status_code == 404:
            response = self._gh("POST", base, json={"tag_name": tag, "name": title[:100],
                                                     "body": "Automated video release.",
                                                     "draft": False, "prerelease": False})
        if response.status_code not in (200, 201):
            raise ServiceError(f"GitHub Release: HTTP {response.status_code} (تحقق من permissions: contents: write).")
        release = response.json()
        existing = self._uploaded_asset_url(release, asset_name)
        if existing:
            return existing
        upload_url = str(release.get("upload_url", "")).split("{", 1)[0]
        if urlparse(upload_url).hostname != "uploads.github.com":
            raise ServiceError("رابط رفع GitHub غير متوقع.")
        with video_path.open("rb") as stream:
            uploaded = self._gh("POST", upload_url, params={"name": asset_name}, data=stream,
                                headers={"Content-Type": "video/mp4"})
        if uploaded.status_code == 422:
            # Another run may have uploaded the same asset in the meantime.
            again = self._gh("GET", f"{base}/tags/{tag}")
            if again.status_code == 200:
                existing = self._uploaded_asset_url(again.json(), asset_name)
                if existing:
                    return existing
        if uploaded.status_code not in (200, 201):
            raise ServiceError(f"رفع الفيديو إلى GitHub Release أعاد HTTP {uploaded.status_code}.")
        url = uploaded.json().get("browser_download_url")
        if not isinstance(url, str) or not url.startswith("https://"):
            raise ServiceError("لم يُرجع GitHub رابط تنزيل صالحًا للملف المرفوع.")
        return url

    def _notify(self, payload: dict[str, Any]) -> None:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["x-make-apikey"] = self.api_key
        last = "unknown"
        for attempt in range(3):
            try:
                response = self.session.post(self.webhook, json=payload, headers=headers, timeout=(15, 60))
            except requests.RequestException as exc:
                last = type(exc).__name__
            else:
                if 200 <= response.status_code < 300:
                    return
                last = f"HTTP {response.status_code}"
                if response.status_code not in (408, 429) and response.status_code < 500:
                    break
            if attempt < 2:
                self._sleep(self.retry_delay)
        raise ServiceError(f"فشل إرسال Webhook إلى Make ({last}).")

    def publish_or_verify(self, video_path: Path, title: str, description: str,
                          previous: dict[str, Any] | None = None) -> dict[str, Any]:
        previous = previous or {}
        if previous.get("status") == "published" and previous.get("video_id"):
            return dict(previous)
        resume_url = previous.get("video_url") if isinstance(previous.get("video_url"), str) else None
        if not resume_url and not video_path.is_file():
            raise ServiceError("ملف الفيديو غير موجود لرفعه.")
        digest = previous.get("sha256") or self._sha256(video_path)
        tag = previous.get("video_id") or f"video-{digest[:16]}"
        url = resume_url or self._ensure_asset(video_path, str(tag), title)
        state = {"status": "uploaded_unconfirmed", "video_id": str(tag), "video_url": url, "sha256": digest}
        payload = {
            "idempotency_key": str(tag),
            "video_url": url,
            "sha256": digest,
            "title": title[:100],
            "description": description[:5000],
            "caption": f"{title}\n\n{description}"[:2000],     # Instagram/TikTok caption limit
            "platforms": self.platforms,
            "youtube_privacy": os.getenv("YOUTUBE_PRIVACY_STATUS", "private"),
        }
        try:
            self._notify(payload)
        except ServiceError as exc:
            raise ServiceError(str(exc), platform_state=state) from None
        return {"status": "published", "video_id": str(tag), "video_url": url, "sha256": digest,
                "delivered_to": "make.com"}


# Self-contained unit tests; external APIs are mocked and never called.
if __name__ == "__main__":
    import unittest
    from unittest.mock import patch

    class ServiceValidationTests(unittest.TestCase):
        class FakeResponse:
            def __init__(self, status_code, payload):
                self.status_code, self.payload = status_code, payload
            def json(self):
                return self.payload

        class FakeSession:
            def __init__(self, response, token_response=None):
                self.response = response
                self.token_response = token_response or ServiceValidationTests.FakeResponse(
                    200, {"access_token": "test-access-token", "expires_in": 3600}
                )
                self.get_calls = []
                self.post_calls = []
            def get(self, *args, **kwargs):
                self.get_calls.append((args, kwargs))
                return self.response
            def post(self, *args, **kwargs):
                self.post_calls.append((args, kwargs))
                return self.token_response

        @classmethod
        def ready_qf_client(cls, response):
            client = object.__new__(QuranFoundationClient)
            client.base_url, client.auth_base_url = QF_API, QF_PRODUCTION_AUTH
            client.session = cls.FakeSession(response)
            client.client_id, client.client_secret = "test-client", "test-secret"
            client.headers = {"x-client-id": "test-client", "x-auth-token": "test-token", "Accept": "application/json"}
            client._access_token_expires_at = float("inf")
            return client

        def test_official_catalog_spellings_of_mishari_are_recognized(self):
            self.assertTrue(_is_mishary_reciter("Mishari Rashid al-`Afasy"))
            self.assertTrue(_is_mishary_reciter("مشاري راشد العفاسي"))
            self.assertFalse(_is_mishary_reciter("AbdulBaset AbdulSamad"))

        def test_quran_verse_response_needs_only_uthmani_text(self):
            payload = {"verse": {"verse_key": "2:152", "verse_number": 152, "chapter_id": 2,
                                 "id": 12345, "text_uthmani": "فَاذْكُرُونِي أَذْكُرْكُمْ"}}
            client = self.ready_qf_client(self.FakeResponse(200, payload))
            verse = client.get_verse(2, 152)
            self.assertEqual(verse.key, "2:152")
            self.assertIn("text_uthmani", verse.__dict__)
            self.assertNotIn("text_imlaei", verse.__dict__)
            self.assertNotIn("text_imlaei", client.session.get_calls[0][1]["params"]["fields"])

        def test_qf_client_credentials_requests_and_caches_access_token(self):
            session = self.FakeSession(self.FakeResponse(200, {"chapters": []}))
            with patch.dict(os.environ, {"QF_CLIENT_ID": "client-id", "QF_CLIENT_SECRET": "client-secret",
                                          "QF_ENV": "production"}, clear=False):
                client = QuranFoundationClient(session=session)
                client._get_json("chapters")
                client._get_json("chapters")
            self.assertEqual(len(session.post_calls), 1)
            args, kwargs = session.post_calls[0]
            self.assertEqual(args[0], f"{QF_PRODUCTION_AUTH}/oauth2/token")
            self.assertEqual(kwargs["auth"], ("client-id", "client-secret"))
            self.assertEqual(kwargs["data"], {"grant_type": "client_credentials", "scope": "content"})
            self.assertEqual(session.get_calls[0][1]["headers"]["x-auth-token"], "test-access-token")

        def test_quran_api_failure_is_not_treated_as_data(self):
            client = self.ready_qf_client(self.FakeResponse(503, {}))
            with self.assertRaises(ServiceError):
                client.get_verse(2, 152)

        def test_chapter_recitation_extracts_ayah_and_offsets_word_segments(self):
            import tempfile
            class DownloadResponse:
                url = "https://download.quranicaudio.com/test/chapter.mp3"
                headers = {"Content-Type": "audio/mpeg"}
                def raise_for_status(self):
                    return None
                def iter_content(self, _size):
                    return [b"x" * 6000]

            class DownloadSession:
                def get(self, *args, **kwargs):
                    return DownloadResponse()

            client = self.ready_qf_client(self.FakeResponse(200, {}))
            client.session = DownloadSession()
            catalog = {"reciters": [{"id": 7, "name": "Mishari Rashid al-`Afasy"}]}
            chapter_data = {"audio_file": {
                "audio_url": "https://download.quranicaudio.com/test/chapter.mp3",
                "timestamps": [{"verse_key": "2:3", "timestamp_from": 10000,
                                "timestamp_to": 13000,
                                "segments": [[1, 9900, 11000], [2, 11000, 12000], [3, 12000, 13100]]}],
            }}
            with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {
                "CHAPTER_RECITER_ID": "7", "RECITATION_LICENSE_URL": "https://example.org/license",
                "RECITATION_LICENSE_CONFIRMED": "true", "RECITATION_ALLOWLIST": "7|https://example.org/license",
                "RECITATION_REQUIRE_MISHARY": "true",
            }):
                with patch.object(client, "_get_json", side_effect=[catalog, chapter_data]) as api_call:
                    def fake_ffmpeg(command, **kwargs):
                        Path(command[-1]).write_bytes(b"y" * 5000)
                    with patch("services.subprocess.run", side_effect=fake_ffmpeg) as ffmpeg:
                        recitation, path = client.get_recitation(2, 3, Path(temp))
                self.assertEqual(api_call.call_args_list[1].args[0], "chapter_recitations/7/2")
                self.assertEqual(recitation.segments, [[1, 200, 1300], [2, 1300, 2300], [3, 2300, 3400]])
                self.assertEqual(recitation.duration_hint, 3600)
                self.assertEqual(len(validate_word_segments(recitation.segments, "الأول الثاني الثالث", 3600)), 3)
                self.assertTrue(path.is_file())
                self.assertIn("9.700", ffmpeg.call_args.args[0])

        def test_word_timings_accept_contiguous_one_based_and_map_to_zero_based(self):
            got = validate_word_segments([[1, 0, 250], [2, 250, 700]], "بسم الله", 800)
            self.assertEqual([x["word_index"] for x in got], [0, 1])

        def test_word_timings_reject_missing_and_bad_rows(self):
            with self.assertRaises(ServiceError):
                validate_word_segments([[0, 0, 100]], "بسم الله", 800)
            with self.assertRaises(ServiceError):
                validate_word_segments([[0, 0, 300], [1, 200, 100]], "بسم الله", 800)

        # ----- Generated video copy -----
        @staticmethod
        def valid_content(**overrides):
            hooks = [{"text": f"تعبك الذي لا يراه أحد يراه الله رقم {i}", **{key: 4 for key in HOOK_SCORE_KEYS}}
                     for i in range(5)]
            prompt = "A lone traveller walking a winding road toward a glowing sunrise over quiet hills"
            data = {
                "hook_candidates": hooks,
                "message": " ".join(f"كلمة{i}" for i in range(45)),
                "call_to_action": "شارك هذا المقطع الآن واجعله صدقة جارية عن والديك ومن تحب",
                "title": "تعبك الذي لا يراه أحد يراه الله وحده",
                "description": "هذه الآية تخاطب قلب كل من أنهكه الصبر. " * 12,
                "keywords": [f"آيات عن الصبر {i}" for i in range(8)],
                "hashtags": ["#الصبر", "#تدبر_القرآن", "#آيات", "#طمأنينة", "#دعاء", "#القرآن_الكريم", "#راحة_نفسية", "#ذكر"],
                "visual_plan": {
                    "opening": {"mood": "night", "image_prompt": prompt, "elements": [
                        {"type": "sitting_person", "position": "center", "animation": "fade_in"},
                        {"type": "moon", "position": "right", "animation": "float"}]},
                    "scenes": [
                        {"mood": "storm", "text_span": "كلمة0 كلمة1 كلمة2", "image_prompt": prompt, "elements": [
                            {"type": "cloud", "position": "left", "animation": "float"},
                            {"type": "rain", "position": "center", "animation": "none"}]},
                        {"mood": "dawn", "text_span": "كلمة15 كلمة16 كلمة17", "image_prompt": prompt, "elements": [
                            {"type": "sun", "position": "right", "animation": "grow"}]},
                        {"mood": "day", "text_span": "كلمة30 كلمة31 كلمة32", "image_prompt": prompt, "elements": [
                            {"type": "seed", "position": "center", "animation": "grow"}]}],
                    "closing": {"mood": "warm", "image_prompt": prompt, "elements": [
                        {"type": "heart", "position": "center", "animation": "pulse"}]}
                }
            }
            data.update(overrides)
            return data

        def test_generated_content_accepts_a_complete_valid_copy(self):
            result = validate_generated_content(self.valid_content())
            self.assertTrue(result["hook"])
            self.assertEqual(len(result["hook_candidates"]), 5)
            self.assertEqual(len(result["hashtags"]), 8)
            self.assertTrue(all(tag.startswith("#") and " " not in tag for tag in result["hashtags"]))
            self.assertEqual(set(result), {"hook", "hook_candidates", "message", "call_to_action", "title",
                                           "description", "keywords", "hashtags", "visual_plan"})

        def test_visual_plan_binds_scenes_to_message_words_in_order(self):
            result = validate_generated_content(self.valid_content())
            self.assertEqual([s["word_start"] for s in result["visual_plan"]["scenes"]], [0, 15, 30])

        def test_visual_plan_rejects_unknown_phrase_wrong_order_and_bad_elements(self):
            good = self.valid_content()["visual_plan"]
            for mutate in (lambda p: p["scenes"][1].update(text_span="عبارة ليست في الرسالة"),
                           lambda p: p["scenes"].reverse(),
                           lambda p: p["scenes"].pop(),
                           lambda p: p["opening"]["elements"][0].update(type="dragon"),
                           lambda p: p["closing"].update(mood="neon")):
                plan = json.loads(json.dumps(good))
                mutate(plan)
                with self.assertRaises(ContentValidationError):
                    validate_generated_content(self.valid_content(visual_plan=plan))

        def test_unsafe_or_missing_image_prompt_only_removes_the_picture(self):
            full = validate_generated_content(self.valid_content())["visual_plan"]
            self.assertTrue(full["opening"]["image_prompt"] and full["closing"]["image_prompt"])
            self.assertTrue(all(s["image_prompt"] for s in full["scenes"]))
            plan = json.loads(json.dumps(self.valid_content()["visual_plan"]))
            plan["scenes"][0]["image_prompt"] = "The prophet standing alone in the desert at dawn with light behind him"
            del plan["scenes"][1]["image_prompt"]
            result = validate_generated_content(self.valid_content(visual_plan=plan))["visual_plan"]
            self.assertEqual((result["scenes"][0]["image_prompt"], result["scenes"][1]["image_prompt"]), ("", ""))
            self.assertTrue(result["scenes"][2]["image_prompt"])

        def test_generation_schema_requires_an_image_prompt_per_scene(self):
            scene = GENERATION_SCHEMA["properties"]["visual_plan"]["properties"]["opening"]
            self.assertIn("image_prompt", scene["required"])
            self.assertIn("image_prompt", scene["properties"])

        def test_generated_content_rejects_rule_violations(self):
            good = self.valid_content()
            def broken(**overrides):
                with self.assertRaises(ContentValidationError, msg=str(list(overrides))):
                    validate_generated_content(self.valid_content(**overrides))
            broken(hook_candidates=good["hook_candidates"][:4])
            broken(message="كلمات قليلة جدا")
            broken(message=" ".join(["كلمة"] * 200))
            broken(call_to_action="هذه عبارة جميلة جدا عن الصبر والأمل في الحياة")      # no share verb
            broken(title="عنوان #وسم هنا قوي جدا")
            broken(title="كلمة " * 30)
            broken(description="وصف قصير")
            broken(description="نص " + "رابط https://example.org " * 40)
            broken(message=good["message"] + " صدق الله العظيم")
            broken(description=good["description"] + " المصدر ابن كثير")
            broken(keywords=["واحدة", "اثنتان"])
            broken(hashtags=["#واحد"])
            same = [dict(good["hook_candidates"][0]) for _ in range(5)]
            broken(hook_candidates=same)
            low = [dict(hook, relevance=2) for hook in good["hook_candidates"]]
            broken(hook_candidates=low)

        def test_generated_content_normalizes_hashtags_and_keywords(self):
            result = validate_generated_content(self.valid_content(
                hashtags=["الصبر", "# تدبر القرآن", "#آيات!", "#آيات", "#راحة_نفسية", "#ذكر", "#دعاء", "#أمل", "#"],
                keywords=["# آيات عن الصبر", "آيات عن الصبر", "تفسير الآية", "تدبر القرآن", "دعاء الفرج", "راحة القلب", "كلمات مؤثرة", "x"]))
            self.assertIn("#الصبر", result["hashtags"])
            self.assertIn("#تدبر_القرآن", result["hashtags"])
            self.assertEqual(result["hashtags"].count("#آيات"), 1)
            self.assertEqual(result["keywords"].count("آيات عن الصبر"), 1)

        def test_content_generation_regenerates_after_a_rule_violation_but_not_after_api_errors(self):
            verse = QuranVerse(2, 3, "2:3", "كلمة " * 12, 1, "البقرة")
            tafsir = TafsirSource("تفسير " * 20, "مصدر", 16, "https://x", "https://l", True)
            review_ok = {"grounded_in_source": True, "no_unsupported_claims": True, "no_contradiction": True,
                         "needs_review": False, "reason": "ok"}
            service = object.__new__(GeminiContentService)
            service.model = "test-model"
            calls = []
            def flaky(prompt, schema, max_tokens):
                calls.append(schema is GENERATION_SCHEMA)
                if schema is REVIEW_SCHEMA:
                    return review_ok
                if calls.count(True) == 1:
                    return self.valid_content(message="قصير")
                return self.valid_content()
            service._generate_json = flaky
            result = service.create_and_review(verse, tafsir, topic="الصبر والابتلاء", message_words=(30, 60))
            self.assertEqual(calls.count(True), 2)
            self.assertEqual(result["review"], review_ok)
            self.assertEqual(result["model"], "test-model")
            def api_down(prompt, schema, max_tokens):
                raise ServiceError("code=404 model not found")
            service._generate_json = api_down
            with self.assertRaisesRegex(ServiceError, "404"):
                service.create_and_review(verse, tafsir)

        def test_content_generation_regenerates_when_the_review_rejects_and_gives_up_after_three(self):
            verse = QuranVerse(2, 3, "2:3", "كلمة " * 12, 1, "البقرة")
            tafsir = TafsirSource("تفسير " * 20, "مصدر", 16, "https://x", "https://l", True)
            service = object.__new__(GeminiContentService)
            service.model = "test-model"
            generations = []
            def rejecting(prompt, schema, max_tokens):
                if schema is REVIEW_SCHEMA:
                    return {"grounded_in_source": False, "no_unsupported_claims": True, "no_contradiction": True,
                            "needs_review": True, "reason": "unsupported claim"}
                generations.append(1)
                return self.valid_content()
            service._generate_json = rejecting
            with self.assertRaisesRegex(ContentValidationError, "unsupported claim"):
                service.create_and_review(verse, tafsir)
            self.assertEqual(len(generations), CONTENT_ATTEMPTS)

        def test_generation_prompt_carries_topic_budget_and_rules(self):
            verse = QuranVerse(2, 3, "2:3", "آية", 1, "البقرة")
            tafsir = TafsirSource("نص التفسير الموثق", "مصدر", 16, "https://x", "https://l", True)
            prompt = GeminiContentService._generation_prompt(verse, tafsir, "بر الوالدين", (42, 61))
            self.assertIn("بر الوالدين", prompt)
            self.assertIn("42 إلى 61 كلمة", prompt)
            self.assertIn("نص التفسير الموثق", prompt)
            self.assertIn("صدقة جارية", prompt)
            self.assertIn("لا تذكر أي مصدر", prompt)
            self.assertIn("image_prompt", prompt)
            self.assertNotIn("الموضوع المقترح", GeminiContentService._generation_prompt(verse, tafsir, None, (42, 61)))

        # ----- Quran words (waqf signs are not words) -----
        def test_quran_words_attach_waqf_signs_to_the_previous_word(self):
            self.assertEqual(quran_words("بسم الله الرحمن"), ["بسم", "الله", "الرحمن"])
            text = "فَٱتَّقُوا۟ ٱللَّهَ ۗ وَمَن يُوقَ ۚ فَأُو۟لَٰٓئِكَ"
            words = quran_words(text)
            self.assertEqual(len(text.split()), 7)
            self.assertEqual(len(words), 5)
            self.assertEqual(words[1], "ٱللَّهَ ۗ")
            self.assertEqual(quran_words("۞ بسم الله"), ["۞ بسم", "الله"])
            self.assertEqual(quran_words(""), [])

        def test_word_segments_accept_verses_that_contain_waqf_signs(self):
            text = "بسم الله \u06d7 الرحمن \u06da الرحيم"
            segments = [[i + 1, i * 400, i * 400 + 380] for i in range(4)]
            got = validate_word_segments(segments, text, 1700)
            self.assertEqual([row["word_index"] for row in got], [0, 1, 2, 3])
            with self.assertRaises(ServiceError):
                validate_word_segments(segments[:3], text, 1700)

        def test_tafsir_license_fails_closed(self):
            with patch.dict(os.environ, {"TAFSIR_ID": "1", "TAFSIR_LICENSE_URL": "https://example.org/terms",
                                         "TAFSIR_COMMERCIAL_LICENSE_VERIFIED": "false", "QF_CLIENT_ID": "x", "QF_CLIENT_SECRET": "y"}):
                client = object.__new__(QuranFoundationClient)
                with self.assertRaises(ServiceError):
                    client.get_tafsir("1:1")

        def test_tafsir_requires_allowlisted_source_and_arabic_text(self):
            payload = {"tafsir": {"resource_id": 7, "verses": {"1:1": {"id": 99}},
                                   "resource_name": "مصدر تفسير موثق", "language_name": "arabic",
                                   "text": "هذا تفسير عربي موثق يشرح معنى الآية ضمن سياقها."}}
            client = self.ready_qf_client(self.FakeResponse(200, payload))
            env = {"TAFSIR_ID": "7", "TAFSIR_LICENSE_URL": "https://example.org/license",
                   "TAFSIR_COMMERCIAL_LICENSE_VERIFIED": "true", "TAFSIR_ALLOWLIST": "7|https://example.org/license"}
            with patch.dict(os.environ, env):
                result = client.get_tafsir("1:1")
            self.assertEqual(result.resource_id, 7)
            self.assertEqual(result.source_name, "مصدر تفسير موثق")
            requested_fields = client.session.get_calls[0][1]["params"]["fields"]
            self.assertIn("language_name", requested_fields)

        def test_english_tafsir_resource_169_fails_with_arabic_muyassar_guidance(self):
            payload = {"tafsir": {"resource_id": 169, "verses": {"2:3": {"id": 99}},
                                   "resource_name": "Ibn Kathir (Abridged)", "language_name": "english",
                                   "text": "English tafsir content long enough to be processed safely."}}
            client = self.ready_qf_client(self.FakeResponse(200, payload))
            env = {"TAFSIR_ID": "169", "TAFSIR_LICENSE_URL": "https://example.org/license",
                   "TAFSIR_COMMERCIAL_LICENSE_VERIFIED": "true",
                   "TAFSIR_ALLOWLIST": "169|https://example.org/license"}
            with patch.dict(os.environ, env):
                with self.assertRaisesRegex(ServiceError, "TAFSIR_ID=169.*TAFSIR_ID=16"):
                    client.get_tafsir("2:3")

        def test_tts_requires_seal_once_in_standalone_segment(self):
            import tempfile
            service = object.__new__(TTSService)
            async def fake_save(text, path):
                Path(path).write_bytes(b"x" * 600)
            service._save = fake_save
            with tempfile.TemporaryDirectory() as temp:
                with self.assertRaises(ServiceError):
                    service.synthesize({"hook": "هوك", "seal": "صدق الله العظيم صدق الله العظيم"}, Path(temp))
                saved = service.synthesize({"hook": "هوك", "seal": "صدق الله العظيم"}, Path(temp))
                self.assertEqual(set(saved), {"hook", "seal"})

        def test_quran_key_rejects_out_of_range_chapter(self):
            with self.assertRaises(ServiceError):
                verse_key(115, 1)

        # ----- Gemini retry / diagnostics -----
        @staticmethod
        def make_gemini(once):
            service = object.__new__(GeminiContentService)
            service.api_key = "AIzaSyFAKEKEYFAKEKEYFAKEKEYFAKEKEY123"
            service.model = "test-model"
            service.sleeps = []
            service._sleep = lambda seconds: service.sleeps.append(seconds)
            service._generate_json_once = once
            return service

        def test_gemini_error_classification_uses_http_code_and_network_class_names(self):
            class FakeAPIError(Exception):
                def __init__(self, code, status):
                    super().__init__("boom")
                    self.code, self.status = code, status
            self.assertEqual(_gemini_exception_info(FakeAPIError(503, "UNAVAILABLE")), (True, 503, "UNAVAILABLE"))
            self.assertEqual(_gemini_exception_info(FakeAPIError(500, "INTERNAL")), (True, 500, "INTERNAL"))
            self.assertEqual(_gemini_exception_info(FakeAPIError(429, "RESOURCE_EXHAUSTED"))[0], True)
            self.assertEqual(_gemini_exception_info(FakeAPIError(400, "INVALID_ARGUMENT"))[0], False)
            self.assertEqual(_gemini_exception_info(FakeAPIError(404, "NOT_FOUND"))[0], False)
            self.assertEqual(_gemini_exception_info(FakeAPIError(403, "PERMISSION_DENIED"))[0], False)
            class TimeoutException(Exception):
                pass
            class ReadTimeout(TimeoutException):
                pass
            self.assertEqual(_gemini_exception_info(ReadTimeout("slow"))[0], True)
            self.assertEqual(_gemini_exception_info(ValueError("programming error"))[0], False)

        def test_gemini_error_description_redacts_api_key_and_truncates(self):
            key = "AIzaSyFAKEKEYFAKEKEYFAKEKEYFAKEKEY123"
            error = RuntimeError(f"request to https://x/?key={key} failed " + "x" * 500)
            detail = _describe_gemini_exception(error, key, 503, "UNAVAILABLE")
            self.assertNotIn(key, detail)
            self.assertIn("code=503", detail)
            self.assertIn("status=UNAVAILABLE", detail)
            self.assertLessEqual(len(detail), 260)

        def test_gemini_retries_transient_failures_with_backoff_then_succeeds(self):
            calls = []
            def once(prompt, schema, max_tokens):
                calls.append(1)
                if len(calls) < 3:
                    raise _GeminiTransient("code=503, status=UNAVAILABLE")
                return {"ok": True}
            service = self.make_gemini(once)
            self.assertEqual(service._generate_json("p", {}, 10), {"ok": True})
            self.assertEqual(len(calls), 3)
            self.assertEqual(service.sleeps, [GEMINI_RETRY_DELAYS[0], GEMINI_RETRY_DELAYS[1]])

        def test_gemini_does_not_retry_permanent_errors(self):
            calls = []
            def once(prompt, schema, max_tokens):
                calls.append(1)
                raise ServiceError("code=404 model not found")
            service = self.make_gemini(once)
            with self.assertRaisesRegex(ServiceError, "404"):
                service._generate_json("p", {}, 10)
            self.assertEqual(len(calls), 1)
            self.assertEqual(service.sleeps, [])

        def test_gemini_gives_up_after_all_attempts_and_reports_last_detail(self):
            calls = []
            def once(prompt, schema, max_tokens):
                calls.append(1)
                raise _GeminiTransient("code=503, status=UNAVAILABLE")
            service = self.make_gemini(once)
            with self.assertRaisesRegex(ServiceError, "code=503"):
                service._generate_json("p", {}, 10)
            self.assertEqual(len(calls), len(GEMINI_RETRY_DELAYS) + 1)
            self.assertEqual(service.sleeps, list(GEMINI_RETRY_DELAYS))

        # ----- TTS voice diagnostics / retry -----
        def test_tts_misspelled_default_voice_is_corrected_to_shakir(self):
            with patch.dict(os.environ, {"TTS_TERMS_CONFIRMED": "true",
                                          "TTS_LICENSE_URL": "https://example.org/terms"}):
                service = TTSService("ar-EG-ShakerNeural")
            self.assertEqual(service.voice, "ar-EG-ShakirNeural")

        def test_tts_unavailable_voice_lists_available_arabic_voices(self):
            import tempfile
            service = object.__new__(TTSService)
            service.voice = "ar-XX-NopeNeural"
            service._voice_checked = False
            async def fake_list_voices():
                return [{"ShortName": "ar-EG-SalmaNeural"}, {"ShortName": "ar-EG-ShakirNeural"},
                        {"ShortName": "en-US-AriaNeural"}]
            with tempfile.TemporaryDirectory() as temp, patch("edge_tts.list_voices", new=fake_list_voices):
                with self.assertRaises(ServiceError) as caught:
                    asyncio.run(service._save("نص", Path(temp) / "x.mp3"))
            message = str(caught.exception)
            self.assertIn("ar-EG-ShakirNeural", message)
            self.assertIn("ar-EG-SalmaNeural", message)
            self.assertNotIn("en-US-AriaNeural", message)

        def test_tts_retries_transient_synthesis_failures(self):
            import tempfile
            service = object.__new__(TTSService)
            sleeps, attempts = [], []
            service._sleep = lambda seconds: sleeps.append(seconds)
            async def flaky_save(text, path):
                attempts.append(1)
                if len(attempts) < 3:
                    raise ConnectionError("temporary")
                Path(path).write_bytes(b"x" * 600)
            service._save = flaky_save
            with tempfile.TemporaryDirectory() as temp:
                saved = service.synthesize({"seal": "صدق الله العظيم"}, Path(temp))
            self.assertEqual(set(saved), {"seal"})
            self.assertEqual(len(attempts), 3)
            self.assertEqual(sleeps, list(TTS_RETRY_DELAYS))

        def test_tts_voice_unavailable_is_not_retried(self):
            import tempfile
            service = object.__new__(TTSService)
            sleeps, attempts = [], []
            service._sleep = lambda seconds: sleeps.append(seconds)
            async def bad_voice(text, path):
                attempts.append(1)
                raise ServiceError("صوت TTS_VOICE غير متاح")
            service._save = bad_voice
            with tempfile.TemporaryDirectory() as temp:
                with self.assertRaises(ServiceError):
                    service.synthesize({"seal": "صدق الله العظيم"}, Path(temp))
            self.assertEqual(len(attempts), 1)
            self.assertEqual(sleeps, [])

        # ----- Surah names / TTS word boundaries -----
        def test_surah_names_table_is_complete_and_used_for_verse_card(self):
            self.assertEqual(len(SURAH_NAMES_AR), 114)
            self.assertEqual((SURAH_NAMES_AR[0], SURAH_NAMES_AR[1], SURAH_NAMES_AR[113]), ("الفاتحة", "البقرة", "الناس"))
            self.assertEqual(len(set(SURAH_NAMES_AR)), 114)
            payload = {"verse": {"verse_key": "2:152", "verse_number": 152, "chapter_id": 2,
                                 "id": 1, "text_uthmani": "فَاذْكُرُونِي أَذْكُرْكُمْ"}}
            verse = self.ready_qf_client(self.FakeResponse(200, payload)).get_verse(2, 152)
            self.assertEqual(verse.chapter_name, "البقرة")

        def test_tts_save_requests_word_boundaries_and_collects_them(self):
            import tempfile
            class FakeCommunicate:
                kwargs = {}
                def __init__(self, **kwargs):
                    FakeCommunicate.kwargs = kwargs
                async def stream(self):
                    yield {"type": "audio", "data": b"a" * 400}
                    yield {"type": "WordBoundary", "offset": 10_000_000, "duration": 2_500_000, "text": "كلمة"}
                    yield {"type": "SentenceBoundary", "offset": 0, "duration": 1, "text": "ignored"}
                    yield {"type": "audio", "data": b"b" * 400}
            service = object.__new__(TTSService)
            service.voice, service._voice_checked = "ar-EG-ShakirNeural", True
            with tempfile.TemporaryDirectory() as temp, patch("edge_tts.Communicate", FakeCommunicate):
                target = Path(temp) / "x.mp3"
                asyncio.run(service._save("كلمة", target))
                self.assertEqual(target.stat().st_size, 800)
            self.assertEqual(FakeCommunicate.kwargs["boundary"], "WordBoundary")
            self.assertEqual(service._last_boundaries, [{"text": "كلمة", "start_ms": 1000, "end_ms": 1250}])

        def test_tts_older_edge_tts_without_boundary_option_still_works(self):
            import tempfile
            class OldCommunicate:
                def __init__(self, text, voice, rate="+0%", pitch="+0Hz"):
                    self.text = text
                async def stream(self):
                    yield {"type": "audio", "data": b"a" * 600}
                    yield {"type": "WordBoundary", "offset": 0, "duration": 1_000_000, "text": "x"}
            service = object.__new__(TTSService)
            service.voice, service._voice_checked = "ar-EG-ShakirNeural", True
            with tempfile.TemporaryDirectory() as temp, patch("edge_tts.Communicate", OldCommunicate):
                asyncio.run(service._save("x", Path(temp) / "x.mp3"))
            self.assertEqual(service._last_boundaries, [{"text": "x", "start_ms": 0, "end_ms": 100}])

        def test_tts_synthesize_exposes_boundaries_per_segment_and_empty_when_unavailable(self):
            import tempfile
            service = object.__new__(TTSService)
            async def fake_save(text, path):
                Path(path).write_bytes(b"x" * 600)
                service._last_boundaries = [{"text": text, "start_ms": 0, "end_ms": 500}] if text == "هوك" else []
            service._save = fake_save
            with tempfile.TemporaryDirectory() as temp:
                service.synthesize({"hook": "هوك", "seal": "صدق الله العظيم", "cta": "شارك"}, Path(temp))
            self.assertEqual(service.word_boundaries["hook"], [{"text": "هوك", "start_ms": 0, "end_ms": 500}])
            self.assertEqual(service.word_boundaries["cta"], [])
            self.assertEqual(set(service.word_boundaries), {"hook", "seal", "cta"})

        # ----- Gemini text-to-speech -----
        @staticmethod
        def pcm(seconds, amplitude=8000, rate=24000):
            count = int(seconds * rate)
            return array("h", [amplitude if (i // 40) % 2 == 0 else -amplitude for i in range(count)]).tobytes()

        @staticmethod
        def make_gemini_tts(model="gemini-3.8-flash-tts"):
            service = object.__new__(GeminiTTSService)
            service.api_key, service.model, service.voice = "k", model, "Algieba"
            service.terms_url, service.provider = GEMINI_TTS_TERMS, f"gemini-tts ({model})"
            service.word_boundaries = {}
            service.sleeps = []
            service._sleep = lambda seconds: service.sleeps.append(seconds)
            return service

        def test_gemini_tts_mime_parsing_and_wav_wrapping(self):
            self.assertEqual(_parse_pcm_mime("audio/L16;codec=pcm;rate=24000"), {"bits_per_sample": 16, "rate": 24000})
            self.assertEqual(_parse_pcm_mime("audio/L16;rate=16000")["rate"], 16000)
            self.assertEqual(_parse_pcm_mime("audio/L16"), {"bits_per_sample": 16, "rate": 24000})
            wav = _pcm_to_wav(self.pcm(1.0), "audio/L16;rate=24000")
            with wave.open(io.BytesIO(wav), "rb") as reader:
                self.assertEqual((reader.getnchannels(), reader.getsampwidth(), reader.getframerate()), (1, 2, 24000))
                self.assertEqual(reader.getnframes(), 24000)
            self.assertEqual(_pcm_to_wav(wav, "audio/wav"), wav)          # already a WAV file
            with self.assertRaises(ServiceError):
                _pcm_to_wav(b"\x00\x01" * 10, "text/plain")
            with self.assertRaises(ServiceError):
                _pcm_to_wav(b"\x00\x01" * 10, "audio/L24;rate=24000")

        def test_gemini_tts_wav_stats_report_duration_and_peak(self):
            seconds, peak = _wav_stats(_pcm_to_wav(self.pcm(2.0, amplitude=1234), "audio/L16;rate=24000"))
            self.assertAlmostEqual(seconds, 2.0, places=3)
            self.assertEqual(peak, 1234)
            with self.assertRaises(_GeminiTransient):
                _wav_stats(b"RIFFnot-a-wav")

        def test_gemini_tts_synthesize_writes_wav_files_without_word_boundaries(self):
            import tempfile
            service = self.make_gemini_tts()
            service._request_audio = lambda text: (self.pcm(2.0), "audio/L16;rate=24000")
            with tempfile.TemporaryDirectory() as temp:
                saved = service.synthesize({"hook": "ما هو الإيمان بالغيب", "seal": "صدق الله العظيم"}, Path(temp))
                self.assertEqual(set(saved), {"hook", "seal"})
                for path in saved.values():
                    self.assertEqual(path.suffix, ".wav")
                    self.assertEqual(path.read_bytes()[:4], b"RIFF")
            self.assertEqual(service.word_boundaries, {"hook": [], "seal": []})

        def test_gemini_tts_requires_the_seal_exactly_once(self):
            import tempfile
            service = self.make_gemini_tts()
            service._request_audio = lambda text: (self.pcm(2.0), "audio/L16;rate=24000")
            with tempfile.TemporaryDirectory() as temp:
                with self.assertRaises(ServiceError):
                    service.synthesize({"hook": "هوك"}, Path(temp))
                with self.assertRaises(ServiceError):
                    service.synthesize({"hook": "صدق الله العظيم", "seal": "صدق الله العظيم"}, Path(temp))

        def test_gemini_tts_retries_implausible_audio_then_succeeds(self):
            import tempfile
            service = self.make_gemini_tts()
            calls = []
            def request(text):
                calls.append(1)
                return (self.pcm(0.1) if len(calls) == 1 else self.pcm(2.0)), "audio/L16;rate=24000"
            service._request_audio = request
            with tempfile.TemporaryDirectory() as temp:
                service.synthesize({"seal": "صدق الله العظيم"}, Path(temp))
            self.assertEqual(len(calls), 2)
            self.assertEqual(service.sleeps, [GEMINI_TTS_RETRY_DELAYS[0]])

        def test_gemini_tts_rejects_silent_and_runaway_audio_after_all_attempts(self):
            import tempfile
            for label, audio in (("silent", self.pcm(2.0, amplitude=10)), ("runaway", self.pcm(20.0))):
                service = self.make_gemini_tts()
                calls = []
                def request(text, audio=audio):
                    calls.append(1)
                    return audio, "audio/L16;rate=24000"
                service._request_audio = request
                with tempfile.TemporaryDirectory() as temp:
                    with self.assertRaises(ServiceError, msg=label):
                        service.synthesize({"seal": "صدق الله العظيم"}, Path(temp))
                self.assertEqual(len(calls), len(GEMINI_TTS_RETRY_DELAYS) + 1, label)
                self.assertEqual(service.sleeps, list(GEMINI_TTS_RETRY_DELAYS), label)

        def test_gemini_tts_does_not_retry_permanent_errors(self):
            import tempfile
            service = self.make_gemini_tts()
            calls = []
            def request(text):
                calls.append(1)
                raise ServiceError("code=403 permission denied")
            service._request_audio = request
            with tempfile.TemporaryDirectory() as temp:
                with self.assertRaisesRegex(ServiceError, "403"):
                    service.synthesize({"seal": "صدق الله العظيم"}, Path(temp))
            self.assertEqual(len(calls), 1)
            self.assertEqual(service.sleeps, [])

        def test_gemini_tts_prompt_depends_on_model_family(self):
            self.assertEqual(self.make_gemini_tts("gemini-3.8-flash-tts")._prompt("نص"), "نص")
            legacy = self.make_gemini_tts("gemini-2.5-pro-preview-tts")._prompt("نص")
            self.assertIn("## Transcript:\nنص", legacy)
            self.assertIn("exactly as written", legacy)

        def test_gemini_tts_extracts_inline_audio_from_response_parts(self):
            class Inline:
                def __init__(self, data, mime_type):
                    self.data, self.mime_type = data, mime_type
            class Part:
                def __init__(self, inline_data=None):
                    self.inline_data = inline_data
            class Content:
                parts = [Part(), Part(Inline(b"ab", "audio/L16;rate=24000")), Part(Inline(b"cd", "audio/L16;rate=24000"))]
            class Candidate:
                content = Content()
            class Response:
                candidates = [Candidate()]
            self.assertEqual(GeminiTTSService._extract_audio(Response()), (b"abcd", "audio/L16;rate=24000"))
            self.assertEqual(GeminiTTSService._extract_audio(object()), (b"", ""))

        def test_gemini_tts_constructor_validates_settings(self):
            base = {"GEMINI_API_KEY1": "key", "GEMINI_API_KEY": "", "TTS_TERMS_CONFIRMED": "true",
                    "GEMINI_TTS_MODEL": "", "GEMINI_TTS_VOICE": "", "GEMINI_TTS_TERMS_URL": ""}
            with patch.dict(os.environ, base):
                service = GeminiTTSService()
                self.assertEqual((service.model, service.voice), (DEFAULT_GEMINI_TTS_MODEL, DEFAULT_GEMINI_TTS_VOICE))
                self.assertEqual(service.terms_url, GEMINI_TTS_TERMS)
                self.assertIn(service.model, service.provider)
            with patch.dict(os.environ, {**base, "GEMINI_API_KEY1": "", "GEMINI_API_KEY": "fallback-key"}):
                self.assertEqual(GeminiTTSService().api_key, "fallback-key")
            with patch.dict(os.environ, {**base, "GEMINI_API_KEY1": "", "GEMINI_API_KEY": ""}):
                with self.assertRaises(ServiceError):
                    GeminiTTSService()
            with patch.dict(os.environ, {**base, "GEMINI_TTS_VOICE": "bad voice!"}):
                with self.assertRaises(ServiceError):
                    GeminiTTSService()
            with patch.dict(os.environ, {**base, "TTS_TERMS_CONFIRMED": "false"}):
                with self.assertRaises(ServiceError):
                    GeminiTTSService()
            with patch.dict(os.environ, {**base, "GEMINI_TTS_TERMS_URL": "http://insecure.example"}):
                with self.assertRaises(ServiceError):
                    GeminiTTSService()

        def test_tts_provider_factory_selects_edge_gemini_or_auto(self):
            env = {"TTS_TERMS_CONFIRMED": "true", "TTS_LICENSE_URL": "https://example.org/terms",
                   "GEMINI_API_KEY1": "key", "GEMINI_API_KEY": "", "GEMINI_TTS_MODEL": "", "GEMINI_TTS_VOICE": ""}
            with patch.dict(os.environ, {**env, "TTS_PROVIDER": ""}):
                self.assertIsInstance(make_tts_service("ar-EG-ShakirNeural"), TTSService)
            with patch.dict(os.environ, {**env, "TTS_PROVIDER": "edge"}):
                self.assertIsInstance(make_tts_service("ar-EG-ShakirNeural"), TTSService)
            with patch.dict(os.environ, {**env, "TTS_PROVIDER": "gemini"}):
                self.assertIsInstance(make_tts_service("ar-EG-ShakirNeural"), GeminiTTSService)
            with patch.dict(os.environ, {**env, "TTS_PROVIDER": "auto"}):
                auto = make_tts_service("ar-EG-ShakirNeural")
                self.assertIsInstance(auto, FallbackTTSService)
                self.assertIsInstance(auto.primary, GeminiTTSService)
                self.assertIsInstance(auto.secondary, TTSService)
            with patch.dict(os.environ, {**env, "TTS_PROVIDER": "auto", "GEMINI_API_KEY1": ""}):
                self.assertIsInstance(make_tts_service("ar-EG-ShakirNeural"), TTSService)
            with patch.dict(os.environ, {**env, "TTS_PROVIDER": "elevenlabs"}):
                with self.assertRaises(ServiceError):
                    make_tts_service()

        def test_fallback_tts_uses_secondary_for_all_segments_and_reports_its_identity(self):
            class Fake:
                def __init__(self, name, voice, fail):
                    self.provider, self.voice, self.terms_url = name, voice, f"https://{name}.example/terms"
                    self.fail, self.word_boundaries, self.calls = fail, {}, 0
                def synthesize(self, texts, output_dir):
                    self.calls += 1
                    if self.fail:
                        raise ServiceError("quota exceeded")
                    self.word_boundaries = {key: [] for key in texts}
                    return {key: Path(f"{self.provider}-{key}.mp3") for key in texts}
            primary, secondary = Fake("gemini", "Algieba", True), Fake("edge", "ar-EG-ShakirNeural", False)
            service = FallbackTTSService(primary, secondary)
            self.assertEqual(service.voice, "Algieba")
            files = service.synthesize({"hook": "هوك", "seal": "صدق الله العظيم"}, Path("."))
            self.assertEqual((primary.calls, secondary.calls), (1, 1))
            self.assertTrue(all(str(path).startswith("edge-") for path in files.values()))
            self.assertEqual((service.voice, service.provider), ("ar-EG-ShakirNeural", "edge"))
            self.assertEqual(service.terms_url, "https://edge.example/terms")
            healthy = FallbackTTSService(Fake("gemini", "Algieba", False), secondary)
            healthy.synthesize({"seal": "صدق الله العظيم"}, Path("."))
            self.assertEqual((healthy.voice, healthy.provider), ("Algieba", "gemini"))

        # ----- Make.com publisher -----
        class FakeHTTP:
            """Shared response queue for both `request` (GitHub) and `post` (Make webhook)."""
            def __init__(self, responses):
                self.responses, self.calls = list(responses), []
            def request(self, method, url, **kwargs):
                self.calls.append(("request", method, url, kwargs))
                return self.responses.pop(0)
            def post(self, url, **kwargs):
                self.calls.append(("post", url, kwargs))
                return self.responses.pop(0)

        MAKE_ENV = {"MAKE_WEBHOOK_URL": "https://hook.eu1.make.com/abc", "MAKE_WEBHOOK_API_KEY": "secret-key",
                    "GITHUB_TOKEN": "gh-token", "GITHUB_REPOSITORY": "owner/repo",
                    "MAKE_PLATFORMS": "facebook,youtube,instagram,tiktok"}

        def make_publisher(self, responses):
            with patch.dict(os.environ, self.MAKE_ENV):
                publisher = MakeWebhookPublisher(session=self.FakeHTTP(responses))
            publisher._sleep = lambda seconds: None
            return publisher

        def test_make_publisher_rejects_unsafe_webhook_and_repository(self):
            for url in ("http://hook.eu1.make.com/abc", "https://evil.example/abc", "https://make.com.evil.example/abc"):
                with patch.dict(os.environ, {**self.MAKE_ENV, "MAKE_WEBHOOK_URL": url}):
                    with self.assertRaises(ServiceError, msg=url):
                        MakeWebhookPublisher()
            with patch.dict(os.environ, {**self.MAKE_ENV, "GITHUB_REPOSITORY": "not a repo"}):
                with self.assertRaises(ServiceError):
                    MakeWebhookPublisher()

        def test_make_publisher_reuses_existing_release_asset_and_sends_payload(self):
            import tempfile
            content = b"fake-mp4-bytes" * 100
            tag = "video-" + hashlib.sha256(content).hexdigest()[:16]
            url = f"https://github.com/owner/repo/releases/download/{tag}/{tag}.mp4"
            release = {"assets": [{"name": f"{tag}.mp4", "state": "uploaded", "browser_download_url": url}]}
            publisher = self.make_publisher([self.FakeResponse(200, release), self.FakeResponse(200, {})])
            with tempfile.TemporaryDirectory() as temp:
                video = Path(temp) / "video.mp4"
                video.write_bytes(content)
                with patch.dict(os.environ, {"YOUTUBE_PRIVACY_STATUS": "unlisted"}):
                    result = publisher.publish_or_verify(video, "العنوان", "الوصف")
            self.assertEqual(result["status"], "published")
            self.assertEqual((result["video_id"], result["video_url"]), (tag, url))
            calls = publisher.session.calls
            self.assertEqual([call[0] for call in calls], ["request", "post"])     # no upload happened
            webhook = calls[1]
            self.assertEqual(webhook[1], "https://hook.eu1.make.com/abc")
            self.assertEqual(webhook[2]["headers"]["x-make-apikey"], "secret-key")
            body = webhook[2]["json"]
            self.assertEqual(body["idempotency_key"], tag)
            self.assertEqual(body["video_url"], url)
            self.assertEqual(body["platforms"], ["facebook", "youtube", "instagram", "tiktok"])
            self.assertEqual(body["youtube_privacy"], "unlisted")
            self.assertLessEqual(len(body["caption"]), 2000)

        def test_make_publisher_creates_release_and_uploads_when_missing(self):
            import tempfile
            created = {"upload_url": "https://uploads.github.com/repos/owner/repo/releases/1/assets{?name,label}", "assets": []}
            uploaded = {"browser_download_url": "https://github.com/owner/repo/releases/download/t/t.mp4"}
            publisher = self.make_publisher([self.FakeResponse(404, {}), self.FakeResponse(201, created),
                                             self.FakeResponse(201, uploaded), self.FakeResponse(200, {})])
            with tempfile.TemporaryDirectory() as temp:
                video = Path(temp) / "video.mp4"
                video.write_bytes(b"x" * 2048)
                result = publisher.publish_or_verify(video, "title", "desc")
            self.assertEqual(result["video_url"], uploaded["browser_download_url"])
            kinds = [(call[0], call[1] if call[0] == "request" else "POST") for call in publisher.session.calls]
            self.assertEqual(kinds, [("request", "GET"), ("request", "POST"), ("request", "POST"), ("post", "POST")])
            self.assertEqual(publisher.session.calls[2][3]["params"], {"name": result["video_id"] + ".mp4"})

        def test_make_publisher_rejects_unexpected_upload_host(self):
            import tempfile
            created = {"upload_url": "https://evil.example/upload{?name}", "assets": []}
            publisher = self.make_publisher([self.FakeResponse(404, {}), self.FakeResponse(201, created)])
            with tempfile.TemporaryDirectory() as temp:
                video = Path(temp) / "video.mp4"
                video.write_bytes(b"x" * 2048)
                with self.assertRaises(ServiceError):
                    publisher.publish_or_verify(video, "title", "desc")

        def test_make_publisher_keeps_id_and_url_when_webhook_keeps_failing(self):
            import tempfile
            url = "https://github.com/owner/repo/releases/download/t/t.mp4"
            release = {"assets": [{"name": "placeholder", "state": "uploaded", "browser_download_url": url}]}
            content = b"y" * 2048
            tag = "video-" + hashlib.sha256(content).hexdigest()[:16]
            release["assets"][0]["name"] = f"{tag}.mp4"
            publisher = self.make_publisher([self.FakeResponse(200, release), self.FakeResponse(500, {}),
                                             self.FakeResponse(502, {}), self.FakeResponse(503, {})])
            with tempfile.TemporaryDirectory() as temp:
                video = Path(temp) / "video.mp4"
                video.write_bytes(content)
                with self.assertRaises(ServiceError) as caught:
                    publisher.publish_or_verify(video, "title", "desc")
            state = caught.exception.platform_state
            self.assertEqual((state["status"], state["video_id"], state["video_url"]), ("uploaded_unconfirmed", tag, url))
            self.assertEqual(sum(1 for call in publisher.session.calls if call[0] == "post"), 3)

        def test_make_publisher_does_not_retry_a_client_error(self):
            import tempfile
            content = b"z" * 2048
            tag = "video-" + hashlib.sha256(content).hexdigest()[:16]
            release = {"assets": [{"name": f"{tag}.mp4", "state": "uploaded",
                                   "browser_download_url": "https://github.com/o/r/releases/download/t/t.mp4"}]}
            publisher = self.make_publisher([self.FakeResponse(200, release), self.FakeResponse(400, {})])
            with tempfile.TemporaryDirectory() as temp:
                video = Path(temp) / "video.mp4"
                video.write_bytes(content)
                with self.assertRaisesRegex(ServiceError, "HTTP 400"):
                    publisher.publish_or_verify(video, "title", "desc")
            self.assertEqual(sum(1 for call in publisher.session.calls if call[0] == "post"), 1)

        def test_make_publisher_resumes_from_saved_url_without_touching_github_or_the_file(self):
            publisher = self.make_publisher([self.FakeResponse(200, {})])
            previous = {"status": "uploaded_unconfirmed", "video_id": "video-abc123",
                        "video_url": "https://github.com/o/r/releases/download/video-abc123/video-abc123.mp4",
                        "sha256": "f" * 64}
            result = publisher.publish_or_verify(Path("/nonexistent/video.mp4"), "title", "desc", previous=previous)
            self.assertEqual(result["status"], "published")
            self.assertEqual([call[0] for call in publisher.session.calls], ["post"])
            self.assertEqual(publisher.session.calls[0][2]["json"]["idempotency_key"], "video-abc123")
            again = publisher.publish_or_verify(Path("/nonexistent/video.mp4"), "title", "desc", previous=result)
            self.assertEqual(again, result)
            self.assertEqual(len(publisher.session.calls), 1)

    unittest.main()
