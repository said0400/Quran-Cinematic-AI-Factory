"""External services for Quran Cinematic AI Factory.

All network access is isolated here. Rights are never inferred from API access;
operators must configure and record the applicable source-specific permissions.
"""
from __future__ import annotations

import asyncio
import html
import json
import os
import re
import subprocess
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

import requests

QF_API = "https://apis.quran.foundation/content/api/v4"
QF_PRELIVE_API = "https://apis-prelive.quran.foundation/content/api/v4"
QF_PRODUCTION_AUTH = "https://oauth2.quran.foundation"
QF_PRELIVE_AUTH = "https://prelive-oauth2.quran.foundation"
QF_DOCS = "https://api-docs.quran.foundation/docs/content_apis_versioned/4.0.0/"
QF_TERMS = "https://api-docs.quran.foundation/legal/developer-terms/"
PEXELS_SEARCH = "https://api.pexels.com/v1/videos/search"
PEXELS_LICENSE = "https://www.pexels.com/license/"
YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
YOUTUBE_READ_SCOPE = "https://www.googleapis.com/auth/youtube.readonly"
YOUTUBE_UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"


class ServiceError(RuntimeError):
    """A sanitized, user-safe service or validation failure."""
    def __init__(self, message: str, platform_state: dict[str, Any] | None = None):
        super().__init__(message)
        self.platform_state = platform_state or {}


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


def validate_word_segments(segments: Any, text_uthmani: str, audio_duration_ms: int,
                           end_tolerance_ms: int = 300) -> list[dict[str, int]]:
    """Validate Quran Foundation [word_index,start_ms,end_ms] timing rows.

    Only contiguous 0-based or 1-based word indexes are accepted. No timing is
    generated or interpolated by this function.
    """
    words = [w for w in (text_uthmani or "").split() if w]
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
            raise ServiceError("توقيت كلمة خارج مدة الصوت أو بترتيب غير صالح.")
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


def _arabic_word_tokens(text: str) -> list[str]:
    normalized = normalize_arabic(text)
    return re.findall(r"[\u0621-\u064a\u0671-\u06d3\u06fa-\u06fc]+", normalized)


def validate_audio_transcription(transcript: str, expected_text: str | Iterable[str], *,
                                 contains_other_speech: bool = False, uncertain: bool = False) -> dict[str, Any]:
    if contains_other_speech or uncertain:
        raise ServiceError("تفريغ التلاوة غير مؤكد أو يتضمن كلامًا إضافيًا؛ رُفض المرشح.")
    heard = _arabic_word_tokens(transcript)
    variants = [expected_text] if isinstance(expected_text, str) else list(expected_text)
    expected_variants = [_arabic_word_tokens(variant) for variant in variants if isinstance(variant, str)]
    matched = next((expected for expected in expected_variants if expected and heard == expected), None)
    if not matched or not heard:
        raise ServiceError("التفريغ الصوتي لا يطابق كلمات الآية بالترتيب؛ رُفض المرشح.")
    return {"status": "exact_match", "word_count": len(matched), "transcription": transcript.strip()}


@dataclass
class QuranVerse:
    chapter: int
    verse: int
    key: str
    text_uthmani: str
    verse_id: int | None
    chapter_name: str = ""
    text_imlaei: str = ""


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
            "fields": "text_uthmani,text_imlaei,verse_key,verse_number,chapter_id,id",
        })
        item = payload.get("verse")
        if not isinstance(item, dict):
            raise ServiceError("استجابة الآية لا تحتوي الحقل الموثق verse.")
        if item.get("verse_key") != key or item.get("verse_number") != int(verse) or item.get("chapter_id") != int(chapter):
            raise ServiceError("مفتاح الآية أو بياناتها لا تطابق الطلب اليدوي.")
        text = item.get("text_uthmani")
        if not isinstance(text, str) or not text.strip() or "�" in text:
            raise ServiceError("النص العثماني مفقود أو تالف.")
        imlaei = item.get("text_imlaei")
        if not isinstance(imlaei, str) or not imlaei.strip() or "�" in imlaei:
            raise ServiceError("النص الإملائي اللازم لمقارنة التفريغ الصوتي مفقود أو تالف.")
        verse_id = item.get("id")
        return QuranVerse(int(chapter), int(verse), key, text.strip(),
                          int(verse_id) if verse_id is not None else None, text_imlaei=imlaei.strip())

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
        segments: list[list[int]] = []
        for row in raw_segments:
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                raise ServiceError(f"المصدر لم يقدّم توقيت كلمة ثلاثيًا للآية {key}؛ اختر قارئ سورة يوفر word-level timestamps.")
            try:
                word_index, start_ms, end_ms = (int(value) for value in row)
            except (TypeError, ValueError):
                raise ServiceError(f"المصدر أعاد توقيت كلمة غير رقمي للآية {key}.") from None
            # The API returns chapter-global milliseconds; convert to the cropped verse timeline.
            segments.append([word_index, start_ms - verse_start, end_ms - verse_start])

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
                "-i", str(chapter_audio), "-ss", f"{verse_start / 1000:.3f}",
                "-t", f"{(verse_end - verse_start) / 1000:.3f}", "-vn",
                "-c:a", "libmp3lame", "-q:a", "2", str(destination),
            ], check=True, capture_output=True, text=True, timeout=180)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            destination.unlink(missing_ok=True)
            chapter_audio.unlink(missing_ok=True)
            raise ServiceError("تعذر اقتطاع الآية من صوت السورة؛ تحقق من تثبيت FFmpeg وصلاحية ملف الصوت.") from None
        if not destination.is_file() or destination.stat().st_size < 4096:
            destination.unlink(missing_ok=True)
            chapter_audio.unlink(missing_ok=True)
            raise ServiceError("ملف التلاوة المقتطع فارغ أو غير مكتمل.")
        chapter_audio.unlink(missing_ok=True)
        return QuranRecitation(reciter_id, reciter_name, full_url, segments, source_url, verse_end - verse_start), destination

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
        payload = self._get_json(f"tafsirs/{resource_id}/by_ayah/{key}", {"fields": "verse_key,resource_name,id"})
        item = payload.get("tafsir")
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise ServiceError("استجابة التفسير لا تطابق بنية tafsir الموثقة.")
        if item.get("resource_id") != resource_id:
            raise ServiceError("معرّف مصدر التفسير في الاستجابة لا يطابق المورد المطلوب.")
        verse_ids = item.get("verses")
        if not isinstance(verse_ids, dict) or key not in verse_ids:
            raise ServiceError("الاستجابة لا تثبت أن نص التفسير يخص مفتاح الآية المطلوب.")
        source_name = item.get("resource_name")
        if not isinstance(source_name, str) or not source_name.strip():
            raise ServiceError("اسم مصدر التفسير غير متوفر.")
        text = _plain_text(item["text"])
        if len(text) < 30:
            raise ServiceError("نص التفسير قصير أو فارغ ولا يكفي للتلخيص.")
        arabic_letters = sum(1 for char in text if "\u0600" <= char <= "\u06ff" and char.isalpha())
        if arabic_letters < 15 or arabic_letters / max(1, len(text)) < 0.10:
            raise ServiceError("نص التفسير لا يبدو عربيًا بما يكفي؛ رُفض بدل تلخيص لغة غير مطلوبة.")
        return TafsirSource(text, source_name.strip(), resource_id, source_url, license_url, True)


HOOK_SCORE_KEYS = (
    "start_strength", "curiosity", "relevance", "clarity", "brevity",
    "uniqueness", "non_misleading", "general_audience",
)
GENERATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "hook_candidates": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "text": {"type": "STRING"},
            **{key: {"type": "INTEGER"} for key in HOOK_SCORE_KEYS},
        }, "required": ["text", *HOOK_SCORE_KEYS]}},
        "explanation": {"type": "STRING"},
        "practical_benefit": {"type": "STRING"},
        "call_to_action": {"type": "STRING"},
    },
    "required": ["hook_candidates", "explanation", "practical_benefit", "call_to_action"],
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
AUDIO_TRANSCRIPTION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "transcript": {"type": "STRING"},
        "contains_other_speech": {"type": "BOOLEAN"},
        "uncertain": {"type": "BOOLEAN"},
    },
    "required": ["transcript", "contains_other_speech", "uncertain"],
}


def validate_generated_content(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ServiceError("مخرجات النموذج ليست كائن JSON.")
    candidates = data.get("hook_candidates")
    if not isinstance(candidates, list) or len(candidates) != 5:
        raise ServiceError("يجب إنشاء خمس صيغ مستقلة للهوك بالضبط.")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("text"), str):
            raise ServiceError("بنية أحد مرشحي الهوك غير صالحة.")
        text = candidate["text"].strip()
        if not text or len(text.split()) > 12 or "صدق الله العظيم" in text or re.search(r"(?:https?://|www\.)", text, re.I):
            raise ServiceError("مرشح الهوك فارغ أو طويل أو يكرر عبارة ما بعد التلاوة.")
        signature = normalize_arabic(text)
        if signature in seen:
            raise ServiceError("مرشحو الهوك مكرّرون.")
        seen.add(signature)
        scores = {}
        for key in HOOK_SCORE_KEYS:
            score = candidate.get(key)
            if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
                raise ServiceError(f"تقييم الهوك {key} خارج المجال 1..5.")
            scores[key] = score
        total = sum(scores.values())
        if scores["non_misleading"] < 4 or scores["relevance"] < 4:
            raise ServiceError("أحد مرشحي الهوك لا يحقق الحد الأدنى للارتباط وعدم التضليل.")
        normalized.append({"text": text, "scores": scores, "total_score": total})
    for key in ("explanation", "practical_benefit", "call_to_action"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip() or len(value) > 1200:
            raise ServiceError(f"حقل {key} فارغ أو غير صالح.")
        if "صدق الله العظيم" in value:
            raise ServiceError("يُمنع تكرار عبارة صدق الله العظيم داخل النص المولّد.")
        if re.search(r"(?:https?://|www\.)", value, re.I):
            raise ServiceError("روابط أو نصوص خارجية غير موثوقة مرفوضة في مخرجات Gemini.")
    best = max(normalized, key=lambda item: (item["total_score"], item["scores"]["non_misleading"], item["scores"]["relevance"]))
    return {"hook": best["text"], "hook_candidates": normalized,
            "explanation": data["explanation"].strip(),
            "practical_benefit": data["practical_benefit"].strip(),
            "call_to_action": data["call_to_action"].strip()}


class GeminiContentService:
    def __init__(self, model: str | None = None):
        self.api_key = _required_env("GEMINI_API_KEY")
        self.model = (model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")).strip()
        if not self.model:
            raise ServiceError("GEMINI_MODEL لا يمكن أن يكون فارغًا.")

    def _generate_json(self, prompt: str, schema: dict[str, Any], max_tokens: int) -> dict[str, Any]:
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=self.api_key)
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
            # Do not include exception text; SDK errors may contain request details.
            raise ServiceError(f"فشل استدعاء Gemini ({type(exc).__name__}).") from None
        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise ServiceError("أعاد Gemini استجابة فارغة.")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            raise ServiceError("استجابة Gemini ليست JSON خالصًا صالحًا.") from None
        if not isinstance(data, dict):
            raise ServiceError("استجابة Gemini لا تطابق الكائن المتوقع.")
        return data

    def verify_recitation_audio(self, audio_path: Path, expected_text: str | Iterable[str]) -> dict[str, Any]:
        if not audio_path.is_file() or audio_path.stat().st_size < 4096:
            raise ServiceError("ملف التلاوة غير صالح لفحص المحتوى الصوتي.")
        if audio_path.stat().st_size > 14 * 1024 * 1024:
            raise ServiceError("حجم التلاوة يتجاوز الحد الآمن للفحص الصوتي المستقل.")
        prompt = (
            "فرّغ التسجيل الصوتي المرفق فقط إلى الكلمات العربية المسموعة، من دون تخمين أو إكمال نص متوقع. "
            "لا أرفق نص الآية عمدًا لتقليل الانحياز. أخرج التفريغ العربي الحرفي في transcript. "
            "اجعل uncertain=true إذا تعذر تمييز كلمة بوضوح، وcontains_other_speech=true إذا وُجد كلام إضافي خارج التلاوة. "
            "لا تصحح التسجيل ولا تضف أية كلمات غير مسموعة. أعد JSON فقط وفق المخطط."
        )
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.model,
                contents=[prompt, types.Part.from_bytes(data=audio_path.read_bytes(), mime_type="audio/mpeg")],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", response_schema=AUDIO_TRANSCRIPTION_SCHEMA,
                    temperature=0.0, max_output_tokens=512,
                ),
            )
        except Exception as exc:
            raise ServiceError(f"تعذر التحقق الصوتي المستقل عبر Gemini ({type(exc).__name__}).") from None
        text = getattr(response, "text", None)
        try:
            data = json.loads(text) if isinstance(text, str) else None
        except json.JSONDecodeError:
            raise ServiceError("نتيجة التفريغ الصوتي ليست JSON صالحًا؛ رُفض المرشح.") from None
        if not isinstance(data, dict) or not isinstance(data.get("transcript"), str):
            raise ServiceError("نتيجة التفريغ الصوتي ناقصة؛ رُفض المرشح.")
        if not isinstance(data.get("contains_other_speech"), bool) or not isinstance(data.get("uncertain"), bool):
            raise ServiceError("علامات الثقة في التفريغ الصوتي غير صالحة؛ رُفض المرشح.")
        return validate_audio_transcription(data["transcript"], expected_text,
                                            contains_other_speech=data["contains_other_speech"],
                                            uncertain=data["uncertain"])

    def create_and_review(self, verse: QuranVerse, tafsir: TafsirSource) -> dict[str, Any]:
        prompt = (
            "أنت محرر عربي مساعد، ولست مصدرًا شرعيًا. استخدم النص التفسيري الموثق المرفق وحده. "
            "لخّص التفسير الموثوق بلغة عربية فصحى بسيطة، وحافظ على المعنى الأساسي، ولا تضف أحكامًا أو أسباب نزول أو معلومات غير موجودة بالمصدر. "
            "أنشئ خمس صيغ مختلفة قصيرة للهوك، وامنح كل واحدة تقييمًا صحيحًا من 1 إلى 5 للمعايير التالية: "
            + ", ".join(HOOK_SCORE_KEYS)
            + ". يجب أن ترتبط بالآية ولا توهم بأنها رسالة غيبية شخصية أو وعد بحل مشكلة، وتجنب التضليل. "
            "قدّم شرحًا موجزًا وفائدة عملية مرتبطة بالآية ودعوة عامة للتفاعل. لا تكتب عبارة «صدق الله العظيم» في أي حقل. "
            "أعد JSON وفق المخطط فقط.\n\n"
            f"الآية الأصلية بالرسم العثماني (للسياق فقط، لا تعِد كتابتها أو تعديلها): {verse.text_uthmani}\n"
            f"المصدر: {tafsir.source_name}\nنص التفسير المسترجع: {tafsir.text}"
        )
        generated = validate_generated_content(self._generate_json(prompt, GENERATION_SCHEMA, 1800))
        review_prompt = (
            "راجع النص المقترح بمقارنته حصريًا بالنص التفسيري المرفق وبالآية الأصلية. أنت لست مفتيًا. "
            "اعتبر كل ادعاء ديني أو تفسيري غير موجود بوضوح في المصدر سببًا لرفضه أو طلب المراجعة. "
            "لا تقيّم الأسلوب فقط. إذا لم تكفِ الأدلة، اجعل needs_review=true. أعد JSON فقط حسب المخطط.\n"
            f"الآية العثمانية: {verse.text_uthmani}\nالتفسير الأصلي من {tafsir.source_name}: {tafsir.text}\n"
            f"النص المراد مراجعته: {json.dumps(generated, ensure_ascii=False)}"
        )
        review = self._generate_json(review_prompt, REVIEW_SCHEMA, 1000)
        for key in ("grounded_in_source", "no_unsupported_claims", "no_contradiction", "needs_review"):
            if not isinstance(review.get(key), bool):
                raise ServiceError("استجابة مراجعة Gemini تحتوي حقولًا منطقية غير صالحة.")
        if not isinstance(review.get("reason"), str):
            raise ServiceError("استجابة المراجعة تفتقر إلى سبب نصي.")
        if review["needs_review"] or not all(review[k] for k in ("grounded_in_source", "no_unsupported_claims", "no_contradiction")):
            raise ServiceError("رفضت مراجعة المحتوى أو طلبت مراجعة بشرية: " + review["reason"][:240])
        generated["review"] = review
        generated["model"] = self.model
        return generated


class PexelsService:
    def __init__(self, session: requests.Session | None = None):
        self.key = _required_env("PEXELS_API_KEY")
        self.session = session or requests.Session()
        if not env_bool("PEXELS_LICENSE_CONFIRMED"):
            raise ServiceError("أكد مراجعة شروط Pexels قبل الاستخدام عبر PEXELS_LICENSE_CONFIRMED=true.")

    def fetch_background(self, topic: str, destination: Path) -> dict[str, Any]:
        queries = {
            "صبر": "quiet ocean sunrise nature",
            "رحمة": "soft forest sunlight nature",
            "أمل": "sunrise landscape nature",
            "تفكر": "mountain lake nature",
            "عدل": "calm nature landscape",
            "عمل": "green field sunrise nature",
        }
        query = next((q for label, q in queries.items() if label in topic), "calm nature landscape sunrise")
        try:
            response = self.session.get(PEXELS_SEARCH, headers={"Authorization": self.key},
                                        params={"query": query, "orientation": "portrait", "size": "medium", "per_page": 20},
                                        timeout=(15, 45))
        except requests.RequestException as exc:
            raise ServiceError(f"تعذر الاتصال بـPexels ({type(exc).__name__}).") from None
        if response.status_code != 200:
            raise ServiceError(f"Pexels أعاد HTTP {response.status_code}.")
        try:
            payload = response.json()
        except ValueError:
            raise ServiceError("استجابة بحث Pexels ليست JSON صالحًا.") from None
        videos = payload.get("videos") if isinstance(payload, dict) else None
        if not isinstance(videos, list) or not videos:
            raise ServiceError("لم يُعثر على فيديو خلفية مرخّص في نتائج Pexels.")
        candidates = []
        for video in videos:
            if not isinstance(video, dict) or not isinstance(video.get("video_files"), list):
                continue
            for item in video["video_files"]:
                if not isinstance(item, dict) or not isinstance(item.get("link"), str):
                    continue
                parsed = urlparse(item["link"])
                if parsed.scheme != "https" or not (parsed.hostname or "").endswith(".pexels.com"):
                    continue
                if item.get("file_type") not in ("video/mp4", "video/webm"):
                    continue
                width, height = item.get("width") or 0, item.get("height") or 0
                candidates.append((int(width or 0) * int(height or 0), video, item))
        if not candidates:
            raise ServiceError("لم تُرجع Pexels رابط فيديو صالحًا على HTTPS.")
        _, video, chosen = max(candidates, key=lambda row: row[0])
        link = chosen["link"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.session.get(link, stream=True, timeout=(20, 120), headers={"User-Agent": "Quran-Cinematic-AI-Factory/1.0"}) as file_response:
                file_response.raise_for_status()
                total = 0
                with destination.open("wb") as out:
                    for chunk in file_response.iter_content(1024 * 1024):
                        if chunk:
                            total += len(chunk)
                            if total > 400 * 1024 * 1024:
                                raise ServiceError("فيديو الخلفية تجاوز الحد الآمن 400 MiB.")
                            out.write(chunk)
        except ServiceError:
            destination.unlink(missing_ok=True)
            raise
        except requests.RequestException as exc:
            destination.unlink(missing_ok=True)
            raise ServiceError(f"تعذر تنزيل فيديو Pexels ({type(exc).__name__}).") from None
        if not destination.exists() or destination.stat().st_size < 4096:
            destination.unlink(missing_ok=True)
            raise ServiceError("فيديو الخلفية فارغ أو غير مكتمل.")
        user = video.get("user") if isinstance(video.get("user"), dict) else {}
        return {
            "provider": "Pexels",
            "query": query,
            "video_id": video.get("id"),
            "source_page": video.get("url"),
            "download_url": link,
            "creator": user.get("name"),
            "license": "Pexels License (operator confirmed); commercial use subject to current terms",
            "license_url": PEXELS_LICENSE,
            "attribution": "Pexels video " + str(video.get("url", "")),
        }


class TTSService:
    """Speech for commentary only. Quran recitation must always remain source audio."""
    def __init__(self, voice: str | None = None):
        self.voice = (voice or os.getenv("TTS_VOICE", "ar-EG-ShakerNeural")).strip()
        if not self.voice:
            raise ServiceError("TTS_VOICE فارغ.")
        if not env_bool("TTS_TERMS_CONFIRMED"):
            raise ServiceError("أكد حقوق الاستخدام التجاري لمزوّد TTS عبر TTS_TERMS_CONFIRMED=true.")
        self.terms_url = _required_env("TTS_LICENSE_URL")
        if urlparse(self.terms_url).scheme != "https":
            raise ServiceError("TTS_LICENSE_URL يجب أن يكون HTTPS.")
        self._voice_checked = False

    async def _save(self, text: str, path: Path) -> None:
        import edge_tts
        if not getattr(self, "_voice_checked", False):
            voices = await edge_tts.list_voices()
            if self.voice not in {item.get("ShortName") for item in voices}:
                raise ServiceError(f"صوت TTS_VOICE غير متاح حاليًا: {self.voice}.")
            self._voice_checked = True
        await edge_tts.Communicate(text=text, voice=self.voice, rate="+0%", pitch="+0Hz").save(str(path))

    def synthesize(self, texts: dict[str, str], output_dir: Path) -> dict[str, Path]:
        forbidden = "صدق الله العظيم"
        if texts.get("seal") != forbidden or sum(value.count(forbidden) for value in texts.values() if isinstance(value, str)) != 1:
            raise ServiceError("يجب إنشاء «صدق الله العظيم» مرة واحدة بالضبط في المقطع المستقل seal.")
        for key, value in texts.items():
            if not isinstance(value, str) or not value.strip() or (key != "seal" and forbidden in value):
                raise ServiceError(f"نص TTS {key} فارغ أو يكرر العبارة المحجوزة.")
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        for name, text in texts.items():
            path = output_dir / f"tts-{name}.mp3"
            try:
                asyncio.run(self._save(text, path))
            except ServiceError:
                path.unlink(missing_ok=True)
                raise
            except Exception as exc:
                path.unlink(missing_ok=True)
                raise ServiceError(f"فشل توليد TTS للمقطع {name} ({type(exc).__name__}).") from None
            if not path.is_file() or path.stat().st_size < 512:
                path.unlink(missing_ok=True)
                raise ServiceError(f"ملف TTS للمقطع {name} غير صالح.")
            results[name] = path
        return results


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

        def test_quran_verse_response_checks_key_and_uthmani_text(self):
            payload = {"verse": {"verse_key": "2:152", "verse_number": 152, "chapter_id": 2,
                                 "id": 12345, "text_uthmani": "فَاذْكُرُونِي أَذْكُرْكُمْ",
                                 "text_imlaei": "فاذكروني أذكركم"}}
            client = self.ready_qf_client(self.FakeResponse(200, payload))
            verse = client.get_verse(2, 152)
            self.assertEqual(verse.key, "2:152")
            self.assertIn("text_uthmani", verse.__dict__)
            self.assertEqual(verse.text_imlaei, "فاذكروني أذكركم")
            self.assertIn("text_imlaei", client.session.get_calls[0][1]["params"]["fields"])

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
                                "segments": [[1, 10000, 11000], [2, 11000, 12000], [3, 12000, 13000]]}],
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
                self.assertEqual(recitation.segments, [[1, 0, 1000], [2, 1000, 2000], [3, 2000, 3000]])
                self.assertEqual(recitation.duration_hint, 3000)
                self.assertTrue(path.is_file())
                self.assertIn("10.000", ffmpeg.call_args.args[0])

        def test_word_timings_accept_contiguous_one_based_and_map_to_zero_based(self):
            got = validate_word_segments([[1, 0, 250], [2, 250, 700]], "بسم الله", 800)
            self.assertEqual([x["word_index"] for x in got], [0, 1])

        def test_word_timings_reject_missing_and_bad_rows(self):
            with self.assertRaises(ServiceError):
                validate_word_segments([[0, 0, 100]], "بسم الله", 800)
            with self.assertRaises(ServiceError):
                validate_word_segments([[0, 0, 300], [1, 200, 100]], "بسم الله", 800)

        def test_audio_transcription_must_match_verse_words_in_order(self):
            result = validate_audio_transcription("فاذكروني أذكركم واشكروا لي ولا تكفرون",
                                                   "فَاذْكُرُونِي أَذْكُرْكُمْ وَاشْكُرُوا لِي وَلَا تَكْفُرُونِ")
            self.assertEqual(result["status"], "exact_match")
            with self.assertRaises(ServiceError):
                validate_audio_transcription("فاذكروني ولا تكفرون", "فَاذْكُرُونِي أَذْكُرْكُمْ")
            with self.assertRaises(ServiceError):
                validate_audio_transcription("فاذكروني", "فَاذْكُرُونِي", uncertain=True)
            with self.assertRaises(ServiceError):
                validate_audio_transcription("فاذكروني آمين", "فَاذْكُرُونِي", contains_other_speech=True)

        def test_audio_transcription_accepts_imlaei_spelling_without_changing_display_text(self):
            result = validate_audio_transcription("الصلاة", ["الصلوة", "الصلاة"])
            self.assertEqual(result["status"], "exact_match")

        def test_generated_json_requires_five_distinct_scored_hooks(self):
            base = {"hook_candidates": [], "explanation": "شرح", "practical_benefit": "فائدة", "call_to_action": "شارك"}
            with self.assertRaises(ServiceError):
                validate_generated_content(base)

        def test_generated_json_accepts_five_hooks_and_rejects_seal_duplication(self):
            candidates = []
            for index in range(5):
                candidates.append({"text": f"تأمل معنى الآية رقم {index}", **{key: 4 for key in HOOK_SCORE_KEYS}})
            content = {"hook_candidates": candidates, "explanation": "شرح موثق", "practical_benefit": "فائدة عملية",
                       "call_to_action": "شارك الفائدة"}
            self.assertTrue(validate_generated_content(content)["hook"])
            content["explanation"] = "صدق الله العظيم"
            with self.assertRaises(ServiceError):
                validate_generated_content(content)

        def test_tafsir_license_fails_closed(self):
            with patch.dict(os.environ, {"TAFSIR_ID": "1", "TAFSIR_LICENSE_URL": "https://example.org/terms",
                                         "TAFSIR_COMMERCIAL_LICENSE_VERIFIED": "false", "QF_CLIENT_ID": "x", "QF_CLIENT_SECRET": "y"}):
                client = object.__new__(QuranFoundationClient)
                with self.assertRaises(ServiceError):
                    client.get_tafsir("1:1")

        def test_tafsir_requires_allowlisted_source_and_arabic_text(self):
            payload = {"tafsir": {"resource_id": 7, "verses": {"1:1": {"id": 99}},
                                   "resource_name": "مصدر تفسير موثق", "text": "هذا تفسير عربي موثق يشرح معنى الآية ضمن سياقها."}}
            client = self.ready_qf_client(self.FakeResponse(200, payload))
            env = {"TAFSIR_ID": "7", "TAFSIR_LICENSE_URL": "https://example.org/license",
                   "TAFSIR_COMMERCIAL_LICENSE_VERIFIED": "true", "TAFSIR_ALLOWLIST": "7|https://example.org/license"}
            with patch.dict(os.environ, env):
                result = client.get_tafsir("1:1")
            self.assertEqual(result.resource_id, 7)
            self.assertEqual(result.source_name, "مصدر تفسير موثق")

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

    unittest.main()
