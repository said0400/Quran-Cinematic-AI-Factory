"""Quran Cinematic AI Factory entry point, production history, and workflow orchestration."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from services import (
    FacebookReelsPublisher, GeminiContentService, MakeWebhookPublisher,
    QuranFoundationClient, ScriptVerse, ServiceError, YouTubePublisher, env_bool, make_tts_service,
    quran_words, validate_word_segments,
)
from procedural_background import (
    apply_images, build_script_timeline, pick_image_keys, render_background, shot_index, timeline_image_keys,
)
from scene_images import SceneImageService
from topics import Topic, TopicError, find_topic, load_topics
from video_engine import (
    VideoError, align_speech_words, build_script_video, build_verse_layout, media_duration, validate_video,
)

PROJECT_ROOT = Path(__file__).resolve().parent
# Spoken-length planning: how many words the narrator can say in the time left after the recited verses.
SPEECH_WORDS_PER_SECOND = 2.3
SPEECH_WORDS_FLOOR, SPEECH_WORDS_CAP = 45, 120
DESCRIPTION_LIMIT = 1900     # title (<=95) + blank line + description stays under the 2000-char caption limit
DEFAULT_TTS_VOICE = "ar-EG-ShakirNeural"
COMMENTARY_STYLES = ("captions", "card")
PLATFORM_KEYS = ("youtube", "facebook", "make")


class FactoryError(RuntimeError):
    pass


class CandidateRejected(FactoryError):
    """A deterministic reason to skip this verse (retrying the same verse cannot help)."""


@dataclass(frozen=True)
class Config:
    timezone_name: str
    daily_video_count: int
    width: int
    height: int
    fps: int
    target_min: int
    target_max: int
    max_candidates: int
    max_retries: int
    tts_voice: str
    output_dir: Path
    history_file: Path
    final_video_name: str
    dim_strength: float = 0.30
    max_recitation_seconds: int = 45
    commentary_style: str = "captions"
    max_scene_images: int = 14
    topics_file: Path = Path("data/topics.json")

    @classmethod
    def from_env(cls, root: Path = PROJECT_ROOT) -> "Config":
        def integer(name: str, default: int, minimum: int, maximum: int) -> int:
            raw = os.getenv(name, "").strip() or str(default)
            try:
                value = int(raw)
            except ValueError:
                raise FactoryError(f"قيمة {name} يجب أن تكون عددًا صحيحًا.") from None
            if not minimum <= value <= maximum:
                raise FactoryError(f"قيمة {name} خارج المجال {minimum}..{maximum}.")
            return value

        def decimal(name: str, default: float, minimum: float, maximum: float) -> float:
            raw = os.getenv(name, "").strip() or str(default)
            try:
                value = float(raw)
            except ValueError:
                raise FactoryError(f"قيمة {name} يجب أن تكون عددًا عشريًا.") from None
            if not minimum <= value <= maximum:
                raise FactoryError(f"قيمة {name} خارج المجال {minimum}..{maximum}.")
            return value
        timezone_name = os.getenv("TZ", "Africa/Casablanca").strip() or "Africa/Casablanca"
        try:
            ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError:
            raise FactoryError(f"المنطقة الزمنية غير معروفة: {timezone_name}.") from None
        output_dir = Path(os.getenv("OUTPUT_DIR", "output").strip() or "output")
        history_file = Path(os.getenv("HISTORY_FILE", "data/production_history.json").strip() or "data/production_history.json")
        if not output_dir.is_absolute():
            output_dir = root / output_dir
        if not history_file.is_absolute():
            history_file = root / history_file
        final_name = os.getenv("FINAL_VIDEO_NAME", "final_output.mp4").strip() or "final_output.mp4"
        if Path(final_name).name != final_name or not final_name.lower().endswith(".mp4"):
            raise FactoryError("FINAL_VIDEO_NAME يجب أن يكون اسم ملف MP4 فقط.")
        target_min = integer("TARGET_MIN_DURATION", 45, 10, 300)
        target_max = integer("TARGET_MAX_DURATION", 60, 10, 600)
        if target_min > target_max:
            raise FactoryError("TARGET_MIN_DURATION لا يمكن أن يتجاوز TARGET_MAX_DURATION.")
        width = integer("VIDEO_WIDTH", 1080, 360, 2160)
        height = integer("VIDEO_HEIGHT", 1920, 640, 3840)
        if width * 16 != height * 9 or width % 2 or height % 2:
            raise FactoryError("يجب أن تكون أبعاد الفيديو زوجية وتحافظ على نسبة 9:16.")
        commentary_style = os.getenv("COMMENTARY_STYLE", "").strip().lower() or "captions"
        if commentary_style not in COMMENTARY_STYLES:
            raise FactoryError("COMMENTARY_STYLE يجب أن يكون captions أو card.")
        topics_file = Path(os.getenv("TOPICS_FILE", "data/topics.json").strip() or "data/topics.json")
        if not topics_file.is_absolute():
            topics_file = root / topics_file
        return cls(
            timezone_name=timezone_name,
            daily_video_count=integer("DAILY_VIDEO_COUNT", 1, 1, 5),
            width=width,
            height=height,
            fps=integer("VIDEO_FPS", 24, 24, 60),
            target_min=target_min,
            target_max=target_max,
            max_candidates=integer("MAX_CANDIDATES_PER_RUN", 5, 1, 20),
            max_retries=integer("MAX_RETRIES", 3, 1, 8),
            tts_voice=os.getenv("TTS_VOICE", DEFAULT_TTS_VOICE).strip() or DEFAULT_TTS_VOICE,
            output_dir=output_dir,
            history_file=history_file,
            final_video_name=final_name,
            dim_strength=decimal("VIDEO_DIM_STRENGTH", 0.30, 0.0, 0.8),
            max_recitation_seconds=integer("MAX_RECITATION_SECONDS", 45, 10, 300),
            commentary_style=commentary_style,
            max_scene_images=integer("MAX_SCENE_IMAGES", 14, 1, 30),
            topics_file=topics_file,
        )


def local_now(config: Config) -> datetime:
    return datetime.now(ZoneInfo(config.timezone_name))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_generation_environment() -> None:
    required = [
        "GEMINI_API_KEY", "QF_CLIENT_ID", "QF_CLIENT_SECRET", "QURAN_TEXT_LICENSE_URL", "CHAPTER_RECITER_ID",
        "RECITATION_LICENSE_URL", "RECITATION_ALLOWLIST", "TAFSIR_ID", "TAFSIR_LICENSE_URL",
        "TAFSIR_ALLOWLIST", "TTS_LICENSE_URL",
    ]
    missing = [key for key in required if not os.getenv(key, "").strip()]
    if missing:
        raise FactoryError("إعدادات إنتاج مطلوبة غير موجودة: " + ", ".join(missing))
    if env_bool("MAKE_PUBLISH_ENABLED"):
        absent = [key for key in ("MAKE_WEBHOOK_URL", "GITHUB_TOKEN", "GITHUB_REPOSITORY")
                  if not os.getenv(key, "").strip()]
        if absent:
            raise FactoryError("إعدادات Make مطلوبة غير موجودة: " + ", ".join(absent))
    if urlparse(os.getenv("QURAN_TEXT_LICENSE_URL", "").strip()).scheme != "https":
        raise FactoryError("QURAN_TEXT_LICENSE_URL يجب أن يكون رابط HTTPS.")
    confirmations = {
        "QURAN_TEXT_LICENSE_CONFIRMED": "حق إعادة نشر النص القرآني من المصدر المحدد",
        "RECITATION_LICENSE_CONFIRMED": "ترخيص إعادة نشر التلاوة",
        "TAFSIR_COMMERCIAL_LICENSE_VERIFIED": "ترخيص التفسير التجاري وإعادة النشر",
        "TTS_TERMS_CONFIRMED": "حقوق استخدام TTS",
    }
    unconfirmed = [label for key, label in confirmations.items() if not env_bool(key)]
    if unconfirmed:
        raise FactoryError("لم تُؤكد المتطلبات الحقوقية التالية: " + "، ".join(unconfirmed))


class ProductionHistory:
    """Atomic JSON history updates with a POSIX process lock."""
    def __init__(self, path: Path):
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + ".lock")

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "runs": []}
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            raise FactoryError(f"سجل الإنتاج تالف أو غير قابل للقراءة: {self.path}") from None
        if not isinstance(payload, dict) or payload.get("version") != 1 or not isinstance(payload.get("runs"), list):
            raise FactoryError("صيغة سجل الإنتاج غير معروفة؛ لن يُستبدل الملف.")
        return payload

    def load(self) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            result = self._read_unlocked()
            if not self.path.exists():
                self._write_unlocked(result)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return result

    def _write_unlocked(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
            dir_fd = os.open(self.path.parent, os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def upsert(self, run: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            payload = self._read_unlocked()
            runs = payload["runs"]
            index = next((i for i, row in enumerate(runs) if row.get("run_id") == run.get("run_id")), None)
            if index is None:
                runs.append(run)
            else:
                # Preserve earlier confirmed platform IDs and source data if a later update omitted them.
                prior = runs[index]
                merged = {**prior, **run}
                old_platforms = prior.get("platforms", {})
                new_platforms = run.get("platforms", {})
                merged["platforms"] = {**old_platforms, **new_platforms}
                runs[index] = merged
            self._write_unlocked(payload)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def get_runs(self) -> list[dict[str, Any]]:
        return self.load()["runs"]


def row_has_any_publication(row: dict[str, Any]) -> bool:
    platforms = row.get("platforms", {})
    return any(isinstance(value, dict) and value.get("status") == "published" and value.get("video_id")
               for value in platforms.values())


def row_has_unresolved_submission(row: dict[str, Any]) -> bool:
    uncertain_states = {"upload_uncertain", "upload_session_uncertain", "upload_session_created",
                        "uploading_unknown", "uploaded_unconfirmed", "pending"}
    for value in row.get("platforms", {}).values():
        if not isinstance(value, dict) or value.get("status") == "published":
            continue
        if value.get("video_id") or value.get("status") in uncertain_states:
            return True
    return False


def choose_auto_candidates(history_rows: list[dict[str, Any]], date_key: str, topics: list[Topic], limit: int) -> list[Topic]:
    """Unused topics in a date-rotated order. A topic with any confirmed or unresolved publication is never chosen
    again: pending and failed rows stay addressable by publish-only recovery instead of producing duplicates."""
    seen = {row.get("verse_key") for row in history_rows
            if row_has_any_publication(row) or row_has_unresolved_submission(row)}
    if not topics:
        return []
    start = sum(ord(ch) for ch in date_key) % len(topics)
    rotated = topics[start:] + topics[:start]
    return [topic for topic in rotated if topic.row_key not in seen][:limit]


LOW_TOPIC_WARNING = 3        # warn when fewer unused topics than this remain


def blocked_topics(history_rows: list[dict[str, Any]], topics: list[Topic]) -> list[str]:
    """Why each unavailable topic is unavailable, e.g. 'cave-309 (published)' or 'x (نشر غير محسوم: make=uploaded_unconfirmed)'."""
    result = []
    for topic in topics:
        rows = [row for row in history_rows if row.get("verse_key") == topic.row_key
                and (row_has_any_publication(row) or row_has_unresolved_submission(row))]
        if not rows:
            continue
        row = max(rows, key=lambda item: item.get("updated_at", ""))
        if row_has_any_publication(row):
            result.append(f"{topic.id} (منشور)")
        else:
            states = ", ".join(f"{name}={value.get('status')}" for name, value in row.get("platforms", {}).items()
                               if isinstance(value, dict) and value.get("status") not in {"not_started", "disabled", "published"})
            result.append(f"{topic.id} (نشر غير محسوم: {states or 'غير معروف'})")
    return result


def already_processed_today(history_rows: list[dict[str, Any]], local_date: str, daily_target: int = 1) -> bool:
    today = [row for row in history_rows if row.get("mode") == "auto" and row.get("local_date") == local_date]
    completed = sum(1 for row in today if row.get("status") == "published")
    if completed >= daily_target:
        return True
    # Known pre-publication failures can be retried the same day. A crashed/in-progress
    # row or uncertain/partial submission blocks fresh uploads until it is recovered.
    return any(row.get("status") in {"started", "generated", "validated", "partially_published"}
               or row_has_unresolved_submission(row) for row in today)


def new_run_record(run_id: str, topic_id: str, mode: str, local_date: str, attempt: int) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "verse_key": f"topic:{topic_id}",        # the key under which history remembers the topic
        "topic_id": topic_id,
        "mode": mode,
        "local_date": local_date,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "source_tafsir": None,
        "media_sources": [],
        "verification": {"status": "not_run", "checks": {}},
        "video_sha256": None,
        "video_path": None,
        "publication_metadata": None,
        "platforms": {
            "youtube": {"status": "not_started", "video_id": None, "attempts": 0},
            "facebook": {"status": "not_started", "video_id": None, "attempts": 0},
            "make": {"status": "not_started", "video_id": None, "attempts": 0},
        },
        "attempt_count": attempt,
        "last_error": None,
        "status": "started",
    }


def update_record(history: ProductionHistory, row: dict[str, Any], report_dir: Path | None = None) -> None:
    row["updated_at"] = utc_now_iso()
    history.upsert(row)
    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{row['run_id']}-report.json"
        temp = report_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, report_path)
        row["report_path"] = str(report_path)
        history.upsert(row)


def publisher_factory(name: str) -> Any:
    if name == "youtube":
        return YouTubePublisher()
    if name == "facebook":
        return FacebookReelsPublisher()
    if name == "make":
        return MakeWebhookPublisher()
    raise FactoryError(f"منصة غير معروفة: {name}")


def publish_platforms(video: Path, row: dict[str, Any], history: ProductionHistory,
                      report_dir: Path, title: str, description: str,
                      make_publisher: Callable[[str], Any] = publisher_factory,
                      retries: int = 3) -> dict[str, Any]:
    """Publish destinations independently; never re-upload a destination already confirmed.

    With MAKE_PUBLISH_ENABLED the only target is Make.com (it fans out to Facebook, YouTube,
    Instagram and TikTok), so the direct YouTube/Facebook publishers are marked disabled. Without it,
    YouTube is always a target and Facebook only when META_PUBLISH_ENABLED is set.
    """
    if env_bool("MAKE_PUBLISH_ENABLED"):
        targets = ["make"]
        for legacy in ("youtube", "facebook"):
            state = row.setdefault("platforms", {}).get(legacy)
            if not isinstance(state, dict) or state.get("status") == "not_started":
                row["platforms"][legacy] = {"status": "disabled", "video_id": None, "attempts": 0}
    else:
        targets = ["youtube"]
        if env_bool("META_PUBLISH_ENABLED"):
            targets.append("facebook")
        else:
            facebook_state = row.setdefault("platforms", {}).get("facebook")
            if not isinstance(facebook_state, dict) or facebook_state.get("status") == "not_started":
                row["platforms"]["facebook"] = {"status": "disabled", "video_id": None, "attempts": 0}
    for name in targets:
        prior = row.setdefault("platforms", {}).setdefault(name, {"status": "not_started", "video_id": None, "attempts": 0})
        if prior.get("status") == "published" and prior.get("video_id"):
            continue
        try:
            publisher = make_publisher(name)
        except Exception as exc:
            prior.update({"status": "failed", "last_error": f"تعذر تهيئة {name}: {type(exc).__name__}"})
            row["last_error"] = prior["last_error"]
            update_record(history, row, report_dir)
            continue
        # First upload is attempted once: a transport failure may happen after the remote server
        # accepted the media, and retrying blindly could create a duplicate. Known IDs are verified
        # repeatedly without creating a second post.
        max_attempts = max(1, retries) if prior.get("video_id") else 1
        for attempt in range(max_attempts):
            prior["attempts"] = int(prior.get("attempts", 0)) + 1
            try:
                result = publisher.publish_or_verify(video, title, description, previous=prior)
                if not isinstance(result, dict) or result.get("status") not in {"published", "pending"}:
                    raise ServiceError(f"استجابة {name} لا تثبت النشر أو حالة الانتظار.")
                prior.update(result)
                prior["attempts"] = int(prior.get("attempts", 0))
                prior.pop("last_error", None)
                update_record(history, row, report_dir)
                if result["status"] == "published":
                    break
                if attempt + 1 < max_attempts:
                    continue
            except Exception as exc:
                state = getattr(exc, "platform_state", None)
                if isinstance(state, dict):
                    prior.update(state)
                uncertain = prior.get("status") in {"upload_uncertain", "upload_session_uncertain", "upload_session_created",
                                                     "uploading_unknown", "uploaded_unconfirmed", "pending"}
                prior.update({"status": prior.get("status") if prior.get("video_id") or uncertain else "failed",
                              "last_error": str(exc)[:400]})
                row["last_error"] = prior["last_error"]
                update_record(history, row, report_dir)
                # Do not retry a fresh/uncertain submission without a server-issued ID.
                if not prior.get("video_id"):
                    break
                if attempt + 1 >= max_attempts:
                    break
    statuses = [row.get("platforms", {}).get(key, {}).get("status") for key in targets]
    if row_has_unresolved_submission(row):
        row["status"] = "partially_published" if any(status == "published" for status in statuses) else "failed"
    elif all(status == "published" for status in statuses):
        row["status"] = "published"
    elif any(status == "published" for status in statuses):
        row["status"] = "partially_published"
    else:
        row["status"] = "failed"
    update_record(history, row, report_dir)
    return row


def _record_sources(items: list[dict[str, Any]], tafsirs: dict[str, Any], media_items: list[dict[str, Any]],
                    tts_terms: str, tts_voice: str, tts_provider: str = "edge-tts") -> list[dict[str, Any]]:
    """The licence trail: every verse text and recitation used, the tafsir read, generated pictures, and the voice."""
    sources: list[dict[str, Any]] = []
    for item in items:
        verse, recitation = item["verse"], item["recitation"]
        sources.append({"kind": "quran_text", "provider": "Quran Foundation Content API v4", "verse_key": verse.key,
                        "fragment": item.get("label"), "source_url": f"{QF_BASE}/verses/by_key/{verse.key}",
                        "license_url": os.getenv("QURAN_TEXT_LICENSE_URL"),
                        "license_status": "operator-confirmed right to reproduce source text in published video"})
        sources.append({"kind": "recitation", "provider": "Quran Foundation Content API v4", "verse_key": verse.key,
                        "fragment": item.get("label"), "reciter_id": recitation.reciter_id,
                        "reciter_name": recitation.reciter_name, "source_url": recitation.source_url,
                        "audio_url": recitation.audio_url, "license_url": os.getenv("RECITATION_LICENSE_URL"),
                        "license_status": "operator-confirmed commercial reproduction permission"})
    for key, tafsir in tafsirs.items():
        sources.append({"kind": "tafsir", "verse_key": key, "source_name": tafsir.source_name,
                        "resource_id": tafsir.resource_id, "source_url": tafsir.source_url, "license_url": tafsir.license_url,
                        "license_status": "operator-confirmed commercial reproduction permission"})
    sources.extend(media_items)
    sources.append({"kind": "tts", "provider": tts_provider, "voice": tts_voice,
                    "license_url": tts_terms, "license_status": "operator-confirmed current usage terms"})
    return sources

# Set immediately below to keep URL constants in reports while avoiding accidental config-dependent values.
QF_BASE = os.getenv(
    "QF_API_BASE_URL",
    "https://apis-prelive.quran.foundation/content/api/v4"
    if os.getenv("QF_ENV", "production").strip().lower() == "prelive"
    else "https://apis.quran.foundation/content/api/v4",
).rstrip("/")


def speech_word_budget(target_max: float, recitation_total: float) -> tuple[int, int]:
    """(minimum, maximum) words the narrator may say, from the time left after the recited verses."""
    spare = float(target_max) - float(recitation_total)
    upper = max(SPEECH_WORDS_FLOOR, min(SPEECH_WORDS_CAP, int(spare * SPEECH_WORDS_PER_SECOND)))
    lower = min(upper, max(40, int(upper * 0.75)))
    return lower, upper


def compose_description(ai: dict[str, Any], limit: int = DESCRIPTION_LIMIT) -> str:
    """Long description + search keywords + hashtags. Never names a source; keywords are trimmed first."""
    body = ai["description"].strip()
    hashtags = " ".join(ai["hashtags"])
    keywords = list(ai["keywords"])
    while True:
        parts = [body]
        if keywords:
            parts.append("الكلمات المفتاحية: " + "، ".join(keywords))
        parts.append(hashtags)
        text = "\n\n".join(parts)
        if len(text) <= limit or not keywords:
            break
        keywords.pop()
    if len(text) > limit:
        allowed = max(0, limit - len(hashtags) - 2)
        text = body[:allowed].rstrip() + "\n\n" + hashtags
    return text


def _title_description(ai: dict[str, Any]) -> tuple[str, str]:
    return ai["title"].strip()[:100], compose_description(ai)[:5000]


def generate_candidate(topic: Topic, attempt: int, mode: str, config: Config, history: ProductionHistory,
                       local_date: str, previous_row: dict[str, Any] | None = None) -> tuple[dict[str, Any], bool]:
    run_id = f"{local_date}-{topic.id}-{uuid.uuid4().hex[:8]}"
    row = new_run_record(run_id, topic.id, mode, local_date, attempt)
    if previous_row:
        # Reprocessing may create a new artifact, but previously confirmed platform posts and IDs
        # remain confirmed and are never uploaded again for this topic.
        row["platforms"] = row["platforms"] | {
            key: dict(value) for key, value in previous_row.get("platforms", {}).items()
            if key in PLATFORM_KEYS and isinstance(value, dict)
        }
    report_dir = config.output_dir / "reports"
    update_record(history, row, report_dir)
    validated_video = False
    try:
        validate_generation_environment()
        quran = QuranFoundationClient()
        checks = row["verification"]["checks"]
        work_dir = config.output_dir / f"work-{run_id}"
        work_dir.mkdir(parents=True, exist_ok=True)

        # 1) The evidence: every verse (or fragment) is fetched, cut and verified before any paid call.
        items: list[dict[str, Any]] = []
        tafsirs: dict[str, Any] = {}
        for ref in topic.verses:
            verse = quran.get_verse(ref.chapter, ref.verse)
            all_words = quran_words(verse.text_uthmani)
            if ref.words is not None:
                if ref.last_word > len(all_words):
                    raise CandidateRejected(f"مقطع الآية {ref.label} يتجاوز عدد كلمات الآية ({len(all_words)}).")
                fragment_text = " ".join(all_words[ref.first_word - 1:ref.last_word])
            else:
                fragment_text = verse.text_uthmani
            try:
                build_verse_layout(fragment_text, verse.chapter_name, verse.verse, config.width, config.height)
            except VideoError as exc:
                raise CandidateRejected(f"تعذر عرض الآية على البطاقة: {exc}") from None
            recitation, audio_file = quran.get_recitation(ref.chapter, ref.verse, work_dir, ref.words, verse.text_uthmani)
            seconds = media_duration(audio_file)
            try:
                timed_words = validate_word_segments(recitation.segments, fragment_text, int(round(seconds * 1000)))
            except ServiceError as exc:
                raise CandidateRejected(str(exc)) from None
            items.append({"ref": ref, "label": ref.label, "verse": verse, "recitation": recitation, "audio": audio_file,
                          "text": fragment_text, "seconds": seconds, "timed_words": timed_words})
            if verse.key not in tafsirs:
                tafsirs[verse.key] = quran.get_tafsir(verse.key)
        recitation_total = sum(item["seconds"] for item in items)
        if mode == "auto" and recitation_total > config.max_recitation_seconds:
            raise CandidateRejected(
                f"زمن التلاوة الكلي أطول من الحد ({recitation_total:.1f}s > {config.max_recitation_seconds}s).")
        checks["topic"] = {"id": topic.id, "question": topic.question, "facts": len(topic.facts)}
        checks["verses"] = [{"label": item["label"], "words": len(item["timed_words"]), "seconds": round(item["seconds"], 3)}
                            for item in items]
        checks["recitation_total_seconds"] = round(recitation_total, 3)
        checks["audio_content_identity"] = {
            "status": "not_independently_transcribed",
            "audio_sent_to_gemini": False,
            "basis": "Quran Foundation verse key and word-level timestamps only",
        }
        row["source_tafsir"] = [{"verse_key": key, "source_name": tafsir.source_name, "resource_id": tafsir.resource_id,
                                 "source_url": tafsir.source_url, "license_url": tafsir.license_url,
                                 "commercial_license_verified_by_operator": tafsir.license_verified_by_operator}
                                for key, tafsir in tafsirs.items()]
        update_record(history, row, report_dir)

        # 2) The script: the idea first, the verses placed inside it as evidence.
        budget = speech_word_budget(config.target_max, recitation_total)
        script_verses = [ScriptVerse(index=number, key=item["verse"].key, text=item["text"],
                                     chapter_name=item["verse"].chapter_name, verse_number=item["verse"].verse,
                                     seconds=item["seconds"]) for number, item in enumerate(items, start=1)]
        ai = GeminiContentService().create_and_review(topic, script_verses, tafsirs, budget)
        row["visual_plan"] = ai["visual_plan"]
        row["script"] = {"hook": ai["hook"], "parts": ai["parts"], "call_to_action": ai["call_to_action"],
                         "spoken_words": ai["spoken_words"], "spoken_word_budget": list(budget)}
        checks["model_content_schema"] = True
        checks["model_source_review"] = ai["review"]
        checks["content"] = {
            "spoken_words": ai["spoken_words"],
            "spoken_word_budget": list(budget),
            "hook_words": len(ai["hook"].split()),
            "parts": [part["type"] for part in ai["parts"]],
            "keywords": len(ai["keywords"]),
            "hashtags": len(ai["hashtags"]),
            "visual_shots": len(ai["visual_plan"]["scenes"]),
        }
        generated_texts = (ai["hook"], *(part["text"] for part in ai["parts"] if part["type"] == "speech"),
                           ai["call_to_action"], ai["title"], ai["description"],
                           " ".join(ai["keywords"]), " ".join(ai["hashtags"]))
        checks["honorific_occurrences_in_generated_text"] = sum(text.count("صدق الله العظيم") for text in generated_texts)
        if checks["honorific_occurrences_in_generated_text"]:
            raise FactoryError("ظهرت العبارة المحجوزة ضمن النص المولّد.")

        # 3) The voice: one clip per stretch of speech; the verses keep the reciter's own audio.
        tts = make_tts_service(config.tts_voice)
        speech_texts = {"hook": ai["hook"]}
        order: list[dict[str, Any]] = [{"kind": "speech", "name": "hook", "role": "hook", "text": ai["hook"]}]
        talk = 0
        for part in ai["parts"]:
            if part["type"] == "speech":
                talk += 1
                speech_texts[f"talk-{talk}"] = part["text"]
                order.append({"kind": "speech", "name": f"talk-{talk}", "role": "body", "text": part["text"]})
            else:
                item = items[part["verse_index"] - 1]
                order.append({"kind": "verse", "name": f"verse-{part['verse_index']}", "text": item["text"],
                              "audio": item["audio"], "segments": item["timed_words"],
                              "chapter_name": item["verse"].chapter_name, "verse_number": item["verse"].verse})
        speech_texts["closing"] = ai["call_to_action"]
        order.append({"kind": "speech", "name": "closing", "role": "closing", "text": ai["call_to_action"]})
        speech_files = tts.synthesize(speech_texts, work_dir / "speech")
        for segment in order:
            if segment["kind"] == "speech":
                segment["audio"] = speech_files[segment["name"]]
                segment["boundaries"] = tts.word_boundaries.get(segment["name"])

        # 4) The timeline: where each segment sits, and where the narrator says each new phrase.
        timeline_segments: list[dict[str, Any]] = []
        cursor = 0.0
        timing_sources: dict[str, str] = {}
        for segment in order:
            length = media_duration(segment["audio"])
            entry: dict[str, Any] = {"kind": segment["kind"], "role": segment.get("role"), "start": cursor, "end": cursor + length}
            if segment["kind"] == "speech":
                entry["timings"], timing_sources[segment["name"]] = align_speech_words(
                    segment["text"], segment.get("boundaries"), length)
            timeline_segments.append(entry)
            cursor += length
        video_duration = max(cursor, float(config.target_min))

        # 5) Pictures: a new one about every 2.5-3 s. Cut the shots first, then generate only the ones that survive the cut.
        timeline = build_script_timeline(ai["visual_plan"], timeline_segments, video_duration)
        shots = shot_index(ai["visual_plan"])
        wanted = pick_image_keys(timeline_image_keys(timeline), config.max_scene_images)
        images = SceneImageService()
        scene_images = images.generate_jobs(
            [(key, shots[key].get("image_prompt")) for key in wanted], work_dir / "images",
            config.width, config.height, theme=ai["visual_plan"].get("visual_theme", ""))
        apply_images(timeline, scene_images)
        checks["scene_images"] = {
            "shots": len(timeline_image_keys(timeline)),
            "timeline_entries": len(timeline),
            "requested": len(wanted),
            "generated": len(scene_images),
            "entries_reusing_a_neighbour_picture": sum(1 for entry in timeline if entry.get("reused_image")),
            "providers": sorted({item["provider"] for item in scene_images.values()}),
            "notes": images.notes[:12],
        }
        media_items: list[dict[str, Any]] = []
        for key, item in scene_images.items():
            media_items.append({"kind": "ai_generated_image", "scene": key, "provider": item["provider"],
                                "model": item["model"], "prompt": item["prompt"], "sha256": item["sha256"],
                                "license_url": os.getenv("IMAGE_GEN_LICENSE_URL"),
                                "license_status": "operator-confirmed current provider terms; AI-generated"})
        background = work_dir / "background.mp4"
        background_info = render_background(
            background, config.width, config.height, config.fps, video_duration, timeline,
            theme=f"{topic.id}|{ai['title']}")
        background_info["speech_timing_sources"] = timing_sources
        checks["background"] = background_info
        update_record(history, row, report_dir)

        # 6) The video.
        output_path = config.output_dir / config.final_video_name
        video_validation = build_script_video(
            background, order, output_path, width=config.width, height=config.height, fps=config.fps,
            target_min_duration=config.target_min, target_max_duration=config.target_max,
            dim_strength=config.dim_strength, commentary_style=config.commentary_style)
        row["video_path"] = str(output_path)
        row["video_sha256"] = sha256_file(output_path)
        checks["video"] = video_validation
        row["verification"]["status"] = "validated"
        row["status"] = "validated"
        row["media_sources"] = _record_sources(
            [{"verse": item["verse"], "recitation": item["recitation"], "label": item["label"]} for item in items],
            tafsirs, media_items, tts.terms_url, tts.voice, getattr(tts, "provider", "edge-tts"))
        row["media_sources"].insert(0, {"kind": "procedural_background", "provider": "Pillow + FFmpeg",
                                        "external_assets": False, "metadata": background_info})
        font_source = video_validation.get("font_source", {})
        row["media_sources"].append({"kind": "font", **font_source})
        validated_video = True
        update_record(history, row, report_dir)
        title, description = _title_description(ai)
        row["publication_metadata"] = {"title": title, "description": description}
        update_record(history, row, report_dir)
        publish_platforms(output_path, row, history, report_dir, title, description,
                          retries=config.max_retries)
        return row, validated_video
    except Exception as exc:
        row["last_error"] = (str(exc) or type(exc).__name__)[:500]
        row["attempt_count"] = attempt
        row["permanent_failure"] = isinstance(exc, CandidateRejected)
        row["status"] = "partially_published" if row_has_any_publication(row) else "failed"
        if not row.get("verification", {}).get("status") == "validated":
            row.setdefault("verification", {})["status"] = "failed"
        update_record(history, row, report_dir)
        print(f"فشل الموضوع {topic.id}: {row['last_error']}", file=sys.stderr)
        return row, validated_video


def find_latest_row(history_rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    matches = [row for row in history_rows if row.get("verse_key") == key and row.get("video_sha256")]
    return max(matches, key=lambda row: row.get("updated_at", "")) if matches else None


def publish_only(args: argparse.Namespace, config: Config, history: ProductionHistory) -> int:
    if not args.topic or not args.input_video:
        raise FactoryError("--publish-only يتطلب --topic و--input-video.")
    key = f"topic:{args.topic.strip()}"
    video = Path(args.input_video).expanduser().resolve()
    rows = history.get_runs()
    row = find_latest_row(rows, key)
    if not row:
        raise FactoryError("لا يوجد سجل فيديو مُتحقق منه لهذا الموضوع؛ لن يُنشأ منشور جديد.")
    validate_video(video, config.width, config.height, config.fps)
    digest = sha256_file(video)
    if digest != row.get("video_sha256"):
        raise FactoryError("بصمة ملف الفيديو لا تطابق النسخة المسجلة؛ رُفضت إعادة النشر.")
    report_dir = config.output_dir / "reports"
    metadata = row.get("publication_metadata") or {}
    row = publish_platforms(video, row, history, report_dir,
                            metadata.get("title", f"تدبر قرآني | {args.topic}"),
                            metadata.get("description", f"فيديو تدبر قرآني. الموضوع: {args.topic}."),
                            retries=config.max_retries)
    print(f"حالة إعادة المحاولة للموضوع {args.topic}: {row['status']}")
    return 0 if row["status"] == "published" else 1


def run(args: argparse.Namespace) -> int:
    try:
        config = Config.from_env()
        now_local = local_now(config)
        local_date = now_local.date().isoformat()
        history = ProductionHistory(config.history_file)
        history_rows = history.load()["runs"]
        if args.publish_only:
            return publish_only(args, config, history)
        topics = load_topics(config.topics_file)
        if args.mode == "manual":
            if not args.topic:
                raise FactoryError("الوضع اليدوي يتطلب --topic.")
            candidates = [find_topic(topics, args.topic)]
        elif args.mode == "auto":
            if already_processed_today(history_rows, local_date, config.daily_video_count):
                print(f"يوجد تشغيل تلقائي مسجل لهذا اليوم في {config.timezone_name}؛ لن يُنشأ فيديو إضافي.")
                return 0
            candidates = choose_auto_candidates(history_rows, local_date, topics, config.max_candidates)
        else:
            raise FactoryError("--mode يجب أن يكون auto أو manual.")
        if args.mode == "manual":
            existing = find_latest_row(history_rows, candidates[0].row_key)
            if existing and row_has_unresolved_submission(existing):
                raise FactoryError("لدى هذا الموضوع جلسة نشر غير محسومة؛ استخدم --publish-only للتحقق من معرّف المنصة قبل إنشاء أي منشور جديد.")
            if existing and row_has_any_publication(existing) and not args.reprocess:
                raise FactoryError("هذا الموضوع له منشور مؤكد سابقًا. استخدم --publish-only لإعادة التحقق أو --reprocess لتوليد جديد؛ لن يتكرر النشر تلقائيًا.")
        if not candidates:
            blocked = blocked_topics(history_rows, topics)
            raise FactoryError(
                f"لا توجد مواضيع متاحة في ملف المواضيع ({len(topics)} موضوعًا). "
                + (("المستبعدة: " + "، ".join(blocked) + ". ") if blocked else "")
                + "أضف مواضيع جديدة إلى data/topics.json (لكل موضوع فيديو واحد). لم يُنشأ فيديو.")
        if args.mode == "auto":
            unused = len(choose_auto_candidates(history_rows, local_date, topics, len(topics)))
            if unused < LOW_TOPIC_WARNING:
                print(f"تنبيه: بقي {unused} موضوعًا غير منشور فقط من {len(topics)}؛ أضف مواضيع قبل أن ينفد البنك.", file=sys.stderr)
        candidate_errors = []
        daily_target = config.daily_video_count if args.mode == "auto" else 1
        completed_today = sum(1 for row in history_rows if row.get("mode") == "auto"
                              and row.get("local_date") == local_date and row.get("status") == "published")
        if args.mode == "auto" and completed_today >= daily_target:
            print(f"اكتمل العدد اليومي ({completed_today}/{daily_target}) في {config.timezone_name}.")
            return 0
        successful_this_run = 0
        for topic in candidates:
            if (any(row.get("verse_key") == topic.row_key and
                    (row_has_any_publication(row) or row_has_unresolved_submission(row)) for row in history_rows)
                    and not (args.mode == "manual" and args.reprocess)):
                continue
            attempts = config.max_retries if args.mode == "auto" else 1
            previous_row = find_latest_row(history_rows, topic.row_key) if args.reprocess else None
            for attempt in range(1, attempts + 1):
                row, validated = generate_candidate(topic, attempt, args.mode, config, history, local_date, previous_row)
                if row.get("status") == "published":
                    print(f"اكتمل إنشاء الفيديو والتحقق والنشر للموضوع {topic.id}.")
                    successful_this_run += 1
                    history_rows = history.get_runs()
                    if completed_today + successful_this_run >= daily_target:
                        return 0
                    break
                if row.get("status") == "partially_published":
                    print("نُشر الفيديو على منصة واحدة على الأقل؛ لا تعِد إنشاءه. يمكن استعادة الملف نفسه ثم استخدام --publish-only للمنصة غير المؤكدة.", file=sys.stderr)
                    return 1
                if validated:
                    print("الفيديو اجتاز الفحوص لكن النشر لم يتأكد؛ أوقف الإنتاج لتجنب إنشاء منشور مكرر.", file=sys.stderr)
                    return 1
                candidate_errors.append(row.get("last_error", "فشل غير محدد"))
                if args.mode == "manual":
                    return 1
                if row.get("permanent_failure"):
                    break          # retrying the same topic cannot help; move to the next candidate
            # Automatic mode moves to another candidate only after a pre-publication failure.
        if completed_today + successful_this_run >= daily_target:
            return 0
        print("لم يكتمل العدد اليومي المطلوب. آخر سبب: " + (candidate_errors[-1] if candidate_errors else "لا يوجد مرشح صالح."), file=sys.stderr)
        return 1
    except (FactoryError, ServiceError, VideoError, TopicError) as exc:
        print(f"فشل التشغيل: {str(exc)[:700]}", file=sys.stderr)
        return 1
    except Exception as exc:
        # Final safeguard: avoid printing environment values or raw SDK/network exceptions.
        print(f"فشل غير متوقع ({type(exc).__name__}). راجع سجل التشغيل دون مشاركة الأسرار.", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quran Cinematic AI Factory")
    parser.add_argument("--mode", choices=("auto", "manual"), default=os.getenv("MODE", "auto").strip() or "auto")
    parser.add_argument("--topic", help="معرّف الموضوع من ملف المواضيع في الوضع اليدوي")
    parser.add_argument("--reprocess", action="store_true", help="السماح بإعادة إنشاء فيديو موضوع سبق نشره دون تكرار الوجهات المنشورة")
    parser.add_argument("--publish-only", action="store_true", help="استئناف الوجهات غير المؤكدة باستخدام ملف الفيديو المطابق للبصمة")
    parser.add_argument("--input-video", help="ملف فيديو مطابق لبصمة سجل الإنتاج عند استخدام --publish-only")
    parser.add_argument("--test", action="store_true", help="تشغيل اختبارات الوحدة المضمنة دون اتصال خارجي")
    return parser


# Embedded tests keep the repository at a small, fixed set of source files.
if __name__ == "__main__":
    if "--test" in sys.argv:
        import unittest
        from unittest.mock import patch

        from services import TafsirSource

        THIS = sys.modules[__name__]      # the module under test is __main__, not "main"

        class MainTests(unittest.TestCase):
            def test_environment_parsing_strips_timezone_whitespace(self):
                with patch.dict(os.environ, {"TZ": " Africa/Casablanca ", "OUTPUT_DIR": "out", "HISTORY_FILE": "data/history.json"}):
                    self.assertEqual(Config.from_env(Path("/tmp/project")).timezone_name, "Africa/Casablanca")

            def test_environment_rejects_minimum_duration_above_maximum(self):
                env = {"TARGET_MIN_DURATION": "70", "TARGET_MAX_DURATION": "60"}
                with patch.dict(os.environ, env):
                    with self.assertRaises(FactoryError):
                        Config.from_env(Path("/tmp/project"))

            def test_environment_rejects_non_vertical_dimensions(self):
                with patch.dict(os.environ, {"VIDEO_WIDTH": "1080", "VIDEO_HEIGHT": "1080"}):
                    with self.assertRaises(FactoryError):
                        Config.from_env(Path("/tmp/project"))

            def test_default_voice_is_the_valid_microsoft_name(self):
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("TTS_VOICE", None)
                    self.assertEqual(Config.from_env(Path("/tmp/project")).tts_voice, "ar-EG-ShakirNeural")

            def test_dim_strength_defaults_parses_and_rejects_out_of_range(self):
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("VIDEO_DIM_STRENGTH", None)
                    self.assertEqual(Config.from_env(Path("/tmp/project")).dim_strength, 0.30)
                with patch.dict(os.environ, {"VIDEO_DIM_STRENGTH": "0.45"}):
                    self.assertEqual(Config.from_env(Path("/tmp/project")).dim_strength, 0.45)
                with patch.dict(os.environ, {"VIDEO_DIM_STRENGTH": ""}):
                    self.assertEqual(Config.from_env(Path("/tmp/project")).dim_strength, 0.30)
                with patch.dict(os.environ, {"VIDEO_DIM_STRENGTH": "0"}):
                    self.assertEqual(Config.from_env(Path("/tmp/project")).dim_strength, 0.0)
                for bad in ("0.95", "-0.1", "dark"):
                    with patch.dict(os.environ, {"VIDEO_DIM_STRENGTH": bad}):
                        with self.assertRaises(FactoryError):
                            Config.from_env(Path("/tmp/project"))

            def test_commentary_style_defaults_to_captions_and_rejects_unknown_values(self):
                with patch.dict(os.environ, {"COMMENTARY_STYLE": ""}):
                    self.assertEqual(Config.from_env(Path("/tmp/project")).commentary_style, "captions")
                with patch.dict(os.environ, {"COMMENTARY_STYLE": " Card "}):
                    self.assertEqual(Config.from_env(Path("/tmp/project")).commentary_style, "card")
                with patch.dict(os.environ, {"COMMENTARY_STYLE": "neon"}):
                    with self.assertRaises(FactoryError):
                        Config.from_env(Path("/tmp/project"))

            def test_max_scene_images_defaults_parses_and_rejects_out_of_range(self):
                with patch.dict(os.environ, {"MAX_SCENE_IMAGES": ""}):
                    self.assertEqual(Config.from_env(Path("/tmp/project")).max_scene_images, 14)
                with patch.dict(os.environ, {"MAX_SCENE_IMAGES": "20"}):
                    self.assertEqual(Config.from_env(Path("/tmp/project")).max_scene_images, 20)
                for bad in ("0", "31", "many"):
                    with patch.dict(os.environ, {"MAX_SCENE_IMAGES": bad}):
                        with self.assertRaises(FactoryError):
                            Config.from_env(Path("/tmp/project"))

            @staticmethod
            def sample_copy(**overrides):
                data = {
                    "hook": "تعبك الذي لا يراه أحد يراه الله",
                    "message": " ".join(["كلمة"] * 45),
                    "call_to_action": "شارك هذا المقطع واجعله صدقة جارية عن والديك",
                    "title": "تعبك الذي لا يراه أحد يراه الله وحده",
                    "description": "هذه الآية تخاطب قلب كل من أنهكه الصبر.\n\nتأمل معناها اليوم.",
                    "keywords": ["آيات عن الصبر", "تدبر القرآن", "دعاء الفرج"],
                    "hashtags": ["#الصبر", "#تدبر_القرآن", "#طمأنينة"],
                }
                data.update(overrides)
                return data

            def test_title_is_the_generated_sentence_and_description_has_keywords_and_hashtags_only(self):
                title, description = _title_description(self.sample_copy())
                self.assertEqual(title, "تعبك الذي لا يراه أحد يراه الله وحده")
                self.assertIn("الكلمات المفتاحية: آيات عن الصبر، تدبر القرآن، دعاء الفرج", description)
                self.assertTrue(description.rstrip().endswith("#الصبر #تدبر_القرآن #طمأنينة"))
                self.assertTrue(description.startswith("هذه الآية تخاطب"))
                lowered = description.lower()
                for forbidden in ("pexels", "quran.foundation", "http", "المصدر", "المصور", "تفسير الميسر", "مشاري"):
                    self.assertNotIn(forbidden, lowered)

            def test_description_never_reads_the_attribution_environment_variables(self):
                env = {name: "نص نسبة يجب ألا يظهر" for name in (
                    "RECITATION_ATTRIBUTION", "TAFSIR_ATTRIBUTION", "QURAN_ATTRIBUTION", "TTS_ATTRIBUTION")}
                with patch.dict(os.environ, env):
                    _, description = _title_description(self.sample_copy())
                self.assertNotIn("نص نسبة", description)

            def test_description_keeps_hashtags_and_trims_keywords_to_fit_the_caption_limit(self):
                copy = self.sample_copy(description="جملة مفيدة. " * 100,
                                        keywords=[f"كلمة مفتاحية رقم {i}" for i in range(15)])
                text = compose_description(copy, limit=1300)
                self.assertLessEqual(len(text), 1300)
                self.assertTrue(text.rstrip().endswith("#الصبر #تدبر_القرآن #طمأنينة"))
                self.assertIn("الكلمات المفتاحية:", text)
                tight = compose_description(copy, limit=len(copy["description"].strip()) + 40)
                self.assertNotIn("الكلمات المفتاحية:", tight)
                self.assertTrue(tight.rstrip().endswith("#طمأنينة"))
                self.assertLessEqual(len(tight), len(copy["description"].strip()) + 40)
                title, full = _title_description(self.sample_copy(description="ن" * 1400,
                                                                  hashtags=[f"#وسم{i}" for i in range(12)]))
                self.assertLessEqual(len(title) + 2 + len(full), 2000)       # fits the Instagram/TikTok caption

            def test_quran_words_helpers_in_services_and_video_engine_agree(self):
                import video_engine
                samples = ["بسم الله الرحمن", "فَٱتَّقُوا۟ ٱللَّهَ ۗ وَمَن يُوقَ ۚ فَأُو۟لَٰٓئِكَ", "۞ بسم الله", "", "  \n "]
                for sample in samples:
                    self.assertEqual(quran_words(sample), video_engine.quran_words(sample), sample)

            def test_daily_guard_only_counts_auto_and_allows_retry_after_known_failure(self):
                published = new_run_record("auto-ok", "topic-x", "auto", "2026-10-21", 1)
                published["status"] = "published"
                failed = new_run_record("auto-failed", "topic-x", "auto", "2026-10-21", 1)
                failed["status"] = "failed"
                manual = new_run_record("manual-ok", "topic-x", "manual", "2026-10-21", 1)
                manual["status"] = "published"
                self.assertTrue(already_processed_today([published], "2026-10-21", 1))
                self.assertFalse(already_processed_today([published], "2026-10-21", 2))
                self.assertFalse(already_processed_today([failed], "2026-10-21", 1))
                self.assertFalse(already_processed_today([manual], "2026-10-21", 1))
                failed["platforms"]["youtube"] = {"status": "upload_uncertain", "video_id": None}
                self.assertTrue(already_processed_today([failed], "2026-10-21", 1))

            def test_partial_failure_never_repeats_successful_platform(self):
                import tempfile
                with tempfile.TemporaryDirectory() as temp:
                    h = ProductionHistory(Path(temp) / "history.json")
                    h.load()
                    row = new_run_record("run-1", "topic-x", "manual", "2026-01-01", 1)
                    row["video_sha256"] = "abc"
                    row["platforms"]["youtube"] = {"status": "published", "video_id": "yt-existing", "attempts": 1}
                    calls = []
                    class FakeFB:
                        def publish_or_verify(self, *args, **kwargs):
                            calls.append("facebook")
                            raise ServiceError("simulated Meta failure")
                    def factory(name):
                        return FakeFB()
                    with patch.dict(os.environ, {"META_PUBLISH_ENABLED": "true", "MAKE_PUBLISH_ENABLED": "false"}):
                        publish_platforms(Path("video.mp4"), row, h, Path(temp) / "reports", "title", "desc", factory, retries=2)
                    self.assertNotIn("youtube", calls)
                    self.assertEqual(calls.count("facebook"), 1)
                    self.assertEqual(row["platforms"]["youtube"]["video_id"], "yt-existing")

            def test_pending_facebook_retries_by_id_and_never_calls_youtube(self):
                import tempfile
                with tempfile.TemporaryDirectory() as temp:
                    h = ProductionHistory(Path(temp) / "history.json")
                    h.load()
                    row = new_run_record("run-2", "topic-x", "manual", "2026-01-01", 1)
                    row["platforms"]["youtube"] = {"status": "published", "video_id": "yt-1", "attempts": 1}
                    row["platforms"]["facebook"] = {"status": "pending", "video_id": "fb-1", "attempts": 1}
                    calls = []
                    class FakeFB:
                        def publish_or_verify(self, *args, **kwargs):
                            calls.append(kwargs["previous"]["video_id"])
                            return {"status": "published", "video_id": "fb-1"} if len(calls) == 2 else {"status": "pending", "video_id": "fb-1"}
                    with patch.dict(os.environ, {"META_PUBLISH_ENABLED": "true", "MAKE_PUBLISH_ENABLED": "false"}):
                        result = publish_platforms(Path("video.mp4"), row, h, Path(temp) / "reports", "title", "desc",
                                                   lambda name: FakeFB(), retries=2)
                    self.assertEqual(calls, ["fb-1", "fb-1"])
                    self.assertEqual(result["status"], "published")

            def test_meta_publication_is_disabled_without_explicit_opt_in(self):
                import tempfile
                with tempfile.TemporaryDirectory() as temp:
                    history = ProductionHistory(Path(temp) / "history.json")
                    history.load()
                    row = new_run_record("safe-default", "topic-x", "manual", "2026-10-21", 1)
                    calls = []
                    class FakePublisher:
                        def publish_or_verify(self, *args, **kwargs):
                            calls.append("youtube")
                            return {"status": "published", "video_id": "yt-test"}
                    with patch.dict(os.environ, {"META_PUBLISH_ENABLED": "false", "MAKE_PUBLISH_ENABLED": "false"}):
                        result = publish_platforms(Path("video.mp4"), row, history, Path(temp) / "reports",
                                                   "title", "desc", lambda name: FakePublisher())
                    self.assertEqual(calls, ["youtube"])
                    self.assertEqual(result["status"], "published")
                    self.assertEqual(result["platforms"]["facebook"]["status"], "disabled")

            def test_make_mode_publishes_only_through_make_and_disables_direct_platforms(self):
                import tempfile
                with tempfile.TemporaryDirectory() as temp:
                    history = ProductionHistory(Path(temp) / "history.json")
                    history.load()
                    row = new_run_record("make-1", "topic-x", "manual", "2026-10-21", 1)
                    created = []
                    class FakeMake:
                        def publish_or_verify(self, video, title, description, previous=None):
                            return {"status": "published", "video_id": "video-abc", "video_url": "https://github.com/x/y/releases/download/video-abc/video-abc.mp4",
                                    "delivered_to": "make.com"}
                    def factory(name):
                        created.append(name)
                        return FakeMake()
                    with patch.dict(os.environ, {"MAKE_PUBLISH_ENABLED": "true", "META_PUBLISH_ENABLED": "true"}):
                        result = publish_platforms(Path("video.mp4"), row, history, Path(temp) / "reports",
                                                   "title", "desc", factory)
                    self.assertEqual(created, ["make"])
                    self.assertEqual(result["status"], "published")
                    self.assertEqual(result["platforms"]["make"]["video_id"], "video-abc")
                    self.assertEqual(result["platforms"]["youtube"]["status"], "disabled")
                    self.assertEqual(result["platforms"]["facebook"]["status"], "disabled")

            def test_make_unconfirmed_delivery_keeps_the_id_and_resumes_by_id(self):
                import tempfile
                with tempfile.TemporaryDirectory() as temp:
                    history = ProductionHistory(Path(temp) / "history.json")
                    history.load()
                    row = new_run_record("make-2", "topic-x", "manual", "2026-10-21", 1)
                    previous_seen = []
                    class FakeMake:
                        def __init__(self, fail):
                            self.fail = fail
                        def publish_or_verify(self, video, title, description, previous=None):
                            previous_seen.append((previous or {}).get("video_id"))
                            if self.fail:
                                raise ServiceError("webhook down", platform_state={
                                    "status": "uploaded_unconfirmed", "video_id": "video-abc", "video_url": "https://example.org/v.mp4"})
                            return {"status": "published", "video_id": "video-abc", "video_url": "https://example.org/v.mp4"}
                    with patch.dict(os.environ, {"MAKE_PUBLISH_ENABLED": "true"}):
                        first = publish_platforms(Path("video.mp4"), row, history, Path(temp) / "reports",
                                                  "title", "desc", lambda name: FakeMake(True), retries=2)
                        self.assertEqual(first["status"], "failed")
                        self.assertEqual(first["platforms"]["make"]["status"], "uploaded_unconfirmed")
                        self.assertEqual(first["platforms"]["make"]["video_id"], "video-abc")
                        self.assertTrue(row_has_unresolved_submission(first))
                        second = publish_platforms(Path("video.mp4"), first, history, Path(temp) / "reports",
                                                   "title", "desc", lambda name: FakeMake(False), retries=2)
                    self.assertEqual(previous_seen, [None, "video-abc"])
                    self.assertEqual(second["status"], "published")

            def test_make_settings_are_required_only_when_make_publishing_is_enabled(self):
                base = {
                    "GEMINI_API_KEY": "k", "QF_CLIENT_ID": "i", "QF_CLIENT_SECRET": "s",
                    "QURAN_TEXT_LICENSE_URL": "https://example.org/q", "CHAPTER_RECITER_ID": "7",
                    "RECITATION_LICENSE_URL": "https://example.org/r", "RECITATION_ALLOWLIST": "7|https://example.org/r",
                    "TAFSIR_ID": "16", "TAFSIR_LICENSE_URL": "https://example.org/t",
                    "TAFSIR_ALLOWLIST": "16|https://example.org/t",
                    "TTS_LICENSE_URL": "https://example.org/tts",
                    "QURAN_TEXT_LICENSE_CONFIRMED": "true", "RECITATION_LICENSE_CONFIRMED": "true",
                    "TAFSIR_COMMERCIAL_LICENSE_VERIFIED": "true",
                    "TTS_TERMS_CONFIRMED": "true",
                }
                with patch.dict(os.environ, {**base, "MAKE_PUBLISH_ENABLED": "false"}):
                    validate_generation_environment()
                with patch.dict(os.environ, {**base, "MAKE_PUBLISH_ENABLED": "true", "MAKE_WEBHOOK_URL": "",
                                              "GITHUB_TOKEN": "t", "GITHUB_REPOSITORY": "o/r"}):
                    with self.assertRaisesRegex(FactoryError, "MAKE_WEBHOOK_URL"):
                        validate_generation_environment()
                with patch.dict(os.environ, {**base, "MAKE_PUBLISH_ENABLED": "true",
                                              "MAKE_WEBHOOK_URL": "https://hook.eu1.make.com/abc",
                                              "GITHUB_TOKEN": "t", "GITHUB_REPOSITORY": "o/r"}):
                    validate_generation_environment()

            def test_image_generation_settings_are_never_required_for_production(self):
                # Pictures are optional: a run without any image configuration must still pass validation.
                base = {
                    "GEMINI_API_KEY": "k", "QF_CLIENT_ID": "i", "QF_CLIENT_SECRET": "s",
                    "QURAN_TEXT_LICENSE_URL": "https://example.org/q", "CHAPTER_RECITER_ID": "7",
                    "RECITATION_LICENSE_URL": "https://example.org/r", "RECITATION_ALLOWLIST": "7|https://example.org/r",
                    "TAFSIR_ID": "16", "TAFSIR_LICENSE_URL": "https://example.org/t",
                    "TAFSIR_ALLOWLIST": "16|https://example.org/t", "TTS_LICENSE_URL": "https://example.org/tts",
                    "QURAN_TEXT_LICENSE_CONFIRMED": "true", "RECITATION_LICENSE_CONFIRMED": "true",
                    "TAFSIR_COMMERCIAL_LICENSE_VERIFIED": "true", "TTS_TERMS_CONFIRMED": "true",
                    "MAKE_PUBLISH_ENABLED": "false", "CLOUDFLARE_ACCOUNT_ID": "", "CLOUDFLARE_API_TOKEN": "",
                    "IMAGE_GEN_LICENSE_CONFIRMED": "", "IMAGE_GEN_LICENSE_URL": "",
                }
                with patch.dict(os.environ, base):
                    validate_generation_environment()
                    service = SceneImageService()
                    self.assertEqual(service.providers, [])
                    self.assertEqual(service.generate_jobs([("scene-1", "A quiet road at dawn with mist over the hills")],
                                                           Path("/tmp/none"), 1080, 1920), {})

            def test_recitation_limit_defaults_parses_and_validates(self):
                with patch.dict(os.environ, {"MAX_RECITATION_SECONDS": ""}):
                    self.assertEqual(Config.from_env(Path("/tmp/project")).max_recitation_seconds, 45)
                with patch.dict(os.environ, {"MAX_RECITATION_SECONDS": "30"}):
                    self.assertEqual(Config.from_env(Path("/tmp/project")).max_recitation_seconds, 30)
                for bad in ("5", "999", "x"):
                    with patch.dict(os.environ, {"MAX_RECITATION_SECONDS": bad}):
                        with self.assertRaises(FactoryError):
                            Config.from_env(Path("/tmp/project"))

            def test_topics_file_defaults_next_to_the_project_and_can_be_overridden(self):
                with patch.dict(os.environ, {"TOPICS_FILE": ""}):
                    self.assertEqual(Config.from_env(Path("/tmp/project")).topics_file, Path("/tmp/project/data/topics.json"))
                with patch.dict(os.environ, {"TOPICS_FILE": "other/t.json"}):
                    self.assertEqual(Config.from_env(Path("/tmp/project")).topics_file, Path("/tmp/project/other/t.json"))

            def test_speech_budget_shrinks_as_the_recitation_grows_and_stays_in_bounds(self):
                short = speech_word_budget(60, 12)
                long_recitation = speech_word_budget(60, 35)
                huge = speech_word_budget(60, 58)
                self.assertGreater(short[1], long_recitation[1])
                self.assertEqual(huge[1], SPEECH_WORDS_FLOOR)
                self.assertEqual(speech_word_budget(200, 5)[1], SPEECH_WORDS_CAP)
                for low, high in (short, long_recitation, huge):
                    self.assertTrue(40 <= low <= high <= SPEECH_WORDS_CAP)

            @staticmethod
            def sample_topics():
                from topics import VerseRef
                return [Topic(f"topic-{n}", f"سؤال رقم {n} طويل بما يكفي؟", (VerseRef(18, 25),), ()) for n in range(1, 5)]

            def test_auto_selection_rotates_by_date_and_skips_published_and_unresolved_topics(self):
                topics = self.sample_topics()
                first = choose_auto_candidates([], "2026-01-02", topics, 10)
                self.assertEqual(sorted(t.id for t in first), sorted(t.id for t in topics))
                self.assertNotEqual([t.id for t in choose_auto_candidates([], "2026-01-02", topics, 10)],
                                    [t.id for t in choose_auto_candidates([], "2026-01-03", topics, 10)])
                self.assertEqual(len(choose_auto_candidates([], "2026-01-02", topics, 2)), 2)
                self.assertEqual(choose_auto_candidates([], "2026-01-02", [], 5), [])
                done = new_run_record("r1", "topic-1", "auto", "2026-01-01", 1)
                done["platforms"]["youtube"] = {"status": "published", "video_id": "yt-1", "attempts": 1}
                stuck = new_run_record("r2", "topic-2", "auto", "2026-01-01", 1)
                stuck["platforms"]["facebook"] = {"status": "uploading_unknown", "video_id": "fb-session"}
                failed = new_run_record("r3", "topic-3", "auto", "2026-01-01", 1)
                failed["status"] = "failed"
                left = [t.id for t in choose_auto_candidates([done, stuck, failed], "2026-01-02", topics, 10)]
                self.assertEqual(sorted(left), ["topic-3", "topic-4"])                      # a known failure stays eligible

            def test_blocked_topics_explain_why_a_topic_is_unavailable(self):
                topics = self.sample_topics()
                done = new_run_record("r1", "topic-1", "auto", "2026-01-01", 1)
                done["platforms"]["make"] = {"status": "published", "video_id": "v1", "attempts": 1}
                stuck = new_run_record("r2", "topic-2", "auto", "2026-01-01", 1)
                stuck["platforms"]["make"] = {"status": "uploaded_unconfirmed", "video_id": "v2", "attempts": 1}
                failed = new_run_record("r3", "topic-3", "auto", "2026-01-01", 1)
                failed["status"] = "failed"
                blocked = blocked_topics([done, stuck, failed], topics)
                self.assertEqual(len(blocked), 2)                                       # a plain failure is not a block
                self.assertIn("topic-1 (منشور)", blocked)
                self.assertTrue(any(item.startswith("topic-2") and "make=uploaded_unconfirmed" in item for item in blocked))
                self.assertEqual(blocked_topics([], topics), [])

            def test_an_exhausted_topic_bank_says_which_topics_were_used_and_a_low_bank_warns(self):
                import io as _io
                import json as _json
                import tempfile
                from argparse import Namespace
                from contextlib import redirect_stderr
                with tempfile.TemporaryDirectory() as temp:
                    path = Path(temp) / "topics.json"
                    path.write_text(_json.dumps({"version": 1, "topics": [
                        {"id": "topic-1", "question": "سؤال رقم واحد طويل بما يكفي؟", "verses": [{"key": "18:25"}]}]},
                        ensure_ascii=False), encoding="utf-8")
                    history = ProductionHistory(Path(temp) / "history.json")
                    done = new_run_record("r1", "topic-1", "auto", "2026-01-01", 1)
                    done["platforms"]["make"] = {"status": "published", "video_id": "v1", "attempts": 1}
                    history.upsert(done)
                    env = {"OUTPUT_DIR": temp, "HISTORY_FILE": str(Path(temp) / "history.json"), "TOPICS_FILE": str(path)}
                    args = Namespace(mode="auto", topic=None, reprocess=False, publish_only=False, input_video=None, test=False)
                    errors = _io.StringIO()
                    with patch.dict(os.environ, env), redirect_stderr(errors):
                        self.assertEqual(run(args), 1)
                    self.assertIn("topic-1 (منشور)", errors.getvalue())
                    self.assertIn("data/topics.json", errors.getvalue())
                    path.write_text(_json.dumps({"version": 1, "topics": [
                        {"id": f"topic-{n}", "question": f"سؤال رقم {n} طويل بما يكفي؟", "verses": [{"key": "18:25"}]} for n in (1, 2)]},
                        ensure_ascii=False), encoding="utf-8")
                    warned = _io.StringIO()
                    with patch.dict(os.environ, env), redirect_stderr(warned), \
                            patch.object(THIS, "generate_candidate", side_effect=lambda *a, **k: ({"status": "published"}, True)):
                        run(args)
                    self.assertIn("بقي 1 موضوعًا", warned.getvalue())

            def test_run_record_remembers_the_topic_under_a_topic_key(self):
                row = new_run_record("run-9", "cave-309", "manual", "2026-01-01", 2)
                self.assertEqual((row["verse_key"], row["topic_id"], row["attempt_count"]), ("topic:cave-309", "cave-309", 2))
                self.assertEqual(Topic("cave-309", "سؤال طويل بما يكفي؟", self.sample_topics()[0].verses).row_key, row["verse_key"])

            def test_record_sources_lists_every_verse_recitation_tafsir_picture_and_the_voice(self):
                class Obj:
                    def __init__(self, **kwargs):
                        self.__dict__.update(kwargs)
                items = [{"verse": Obj(key="18:25"), "label": "18:25",
                          "recitation": Obj(reciter_id=7, reciter_name="Mishari", source_url="https://x", audio_url="https://y")},
                         {"verse": Obj(key="18:26"), "label": "18:26#1-5",
                          "recitation": Obj(reciter_id=7, reciter_name="Mishari", source_url="https://x", audio_url="https://y")}]
                tafsirs = {"18:25": Obj(source_name="t", resource_id=16, source_url="https://z", license_url="https://l"),
                           "18:26": Obj(source_name="t", resource_id=16, source_url="https://z", license_url="https://l")}
                images = [{"kind": "ai_generated_image", "scene": "opening-1"}, {"kind": "ai_generated_image", "scene": "scene-1"}]
                sources = _record_sources(items, tafsirs, images, "https://terms", "ar-EG-ShakirNeural")
                kinds = [s["kind"] for s in sources]
                self.assertEqual((kinds.count("quran_text"), kinds.count("recitation"), kinds.count("tafsir"),
                                  kinds.count("ai_generated_image")), (2, 2, 2, 2))
                self.assertEqual([s["fragment"] for s in sources if s["kind"] == "recitation"], ["18:25", "18:26#1-5"])
                self.assertEqual(sources[-1]["kind"], "tts")
                self.assertEqual((sources[-1]["voice"], sources[-1]["provider"]), ("ar-EG-ShakirNeural", "edge-tts"))
                other = _record_sources(items, tafsirs, [], "https://terms", "Algieba", "gemini-tts (m)")
                self.assertEqual((other[-1]["provider"], other[-1]["voice"]), ("gemini-tts (m)", "Algieba"))

            # ----- A whole candidate with stand-ins for every network service (real FFmpeg audio and video) -----
            WORDS = "واحد اثنان ثلاثة أربعة خمسة ستة"

            @classmethod
            def _stand_ins(cls, temp, **fake):
                import subprocess
                captured = {"tafsir": [], "calls": {}}

                def sine(path, seconds):
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                                    "-i", f"sine=frequency=330:duration={seconds}", "-ar", "48000", str(path)], check=True)

                class Verse:
                    def __init__(self, chapter, verse):
                        self.key, self.chapter, self.verse = f"{chapter}:{verse}", chapter, verse
                        self.text_uthmani, self.chapter_name = fake.get("text", cls.WORDS), "الكهف"

                class Recitation:
                    reciter_id, reciter_name = 7, "Mishari Rashid al-`Afasy"
                    source_url, audio_url = "https://x.example/s", "https://x.example/a.mp3"

                class FakeQuran:
                    def get_verse(self, chapter, verse):
                        return Verse(chapter, verse)
                    def get_recitation(self, chapter, verse, directory, words=None, verse_text=None):
                        if fake.get("recitation_error"):
                            raise ServiceError(fake["recitation_error"])
                        count = (words[1] - words[0] + 1) if words else len(quran_words(verse_text))
                        seconds = fake.get("seconds", 3.0)
                        path = Path(directory) / f"rec-{chapter}-{verse}-{words[0] if words else 0}.wav"
                        sine(path, seconds)
                        step = (seconds * 1000 - 200) / count
                        recitation = Recitation()
                        recitation.segments = fake.get("segments") or [[i + 1, int(100 + i * step), int(100 + (i + 1) * step - 20)]
                                                                       for i in range(count)]
                        return recitation, path
                    def get_tafsir(self, key):
                        captured["tafsir"].append(key)
                        return TafsirSource("هذا تفسير عربي موثق يشرح معنى الآية ضمن سياقها بما يكفي للاختبار.", "مصدر", 16,
                                            "https://x.example/t", "https://x.example/l", True)

                class FakeTTS:
                    provider, voice, terms_url = "fake-tts", "ar-EG-ShakirNeural", "https://example.org/tts"
                    word_boundaries: dict = {}
                    def synthesize(self, texts, directory):
                        captured["tts_texts"] = dict(texts)
                        files = {}
                        for name, text in texts.items():
                            files[name] = Path(directory) / f"{name}.wav"
                            sine(files[name], max(1.0, 0.3 * len(text.split())))
                        return files

                shot = lambda mood, start=None, prompt="": ({"mood": mood, "elements": [{"type": "light_rays", "position": "center",
                                                             "animation": "fade_in", "size": "medium"}], "image_prompt": prompt}
                                                            | ({} if start is None else {"word_start": start}))
                prompt = "A lone traveller walking a winding road toward a glowing sunrise over quiet hills"
                ai = {
                    "hook": "لماذا لم يقل القرآن ثلاثمائة وتسع سنين مباشرة مع أن المجموع نفسه؟ الجواب أدق مما تتخيل.",
                    "hook_candidates": [], "parts": [
                        {"type": "speech", "text": "اسمع، قال تعالى:"}, {"type": "verse", "verse_index": 1},
                        {"type": "speech", "text": " ".join(f"فكرة{i}" for i in range(20))},
                        {"type": "verse", "verse_index": 2},
                        {"type": "speech", "text": " ".join(f"ختام{i}" for i in range(6))}],
                    "call_to_action": "تابعني لنتدبر دقة القرآن بحسبان",
                    "title": "لماذا لم يقل القرآن ثلاثمائة وتسع سنين مباشرة", "description": "وصف طويل. " * 40,
                    "keywords": ["آيات الكهف", "تدبر القرآن"], "hashtags": ["#الكهف", "#تدبر_القرآن"],
                    "visual_plan": {"visual_theme": "warm dusk light", "opening": {"shots": [shot("night", prompt=prompt), shot("storm", prompt=prompt)]},
                                    "scenes": [shot("dawn", 0, prompt), shot("day", 3, prompt), shot("storm", 13, prompt),
                                               shot("warm", 23, prompt), shot("warm", 29, prompt)]},
                    "body_text": "", "spoken_words": 60, "review": {"reason": "ok"}, "model": "fake-model",
                }
                if fake.get("ai_update"):
                    ai.update(fake["ai_update"])

                class FakeGemini:
                    def create_and_review(self, topic, verses, tafsirs, budget):
                        if fake.get("paid_call_fails"):
                            raise AssertionError("paid call must not happen")
                        captured.update(topic=topic, verses=verses, tafsirs=tafsirs, budget=budget)
                        return ai

                def fake_publish(video, row, history, report_dir, title, description, make_publisher=None, retries=3):
                    captured["published"] = (Path(video).name, title)
                    row["status"] = "published"
                    row["platforms"]["youtube"] = {"status": "published", "video_id": "yt-fake", "attempts": 1}
                    update_record(history, row, report_dir)
                    return row

                patches = [patch.object(THIS, "validate_generation_environment"),
                           patch.object(THIS, "QuranFoundationClient", FakeQuran),
                           patch.object(THIS, "GeminiContentService", FakeGemini),
                           patch.object(THIS, "make_tts_service", lambda voice=None: FakeTTS()),
                           patch.object(THIS, "publish_platforms", fake_publish)]
                return patches, captured

            def _generate(self, topic, mode="auto", env=None, **fake):
                import tempfile
                from contextlib import ExitStack
                with tempfile.TemporaryDirectory() as temp:
                    patches, captured = self._stand_ins(temp, **fake)
                    environment = {"OUTPUT_DIR": temp, "HISTORY_FILE": temp + "/history.json", "VIDEO_WIDTH": "360",
                                   "VIDEO_HEIGHT": "640", "TARGET_MIN_DURATION": "10", "TARGET_MAX_DURATION": "60",
                                   "IMAGE_GEN_LICENSE_CONFIRMED": "", **(env or {})}
                    with patch.dict(os.environ, environment), ExitStack() as stack:
                        for item in patches:
                            stack.enter_context(item)
                        config = Config.from_env(Path(temp))
                        history = ProductionHistory(config.history_file)
                        row, validated = generate_candidate(topic, 1, mode, config, history, "2026-10-24")
                        captured["video_exists"] = (config.output_dir / config.final_video_name).is_file()
                        captured["history_rows"] = history.get_runs()
                    return row, validated, captured

            def test_a_complete_candidate_is_built_from_the_idea_the_verses_and_a_script_that_places_them(self):
                from topics import VerseRef
                topic = Topic("cave-309", "لماذا لم يقل القرآن ثلاثمائة وتسع سنين مباشرة؟",
                              (VerseRef(18, 25), VerseRef(18, 26, 1, 3)), ("السنة الشمسية نحو ثلاثمائة وخمسة وستين يومًا.",))
                row, validated, seen = self._generate(topic)
                self.assertEqual((row["status"], validated, seen["video_exists"]), ("published", True, True))
                self.assertEqual(row["verse_key"], "topic:cave-309")
                checks = row["verification"]["checks"]
                self.assertEqual([v["label"] for v in checks["verses"]], ["18:25", "18:26#1-3"])
                self.assertAlmostEqual(checks["recitation_total_seconds"], 6.0, delta=0.3)
                self.assertEqual([v.text for v in seen["verses"]], [self.WORDS, "واحد اثنان ثلاثة"])   # the fragment, exactly as recited
                self.assertEqual([v.index for v in seen["verses"]], [1, 2])
                self.assertEqual(seen["budget"], speech_word_budget(60, checks["recitation_total_seconds"]))
                self.assertEqual(seen["topic"].id, "cave-309")
                self.assertEqual(sorted(seen["tafsirs"]), ["18:25", "18:26"])
                self.assertEqual(list(seen["tts_texts"]), ["hook", "talk-1", "talk-2", "talk-3", "closing"])
                video = checks["video"]
                self.assertEqual(list(video["audio_segments"]),
                                 ["hook", "talk-1", "verse-1", "talk-2", "verse-2", "talk-3", "closing"])
                self.assertEqual(video["verse_count"], 2)
                self.assertEqual(video["commentary_style"], "captions")
                self.assertEqual(checks["content"]["parts"], ["speech", "verse", "speech", "verse", "speech"])
                kinds = [s["kind"] for s in row["media_sources"]]
                self.assertEqual((kinds.count("quran_text"), kinds.count("recitation"), kinds.count("tafsir"),
                                  kinds.count("tts"), kinds.count("font"), kinds.count("procedural_background")), (2, 2, 2, 1, 1, 1))
                self.assertEqual(seen["published"][0], "final_output.mp4")
                self.assertTrue(row["video_sha256"])
                self.assertEqual(checks["scene_images"]["generated"], 0)             # no image provider configured: procedural scenes
                self.assertTrue(checks["scene_images"]["notes"])
                self.assertEqual(len(seen["history_rows"]), 1)

            def test_generated_pictures_are_requested_only_for_the_shots_that_survive_the_cut_and_recorded(self):
                import io as _io
                from PIL import Image as _Image, ImageChops as _ImageChops
                from topics import VerseRef
                requested = []

                class Provider:
                    name, model = "cloudflare", "fake-model"
                    def generate(self, prompt, width, height, seed):
                        requested.append(prompt)
                        gradient = _Image.linear_gradient("L").resize((512, 512))
                        noise = _Image.effect_noise((512, 512), 40)
                        buffer = _io.BytesIO()
                        _Image.merge("RGB", (_ImageChops.add(gradient, noise, scale=2), gradient, noise)).save(buffer, "PNG")
                        return buffer.getvalue()

                real_service = SceneImageService

                def service():
                    result = real_service(providers=[Provider()])
                    result._sleep = lambda seconds: None
                    return result
                topic = Topic("cave-309", "لماذا لم يقل القرآن ثلاثمائة وتسع سنين مباشرة؟", (VerseRef(18, 25), VerseRef(18, 26, 1, 3)), ())
                with patch.object(THIS, "SceneImageService", service):
                    row, validated, _ = self._generate(topic, env={"MAX_SCENE_IMAGES": "4"})
                self.assertTrue(validated)
                info = row["verification"]["checks"]["scene_images"]
                self.assertEqual((info["requested"], info["generated"]), (4, 4))
                self.assertEqual(len(requested), 4)
                self.assertTrue(all("warm dusk light" in prompt for prompt in requested))     # the shared theme travels with every prompt
                self.assertEqual(len([s for s in row["media_sources"] if s["kind"] == "ai_generated_image"]), 4)
                self.assertGreaterEqual(info["entries_reusing_a_neighbour_picture"], 1)      # shots beyond the budget reuse a neighbour

            def test_a_fragment_beyond_the_verse_a_timing_mismatch_and_an_overlong_recitation_are_permanent_rejections(self):
                from topics import VerseRef
                beyond = Topic("t-beyond", "سؤال طويل بما يكفي هنا؟", (VerseRef(18, 25, 1, 50),))
                row, validated, _ = self._generate(beyond, paid_call_fails=True)
                self.assertEqual((row["status"], row["permanent_failure"], validated), ("failed", True, False))
                self.assertIn("يتجاوز", row["last_error"])
                plain = Topic("t-plain", "سؤال طويل بما يكفي هنا؟", (VerseRef(18, 25),))
                row, _, _ = self._generate(plain, paid_call_fails=True, segments=[[1, 0, 500]])
                self.assertTrue(row["permanent_failure"])
                self.assertIn("التوقيتات", row["last_error"])
                two = Topic("t-two", "سؤال طويل بما يكفي هنا؟", (VerseRef(18, 25), VerseRef(18, 26)))
                row, _, _ = self._generate(two, paid_call_fails=True, seconds=30.0)
                self.assertTrue(row["permanent_failure"])
                self.assertIn("أطول", row["last_error"])
                manual, _, _ = self._generate(two, mode="manual", paid_call_fails=True, seconds=30.0)      # the cap is for automatic runs only
                self.assertFalse(manual["permanent_failure"])
                self.assertIn("paid call", manual["last_error"])

            def test_a_failing_recitation_source_is_a_normal_retryable_failure(self):
                from topics import VerseRef
                row, validated, _ = self._generate(Topic("t-err", "سؤال طويل بما يكفي هنا؟", (VerseRef(18, 25),)),
                                                   recitation_error="reached recitation")
                self.assertEqual((row["status"], validated), ("failed", False))
                self.assertFalse(row["permanent_failure"])
                self.assertIn("reached recitation", row["last_error"])

            def test_deterministic_rejection_moves_to_the_next_topic_without_retrying(self):
                import json as _json
                import tempfile
                from argparse import Namespace
                calls = []
                def fake_generate(topic, attempt, mode, config, history, local_date, previous_row=None):
                    calls.append((topic.id, attempt))
                    if len(calls) == 1:
                        return {"status": "failed", "permanent_failure": True, "last_error": "short"}, False
                    row = new_run_record("ok", topic.id, mode, local_date, attempt)
                    row["status"] = "published"
                    return row, True
                with tempfile.TemporaryDirectory() as temp:
                    topics_path = Path(temp) / "topics.json"
                    topics_path.write_text(_json.dumps({"version": 1, "topics": [
                        {"id": f"topic-{n}", "question": f"سؤال رقم {n} طويل بما يكفي؟", "verses": [{"key": "18:25"}]} for n in (1, 2, 3)]},
                        ensure_ascii=False), encoding="utf-8")
                    with patch.dict(os.environ, {"OUTPUT_DIR": temp, "HISTORY_FILE": temp + "/history.json", "MAX_RETRIES": "5",
                                                 "TOPICS_FILE": str(topics_path)}), \
                            patch.object(THIS, "generate_candidate", fake_generate):
                        code = run(Namespace(mode="auto", topic=None, reprocess=False, publish_only=False, input_video=None, test=False))
                self.assertEqual(code, 0)
                self.assertEqual(len(calls), 2)                              # the rejected topic was tried once, not five times
                self.assertNotEqual(calls[0][0], calls[1][0])
                self.assertEqual((calls[0][1], calls[1][1]), (1, 1))

            def test_manual_mode_needs_a_known_topic_and_refuses_a_published_or_unresolved_one(self):
                import json as _json
                import tempfile
                from argparse import Namespace
                with tempfile.TemporaryDirectory() as temp:
                    topics_path = Path(temp) / "topics.json"
                    topics_path.write_text(_json.dumps({"version": 1, "topics": [
                        {"id": "topic-1", "question": "سؤال رقم واحد طويل بما يكفي؟", "verses": [{"key": "18:25"}]}]},
                        ensure_ascii=False), encoding="utf-8")
                    env = {"OUTPUT_DIR": temp, "HISTORY_FILE": temp + "/history.json", "TOPICS_FILE": str(topics_path)}
                    args = lambda **kw: Namespace(**{"mode": "manual", "topic": None, "reprocess": False, "publish_only": False,
                                                     "input_video": None, "test": False, **kw})
                    with patch.dict(os.environ, env):
                        self.assertEqual(run(args()), 1)                              # no --topic
                        self.assertEqual(run(args(topic="nope")), 1)                  # unknown topic
                        history = ProductionHistory(Path(temp) / "history.json")
                        done = new_run_record("r1", "topic-1", "manual", "2026-01-01", 1)
                        done["video_sha256"] = "abc"
                        done["platforms"]["youtube"] = {"status": "published", "video_id": "yt-1", "attempts": 1}
                        history.upsert(done)
                        with patch.object(THIS, "generate_candidate", side_effect=AssertionError("must not generate")):
                            self.assertEqual(run(args(topic="topic-1")), 1)           # already published, no --reprocess
                        done["platforms"]["youtube"] = {"status": "upload_uncertain", "video_id": None, "attempts": 1}
                        history.upsert(done)
                        with patch.object(THIS, "generate_candidate", side_effect=AssertionError("must not generate")):
                            self.assertEqual(run(args(topic="topic-1")), 1)           # unresolved submission

            def test_the_parser_takes_a_topic_instead_of_a_chapter_and_verse(self):
                args = build_parser().parse_args(["--mode", "manual", "--topic", "cave-309"])
                self.assertEqual((args.mode, args.topic), ("manual", "cave-309"))
                with self.assertRaises(SystemExit):
                    build_parser().parse_args(["--chapter", "2"])

            def test_history_creates_valid_json_and_prevents_duplicate_publication(self):
                import tempfile
                with tempfile.TemporaryDirectory() as temp:
                    history = ProductionHistory(Path(temp) / "data/history.json")
                    history.load()
                    topics = self.sample_topics()
                    self.assertEqual(len(choose_auto_candidates([], "2026-01-02", topics, 100)), 4)
                    row = new_run_record("test-1", topics[0].id, "manual", "2026-01-01", 1)
                    row["platforms"]["youtube"] = {"status": "published", "video_id": "yt-1", "attempts": 1}
                    history.upsert(row)
                    reread = history.get_runs()
                    self.assertEqual(len(reread), 1)
                    self.assertTrue(row_has_any_publication(reread[0]))
                    self.assertNotIn(topics[0].id, [t.id for t in choose_auto_candidates(reread, "2026-01-02", topics, 100)])

            def test_publish_only_fingerprint_requires_exact_artifact(self):
                import tempfile
                with tempfile.TemporaryDirectory() as temp:
                    h = ProductionHistory(Path(temp) / "history.json")
                    row = new_run_record("run-1", "cave-309", "manual", "2026-01-01", 1)
                    row["video_sha256"] = "expected"
                    h.upsert(row)
                    self.assertEqual(find_latest_row(h.get_runs(), "topic:cave-309")["video_sha256"], "expected")

        unittest.main(argv=[sys.argv[0]])
    else:
        sys.exit(run(build_parser().parse_args()))
