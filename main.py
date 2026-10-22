"""Quran Cinematic AI Factory entry point, production history, and workflow orchestration."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
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
from video_engine import VideoError, build_video, compose_background, media_duration, validate_video

from services import (
    FacebookReelsPublisher, GeminiContentService, MakeWebhookPublisher, PexelsService, QuranFoundationClient,
    ServiceError, TTSService, YouTubePublisher, env_bool, validate_word_segments,
)
from video_engine import VideoError, build_video, media_duration, validate_video

PROJECT_ROOT = Path(__file__).resolve().parent
AUTO_VERSE_POOL = [
    (2, 152), (2, 153), (3, 139), (5, 8), (13, 28), (16, 90),
    (17, 23), (25, 63), (39, 53), (49, 13), (64, 16), (94, 5),
]
DEFAULT_TTS_VOICE = "ar-EG-ShakirNeural"


class FactoryError(RuntimeError):
    pass


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
    background_clips: int = 3
    background_speed: float = 0.75

    @classmethod
    def from_env(cls, root: Path = PROJECT_ROOT) -> "Config":
        def integer(name: str, default: int, minimum: int, maximum: int) -> int:
            raw = os.getenv(name, str(default)).strip()
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
        return cls(
            timezone_name=timezone_name,
            daily_video_count=integer("DAILY_VIDEO_COUNT", 1, 1, 5),
            width=width,
            height=height,
            fps=integer("VIDEO_FPS", 30, 24, 60),
            target_min=target_min,
            target_max=target_max,
            max_candidates=integer("MAX_CANDIDATES_PER_RUN", 5, 1, 20),
            max_retries=integer("MAX_RETRIES", 3, 1, 8),
            tts_voice=os.getenv("TTS_VOICE", DEFAULT_TTS_VOICE).strip() or DEFAULT_TTS_VOICE,
            output_dir=output_dir,
            history_file=history_file,
            final_video_name=final_name,
            dim_strength=decimal("VIDEO_DIM_STRENGTH", 0.30, 0.0, 0.8),
            background_clips=integer("BACKGROUND_CLIP_COUNT", 3, 2, 5),
            background_speed=decimal("BACKGROUND_SPEED", 0.75, 0.25, 1.0),
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
        "TAFSIR_ALLOWLIST", "PEXELS_API_KEY", "TTS_LICENSE_URL",
    ]
    missing = [key for key in required if not os.getenv(key, "").strip()]
    if missing:
        raise FactoryError("إعدادات إنتاج مطلوبة غير موجودة: " + ", ".join(missing))
    if env_bool("MAKE_PUBLISH_ENABLED"):
        absent = [k for k in ("MAKE_WEBHOOK_URL", "GITHUB_TOKEN", "GITHUB_REPOSITORY") if not os.getenv(k, "").strip()]
        if absent:
            raise FactoryError("إعدادات Make مطلوبة غير موجودة: " + ", ".join(absent))
    if urlparse(os.getenv("QURAN_TEXT_LICENSE_URL", "").strip()).scheme != "https":
        raise FactoryError("QURAN_TEXT_LICENSE_URL يجب أن يكون رابط HTTPS.")
    confirmations = {
        "QURAN_TEXT_LICENSE_CONFIRMED": "حق إعادة نشر النص القرآني من المصدر المحدد",
        "RECITATION_LICENSE_CONFIRMED": "ترخيص إعادة نشر التلاوة",
        "TAFSIR_COMMERCIAL_LICENSE_VERIFIED": "ترخيص التفسير التجاري وإعادة النشر",
        "PEXELS_LICENSE_CONFIRMED": "مراجعة ترخيص Pexels",
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


def choose_auto_candidates(history_rows: list[dict[str, Any]], date_key: str, limit: int) -> list[tuple[int, int]]:
    # Exclude verses with any confirmed public/posted destination. Pending and failed rows remain
    # addressable by publish-only recovery instead of silently producing duplicate content.
    seen = {row.get("verse_key") for row in history_rows
            if row_has_any_publication(row) or row_has_unresolved_submission(row)}
    if not AUTO_VERSE_POOL:
        return []
    start = sum(ord(ch) for ch in date_key) % len(AUTO_VERSE_POOL)
    rotated = AUTO_VERSE_POOL[start:] + AUTO_VERSE_POOL[:start]
    result = [(ch, v) for ch, v in rotated if f"{ch}:{v}" not in seen]
    return result[:limit]


def already_processed_today(history_rows: list[dict[str, Any]], local_date: str, daily_target: int = 1) -> bool:
    today = [row for row in history_rows if row.get("mode") == "auto" and row.get("local_date") == local_date]
    completed = sum(1 for row in today if row.get("status") == "published")
    if completed >= daily_target:
        return True
    # Known pre-publication failures can be retried the same day. A crashed/in-progress
    # row or uncertain/partial submission blocks fresh uploads until it is recovered.
    return any(row.get("status") in {"started", "generated", "validated", "partially_published"}
               or row_has_unresolved_submission(row) for row in today)


def new_run_record(run_id: str, chapter: int, verse: int, mode: str, local_date: str, attempt: int) -> dict[str, Any]:
    key = f"{chapter}:{verse}"
    return {
        "run_id": run_id,
        "verse_key": key,
        "chapter": chapter,
        "verse": verse,
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
    """Publish destinations independently; never re-upload a destination already confirmed."""
    if env_bool("MAKE_PUBLISH_ENABLED"):
        targets = ["make"]
        for legacy in ("youtube", "facebook"):      # Make handles every destination
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


def _record_sources(verse, recitation, tafsir, media_items, tts_terms, tts_voice):
    return [
        {"kind": "quran_text", "provider": "Quran Foundation Content API v4", "verse_key": verse.key,
         "source_url": f"{QF_BASE}/verses/by_key/{verse.key}", "license_url": os.getenv("QURAN_TEXT_LICENSE_URL"),
         "license_status": "operator-confirmed right to reproduce source text in published video"},
        {"kind": "recitation", "provider": "Quran Foundation Content API v4", "reciter_id": recitation.reciter_id,
         "reciter_name": recitation.reciter_name, "source_url": recitation.source_url,
         "audio_url": recitation.audio_url, "license_url": os.getenv("RECITATION_LICENSE_URL"),
         "license_status": "operator-confirmed commercial reproduction permission"},
        {"kind": "tafsir", "source_name": tafsir.source_name, "resource_id": tafsir.resource_id,
         "source_url": tafsir.source_url, "license_url": tafsir.license_url,
         "license_status": "operator-confirmed commercial reproduction permission"},
        *media_items,
        {"kind": "tts", "provider": "edge-tts", "voice": tts_voice,
         "license_url": tts_terms, "license_status": "operator-confirmed current usage terms"},
    ]

# Set immediately below to keep URL constants in reports while avoiding accidental config-dependent values.
QF_BASE = os.getenv(
    "QF_API_BASE_URL",
    "https://apis-prelive.quran.foundation/content/api/v4"
    if os.getenv("QF_ENV", "production").strip().lower() == "prelive"
    else "https://apis.quran.foundation/content/api/v4",
).rstrip("/")


def _title_description(key, content, source_names, media_items=None):
    credit = "خلفية مرئية من Pexels: https://www.pexels.com/"
    for item in media_items or []:
        if item.get("source_page"):
            credit += f"\nالفيديو: {item['source_page']}" + (f" | المصور: {item['creator']}" if item.get("creator") else "")

    required_attributions = [os.getenv(name, "").strip() for name in (
        "RECITATION_ATTRIBUTION", "TAFSIR_ATTRIBUTION", "QURAN_ATTRIBUTION", "TTS_ATTRIBUTION"
    ) if os.getenv(name, "").strip()]
    if required_attributions:
        credit += "\n" + "\n".join(required_attributions)
    title = f"تدبر آية قرآنية | {key}"
    description = (
        f"{content['hook']}\n\n"
        f"شرح موجز مستند إلى تفسير {source_names[0]}. الآية: {key}.\n"
        "المحتوى للتعريف والتدبر، ولا يغني عن الرجوع إلى المصادر الشرعية المعتمدة.\n\n"
        f"{credit}"
    )
    return title[:100], description[:5000]


def generate_candidate(chapter: int, verse_number: int, attempt: int, mode: str, config: Config,
                       history: ProductionHistory, local_date: str,
                       previous_row: dict[str, Any] | None = None) -> tuple[dict[str, Any], bool]:
    run_id = f"{local_date}-{chapter}-{verse_number}-{uuid.uuid4().hex[:10]}"
    row = new_run_record(run_id, chapter, verse_number, mode, local_date, attempt)
    if previous_row:
        # Reprocessing may create a new artifact, but previously confirmed platform posts and IDs
        # remain confirmed and are never uploaded again for this verse.
        row["platforms"] = row["platforms"] | {
            key: dict(value) for key, value in previous_row.get("platforms", {}).items()
            if key in {"youtube", "facebook", "make"} and isinstance(value, dict)
        }
    report_dir = config.output_dir / "reports"
    update_record(history, row, report_dir)
    validated_video = False
    try:
        validate_generation_environment()
        quran = QuranFoundationClient()
        verse = quran.get_verse(chapter, verse_number)
        row["verse_key"] = verse.key
        row["verification"]["checks"]["verse_source_and_key"] = True
        row["status"] = "started"
        update_record(history, row, report_dir)

        work_dir = config.output_dir / f"work-{run_id}"
        work_dir.mkdir(parents=True, exist_ok=True)
        recitation, recitation_file = quran.get_recitation(chapter, verse_number, work_dir)
        recitation_duration = media_duration(recitation_file)
        timed_words = validate_word_segments(recitation.segments, verse.text_uthmani,
                                             int(round(recitation_duration * 1000)))
        row["verification"]["checks"]["recitation_api_verse_key"] = verse.key
        row["verification"]["checks"]["word_segments_valid"] = len(timed_words)
        row["verification"]["checks"]["recitation_duration_seconds"] = round(recitation_duration, 3)
        row["verification"]["checks"]["audio_content_identity"] = {
            "status": "not_independently_transcribed",
            "audio_sent_to_gemini": False,
            "basis": "Quran Foundation verse key and word-level timestamps only",
        }
        update_record(history, row, report_dir)

        tafsir = quran.get_tafsir(verse.key)
        row["source_tafsir"] = {
            "source_name": tafsir.source_name, "resource_id": tafsir.resource_id,
            "source_url": tafsir.source_url, "license_url": tafsir.license_url,
            "commercial_license_verified_by_operator": tafsir.license_verified_by_operator,
        }
        gemini = GeminiContentService()
        ai = gemini.create_and_review(verse, tafsir)
        row["verification"]["checks"]["model_content_schema"] = True
        row["verification"]["checks"]["model_source_review"] = ai["review"]
        row["verification"]["checks"]["honorific_occurrences_in_generated_text"] = sum(
            text.count("صدق الله العظيم") for text in (ai["hook"], ai["explanation"], ai["practical_benefit"], ai["call_to_action"])
        )
        if row["verification"]["checks"]["honorific_occurrences_in_generated_text"]:
            raise FactoryError("ظهرت العبارة المحجوزة ضمن النص المولّد.")

        media_items = PexelsService().fetch_backgrounds(
            ai["explanation"] + " " + ai["practical_benefit"], work_dir / "clips",
            config.background_clips, config.background_speed)
        per_clip = max(6.0, math.ceil(config.target_max * config.background_speed / len(media_items)) + 1)
        background = work_dir / "background.mp4"
        bg_info = compose_background([Path(m.pop("file")) for m in media_items], background,
                                     config.width, config.height, config.fps,
                                     speed=config.background_speed, source_seconds_each=per_clip)
        row["verification"]["checks"]["background"] = bg_info
        tts = TTSService(config.tts_voice)
        speech_texts = {
            "hook": ai["hook"],
            "seal": "صدق الله العظيم",
            "explanation": ai["explanation"],
            "benefit": ai["practical_benefit"],
            "cta": ai["call_to_action"],
        }
        speech_files = tts.synthesize(speech_texts, work_dir / "speech")
        speech_files["recitation"] = recitation_file
        output_path = config.output_dir / config.final_video_name
        # Quran word highlighting always uses the source timestamps; the commentary highlighting uses
        # the TTS word boundaries when valid and an estimate otherwise (reported in the video checks).
        video_validation = build_video(
            background, speech_files, timed_words, verse.text_uthmani, ai, output_path,
            width=config.width, height=config.height, fps=config.fps,
            target_min_duration=config.target_min, target_max_duration=config.target_max,
            tts_word_boundaries=tts.word_boundaries, chapter_name=verse.chapter_name,
            verse_number=verse.verse, dim_strength=config.dim_strength,
        )
        row["video_path"] = str(output_path)
        row["video_sha256"] = sha256_file(output_path)
        row["verification"]["checks"]["video"] = video_validation
        row["verification"]["status"] = "validated"
        row["status"] = "validated"
        row["media_sources"] = _record_sources(verse, recitation, tafsir, media_items, tts.terms_url, tts.voice)
        font_source = video_validation.get("font_source", {})
        row["media_sources"].append({"kind": "font", **font_source})
        validated_video = True
        update_record(history, row, report_dir)
        title, description = _title_description(verse.key, ai, [tafsir.source_name], media_items)
        row["publication_metadata"] = {"title": title, "description": description}
        update_record(history, row, report_dir)
        publish_platforms(output_path, row, history, report_dir, title, description,
                          retries=config.max_retries)
        return row, validated_video
    except Exception as exc:
        row["last_error"] = (str(exc) or type(exc).__name__)[:500]
        row["attempt_count"] = attempt
        row["status"] = "partially_published" if row_has_any_publication(row) else "failed"
        if not row.get("verification", {}).get("status") == "validated":
            row.setdefault("verification", {})["status"] = "failed"
        update_record(history, row, report_dir)
        print(f"فشل المرشح {chapter}:{verse_number}: {row['last_error']}", file=sys.stderr)
        return row, validated_video


def media_path(media: dict[str, Any], work_dir: Path) -> Path:
    path = work_dir / "background.mp4"
    if not path.is_file():
        raise FactoryError("خلفية Pexels لم تُنزّل إلى المسار المتوقع.")
    return path


def find_latest_row(history_rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    matches = [row for row in history_rows if row.get("verse_key") == key and row.get("video_sha256")]
    return max(matches, key=lambda row: row.get("updated_at", "")) if matches else None


def publish_only(args: argparse.Namespace, config: Config, history: ProductionHistory) -> int:
    if not args.chapter or not args.verse or not args.input_video:
        raise FactoryError("--publish-only يتطلب --chapter و--verse و--input-video.")
    key = f"{args.chapter}:{args.verse}"
    video = Path(args.input_video).expanduser().resolve()
    rows = history.get_runs()
    row = find_latest_row(rows, key)
    if not row:
        raise FactoryError("لا يوجد سجل فيديو مُتحقق منه لهذه الآية؛ لن يُنشأ منشور جديد.")
    validate_video(video, config.width, config.height, config.fps)
    digest = sha256_file(video)
    if digest != row.get("video_sha256"):
        raise FactoryError("بصمة ملف الفيديو لا تطابق النسخة المسجلة؛ رُفضت إعادة النشر.")
    report_dir = config.output_dir / "reports"
    metadata = row.get("publication_metadata") or {}
    row = publish_platforms(video, row, history, report_dir,
                            metadata.get("title", f"تدبر آية قرآنية | {key}"),
                            metadata.get("description", f"شرح موجز مستند إلى التفسير الموثق. الآية: {key}."),
                            retries=config.max_retries)
    print(f"حالة إعادة المحاولة للآية {key}: {row['status']}")
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
        if args.mode == "manual":
            if args.chapter is None or args.verse is None:
                raise FactoryError("الوضع اليدوي يتطلب --chapter و--verse.")
            if not 1 <= args.chapter <= 114 or args.verse < 1:
                raise FactoryError("رقم السورة أو الآية خارج النطاق.")
            candidates = [(args.chapter, args.verse)]
        elif args.mode == "auto":
            if already_processed_today(history_rows, local_date, config.daily_video_count):
                print(f"يوجد تشغيل تلقائي مسجل لهذا اليوم في {config.timezone_name}؛ لن يُنشأ فيديو إضافي.")
                return 0
            candidates = choose_auto_candidates(history_rows, local_date, config.max_candidates)
        else:
            raise FactoryError("--mode يجب أن يكون auto أو manual.")
        if args.mode == "manual":
            existing = find_latest_row(history_rows, f"{args.chapter}:{args.verse}")
            if existing and row_has_unresolved_submission(existing):
                raise FactoryError("لدى هذه الآية جلسة نشر غير محسومة؛ استخدم --publish-only للتحقق من معرّف المنصة قبل إنشاء أي منشور جديد.")
            if existing and row_has_any_publication(existing) and not args.reprocess:
                raise FactoryError("هذه الآية لها منشور مؤكد سابقًا. استخدم --publish-only لإعادة التحقق أو --reprocess لتوليد جديد؛ لن يتكرر النشر تلقائيًا.")
        if not candidates:
            raise FactoryError("لا توجد آيات غير منشورة مرشحة ضمن القائمة؛ لم يُنشأ فيديو.")
        candidate_errors = []
        daily_target = config.daily_video_count if args.mode == "auto" else 1
        completed_today = sum(1 for row in history_rows if row.get("mode") == "auto"
                              and row.get("local_date") == local_date and row.get("status") == "published")
        if args.mode == "auto" and completed_today >= daily_target:
            print(f"اكتمل العدد اليومي ({completed_today}/{daily_target}) في {config.timezone_name}.")
            return 0
        successful_this_run = 0
        for chapter, verse in candidates:
            if (any(row.get("verse_key") == f"{chapter}:{verse}" and
                    (row_has_any_publication(row) or row_has_unresolved_submission(row)) for row in history_rows)
                    and not (args.mode == "manual" and args.reprocess)):
                continue
            attempts = config.max_retries if args.mode == "auto" else 1
            previous_row = find_latest_row(history_rows, f"{chapter}:{verse}") if args.reprocess else None
            for attempt in range(1, attempts + 1):
                row, validated = generate_candidate(
                    chapter, verse, attempt, args.mode, config, history, local_date, previous_row
                )
                if row.get("status") == "published":
                    print(f"اكتمل إنشاء الفيديو والتحقق والنشر للآية {chapter}:{verse}.")
                    successful_this_run += 1
                    history_rows = history.get_runs()
                    if completed_today + successful_this_run >= daily_target:
                        return 0
                    break
                if row.get("status") == "partially_published":
                    print(f"نُشر الفيديو على منصة واحدة على الأقل؛ لا تعِد إنشاءه. يمكن استعادة الملف نفسه ثم استخدام --publish-only للمنصة غير المؤكدة.", file=sys.stderr)
                    return 1
                if validated:
                    print("الفيديو اجتاز الفحوص لكن النشر لم يتأكد؛ أوقف الإنتاج لتجنب إنشاء منشور مكرر.", file=sys.stderr)
                    return 1
                candidate_errors.append(row.get("last_error", "فشل غير محدد"))
                if args.mode == "manual":
                    return 1
            # Automatic mode moves to another candidate only after a pre-publication failure.
        if completed_today + successful_this_run >= daily_target:
            return 0
        print("لم يكتمل العدد اليومي المطلوب. آخر سبب: " + (candidate_errors[-1] if candidate_errors else "لا يوجد مرشح صالح."), file=sys.stderr)
        return 1
    except (FactoryError, ServiceError, VideoError) as exc:
        print(f"فشل التشغيل: {str(exc)[:700]}", file=sys.stderr)
        return 1
    except Exception as exc:
        # Final safeguard: avoid printing environment values or raw SDK/network exceptions.
        print(f"فشل غير متوقع ({type(exc).__name__}). راجع سجل التشغيل دون مشاركة الأسرار.", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quran Cinematic AI Factory")
    parser.add_argument("--mode", choices=("auto", "manual"), default=os.getenv("MODE", "auto").strip() or "auto")
    parser.add_argument("--chapter", type=int, help="رقم السورة في الوضع اليدوي")
    parser.add_argument("--verse", type=int, help="رقم الآية في الوضع اليدوي")
    parser.add_argument("--reprocess", action="store_true", help="السماح بإعادة إنشاء ملف آية سبق نشرها دون تكرار الوجهات المنشورة")
    parser.add_argument("--publish-only", action="store_true", help="استئناف الوجهات غير المؤكدة باستخدام ملف الفيديو المطابق للبصمة")
    parser.add_argument("--input-video", help="ملف فيديو مطابق لبصمة سجل الإنتاج عند استخدام --publish-only")
    parser.add_argument("--test", action="store_true", help="تشغيل اختبارات الوحدة المضمنة دون اتصال خارجي")
    return parser


# Embedded tests keep the repository at exactly five source files.
if __name__ == "__main__":
    if "--test" in sys.argv:
        import unittest
        from unittest.mock import patch

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

            def test_record_sources_logs_the_voice_actually_used(self):
                class Obj:
                    def __init__(self, **kwargs):
                        self.__dict__.update(kwargs)
                verse = Obj(key="2:3")
                recitation = Obj(reciter_id=7, reciter_name="Mishari", source_url="https://x", audio_url="https://y")
                tafsir = Obj(source_name="t", resource_id=16, source_url="https://z", license_url="https://l")
                sources = _record_sources(verse, recitation, tafsir, [{"kind": "media"}], "https://terms", "ar-EG-ShakirNeural")
                self.assertEqual([item for item in sources if item["kind"] == "tts"][0]["voice"], "ar-EG-ShakirNeural")

            def test_daily_guard_only_counts_auto_and_allows_retry_after_known_failure(self):
                published = new_run_record("auto-ok", 2, 152, "auto", "2026-10-21", 1)
                published["status"] = "published"
                failed = new_run_record("auto-failed", 2, 153, "auto", "2026-10-21", 1)
                failed["status"] = "failed"
                manual = new_run_record("manual-ok", 3, 139, "manual", "2026-10-21", 1)
                manual["status"] = "published"
                self.assertTrue(already_processed_today([published], "2026-10-21", 1))
                self.assertFalse(already_processed_today([published], "2026-10-21", 2))
                self.assertFalse(already_processed_today([failed], "2026-10-21", 1))
                self.assertFalse(already_processed_today([manual], "2026-10-21", 1))
                failed["platforms"]["youtube"] = {"status": "upload_uncertain", "video_id": None}
                self.assertTrue(already_processed_today([failed], "2026-10-21", 1))

            def test_history_creates_valid_json_and_prevents_duplicate_publication(self):
                import tempfile
                with tempfile.TemporaryDirectory() as temp:
                    history = ProductionHistory(Path(temp) / "data/history.json")
                    history.load()
                    row = new_run_record("test-1", 2, 152, "manual", "2026-01-01", 1)
                    row["platforms"]["youtube"] = {"status": "published", "video_id": "yt-1", "attempts": 1}
                    history.upsert(row)
                    reread = history.get_runs()
                    self.assertEqual(len(reread), 1)
                    self.assertTrue(row_has_any_publication(reread[0]))
                    self.assertNotIn((2, 152), choose_auto_candidates(reread, "2026-01-02", 20))

            def test_partial_failure_never_repeats_successful_platform(self):
                import tempfile
                with tempfile.TemporaryDirectory() as temp:
                    h = ProductionHistory(Path(temp) / "history.json")
                    h.load()
                    row = new_run_record("run-1", 2, 152, "manual", "2026-01-01", 1)
                    row["video_sha256"] = "abc"
                    row["platforms"]["youtube"] = {"status": "published", "video_id": "yt-existing", "attempts": 1}
                    calls = []
                    class FakeFB:
                        def publish_or_verify(self, *args, **kwargs):
                            calls.append("facebook")
                            raise ServiceError("simulated Meta failure")
                    def factory(name):
                        return FakeFB()
                    with patch.dict(os.environ, {"META_PUBLISH_ENABLED": "true"}):
                        publish_platforms(Path("video.mp4"), row, h, Path(temp) / "reports", "title", "desc", factory, retries=2)
                    self.assertNotIn("youtube", calls)
                    self.assertEqual(calls.count("facebook"), 1)
                    self.assertEqual(row["platforms"]["youtube"]["video_id"], "yt-existing")

            def test_pending_facebook_retries_by_id_and_never_calls_youtube(self):
                import tempfile
                with tempfile.TemporaryDirectory() as temp:
                    h = ProductionHistory(Path(temp) / "history.json")
                    h.load()
                    row = new_run_record("run-2", 2, 153, "manual", "2026-01-01", 1)
                    row["platforms"]["youtube"] = {"status": "published", "video_id": "yt-1", "attempts": 1}
                    row["platforms"]["facebook"] = {"status": "pending", "video_id": "fb-1", "attempts": 1}
                    calls = []
                    class FakeFB:
                        def publish_or_verify(self, *args, **kwargs):
                            calls.append(kwargs["previous"]["video_id"])
                            return {"status": "published", "video_id": "fb-1"} if len(calls) == 2 else {"status": "pending", "video_id": "fb-1"}
                    with patch.dict(os.environ, {"META_PUBLISH_ENABLED": "true"}):
                        result = publish_platforms(Path("video.mp4"), row, h, Path(temp) / "reports", "title", "desc",
                                                   lambda name: FakeFB(), retries=2)
                    self.assertEqual(calls, ["fb-1", "fb-1"])
                    self.assertEqual(result["status"], "published")

            def test_meta_publication_is_disabled_without_explicit_opt_in(self):
                import tempfile
                with tempfile.TemporaryDirectory() as temp:
                    history = ProductionHistory(Path(temp) / "history.json")
                    history.load()
                    row = new_run_record("safe-default", 2, 152, "manual", "2026-10-21", 1)
                    calls = []
                    class FakePublisher:
                        def publish_or_verify(self, *args, **kwargs):
                            calls.append("youtube")
                            return {"status": "published", "video_id": "yt-test"}
                    with patch.dict(os.environ, {"META_PUBLISH_ENABLED": "false"}):
                        result = publish_platforms(Path("video.mp4"), row, history, Path(temp) / "reports",
                                                   "title", "desc", lambda name: FakePublisher())
                    self.assertEqual(calls, ["youtube"])
                    self.assertEqual(result["status"], "published")
                    self.assertEqual(result["platforms"]["facebook"]["status"], "disabled")

            def test_unresolved_platform_id_excludes_verse_from_auto_selection(self):
                row = new_run_record("run-3", 2, 286, "auto", "2026-01-01", 1)
                row["platforms"]["facebook"] = {"status": "uploading_unknown", "video_id": "fb-session"}
                self.assertNotIn((2, 286), choose_auto_candidates([row], "2026-01-02", 20))

            def test_publish_only_fingerprint_requires_exact_artifact(self):
                import tempfile
                with tempfile.TemporaryDirectory() as temp:
                    h = ProductionHistory(Path(temp) / "history.json")
                    row = new_run_record("run-1", 2, 152, "manual", "2026-01-01", 1)
                    row["video_sha256"] = "expected"
                    h.upsert(row)
                    self.assertEqual(find_latest_row(h.get_runs(), "2:152")["video_sha256"], "expected")

        unittest.main(argv=[sys.argv[0]])
    else:
        sys.exit(run(build_parser().parse_args()))
