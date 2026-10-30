"""Free AI scene images for the video background.

All network access for images lives here. Providers (in order of IMAGE_PROVIDERS, default
"cloudflare"): Cloudflare Workers AI (FLUX.1 schnell, free daily allowance) and, only when you list
it, Pollinations. Every failure degrades gracefully: a scene without an image is drawn by the
built-in procedural renderer, so the daily production never stops because of a picture.

Safety: prompts come from a model that read the tafsir, so they are sanitised (English only, no
sacred figures, no faces, no links) before reaching an image model, and a house style that forbids
faces and text is appended in code, not left to the model.
"""
from __future__ import annotations

import base64
import hashlib
import io
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import requests
from PIL import Image

CF_API = "https://api.cloudflare.com/client/v4"
DEFAULT_CF_MODEL = "@cf/black-forest-labs/flux-1-schnell"
POLLINATIONS_URL = "https://image.pollinations.ai/prompt/"
KNOWN_PROVIDERS = ("cloudflare", "pollinations")
CF_QUOTA_CODES = {4006}                       # "daily free allocation exceeded"
RETRY_DELAYS = (2.0, 6.0)
MIN_IMAGE_SIDE = 256
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
PROMPT_MIN_CHARS, PROMPT_MAX_CHARS = 20, 400

STYLE_SUFFIX = ("cinematic vertical illustration, soft dramatic lighting, rich colour palette, depth and atmosphere, "
                "no text, no letters, no logos, no faces, no portraits, people only as small distant silhouettes")
FORBIDDEN_PROMPT_TERMS = re.compile(
    r"\b(prophet|prophets|messenger|muhammad|mohammed|mohammad|allah|god|angel|angels|jesus|moses|abraham|noah|"
    r"quran|koran|bible|kaaba|kaabah|mecca|makkah|face|faces|portrait|selfie|nude|naked|topless|erotic|"
    r"blood|gore|corpse|weapon|weapons|gun|guns|swastika|flag|idol|idols|cross|logo|brand)\b|https?:|www\.",
    re.I)


class ImageGenerationError(RuntimeError):
    """A sanitised failure of one image request (never contains a secret)."""


class ImageQuotaError(ImageGenerationError):
    """The provider's free allowance (or rate limit) is exhausted: stop using it for this run."""


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def sanitize_image_prompt(text: Any) -> str:
    """Return a safe English prompt, or "" when the text must not be sent to an image model."""
    if not isinstance(text, str):
        return ""
    value = re.sub(r"\s+", " ", text).strip().strip("\"'")
    if not PROMPT_MIN_CHARS <= len(value) <= PROMPT_MAX_CHARS:
        return ""
    letters = [char for char in value if char.isalpha()]
    if not letters or sum(1 for char in letters if char.isascii()) / len(letters) < 0.95:
        return ""
    if FORBIDDEN_PROMPT_TERMS.search(value):
        return ""
    return value


def build_final_prompt(prompt: str) -> str:
    return f"{prompt.rstrip('. ')}. {STYLE_SUFFIX}"


def normalize_image(data: bytes, destination: Path) -> dict[str, Any]:
    """Validate downloaded bytes and store them as a clean RGB JPEG."""
    if not isinstance(data, (bytes, bytearray)) or not 2048 <= len(data) <= MAX_IMAGE_BYTES:
        raise ImageGenerationError("حجم الصورة المستلمة غير معقول.")
    try:
        image = Image.open(io.BytesIO(data))
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ImageGenerationError("أبعاد الصورة المستلمة كبيرة جدًا.")
        image.load()
    except ImageGenerationError:
        raise
    except Exception:
        raise ImageGenerationError("الصورة المستلمة غير قابلة للقراءة.") from None
    if min(image.size) < MIN_IMAGE_SIDE:
        raise ImageGenerationError("الصورة المستلمة صغيرة جدًا.")
    image = image.convert("RGB")
    low, high = image.convert("L").resize((32, 32)).getextrema()
    if high - low < 12:
        raise ImageGenerationError("الصورة المستلمة شبه فارغة.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, "JPEG", quality=92)
    return {"width": image.width, "height": image.height,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}


class CloudflareProvider:
    """Cloudflare Workers AI REST API. FLUX.1 schnell returns a square image (it ignores size)."""
    name = "cloudflare"
    _sleep = staticmethod(time.sleep)

    def __init__(self, session: requests.Session | None = None):
        self.account = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
        self.token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
        if not re.fullmatch(r"[0-9a-f]{32}", self.account) or not self.token:
            raise ImageGenerationError("CLOUDFLARE_ACCOUNT_ID أو CLOUDFLARE_API_TOKEN غير مضبوط أو غير صالح.")
        self.model = os.getenv("CF_IMAGE_MODEL", "").strip() or DEFAULT_CF_MODEL
        if not re.fullmatch(r"@cf/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+", self.model):
            raise ImageGenerationError("CF_IMAGE_MODEL غير صالح.")
        self.send_size = env_flag("CF_IMAGE_SIZE_PARAMS")       # for SDXL-type models that accept width/height
        self.session = session or requests.Session()

    def _body(self, prompt: str, width: int, height: int, seed: int) -> dict[str, Any]:
        body: dict[str, Any] = {"prompt": prompt}
        if "flux" in self.model:
            body["steps"] = 4
        if self.send_size:
            body.update(width=int(width), height=int(height), seed=int(seed) % 2_147_483_647)
        return body

    def generate(self, prompt: str, width: int, height: int, seed: int) -> bytes:
        url = f"{CF_API}/accounts/{self.account}/ai/run/{self.model}"
        headers = {"Authorization": f"Bearer {self.token}"}
        body = self._body(prompt, width, height, seed)
        rate_limited = False
        last = "unknown"
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                response = self.session.post(url, headers=headers, json=body, timeout=(15, 120))
            except requests.RequestException as exc:
                last = type(exc).__name__
            else:
                status = response.status_code
                if status == 429:
                    if rate_limited:
                        raise ImageQuotaError("حد Cloudflare المجاني أو معدل الطلبات استُنفد.")
                    rate_limited, last = True, "HTTP 429"
                elif status in (401, 403):
                    raise ImageGenerationError("رفض Cloudflare المصادقة؛ تحقق من صلاحية Workers AI للرمز.")
                elif status >= 500 or status in (408, 425):
                    last = f"HTTP {status}"
                elif status != 200:
                    raise ImageGenerationError(f"Cloudflare أعاد HTTP {status}.")
                else:
                    return self._extract(response)
            if attempt < len(RETRY_DELAYS):
                self._sleep(RETRY_DELAYS[attempt])
        raise ImageGenerationError(f"فشل طلب صورة Cloudflare ({last}).")

    @staticmethod
    def _extract(response: Any) -> bytes:
        content_type = (response.headers.get("Content-Type", "") or "").split(";", 1)[0].lower()
        if content_type.startswith("image/"):
            return bytes(response.content)
        try:
            payload = response.json()
        except ValueError:
            raise ImageGenerationError("استجابة Cloudflare ليست JSON صالحًا.") from None
        if not isinstance(payload, dict):
            raise ImageGenerationError("بنية استجابة Cloudflare غير صالحة.")
        if payload.get("success") is False:
            errors = payload.get("errors") or [{}]
            code = errors[0].get("code") if isinstance(errors[0], dict) else None
            if code in CF_QUOTA_CODES:
                raise ImageQuotaError("حد Cloudflare المجاني اليومي استُنفد.")
            raise ImageGenerationError(f"رفض Cloudflare الطلب (code={code}).")
        result = payload.get("result")
        encoded = result.get("image") if isinstance(result, dict) else None
        if not isinstance(encoded, str) or not encoded:
            raise ImageGenerationError("استجابة Cloudflare لا تحتوي صورة.")
        try:
            return base64.b64decode(encoded, validate=False)
        except ValueError:
            raise ImageGenerationError("صورة Cloudflare بترميز base64 غير صالح.") from None


class PollinationsProvider:
    """Pollinations: no key, but rate limited and with terms that may change; opt-in only."""
    name = "pollinations"
    model = "flux"
    _sleep = staticmethod(time.sleep)

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()

    def generate(self, prompt: str, width: int, height: int, seed: int) -> bytes:
        url = POLLINATIONS_URL + quote(prompt, safe="")
        params = {"width": int(width), "height": int(height), "model": self.model, "nologo": "true", "seed": int(seed)}
        last = "unknown"
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                response = self.session.get(url, params=params, timeout=(15, 120))
            except requests.RequestException as exc:
                last = type(exc).__name__
            else:
                if response.status_code == 429:
                    raise ImageQuotaError("حد معدل Pollinations استُنفد.")
                if response.status_code == 200:
                    kind = (response.headers.get("Content-Type", "") or "").split(";", 1)[0].lower()
                    if not kind.startswith("image/"):
                        raise ImageGenerationError("استجابة Pollinations ليست صورة.")
                    return bytes(response.content)
                last = f"HTTP {response.status_code}"
                if response.status_code < 500:
                    break
            if attempt < len(RETRY_DELAYS):
                self._sleep(RETRY_DELAYS[attempt])
        raise ImageGenerationError(f"فشل طلب صورة Pollinations ({last}).")


class SceneImageService:
    """One image per scene of the visual plan; scenes that fail simply get no image."""
    pause_seconds = 1.0
    _sleep = staticmethod(time.sleep)

    def __init__(self, providers: list[Any] | None = None):
        self.notes: list[str] = []
        self.providers: list[Any] = providers if providers is not None else self._providers_from_env()

    def _providers_from_env(self) -> list[Any]:
        if not env_flag("IMAGE_GEN_LICENSE_CONFIRMED") or urlparse(os.getenv("IMAGE_GEN_LICENSE_URL", "").strip()).scheme != "https":
            self.notes.append("توليد الصور معطل: اضبط IMAGE_GEN_LICENSE_CONFIRMED=true وIMAGE_GEN_LICENSE_URL (HTTPS) بعد مراجعة الشروط.")
            return []
        names = [part.strip().lower() for part in (os.getenv("IMAGE_PROVIDERS", "").strip() or "cloudflare").split(",") if part.strip()]
        providers: list[Any] = []
        for name in names:
            if name not in KNOWN_PROVIDERS:
                self.notes.append(f"مزوّد صور غير معروف: {name}.")
                continue
            try:
                providers.append(CloudflareProvider() if name == "cloudflare" else PollinationsProvider())
            except ImageGenerationError as exc:
                self.notes.append(f"{name}: {str(exc)[:160]}")
        if not providers:
            self.notes.append("لا يوجد مزوّد صور صالح؛ ستُرسم المشاهد إجرائيًا.")
        return providers

    def generate(self, plan: dict[str, Any], out_dir: Path, width: int, height: int) -> dict[str, dict[str, Any]]:
        """Return {"opening"|"scene-N"|"closing": {path, provider, model, prompt, sha256, ...}}."""
        if not self.providers:
            return {}
        jobs = ([("opening", plan.get("opening"))]
                + [(f"scene-{number}", scene) for number, scene in enumerate(plan.get("scenes", []), start=1)]
                + [("closing", plan.get("closing"))])
        results: dict[str, dict[str, Any]] = {}
        exhausted: set[str] = set()
        for key, scene in jobs:
            prompt = sanitize_image_prompt((scene or {}).get("image_prompt"))
            if not prompt:
                self.notes.append(f"{key}: لا يوجد image_prompt صالح.")
                continue
            final_prompt = build_final_prompt(prompt)
            seed = int(hashlib.sha256(f"{key}|{prompt}".encode("utf-8")).hexdigest()[:8], 16)
            for provider in self.providers:
                if provider.name in exhausted:
                    continue
                try:
                    data = provider.generate(final_prompt, width, height, seed)
                    info = normalize_image(data, out_dir / f"{key}.jpg")
                except ImageQuotaError as exc:
                    exhausted.add(provider.name)
                    self.notes.append(f"{provider.name}: {exc}")
                    continue
                except ImageGenerationError as exc:
                    self.notes.append(f"{key}/{provider.name}: {str(exc)[:160]}")
                    continue
                results[key] = {"path": str(out_dir / f"{key}.jpg"), "provider": provider.name,
                                "model": getattr(provider, "model", ""), "prompt": prompt, **info}
                break
            self._sleep(self.pause_seconds)
        return results


# ------------------------------------------------------------------ tests
if __name__ == "__main__":
    import tempfile
    import unittest
    from unittest.mock import patch

    def png_bytes(size=(300, 300), flat=False):
        if flat:
            image = Image.new("RGB", size, (40, 40, 40))
        else:                                          # gradient + noise: compresses like a real picture
            from PIL import ImageChops
            gradient = Image.linear_gradient("L").resize(size)
            noise = Image.effect_noise(size, 40)
            image = Image.merge("RGB", (ImageChops.add(gradient, noise, scale=2), gradient, noise))
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        return buffer.getvalue()

    class Resp:
        def __init__(self, status=200, payload=None, content=b"", content_type="application/json"):
            self.status_code, self._payload, self.content = status, payload, content
            self.headers = {"Content-Type": content_type}
        def json(self):
            if self._payload is None:
                raise ValueError("no json")
            return self._payload

    class Session:
        def __init__(self, responses):
            self.responses, self.calls = list(responses), []
        def post(self, url, **kwargs):
            self.calls.append(("post", url, kwargs)); return self.responses.pop(0)
        def get(self, url, **kwargs):
            self.calls.append(("get", url, kwargs)); return self.responses.pop(0)

    ENV = {"CLOUDFLARE_ACCOUNT_ID": "a" * 32, "CLOUDFLARE_API_TOKEN": "token-secret", "CF_IMAGE_MODEL": "",
           "CF_IMAGE_SIZE_PARAMS": ""}
    OK = {"success": True, "result": {"image": base64.b64encode(png_bytes()).decode()}}

    def cloudflare(responses):
        with patch.dict(os.environ, ENV):
            provider = CloudflareProvider(session=Session(responses))
        provider._sleep = lambda seconds: None
        return provider

    class SceneImageTests(unittest.TestCase):
        def test_prompt_sanitiser_keeps_safe_english_and_drops_everything_else(self):
            good = "A lone traveller walking a winding road toward a glowing sunrise over quiet hills"
            self.assertEqual(sanitize_image_prompt(f"  {good}  "), good)
            for bad in ("قصة طويلة بالعربية عن الصبر والأمل في الحياة اليومية", "short", "x" * 500,
                        "The prophet standing in the desert at dawn with light behind him",
                        "An angel above the clouds over a sleeping city at night",
                        "A close portrait of a sad man looking at the camera in the rain",
                        "A calm sea at dusk, see https://example.org for details", None, 42):
                self.assertEqual(sanitize_image_prompt(bad), "", bad)

        def test_final_prompt_appends_the_house_style_in_code(self):
            final = build_final_prompt("A quiet window with rain on the glass.")
            self.assertTrue(final.endswith(STYLE_SUFFIX))
            self.assertIn("no faces", final)

        def test_cloudflare_request_shape_and_json_image(self):
            provider = cloudflare([Resp(200, OK)])
            data = provider.generate("a prompt", 1080, 1920, 7)
            self.assertEqual(Image.open(io.BytesIO(data)).size, (300, 300))
            _, url, kwargs = provider.session.calls[0]
            self.assertEqual(url, f"{CF_API}/accounts/{'a' * 32}/ai/run/{DEFAULT_CF_MODEL}")
            self.assertEqual(kwargs["headers"], {"Authorization": "Bearer token-secret"})
            self.assertEqual(kwargs["json"], {"prompt": "a prompt", "steps": 4})        # no size for FLUX schnell

        def test_cloudflare_can_send_size_for_sdxl_type_models(self):
            with patch.dict(os.environ, {**ENV, "CF_IMAGE_MODEL": "@cf/stabilityai/stable-diffusion-xl-base-1.0",
                                         "CF_IMAGE_SIZE_PARAMS": "true"}):
                provider = CloudflareProvider(session=Session([Resp(200, OK)]))
            provider.generate("p", 768, 1344, 5)
            body = provider.session.calls[0][2]["json"]
            self.assertEqual((body["width"], body["height"], body["seed"]), (768, 1344, 5))
            self.assertNotIn("steps", body)

        def test_cloudflare_accepts_raw_image_bytes(self):
            raw = png_bytes()
            provider = cloudflare([Resp(200, None, raw, "image/png")])
            self.assertEqual(provider.generate("p", 1, 1, 1), raw)

        def test_cloudflare_errors(self):
            with self.assertRaises(ImageGenerationError):                                  # auth: no retry
                provider = cloudflare([Resp(401, {})]); provider.generate("p", 1, 1, 1)
            self.assertEqual(len(provider.session.calls), 1)
            provider = cloudflare([Resp(429, {}), Resp(429, {})])                          # second 429 = quota
            with self.assertRaises(ImageQuotaError):
                provider.generate("p", 1, 1, 1)
            provider = cloudflare([Resp(200, {"success": False, "errors": [{"code": 4006}]})])
            with self.assertRaises(ImageQuotaError):
                provider.generate("p", 1, 1, 1)
            provider = cloudflare([Resp(500, {}), Resp(502, {}), Resp(200, OK)])           # transient: retried
            self.assertTrue(provider.generate("p", 1, 1, 1))
            provider = cloudflare([Resp(500, {})] * 3)
            with self.assertRaises(ImageGenerationError):
                provider.generate("p", 1, 1, 1)
            with self.assertRaises(ImageGenerationError):
                cloudflare([Resp(200, {"success": True, "result": {}})]).generate("p", 1, 1, 1)

        def test_cloudflare_rejects_bad_credentials_and_model_names(self):
            for env in ({"CLOUDFLARE_ACCOUNT_ID": "short"}, {"CLOUDFLARE_API_TOKEN": ""}, {"CF_IMAGE_MODEL": "evil/model"}):
                with patch.dict(os.environ, {**ENV, **env}):
                    with self.assertRaises(ImageGenerationError, msg=str(env)):
                        CloudflareProvider()

        def test_pollinations_requires_image_content_type_and_reports_quota(self):
            provider = PollinationsProvider(Session([Resp(200, None, png_bytes(), "image/jpeg")]))
            self.assertTrue(provider.generate("a b/c", 1080, 1920, 3))
            _, url, kwargs = provider.session.calls[0]
            self.assertTrue(url.startswith(POLLINATIONS_URL + "a%20b%2Fc"))
            self.assertEqual((kwargs["params"]["width"], kwargs["params"]["nologo"]), (1080, "true"))
            with self.assertRaises(ImageGenerationError):
                PollinationsProvider(Session([Resp(200, None, b"<html>", "text/html")])).generate("p", 1, 1, 1)
            with self.assertRaises(ImageQuotaError):
                PollinationsProvider(Session([Resp(429, {})])).generate("p", 1, 1, 1)

        def test_normalize_image_rejects_tiny_blank_corrupt_and_stores_a_jpeg(self):
            with tempfile.TemporaryDirectory() as temp:
                target = Path(temp) / "a" / "x.jpg"
                info = normalize_image(png_bytes((400, 600)), target)
                self.assertEqual((info["width"], info["height"]), (400, 600))
                self.assertEqual(Image.open(target).format, "JPEG")
                for bad in (png_bytes((100, 100)), png_bytes((300, 300), flat=True), b"not an image" * 400, b"x"):
                    with self.assertRaises(ImageGenerationError):
                        normalize_image(bad, Path(temp) / "bad.jpg")

        @staticmethod
        def plan():
            prompt = "A lone traveller walking a winding road toward a glowing sunrise over quiet hills"
            return {"opening": {"image_prompt": prompt}, "closing": {"image_prompt": prompt},
                    "scenes": [{"image_prompt": prompt}, {"image_prompt": "الصبر"}, {"image_prompt": prompt}]}

        class Fake:
            def __init__(self, name, behaviour):
                self.name, self.model, self.behaviour, self.calls = name, "m", behaviour, 0
            def generate(self, prompt, width, height, seed):
                self.calls += 1
                outcome = self.behaviour(self.calls)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

        def run_service(self, providers):
            service = SceneImageService(providers=providers)
            service._sleep = lambda seconds: None
            with tempfile.TemporaryDirectory() as temp:
                return service, service.generate(self.plan(), Path(temp) / "images", 1080, 1920)

        def test_service_generates_one_image_per_valid_scene_and_skips_bad_prompts(self):
            first = self.Fake("cloudflare", lambda n: png_bytes())
            service, results = self.run_service([first])
            self.assertEqual(sorted(results), ["closing", "opening", "scene-1", "scene-3"])
            self.assertEqual(first.calls, 4)
            self.assertTrue(any("scene-2" in note for note in service.notes))
            self.assertEqual(results["opening"]["provider"], "cloudflare")

        def test_service_falls_back_to_the_next_provider_and_stops_using_an_exhausted_one(self):
            first = self.Fake("cloudflare", lambda n: ImageQuotaError("quota"))
            second = self.Fake("pollinations", lambda n: png_bytes())
            _, results = self.run_service([first, second])
            self.assertEqual(first.calls, 1)                                   # never asked again after quota
            self.assertEqual(second.calls, 4)
            self.assertTrue(all(item["provider"] == "pollinations" for item in results.values()))

        def test_service_survives_total_failure_and_bad_image_bytes(self):
            _, results = self.run_service([self.Fake("cloudflare", lambda n: ImageGenerationError("down"))])
            self.assertEqual(results, {})
            _, results = self.run_service([self.Fake("cloudflare", lambda n: b"garbage" * 500)])
            self.assertEqual(results, {})

        def test_service_is_disabled_without_license_confirmation_or_credentials(self):
            env = {**ENV, "IMAGE_PROVIDERS": "", "IMAGE_GEN_LICENSE_CONFIRMED": "", "IMAGE_GEN_LICENSE_URL": ""}
            with patch.dict(os.environ, env):
                service = SceneImageService()
            self.assertEqual(service.providers, [])
            self.assertTrue(service.notes)
            ready = {**env, "IMAGE_GEN_LICENSE_CONFIRMED": "true", "IMAGE_GEN_LICENSE_URL": "https://example.org/terms"}
            with patch.dict(os.environ, ready):
                self.assertEqual([p.name for p in SceneImageService().providers], ["cloudflare"])
            with patch.dict(os.environ, {**ready, "IMAGE_PROVIDERS": "cloudflare,pollinations,bogus"}):
                service = SceneImageService()
                self.assertEqual([p.name for p in service.providers], ["cloudflare", "pollinations"])
                self.assertTrue(any("bogus" in note for note in service.notes))
            with patch.dict(os.environ, {**ready, "CLOUDFLARE_API_TOKEN": ""}):
                self.assertEqual(SceneImageService().providers, [])
            with patch.dict(os.environ, {**ready, "IMAGE_GEN_LICENSE_URL": "http://insecure"}):
                self.assertEqual(SceneImageService().providers, [])

    unittest.main()
