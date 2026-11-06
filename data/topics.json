"""Topic bank for the "idea first, verse as evidence" video template.

Each topic is an idea (a curiosity question), the facts the script may rely on (written and verified by
the operator, never invented by the model), and the verses (or verse fragments) that are recited at
the moment the script says "قال تعالى". The model only writes the script around them.

File format (data/topics.json):
{
  "version": 1,
  "topics": [
    {"id": "cave-309",
     "question": "لماذا لم يقل القرآن «ثلاثمائة وتسع سنين» مباشرة، مع أن المجموع نفسه؟",
     "verses": [{"key": "18:25"}, {"key": "18:26", "words": [1, 5]}],
     "facts": ["السنة الشمسية نحو 365 يومًا ..."]}
  ]
}
`words` is optional: [first, last], 1-based and inclusive, counted like video_engine.quran_words.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_VERSES_PER_TOPIC = 3
MAX_FACTS = 14
MAX_FACT_CHARS = 500
TOPIC_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9\-]{2,59}")


class TopicError(ValueError):
    """The topic bank is malformed; nothing is produced from a topic that cannot be trusted."""


@dataclass(frozen=True)
class VerseRef:
    chapter: int
    verse: int
    first_word: int | None = None
    last_word: int | None = None

    @property
    def key(self) -> str:
        return f"{self.chapter}:{self.verse}"

    @property
    def words(self) -> tuple[int, int] | None:
        return None if self.first_word is None else (self.first_word, int(self.last_word))

    @property
    def label(self) -> str:
        return self.key if self.words is None else f"{self.key}#{self.first_word}-{self.last_word}"


@dataclass(frozen=True)
class Topic:
    id: str
    question: str
    verses: tuple[VerseRef, ...]
    facts: tuple[str, ...] = ()
    angle: str = ""

    @property
    def row_key(self) -> str:
        """The key under which production history remembers this topic."""
        return f"topic:{self.id}"


def _parse_verse(raw: Any, topic_id: str) -> VerseRef:
    if not isinstance(raw, dict) or not isinstance(raw.get("key"), str):
        raise TopicError(f"{topic_id}: كل آية يجب أن تكون كائنًا فيه الحقل key.")
    match = re.fullmatch(r"(\d{1,3}):(\d{1,3})", raw["key"].strip())
    if not match or not 1 <= int(match.group(1)) <= 114 or int(match.group(2)) < 1:
        raise TopicError(f"{topic_id}: مفتاح الآية {raw['key']!r} غير صالح (المطلوب سورة:آية).")
    chapter, verse = int(match.group(1)), int(match.group(2))
    words = raw.get("words")
    if words is None:
        return VerseRef(chapter, verse)
    if (not isinstance(words, list) or len(words) != 2 or not all(isinstance(n, int) and not isinstance(n, bool) for n in words)
            or not 1 <= words[0] <= words[1] <= 200):
        raise TopicError(f"{topic_id}: words يجب أن تكون [أول, آخر] بأعداد صحيحة 1-based وأول ≤ آخر.")
    return VerseRef(chapter, verse, words[0], words[1])


def parse_topics(raw: Any) -> list[Topic]:
    if not isinstance(raw, dict) or raw.get("version") != 1 or not isinstance(raw.get("topics"), list):
        raise TopicError("صيغة ملف المواضيع غير معروفة (المطلوب version=1 وقائمة topics).")
    topics: list[Topic] = []
    seen: set[str] = set()
    for item in raw["topics"]:
        if not isinstance(item, dict):
            raise TopicError("كل موضوع يجب أن يكون كائنًا.")
        topic_id = item.get("id")
        if not isinstance(topic_id, str) or not TOPIC_ID_PATTERN.fullmatch(topic_id):
            raise TopicError(f"معرّف الموضوع {topic_id!r} غير صالح (حروف إنجليزية صغيرة وأرقام وشرطات، 3 إلى 60).")
        if topic_id in seen:
            raise TopicError(f"معرّف الموضوع {topic_id} مكرّر.")
        seen.add(topic_id)
        question = item.get("question")
        if not isinstance(question, str) or not 10 <= len(question.strip()) <= 300:
            raise TopicError(f"{topic_id}: question يجب أن يكون نصًا من 10 إلى 300 حرف.")
        verses_raw = item.get("verses")
        if not isinstance(verses_raw, list) or not 1 <= len(verses_raw) <= MAX_VERSES_PER_TOPIC:
            raise TopicError(f"{topic_id}: عدد الآيات يجب أن يكون من 1 إلى {MAX_VERSES_PER_TOPIC}.")
        verses = tuple(_parse_verse(entry, topic_id) for entry in verses_raw)
        if len({v.label for v in verses}) != len(verses):
            raise TopicError(f"{topic_id}: آية أو مقطع مكرّر.")
        facts_raw = item.get("facts", [])
        if (not isinstance(facts_raw, list) or len(facts_raw) > MAX_FACTS
                or not all(isinstance(f, str) and 5 <= len(f.strip()) <= MAX_FACT_CHARS for f in facts_raw)):
            raise TopicError(f"{topic_id}: facts قائمة نصوص (حتى {MAX_FACTS}، كل نص من 5 إلى {MAX_FACT_CHARS} حرف).")
        angle = item.get("angle", "")
        if not isinstance(angle, str) or len(angle) > 300:
            raise TopicError(f"{topic_id}: angle يجب أن يكون نصًا حتى 300 حرف.")
        topics.append(Topic(topic_id, question.strip(), verses, tuple(f.strip() for f in facts_raw), angle.strip()))
    if not topics:
        raise TopicError("ملف المواضيع فارغ.")
    return topics


def load_topics(path: Path) -> list[Topic]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError:
        raise TopicError(f"تعذر قراءة ملف المواضيع: {path}") from None
    except json.JSONDecodeError as exc:
        raise TopicError(f"ملف المواضيع ليس JSON صالحًا ({exc.msg} عند السطر {exc.lineno}).") from None
    return parse_topics(raw)


def find_topic(topics: list[Topic], topic_id: str) -> Topic:
    for topic in topics:
        if topic.id == (topic_id or "").strip():
            return topic
    raise TopicError(f"لا يوجد موضوع بالمعرّف {topic_id!r}. المتاح: " + ", ".join(t.id for t in topics))


if __name__ == "__main__":
    import tempfile
    import unittest

    def sample(**overrides):
        topic = {"id": "cave-309", "question": "لماذا لم يقل القرآن ثلاثمائة وتسع سنين مباشرة؟",
                 "verses": [{"key": "18:25"}, {"key": "18:26", "words": [1, 5]}],
                 "facts": ["السنة الشمسية نحو 365 يومًا والقمرية نحو 354 يومًا."]}
        topic.update(overrides)
        return {"version": 1, "topics": [topic]}

    class TopicTests(unittest.TestCase):
        def test_valid_topic_parses_with_fragments_and_labels(self):
            topic = parse_topics(sample())[0]
            self.assertEqual((topic.id, topic.row_key), ("cave-309", "topic:cave-309"))
            self.assertEqual([v.key for v in topic.verses], ["18:25", "18:26"])
            self.assertIsNone(topic.verses[0].words)
            self.assertEqual((topic.verses[1].words, topic.verses[1].label), ((1, 5), "18:26#1-5"))
            self.assertEqual(len(topic.facts), 1)

        def test_facts_and_angle_are_optional(self):
            raw = sample()
            del raw["topics"][0]["facts"]
            self.assertEqual(parse_topics(raw)[0].facts, ())

        def test_malformed_topics_are_rejected_with_a_reason(self):
            bad = [({"version": 2, "topics": []}, "صيغة"), ({"version": 1, "topics": []}, "فارغ"),
                   (sample(id="Bad ID"), "معرّف"), (sample(id="ab"), "معرّف"),
                   (sample(question="قصير"), "question"),
                   (sample(verses=[]), "عدد الآيات"), (sample(verses=[{"key": "1:1"}] * 2), "مكرّر"),
                   (sample(verses=[{"key": "18:25"}] * 1 + [{"key": "2:1"}, {"key": "3:1"}, {"key": "4:1"}]), "عدد الآيات"),
                   (sample(verses=[{"key": "115:1"}]), "غير صالح"), (sample(verses=[{"key": "x"}]), "غير صالح"),
                   (sample(verses=[{"key": "18:26", "words": [5, 1]}]), "words"),
                   (sample(verses=[{"key": "18:26", "words": [0, 3]}]), "words"),
                   (sample(verses=[{"key": "18:26", "words": "1-5"}]), "words"),
                   (sample(facts="نص واحد"), "facts"), (sample(facts=["x"]), "facts"),
                   (sample(facts=["ن" * 600]), "facts"), (sample(angle="ز" * 400), "angle")]
            for raw, reason in bad:
                with self.assertRaisesRegex(TopicError, reason):
                    parse_topics(raw)

        def test_duplicate_ids_are_rejected(self):
            raw = sample()
            raw["topics"].append(dict(raw["topics"][0]))
            with self.assertRaisesRegex(TopicError, "مكرّر"):
                parse_topics(raw)

        def test_load_and_find(self):
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "topics.json"
                path.write_text(json.dumps(sample(), ensure_ascii=False), encoding="utf-8")
                topics = load_topics(path)
                self.assertEqual(find_topic(topics, " cave-309 ").id, "cave-309")
                with self.assertRaisesRegex(TopicError, "المتاح"):
                    find_topic(topics, "nope")
                path.write_text("{not json", encoding="utf-8")
                with self.assertRaisesRegex(TopicError, "JSON"):
                    load_topics(path)
                with self.assertRaisesRegex(TopicError, "تعذر قراءة"):
                    load_topics(Path(temp) / "missing.json")

    unittest.main()
