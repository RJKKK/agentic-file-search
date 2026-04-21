"""
Context budget management for LLM calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from google.genai.types import Content, Part

_UNIT_RE = re.compile(r"\[UNIT\s+(\d+)\b", re.IGNORECASE)


@dataclass
class _MessageRecord:
    index: int
    role: str
    original_text: str
    text: str
    bucket: str
    priority: int
    dropped: bool = False


class ContextBudgetManager:
    """Hard-budget context compaction with anchor-centered radial compression."""

    def __init__(
        self,
        *,
        max_input_tokens: int = 12000,
        min_recent_messages: int = 6,
    ) -> None:
        self.max_input_tokens = max_input_tokens
        self.min_recent_messages = min_recent_messages

    def compact_history(
        self,
        history: list[Content],
        *,
        anchor_unit_no: int | None = None,
    ) -> tuple[list[Content], dict[str, int | float | str | bool]]:
        if not history:
            return [], self._empty_stats()

        records = self._build_records(history, anchor_unit_no=anchor_unit_no)
        before_tokens = self._estimate_tokens_records(records)

        self._apply_bucket_caps(
            records,
            caps={
                "current": 3600,
                "ring0": 3000,
                "ring1": 1800,
                "ring2": 900,
                "history": 900,
            },
        )
        if self._estimate_tokens_records(records) > self.max_input_tokens:
            self._apply_bucket_caps(
                records,
                caps={
                    "current": 3000,
                    "ring0": 2200,
                    "ring1": 1200,
                    "ring2": 600,
                    "history": 450,
                },
            )

        if self._estimate_tokens_records(records) > self.max_input_tokens:
            self._drop_oldest_bucket(
                records,
                bucket="history",
                min_keep=self.min_recent_messages,
            )
        if self._estimate_tokens_records(records) > self.max_input_tokens:
            self._drop_oldest_bucket(records, bucket="ring2", min_keep=3)
        if self._estimate_tokens_records(records) > self.max_input_tokens:
            self._drop_oldest_bucket(records, bucket="ring1", min_keep=2)

        if self._estimate_tokens_records(records) > self.max_input_tokens:
            self._apply_bucket_caps(
                records,
                caps={
                    "current": 1200,
                    "ring0": 900,
                    "ring1": 600,
                    "ring2": 350,
                    "history": 220,
                },
            )

        while self._estimate_tokens_records(records) > self.max_input_tokens:
            alive = [record for record in records if not record.dropped]
            if len(alive) <= 1:
                break
            candidates = [
                record
                for record in alive
                if record.bucket != "current" and record.priority < 100
            ]
            if not candidates:
                candidates = alive[:-1]
            if not candidates:
                break
            oldest = min(candidates, key=lambda record: (record.priority, record.index))
            oldest.dropped = True

        self._hard_cap(records)

        output = self._records_to_content(records, history)
        after_tokens = self._estimate_tokens_content(output)
        truncated_messages = sum(
            1 for record in records if not record.dropped and record.text != record.original_text
        )
        dropped_messages = sum(1 for record in records if record.dropped)
        warning = after_tokens > self.max_input_tokens

        stats: dict[str, int | float | str | bool] = {
            "before_tokens": before_tokens,
            "after_tokens": after_tokens,
            "hard_limit_tokens": self.max_input_tokens,
            "truncated_messages": truncated_messages,
            "dropped_messages": dropped_messages,
            "compression_ratio": round(after_tokens / max(before_tokens, 1), 4),
            "overflow_warning": warning,
        }
        if warning:
            stats["warning"] = (
                "Context still exceeds hard budget after compaction; consider "
                "asking for narrower scope."
            )
        return output, stats

    def _hard_cap(self, records: list[_MessageRecord]) -> None:
        """Final deterministic clamp to enforce hard token limits."""
        guard = 0
        while self._estimate_tokens_records(records) > self.max_input_tokens and guard < 200:
            guard += 1
            candidates = [
                record
                for record in records
                if not record.dropped and record.bucket != "current"
            ]
            if not candidates:
                candidates = [record for record in records if not record.dropped]
            if not candidates:
                break
            target = min(candidates, key=lambda record: (record.priority, record.index))
            if len(target.text) > 140:
                target.text = self._compress_text(
                    target.text,
                    max_chars=max(120, int(len(target.text) * 0.65)),
                    bucket=target.bucket,
                )
                continue
            target.dropped = True

    @staticmethod
    def _empty_stats() -> dict[str, int | float | str | bool]:
        return {
            "before_tokens": 0,
            "after_tokens": 0,
            "hard_limit_tokens": 0,
            "truncated_messages": 0,
            "dropped_messages": 0,
            "compression_ratio": 1.0,
            "overflow_warning": False,
        }

    def _build_records(
        self,
        history: list[Content],
        *,
        anchor_unit_no: int | None,
    ) -> list[_MessageRecord]:
        records: list[_MessageRecord] = []
        last_index = len(history) - 1
        for idx, content in enumerate(history):
            text = self._content_text(content)
            bucket = self._classify_bucket(text, anchor_unit_no=anchor_unit_no)
            priority = self._priority_for_bucket(bucket=bucket, idx=idx, last_index=last_index)
            role = getattr(content, "role", "user") or "user"
            records.append(
                _MessageRecord(
                    index=idx,
                    role=str(role),
                    original_text=text,
                    text=text,
                    bucket=bucket,
                    priority=priority,
                )
            )
        return records

    @staticmethod
    def _classify_bucket(text: str, *, anchor_unit_no: int | None) -> str:
        if not text:
            return "history"
        if "Tool result for parse_file" not in text:
            return "history"
        if anchor_unit_no is None:
            return "ring2"

        unit_nos = [int(match.group(1)) for match in _UNIT_RE.finditer(text)]
        if not unit_nos:
            return "ring2"
        distance = min(abs(unit_no - anchor_unit_no) for unit_no in unit_nos)
        if distance == 0:
            return "ring0"
        if distance <= 2:
            return "ring1"
        return "ring2"

    @staticmethod
    def _priority_for_bucket(*, bucket: str, idx: int, last_index: int) -> int:
        if idx == last_index:
            return 100
        if bucket == "ring0":
            return 90
        if bucket == "ring1":
            return 80
        if bucket == "ring2":
            return 70
        return 40 + min(idx, 20)

    def _apply_bucket_caps(
        self,
        records: list[_MessageRecord],
        *,
        caps: dict[str, int],
    ) -> None:
        for record in records:
            if record.dropped:
                continue
            cap = caps.get(record.bucket, caps["history"])
            record.text = self._compress_text(record.text, max_chars=cap, bucket=record.bucket)

    def _drop_oldest_bucket(
        self,
        records: list[_MessageRecord],
        *,
        bucket: str,
        min_keep: int,
    ) -> None:
        while self._estimate_tokens_records(records) > self.max_input_tokens:
            alive = [record for record in records if not record.dropped]
            if len(alive) <= min_keep:
                break
            candidates = [
                record
                for record in records
                if not record.dropped and record.bucket == bucket
            ]
            if not candidates:
                break
            oldest = min(candidates, key=lambda record: record.index)
            oldest.dropped = True

    @staticmethod
    def _compress_text(text: str, *, max_chars: int, bucket: str) -> str:
        if len(text) <= max_chars:
            return text
        if max_chars < 120:
            return text[: max_chars - 20] + "\n...[truncated]..."

        if bucket == "ring2":
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            highlights = [line for line in lines if line.startswith("[UNIT ") or line.startswith("#")]
            highlight_blob = "\n".join(highlights[:8])
            head = text[: max_chars // 3]
            merged = f"{head}\n\n...[ring2 compressed]...\n{highlight_blob}".strip()
            if len(merged) > max_chars:
                return merged[: max_chars - 20] + "\n...[truncated]..."
            return merged

        head = int(max_chars * 0.6)
        tail = max_chars - head - 30
        return f"{text[:head]}\n...[compressed]...\n{text[-tail:]}"

    @staticmethod
    def _content_text(content: Content) -> str:
        parts = getattr(content, "parts", None) or []
        texts: list[str] = []
        for part in parts:
            text = getattr(part, "text", None)
            if text is not None:
                texts.append(str(text))
        return "\n".join(texts).strip()

    def _records_to_content(
        self,
        records: list[_MessageRecord],
        history: list[Content],
    ) -> list[Content]:
        output: list[Content] = []
        by_index = {record.index: record for record in records}
        for idx, original in enumerate(history):
            record = by_index[idx]
            if record.dropped:
                continue
            if record.text == record.original_text:
                output.append(original)
                continue
            output.append(
                Content(
                    role=record.role,
                    parts=[Part.from_text(text=record.text)],
                )
            )
        return output

    @staticmethod
    def _estimate_tokens_text(text: str) -> int:
        # Lightweight, model-agnostic approximation.
        return max(1, (len(text) + 3) // 4)

    def _estimate_tokens_records(self, records: list[_MessageRecord]) -> int:
        return sum(
            self._estimate_tokens_text(record.text)
            for record in records
            if not record.dropped
        )

    def _estimate_tokens_content(self, content: list[Content]) -> int:
        return sum(self._estimate_tokens_text(self._content_text(item)) for item in content)
