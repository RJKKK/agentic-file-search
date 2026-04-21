"""
Session-scoped structured context state for agent evidence management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def _snippet(text: str, *, limit: int = 220) -> str:
    normalized = _normalize_text(text)
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."


def compress_unit_ranges(unit_nos: list[int] | set[int] | tuple[int, ...]) -> list[dict[str, int]]:
    ordered = sorted({int(unit_no) for unit_no in unit_nos if int(unit_no) > 0})
    if not ordered:
        return []
    ranges: list[dict[str, int]] = []
    start = ordered[0]
    end = ordered[0]
    for unit_no in ordered[1:]:
        if unit_no == end + 1:
            end = unit_no
            continue
        ranges.append({"start": start, "end": end})
        start = end = unit_no
    ranges.append({"start": start, "end": end})
    return ranges


def render_ranges(ranges: list[dict[str, int]]) -> str:
    if not ranges:
        return "-"
    labels: list[str] = []
    for item in ranges:
        start = int(item["start"])
        end = int(item["end"])
        labels.append(f"{start}" if start == end else f"{start}-{end}")
    return ", ".join(labels)


@dataclass(slots=True)
class EvidenceUnit:
    evidence_id: str
    kind: str
    document_id: str | None
    file_path: str
    unit_no: int | None
    source_locator: str | None
    heading: str | None
    text: str
    snippet: str
    score: float | None = None
    cited: bool = False
    promoted: bool = False
    last_used_turn: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "document_id": self.document_id,
            "file_path": self.file_path,
            "unit_no": self.unit_no,
            "source_locator": self.source_locator,
            "heading": self.heading,
            "snippet": self.snippet,
            "score": self.score,
            "cited": self.cited,
            "promoted": self.promoted,
            "last_used_turn": self.last_used_turn,
        }


@dataclass(slots=True)
class DocumentCoverage:
    document_id: str
    file_path: str
    label: str
    total_units: int | None = None
    requested_units: set[int] = field(default_factory=set)
    retrieved_units: set[int] = field(default_factory=set)
    active_ranges: list[dict[str, int]] = field(default_factory=list)
    summarized_ranges: list[dict[str, int]] = field(default_factory=list)
    last_anchor: int | None = None

    def note_requested_range(self, start: int, end: int) -> None:
        lo = max(min(int(start), int(end)), 1)
        hi = max(int(start), int(end))
        self.requested_units.update(range(lo, hi + 1))

    def note_retrieved_units(self, unit_nos: list[int]) -> set[int]:
        cleaned = {int(unit_no) for unit_no in unit_nos if int(unit_no) > 0}
        new_units = cleaned - self.retrieved_units
        self.retrieved_units.update(cleaned)
        return new_units

    def coverage_ranges(self) -> list[dict[str, int]]:
        return compress_unit_ranges(self.retrieved_units)

    def missing_ranges(self) -> list[dict[str, int]]:
        if not self.requested_units:
            return []
        if self.total_units is not None:
            requested = {
                unit_no for unit_no in self.requested_units if unit_no <= int(self.total_units)
            }
        else:
            requested = set(self.requested_units)
        missing = requested - self.retrieved_units
        return compress_unit_ranges(missing)

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "file_path": self.file_path,
            "label": self.label,
            "total_units": self.total_units,
            "retrieved_ranges": self.coverage_ranges(),
            "active_ranges": list(self.active_ranges),
            "summarized_ranges": list(self.summarized_ranges),
            "missing_ranges": self.missing_ranges(),
            "last_anchor": self.last_anchor,
        }


@dataclass(slots=True)
class ContextPlanResult:
    applied: bool
    operation: str
    payload: dict[str, Any]


@dataclass(slots=True)
class ContextState:
    task: str
    selected_documents: dict[str, dict[str, str]] = field(default_factory=dict)
    collection_name: str | None = None
    evidence_units: dict[str, EvidenceUnit] = field(default_factory=dict)
    coverage_by_document: dict[str, DocumentCoverage] = field(default_factory=dict)
    working_summary: list[str] = field(default_factory=list)
    open_gaps: list[dict[str, Any]] = field(default_factory=list)
    compaction_log: list[dict[str, Any]] = field(default_factory=list)
    promoted_evidence_units: list[str] = field(default_factory=list)
    active_document_id: str | None = None
    active_file_path: str | None = None
    active_ranges: list[dict[str, int]] = field(default_factory=list)
    turn_counter: int = 0
    no_new_coverage_streak: int = 0

    def register_documents(self, documents: list[dict[str, str]]) -> None:
        for item in documents:
            doc_id = str(item["document_id"])
            self.selected_documents[doc_id] = {
                "label": str(item.get("label") or doc_id),
                "file_path": str(item.get("file_path") or ""),
            }
            self._ensure_coverage(
                document_id=doc_id,
                file_path=str(item.get("file_path") or ""),
                label=str(item.get("label") or doc_id),
            )

    def _ensure_coverage(
        self,
        *,
        document_id: str,
        file_path: str,
        label: str | None = None,
    ) -> DocumentCoverage:
        coverage = self.coverage_by_document.get(document_id)
        if coverage is None:
            coverage = DocumentCoverage(
                document_id=document_id,
                file_path=file_path,
                label=label or file_path or document_id,
            )
            self.coverage_by_document[document_id] = coverage
        else:
            if file_path:
                coverage.file_path = file_path
            if label:
                coverage.label = label
        return coverage

    def _bump_turn(self) -> int:
        self.turn_counter += 1
        return self.turn_counter

    def set_active_scope(
        self,
        *,
        document_id: str | None,
        file_path: str | None,
        ranges: list[dict[str, int]] | None,
    ) -> dict[str, Any]:
        self.active_document_id = document_id
        self.active_file_path = file_path
        self.active_ranges = list(ranges or [])
        if document_id and document_id in self.coverage_by_document:
            self.coverage_by_document[document_id].active_ranges = list(self.active_ranges)
        return {
            "active_document_id": self.active_document_id,
            "active_file_path": self.active_file_path,
            "active_ranges": list(self.active_ranges),
        }

    def ingest_parse_result(
        self,
        *,
        document_id: str,
        file_path: str,
        label: str,
        units: list[dict[str, Any]],
        total_units: int | None,
        focus_hint: str | None,
        anchor: int | None,
        window: int,
        max_units: int | None,
    ) -> dict[str, Any]:
        turn = self._bump_turn()
        coverage = self._ensure_coverage(
            document_id=document_id,
            file_path=file_path,
            label=label,
        )
        if total_units is not None:
            coverage.total_units = int(total_units)

        if anchor is not None:
            coverage.last_anchor = int(anchor)
            coverage.note_requested_range(anchor - int(window), anchor + int(window))

        returned_unit_nos = [int(unit["unit_no"]) for unit in units if unit.get("unit_no") is not None]
        new_units = coverage.note_retrieved_units(returned_unit_nos)
        returned_ranges = compress_unit_ranges(returned_unit_nos)
        coverage.active_ranges = list(returned_ranges)
        self.set_active_scope(
            document_id=document_id,
            file_path=file_path,
            ranges=returned_ranges,
        )

        top_snippets: list[str] = []
        for unit in units:
            unit_no = int(unit["unit_no"])
            evidence_id = f"{document_id}:unit:{unit_no}"
            snippet = _snippet(str(unit.get("markdown") or ""))
            self.evidence_units[evidence_id] = EvidenceUnit(
                evidence_id=evidence_id,
                kind="parsed_unit",
                document_id=document_id,
                file_path=file_path,
                unit_no=unit_no,
                source_locator=str(unit.get("source_locator") or f"unit-{unit_no}"),
                heading=str(unit.get("heading") or "") or None,
                text=str(unit.get("markdown") or ""),
                snippet=snippet,
                last_used_turn=turn,
            )
            if len(top_snippets) < 4 and snippet:
                heading = str(unit.get("heading") or "") or f"UNIT {unit_no}"
                top_snippets.append(f"[UNIT {unit_no}] {heading}: {snippet}")

        if new_units:
            self.no_new_coverage_streak = 0
        else:
            self.no_new_coverage_streak += 1

        missing_ranges = coverage.missing_ranges()
        self.open_gaps = [
            gap
            for gap in self.open_gaps
            if not (
                gap.get("document_id") == document_id
                and gap.get("kind") in {"coverage_gap", "no_new_coverage"}
            )
        ]
        if missing_ranges:
            self.open_gaps.append(
                {
                    "kind": "coverage_gap",
                    "document_id": document_id,
                    "file_path": file_path,
                    "ranges": missing_ranges,
                    "message": (
                        f"Requested coverage has missing units: {render_ranges(missing_ranges)}."
                    ),
                }
            )
        if self.no_new_coverage_streak >= 2:
            stale_ranges = list(coverage.active_ranges)
            coverage.summarized_ranges = stale_ranges
            coverage.active_ranges = []
            self.set_active_scope(
                document_id=document_id,
                file_path=file_path,
                ranges=[],
            )
            self.open_gaps.append(
                {
                    "kind": "no_new_coverage",
                    "document_id": document_id,
                    "file_path": file_path,
                    "stale_ranges": stale_ranges,
                    "message": (
                        "The last two evidence reads added no new coverage. "
                        "The current active range should be treated as stale. "
                        "Run a new search, switch anchors, or parse a different range."
                    ),
                }
            )

        summary = (
            f"Focused parse for {label}: units {render_ranges(returned_ranges)}"
            f"; new units={len(new_units)}; total_units="
            f"{total_units if total_units is not None else '?'}."
        )
        if focus_hint:
            summary += f" Focus hint={focus_hint!r}."
        self.working_summary.append(summary)
        self.working_summary = self.working_summary[-8:]

        return {
            "document_id": document_id,
            "file_path": file_path,
            "anchor": anchor,
            "window": window,
            "max_units": max_units,
            "focus_hint": focus_hint,
            "returned_unit_nos": returned_unit_nos,
            "returned_ranges": returned_ranges,
            "total_units": total_units,
            "new_units_added": len(new_units),
            "top_snippets": top_snippets,
            "coverage_gap": missing_ranges,
            "source_truncated": False,
            "summary_for_model": summary,
        }

    def ingest_search_results(
        self,
        *,
        query: str,
        filters: str | None,
        hits: list[dict[str, Any]],
        limit: int,
    ) -> dict[str, Any]:
        turn = self._bump_turn()
        top_hits: list[dict[str, Any]] = []
        for index, hit in enumerate(hits[:limit], start=1):
            doc_id = str(hit["doc_id"])
            file_path = str(hit["absolute_path"])
            label = self.selected_documents.get(doc_id, {}).get("label") or file_path or doc_id
            coverage = self._ensure_coverage(
                document_id=doc_id,
                file_path=file_path,
                label=label,
            )
            source_unit_no = hit.get("source_unit_no")
            if source_unit_no is not None:
                coverage.note_retrieved_units([int(source_unit_no)])
            evidence_id = (
                f"{doc_id}:search:{int(source_unit_no)}"
                if source_unit_no is not None
                else f"{doc_id}:search:{index}"
            )
            self.evidence_units[evidence_id] = EvidenceUnit(
                evidence_id=evidence_id,
                kind="search_hit",
                document_id=doc_id,
                file_path=file_path,
                unit_no=int(source_unit_no) if source_unit_no is not None else None,
                source_locator=None,
                heading=None,
                text=str(hit.get("text") or ""),
                snippet=_snippet(str(hit.get("text") or "")),
                score=float(hit.get("score")) if hit.get("score") is not None else None,
                last_used_turn=turn,
            )
            top_hits.append(
                {
                    "doc_id": doc_id,
                    "file_path": file_path,
                    "source_unit_no": source_unit_no,
                    "score": hit.get("score"),
                    "snippet": _snippet(str(hit.get("text") or "")),
                }
            )

        if top_hits:
            first_hit = top_hits[0]
            ranges = (
                [{"start": int(first_hit["source_unit_no"]), "end": int(first_hit["source_unit_no"])}]
                if first_hit.get("source_unit_no") is not None
                else []
            )
            self.set_active_scope(
                document_id=str(first_hit["doc_id"]),
                file_path=str(first_hit["file_path"]),
                ranges=ranges,
            )
            self.no_new_coverage_streak = 0
        else:
            self.no_new_coverage_streak += 1

        summary = f"Search for {query!r} returned {len(top_hits)} hits."
        if filters:
            summary += f" Filters={filters!r}."
        self.working_summary.append(summary)
        self.working_summary = self.working_summary[-8:]
        return {
            "query": query,
            "filters": filters,
            "limit": limit,
            "hit_count": len(hits),
            "top_hits": top_hits,
            "summary_for_model": summary,
        }

    def ingest_document_read(
        self,
        *,
        document_id: str,
        file_path: str,
        label: str,
        body: str,
    ) -> dict[str, Any]:
        turn = self._bump_turn()
        self._ensure_coverage(
            document_id=document_id,
            file_path=file_path,
            label=label,
        )
        evidence_id = f"{document_id}:document"
        snippet = _snippet(body, limit=320)
        self.evidence_units[evidence_id] = EvidenceUnit(
            evidence_id=evidence_id,
            kind="document_body",
            document_id=document_id,
            file_path=file_path,
            unit_no=None,
            source_locator=None,
            heading=label,
            text=body,
            snippet=snippet,
            last_used_turn=turn,
        )
        self.set_active_scope(document_id=document_id, file_path=file_path, ranges=[])
        summary = f"Read indexed document {label}. Stored a condensed document-level excerpt."
        self.working_summary.append(summary)
        self.working_summary = self.working_summary[-8:]
        return {
            "document_id": document_id,
            "file_path": file_path,
            "summary_for_model": summary,
            "snippet": snippet,
        }

    def apply_context_plan(self, plan: dict[str, Any] | None) -> ContextPlanResult | None:
        if not isinstance(plan, dict):
            return None
        operation = str(plan.get("operation") or "").strip()
        if not operation:
            return None

        document_id = str(plan.get("document_id") or "").strip() or None
        file_path = str(plan.get("file_path") or "").strip() or None
        ranges = [
            {
                "start": int(item["start_unit"]),
                "end": int(item.get("end_unit", item["start_unit"])),
            }
            for item in plan.get("ranges", [])
            if isinstance(item, dict) and item.get("start_unit") is not None
        ]

        if document_id is not None and document_id not in self.selected_documents:
            return ContextPlanResult(
                applied=False,
                operation=operation,
                payload={"error": "document is outside the selected scope"},
            )

        if operation in {"keep_ranges", "expand_range", "narrow_to_range", "switch_document"}:
            payload = self.set_active_scope(
                document_id=document_id or self.active_document_id,
                file_path=file_path or self.active_file_path,
                ranges=ranges or self.active_ranges,
            )
            payload["operation"] = operation
            return ContextPlanResult(applied=True, operation=operation, payload=payload)

        if operation == "promote_evidence_units":
            evidence_ids = [
                str(item)
                for item in plan.get("evidence_ids", [])
                if isinstance(item, str) and item in self.evidence_units
            ]
            for evidence_id in evidence_ids:
                self.evidence_units[evidence_id].promoted = True
                if evidence_id not in self.promoted_evidence_units:
                    self.promoted_evidence_units.append(evidence_id)
            payload = {
                "operation": operation,
                "promoted_evidence_units": list(self.promoted_evidence_units),
            }
            return ContextPlanResult(applied=True, operation=operation, payload=payload)

        if operation == "summarize_stale_ranges":
            doc_id = document_id or self.active_document_id
            if doc_id is None or doc_id not in self.coverage_by_document:
                return ContextPlanResult(
                    applied=False,
                    operation=operation,
                    payload={"error": "no document available to summarize"},
                )
            coverage = self.coverage_by_document[doc_id]
            stale_ranges = [
                item
                for item in coverage.coverage_ranges()
                if item not in coverage.active_ranges
            ]
            coverage.summarized_ranges = stale_ranges
            payload = {
                "operation": operation,
                "document_id": doc_id,
                "summarized_ranges": stale_ranges,
            }
            return ContextPlanResult(applied=True, operation=operation, payload=payload)

        if operation == "stop_insufficient_evidence":
            note = str(plan.get("note") or "").strip() or (
                "Evidence is still insufficient after automatic context compaction."
            )
            self.open_gaps.append({"kind": "insufficient_evidence", "message": note})
            return ContextPlanResult(
                applied=True,
                operation=operation,
                payload={"operation": operation, "message": note},
            )

        return ContextPlanResult(
            applied=False,
            operation=operation,
            payload={"error": "unsupported context operation"},
        )

    def mark_cited_sources(self, cited_sources: list[str]) -> None:
        if not cited_sources:
            return
        for evidence in self.evidence_units.values():
            source = evidence.file_path
            locator = evidence.source_locator or ""
            citation_label = f"{source}#{locator}" if locator else source
            if citation_label in cited_sources or source in cited_sources:
                evidence.cited = True

    def build_context_pack(
        self,
        *,
        anchor_unit_no: int | None = None,
        max_chars: int = 7000,
    ) -> tuple[str, dict[str, Any]]:
        compaction_actions: list[dict[str, Any]] = []
        coverage_payload = {
            doc_id: coverage.as_dict()
            for doc_id, coverage in self.coverage_by_document.items()
        }

        def evidence_priority(item: EvidenceUnit) -> tuple[int, int, float, int]:
            citation_rank = 0 if item.cited else 1
            promotion_rank = 0 if item.promoted else 1
            if anchor_unit_no is not None and item.unit_no is not None:
                distance = abs(int(item.unit_no) - int(anchor_unit_no))
            else:
                distance = 9999
            score_rank = -float(item.score or 0.0)
            return (citation_rank, promotion_rank, distance, int(score_rank))

        active_document_id = self.active_document_id
        active_ranges = list(self.active_ranges)
        coverage_label_lines = []
        for coverage in self.coverage_by_document.values():
            coverage_label_lines.append(
                f"- {coverage.label}: coverage={render_ranges(coverage.coverage_ranges())}; "
                f"active={render_ranges(coverage.active_ranges)}; "
                f"missing={render_ranges(coverage.missing_ranges())}; "
                f"total={coverage.total_units if coverage.total_units is not None else '?'}"
            )

        active_evidence = [
            evidence
            for evidence in self.evidence_units.values()
            if (
                active_document_id is not None
                and evidence.document_id == active_document_id
                and evidence.unit_no is not None
                and any(
                    int(rng["start"]) <= int(evidence.unit_no) <= int(rng["end"])
                    for rng in active_ranges
                )
            )
        ]
        active_evidence.sort(key=evidence_priority)

        supporting_evidence = [
            evidence
            for evidence in self.evidence_units.values()
            if evidence not in active_evidence
        ]
        supporting_evidence.sort(key=evidence_priority)

        sections: list[str] = [
            "=== STRUCTURED CONTEXT PACK ===",
            f"Task: {self.task}",
            "",
            "Selected scope:",
        ]
        if self.selected_documents:
            sections.extend(
                [
                    f"- {item['label']} ({doc_id})"
                    for doc_id, item in self.selected_documents.items()
                ]
            )
        else:
            sections.append("- (none)")
        sections.extend(["", "Coverage by document:"])
        sections.extend(coverage_label_lines or ["- (no coverage yet)"])

        if self.working_summary:
            sections.extend(["", "Working summary:"])
            sections.extend(f"- {item}" for item in self.working_summary[-5:])

        if active_evidence:
            sections.extend(["", "Active evidence window (raw excerpts):"])
            for evidence in active_evidence[:8]:
                label = evidence.heading or evidence.source_locator or evidence.evidence_id
                prefix = f"[UNIT {evidence.unit_no}] " if evidence.unit_no is not None else ""
                sections.append(f"{prefix}{label}")
                sections.append(evidence.text)
                sections.append("")

        if supporting_evidence:
            sections.extend(["", "Supporting evidence snippets:"])
            for evidence in supporting_evidence[:8]:
                label = evidence.heading or evidence.source_locator or evidence.evidence_id
                prefix = f"[UNIT {evidence.unit_no}] " if evidence.unit_no is not None else ""
                sections.append(f"- {prefix}{label}: {evidence.snippet}")

        if self.open_gaps:
            sections.extend(["", "Open gaps:"])
            for gap in self.open_gaps[-4:]:
                if gap.get("ranges"):
                    sections.append(
                        f"- {gap.get('message')} ranges={render_ranges(list(gap['ranges']))}"
                    )
                elif gap.get("stale_ranges"):
                    sections.append(
                        f"- {gap.get('message')} stale_ranges="
                        f"{render_ranges(list(gap['stale_ranges']))}"
                    )
                else:
                    sections.append(f"- {gap.get('message')}")

        text = "\n".join(section for section in sections if section is not None).strip()
        if len(text) > max_chars:
            compaction_actions.append(
                {
                    "action": "summarize_stale_ranges",
                    "reason": "context pack exceeded max_chars; using summaries for stale ranges",
                }
            )
            stale_labels: list[str] = []
            for coverage in self.coverage_by_document.values():
                stale_ranges = [
                    item
                    for item in coverage.coverage_ranges()
                    if item not in coverage.active_ranges
                ]
                if stale_ranges:
                    coverage.summarized_ranges = stale_ranges
                    stale_labels.append(
                        f"- {coverage.label}: summarized={render_ranges(stale_ranges)}"
                    )
            compact_sections = sections[:]
            if stale_labels:
                compact_sections.extend(["", "Summarized stale ranges:"])
                compact_sections.extend(stale_labels)
            compact_text = "\n".join(compact_sections).strip()
            if len(compact_text) <= max_chars:
                text = compact_text
            else:
                compaction_actions.append(
                    {
                        "action": "drop_supporting_snippets",
                        "reason": "supporting snippets dropped to respect hard context pack limit",
                    }
                )
                compact_sections = [
                    section
                    for section in compact_sections
                    if not section.startswith("- [UNIT") or len(section) < 320
                ]
                text = "\n".join(compact_sections).strip()
                if len(text) > max_chars:
                    text = text[: max_chars - 18].rstrip() + "\n...[truncated]..."

        self.compaction_log.extend(compaction_actions)
        self.compaction_log = self.compaction_log[-10:]
        return text, {
            "context_scope": {
                "active_document_id": self.active_document_id,
                "active_file_path": self.active_file_path,
                "active_ranges": list(self.active_ranges),
            },
            "coverage_by_document": coverage_payload,
            "compaction_actions": compaction_actions,
            "active_ranges": list(self.active_ranges),
            "promoted_evidence_units": list(self.promoted_evidence_units),
            "open_gaps": list(self.open_gaps[-4:]),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "context_scope": {
                "active_document_id": self.active_document_id,
                "active_file_path": self.active_file_path,
                "active_ranges": list(self.active_ranges),
            },
            "coverage_by_document": {
                doc_id: coverage.as_dict()
                for doc_id, coverage in self.coverage_by_document.items()
            },
            "working_summary": list(self.working_summary),
            "open_gaps": list(self.open_gaps),
            "compaction_actions": list(self.compaction_log),
            "promoted_evidence_units": list(self.promoted_evidence_units),
            "evidence_units": [
                evidence.as_dict()
                for evidence in sorted(
                    self.evidence_units.values(),
                    key=lambda item: (item.document_id or "", item.unit_no or 0, item.evidence_id),
                )
            ],
        }
