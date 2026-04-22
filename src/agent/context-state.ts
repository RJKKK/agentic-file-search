/*
Reference: legacy/python/src/fs_explorer/context_state.py
Reference: legacy/python/src/fs_explorer/agent.py
*/

import { snippet } from "../runtime/fs-utils.js";

export interface RegisteredDocument {
  documentId: string;
  label: string;
  filePath: string;
}

export interface EvidenceUnit {
  evidenceId: string;
  kind: string;
  documentId: string | null;
  filePath: string;
  unitNo: number | null;
  sourceLocator: string | null;
  heading: string | null;
  text: string;
  snippet: string;
  score: number | null;
  cited: boolean;
  promoted: boolean;
  lastUsedTurn: number;
}

interface DocumentCoverage {
  documentId: string;
  filePath: string;
  label: string;
  totalUnits: number | null;
  requestedUnits: Set<number>;
  retrievedUnits: Set<number>;
  activeRanges: Array<{ start: number; end: number }>;
  summarizedRanges: Array<{ start: number; end: number }>;
  lastAnchor: number | null;
}

export interface ContextPlanResult {
  applied: boolean;
  operation: string;
  payload: Record<string, unknown>;
}

export interface ContextStateSnapshot {
  context_scope: {
    active_document_id: string | null;
    active_file_path: string | null;
    active_ranges: Array<{ start: number; end: number }>;
  };
  coverage_by_document: Record<string, Record<string, unknown>>;
  working_summary: string[];
  open_gaps: Array<Record<string, unknown>>;
  compaction_actions: Array<Record<string, unknown>>;
  promoted_evidence_units: string[];
  evidence_units: Array<Record<string, unknown>>;
}

export function compressUnitRanges(unitNos: Iterable<number>): Array<{ start: number; end: number }> {
  const ordered = [...new Set([...unitNos].map((unitNo) => Number(unitNo)).filter((unitNo) => unitNo > 0))]
    .sort((left, right) => left - right);
  if (ordered.length === 0) {
    return [];
  }
  const ranges: Array<{ start: number; end: number }> = [];
  let start = ordered[0];
  let end = ordered[0];
  for (const unitNo of ordered.slice(1)) {
    if (unitNo === end + 1) {
      end = unitNo;
      continue;
    }
    ranges.push({ start, end });
    start = unitNo;
    end = unitNo;
  }
  ranges.push({ start, end });
  return ranges;
}

export function renderRanges(ranges: Array<{ start: number; end: number }>): string {
  if (ranges.length === 0) {
    return "-";
  }
  return ranges
    .map((item) => (item.start === item.end ? `${item.start}` : `${item.start}-${item.end}`))
    .join(", ");
}

function pythonLikeRepr(value: string): string {
  return `'${String(value).replace(/\\/g, "\\\\").replace(/'/g, "\\'")}'`;
}

export class ContextState {
  private task = "";

  private readonly selectedDocuments = new Map<string, { label: string; filePath: string }>();

  private readonly evidenceUnits = new Map<string, EvidenceUnit>();

  private readonly coverageByDocument = new Map<string, DocumentCoverage>();

  private readonly workingSummary: string[] = [];

  private readonly openGaps: Array<Record<string, unknown>> = [];

  private readonly compactionLog: Array<Record<string, unknown>> = [];

  private readonly promotedEvidenceUnits: string[] = [];

  private activeDocumentId: string | null = null;

  private activeFilePath: string | null = null;

  private activeRanges: Array<{ start: number; end: number }> = [];

  private turnCounter = 0;

  private noNewCoverageStreak = 0;

  setTask(task: string): void {
    this.task = String(task ?? "");
  }

  registerDocuments(documents: RegisteredDocument[]): void {
    for (const item of documents) {
      this.selectedDocuments.set(item.documentId, {
        label: item.label || item.documentId,
        filePath: item.filePath || "",
      });
      this.ensureCoverage(item.documentId, item.filePath || "", item.label || item.documentId);
    }
  }

  setActiveScope(input: {
    documentId?: string | null;
    filePath?: string | null;
    ranges?: Array<{ start: number; end: number }>;
  }): Record<string, unknown> {
    this.activeDocumentId = input.documentId ?? null;
    this.activeFilePath = input.filePath ?? null;
    this.activeRanges = [...(input.ranges ?? [])];
    if (this.activeDocumentId && this.coverageByDocument.has(this.activeDocumentId)) {
      this.coverageByDocument.get(this.activeDocumentId)!.activeRanges = [...this.activeRanges];
    }
    return {
      active_document_id: this.activeDocumentId,
      active_file_path: this.activeFilePath,
      active_ranges: [...this.activeRanges],
    };
  }

  ingestParseResult(input: {
    documentId: string;
    filePath: string;
    label: string;
    units: Array<{
      unit_no?: number | null;
      source_locator?: string | null;
      heading?: string | null;
      markdown?: string | null;
    }>;
    totalUnits?: number | null;
    focusHint?: string | null;
    anchor?: number | null;
    window: number;
    maxUnits?: number | null;
  }): Record<string, unknown> {
    const turn = this.bumpTurn();
    const coverage = this.ensureCoverage(input.documentId, input.filePath, input.label);
    if (input.totalUnits != null) {
      coverage.totalUnits = Number(input.totalUnits);
    }

    if (input.anchor != null) {
      coverage.lastAnchor = Number(input.anchor);
      this.noteRequestedRange(
        coverage,
        Number(input.anchor) - Number(input.window),
        Number(input.anchor) + Number(input.window),
      );
    }

    const returnedUnitNos = input.units
      .map((unit) => unit.unit_no)
      .filter((unitNo): unitNo is number => unitNo != null)
      .map((unitNo) => Number(unitNo));

    const newUnits = this.noteRetrievedUnits(coverage, returnedUnitNos);
    const returnedRanges = compressUnitRanges(returnedUnitNos);
    coverage.activeRanges = [...returnedRanges];
    this.setActiveScope({
      documentId: input.documentId,
      filePath: input.filePath,
      ranges: returnedRanges,
    });

    const topSnippets: string[] = [];
    for (const unit of input.units) {
      const unitNo = unit.unit_no != null ? Number(unit.unit_no) : null;
      if (unitNo == null) {
        continue;
      }
      const evidenceId = `${input.documentId}:unit:${unitNo}`;
      const excerpt = snippet(String(unit.markdown ?? ""));
      this.evidenceUnits.set(evidenceId, {
        evidenceId,
        kind: "parsed_unit",
        documentId: input.documentId,
        filePath: input.filePath,
        unitNo,
        sourceLocator: String(unit.source_locator ?? `unit-${unitNo}`),
        heading: String(unit.heading ?? "") || null,
        text: String(unit.markdown ?? ""),
        snippet: excerpt,
        score: null,
        cited: false,
        promoted: false,
        lastUsedTurn: turn,
      });
      if (topSnippets.length < 4 && excerpt) {
        const heading = String(unit.heading ?? "") || `UNIT ${unitNo}`;
        topSnippets.push(`[UNIT ${unitNo}] ${heading}: ${excerpt}`);
      }
    }

    if (newUnits.size > 0) {
      this.noNewCoverageStreak = 0;
    } else {
      this.noNewCoverageStreak += 1;
    }

    const missingRanges = this.missingRanges(coverage);
    this.removeOpenGap(input.documentId, ["coverage_gap", "no_new_coverage"]);
    if (missingRanges.length > 0) {
      this.openGaps.push({
        kind: "coverage_gap",
        document_id: input.documentId,
        file_path: input.filePath,
        ranges: missingRanges,
        message: `Requested coverage has missing units: ${renderRanges(missingRanges)}.`,
      });
    }

    if (this.noNewCoverageStreak >= 2) {
      const staleRanges = [...coverage.activeRanges];
      coverage.summarizedRanges = staleRanges;
      coverage.activeRanges = [];
      this.setActiveScope({
        documentId: input.documentId,
        filePath: input.filePath,
        ranges: [],
      });
      this.openGaps.push({
        kind: "no_new_coverage",
        document_id: input.documentId,
        file_path: input.filePath,
        stale_ranges: staleRanges,
        message:
          "The last two evidence reads added no new coverage. The current active range should be treated as stale. Run a new search, switch anchors, or parse a different range.",
      });
    }

    let summary = `Focused parse for ${input.label}: units ${renderRanges(returnedRanges)}; new units=${newUnits.size}; total_units=${input.totalUnits ?? "?"}.`;
    if (input.focusHint) {
      summary += ` Focus hint=${pythonLikeRepr(String(input.focusHint))}.`;
    }
    this.workingSummary.push(summary);
    this.trimSummary();

    return {
      document_id: input.documentId,
      file_path: input.filePath,
      anchor: input.anchor ?? null,
      window: input.window,
      max_units: input.maxUnits ?? null,
      focus_hint: input.focusHint ?? null,
      returned_unit_nos: returnedUnitNos,
      returned_ranges: returnedRanges,
      total_units: input.totalUnits ?? null,
      new_units_added: newUnits.size,
      top_snippets: topSnippets,
      coverage_gap: missingRanges,
      source_truncated: false,
      summary_for_model: summary,
    };
  }

  ingestSearchResults(input: {
    query: string;
    filters?: string | null;
    hits: Array<{
      doc_id: string;
      absolute_path: string;
      source_unit_no?: number | null;
      score?: number | null;
      text?: string | null;
    }>;
    limit: number;
  }): Record<string, unknown> {
    const turn = this.bumpTurn();
    const topHits: Array<Record<string, unknown>> = [];
    for (const [index, hit] of input.hits.slice(0, input.limit).entries()) {
      const docId = String(hit.doc_id);
      const filePath = String(hit.absolute_path);
      const label =
        this.selectedDocuments.get(docId)?.label ??
        filePath.split(/[\\/]/).at(-1) ??
        filePath;
      const coverage = this.ensureCoverage(docId, filePath, label);
      if (hit.source_unit_no != null) {
        this.noteRetrievedUnits(coverage, [Number(hit.source_unit_no)]);
      }
      const evidenceId =
        hit.source_unit_no != null ? `${docId}:search:${Number(hit.source_unit_no)}` : `${docId}:search:${index + 1}`;
      const text = String(hit.text ?? "");
      this.evidenceUnits.set(evidenceId, {
        evidenceId,
        kind: "search_hit",
        documentId: docId,
        filePath,
        unitNo: hit.source_unit_no != null ? Number(hit.source_unit_no) : null,
        sourceLocator: null,
        heading: null,
        text,
        snippet: snippet(text),
        score: hit.score != null ? Number(hit.score) : null,
        cited: false,
        promoted: false,
        lastUsedTurn: turn,
      });
      topHits.push({
        doc_id: docId,
        file_path: filePath,
        source_unit_no: hit.source_unit_no ?? null,
        score: hit.score ?? null,
        snippet: snippet(text),
      });
    }

    if (topHits.length > 0) {
      const firstHit = topHits[0];
      const ranges =
        firstHit.source_unit_no != null
          ? [{ start: Number(firstHit.source_unit_no), end: Number(firstHit.source_unit_no) }]
          : [];
      this.setActiveScope({
        documentId: String(firstHit.doc_id),
        filePath: String(firstHit.file_path),
        ranges,
      });
      this.noNewCoverageStreak = 0;
    } else {
      this.noNewCoverageStreak += 1;
    }

    let summary = `Search for ${pythonLikeRepr(input.query)} returned ${topHits.length} hits.`;
    if (input.filters) {
      summary += ` Filters=${pythonLikeRepr(String(input.filters))}.`;
    }
    this.workingSummary.push(summary);
    this.trimSummary();
    return {
      query: input.query,
      filters: input.filters ?? null,
      limit: input.limit,
      hit_count: input.hits.length,
      top_hits: topHits,
      summary_for_model: summary,
    };
  }

  ingestPageRead(input: {
    documentId?: string | null;
    filePath: string;
    content: string;
    pageNo?: number | null;
    heading?: string | null;
    sourceLocator?: string | null;
    score?: number | null;
  }): Record<string, unknown> {
    const documentId =
      input.documentId ??
      String(input.filePath.split(/[\\/]/).slice(0, -1).join("/") || input.filePath);
    const label =
      this.selectedDocuments.get(documentId)?.label ??
      input.heading ??
      input.filePath.split(/[\\/]/).at(-1) ??
      input.filePath;
    return this.ingestParseResult({
      documentId,
      filePath: input.filePath,
      label,
      units: [
        {
          unit_no: input.pageNo ?? null,
          source_locator: input.sourceLocator ?? null,
          heading: input.heading ?? null,
          markdown: input.content,
        },
      ],
      totalUnits: null,
      focusHint: null,
      anchor: input.pageNo ?? null,
      window: 0,
      maxUnits: 1,
    });
  }

  ingestDocumentRead(input: {
    documentId: string;
    filePath: string;
    label: string;
    body: string;
  }): Record<string, unknown> {
    const turn = this.bumpTurn();
    this.ensureCoverage(input.documentId, input.filePath, input.label);
    const evidenceId = `${input.documentId}:document`;
    const excerpt = snippet(input.body, 320);
    this.evidenceUnits.set(evidenceId, {
      evidenceId,
      kind: "document_body",
      documentId: input.documentId,
      filePath: input.filePath,
      unitNo: null,
      sourceLocator: null,
      heading: input.label,
      text: input.body,
      snippet: excerpt,
      score: null,
      cited: false,
      promoted: false,
      lastUsedTurn: turn,
    });
    this.setActiveScope({ documentId: input.documentId, filePath: input.filePath, ranges: [] });
    const summary = `Read indexed document ${input.label}. Stored a condensed document-level excerpt.`;
    this.workingSummary.push(summary);
    this.trimSummary();
    return {
      document_id: input.documentId,
      file_path: input.filePath,
      summary_for_model: summary,
      snippet: excerpt,
    };
  }

  releaseDocumentEvidence(input: {
    documentId: string;
    summarizedRanges?: Array<{ start: number; end: number }>;
    reason?: string;
  }): Record<string, unknown> {
    const documentId = String(input.documentId || "").trim();
    const coverage = documentId ? this.coverageByDocument.get(documentId) : null;
    if (!documentId || !coverage) {
      return {
        released: false,
        document_id: documentId || null,
        reason: "document has no tracked coverage",
      };
    }

    const summarizedRanges =
      input.summarizedRanges && input.summarizedRanges.length
        ? input.summarizedRanges
        : coverage.activeRanges.length
          ? [...coverage.activeRanges]
          : this.coverageRanges(coverage);
    const existingSummarized = coverage.summarizedRanges.flatMap((range) => {
      const units: number[] = [];
      for (let unitNo = range.start; unitNo <= range.end; unitNo += 1) {
        units.push(unitNo);
      }
      return units;
    });
    const newSummarized = summarizedRanges.flatMap((range) => {
      const units: number[] = [];
      for (let unitNo = range.start; unitNo <= range.end; unitNo += 1) {
        units.push(unitNo);
      }
      return units;
    });
    coverage.summarizedRanges = compressUnitRanges([...existingSummarized, ...newSummarized]);
    coverage.activeRanges = [];

    let releasedUnits = 0;
    for (const evidence of this.evidenceUnits.values()) {
      if (evidence.documentId !== documentId || !evidence.text) {
        continue;
      }
      if (evidence.kind === "parsed_unit" || evidence.kind === "document_body") {
        evidence.text = "";
        evidence.promoted = evidence.cited;
        releasedUnits += 1;
      }
    }

    if (this.activeDocumentId === documentId) {
      this.setActiveScope({
        documentId,
        filePath: coverage.filePath,
        ranges: [],
      });
    }

    const summary =
      `Released raw evidence for ${coverage.label}: summarized=${renderRanges(coverage.summarizedRanges)}; ` +
      `snippets and citations remain available.`;
    this.workingSummary.push(summary);
    this.trimSummary();
    this.compactionLog.push({
      action: "release_batch_raw_context",
      document_id: documentId,
      summarized_ranges: [...coverage.summarizedRanges],
      released_units: releasedUnits,
      reason: input.reason ?? "batch completed",
    });
    while (this.compactionLog.length > 10) {
      this.compactionLog.shift();
    }

    return {
      released: true,
      document_id: documentId,
      file_path: coverage.filePath,
      summarized_ranges: [...coverage.summarizedRanges],
      active_ranges: [],
      released_units: releasedUnits,
      coverage: this.coverageAsDict(coverage),
      summary_for_model: summary,
    };
  }

  applyContextPlan(plan: Record<string, unknown> | null | undefined): ContextPlanResult | null {
    if (!plan || typeof plan !== "object" || Array.isArray(plan)) {
      return null;
    }
    const operation = String(plan.operation ?? "").trim();
    if (!operation) {
      return null;
    }
    const documentId = String(plan.document_id ?? "").trim() || null;
    const filePath = String(plan.file_path ?? "").trim() || null;
    const ranges = Array.isArray(plan.ranges)
      ? plan.ranges
          .filter((item): item is Record<string, unknown> => !!item && typeof item === "object")
          .filter((item) => item.start_unit != null)
          .map((item) => ({
            start: Number(item.start_unit),
            end: Number(item.end_unit ?? item.start_unit),
          }))
      : [];

    if (documentId && !this.selectedDocuments.has(documentId)) {
      return {
        applied: false,
        operation,
        payload: { error: "document is outside the selected scope" },
      };
    }

    if (["keep_ranges", "expand_range", "narrow_to_range", "switch_document"].includes(operation)) {
      const payload = this.setActiveScope({
        documentId: documentId ?? this.activeDocumentId,
        filePath: filePath ?? this.activeFilePath,
        ranges: ranges.length ? ranges : this.activeRanges,
      });
      return { applied: true, operation, payload };
    }

    if (operation === "promote_evidence_units") {
      const evidenceIds = Array.isArray(plan.evidence_ids) ? plan.evidence_ids.map(String) : [];
      for (const evidenceId of evidenceIds) {
        const evidence = this.evidenceUnits.get(evidenceId);
        if (!evidence) {
          continue;
        }
        evidence.promoted = true;
        if (!this.promotedEvidenceUnits.includes(evidenceId)) {
          this.promotedEvidenceUnits.push(evidenceId);
        }
      }
      return {
        applied: true,
        operation,
        payload: {
          operation,
          promoted_evidence_units: [...this.promotedEvidenceUnits],
        },
      };
    }

    if (operation === "summarize_stale_ranges") {
      const docId = documentId ?? this.activeDocumentId;
      if (!docId || !this.coverageByDocument.has(docId)) {
        return {
          applied: false,
          operation,
          payload: { error: "no document available to summarize" },
        };
      }
      const coverage = this.coverageByDocument.get(docId)!;
      const staleRanges = this.coverageRanges(coverage).filter(
        (item) =>
          !coverage.activeRanges.some(
            (active) => active.start === item.start && active.end === item.end,
          ),
      );
      coverage.summarizedRanges = staleRanges;
      return {
        applied: true,
        operation,
        payload: {
          operation,
          document_id: docId,
          summarized_ranges: staleRanges,
        },
      };
    }

    if (operation === "stop_insufficient_evidence") {
      const note =
        String(plan.note ?? "").trim() ||
        "Evidence is still insufficient after automatic context compaction.";
      this.openGaps.push({ kind: "insufficient_evidence", message: note });
      return {
        applied: true,
        operation,
        payload: { operation, message: note },
      };
    }

    return {
      applied: false,
      operation,
      payload: { error: "unsupported context operation" },
    };
  }

  markCitedSources(citedSources: string[]): void {
    if (!citedSources.length) {
      return;
    }
    for (const evidence of this.evidenceUnits.values()) {
      const source = evidence.filePath;
      const locator = evidence.sourceLocator ?? "";
      const citationLabel = locator ? `${source}#${locator}` : source;
      if (citedSources.includes(citationLabel) || citedSources.includes(source)) {
        evidence.cited = true;
      }
    }
  }

  buildContextPack(maxChars = 7000, anchorUnitNo: number | null = null): { text: string; stats: Record<string, unknown> } {
    const compactionActions: Array<Record<string, unknown>> = [];
    const coveragePayload = Object.fromEntries(
      [...this.coverageByDocument.entries()].map(([docId, coverage]) => [docId, this.coverageAsDict(coverage)]),
    );

    const evidencePriority = (item: EvidenceUnit): [number, number, number, number] => {
      const citationRank = item.cited ? 0 : 1;
      const promotionRank = item.promoted ? 0 : 1;
      const distance =
        anchorUnitNo != null && item.unitNo != null ? Math.abs(item.unitNo - anchorUnitNo) : 9999;
      const scoreRank = -Math.trunc(item.score ?? 0);
      return [citationRank, promotionRank, distance, scoreRank];
    };

    const activeEvidence = [...this.evidenceUnits.values()]
      .filter(
        (item) =>
          this.activeDocumentId != null &&
          item.documentId === this.activeDocumentId &&
          item.unitNo != null &&
          this.activeRanges.some((range) => item.unitNo! >= range.start && item.unitNo! <= range.end),
      )
      .sort((left, right) => {
        const a = evidencePriority(left);
        const b = evidencePriority(right);
        for (let index = 0; index < a.length; index += 1) {
          if (a[index] !== b[index]) {
            return a[index] - b[index];
          }
        }
        return 0;
      });

    const supportingEvidence = [...this.evidenceUnits.values()]
      .filter((item) => !activeEvidence.some((active) => active.evidenceId === item.evidenceId))
      .sort((left, right) => {
        const a = evidencePriority(left);
        const b = evidencePriority(right);
        for (let index = 0; index < a.length; index += 1) {
          if (a[index] !== b[index]) {
            return a[index] - b[index];
          }
        }
        return 0;
      });

    const sections: string[] = [
      "=== STRUCTURED CONTEXT PACK ===",
      `Task: ${this.task || "-"}`,
      "",
      "Selected scope:",
    ];

    if (this.selectedDocuments.size > 0) {
      sections.push(
        ...[...this.selectedDocuments.entries()].map(
          ([docId, item]) => `- ${item.label} (${docId})`,
        ),
      );
    } else {
      sections.push("- (none)");
    }

    const coverageLabelLines = [...this.coverageByDocument.values()].map(
      (coverage) =>
        `- ${coverage.label}: coverage=${renderRanges(this.coverageRanges(coverage))}; active=${renderRanges(
          coverage.activeRanges,
        )}; missing=${renderRanges(this.missingRanges(coverage))}; total=${coverage.totalUnits ?? "?"}`,
    );
    sections.push("", "Coverage by document:", ...(coverageLabelLines.length ? coverageLabelLines : ["- (no coverage yet)"]));

    if (this.workingSummary.length) {
      sections.push("", "Working summary:");
      sections.push(...this.workingSummary.slice(-5).map((item) => `- ${item}`));
    }

    if (activeEvidence.length) {
      sections.push("", "Active evidence window (raw excerpts):");
      for (const evidence of activeEvidence.slice(0, 8)) {
        const label = evidence.heading || evidence.sourceLocator || evidence.evidenceId;
        const prefix = evidence.unitNo != null ? `[UNIT ${evidence.unitNo}] ` : "";
        sections.push(`${prefix}${label}`);
        sections.push(evidence.text);
        sections.push("");
      }
    }

    if (supportingEvidence.length) {
      sections.push("", "Supporting evidence snippets:");
      for (const evidence of supportingEvidence.slice(0, 8)) {
        const label = evidence.heading || evidence.sourceLocator || evidence.evidenceId;
        const prefix = evidence.unitNo != null ? `[UNIT ${evidence.unitNo}] ` : "";
        sections.push(`- ${prefix}${label}: ${evidence.snippet}`);
      }
    }

    if (this.openGaps.length) {
      sections.push("", "Open gaps:");
      for (const gap of this.openGaps.slice(-4)) {
        if (Array.isArray(gap.ranges)) {
          sections.push(`- ${String(gap.message ?? "")} ranges=${renderRanges(gap.ranges as Array<{start:number;end:number}> )}`);
        } else if (Array.isArray(gap.stale_ranges)) {
          sections.push(`- ${String(gap.message ?? "")} stale_ranges=${renderRanges(gap.stale_ranges as Array<{start:number;end:number}> )}`);
        } else {
          sections.push(`- ${String(gap.message ?? "")}`);
        }
      }
    }

    let text = sections.filter((section) => section != null).join("\n").trim();
    if (text.length > maxChars) {
      compactionActions.push({
        action: "summarize_stale_ranges",
        reason: "context pack exceeded max_chars; using summaries for stale ranges",
      });
      const staleLabels: string[] = [];
      for (const coverage of this.coverageByDocument.values()) {
        const staleRanges = this.coverageRanges(coverage).filter(
          (item) =>
            !coverage.activeRanges.some(
              (active) => active.start === item.start && active.end === item.end,
            ),
        );
        if (staleRanges.length > 0) {
          coverage.summarizedRanges = staleRanges;
          staleLabels.push(`- ${coverage.label}: summarized=${renderRanges(staleRanges)}`);
        }
      }
      let compactSections = [...sections];
      if (staleLabels.length > 0) {
        compactSections = [...compactSections, "", "Summarized stale ranges:", ...staleLabels];
      }
      const compactText = compactSections.join("\n").trim();
      if (compactText.length <= maxChars) {
        text = compactText;
      } else {
        compactionActions.push({
          action: "drop_supporting_snippets",
          reason: "supporting snippets dropped to respect hard context pack limit",
        });
        compactSections = compactSections.filter(
          (section) => !section.startsWith("- [UNIT") || section.length < 320,
        );
        text = compactSections.join("\n").trim();
        if (text.length > maxChars) {
          text = `${text.slice(0, Math.max(0, maxChars - 18)).trimEnd()}\n...[truncated]...`;
        }
      }
    }
    this.compactionLog.push(...compactionActions);
    while (this.compactionLog.length > 10) {
      this.compactionLog.shift();
    }

    return {
      text,
      stats: {
        context_scope: {
          active_document_id: this.activeDocumentId,
          active_file_path: this.activeFilePath,
          active_ranges: [...this.activeRanges],
        },
        coverage_by_document: coveragePayload,
        compaction_actions: [...compactionActions],
        active_ranges: [...this.activeRanges],
        promoted_evidence_units: [...this.promotedEvidenceUnits],
        open_gaps: [...this.openGaps.slice(-4)],
      },
    };
  }

  bestEffortAnswer(): string {
    const contextPack = this.buildContextPack(5000).text;
    return [
      "Based on the evidence already collected, here is the best-effort answer. The evidence may be incomplete, so any missing parts should be treated as uncertain.",
      "",
      contextPack,
    ].join("\n");
  }

  snapshot(): ContextStateSnapshot {
    const evidenceUnits = [...this.evidenceUnits.values()]
      .sort(
        (left, right) =>
          (left.documentId ?? "").localeCompare(right.documentId ?? "") ||
          (left.unitNo ?? 0) - (right.unitNo ?? 0) ||
          left.evidenceId.localeCompare(right.evidenceId),
      )
      .map((evidence) => ({
        evidence_id: evidence.evidenceId,
        kind: evidence.kind,
        document_id: evidence.documentId,
        file_path: evidence.filePath,
        unit_no: evidence.unitNo,
        source_locator: evidence.sourceLocator,
        heading: evidence.heading,
        snippet: evidence.snippet,
        score: evidence.score,
        cited: evidence.cited,
        promoted: evidence.promoted,
        last_used_turn: evidence.lastUsedTurn,
      }));

    return {
      context_scope: {
        active_document_id: this.activeDocumentId,
        active_file_path: this.activeFilePath,
        active_ranges: [...this.activeRanges],
      },
      coverage_by_document: Object.fromEntries(
        [...this.coverageByDocument.entries()].map(([docId, coverage]) => [
          docId,
          this.coverageAsDict(coverage),
        ]),
      ),
      working_summary: [...this.workingSummary],
      open_gaps: [...this.openGaps],
      compaction_actions: [...this.compactionLog],
      promoted_evidence_units: [...this.promotedEvidenceUnits],
      evidence_units: evidenceUnits,
    };
  }

  private ensureCoverage(documentId: string, filePath: string, label: string): DocumentCoverage {
    const existing = this.coverageByDocument.get(documentId);
    if (existing) {
      if (filePath) {
        existing.filePath = filePath;
      }
      if (label) {
        existing.label = label;
      }
      return existing;
    }
    const created: DocumentCoverage = {
      documentId,
      filePath,
      label: label || filePath || documentId,
      totalUnits: null,
      requestedUnits: new Set<number>(),
      retrievedUnits: new Set<number>(),
      activeRanges: [],
      summarizedRanges: [],
      lastAnchor: null,
    };
    this.coverageByDocument.set(documentId, created);
    return created;
  }

  private noteRetrievedUnits(coverage: DocumentCoverage, unitNos: number[]): Set<number> {
    const cleaned = new Set(unitNos.filter((item) => Number(item) > 0).map(Number));
    const newUnits = new Set<number>();
    for (const unitNo of cleaned) {
      if (!coverage.retrievedUnits.has(unitNo)) {
        newUnits.add(unitNo);
      }
      coverage.retrievedUnits.add(unitNo);
    }
    return newUnits;
  }

  private noteRequestedRange(coverage: DocumentCoverage, start: number, end: number): void {
    const lo = Math.max(Math.min(Number(start), Number(end)), 1);
    const hi = Math.max(Number(start), Number(end));
    for (let unitNo = lo; unitNo <= hi; unitNo += 1) {
      coverage.requestedUnits.add(unitNo);
    }
  }

  private coverageRanges(coverage: DocumentCoverage): Array<{ start: number; end: number }> {
    return compressUnitRanges(coverage.retrievedUnits);
  }

  private missingRanges(coverage: DocumentCoverage): Array<{ start: number; end: number }> {
    if (coverage.requestedUnits.size === 0) {
      return [];
    }
    const requested =
      coverage.totalUnits != null
        ? new Set([...coverage.requestedUnits].filter((unitNo) => unitNo <= coverage.totalUnits!))
        : new Set(coverage.requestedUnits);
    const missing = [...requested].filter((unitNo) => !coverage.retrievedUnits.has(unitNo));
    return compressUnitRanges(missing);
  }

  private coverageAsDict(coverage: DocumentCoverage): Record<string, unknown> {
    return {
      document_id: coverage.documentId,
      file_path: coverage.filePath,
      label: coverage.label,
      total_units: coverage.totalUnits,
      retrieved_ranges: this.coverageRanges(coverage),
      active_ranges: [...coverage.activeRanges],
      summarized_ranges: [...coverage.summarizedRanges],
      missing_ranges: this.missingRanges(coverage),
      last_anchor: coverage.lastAnchor,
    };
  }

  private removeOpenGap(documentId: string, kinds: string[]): void {
    for (let index = this.openGaps.length - 1; index >= 0; index -= 1) {
      const gap = this.openGaps[index];
      if (gap.document_id === documentId && kinds.includes(String(gap.kind ?? ""))) {
        this.openGaps.splice(index, 1);
      }
    }
  }

  private trimSummary(): void {
    while (this.workingSummary.length > 8) {
      this.workingSummary.shift();
    }
  }

  private bumpTurn(): number {
    this.turnCounter += 1;
    return this.turnCounter;
  }
}
