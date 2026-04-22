/*
Reference: legacy/python/src/fs_explorer/context_budget.py
Reference: legacy/python/src/fs_explorer/agent.py
*/

import type { AgentMessage } from "../types/messages.js";

const UNIT_RE = /\[UNIT\s+(\d+)\b/gi;

interface MessageRecord {
  index: number;
  role: string;
  originalText: string;
  text: string;
  bucket: string;
  priority: number;
  dropped: boolean;
}

export interface ContextBudgetStats {
  beforeChars: number;
  afterChars: number;
  hardLimitChars: number;
  droppedMessages: number;
  truncatedMessages: number;
  compressionRatio: number;
  overflowWarning: boolean;
}

export class ContextBudgetManager {
  constructor(
    private readonly config: {
      maxInputChars?: number;
      minRecentMessages?: number;
    } = {},
  ) {}

  compactHistory(
    history: AgentMessage[],
    options: { anchorUnitNo?: number | null } = {},
  ): {
    messages: AgentMessage[];
    stats: ContextBudgetStats;
  } {
    const maxInputChars = this.config.maxInputChars ?? 12_000;
    const minRecentMessages = this.config.minRecentMessages ?? 6;
    if (history.length === 0) {
      return {
        messages: [],
        stats: {
          beforeChars: 0,
          afterChars: 0,
          hardLimitChars: maxInputChars,
          droppedMessages: 0,
          truncatedMessages: 0,
          compressionRatio: 1,
          overflowWarning: false,
        },
      };
    }

    const records = this.buildRecords(history, options.anchorUnitNo ?? null);
    const beforeChars = this.estimateChars(records);

    this.applyBucketCaps(records, {
      current: 3600,
      ring0: 3000,
      ring1: 1800,
      ring2: 900,
      history: 900,
    });
    if (this.estimateChars(records) > maxInputChars) {
      this.applyBucketCaps(records, {
        current: 3000,
        ring0: 2200,
        ring1: 1200,
        ring2: 600,
        history: 450,
      });
    }

    if (this.estimateChars(records) > maxInputChars) {
      this.dropOldestBucket(records, "history", minRecentMessages);
    }
    if (this.estimateChars(records) > maxInputChars) {
      this.dropOldestBucket(records, "ring2", 3);
    }
    if (this.estimateChars(records) > maxInputChars) {
      this.dropOldestBucket(records, "ring1", 2);
    }

    if (this.estimateChars(records) > maxInputChars) {
      this.applyBucketCaps(records, {
        current: 1200,
        ring0: 900,
        ring1: 600,
        ring2: 350,
        history: 220,
      });
    }

    while (this.estimateChars(records) > maxInputChars) {
      const alive = records.filter((record) => !record.dropped);
      if (alive.length <= 1) {
        break;
      }
      let candidates = alive.filter(
        (record) => record.bucket !== "current" && record.priority < 100,
      );
      if (candidates.length === 0) {
        candidates = alive.slice(0, -1);
      }
      if (candidates.length === 0) {
        break;
      }
      candidates.sort((left, right) => left.priority - right.priority || left.index - right.index);
      candidates[0].dropped = true;
    }

    this.hardCap(records, maxInputChars);

    const output = this.recordsToMessages(records, history);
    const afterChars = output.reduce((sum, message) => sum + message.content.length, 0);
    const truncatedMessages = records.filter(
      (record) => !record.dropped && record.text !== record.originalText,
    ).length;
    const droppedMessages = records.filter((record) => record.dropped).length;
    const overflowWarning = afterChars > maxInputChars;

    return {
      messages: output,
      stats: {
        beforeChars,
        afterChars,
        hardLimitChars: maxInputChars,
        droppedMessages,
        truncatedMessages,
        compressionRatio: Number((afterChars / Math.max(beforeChars, 1)).toFixed(4)),
        overflowWarning,
      },
    };
  }

  private buildRecords(history: AgentMessage[], anchorUnitNo: number | null): MessageRecord[] {
    const lastIndex = history.length - 1;
    return history.map((message, index) => {
      const text = message.content;
      const bucket = this.classifyBucket(text, anchorUnitNo);
      return {
        index,
        role: message.role,
        originalText: text,
        text,
        bucket,
        priority: this.priorityForBucket(bucket, index, lastIndex),
        dropped: false,
      };
    });
  }

  private classifyBucket(text: string, anchorUnitNo: number | null): string {
    if (!text) {
      return "history";
    }
    if (!/STRUCTURED CONTEXT PACK|Read receipt:|Document receipt:/i.test(text)) {
      return "history";
    }
    if (anchorUnitNo == null) {
      return "ring2";
    }
    const unitNos = [...text.matchAll(UNIT_RE)].map((match) => Number.parseInt(match[1], 10));
    if (unitNos.length === 0) {
      return "ring2";
    }
    const distance = Math.min(...unitNos.map((unitNo) => Math.abs(unitNo - anchorUnitNo)));
    if (distance === 0) {
      return "ring0";
    }
    if (distance <= 2) {
      return "ring1";
    }
    return "ring2";
  }

  private priorityForBucket(bucket: string, index: number, lastIndex: number): number {
    if (index === lastIndex) {
      return 100;
    }
    if (bucket === "ring0") {
      return 90;
    }
    if (bucket === "ring1") {
      return 80;
    }
    if (bucket === "ring2") {
      return 70;
    }
    return 40 + Math.min(index, 20);
  }

  private applyBucketCaps(records: MessageRecord[], caps: Record<string, number>): void {
    for (const record of records) {
      if (record.dropped) {
        continue;
      }
      const cap = caps[record.bucket] ?? caps.history;
      record.text = this.compressText(record.text, cap, record.bucket);
    }
  }

  private dropOldestBucket(records: MessageRecord[], bucket: string, minKeep: number): void {
    while (this.estimateChars(records) > (this.config.maxInputChars ?? 12_000)) {
      const alive = records.filter((record) => !record.dropped);
      if (alive.length <= minKeep) {
        break;
      }
      const candidates = records.filter(
        (record) => !record.dropped && record.bucket === bucket,
      );
      if (candidates.length === 0) {
        break;
      }
      candidates.sort((left, right) => left.index - right.index);
      candidates[0].dropped = true;
    }
  }

  private hardCap(records: MessageRecord[], maxInputChars: number): void {
    let guard = 0;
    while (this.estimateChars(records) > maxInputChars && guard < 200) {
      guard += 1;
      let candidates = records.filter((record) => !record.dropped && record.bucket !== "current");
      if (candidates.length === 0) {
        candidates = records.filter((record) => !record.dropped);
      }
      if (candidates.length === 0) {
        break;
      }
      candidates.sort((left, right) => left.priority - right.priority || left.index - right.index);
      const target = candidates[0];
      if (target.text.length > 140) {
        target.text = this.compressText(target.text, Math.max(120, Math.floor(target.text.length * 0.65)), target.bucket);
      } else {
        target.dropped = true;
      }
    }
  }

  private compressText(text: string, maxChars: number, bucket: string): string {
    if (text.length <= maxChars) {
      return text;
    }
    if (bucket === "current" || bucket === "ring0") {
      const head = text.slice(0, Math.max(0, Math.floor(maxChars * 0.55))).trimEnd();
      const tail = text
        .slice(Math.max(0, text.length - Math.floor(maxChars * 0.3)))
        .trimStart();
      return `${head}\n...\n${tail}`.slice(0, maxChars);
    }
    return `${text.slice(0, Math.max(0, maxChars - 3)).trimEnd()}...`;
  }

  private estimateChars(records: MessageRecord[]): number {
    return records
      .filter((record) => !record.dropped)
      .reduce((sum, record) => sum + record.text.length, 0);
  }

  private recordsToMessages(records: MessageRecord[], history: AgentMessage[]): AgentMessage[] {
    return records
      .filter((record) => !record.dropped)
      .map((record) => ({
        ...history[record.index],
        content: record.text,
      }));
  }
}
