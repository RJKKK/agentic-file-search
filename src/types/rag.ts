/*
Reference: legacy/python/src/fs_explorer/server.py
*/

export type TraditionalRetrievalMode = "keyword" | "semantic" | "hybrid";

export interface TraditionalRagUsedChunk {
  document_chunk_id: string;
  retrieval_chunk_ids: string[];
  page_nos: number[];
  bboxes: Array<[number, number, number, number]>;
  score: number;
  content: string;
  summary_text?: string | null;
  source_link: string;
  compression_applied: boolean;
}

export interface TraditionalRagQueryResult {
  mode: TraditionalRetrievalMode;
  question: string;
  answer: string;
  used_chunks: TraditionalRagUsedChunk[];
  warnings?: string[];
}
