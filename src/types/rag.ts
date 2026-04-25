/*
Reference: legacy/python/src/fs_explorer/server.py
*/

export type TraditionalRetrievalMode = "keyword" | "semantic" | "hybrid";

export interface TraditionalRagChunkReference {
  citation_no: number;
  document_chunk_id: string;
  document_id: string;
  document_name: string;
  source_locator: string | null;
  show_full_chunk_detail: boolean;
  retrieval_chunk_ids: string[];
  page_nos: number[];
  bboxes: Array<[number, number, number, number]>;
  score: number;
  source_link: string;
  compression_applied: boolean;
}

export interface TraditionalRagUsedChunk extends TraditionalRagChunkReference {}

export interface TraditionalRagQueryResult {
  mode: TraditionalRetrievalMode;
  question: string;
  answer: string;
  used_chunks: TraditionalRagUsedChunk[];
  warnings?: string[];
}

export interface TraditionalRagRetrieveResult {
  mode: TraditionalRetrievalMode;
  question: string;
  retrieved_chunks: TraditionalRagChunkReference[];
  warnings?: string[];
}
