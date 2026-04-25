/*
Reference: legacy/python/src/fs_explorer/server.py
*/

export type TraditionalRetrievalMode = "keyword" | "semantic" | "hybrid";
export type TraditionalRagReferenceKind = "retrieval_chunk" | "document_chunk";

export interface TraditionalRagCerContextRef {
  reference_id: string;
  reference_kind: TraditionalRagReferenceKind;
  role: "center" | "context";
  source_locator: string | null;
  source_link: string;
  page_nos: number[];
}

export interface TraditionalRagChunkReference {
  citation_no: number;
  reference_id: string;
  reference_kind: TraditionalRagReferenceKind;
  document_id: string;
  document_name: string;
  source_locator: string | null;
  show_full_chunk_detail: boolean;
  document_chunk_ids: string[];
  retrieval_unit_ids: string[];
  page_nos: number[];
  bboxes: Array<[number, number, number, number]>;
  score: number;
  source_link: string;
  compression_applied: boolean;
  debug_cer_applied?: boolean;
  debug_cer_context_refs?: TraditionalRagCerContextRef[];
  debug_enriched_content?: string;
}

export interface TraditionalRagUsedChunk extends TraditionalRagChunkReference {}
export interface TraditionalRagDetailChunk extends TraditionalRagChunkReference {}

export interface TraditionalRagQueryResult {
  mode: TraditionalRetrievalMode;
  question: string;
  answer: string;
  used_chunks: TraditionalRagUsedChunk[];
  detail_chunks?: TraditionalRagDetailChunk[];
  warnings?: string[];
}

export interface TraditionalRagRetrieveResult {
  mode: TraditionalRetrievalMode;
  question: string;
  retrieved_chunks: TraditionalRagChunkReference[];
  warnings?: string[];
}

export interface TraditionalRagPromptMessage {
  role: "system" | "user";
  content: string;
}

export interface TraditionalRagPromptPreviewResult {
  mode: TraditionalRetrievalMode;
  question: string;
  prompt_variant: "stream";
  model: string | null;
  temperature: number;
  messages: TraditionalRagPromptMessage[];
  system_prompt: string;
  user_prompt: string;
  request_body_json: string;
  warnings?: string[];
}
