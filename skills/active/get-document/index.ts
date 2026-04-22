/*
Reference: legacy/python/src/fs_explorer/agent.py
Reference: legacy/python/src/fs_explorer/context_state.py
*/

import type { SkillModule } from "../../../src/types/skills.js";

export const skillModule: SkillModule = {
  tools: {
    async get_document(input, context) {
      const docId = String(input.doc_id ?? "");
      if (!docId) {
        return { output: "get_document requires doc_id." };
      }
      if (context.services.indexSearch) {
        const indexed = await context.services.indexSearch.getDocument(docId);
        if (!indexed.structured) {
          return { output: indexed.rendered };
        }
        const summary = context.contextState.ingestDocumentRead({
          documentId: indexed.structured.document_id,
          filePath: indexed.structured.absolute_path,
          label: indexed.structured.label,
          body: indexed.structured.content,
        });
        context.services.indexSearch.emit("evidence_added", {
          tool_name: "get_document",
          summary,
        });
        context.services.indexSearch.emit("context_scope_updated", {
          context_scope: context.contextState.snapshot().context_scope,
        });
        return {
          output: indexed.rendered,
          receipt: `Document receipt: doc_id=${docId}; path=${indexed.structured.absolute_path}; stored a condensed excerpt for later reasoning.`,
        };
      }
      const catalog = context.services.documentCatalog;
      if (!catalog) {
        return { output: "No document catalog configured." };
      }
      const document = await catalog.getDocument(docId);
      if (!document) {
        return { output: `No indexed document found for doc_id=${JSON.stringify(docId)}` };
      }

      // Reference: legacy/python/src/fs_explorer/agent.py get_document context ingestion path.
      context.contextState.ingestDocumentRead({
        documentId: document.id,
        filePath: document.absolutePath,
        label: document.label,
        body: document.content,
      });

      return {
        output: [
          `=== DOCUMENT ${document.id} ===`,
          `Path: ${document.absolutePath}`,
          "",
          document.content,
        ].join("\n"),
        receipt: `Document receipt: doc_id=${document.id}; path=${document.absolutePath}; stored a condensed excerpt for later reasoning.`,
      };
    },
  },
};
