/*
Reference: legacy/python/src/fs_explorer/agent.py
Reference: legacy/python/src/fs_explorer/storage/postgres.py
*/

import type { SkillModule } from "../../../src/types/skills.js";

export const skillModule: SkillModule = {
  tools: {
    async list_indexed_documents(_input, context) {
      const catalog = context.services.documentCatalog;
      if (!catalog) {
        return { output: "No document catalog configured." };
      }
      const documents = await catalog.listDocuments();
      if (documents.length === 0) {
        return { output: "No indexed documents found for the active corpus." };
      }
      return {
        output: [
          "=== INDEXED DOCUMENTS ===",
          ...documents.map(
            (document, index) =>
              `[${index + 1}] doc_id=${document.id} source=${document.absolutePath} ${document.pagesDir ? `pages_dir=${document.pagesDir} ` : ""}page_count=${document.pageCount ?? 0} name=${document.label}`,
          ),
          "",
          "Use glob/grep/read on the pages_dir to answer questions page-by-page.",
        ].join("\n"),
        receipt: "Document list receipt: listed indexed documents.",
      };
    },
  },
};
