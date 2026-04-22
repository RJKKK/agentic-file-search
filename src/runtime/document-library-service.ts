/*
Reference: legacy/python/src/fs_explorer/server.py
Reference: legacy/python/src/fs_explorer/document_library.py
Reference: legacy/python/src/fs_explorer/document_pages.py
*/

import { createHash, randomBytes } from "node:crypto";
import { mkdtemp, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { extname, join } from "node:path";

import type { PythonDocumentParserExecutor } from "./document-parsing.js";
import { PythonDocumentParserBridge } from "./document-parsing.js";
import {
  buildDocumentObjectKey,
  buildDocumentPagesKeyPrefix,
  createLibraryDocumentCatalog,
  ensureLibraryCorpus,
  getLibraryCorpusId,
  materializeDocument,
  serializeDocumentSummary,
} from "./document-library.js";
import { loadDocumentPages, pageRecordFromManifest } from "./document-pages.js";
import { LocalBlobStore } from "./blob-store.js";
import { persistDocumentPages, validateStorageFilename } from "./page-store.js";
import type {
  BlobStore,
  DeleteDocumentResult,
  ReparseDocumentResult,
  UploadDocumentInput,
  UploadDocumentResult,
} from "../types/library.js";
import type { ParsedDocument } from "../types/parsing.js";
import type {
  SqliteStorageBackend,
  StorageDocumentRecord,
  StorageImageSemanticRecord,
  StoredDocument,
} from "../types/storage.js";

function randomDocumentId(): string {
  return randomBytes(16).toString("hex");
}

function computeBufferSha256(data: Uint8Array): string {
  return createHash("sha256").update(data).digest("hex");
}

async function parseDocumentFromBytes(input: {
  parser: PythonDocumentParserExecutor;
  documentId: string;
  filename: string;
  data: Uint8Array;
}): Promise<ParsedDocument> {
  const extension = extname(input.filename).toLowerCase() || ".bin";
  const tempDir = await mkdtemp(join(tmpdir(), "fs-explorer-parse-"));
  const tempPath = join(tempDir, `source-${input.documentId}${extension}`);
  try {
    await writeFile(tempPath, input.data);
    return await input.parser.parseDocument(tempPath);
  } finally {
    await rm(tempDir, { recursive: true, force: true }).catch(() => undefined);
  }
}

function imageRecordsFromParsedDocument(
  documentId: string,
  parsedDocument: ParsedDocument,
): StorageImageSemanticRecord[] {
  return parsedDocument.units.flatMap((unit) =>
    unit.images.map((image) => ({
      imageHash: image.image_hash,
      sourceDocumentId: documentId,
      sourcePageNo: unit.unit_no,
      sourceImageIndex: image.image_index,
      mimeType: image.mime_type ?? null,
      width: image.width ?? null,
      height: image.height ?? null,
    })),
  );
}

export class DocumentLibraryService {
  constructor(
    readonly storage: SqliteStorageBackend,
    readonly blobStore: BlobStore = new LocalBlobStore(),
    private readonly parser: PythonDocumentParserExecutor = new PythonDocumentParserBridge(),
  ) {}

  createDocumentCatalog() {
    return createLibraryDocumentCatalog({
      storage: this.storage,
      blobStore: this.blobStore,
    });
  }

  async uploadDocument(input: UploadDocumentInput): Promise<UploadDocumentResult> {
    const filename = validateStorageFilename(input.filename || "");
    const corpusId = ensureLibraryCorpus(this.storage);
    const existing = this.storage.listDocuments(corpusId, false);
    const normalizedFilename = filename.toLowerCase();
    if (
      existing.some(
        (item) => String(item.original_filename || "").toLowerCase() === normalizedFilename,
      )
    ) {
      throw new Error("A document with the same filename already exists.");
    }

    const docId = randomDocumentId();
    const sourceObjectKey = buildDocumentObjectKey(docId, filename);
    const pagesPrefix = buildDocumentPagesKeyPrefix(filename);
    let storedSource = false;
    let storedPages = false;
    try {
      const blobHead = await this.blobStore.put({
        objectKey: sourceObjectKey,
        data: input.data,
      });
      storedSource = true;
      const sourceHash = computeBufferSha256(input.data);
      // Parse from an ASCII temp path so Windows/Python can handle mojibake or surrogate filenames.
      const parsedDocument = await parseDocumentFromBytes({
        parser: this.parser,
        documentId: docId,
        filename,
        data: input.data,
      });
      const pages = await persistDocumentPages({
        blobStore: this.blobStore,
        documentId: docId,
        originalFilename: filename,
        contentType: input.contentType ?? null,
        parsedDocument,
        syntheticPages: extname(filename).toLowerCase() !== ".pdf",
      });
      storedPages = true;
      const fileInfo = await stat(blobHead.absolutePath);
      const documentRecord: StorageDocumentRecord = {
        id: docId,
        corpusId,
        relativePath: filename,
        absolutePath: blobHead.absolutePath,
        content: "",
        metadataJson: "{}",
        fileMtime: Number(fileInfo.mtimeMs / 1000),
        fileSize: Number(blobHead.size),
        contentSha256: sourceHash,
        originalFilename: filename,
        objectKey: sourceObjectKey,
        sourceObjectKey,
        pagesPrefix,
        storageUri: blobHead.storageUri,
        contentType: input.contentType ?? null,
        uploadStatus: "pages_ready",
        pageCount: pages.length,
      };
      this.storage.upsertDocumentStub(documentRecord);
      this.storage.syncDocumentPages(
        docId,
        pages.map((page) => pageRecordFromManifest(page, { documentId: docId })),
      );
      this.storage.upsertImageSemantics(imageRecordsFromParsedDocument(docId, parsedDocument));
      this.storage.updateDocumentParseState(docId, sourceHash, true);
      const document = this.requireDocument(docId);
      return {
        document: serializeDocumentSummary(document),
        uploadResult: {
          corpus_id: document.corpus_id,
          storage_uri: String(document.storage_uri || ""),
          pages_generated: pages.length,
          page_count: pages.length,
          page_naming_scheme: "page-0001.md",
        },
      };
    } catch (error) {
      try {
        if (storedPages) {
          await this.blobStore.deletePrefix({ prefix: pagesPrefix });
        }
        if (storedSource) {
          await this.blobStore.delete({ objectKey: sourceObjectKey });
        }
      } catch {
        // Best-effort cleanup to mirror the legacy upload rollback path.
      }
      throw error;
    }
  }

  async reparseDocument(input: {
    docId: string;
    force?: boolean;
  }): Promise<ReparseDocumentResult> {
    const document = this.requireDocument(input.docId);
    const sourceObjectKey = String(document.source_object_key || document.object_key || "");
    if (!sourceObjectKey) {
      throw new Error("Document source object key is missing.");
    }
    const sourceData = await this.blobStore.get({ objectKey: sourceObjectKey });
    const sourceHash = computeBufferSha256(sourceData);
    if (
      !input.force &&
      String(document.parsed_content_sha256 || "") === sourceHash &&
      Number(document.page_count || 0) > 0
    ) {
      const pages = await loadDocumentPages({
        storage: this.storage,
        blobStore: this.blobStore,
        documentId: document.id,
      });
      return {
        documentId: document.id,
        pageCount: pages.length,
        pagesUpdated: 0,
        fromCache: true,
        pages,
        pageNamingScheme: "page-0001.md",
      };
    }

    const parsedDocument = await parseDocumentFromBytes({
      parser: this.parser,
      documentId: document.id,
      filename: document.original_filename || document.relative_path || sourceObjectKey,
      data: sourceData,
    });
    const storedPages = await persistDocumentPages({
      blobStore: this.blobStore,
      documentId: document.id,
      originalFilename: document.original_filename || document.relative_path,
      contentType: document.content_type,
      parsedDocument,
      syntheticPages: extname(document.original_filename || document.relative_path).toLowerCase() !== ".pdf",
    });
    this.storage.syncDocumentPages(
      document.id,
      storedPages.map((page) => pageRecordFromManifest(page, { documentId: document.id })),
    );
    this.storage.upsertImageSemantics(imageRecordsFromParsedDocument(document.id, parsedDocument));
    this.storage.updateDocumentParseState(document.id, sourceHash, true);
    const pages = await loadDocumentPages({
      storage: this.storage,
      blobStore: this.blobStore,
      documentId: document.id,
    });
    return {
      documentId: document.id,
      pageCount: pages.length,
      pagesUpdated: storedPages.length,
      fromCache: false,
      pages,
      pageNamingScheme: "page-0001.md",
    };
  }

  async deleteDocument(input: { docId: string }): Promise<DeleteDocumentResult> {
    const document = this.requireDocument(input.docId);
    const pagesPrefix = String(document.pages_prefix || "");
    if (pagesPrefix) {
      await this.blobStore.deletePrefix({ prefix: pagesPrefix }).catch(() => undefined);
    }
    const sourceObjectKey = String(document.source_object_key || document.object_key || "");
    if (sourceObjectKey) {
      await this.blobStore.delete({ objectKey: sourceObjectKey }).catch(() => undefined);
    }
    const deletedDocument = this.storage.deleteDocument(document.id);
    if (!deletedDocument) {
      throw new Error("Document not found");
    }
    return {
      document: {
        ...serializeDocumentSummary({ ...deletedDocument, is_deleted: true }),
        is_deleted: true,
        status: "deleted",
      },
      deleted: true,
    };
  }

  async materializeDocument(docId: string): Promise<StoredDocument> {
    return materializeDocument({
      storage: this.storage,
      blobStore: this.blobStore,
      document: this.requireDocument(docId),
    });
  }

  private requireDocument(docId: string): StoredDocument {
    const document = this.storage.getDocument(docId);
    if (!document) {
      throw new Error("Document not found");
    }
    return document;
  }
}
