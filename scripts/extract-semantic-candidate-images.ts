import { mkdir, rm, writeFile } from "node:fs/promises";
import { basename, resolve } from "node:path";

import { evaluateImageSemanticCandidateInspection } from "../src/runtime/image-semantic-screening.js";
import { PythonDocumentAssetBridge } from "../src/runtime/python-document-assets.js";

interface CliOptions {
  pdfPath: string;
  outputDir: string;
  pageNos: number[] | null;
}

interface CandidateManifestEntry {
  file_name: string;
  file_path: string;
  page_no: number;
  image_index: number;
  image_hash: string;
  original_mime_type: string | null;
  output_mime_type: string;
  width: number | null;
  height: number | null;
  same_hash_on_page_count: number;
  has_text: boolean;
  interference_score: number;
  reason: string;
  compressed_byte_size: number;
}

function usage(): never {
  throw new Error(
    [
      "Usage: npm run dev:extract-semantic-images -- <file.pdf> [--out <dir>] [--pages 1,2,3]",
      "Example: npm run dev:extract-semantic-images -- ./data/sample.pdf",
    ].join("\n"),
  );
}

function parsePageNos(raw: string): number[] {
  return [...new Set(raw.split(",").map((item) => Number.parseInt(item.trim(), 10)).filter((value) => value > 0))];
}

function parseArgs(argv: string[]): CliOptions {
  let pdfPath = "";
  let outputDir = "";
  let pageNos: number[] | null = null;
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    const next = argv[index + 1];
    if (!token) {
      continue;
    }
    if (!token.startsWith("--") && !pdfPath) {
      pdfPath = token;
      continue;
    }
    if (token === "--pdf" && next) {
      pdfPath = next;
      index += 1;
      continue;
    }
    if (token === "--out" && next) {
      outputDir = next;
      index += 1;
      continue;
    }
    if (token === "--pages" && next) {
      pageNos = parsePageNos(next);
      index += 1;
      continue;
    }
  }
  if (!pdfPath) {
    usage();
  }
  const resolvedPdfPath = resolve(pdfPath);
  return {
    pdfPath: resolvedPdfPath,
    outputDir: resolve(outputDir || `${resolvedPdfPath}.semantic-images`),
    pageNos: pageNos?.length ? pageNos : null,
  };
}

function extensionFromMimeType(mimeType: string | null): string {
  switch (mimeType) {
    case "image/jpeg":
      return "jpg";
    case "image/png":
      return "png";
    case "image/webp":
      return "webp";
    case "image/gif":
      return "gif";
    case "image/bmp":
      return "bmp";
    default:
      return "bin";
  }
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const bridge = new PythonDocumentAssetBridge();
  const images = await bridge.extractPdfImages(options.pdfPath, options.pageNos);
  await rm(options.outputDir, { recursive: true, force: true });
  await mkdir(options.outputDir, { recursive: true });

  const sameHashOnPageCounts = new Map<string, number>();
  for (const image of images) {
    const key = `${image.page_no}:${image.image_hash}`;
    sameHashOnPageCounts.set(key, (sameHashOnPageCounts.get(key) ?? 0) + 1);
  }

  const selectedImages: CandidateManifestEntry[] = [];
  let droppedCount = 0;
  for (const image of images) {
    const inspection = await bridge.inspectImage(image.bytes_base64, image.mime_type);
    if (!inspection) {
      droppedCount += 1;
      continue;
    }
    const context = {
      sameHashOnPageCount: sameHashOnPageCounts.get(`${image.page_no}:${image.image_hash}`) ?? 1,
    };
    const decision = evaluateImageSemanticCandidateInspection(inspection, context);
    if (decision.shouldDrop) {
      droppedCount += 1;
      continue;
    }
    const outputMimeType = inspection.output_mime_type;
    const extension = extensionFromMimeType(outputMimeType);
    const fileName = `page-${String(image.page_no).padStart(4, "0")}-image-${String(image.image_index).padStart(2, "0")}-${image.image_hash.slice(0, 12)}.${extension}`;
    const filePath = resolve(options.outputDir, fileName);
    await writeFile(filePath, Buffer.from(inspection.compressed_bytes_base64, "base64"));
    selectedImages.push({
      file_name: fileName,
      file_path: filePath,
      page_no: image.page_no,
      image_index: image.image_index,
      image_hash: image.image_hash,
      original_mime_type: image.mime_type,
      output_mime_type: outputMimeType,
      width: inspection.width ?? image.width,
      height: inspection.height ?? image.height,
      same_hash_on_page_count: context.sameHashOnPageCount,
      has_text: Boolean(inspection.has_text),
      interference_score: Number(inspection.interference_score ?? 0),
      reason: decision.reason,
      compressed_byte_size: inspection.compressed_byte_size,
    });
  }

  const manifestPath = resolve(options.outputDir, "manifest.json");
  await writeFile(
    manifestPath,
    `${JSON.stringify(
      {
        pdf_path: options.pdfPath,
        pdf_name: basename(options.pdfPath),
        output_dir: options.outputDir,
        page_nos: options.pageNos,
        extracted_image_count: images.length,
        selected_image_count: selectedImages.length,
        dropped_image_count: droppedCount,
        images: selectedImages,
      },
      null,
      2,
    )}\n`,
    "utf8",
  );

  console.log(
    JSON.stringify(
      {
        ok: true,
        pdf_path: options.pdfPath,
        output_dir: options.outputDir,
        manifest_path: manifestPath,
        extracted_image_count: images.length,
        selected_image_count: selectedImages.length,
        dropped_image_count: droppedCount,
      },
      null,
      2,
    ),
  );
}

void main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
