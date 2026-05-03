import { mkdir, writeFile } from "node:fs/promises";
import { basename, resolve } from "node:path";

import { PythonDocumentAssetBridge } from "../src/runtime/python-document-assets.js";

interface CliOptions {
  pdfPath: string;
  outputDir: string;
  pageNos: number[] | null;
}

function usage(): never {
  throw new Error(
    [
      "Usage: npm run dev:extract-pdf-images -- <file.pdf> [--out <dir>] [--pages 1,2,3]",
      "Example: npm run dev:extract-pdf-images -- ./data/sample.pdf",
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
    outputDir: resolve(outputDir || `${resolvedPdfPath}.images`),
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
  await mkdir(options.outputDir, { recursive: true });

  const manifest = [];
  for (const image of images) {
    const extension = extensionFromMimeType(image.mime_type);
    const fileName = `page-${String(image.page_no).padStart(4, "0")}-image-${String(image.image_index).padStart(2, "0")}-${image.image_hash.slice(0, 12)}.${extension}`;
    const filePath = resolve(options.outputDir, fileName);
    await writeFile(filePath, Buffer.from(image.bytes_base64, "base64"));
    manifest.push({
      file_name: fileName,
      file_path: filePath,
      page_no: image.page_no,
      image_index: image.image_index,
      image_hash: image.image_hash,
      mime_type: image.mime_type,
      width: image.width,
      height: image.height,
      byte_size: image.byte_size,
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
        image_count: manifest.length,
        images: manifest,
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
        image_count: manifest.length,
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
