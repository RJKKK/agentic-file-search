import { readFile, readdir, writeFile } from "node:fs/promises";
import { basename, dirname, resolve } from "node:path";

import {
  evaluateImageSemanticCandidateInspection,
  type ImageSemanticCandidateContext,
} from "../src/runtime/image-semantic-screening.js";
import { PythonDocumentAssetBridge } from "../src/runtime/python-document-assets.js";

interface CliOptions {
  imagePath: string;
  mimeType: string | null;
  compressedOutputPath: string | null;
}

interface ExtractedImageManifestEntry {
  file_name?: string;
  page_no?: number;
  image_hash?: string;
}

interface ExtractedImageManifest {
  images?: ExtractedImageManifestEntry[];
}

function usage(): never {
  throw new Error(
    [
      "Usage: npm run dev:test-image-candidate -- <image-file> [--write-compressed <file>]",
      "Example: npm run dev:test-image-candidate -- ./tmp/page-0001-image-01.png",
    ].join("\n"),
  );
}

function guessMimeType(filePath: string): string | null {
  const lowered = filePath.toLowerCase();
  if (lowered.endsWith(".png")) {
    return "image/png";
  }
  if (lowered.endsWith(".jpg") || lowered.endsWith(".jpeg")) {
    return "image/jpeg";
  }
  if (lowered.endsWith(".webp")) {
    return "image/webp";
  }
  if (lowered.endsWith(".gif")) {
    return "image/gif";
  }
  if (lowered.endsWith(".bmp")) {
    return "image/bmp";
  }
  return null;
}

function parseArgs(argv: string[]): CliOptions {
  let imagePath = "";
  let mimeType: string | null = null;
  let compressedOutputPath: string | null = null;
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    const next = argv[index + 1];
    if (!token) {
      continue;
    }
    if (!token.startsWith("--") && !imagePath) {
      imagePath = token;
      continue;
    }
    if (token === "--image" && next) {
      imagePath = next;
      index += 1;
      continue;
    }
    if (token === "--mime" && next) {
      mimeType = next;
      index += 1;
      continue;
    }
    if (token === "--write-compressed" && next) {
      compressedOutputPath = next;
      index += 1;
      continue;
    }
  }
  if (!imagePath) {
    usage();
  }
  const resolvedImagePath = resolve(imagePath);
  return {
    imagePath: resolvedImagePath,
    mimeType: mimeType?.trim() || guessMimeType(resolvedImagePath),
    compressedOutputPath: compressedOutputPath ? resolve(compressedOutputPath) : null,
  };
}

function shortHashFromFilename(filePath: string): string | null {
  const match = basename(filePath).match(/-([a-f0-9]{12})\.[a-z0-9]+$/i);
  return match?.[1]?.toLowerCase() ?? null;
}

function pageNoFromFilename(filePath: string): number | null {
  const match = basename(filePath).match(/^page-(\d+)-image-\d+-[a-f0-9]{12}\.[a-z0-9]+$/i);
  return match ? Number.parseInt(match[1] ?? "", 10) : null;
}

async function discoverCandidateContext(imagePath: string): Promise<ImageSemanticCandidateContext> {
  const directoryPath = dirname(imagePath);
  const fileName = basename(imagePath);
  const manifestPath = resolve(directoryPath, "manifest.json");
  try {
    const manifestRaw = await readFile(manifestPath, "utf8");
    const manifest = JSON.parse(manifestRaw) as ExtractedImageManifest;
    const currentEntry = manifest.images?.find((entry) => entry.file_name === fileName);
    if (currentEntry?.image_hash && currentEntry.page_no) {
      const sameHashOnPageCount = (manifest.images ?? []).filter(
        (entry) =>
          entry.page_no === currentEntry.page_no && entry.image_hash === currentEntry.image_hash,
      ).length;
      return { sameHashOnPageCount };
    }
  } catch {
    // Fall through to filename-based sibling scan.
  }

  const shortHash = shortHashFromFilename(imagePath);
  const pageNo = pageNoFromFilename(imagePath);
  if (!shortHash || !pageNo) {
    return {};
  }
  try {
    const siblingNames = await readdir(directoryPath);
    const sameHashOnPageCount = siblingNames.filter((siblingName) => {
      const siblingPageNo = pageNoFromFilename(siblingName);
      const siblingShortHash = shortHashFromFilename(siblingName);
      return siblingPageNo === pageNo && siblingShortHash === shortHash;
    }).length;
    return { sameHashOnPageCount };
  } catch {
    return {};
  }
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const bridge = new PythonDocumentAssetBridge();
  const imageBytes = await readFile(options.imagePath);
  const inspection = await bridge.inspectImage(imageBytes.toString("base64"), options.mimeType);
  if (!inspection) {
    throw new Error("Image inspection returned no result.");
  }
  const context = await discoverCandidateContext(options.imagePath);
  const decision = evaluateImageSemanticCandidateInspection(inspection, context);

  if (options.compressedOutputPath) {
    await writeFile(
      options.compressedOutputPath,
      Buffer.from(inspection.compressed_bytes_base64, "base64"),
    );
  }

  console.log(
    JSON.stringify(
      {
        ok: true,
        image_path: options.imagePath,
        mime_type: options.mimeType,
        supported: inspection.supported,
        has_text: inspection.has_text,
        interference_score: inspection.interference_score,
        width: inspection.width ?? null,
        height: inspection.height ?? null,
        pixel_area: inspection.pixel_area ?? null,
        aspect_ratio: inspection.aspect_ratio ?? null,
        grayscale_stddev: inspection.grayscale_stddev ?? null,
        edge_density: inspection.edge_density ?? null,
        grayscale_entropy: inspection.grayscale_entropy ?? null,
        grayscale_mean: inspection.grayscale_mean ?? null,
        hue_stddev: inspection.hue_stddev ?? null,
        same_hash_on_page_count: context.sameHashOnPageCount ?? null,
        compressed_byte_size: inspection.compressed_byte_size,
        output_mime_type: inspection.output_mime_type,
        is_candidate: decision.isCandidate,
        should_drop: decision.shouldDrop,
        reason: decision.reason,
        compressed_output_path: options.compressedOutputPath,
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
