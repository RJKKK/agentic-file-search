/*
Reference: legacy/python/src/fs_explorer/document_parsing.py
Reference: legacy/python/src/fs_explorer/server.py
*/

import type { PythonImageInspection } from "./python-document-assets.js";

export const IMAGE_SEMANTIC_INTERFERENCE_DROP_THRESHOLD = 0.65;
const LOW_INFORMATION_GRAYSCALE_STDDEV_THRESHOLD = 8;
const LOW_INFORMATION_MIN_SIDE_THRESHOLD = 64;
const LOW_INFORMATION_AREA_THRESHOLD = 20_000;
const LOW_INFORMATION_ASPECT_RATIO_THRESHOLD = 4;
const MONOTONE_DECORATIVE_EDGE_DENSITY_THRESHOLD = 0.003;
const MONOTONE_DECORATIVE_ENTROPY_THRESHOLD = 0.9;
const MONOTONE_DECORATIVE_HUE_STDDEV_THRESHOLD = 3;
const FALSE_TEXT_BACKGROUND_EDGE_DENSITY_THRESHOLD = 0.0002;
const FALSE_TEXT_BACKGROUND_ENTROPY_THRESHOLD = 0.75;
const FALSE_TEXT_BACKGROUND_STDDEV_THRESHOLD = 12;
const FALSE_TEXT_BACKGROUND_MEAN_THRESHOLD = 220;
const FALSE_TEXT_PALE_DECORATIVE_EDGE_DENSITY_THRESHOLD = 0.006;
const FALSE_TEXT_PALE_DECORATIVE_ENTROPY_THRESHOLD = 0.7;
const FALSE_TEXT_PALE_DECORATIVE_STDDEV_THRESHOLD = 15;
const FALSE_TEXT_PALE_DECORATIVE_MEAN_THRESHOLD = 232;
const WASHED_OUT_BACKGROUND_EDGE_DENSITY_THRESHOLD = 0.0001;
const WASHED_OUT_BACKGROUND_ENTROPY_THRESHOLD = 0.7;
const WASHED_OUT_BACKGROUND_STDDEV_THRESHOLD = 9;
const WASHED_OUT_BACKGROUND_MEAN_THRESHOLD = 230;
const PALE_DECORATIVE_BACKGROUND_MEAN_THRESHOLD = 235;
const PALE_DECORATIVE_BACKGROUND_ENTROPY_THRESHOLD = 0.65;
const PALE_DECORATIVE_BACKGROUND_EDGE_DENSITY_THRESHOLD = 0.005;
const PALE_DECORATIVE_BACKGROUND_ASPECT_RATIO_THRESHOLD = 3.5;
const PALE_SCENIC_DECORATIVE_MEAN_THRESHOLD = 232;
const PALE_SCENIC_DECORATIVE_ENTROPY_THRESHOLD = 0.7;
const PALE_SCENIC_DECORATIVE_EDGE_DENSITY_THRESHOLD = 0.006;
const PALE_SCENIC_DECORATIVE_STDDEV_THRESHOLD = 15;
const SCENIC_DECORATIVE_MEAN_THRESHOLD = 227;
const SCENIC_DECORATIVE_ENTROPY_THRESHOLD = 0.75;
const SCENIC_DECORATIVE_EDGE_DENSITY_THRESHOLD = 0.015;
const SCENIC_DECORATIVE_STDDEV_THRESHOLD = 22;
const LANDSCAPE_DECORATIVE_BANNER_ASPECT_RATIO_THRESHOLD = 3;
const LANDSCAPE_DECORATIVE_BANNER_ENTROPY_THRESHOLD = 0.87;
const LANDSCAPE_DECORATIVE_BANNER_EDGE_DENSITY_THRESHOLD = 0.012;
const LANDSCAPE_DECORATIVE_BANNER_MIN_MEAN_THRESHOLD = 140;
const LANDSCAPE_DECORATIVE_BANNER_MAX_MEAN_THRESHOLD = 205;
const LANDSCAPE_DECORATIVE_BANNER_HUE_STDDEV_THRESHOLD = 21;
const BANNER_DECORATIVE_ASPECT_RATIO_THRESHOLD = 6;
const BANNER_DECORATIVE_ENTROPY_THRESHOLD = 0.9;
const BANNER_DECORATIVE_HUE_STDDEV_THRESHOLD = 2;
const BANNER_DECORATIVE_EDGE_DENSITY_THRESHOLD = 0.05;
const BANNER_DECORATIVE_MIN_MEAN_THRESHOLD = 120;
const BANNER_DECORATIVE_MAX_MEAN_THRESHOLD = 180;
const LOW_ENTROPY_DECORATIVE_ENTROPY_THRESHOLD = 0.25;
const LOW_ENTROPY_DECORATIVE_ASPECT_RATIO_THRESHOLD = 2.2;
const LOW_ENTROPY_DECORATIVE_MIN_SIDE_THRESHOLD = 40;
const REPEATED_BANNER_ASPECT_RATIO_THRESHOLD = 4.5;
const REPEATED_BANNER_MAX_MIN_SIDE_THRESHOLD = 160;
const REPEATED_BANNER_ENTROPY_THRESHOLD = 1.2;

export interface ImageSemanticCandidateDecision {
  isCandidate: boolean;
  shouldDrop: boolean;
  reason:
    | "accepted_for_semantic_screening"
    | "accepted_without_opencv_screening"
    | "dropped_by_interference_filter";
}

export interface ImageSemanticCandidateContext {
  sameHashOnPageCount?: number | null;
}

export function evaluateImageSemanticCandidateInspection(
  inspection: Pick<
    PythonImageInspection,
    | "supported"
    | "has_text"
    | "interference_score"
    | "width"
    | "height"
    | "pixel_area"
    | "aspect_ratio"
    | "grayscale_stddev"
    | "edge_density"
    | "grayscale_entropy"
    | "grayscale_mean"
    | "hue_stddev"
  >,
  context: ImageSemanticCandidateContext = {},
): ImageSemanticCandidateDecision {
  const supported = Boolean(inspection.supported);
  if (!supported) {
    return {
      isCandidate: true,
      shouldDrop: false,
      reason: "accepted_without_opencv_screening",
    };
  }
  const hasText = Boolean(inspection.has_text);
  const interferenceScore = Number(inspection.interference_score ?? 0);
  const width = Math.max(Number(inspection.width ?? 0), 0);
  const height = Math.max(Number(inspection.height ?? 0), 0);
  const minSide = Math.min(width || Number.POSITIVE_INFINITY, height || Number.POSITIVE_INFINITY);
  const pixelArea =
    Number.isFinite(Number(inspection.pixel_area)) && Number(inspection.pixel_area) > 0
      ? Number(inspection.pixel_area)
      : width > 0 && height > 0
        ? width * height
        : Number.POSITIVE_INFINITY;
  const aspectRatio =
    Number.isFinite(Number(inspection.aspect_ratio)) && Number(inspection.aspect_ratio) > 0
      ? Number(inspection.aspect_ratio)
      : width > 0 && height > 0
        ? Math.max(width, height) / Math.max(Math.min(width, height), 1)
        : 1;
  const grayscaleStddev = Math.max(Number(inspection.grayscale_stddev ?? 0), 0);
  const edgeDensity = Math.max(Number(inspection.edge_density ?? 0), 0);
  const grayscaleEntropy = Math.max(Number(inspection.grayscale_entropy ?? 0), 0);
  const grayscaleMean = Math.max(Number(inspection.grayscale_mean ?? 0), 0);
  const hueStddev = Math.max(Number(inspection.hue_stddev ?? 0), 0);
  const sameHashOnPageCount = Math.max(Number(context.sameHashOnPageCount ?? 1), 1);
  const lowInformationDecorative =
    !hasText &&
    grayscaleStddev <= LOW_INFORMATION_GRAYSCALE_STDDEV_THRESHOLD &&
    (minSide <= LOW_INFORMATION_MIN_SIDE_THRESHOLD ||
      pixelArea <= LOW_INFORMATION_AREA_THRESHOLD ||
      aspectRatio >= LOW_INFORMATION_ASPECT_RATIO_THRESHOLD);
  const monotoneDecorative =
    !hasText &&
    edgeDensity <= MONOTONE_DECORATIVE_EDGE_DENSITY_THRESHOLD &&
    grayscaleEntropy <= MONOTONE_DECORATIVE_ENTROPY_THRESHOLD &&
    hueStddev <= MONOTONE_DECORATIVE_HUE_STDDEV_THRESHOLD;
  const falsePositiveTextBackground =
    hasText &&
    edgeDensity <= FALSE_TEXT_BACKGROUND_EDGE_DENSITY_THRESHOLD &&
    grayscaleEntropy <= FALSE_TEXT_BACKGROUND_ENTROPY_THRESHOLD &&
    grayscaleStddev <= FALSE_TEXT_BACKGROUND_STDDEV_THRESHOLD &&
    grayscaleMean >= FALSE_TEXT_BACKGROUND_MEAN_THRESHOLD;
  const falsePositivePaleDecorative =
    hasText &&
    edgeDensity <= FALSE_TEXT_PALE_DECORATIVE_EDGE_DENSITY_THRESHOLD &&
    grayscaleEntropy <= FALSE_TEXT_PALE_DECORATIVE_ENTROPY_THRESHOLD &&
    grayscaleStddev <= FALSE_TEXT_PALE_DECORATIVE_STDDEV_THRESHOLD &&
    grayscaleMean >= FALSE_TEXT_PALE_DECORATIVE_MEAN_THRESHOLD;
  const washedOutBackground =
    edgeDensity <= WASHED_OUT_BACKGROUND_EDGE_DENSITY_THRESHOLD &&
    grayscaleEntropy <= WASHED_OUT_BACKGROUND_ENTROPY_THRESHOLD &&
    grayscaleStddev <= WASHED_OUT_BACKGROUND_STDDEV_THRESHOLD &&
    grayscaleMean >= WASHED_OUT_BACKGROUND_MEAN_THRESHOLD;
  const paleDecorativeBackground =
    !hasText &&
    grayscaleMean >= PALE_DECORATIVE_BACKGROUND_MEAN_THRESHOLD &&
    grayscaleEntropy <= PALE_DECORATIVE_BACKGROUND_ENTROPY_THRESHOLD &&
    (
      edgeDensity <= PALE_DECORATIVE_BACKGROUND_EDGE_DENSITY_THRESHOLD ||
      aspectRatio >= PALE_DECORATIVE_BACKGROUND_ASPECT_RATIO_THRESHOLD
    );
  const paleScenicDecorative =
    !hasText &&
    grayscaleMean >= PALE_SCENIC_DECORATIVE_MEAN_THRESHOLD &&
    grayscaleEntropy <= PALE_SCENIC_DECORATIVE_ENTROPY_THRESHOLD &&
    grayscaleStddev <= PALE_SCENIC_DECORATIVE_STDDEV_THRESHOLD &&
    edgeDensity <= PALE_SCENIC_DECORATIVE_EDGE_DENSITY_THRESHOLD;
  const scenicDecorativeBackdrop =
    grayscaleMean >= SCENIC_DECORATIVE_MEAN_THRESHOLD &&
    grayscaleEntropy <= SCENIC_DECORATIVE_ENTROPY_THRESHOLD &&
    grayscaleStddev <= SCENIC_DECORATIVE_STDDEV_THRESHOLD &&
    edgeDensity <= SCENIC_DECORATIVE_EDGE_DENSITY_THRESHOLD;
  const landscapeDecorativeBanner =
    aspectRatio >= LANDSCAPE_DECORATIVE_BANNER_ASPECT_RATIO_THRESHOLD &&
    grayscaleEntropy <= LANDSCAPE_DECORATIVE_BANNER_ENTROPY_THRESHOLD &&
    edgeDensity <= LANDSCAPE_DECORATIVE_BANNER_EDGE_DENSITY_THRESHOLD &&
    grayscaleMean >= LANDSCAPE_DECORATIVE_BANNER_MIN_MEAN_THRESHOLD &&
    grayscaleMean <= LANDSCAPE_DECORATIVE_BANNER_MAX_MEAN_THRESHOLD &&
    hueStddev <= LANDSCAPE_DECORATIVE_BANNER_HUE_STDDEV_THRESHOLD;
  const bannerDecorativeBackdrop =
    aspectRatio >= BANNER_DECORATIVE_ASPECT_RATIO_THRESHOLD &&
    grayscaleEntropy <= BANNER_DECORATIVE_ENTROPY_THRESHOLD &&
    hueStddev <= BANNER_DECORATIVE_HUE_STDDEV_THRESHOLD &&
    edgeDensity <= BANNER_DECORATIVE_EDGE_DENSITY_THRESHOLD &&
    grayscaleMean >= BANNER_DECORATIVE_MIN_MEAN_THRESHOLD &&
    grayscaleMean <= BANNER_DECORATIVE_MAX_MEAN_THRESHOLD;
  const lowEntropyDecorativeShape =
    !hasText &&
    grayscaleEntropy <= LOW_ENTROPY_DECORATIVE_ENTROPY_THRESHOLD &&
    (
      aspectRatio >= LOW_ENTROPY_DECORATIVE_ASPECT_RATIO_THRESHOLD ||
      minSide <= LOW_ENTROPY_DECORATIVE_MIN_SIDE_THRESHOLD
    );
  const repeatedBannerDecorative =
    !hasText &&
    sameHashOnPageCount >= 2 &&
    aspectRatio >= REPEATED_BANNER_ASPECT_RATIO_THRESHOLD &&
    minSide <= REPEATED_BANNER_MAX_MIN_SIDE_THRESHOLD &&
    grayscaleEntropy <= REPEATED_BANNER_ENTROPY_THRESHOLD;
  const shouldDrop =
    (!hasText &&
      (
        interferenceScore >= IMAGE_SEMANTIC_INTERFERENCE_DROP_THRESHOLD ||
        lowInformationDecorative ||
        monotoneDecorative ||
        washedOutBackground ||
        paleDecorativeBackground ||
        paleScenicDecorative ||
        scenicDecorativeBackdrop ||
        landscapeDecorativeBanner ||
        bannerDecorativeBackdrop ||
        lowEntropyDecorativeShape ||
        repeatedBannerDecorative
      )) ||
    falsePositiveTextBackground ||
    falsePositivePaleDecorative ||
    scenicDecorativeBackdrop ||
    landscapeDecorativeBanner ||
    bannerDecorativeBackdrop;
  if (shouldDrop) {
    return {
      isCandidate: false,
      shouldDrop: true,
      reason: "dropped_by_interference_filter",
    };
  }
  return {
    isCandidate: true,
    shouldDrop: false,
    reason: supported ? "accepted_for_semantic_screening" : "accepted_without_opencv_screening",
  };
}
