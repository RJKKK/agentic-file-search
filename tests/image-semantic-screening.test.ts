import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { evaluateImageSemanticCandidateInspection } from "../src/runtime/image-semantic-screening.js";

describe("image semantic screening", () => {
  it("drops non-text images with high interference", () => {
    assert.deepEqual(
      evaluateImageSemanticCandidateInspection({
        supported: true,
        has_text: false,
        interference_score: 0.65,
      }),
      {
        isCandidate: false,
        shouldDrop: true,
        reason: "dropped_by_interference_filter",
      },
    );
  });

  it("keeps text-bearing images as candidates", () => {
    assert.deepEqual(
      evaluateImageSemanticCandidateInspection({
        supported: true,
        has_text: true,
        interference_score: 0.99,
      }),
      {
        isCandidate: true,
        shouldDrop: false,
        reason: "accepted_for_semantic_screening",
      },
    );
  });

  it("keeps unsupported formats out of the drop path", () => {
    assert.deepEqual(
      evaluateImageSemanticCandidateInspection({
        supported: false,
        has_text: false,
        interference_score: 0,
      }),
      {
        isCandidate: true,
        shouldDrop: false,
        reason: "accepted_without_opencv_screening",
      },
    );
  });

  it("drops low-information decorative strips", () => {
    assert.deepEqual(
      evaluateImageSemanticCandidateInspection({
        supported: true,
        has_text: false,
        interference_score: 0.42,
        width: 47,
        height: 235,
        pixel_area: 11045,
        aspect_ratio: 5,
        grayscale_stddev: 5.95,
      }),
      {
        isCandidate: false,
        shouldDrop: true,
        reason: "dropped_by_interference_filter",
      },
    );
  });

  it("drops monotone decorative background crops", () => {
    assert.deepEqual(
      evaluateImageSemanticCandidateInspection({
        supported: true,
        has_text: false,
        interference_score: 0.45,
        width: 389,
        height: 780,
        pixel_area: 303420,
        aspect_ratio: 2.01,
        grayscale_stddev: 15.46,
        edge_density: 0.0012,
        grayscale_entropy: 0.72,
        hue_stddev: 1.74,
      }),
      {
        isCandidate: false,
        shouldDrop: true,
        reason: "dropped_by_interference_filter",
      },
    );
  });

  it("drops washed-out textured backgrounds even when text detection is a false positive", () => {
    assert.deepEqual(
      evaluateImageSemanticCandidateInspection({
        supported: true,
        has_text: true,
        interference_score: 0.4,
        width: 3440,
        height: 1378,
        pixel_area: 4740320,
        aspect_ratio: 2.5,
        grayscale_stddev: 10.74,
        edge_density: 0.000026,
        grayscale_entropy: 0.623,
        grayscale_mean: 236.53,
        hue_stddev: 49.13,
      }),
      {
        isCandidate: false,
        shouldDrop: true,
        reason: "dropped_by_interference_filter",
      },
    );
  });

  it("drops washed-out low-texture background panels without text", () => {
    assert.deepEqual(
      evaluateImageSemanticCandidateInspection({
        supported: true,
        has_text: false,
        interference_score: 0.33,
        width: 1153,
        height: 2805,
        pixel_area: 3234165,
        aspect_ratio: 2.43,
        grayscale_stddev: 7.92,
        edge_density: 0,
        grayscale_entropy: 0.534,
        grayscale_mean: 242.95,
        hue_stddev: 4.86,
      }),
      {
        isCandidate: false,
        shouldDrop: true,
        reason: "dropped_by_interference_filter",
      },
    );
  });

  it("drops repeated banner decorations when the same image hash repeats on a page", () => {
    assert.deepEqual(
      evaluateImageSemanticCandidateInspection(
        {
          supported: true,
          has_text: false,
          interference_score: 0.55,
          width: 614,
          height: 106,
          pixel_area: 65084,
          aspect_ratio: 5.79,
          grayscale_stddev: 70.26,
          edge_density: 0.0592,
          grayscale_entropy: 0.668,
          grayscale_mean: 65.78,
          hue_stddev: 33.78,
        },
        {
          sameHashOnPageCount: 2,
        },
      ),
      {
        isCandidate: false,
        shouldDrop: true,
        reason: "dropped_by_interference_filter",
      },
    );
  });

  it("keeps non-repeated banner-like images available for semantic screening", () => {
    assert.deepEqual(
      evaluateImageSemanticCandidateInspection(
        {
          supported: true,
          has_text: false,
          interference_score: 0.55,
          width: 614,
          height: 106,
          pixel_area: 65084,
          aspect_ratio: 5.79,
          grayscale_stddev: 70.26,
          edge_density: 0.0592,
          grayscale_entropy: 0.668,
          grayscale_mean: 65.78,
          hue_stddev: 33.78,
        },
        {
          sameHashOnPageCount: 1,
        },
      ),
      {
        isCandidate: true,
        shouldDrop: false,
        reason: "accepted_for_semantic_screening",
      },
    );
  });

  it("drops pale decorative background crops", () => {
    assert.deepEqual(
      evaluateImageSemanticCandidateInspection({
        supported: true,
        has_text: false,
        interference_score: 0.38,
        width: 1600,
        height: 1053,
        pixel_area: 1684800,
        aspect_ratio: 1.52,
        grayscale_stddev: 9.46,
        edge_density: 0.000148,
        grayscale_entropy: 0.593,
        grayscale_mean: 238.94,
        hue_stddev: 51.34,
      }),
      {
        isCandidate: false,
        shouldDrop: true,
        reason: "dropped_by_interference_filter",
      },
    );
  });

  it("drops low-entropy dark decorative bars and blobs", () => {
    assert.deepEqual(
      evaluateImageSemanticCandidateInspection({
        supported: true,
        has_text: false,
        interference_score: 0.35,
        width: 196,
        height: 500,
        pixel_area: 98000,
        aspect_ratio: 2.55,
        grayscale_stddev: 98.76,
        edge_density: 0.0122,
        grayscale_entropy: 0.196,
        grayscale_mean: 72.83,
        hue_stddev: 42.76,
      }),
      {
        isCandidate: false,
        shouldDrop: true,
        reason: "dropped_by_interference_filter",
      },
    );
  });

  it("drops pale decorative backgrounds even when text detection is a false positive", () => {
    assert.deepEqual(
      evaluateImageSemanticCandidateInspection({
        supported: true,
        has_text: true,
        interference_score: 0.38,
        width: 2013,
        height: 1325,
        pixel_area: 2667225,
        aspect_ratio: 1.52,
        grayscale_stddev: 9.49,
        edge_density: 0.000139,
        grayscale_entropy: 0.595,
        grayscale_mean: 238.94,
        hue_stddev: 51.33,
      }),
      {
        isCandidate: false,
        shouldDrop: true,
        reason: "dropped_by_interference_filter",
      },
    );
  });

  it("drops pale scenic decorative backgrounds misread as text", () => {
    assert.deepEqual(
      evaluateImageSemanticCandidateInspection({
        supported: true,
        has_text: true,
        interference_score: 0.39,
        width: 1600,
        height: 749,
        pixel_area: 1198400,
        aspect_ratio: 2.14,
        grayscale_stddev: 10.79,
        edge_density: 0.00104,
        grayscale_entropy: 0.61,
        grayscale_mean: 238.0,
        hue_stddev: 49.0,
      }),
      {
        isCandidate: false,
        shouldDrop: true,
        reason: "dropped_by_interference_filter",
      },
    );
  });

  it("drops smaller pale scenic decorative crops misread as text", () => {
    assert.deepEqual(
      evaluateImageSemanticCandidateInspection({
        supported: true,
        has_text: true,
        interference_score: 0.42,
        width: 208,
        height: 382,
        pixel_area: 79456,
        aspect_ratio: 1.84,
        grayscale_stddev: 14.04,
        edge_density: 0.00516,
        grayscale_entropy: 0.689,
        grayscale_mean: 232.62,
        hue_stddev: 19.92,
      }),
      {
        isCandidate: false,
        shouldDrop: true,
        reason: "dropped_by_interference_filter",
      },
    );
  });

  it("drops smaller pale scenic decorative crops without text", () => {
    assert.deepEqual(
      evaluateImageSemanticCandidateInspection({
        supported: true,
        has_text: false,
        interference_score: 0.41,
        width: 208,
        height: 382,
        pixel_area: 79456,
        aspect_ratio: 1.84,
        grayscale_stddev: 13.97,
        edge_density: 0.00415,
        grayscale_entropy: 0.687,
        grayscale_mean: 232.62,
        hue_stddev: 12.38,
      }),
      {
        isCandidate: false,
        shouldDrop: true,
        reason: "dropped_by_interference_filter",
      },
    );
  });

  it("drops scenic decorative backdrop crops regardless of text false positives", () => {
    assert.deepEqual(
      evaluateImageSemanticCandidateInspection({
        supported: true,
        has_text: true,
        interference_score: 0.46,
        width: 394,
        height: 323,
        pixel_area: 127262,
        aspect_ratio: 1.22,
        grayscale_stddev: 20.25,
        edge_density: 0.0138,
        grayscale_entropy: 0.737,
        grayscale_mean: 227.7,
        hue_stddev: 7.04,
      }),
      {
        isCandidate: false,
        shouldDrop: true,
        reason: "dropped_by_interference_filter",
      },
    );
  });

  it("drops wide decorative banners with low hue variation", () => {
    assert.deepEqual(
      evaluateImageSemanticCandidateInspection({
        supported: true,
        has_text: true,
        interference_score: 0.54,
        width: 706,
        height: 72,
        pixel_area: 50832,
        aspect_ratio: 9.81,
        grayscale_stddev: 28.17,
        edge_density: 0.0339,
        grayscale_entropy: 0.827,
        grayscale_mean: 135.01,
        hue_stddev: 0.85,
      }),
      {
        isCandidate: false,
        shouldDrop: true,
        reason: "dropped_by_interference_filter",
      },
    );
  });

  it("drops wide landscape decorative banners", () => {
    assert.deepEqual(
      evaluateImageSemanticCandidateInspection({
        supported: true,
        has_text: true,
        interference_score: 0.47,
        width: 1600,
        height: 467,
        pixel_area: 747200,
        aspect_ratio: 3.43,
        grayscale_stddev: 29.25,
        edge_density: 0.00294,
        grayscale_entropy: 0.808,
        grayscale_mean: 196.51,
        hue_stddev: 4.94,
      }),
      {
        isCandidate: false,
        shouldDrop: true,
        reason: "dropped_by_interference_filter",
      },
    );
  });

  it("drops scenic ribbon banners with slightly higher edge density", () => {
    assert.deepEqual(
      evaluateImageSemanticCandidateInspection({
        supported: true,
        has_text: false,
        interference_score: 0.45,
        width: 662,
        height: 125,
        pixel_area: 82750,
        aspect_ratio: 5.296,
        grayscale_stddev: 33.7,
        edge_density: 0.0114,
        grayscale_entropy: 0.64,
        grayscale_mean: 200.5,
        hue_stddev: 1.1,
      }),
      {
        isCandidate: false,
        shouldDrop: true,
        reason: "dropped_by_interference_filter",
      },
    );
  });

  it("drops slightly darker scenic decorative crops", () => {
    assert.deepEqual(
      evaluateImageSemanticCandidateInspection({
        supported: true,
        has_text: true,
        interference_score: 0.46,
        width: 300,
        height: 276,
        pixel_area: 82800,
        aspect_ratio: 1.09,
        grayscale_stddev: 21.86,
        edge_density: 0.0144,
        grayscale_entropy: 0.726,
        grayscale_mean: 227.63,
        hue_stddev: 6.69,
      }),
      {
        isCandidate: false,
        shouldDrop: true,
        reason: "dropped_by_interference_filter",
      },
    );
  });
});
