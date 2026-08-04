import { execFileSync } from 'node:child_process';
import { FEATURE_VERSION, computeClipFeatures } from '../lib/features.js';
import { applyManualKeepSegments, buildSegments } from '../lib/analysis.js';
import { trainClassifier, predictKeepProbability } from '../lib/model.js';

function checkBinary(binary) {
  execFileSync(binary, ['-version'], { stdio: 'ignore' });
}

checkBinary('ffmpeg');
checkBinary('ffprobe');

const featureA = Array.from({ length: 160 }, (_, index) => Math.sin(index) * 0.1);
const featureB = Array.from({ length: 160 }, (_, index) => Math.cos(index) * 0.1 + 1);
const rows = [];
for (let index = 0; index < 10; index += 1) {
  rows.push({ label: 'cut', features: featureA.map(value => value + index * 0.001) });
  rows.push({ label: 'keep', features: featureB.map(value => value + index * 0.001) });
}

const model = trainClassifier(rows, { epochs: 80 });
const cutProbability = predictKeepProbability(model, featureA);
const keepProbability = predictKeepProbability(model, featureB);

if (cutProbability >= 0.4 || keepProbability <= 0.6) {
  throw new Error('Classifier self-check failed.');
}

const segments = buildSegments(
  [
    { start: 0, end: 1, probability: 0.1 },
    { start: 1, end: 2, probability: 0.8 },
    { start: 2, end: 3, probability: 0.9 },
    { start: 3, end: 4, probability: 0.2 }
  ],
  4,
  {
    threshold: 0.5,
    smoothingWindow: 1,
    minKeepDuration: 1,
    paddingBefore: 0,
    paddingAfter: 0,
    mergeGap: 0
  }
);

if (segments.keepSegments.length !== 1 || segments.keepSegments[0].start !== 1) {
  throw new Error('Segment builder self-check failed.');
}

const adjusted = applyManualKeepSegments(segments, [{ start: 0.5, end: 3.5 }], 4);
if (
  adjusted.keepSegments[0].start !== 0.5
  || adjusted.keepSegments[0].end !== 3.5
  || adjusted.editedDuration !== 3
  || adjusted.cutSegments.length !== 2
) {
  throw new Error('Manual segment adjustment self-check failed.');
}

if (!FEATURE_VERSION || typeof computeClipFeatures !== 'function') {
  throw new Error('Feature extractor self-check failed.');
}

console.log('Self-check passed.');
