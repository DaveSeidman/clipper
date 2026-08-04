import { execFileSync } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import { trainClassifier } from '../lib/model.js';
import { jobDir, jobFile } from '../lib/paths.js';

const PRESETS = {
  'farneback-96': {
    method: 'farneback',
    width: 96,
    height: 54,
    fps: 10,
    gridWidth: 8,
    gridHeight: 6
  },
  'farneback-160': {
    method: 'farneback',
    width: 160,
    height: 90,
    fps: 12,
    gridWidth: 10,
    gridHeight: 6
  },
  'farneback-256': {
    method: 'farneback',
    width: 256,
    height: 144,
    fps: 12,
    gridWidth: 12,
    gridHeight: 8
  },
  'dis-160': {
    method: 'dis',
    width: 160,
    height: 90,
    fps: 12,
    gridWidth: 10,
    gridHeight: 6
  },
  'farneback-160-context3': {
    method: 'farneback',
    width: 160,
    height: 90,
    fps: 12,
    contextSeconds: 3,
    gridWidth: 10,
    gridHeight: 6
  },
  'farneback-160-context5': {
    method: 'farneback',
    width: 160,
    height: 90,
    fps: 12,
    contextSeconds: 5,
    gridWidth: 10,
    gridHeight: 6
  },
  'farneback-160-context7': {
    method: 'farneback',
    width: 160,
    height: 90,
    fps: 12,
    contextSeconds: 7,
    gridWidth: 10,
    gridHeight: 6
  }
};

const [jobId, presetName, ...flags] = process.argv.slice(2);
if (!jobId || !PRESETS[presetName]) {
  console.error(`Usage: node server/scripts/run-motion-experiment.js <job-id> <${Object.keys(PRESETS).join('|')}> [--force]`);
  process.exit(1);
}

const config = PRESETS[presetName];
const experimentDir = path.join(jobDir(jobId), 'experiments');
const featurePath = path.join(experimentDir, `${presetName}-features.json`);
const resultPath = path.join(experimentDir, `${presetName}-result.json`);
await fs.mkdir(experimentDir, { recursive: true });

const pythonArgs = [
  path.join(process.cwd(), 'server', 'scripts', 'extract-flow-features.py'),
  '--job', jobFile(jobId),
  '--output', featurePath,
  '--method', config.method,
  '--width', String(config.width),
  '--height', String(config.height),
  '--fps', String(config.fps),
  '--context-seconds', String(config.contextSeconds || 1),
  '--grid-width', String(config.gridWidth),
  '--grid-height', String(config.gridHeight)
];
if (flags.includes('--force')) pythonArgs.push('--force');

execFileSync('python3', pythonArgs, { stdio: 'inherit' });
const featureCache = JSON.parse(await fs.readFile(featurePath, 'utf8'));
const trainingStarted = performance.now();
const model = trainClassifier(featureCache.rows, {
  featureVersion: `opencv-${presetName}-v1`
});
const trainingMs = Number((performance.now() - trainingStarted).toFixed(2));
const result = {
  version: 1,
  jobId,
  preset: presetName,
  createdAt: new Date().toISOString(),
  config: featureCache.config,
  featureDimensions: featureCache.featureDimensions,
  extractionSeconds: featureCache.extractionSeconds,
  trainingMs,
  metrics: model.metrics,
  hyperparameters: model.hyperparameters,
  split: model.split,
  modelCheck: {
    matches: model.validationReview.filter(row => row.correct).length,
    total: model.validationReview.length,
    rows: model.validationReview
  }
};
await fs.writeFile(resultPath, JSON.stringify(result, null, 2));
console.log(JSON.stringify(result, null, 2));
