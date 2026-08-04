import fs from 'node:fs/promises';
import path from 'node:path';
import { runProcess } from './ffmpeg.js';
import { DATA_DIR, ROOT_DIR, jobDir, jobFile } from './paths.js';

export const DEFAULT_VIDEOMAE_MODEL = process.env.CLIPPER_VIDEOMAE_MODEL || 'MCG-NJU/videomae-base';
export const DEFAULT_VIDEOMAE_CONTEXT_SECONDS = Number(process.env.CLIPPER_VIDEOMAE_CONTEXT_SECONDS || 5);
export const DEFAULT_VIDEOMAE_NUM_FRAMES = Number(process.env.CLIPPER_VIDEOMAE_NUM_FRAMES || 16);
export const DEFAULT_VIDEOMAE_IMAGE_SIZE = Number(process.env.CLIPPER_VIDEOMAE_IMAGE_SIZE || 224);
export const VIDEOMAE_FEATURE_VERSION = 'videomae-embedding-v1';

const PYTHON_PATH = process.env.CLIPPER_VIDEOMAE_PYTHON
  ? path.resolve(process.env.CLIPPER_VIDEOMAE_PYTHON)
  : path.join(ROOT_DIR, '.venv-videomae', 'bin', 'python');
const SCRIPT_PATH = path.join(ROOT_DIR, 'server', 'scripts', 'extract-videomae-embeddings.py');
const MODEL_CACHE_DIR = path.join(DATA_DIR, 'models', 'huggingface');

function safeName(value) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
}

function numberOption(value, fallback, minimum, maximum) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(minimum, Math.min(maximum, parsed));
}

async function assertVisionRuntime() {
  try {
    await fs.access(PYTHON_PATH);
  } catch {
    const error = new Error('VideoMAE is not installed. Run `npm run install:vision`, then try again.');
    error.status = 503;
    throw error;
  }
}

async function extractEmbeddings(job, options) {
  await assertVisionRuntime();
  await fs.mkdir(MODEL_CACHE_DIR, { recursive: true });
  await fs.mkdir(path.dirname(options.outputPath), { recursive: true });

  const args = [
    SCRIPT_PATH,
    '--job', jobFile(job.id),
    '--output', options.outputPath,
    '--mode', options.mode,
    '--model-id', options.modelId,
    '--cache-dir', MODEL_CACHE_DIR,
    '--context-seconds', String(options.contextSeconds),
    '--num-frames', String(options.numFrames),
    '--image-size', String(options.imageSize),
    '--batch-size', String(options.batchSize),
    '--decode-workers', String(options.decodeWorkers),
    '--device', process.env.CLIPPER_VIDEOMAE_DEVICE || 'auto'
  ];
  if (options.mode === 'timeline') args.push('--interval', String(options.interval));
  if (process.env.CLIPPER_VIDEOMAE_OFFLINE === '1') args.push('--local-files-only');

  try {
    await runProcess(PYTHON_PATH, args, {
      cwd: ROOT_DIR,
      onStderr: chunk => process.stderr.write(`[videomae] ${chunk}`)
    });
  } catch (error) {
    if (error.stderr?.includes('local-files-only') || error.stderr?.includes('LocalEntryNotFoundError')) {
      error.message = 'The VideoMAE model is not cached locally. Disable offline mode once so it can be downloaded.';
    } else if (error.stderr?.includes('No module named')) {
      error.message = 'VideoMAE dependencies are incomplete. Run `npm run install:vision`.';
    } else {
      error.message = `VideoMAE embedding extraction failed: ${error.stderr?.trim().split('\n').at(-1) || error.message}`;
    }
    error.status = 500;
    throw error;
  }

  return JSON.parse(await fs.readFile(options.outputPath, 'utf8'));
}

function configurationFrom(model = {}) {
  const embedding = model.embedding || {};
  return {
    modelId: embedding.modelId || DEFAULT_VIDEOMAE_MODEL,
    contextSeconds: numberOption(embedding.contextSeconds, DEFAULT_VIDEOMAE_CONTEXT_SECONDS, 1, 15),
    numFrames: numberOption(embedding.numFrames, DEFAULT_VIDEOMAE_NUM_FRAMES, 8, 32),
    imageSize: numberOption(embedding.imageSize, DEFAULT_VIDEOMAE_IMAGE_SIZE, 112, 384),
    batchSize: numberOption(process.env.CLIPPER_VIDEOMAE_BATCH_SIZE, 4, 1, 16),
    decodeWorkers: numberOption(process.env.CLIPPER_VIDEOMAE_DECODE_WORKERS, 4, 1, 8)
  };
}

export function isVideoMaeModel(model) {
  return model?.backend === 'videomae' || model?.featureVersion?.startsWith('videomae-');
}

export async function ensureVideoMaeSampleEmbeddings(job) {
  const config = configurationFrom();
  const outputPath = path.join(
    jobDir(job.id),
    'embeddings',
    `${safeName(config.modelId)}-context${config.contextSeconds}-samples.json`
  );
  const payload = await extractEmbeddings(job, {
    ...config,
    mode: 'samples',
    outputPath
  });
  const featuresById = new Map(payload.rows.map(row => [row.id, row.features]));
  const rows = job.samples
    .filter(sample => sample.label === 'keep' || sample.label === 'cut')
    .map(sample => ({
      id: sample.id,
      label: sample.label,
      features: featuresById.get(sample.id)
    }));

  if (rows.some(row => !row.features?.length)) {
    throw new Error('VideoMAE did not return an embedding for every labeled sample.');
  }

  return {
    rows,
    metadata: {
      ...payload.config,
      device: payload.device,
      embeddingDimension: payload.embeddingDimension
    }
  };
}

export async function extractVideoMaeTimelineEmbeddings(job, model, options = {}) {
  const config = configurationFrom(model);
  const interval = numberOption(options.interval, 1, 0.5, 5);
  const outputPath = path.join(
    jobDir(job.id),
    'embeddings',
    `${safeName(config.modelId)}-context${config.contextSeconds}-timeline-${interval}.json`
  );
  const payload = await extractEmbeddings(job, {
    ...config,
    mode: 'timeline',
    interval,
    outputPath
  });
  return payload.rows.map(row => ({
    start: row.start,
    end: row.end,
    features: row.features
  }));
}
