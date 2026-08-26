import fs from 'node:fs/promises';
import path from 'node:path';
import { nanoid } from 'nanoid';
import { createPreviewClip, extractClipFrames, probeVideo } from './ffmpeg.js';
import { computeOpticalFlowFeatures, FLOW_FEATURE_VERSION, FLOW_SAMPLE_FPS } from './features.js';
import { JOBS_DIR, jobDir, jobFile, publicMediaPath, roundTime } from './paths.js';

export async function loadJob(jobId) {
  const filePath = jobFile(jobId);
  try {
    const text = await fs.readFile(filePath, 'utf8');
    return JSON.parse(text);
  } catch (error) {
    if (error.code === 'ENOENT') {
      const notFound = new Error('Job not found.');
      notFound.status = 404;
      throw notFound;
    }
    throw error;
  }
}

export async function saveJob(job) {
  await fs.mkdir(jobDir(job.id), { recursive: true });
  job.updatedAt = new Date().toISOString();
  await fs.writeFile(jobFile(job.id), JSON.stringify(job, null, 2));
  const labeledSamples = job.samples.filter(sample => sample.label).map(sample => ({
    id: sample.id,
    start: sample.start,
    end: sample.end,
    duration: sample.duration,
    label: sample.label
  }));
  await fs.writeFile(path.join(jobDir(job.id), 'labels.json'), JSON.stringify({
    version: 1,
    jobId: job.id,
    source: {
      originalName: job.source.originalName,
      duration: job.source.duration,
      size: job.source.size
    },
    updatedAt: job.updatedAt,
    counts: {
      reviewed: labeledSamples.length,
      keep: labeledSamples.filter(sample => sample.label === 'keep').length,
      cut: labeledSamples.filter(sample => sample.label === 'cut').length,
      unsure: labeledSamples.filter(sample => sample.label === 'unsure').length
    },
    labels: labeledSamples
  }, null, 2));
  return job;
}

function extensionFor(filename) {
  const ext = path.extname(filename || '').toLowerCase();
  return ext && ext.length <= 8 ? ext : '.mp4';
}

export async function createJobFromUpload(file) {
  const id = nanoid(10);
  const dir = path.join(JOBS_DIR, id);
  await fs.mkdir(dir, { recursive: true });

  const extension = extensionFor(file.originalname);
  const inputRelativePath = `original${extension}`;
  const inputPath = path.join(dir, inputRelativePath);
  await fs.rename(file.path, inputPath);

  const metadata = await probeVideo(inputPath);
  const job = {
    id,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    status: 'uploaded',
    source: {
      originalName: file.originalname,
      mimeType: file.mimetype,
      size: file.size,
      relativePath: inputRelativePath,
      ...metadata,
      duration: roundTime(metadata.duration)
    },
    samples: [],
    model: null,
    analysis: null,
    export: null
  };

  await saveJob(job);
  return job;
}

function seededRandom(seed) {
  let value = seed;
  return () => {
    value = (value * 1664525 + 1013904223) % 4294967296;
    return value / 4294967296;
  };
}

function makeSampleStarts(duration, count, clipDuration, seedText, existingStarts = []) {
  const seed = [...seedText].reduce((sum, char) => sum + char.charCodeAt(0), 0) || 1;
  const random = seededRandom(seed);
  const safeDuration = Math.max(clipDuration, duration);
  const maxStart = Math.max(0, safeDuration - clipDuration);
  const existing = new Set(existingStarts.map(start => roundTime(start).toFixed(3)));
  const starts = [];
  let attempts = 0;

  while (starts.length < count && attempts < count * 40) {
    const candidate = roundTime(random() * maxStart);
    const key = candidate.toFixed(3);
    const tooClose = [...existing, ...starts.map(start => start.toFixed(3))].some(value => {
      return Math.abs(Number(value) - candidate) < Math.max(clipDuration * 0.75, 0.75);
    });

    if (!existing.has(key) && !tooClose) starts.push(candidate);
    attempts += 1;
  }

  if (starts.length < count) {
    const bucket = maxStart / count || 0;
    for (let index = 0; starts.length < count && index < count * 2; index += 1) {
      const base = bucket * (index % count) + bucket / 2;
      const jitter = bucket * (random() - 0.5) * 0.6;
      const candidate = roundTime(Math.max(0, Math.min(maxStart, base + jitter)));
      const key = candidate.toFixed(3);
      if (!existing.has(key) && !starts.some(start => start.toFixed(3) === key)) starts.push(candidate);
    }
  }

  return starts.sort((a, b) => a - b);
}

export async function generateSamples(job, options = {}) {
  const count = Math.max(20, Math.min(200, Number(options.count || 200)));
  const clipDuration = Math.max(0.5, Math.min(3, Number(options.clipDuration || 1)));
  const labels = options.labels || {};
  const sourcePath = path.join(jobDir(job.id), job.source.relativePath);
  const previewDir = path.join(jobDir(job.id), 'previews');

  await fs.mkdir(previewDir, { recursive: true });

  const existingSamples = (job.samples || []).map(sample => ({
    ...sample,
    label: labels[sample.id] || sample.label || null
  }));
  const existingStarts = existingSamples.map(sample => sample.start);
  const starts = makeSampleStarts(
    job.source.duration,
    count,
    clipDuration,
    `${job.id}-${existingSamples.length}`,
    existingStarts
  );
  const samples = [...existingSamples];

  for (let index = 0; index < starts.length; index += 1) {
    const id = `sample-${String(samples.length + 1).padStart(3, '0')}`;
    const start = starts[index];
    const end = roundTime(Math.min(job.source.duration, start + clipDuration));
    const clipRelativePath = path.join('previews', `${id}.mp4`);
    const clipPath = path.join(jobDir(job.id), clipRelativePath);
    await createPreviewClip(sourcePath, clipPath, start, Math.max(0.1, end - start));

    samples.push({
      id,
      start,
      end,
      duration: roundTime(end - start),
      clipRelativePath,
      label: null
    });
  }

  job.samples = samples;
  job.model = null;
  job.analysis = null;
  job.export = null;
  job.status = 'samples-ready';
  await saveJob(job);
  return job;
}

export async function ensureOpticalFlowFeatures(job) {
  const sourcePath = path.join(jobDir(job.id), job.source.relativePath);
  const framesRoot = path.join(jobDir(job.id), 'flow-frames');
  await fs.mkdir(framesRoot, { recursive: true });

  for (const sample of job.samples) {
    if (sample.label !== 'keep' && sample.label !== 'cut') continue;
    sample.featureSets ||= {};
    if (sample.featureVersion === FLOW_FEATURE_VERSION && sample.features?.length) {
      sample.featureSets[FLOW_FEATURE_VERSION] ||= sample.features;
      continue;
    }

    const frameDir = path.join(framesRoot, sample.id);
    const framePaths = await extractClipFrames(
      sourcePath,
      frameDir,
      sample.start,
      Math.max(0.1, sample.duration),
      FLOW_SAMPLE_FPS
    );
    sample.features = await computeOpticalFlowFeatures(framePaths);
    sample.featureVersion = FLOW_FEATURE_VERSION;
    sample.featureSets[FLOW_FEATURE_VERSION] = sample.features;
  }

  await saveJob(job);
  return job;
}

export function toPublicJob(job) {
  return {
    ...job,
    labelsUrl: publicMediaPath(job.id, 'labels.json'),
    source: {
      ...job.source,
      url: publicMediaPath(job.id, job.source.relativePath)
    },
    samples: job.samples.map(sample => ({
      id: sample.id,
      start: sample.start,
      end: sample.end,
      duration: sample.duration,
      label: sample.label,
      clipUrl: publicMediaPath(job.id, sample.clipRelativePath)
    })),
    export: job.export
      ? {
          ...job.export,
          url: publicMediaPath(job.id, job.export.relativePath)
        }
      : null
  };
}
