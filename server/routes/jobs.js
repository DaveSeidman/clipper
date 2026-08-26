import express from 'express';
import fs from 'node:fs/promises';
import path from 'node:path';
import multer from 'multer';
import { analyzeTimeline, applyManualKeepSegments } from '../lib/analysis.js';
import { exportSegments } from '../lib/ffmpeg.js';
import { trainClassifier } from '../lib/model.js';
import { JOBS_DIR, UPLOADS_DIR, jobDir, roundTime } from '../lib/paths.js';
import { createJobFromUpload, generateSamples, loadJob, saveJob, toPublicJob } from '../lib/jobs.js';
import {
  VIDEOMAE_FEATURE_VERSION,
  ensureVideoMaeSampleEmbeddings,
  extractVideoMaeTimelineEmbeddings,
  isVideoMaeModel
} from '../lib/videomae.js';

const upload = multer({
  dest: UPLOADS_DIR,
  limits: {
    fileSize: 20 * 1024 * 1024 * 1024
  }
});

const router = express.Router();
const MIN_TRAINING_LABELS_PER_CLASS = 40;

function asyncRoute(handler) {
  return async (req, res, next) => {
    try {
      await handler(req, res, next);
    } catch (error) {
      next(error);
    }
  };
}

router.post(
  '/',
  upload.single('video'),
  asyncRoute(async (req, res) => {
    if (!req.file) {
      res.status(400).json({ error: 'Upload a video file.' });
      return;
    }

    const job = await createJobFromUpload(req.file);
    res.status(201).json({ job: toPublicJob(job) });
  })
);

router.get(
  '/latest',
  asyncRoute(async (req, res) => {
    const entries = await fs.readdir(JOBS_DIR, { withFileTypes: true });
    const jobs = await Promise.all(entries
      .filter(entry => entry.isDirectory())
      .map(entry => loadJob(entry.name).catch(() => null)));
    const latest = jobs
      .filter(Boolean)
      .sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt))[0];

    if (!latest) {
      res.status(404).json({ error: 'No saved jobs.' });
      return;
    }
    res.json({ job: toPublicJob(latest) });
  })
);

router.get(
  '/:jobId',
  asyncRoute(async (req, res) => {
    const job = await loadJob(req.params.jobId);
    res.json({ job: toPublicJob(job) });
  })
);

router.post(
  '/:jobId/samples',
  asyncRoute(async (req, res) => {
    const job = await loadJob(req.params.jobId);
    const updated = await generateSamples(job, req.body || {});
    res.json({ job: toPublicJob(updated) });
  })
);

router.put(
  '/:jobId/labels',
  asyncRoute(async (req, res) => {
    const job = await loadJob(req.params.jobId);
    const labels = req.body?.labels || {};
    const validLabels = new Set(['keep', 'cut', 'unsure']);

    for (const sample of job.samples) {
      const nextLabel = labels[sample.id];
      if (validLabels.has(nextLabel)) sample.label = nextLabel;
    }

    await saveJob(job);
    res.json({
      savedAt: job.updatedAt,
      labelsUrl: toPublicJob(job).labelsUrl
    });
  })
);

router.post(
  '/:jobId/train',
  asyncRoute(async (req, res) => {
    const job = await loadJob(req.params.jobId);
    const labels = req.body?.labels || {};

    job.samples = job.samples.map(sample => ({
      ...sample,
      label: labels[sample.id] || sample.label
    }));

    const keepCount = job.samples.filter(sample => sample.label === 'keep').length;
    const cutCount = job.samples.filter(sample => sample.label === 'cut').length;
    if (keepCount < MIN_TRAINING_LABELS_PER_CLASS || cutCount < MIN_TRAINING_LABELS_PER_CLASS) {
      res.status(400).json({
        error: `Training needs at least ${MIN_TRAINING_LABELS_PER_CLASS} keep and ${MIN_TRAINING_LABELS_PER_CLASS} cut examples.`
      });
      return;
    }

    await saveJob(job);
    const embeddingPayload = await ensureVideoMaeSampleEmbeddings(job);
    job.model = trainClassifier(embeddingPayload.rows, {
      featureVersion: VIDEOMAE_FEATURE_VERSION,
      modelType: 'videomae-embedding-logistic-layer',
      backend: 'videomae',
      embedding: embeddingPayload.metadata,
      balanceClasses: true
    });
    job.analysis = null;
    job.export = null;
    job.status = 'trained';
    await saveJob(job);
    res.json({ job: toPublicJob(job) });
  })
);

router.post(
  '/:jobId/analyze',
  asyncRoute(async (req, res) => {
    const job = await loadJob(req.params.jobId);
    if (!job.model) {
      res.status(400).json({ error: 'Train the classifier before analysis.' });
      return;
    }

    const streamsProgress = req.get('accept')?.includes('application/x-ndjson');
    if (streamsProgress) {
      res.status(200);
      res.set({
        'Content-Type': 'application/x-ndjson; charset=utf-8',
        'Cache-Control': 'no-cache, no-transform',
        'X-Accel-Buffering': 'no'
      });
      res.flushHeaders();
    }

    const sendProgress = streamsProgress
      ? progress => res.write(`${JSON.stringify({ type: 'progress', progress })}\n`)
      : undefined;

    try {
      const sourcePath = path.join(jobDir(job.id), job.source.relativePath);
      const embeddingRows = isVideoMaeModel(job.model)
        ? await extractVideoMaeTimelineEmbeddings(job, job.model, req.body || {}, sendProgress)
        : null;
      job.analysis = await analyzeTimeline({
        inputPath: sourcePath,
        jobPath: jobDir(job.id),
        model: job.model,
        duration: job.source.duration,
        options: req.body || {},
        embeddingRows
      });
      job.export = null;
      job.status = 'analyzed';
      await saveJob(job);
      const payload = { job: toPublicJob(job) };
      if (streamsProgress) {
        res.end(`${JSON.stringify({ type: 'result', ...payload })}\n`);
      } else {
        res.json(payload);
      }
    } catch (error) {
      if (!streamsProgress) throw error;
      res.end(`${JSON.stringify({ type: 'error', error: error.message || 'Timeline analysis failed.' })}\n`);
    }
  })
);

router.post(
  '/:jobId/export',
  asyncRoute(async (req, res) => {
    const job = await loadJob(req.params.jobId);
    if (!job.analysis?.keepSegments?.length) {
      res.status(400).json({ error: 'Analyze the video before exporting.' });
      return;
    }

    const requestedFormat = req.body?.format === 'mov' ? 'mov' : 'mp4';
    const relativePath = path.join('exports', `clipper-${job.id}.${requestedFormat}`);
    const outputPath = path.join(jobDir(job.id), relativePath);
    const workDir = path.join(jobDir(job.id), 'export-work');
    const sourcePath = path.join(jobDir(job.id), job.source.relativePath);

    await exportSegments({
      inputPath: sourcePath,
      outputPath,
      workDir,
      segments: job.analysis.keepSegments,
      format: requestedFormat
    });

    const stat = await fs.stat(outputPath);
    job.export = {
      createdAt: new Date().toISOString(),
      relativePath,
      size: stat.size,
      format: requestedFormat,
      duration: roundTime(job.analysis.editedDuration)
    };
    job.status = 'exported';
    await saveJob(job);
    res.json({ job: toPublicJob(job) });
  })
);

router.put(
  '/:jobId/segments',
  asyncRoute(async (req, res) => {
    const job = await loadJob(req.params.jobId);
    if (!job.analysis) {
      res.status(400).json({ error: 'Analyze the video before adjusting segments.' });
      return;
    }

    job.analysis = applyManualKeepSegments(
      job.analysis,
      req.body?.segments,
      job.source.duration
    );
    job.export = null;
    job.status = 'analyzed';
    await saveJob(job);
    res.json({ job: toPublicJob(job) });
  })
);

export default router;
