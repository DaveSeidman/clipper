import path from 'node:path';
import {
  computeClipFeatures,
  computeOpticalFlowFeatures,
  FLOW_FEATURE_VERSION,
  FLOW_SAMPLE_FPS
} from './features.js';
import { extractTimelineFrames } from './ffmpeg.js';
import { roundTime } from './paths.js';
import { predictKeepProbability } from './model.js';

function nearestFramePaths(frames, fps, start, duration) {
  const offsets = [0.1, 0.5, 0.9].map(offset => Math.min(duration - 0.02, Math.max(0, offset)));
  const paths = [];
  for (const offset of offsets) {
    const index = Math.max(0, Math.min(frames.length - 1, Math.round((start + offset) * fps)));
    paths.push(frames[index].path);
  }
  return paths;
}

function flowFramePaths(frames, fps, start, duration) {
  return Array.from({ length: FLOW_SAMPLE_FPS }, (_, index) => {
    const offset = ((index + 0.5) / FLOW_SAMPLE_FPS) * duration;
    const frameIndex = Math.max(0, Math.min(frames.length - 1, Math.floor((start + offset) * fps)));
    return frames[frameIndex].path;
  });
}

function movingAverage(items, radius) {
  return items.map((item, index) => {
    let sum = 0;
    let count = 0;
    for (let cursor = Math.max(0, index - radius); cursor <= Math.min(items.length - 1, index + radius); cursor += 1) {
      sum += items[cursor].probability;
      count += 1;
    }
    return {
      ...item,
      smoothedProbability: sum / count
    };
  });
}

function runsToSegments(rows, threshold, interval, duration) {
  const segments = [];
  let active = null;

  for (const row of rows) {
    const keep = row.smoothedProbability >= threshold;
    if (keep && !active) {
      active = {
        start: row.start,
        end: Math.min(duration, row.start + interval),
        probabilities: [row.smoothedProbability]
      };
    } else if (keep && active) {
      active.end = Math.min(duration, row.start + interval);
      active.probabilities.push(row.smoothedProbability);
    } else if (!keep && active) {
      segments.push(active);
      active = null;
    }
  }

  if (active) segments.push(active);
  return segments;
}

function average(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function normalizeSegments(segments) {
  return segments.map(segment => ({
    start: roundTime(segment.start),
    end: roundTime(segment.end),
    duration: roundTime(segment.end - segment.start),
    averageProbability: Number(average(segment.probabilities || [segment.averageProbability || 0]).toFixed(4))
  }));
}

function mergeCloseSegments(segments, mergeGap) {
  if (!segments.length) return [];
  const merged = [segments[0]];

  for (const segment of segments.slice(1)) {
    const previous = merged[merged.length - 1];
    if (segment.start - previous.end <= mergeGap) {
      previous.end = Math.max(previous.end, segment.end);
      previous.probabilities = [...(previous.probabilities || []), ...(segment.probabilities || [])];
    } else {
      merged.push(segment);
    }
  }

  return merged;
}

function complementSegments(keepSegments, duration) {
  const cutSegments = [];
  let cursor = 0;
  for (const segment of keepSegments) {
    if (segment.start > cursor) {
      cutSegments.push({
        start: roundTime(cursor),
        end: roundTime(segment.start),
        duration: roundTime(segment.start - cursor)
      });
    }
    cursor = Math.max(cursor, segment.end);
  }
  if (cursor < duration) {
    cutSegments.push({
      start: roundTime(cursor),
      end: roundTime(duration),
      duration: roundTime(duration - cursor)
    });
  }
  return cutSegments;
}

export function applyManualKeepSegments(analysis, requestedSegments, duration) {
  if (!analysis || !Array.isArray(requestedSegments) || !requestedSegments.length) {
    const error = new Error('Provide at least one keep segment.');
    error.status = 400;
    throw error;
  }

  const sorted = requestedSegments
    .map(segment => ({
      start: Math.max(0, Math.min(duration, Number(segment.start))),
      end: Math.max(0, Math.min(duration, Number(segment.end)))
    }))
    .sort((a, b) => a.start - b.start);

  for (let index = 0; index < sorted.length; index += 1) {
    const segment = sorted[index];
    if (!Number.isFinite(segment.start) || !Number.isFinite(segment.end) || segment.end - segment.start < 0.1) {
      const error = new Error('Each keep segment must have valid start and end times at least 0.1 seconds apart.');
      error.status = 400;
      throw error;
    }
    if (index > 0 && segment.start < sorted[index - 1].end) {
      const error = new Error('Keep segments cannot overlap.');
      error.status = 400;
      throw error;
    }
  }

  const keepSegments = normalizeSegments(sorted.map(segment => ({
    ...segment,
    probabilities: (analysis.rows || [])
      .filter(row => row.start < segment.end && row.end > segment.start)
      .map(row => row.smoothedProbability)
  })));
  const editedDuration = roundTime(keepSegments.reduce((sum, segment) => sum + segment.duration, 0));

  return {
    ...analysis,
    keepSegments,
    cutSegments: complementSegments(keepSegments, duration),
    editedDuration,
    sourceDuration: roundTime(duration),
    keepRatio: duration ? Number((editedDuration / duration).toFixed(4)) : 0,
    manuallyAdjustedAt: new Date().toISOString()
  };
}

export function buildSegments(rows, duration, options = {}) {
  const interval = Number(options.interval || 1);
  const threshold = Number(options.threshold ?? 0.52);
  const smoothingWindow = Math.max(1, Number(options.smoothingWindow || 5));
  const radius = Math.floor(smoothingWindow / 2);
  const mergeGap = Number(options.mergeGap ?? 2.5);
  const minKeepDuration = Number(options.minKeepDuration ?? 2);
  const paddingBefore = Number(options.paddingBefore ?? 1);
  const paddingAfter = Number(options.paddingAfter ?? 1.5);

  const smoothed = movingAverage(rows, radius);
  let keepSegments = runsToSegments(smoothed, threshold, interval, duration);
  keepSegments = keepSegments.filter(segment => segment.end - segment.start >= minKeepDuration);
  keepSegments = mergeCloseSegments(keepSegments, mergeGap);
  keepSegments = keepSegments.map(segment => ({
    ...segment,
    start: Math.max(0, segment.start - paddingBefore),
    end: Math.min(duration, segment.end + paddingAfter)
  }));
  keepSegments = mergeCloseSegments(keepSegments, 0.01);

  const normalizedKeep = normalizeSegments(keepSegments);
  const editedDuration = normalizedKeep.reduce((sum, segment) => sum + segment.duration, 0);

  return {
    options: {
      interval,
      threshold,
      smoothingWindow,
      mergeGap,
      minKeepDuration,
      paddingBefore,
      paddingAfter
    },
    rows: smoothed.map(row => ({
      start: roundTime(row.start),
      end: roundTime(row.end),
      probability: Number(row.probability.toFixed(4)),
      smoothedProbability: Number(row.smoothedProbability.toFixed(4)),
      decision: row.smoothedProbability >= threshold ? 'keep' : 'cut'
    })),
    keepSegments: normalizedKeep,
    cutSegments: complementSegments(normalizedKeep, duration),
    editedDuration: roundTime(editedDuration),
    sourceDuration: roundTime(duration),
    keepRatio: duration ? Number((editedDuration / duration).toFixed(4)) : 0
  };
}

export async function analyzeTimeline({ inputPath, jobPath, model, duration, options = {}, embeddingRows = null }) {
  const interval = Number(options.interval || 1);
  if (embeddingRows?.length) {
    const rows = embeddingRows.map(row => ({
      start: row.start,
      end: row.end,
      probability: predictKeepProbability(model, row.features)
    }));
    return buildSegments(rows, duration, {
      ...options,
      interval
    });
  }

  const useOpticalFlow = model.featureVersion === FLOW_FEATURE_VERSION;
  const fps = useOpticalFlow ? FLOW_SAMPLE_FPS : Number(options.sampleFps || 3);
  const framesDir = path.join(jobPath, 'timeline-frames');
  const frames = await extractTimelineFrames(inputPath, framesDir, fps);
  const rows = [];

  for (let start = 0; start < duration; start += interval) {
    const end = Math.min(duration, start + interval);
    const framePaths = useOpticalFlow
      ? flowFramePaths(frames, fps, start, end - start)
      : nearestFramePaths(frames, fps, start, end - start);
    const features = useOpticalFlow
      ? await computeOpticalFlowFeatures(framePaths)
      : await computeClipFeatures(framePaths);
    const probability = predictKeepProbability(model, features);
    rows.push({
      start,
      end,
      probability,
      features
    });
  }

  return buildSegments(rows, duration, {
    ...options,
    interval
  });
}
