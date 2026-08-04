import fs from 'node:fs/promises';
import { execFileSync } from 'node:child_process';

const API_BASE = process.env.CLIPPER_API || 'http://127.0.0.1:3030';
const videoPath = '/private/tmp/clipper-smoke.mp4';

function createSmokeVideo() {
  execFileSync('ffmpeg', [
    '-hide_banner',
    '-loglevel',
    'error',
    '-y',
    '-f',
    'lavfi',
    '-i',
    'color=c=black:size=640x360:rate=30:duration=10',
    '-f',
    'lavfi',
    '-i',
    'testsrc2=size=640x360:rate=30:duration=10',
    '-filter_complex',
    '[0:v][1:v]concat=n=2:v=1:a=0,format=yuv420p[v]',
    '-map',
    '[v]',
    '-c:v',
    'libx264',
    '-movflags',
    '+faststart',
    videoPath
  ]);
}

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(`${path} failed: ${payload.error || response.statusText}`);
  }
  return payload;
}

async function main() {
  createSmokeVideo();
  const form = new FormData();
  const blob = new Blob([await fs.readFile(videoPath)], { type: 'video/mp4' });
  form.append('video', blob, 'clipper-smoke.mp4');

  let payload = await request('/api/jobs', {
    method: 'POST',
    body: form
  });

  const jobId = payload.job.id;
  payload = await request(`/api/jobs/${jobId}/samples`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ count: 20, clipDuration: 1 })
  });

  const partialLabels = Object.fromEntries(
    payload.job.samples.slice(0, 6).map((sample, index) => [sample.id, index < 3 ? 'cut' : 'keep'])
  );
  const initialSampleCount = payload.job.samples.length;

  payload = await request(`/api/jobs/${jobId}/samples`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ count: 120, clipDuration: 1, labels: partialLabels })
  });

  if (payload.job.samples.length <= initialSampleCount) {
    throw new Error('Appending samples did not increase the training set.');
  }

  for (const [sampleId, label] of Object.entries(partialLabels)) {
    const sample = payload.job.samples.find(item => item.id === sampleId);
    if (sample?.label !== label) {
      throw new Error('Appending samples did not preserve existing labels.');
    }
  }

  const labels = Object.fromEntries(
    payload.job.samples.map(sample => [sample.id, sample.start < 10 ? 'cut' : 'keep'])
  );
  const keepCount = Object.values(labels).filter(label => label === 'keep').length;
  const cutCount = Object.values(labels).filter(label => label === 'cut').length;
  if (keepCount < 50 || cutCount < 50) {
    throw new Error(`Smoke labels are imbalanced: ${keepCount} keep / ${cutCount} cut`);
  }

  payload = await request(`/api/jobs/${jobId}/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ labels })
  });

  if (payload.job.model.backend !== 'videomae') {
    throw new Error(`Unexpected model backend: ${payload.job.model.backend}`);
  }
  if (payload.job.model.metrics.accuracy < 0.85) {
    throw new Error(`Model accuracy too low: ${payload.job.model.metrics.accuracy}`);
  }

  payload = await request(`/api/jobs/${jobId}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      threshold: 0.5,
      smoothingWindow: 1,
      minKeepDuration: 1,
      mergeGap: 1,
      paddingBefore: 0,
      paddingAfter: 0
    })
  });

  if (!payload.job.analysis.keepSegments.length) {
    throw new Error('Analysis produced no keep segments.');
  }

  payload = await request(`/api/jobs/${jobId}/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ format: 'mp4' })
  });

  const download = await fetch(`${API_BASE}${payload.job.export.url}`);
  if (!download.ok) {
    throw new Error('Export download check failed.');
  }

  console.log(`Smoke test passed for job ${jobId}.`);
}

main().catch(error => {
  console.error(error.message);
  process.exit(1);
});
