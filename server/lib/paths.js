import fs from 'node:fs/promises';
import path from 'node:path';

export const ROOT_DIR = process.cwd();
export const DATA_DIR = process.env.CLIPPER_DATA_DIR
  ? path.resolve(process.env.CLIPPER_DATA_DIR)
  : path.join(ROOT_DIR, 'data');
export const JOBS_DIR = path.join(DATA_DIR, 'jobs');
export const UPLOADS_DIR = path.join(DATA_DIR, 'uploads');

export async function ensureBaseDirs() {
  await fs.mkdir(JOBS_DIR, { recursive: true });
  await fs.mkdir(UPLOADS_DIR, { recursive: true });
}

export function assertSafeJobId(jobId) {
  if (!/^[A-Za-z0-9_-]+$/.test(jobId)) {
    const error = new Error('Invalid job id.');
    error.status = 400;
    throw error;
  }
}

export function jobDir(jobId) {
  assertSafeJobId(jobId);
  return path.join(JOBS_DIR, jobId);
}

export function jobFile(jobId) {
  return path.join(jobDir(jobId), 'job.json');
}

export function publicMediaPath(jobId, relativePath) {
  return `/media/jobs/${jobId}/${relativePath.split(path.sep).join('/')}`;
}

export function roundTime(value) {
  return Math.max(0, Math.round(Number(value) * 1000) / 1000);
}
