import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';

function asTime(value) {
  return Number(value).toFixed(3);
}

export function runProcess(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: options.cwd,
      stdio: ['ignore', 'pipe', 'pipe']
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', chunk => {
      const text = chunk.toString();
      stdout += text;
      options.onStdout?.(text);
    });

    child.stderr.on('data', chunk => {
      const text = chunk.toString();
      stderr += text;
      options.onStderr?.(text);
    });

    child.on('error', reject);
    child.on('close', code => {
      if (code === 0) {
        resolve({ stdout, stderr });
        return;
      }

      const error = new Error(`${command} exited with code ${code}`);
      error.stdout = stdout;
      error.stderr = stderr;
      error.code = code;
      reject(error);
    });
  });
}

function parseRate(rate) {
  if (!rate || rate === '0/0') return null;
  const [num, den] = rate.split('/').map(Number);
  if (!den) return Number(rate);
  return num / den;
}

export async function probeVideo(inputPath) {
  const { stdout } = await runProcess('ffprobe', [
    '-v',
    'error',
    '-print_format',
    'json',
    '-show_format',
    '-show_streams',
    inputPath
  ]);
  const data = JSON.parse(stdout);
  const videoStream = data.streams.find(stream => stream.codec_type === 'video');
  const audioStream = data.streams.find(stream => stream.codec_type === 'audio');

  if (!videoStream) {
    const error = new Error('The uploaded file does not contain a video stream.');
    error.status = 400;
    throw error;
  }

  return {
    duration: Number(data.format?.duration || videoStream.duration || 0),
    bitRate: Number(data.format?.bit_rate || 0),
    formatName: data.format?.format_name || '',
    width: Number(videoStream.width || 0),
    height: Number(videoStream.height || 0),
    fps: parseRate(videoStream.avg_frame_rate || videoStream.r_frame_rate),
    codec: videoStream.codec_name,
    hasAudio: Boolean(audioStream),
    audioCodec: audioStream?.codec_name || null
  };
}

export async function createPreviewClip(inputPath, outputPath, start, duration) {
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await runProcess('ffmpeg', [
    '-hide_banner',
    '-loglevel',
    'error',
    '-y',
    '-ss',
    asTime(start),
    '-i',
    inputPath,
    '-t',
    asTime(duration),
    '-map',
    '0:v:0',
    '-an',
    '-vf',
    'scale=-2:360',
    '-c:v',
    'libx264',
    '-preset',
    'veryfast',
    '-crf',
    '25',
    '-movflags',
    '+faststart',
    outputPath
  ]);
}

export async function extractClipFrames(inputPath, outputDir, start, duration, fps = 3) {
  await fs.rm(outputDir, { recursive: true, force: true });
  await fs.mkdir(outputDir, { recursive: true });
  await runProcess('ffmpeg', [
    '-hide_banner',
    '-loglevel',
    'error',
    '-y',
    '-ss',
    asTime(start),
    '-i',
    inputPath,
    '-t',
    asTime(duration),
    '-vf',
    `fps=${fps},scale=96:54`,
    '-pix_fmt',
    'rgb24',
    path.join(outputDir, 'frame_%03d.ppm')
  ]);

  const files = await fs.readdir(outputDir);
  return files
    .filter(file => file.endsWith('.ppm'))
    .sort()
    .map(file => path.join(outputDir, file));
}

export async function extractTimelineFrames(inputPath, outputDir, fps = 3) {
  await fs.rm(outputDir, { recursive: true, force: true });
  await fs.mkdir(outputDir, { recursive: true });
  await runProcess('ffmpeg', [
    '-hide_banner',
    '-loglevel',
    'error',
    '-y',
    '-i',
    inputPath,
    '-vf',
    `fps=${fps},scale=96:54`,
    '-pix_fmt',
    'rgb24',
    path.join(outputDir, 'frame_%06d.ppm')
  ]);

  const files = await fs.readdir(outputDir);
  return files
    .filter(file => file.endsWith('.ppm'))
    .sort()
    .map((file, index) => ({
      path: path.join(outputDir, file),
      time: index / fps
    }));
}

export async function exportSegments({ inputPath, outputPath, workDir, segments, format = 'mp4' }) {
  await fs.rm(workDir, { recursive: true, force: true });
  await fs.mkdir(workDir, { recursive: true });

  const segmentPaths = [];

  for (let index = 0; index < segments.length; index += 1) {
    const segment = segments[index];
    const segmentPath = path.join(workDir, `segment_${String(index).padStart(4, '0')}.${format}`);
    const duration = Math.max(0.05, segment.end - segment.start);

    await runProcess('ffmpeg', [
      '-hide_banner',
      '-loglevel',
      'error',
      '-y',
      '-ss',
      asTime(segment.start),
      '-i',
      inputPath,
      '-t',
      asTime(duration),
      '-map',
      '0:v:0',
      '-map',
      '0:a?',
      '-sn',
      '-c:v',
      'libx264',
      '-preset',
      'veryfast',
      '-crf',
      '20',
      '-c:a',
      'aac',
      '-movflags',
      '+faststart',
      segmentPath
    ]);

    segmentPaths.push(segmentPath);
  }

  const concatPath = path.join(workDir, 'concat.txt');
  const concatBody = segmentPaths
    .map(segmentPath => `file '${segmentPath.replaceAll("'", "'\\''")}'`)
    .join('\n');
  await fs.writeFile(concatPath, concatBody);
  await fs.mkdir(path.dirname(outputPath), { recursive: true });

  await runProcess('ffmpeg', [
    '-hide_banner',
    '-loglevel',
    'error',
    '-y',
    '-f',
    'concat',
    '-safe',
    '0',
    '-i',
    concatPath,
    '-c',
    'copy',
    '-movflags',
    '+faststart',
    outputPath
  ]);
}
