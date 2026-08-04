import fs from 'node:fs/promises';

export const FEATURE_VERSION = 'motion-color-grid-v1';
export const FLOW_FEATURE_VERSION = 'block-optical-flow-v1';
export const FLOW_SAMPLE_FPS = 10;

const WIDTH = 96;
const HEIGHT = 54;
const GRID_W = 8;
const GRID_H = 6;
const CHANNEL_BINS = 4;

function luma(r, g, b) {
  return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
}

function emptyArray(length) {
  return Array.from({ length }, () => 0);
}

async function readFrame(filePath) {
  const file = await fs.readFile(filePath);
  const headerEnd = findPpmHeaderEnd(file);
  const header = file.subarray(0, headerEnd).toString('ascii');
  const tokens = header
    .replace(/#[^\n\r]*/g, '')
    .trim()
    .split(/\s+/);

  if (tokens[0] !== 'P6') {
    throw new Error(`Unsupported frame format in ${filePath}.`);
  }

  const width = Number(tokens[1]);
  const height = Number(tokens[2]);
  const maxValue = Number(tokens[3]);
  const data = file.subarray(headerEnd);
  if (width !== WIDTH || height !== HEIGHT || maxValue !== 255 || data.length < width * height * 3) {
    throw new Error(`Unexpected frame dimensions in ${filePath}.`);
  }

  const pixels = width * height;
  const lumaValues = new Float32Array(pixels);
  const grid = emptyArray(GRID_W * GRID_H);
  const edgeGrid = emptyArray(GRID_W * GRID_H);
  const hist = emptyArray(CHANNEL_BINS * 3);
  const means = [0, 0, 0];
  const variances = [0, 0, 0];
  const counts = emptyArray(GRID_W * GRID_H);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const pixelIndex = y * width + x;
      const dataIndex = pixelIndex * 3;
      const r = data[dataIndex];
      const g = data[dataIndex + 1];
      const b = data[dataIndex + 2];
      const lum = luma(r, g, b);
      lumaValues[pixelIndex] = lum;

      means[0] += r / 255;
      means[1] += g / 255;
      means[2] += b / 255;

      hist[Math.min(CHANNEL_BINS - 1, Math.floor((r / 256) * CHANNEL_BINS))] += 1;
      hist[CHANNEL_BINS + Math.min(CHANNEL_BINS - 1, Math.floor((g / 256) * CHANNEL_BINS))] += 1;
      hist[CHANNEL_BINS * 2 + Math.min(CHANNEL_BINS - 1, Math.floor((b / 256) * CHANNEL_BINS))] += 1;

      const gx = Math.min(GRID_W - 1, Math.floor((x / width) * GRID_W));
      const gy = Math.min(GRID_H - 1, Math.floor((y / height) * GRID_H));
      const cell = gy * GRID_W + gx;
      grid[cell] += lum;
      counts[cell] += 1;
    }
  }

  means[0] /= pixels;
  means[1] /= pixels;
  means[2] /= pixels;

  for (let index = 0; index < pixels; index += 1) {
    const dataIndex = index * 3;
    variances[0] += Math.pow(data[dataIndex] / 255 - means[0], 2);
    variances[1] += Math.pow(data[dataIndex + 1] / 255 - means[1], 2);
    variances[2] += Math.pow(data[dataIndex + 2] / 255 - means[2], 2);
  }

  for (let cell = 0; cell < grid.length; cell += 1) {
    grid[cell] = counts[cell] ? grid[cell] / counts[cell] : 0;
  }

  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const pixelIndex = y * width + x;
      const dx = Math.abs(lumaValues[pixelIndex + 1] - lumaValues[pixelIndex - 1]);
      const dy = Math.abs(lumaValues[pixelIndex + width] - lumaValues[pixelIndex - width]);
      const edge = Math.min(1, dx + dy);
      const gx = Math.min(GRID_W - 1, Math.floor((x / width) * GRID_W));
      const gy = Math.min(GRID_H - 1, Math.floor((y / height) * GRID_H));
      edgeGrid[gy * GRID_W + gx] += edge;
    }
  }

  for (let cell = 0; cell < edgeGrid.length; cell += 1) {
    edgeGrid[cell] = counts[cell] ? edgeGrid[cell] / counts[cell] : 0;
  }

  return {
    lumaValues,
    width,
    height,
    grid,
    edgeGrid,
    colorMeans: means,
    colorStd: variances.map(value => Math.sqrt(value / pixels)),
    hist: hist.map(value => value / pixels)
  };
}

function findPpmHeaderEnd(buffer) {
  let whitespaceRuns = 0;
  let inComment = false;

  for (let index = 0; index < buffer.length; index += 1) {
    const char = buffer[index];
    if (inComment) {
      if (char === 10 || char === 13) inComment = false;
      continue;
    }
    if (char === 35) {
      inComment = true;
      continue;
    }
    if (char === 9 || char === 10 || char === 13 || char === 32) {
      if (index > 0 && !(buffer[index - 1] === 9 || buffer[index - 1] === 10 || buffer[index - 1] === 13 || buffer[index - 1] === 32)) {
        whitespaceRuns += 1;
        if (whitespaceRuns === 4) return index + 1;
      }
    }
  }

  throw new Error('Invalid PPM header.');
}

function meanVectors(vectors) {
  const length = vectors[0].length;
  const out = emptyArray(length);
  for (const vector of vectors) {
    for (let index = 0; index < length; index += 1) out[index] += vector[index];
  }
  return out.map(value => value / vectors.length);
}

function stdVectors(vectors, means) {
  const length = vectors[0].length;
  const out = emptyArray(length);
  for (const vector of vectors) {
    for (let index = 0; index < length; index += 1) {
      out[index] += Math.pow(vector[index] - means[index], 2);
    }
  }
  return out.map(value => Math.sqrt(value / vectors.length));
}

function motionFeatures(frames) {
  if (frames.length < 2) {
    return {
      grid: emptyArray(GRID_W * GRID_H),
      summary: [0, 0, 0, 0]
    };
  }

  const grid = emptyArray(GRID_W * GRID_H);
  const counts = emptyArray(GRID_W * GRID_H);
  const diffs = [];

  for (let pair = 1; pair < frames.length; pair += 1) {
    const previous = frames[pair - 1];
    const current = frames[pair];
    for (let y = 0; y < current.height; y += 1) {
      for (let x = 0; x < current.width; x += 1) {
        const index = y * current.width + x;
        const diff = Math.abs(current.lumaValues[index] - previous.lumaValues[index]);
        diffs.push(diff);
        const gx = Math.min(GRID_W - 1, Math.floor((x / current.width) * GRID_W));
        const gy = Math.min(GRID_H - 1, Math.floor((y / current.height) * GRID_H));
        const cell = gy * GRID_W + gx;
        grid[cell] += diff;
        counts[cell] += 1;
      }
    }
  }

  for (let cell = 0; cell < grid.length; cell += 1) {
    grid[cell] = counts[cell] ? grid[cell] / counts[cell] : 0;
  }

  diffs.sort((a, b) => a - b);
  const mean = diffs.reduce((sum, value) => sum + value, 0) / diffs.length;
  const p90 = diffs[Math.floor(diffs.length * 0.9)] || 0;
  const p98 = diffs[Math.floor(diffs.length * 0.98)] || 0;
  const activeRatio = diffs.filter(value => value > 0.08).length / diffs.length;

  return {
    grid,
    summary: [mean, p90, p98, activeRatio]
  };
}

function gradientMagnitude(frame) {
  const values = new Float32Array(frame.width * frame.height);
  for (let y = 1; y < frame.height - 1; y += 1) {
    for (let x = 1; x < frame.width - 1; x += 1) {
      const index = y * frame.width + x;
      const dx = frame.lumaValues[index + 1] - frame.lumaValues[index - 1];
      const dy = frame.lumaValues[index + frame.width] - frame.lumaValues[index - frame.width];
      values[index] = Math.sqrt(dx * dx + dy * dy);
    }
  }
  return values;
}

function median(values) {
  if (!values.length) return 0;
  const ordered = [...values].sort((a, b) => a - b);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2;
}

function estimateBlockFlow(previous, current) {
  const previousGradient = gradientMagnitude(previous);
  const currentGradient = gradientMagnitude(current);
  const searchRadius = 3;
  const vectors = [];

  for (let gridY = 0; gridY < GRID_H; gridY += 1) {
    for (let gridX = 0; gridX < GRID_W; gridX += 1) {
      const startX = Math.max(searchRadius + 1, Math.floor((gridX / GRID_W) * previous.width));
      const endX = Math.min(previous.width - searchRadius - 1, Math.floor(((gridX + 1) / GRID_W) * previous.width));
      const startY = Math.max(searchRadius + 1, Math.floor((gridY / GRID_H) * previous.height));
      const endY = Math.min(previous.height - searchRadius - 1, Math.floor(((gridY + 1) / GRID_H) * previous.height));
      let texture = 0;
      let sampleCount = 0;

      for (let y = startY; y < endY; y += 2) {
        for (let x = startX; x < endX; x += 2) {
          texture += previousGradient[y * previous.width + x];
          sampleCount += 1;
        }
      }

      texture /= Math.max(1, sampleCount);
      let bestDx = 0;
      let bestDy = 0;
      let bestScore = Number.POSITIVE_INFINITY;
      let stationaryScore = Number.POSITIVE_INFINITY;

      for (let dy = -searchRadius; dy <= searchRadius; dy += 1) {
        for (let dx = -searchRadius; dx <= searchRadius; dx += 1) {
          let score = 0;
          for (let y = startY; y < endY; y += 2) {
            for (let x = startX; x < endX; x += 2) {
              const previousValue = previousGradient[y * previous.width + x];
              const currentValue = currentGradient[(y + dy) * current.width + x + dx];
              score += Math.abs(previousValue - currentValue);
            }
          }
          score /= Math.max(1, sampleCount);
          if (dx === 0 && dy === 0) stationaryScore = score;
          if (score < bestScore) {
            bestScore = score;
            bestDx = dx;
            bestDy = dy;
          }
        }
      }

      const improvement = (stationaryScore - bestScore) / Math.max(0.0001, stationaryScore);
      if (texture < 0.018 || improvement < 0.045) {
        bestDx = 0;
        bestDy = 0;
      }
      vectors.push({ dx: bestDx, dy: bestDy });
    }
  }

  const globalDx = median(vectors.map(vector => vector.dx));
  const globalDy = median(vectors.map(vector => vector.dy));
  return vectors.map(vector => ({
    dx: vector.dx - globalDx,
    dy: vector.dy - globalDy
  }));
}

/**
 * Computes movement-only features. Pixel brightness is used only to estimate
 * displacement; the returned vector contains flow direction and magnitude,
 * never color or absolute luminance.
 */
export async function computeOpticalFlowFeatures(framePaths) {
  if (framePaths.length < 2) {
    throw new Error('Optical flow needs at least two frames.');
  }

  const frames = [];
  for (const framePath of framePaths) {
    frames.push(await readFrame(framePath));
  }
  while (frames.length < FLOW_SAMPLE_FPS) frames.push(frames[frames.length - 1]);
  if (frames.length > FLOW_SAMPLE_FPS) frames.length = FLOW_SAMPLE_FPS;

  const pairFlows = [];
  for (let index = 1; index < frames.length; index += 1) {
    pairFlows.push(estimateBlockFlow(frames[index - 1], frames[index]));
  }

  const searchRadius = 3;
  const cellCount = GRID_W * GRID_H;
  const meanDx = emptyArray(cellCount);
  const meanDy = emptyArray(cellCount);
  const meanMagnitude = emptyArray(cellCount);
  const stdMagnitude = emptyArray(cellCount);
  const maxMagnitude = emptyArray(cellCount);
  const directionHistogram = emptyArray(8);
  const magnitudeHistogram = emptyArray(6);
  const pairActivity = [];
  const pairActiveRatio = [];
  const allMagnitudes = [];

  for (const flow of pairFlows) {
    let activity = 0;
    let activeCells = 0;
    for (let cell = 0; cell < cellCount; cell += 1) {
      const dx = flow[cell].dx / searchRadius;
      const dy = flow[cell].dy / searchRadius;
      const magnitude = Math.min(1, Math.sqrt(dx * dx + dy * dy) / Math.sqrt(2));
      meanDx[cell] += dx;
      meanDy[cell] += dy;
      meanMagnitude[cell] += magnitude;
      maxMagnitude[cell] = Math.max(maxMagnitude[cell], magnitude);
      allMagnitudes.push(magnitude);
      activity += magnitude;
      if (magnitude > 0.12) activeCells += 1;

      if (magnitude > 0) {
        const angle = Math.atan2(dy, dx);
        const directionBin = Math.min(7, Math.floor(((angle + Math.PI) / (Math.PI * 2)) * 8));
        directionHistogram[directionBin] += magnitude;
      }
      magnitudeHistogram[Math.min(5, Math.floor(magnitude * 6))] += 1;
    }
    pairActivity.push(activity / cellCount);
    pairActiveRatio.push(activeCells / cellCount);
  }

  for (let cell = 0; cell < cellCount; cell += 1) {
    meanDx[cell] /= pairFlows.length;
    meanDy[cell] /= pairFlows.length;
    meanMagnitude[cell] /= pairFlows.length;
    for (const flow of pairFlows) {
      const dx = flow[cell].dx / searchRadius;
      const dy = flow[cell].dy / searchRadius;
      const magnitude = Math.min(1, Math.sqrt(dx * dx + dy * dy) / Math.sqrt(2));
      stdMagnitude[cell] += Math.pow(magnitude - meanMagnitude[cell], 2);
    }
    stdMagnitude[cell] = Math.sqrt(stdMagnitude[cell] / pairFlows.length);
  }

  const directionTotal = directionHistogram.reduce((sum, value) => sum + value, 0) || 1;
  const magnitudeTotal = magnitudeHistogram.reduce((sum, value) => sum + value, 0) || 1;
  const orderedMagnitudes = [...allMagnitudes].sort((a, b) => a - b);
  const flowMean = allMagnitudes.reduce((sum, value) => sum + value, 0) / allMagnitudes.length;
  const flowStd = Math.sqrt(
    allMagnitudes.reduce((sum, value) => sum + Math.pow(value - flowMean, 2), 0) / allMagnitudes.length
  );
  const percentile = ratio => orderedMagnitudes[Math.min(orderedMagnitudes.length - 1, Math.floor(orderedMagnitudes.length * ratio))] || 0;

  return [
    ...meanDx,
    ...meanDy,
    ...meanMagnitude,
    ...stdMagnitude,
    ...maxMagnitude,
    ...directionHistogram.map(value => value / directionTotal),
    ...magnitudeHistogram.map(value => value / magnitudeTotal),
    ...pairActivity,
    ...pairActiveRatio,
    flowMean,
    flowStd,
    percentile(0.5),
    percentile(0.75),
    percentile(0.9),
    percentile(0.95),
    orderedMagnitudes.at(-1) || 0,
    allMagnitudes.filter(value => value > 0.12).length / allMagnitudes.length
  ];
}

export async function computeClipFeatures(framePaths) {
  if (!framePaths.length) {
    throw new Error('Cannot compute features without at least one frame.');
  }

  const frames = [];
  for (const framePath of framePaths) {
    frames.push(await readFrame(framePath));
  }
  while (frames.length < 3) frames.push(frames[frames.length - 1]);

  const grids = frames.map(frame => frame.grid);
  const edges = frames.map(frame => frame.edgeGrid);
  const colors = frames.map(frame => frame.colorMeans);
  const colorStds = frames.map(frame => frame.colorStd);
  const hists = frames.map(frame => frame.hist);
  const gridMean = meanVectors(grids);
  const gridStd = stdVectors(grids, gridMean);
  const edgeMean = meanVectors(edges);
  const colorMean = meanVectors(colors);
  const colorStdMean = meanVectors(colorStds);
  const histMean = meanVectors(hists);
  const motion = motionFeatures(frames);

  return [
    ...gridMean,
    ...gridStd,
    ...edgeMean,
    ...motion.grid,
    ...motion.summary,
    ...colorMean,
    ...colorStdMean,
    ...histMean
  ];
}
