import { FEATURE_VERSION } from './features.js';

function sigmoid(value) {
  if (value < -35) return 0;
  if (value > 35) return 1;
  return 1 / (1 + Math.exp(-value));
}

function dot(a, b) {
  let sum = 0;
  for (let index = 0; index < a.length; index += 1) sum += a[index] * b[index];
  return sum;
}

function summarizeColumns(rows) {
  const length = rows[0].length;
  const mean = Array.from({ length }, () => 0);
  const std = Array.from({ length }, () => 0);

  for (const row of rows) {
    for (let index = 0; index < length; index += 1) mean[index] += row[index];
  }
  for (let index = 0; index < length; index += 1) mean[index] /= rows.length;

  for (const row of rows) {
    for (let index = 0; index < length; index += 1) {
      std[index] += Math.pow(row[index] - mean[index], 2);
    }
  }
  for (let index = 0; index < length; index += 1) {
    std[index] = Math.sqrt(std[index] / rows.length) || 1;
  }

  return { mean, std };
}

function normalize(row, mean, std) {
  return row.map((value, index) => (value - mean[index]) / std[index]);
}

function lossFor(rows, labels, weights, bias, l2, classWeights = { keep: 1, cut: 1 }) {
  let loss = 0;
  let correct = 0;
  let totalWeight = 0;
  for (let row = 0; row < rows.length; row += 1) {
    const probability = sigmoid(dot(rows[row], weights) + bias);
    const clipped = Math.min(0.999999, Math.max(0.000001, probability));
    const sampleWeight = labels[row] ? classWeights.keep : classWeights.cut;
    loss += sampleWeight * (-labels[row] * Math.log(clipped) - (1 - labels[row]) * Math.log(1 - clipped));
    totalWeight += sampleWeight;
    correct += (probability >= 0.5 ? 1 : 0) === labels[row] ? 1 : 0;
  }
  const penalty = weights.reduce((sum, value) => sum + value * value, 0) * l2 * 0.5;
  return {
    loss: loss / Math.max(1, totalWeight) + penalty,
    accuracy: correct / rows.length
  };
}

function hashText(text) {
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash = Math.imul(hash ^ text.charCodeAt(index), 16777619);
  }
  return hash >>> 0;
}

function stratifiedSplit(trainingRows, validationRatio) {
  const validationRows = [];

  for (const label of ['cut', 'keep']) {
    const group = trainingRows
      .map((row, index) => ({ row, index }))
      .filter(item => item.row.label === label)
      .sort((a, b) => {
        const scoreA = hashText(`${a.row.id || a.index}:${label}`);
        const scoreB = hashText(`${b.row.id || b.index}:${label}`);
        return scoreA - scoreB || a.index - b.index;
      });
    const validationCount = group.length >= 5
      ? Math.min(group.length - 1, Math.max(1, Math.floor(group.length * validationRatio)))
      : 0;
    validationRows.push(...group.slice(0, validationCount).map(item => item.row));
  }

  const validationSet = new Set(validationRows);
  return {
    training: trainingRows.filter(row => !validationSet.has(row)),
    validation: validationRows
  };
}

function initializeWeights(featureCount) {
  return Array.from({ length: featureCount }, (_, index) => {
    const phase = ((index * 9301 + 49297) % 233280) / 233280;
    return (phase - 0.5) * 0.02;
  });
}

function selectAuditRows(validationRows) {
  return ['cut', 'keep'].flatMap(label =>
    validationRows
      .filter(row => row.label === label)
      .sort((a, b) => hashText(`${a.id}:model-review`) - hashText(`${b.id}:model-review`))
      .slice(0, 5)
  );
}

export function trainClassifier(trainingRows, options = {}) {
  const allLabels = trainingRows.map(row => (row.label === 'keep' ? 1 : 0));
  const keepCount = allLabels.filter(Boolean).length;
  const cutCount = allLabels.length - keepCount;

  if (!keepCount || !cutCount) {
    const error = new Error('Training needs both keep and cut examples.');
    error.status = 400;
    throw error;
  }

  const validationRatio = Math.max(0, Math.min(0.4, Number(options.validationRatio ?? 0.2)));
  const split = stratifiedSplit(trainingRows, validationRatio);
  const auditRows = selectAuditRows(split.validation);
  const auditSet = new Set(auditRows);
  const refitSourceRows = trainingRows.filter(row => !auditSet.has(row));
  const labels = split.training.map(row => (row.label === 'keep' ? 1 : 0));
  const features = split.training.map(row => row.features);
  const validationLabels = split.validation.map(row => (row.label === 'keep' ? 1 : 0));
  const searchNormalization = summarizeColumns(features);
  const rows = features.map(feature => normalize(feature, searchNormalization.mean, searchNormalization.std));
  const validationRows = split.validation.map(row => normalize(row.features, searchNormalization.mean, searchNormalization.std));
  const trainingKeepCount = labels.filter(Boolean).length;
  const trainingCutCount = labels.length - trainingKeepCount;
  const balanceClasses = Boolean(options.balanceClasses);
  const classWeights = balanceClasses
    ? {
        keep: labels.length / (2 * trainingKeepCount),
        cut: labels.length / (2 * trainingCutCount)
      }
    : { keep: 1, cut: 1 };
  const featureCount = rows[0].length;
  let weights = initializeWeights(featureCount);

  let bias = Math.log(trainingKeepCount / trainingCutCount);
  const epochs = Number(options.epochs || 6000);
  const baseLearningRate = Number(options.learningRate || 0.06);
  const l2 = Number(options.l2 || 0.0025);
  const earlyStoppingPatience = Number(options.earlyStoppingPatience || 600);
  const minDelta = Number(options.minDelta || 0.00001);
  const history = [];
  let bestWeights = [...weights];
  let bestBias = bias;
  let bestEpoch = 1;
  let bestMonitoredLoss = Number.POSITIVE_INFINITY;
  let bestValidationAccuracy = Number.NEGATIVE_INFINITY;
  let epochsRun = 0;

  for (let epoch = 1; epoch <= epochs; epoch += 1) {
    const learningRate = baseLearningRate * (1 - epoch / (epochs * 1.5));
    const gradWeights = Array.from({ length: featureCount }, () => 0);
    let gradBias = 0;
    let totalWeight = 0;

    for (let row = 0; row < rows.length; row += 1) {
      const probability = sigmoid(dot(rows[row], weights) + bias);
      const sampleWeight = labels[row] ? classWeights.keep : classWeights.cut;
      const error = (probability - labels[row]) * sampleWeight;
      totalWeight += sampleWeight;
      gradBias += error;
      for (let feature = 0; feature < featureCount; feature += 1) {
        gradWeights[feature] += error * rows[row][feature];
      }
    }

    for (let feature = 0; feature < featureCount; feature += 1) {
      const gradient = gradWeights[feature] / totalWeight + l2 * weights[feature];
      weights[feature] -= learningRate * gradient;
    }
    bias -= learningRate * (gradBias / totalWeight);
    epochsRun = epoch;

    if (epoch === 1 || epoch % 25 === 0 || epoch === epochs) {
      const trainingSnapshot = lossFor(rows, labels, weights, bias, l2, classWeights);
      const validationSnapshot = validationRows.length
        ? lossFor(validationRows, validationLabels, weights, bias, 0, classWeights)
        : null;
      const monitoredLoss = validationSnapshot?.loss ?? trainingSnapshot.loss;
      const historyPoint = {
        epoch,
        loss: Number(trainingSnapshot.loss.toFixed(5)),
        accuracy: Number(trainingSnapshot.accuracy.toFixed(4))
      };
      if (validationSnapshot) {
        historyPoint.validationLoss = Number(validationSnapshot.loss.toFixed(5));
        historyPoint.validationAccuracy = Number(validationSnapshot.accuracy.toFixed(4));
      }
      history.push(historyPoint);

      const improved = validationSnapshot
        ? validationSnapshot.accuracy > bestValidationAccuracy
          || (validationSnapshot.accuracy === bestValidationAccuracy && monitoredLoss < bestMonitoredLoss - minDelta)
        : monitoredLoss < bestMonitoredLoss - minDelta;

      if (improved) {
        bestMonitoredLoss = monitoredLoss;
        if (validationSnapshot) bestValidationAccuracy = validationSnapshot.accuracy;
        bestWeights = [...weights];
        bestBias = bias;
        bestEpoch = epoch;
      } else if (epoch - bestEpoch >= earlyStoppingPatience) {
        break;
      }
    }
  }

  const validationMetrics = validationRows.length
    ? lossFor(validationRows, validationLabels, bestWeights, bestBias, 0, classWeights)
    : null;
  const refitLabels = refitSourceRows.map(row => (row.label === 'keep' ? 1 : 0));
  const refitKeepCount = refitLabels.filter(Boolean).length;
  const refitCutCount = refitLabels.length - refitKeepCount;
  const { mean, std } = summarizeColumns(refitSourceRows.map(row => row.features));
  const refitRows = refitSourceRows.map(row => normalize(row.features, mean, std));
  const finalClassWeights = balanceClasses
    ? {
        keep: refitLabels.length / (2 * refitKeepCount),
        cut: refitLabels.length / (2 * refitCutCount)
      }
    : { keep: 1, cut: 1 };

  // Validation chooses the stopping point; the deployable model then gets to
  // learn from every non-audit example for exactly that many steps.
  weights = initializeWeights(featureCount);
  bias = Math.log(refitKeepCount / refitCutCount);
  for (let epoch = 1; epoch <= bestEpoch; epoch += 1) {
    const learningRate = baseLearningRate * (1 - epoch / (epochs * 1.5));
    const gradWeights = Array.from({ length: featureCount }, () => 0);
    let gradBias = 0;
    let totalWeight = 0;

    for (let row = 0; row < refitRows.length; row += 1) {
      const probability = sigmoid(dot(refitRows[row], weights) + bias);
      const sampleWeight = refitLabels[row] ? finalClassWeights.keep : finalClassWeights.cut;
      const error = (probability - refitLabels[row]) * sampleWeight;
      totalWeight += sampleWeight;
      gradBias += error;
      for (let feature = 0; feature < featureCount; feature += 1) {
        gradWeights[feature] += error * refitRows[row][feature];
      }
    }

    for (let feature = 0; feature < featureCount; feature += 1) {
      const gradient = gradWeights[feature] / totalWeight + l2 * weights[feature];
      weights[feature] -= learningRate * gradient;
    }
    bias -= learningRate * (gradBias / totalWeight);
  }

  const metrics = lossFor(refitRows, refitLabels, weights, bias, l2, finalClassWeights);
  const validationReview = auditRows.map(row => {
    const keepProbability = sigmoid(dot(normalize(row.features, mean, std), weights) + bias);
    const predictedLabel = keepProbability >= 0.5 ? 'keep' : 'cut';
    return {
      id: row.id,
      actualLabel: row.label,
      predictedLabel,
      keepProbability: Number(keepProbability.toFixed(4)),
      confidence: Number((predictedLabel === 'keep' ? keepProbability : 1 - keepProbability).toFixed(4)),
      correct: predictedLabel === row.label
    };
  });

  return {
    type: options.modelType || (options.featureVersion ? 'regularized-logistic-flow-layer' : 'regularized-logistic-top-layer'),
    ...(options.backend ? { backend: options.backend } : {}),
    ...(options.embedding ? { embedding: options.embedding } : {}),
    featureVersion: options.featureVersion || FEATURE_VERSION,
    createdAt: new Date().toISOString(),
    classes: ['cut', 'keep'],
    counts: {
      keep: keepCount,
      cut: cutCount
    },
    normalization: { mean, std },
    weights,
    bias,
    hyperparameters: {
      epochs,
      epochsRun,
      bestEpoch,
      refitEpochs: bestEpoch,
      learningRate: baseLearningRate,
      l2,
      balanceClasses,
      validationRatio,
      earlyStoppingPatience
    },
    split: {
      training: split.training.length,
      validation: split.validation.length,
      refit: refitSourceRows.length,
      audit: auditRows.length
    },
    validationReview,
    history,
    metrics: {
      loss: Number(metrics.loss.toFixed(5)),
      accuracy: Number(metrics.accuracy.toFixed(4)),
      validationLoss: validationMetrics ? Number(validationMetrics.loss.toFixed(5)) : null,
      validationAccuracy: validationMetrics ? Number(validationMetrics.accuracy.toFixed(4)) : null
    }
  };
}

export function predictKeepProbability(model, features) {
  const normalized = normalize(features, model.normalization.mean, model.normalization.std);
  return sigmoid(dot(normalized, model.weights) + model.bias);
}

export function scoreRows(model, rows) {
  return rows.map(row => ({
    ...row,
    probability: predictKeepProbability(model, row.features)
  }));
}
