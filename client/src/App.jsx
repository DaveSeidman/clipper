import {
  Check,
  CircleHelp,
  Download,
  Film,
  GraduationCap,
  Minus,
  Pause,
  Play,
  Plus,
  RefreshCcw,
  RotateCcw,
  Save,
  Scissors,
  SlidersHorizontal,
  Upload
} from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import {
  analyzeJob,
  exportJob,
  generateSamples,
  loadJob,
  loadLatestJob,
  saveLabels,
  saveSegments,
  trainJob,
  uploadVideo
} from './api.js';

const DEFAULT_ANALYSIS = {
  interval: 1,
  sampleFps: 3,
  threshold: 0.52,
  smoothingWindow: 5,
  mergeGap: 2.5,
  minKeepDuration: 2,
  paddingBefore: 1,
  paddingAfter: 1.5
};

const TRAINING_SAMPLE_COUNT = 200;
const LABEL_TARGET = 40;
const LAST_JOB_STORAGE_KEY = 'clipper:last-job-id';
const START_FRESH_STORAGE_KEY = 'clipper:start-fresh';

function formatTime(seconds = 0) {
  const safe = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safe / 60);
  const wholeSeconds = Math.floor(safe % 60);
  const tenths = Math.floor((safe % 1) * 10);
  return `${minutes}:${String(wholeSeconds).padStart(2, '0')}.${tenths}`;
}

function formatSize(bytes = 0) {
  if (!bytes) return '0 MB';
  const mb = bytes / 1024 / 1024;
  return `${mb.toFixed(mb >= 100 ? 0 : 1)} MB`;
}

function metric(value, fallback = 'n/a') {
  return Number.isFinite(value) ? value : fallback;
}

function shuffleSamples(samples) {
  let seed = samples.reduce((hash, sample) => {
    const key = `${sample.id}:${sample.start}`;
    for (let index = 0; index < key.length; index += 1) {
      hash = Math.imul(hash ^ key.charCodeAt(index), 16777619);
    }
    return hash;
  }, 2166136261) >>> 0;

  const shuffled = [...samples];
  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    const swapIndex = seed % (index + 1);
    [shuffled[index], shuffled[swapIndex]] = [shuffled[swapIndex], shuffled[index]];
  }
  return shuffled;
}

function App() {
  const [job, setJob] = useState(null);
  const [restoringJob, setRestoringJob] = useState(true);
  const [labels, setLabels] = useState({});
  const [busy, setBusy] = useState(null);
  const [loadingMoreSamples, setLoadingMoreSamples] = useState(false);
  const [error, setError] = useState('');
  const [analysisOptions, setAnalysisOptions] = useState(DEFAULT_ANALYSIS);
  const [exportFormat, setExportFormat] = useState('mp4');
  const [analysisDirty, setAnalysisDirty] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(null);

  const previousJobIdRef = useRef(null);
  const autoGenerationAttemptRef = useRef(null);
  const labelSaveQueueRef = useRef(Promise.resolve());

  useEffect(() => {
    let active = true;

    async function restoreLastJob() {
      try {
        if (window.localStorage.getItem(START_FRESH_STORAGE_KEY) === '1') return;
        const storedJobId = window.localStorage.getItem(LAST_JOB_STORAGE_KEY);
        let payload = null;
        if (storedJobId) {
          payload = await loadJob(storedJobId).catch(() => null);
        }
        payload ||= await loadLatestJob();
        if (!active) return;
        setJob(payload.job);
        window.localStorage.setItem(LAST_JOB_STORAGE_KEY, payload.job.id);
      } catch (taskError) {
        if (active && taskError.message !== 'No saved jobs.') setError(taskError.message);
      } finally {
        if (active) setRestoringJob(false);
      }
    }

    restoreLastJob();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!job?.samples?.length) return;
    const serverLabels = Object.fromEntries(job.samples.filter(sample => sample.label).map(sample => [sample.id, sample.label]));
    if (previousJobIdRef.current !== job.id) {
      previousJobIdRef.current = job.id;
      setLabels(serverLabels);
      return;
    }

    const sampleIds = new Set(job.samples.map(sample => sample.id));
    setLabels(current => {
      const merged = { ...serverLabels };
      for (const [sampleId, label] of Object.entries(current)) {
        if (sampleIds.has(sampleId)) merged[sampleId] = label;
      }
      return merged;
    });
  }, [job?.id, job?.samples]);

  useEffect(() => {
    if (!analysisDirty) return undefined;
    const warnBeforeUnload = event => {
      event.preventDefault();
      event.returnValue = '';
    };
    window.addEventListener('beforeunload', warnBeforeUnload);
    return () => window.removeEventListener('beforeunload', warnBeforeUnload);
  }, [analysisDirty]);

  const reviewedCount = Object.keys(labels).length;
  const keepCount = Object.values(labels).filter(label => label === 'keep').length;
  const cutCount = Object.values(labels).filter(label => label === 'cut').length;
  const unsureCount = Object.values(labels).filter(label => label === 'unsure').length;
  const canTrain = keepCount >= LABEL_TARGET && cutCount >= LABEL_TARGET;

  useEffect(() => {
    if (!job?.samples?.length || loadingMoreSamples) return;

    const remainingCount = Math.max(0, job.samples.length - reviewedCount);
    const refillThreshold = Math.max(10, Math.ceil(job.samples.length * 0.1));
    const needsMoreClassLabels = keepCount < LABEL_TARGET || cutCount < LABEL_TARGET;
    if (job.model && !needsMoreClassLabels) return;
    if (!needsMoreClassLabels || remainingCount > refillThreshold) return;

    const attemptKey = `${job.id}:${job.samples.length}:${reviewedCount}`;
    if (autoGenerationAttemptRef.current === attemptKey) return;
    autoGenerationAttemptRef.current = attemptKey;

    setLoadingMoreSamples(true);
    setError('');
    generateSamples(job.id, {
      count: TRAINING_SAMPLE_COUNT,
      clipDuration: 1,
      labels
    })
      .then(payload => setJob(payload.job))
      .catch(taskError => setError(taskError.message))
      .finally(() => setLoadingMoreSamples(false));
  }, [job?.id, job?.samples?.length, job?.model, reviewedCount, keepCount, cutCount, labels, loadingMoreSamples]);

  async function run(label, task) {
    setBusy(label);
    setError('');
    try {
      await task();
    } catch (taskError) {
      setError(taskError.message);
    } finally {
      setBusy(null);
    }
  }

  function setSampleLabel(sampleId, label) {
    setLabels(current => ({ ...current, [sampleId]: label }));
    if (job?.id) {
      labelSaveQueueRef.current = labelSaveQueueRef.current
        .then(() => saveLabels(job.id, { [sampleId]: label }))
        .catch(taskError => setError(`Label save failed: ${taskError.message}`));
    }
  }

  function startOver() {
    const confirmed = window.confirm(
      `Start over with a new video? Your current job and saved labels will remain available on this computer.${
        analysisDirty ? ' Unsaved segment adjustments will be discarded.' : ''
      }`
    );
    if (!confirmed) return;

    window.localStorage.removeItem(LAST_JOB_STORAGE_KEY);
    window.localStorage.setItem(START_FRESH_STORAGE_KEY, '1');
    previousJobIdRef.current = null;
    autoGenerationAttemptRef.current = null;
    setJob(null);
    setLabels({});
    setLoadingMoreSamples(false);
    setError('');
    setAnalysisOptions(DEFAULT_ANALYSIS);
    setExportFormat('mp4');
    setAnalysisDirty(false);
    setAnalysisProgress(null);
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <Film size={22} />
          <div>
            <h1>Clipper</h1>
            <span>{job ? job.source.originalName : 'Local video editor'}</span>
          </div>
        </div>
        <div className="topbar-actions">
          <div className="status-strip">
            <StatusPill active={Boolean(job)} label="Video" />
            <StatusPill active={Boolean(job?.samples?.length)} label="Samples" />
            <StatusPill active={Boolean(job?.model)} label="Model" />
            <StatusPill active={Boolean(job?.analysis)} label="Preview" />
            <StatusPill active={Boolean(job?.export)} label="Export" />
          </div>
          {job ? (
            <button className="secondary-button start-over-button" type="button" disabled={Boolean(busy)} onClick={startOver}>
              <RotateCcw size={15} />
              Start over
            </button>
          ) : null}
        </div>
      </header>

      {error ? <div className="error-line">{error}</div> : null}

      {restoringJob ? (
        <section className="upload-panel" role="status">
          <RefreshCcw size={28} className="spin" />
          <strong>Restoring your last session</strong>
        </section>
      ) : !job ? (
        <UploadPanel
          busy={busy}
          onUpload={file =>
            run('Uploading', async () => {
              const uploadPayload = await uploadVideo(file);
              setJob(uploadPayload.job);
              setAnalysisDirty(false);
              window.localStorage.removeItem(START_FRESH_STORAGE_KEY);
              window.localStorage.setItem(LAST_JOB_STORAGE_KEY, uploadPayload.job.id);
              setBusy('Creating training samples');
              const samplesPayload = await generateSamples(uploadPayload.job.id, {
                count: TRAINING_SAMPLE_COUNT,
                clipDuration: 1
              });
              setJob(samplesPayload.job);
            })
          }
        />
      ) : (
        <div className="workspace">
          <aside className="side-panel">
            <ActionPanel
              job={job}
              busy={busy}
              loadingMoreSamples={loadingMoreSamples}
              canTrain={canTrain}
              labels={{ reviewedCount, keepCount, cutCount, unsureCount }}
              analysisOptions={analysisOptions}
              setAnalysisOptions={setAnalysisOptions}
              exportFormat={exportFormat}
              setExportFormat={setExportFormat}
              segmentsDirty={analysisDirty}
              onTrain={() =>
                run('Training', async () => {
                  const payload = await trainJob(job.id, labels);
                  setJob(payload.job);
                  setAnalysisDirty(false);
                })
              }
              onAnalyze={() =>
                run('Analyzing timeline', async () => {
                  setAnalysisProgress({ completed: 0, total: null });
                  try {
                    const payload = await analyzeJob(job.id, analysisOptions, setAnalysisProgress);
                    setJob(payload.job);
                    setAnalysisDirty(false);
                  } finally {
                    setAnalysisProgress(null);
                  }
                })
              }
              onExport={() =>
                run('Exporting', async () => {
                  const payload = await exportJob(job.id, exportFormat);
                  setJob(payload.job);
                })
              }
            />
          </aside>

          <section className="main-panel">
            {job.analysis ? (
              <PreviewStage
                job={job}
                dirty={analysisDirty}
                busy={Boolean(busy)}
                onDirtyChange={setAnalysisDirty}
                onSaveSegments={segments =>
                  run('Saving segment changes', async () => {
                    const payload = await saveSegments(job.id, segments);
                    setJob(payload.job);
                    setAnalysisDirty(false);
                  })
                }
              />
            ) : job.samples?.length ? (
              <TrainingStage job={job} labels={labels} onLabel={setSampleLabel} />
            ) : (
              <EmptyStage />
            )}
          </section>
        </div>
      )}

      {busy ? (
        <div className="busy-overlay" role="status">
          <div className="busy-panel">
            <RefreshCcw size={20} className="spin" />
            <div className="busy-content">
              <span>{busy}</span>
              {busy === 'Analyzing timeline' ? <AnalysisProgress progress={analysisProgress} /> : null}
            </div>
          </div>
        </div>
      ) : null}
    </main>
  );
}

function AnalysisProgress({ progress }) {
  const hasTotal = Number.isFinite(progress?.total) && progress.total > 0;
  const percentage = hasTotal
    ? Math.min(100, Math.round((progress.completed / progress.total) * 100))
    : 0;
  return (
    <div className="analysis-progress-wrap">
      <div
        className={`analysis-progress${hasTotal ? '' : ' indeterminate'}`}
        role="progressbar"
        aria-label="Timeline embedding progress"
        aria-valuemin="0"
        aria-valuemax={hasTotal ? progress.total : undefined}
        aria-valuenow={hasTotal ? progress.completed : undefined}
      >
        <span style={hasTotal ? { width: `${percentage}%` } : undefined} />
      </div>
      <small>
        {hasTotal
          ? `Embedded ${progress.completed.toLocaleString()} of ${progress.total.toLocaleString()} windows · ${percentage}%`
          : 'Preparing timeline embeddings…'}
      </small>
    </div>
  );
}

function StatusPill({ active, label }) {
  return <span className={active ? 'status-pill active' : 'status-pill'}>{label}</span>;
}

function UploadPanel({ onUpload, busy }) {
  return (
    <section className="upload-panel">
      <label className="drop-zone" aria-disabled={Boolean(busy)}>
        <Upload size={32} />
        <input
          type="file"
          accept="video/*,.mov,.mp4,.m4v"
          disabled={Boolean(busy)}
          onChange={event => {
            const file = event.target.files?.[0];
            event.target.value = '';
            if (file) onUpload(file);
          }}
        />
        <strong>Choose a video to begin</strong>
        <span>Uploads automatically · MOV, MP4, M4V</span>
      </label>
    </section>
  );
}

function SourcePanel({ job, compact = false }) {
  return (
    <section className={`panel-block source-block${compact ? ' compact' : ''}`}>
      <h2>Source</h2>
      <dl>
        <div>
          <dt>Duration</dt>
          <dd>{formatTime(job.source.duration)}</dd>
        </div>
        <div>
          <dt>Frame</dt>
          <dd>
            {job.source.width}x{job.source.height}
          </dd>
        </div>
        <div>
          <dt>FPS</dt>
          <dd>{job.source.fps ? job.source.fps.toFixed(2) : 'n/a'}</dd>
        </div>
        <div>
          <dt>Size</dt>
          <dd>{formatSize(job.source.size)}</dd>
        </div>
      </dl>
    </section>
  );
}

function ActionPanel({
  job,
  busy,
  loadingMoreSamples,
  canTrain,
  labels,
  analysisOptions,
  setAnalysisOptions,
  exportFormat,
  setExportFormat,
  segmentsDirty,
  onTrain,
  onAnalyze,
  onExport
}) {
  return (
    <section className="panel-block actions-block">
      <h2>Labeling</h2>
      <div className="label-goals">
        <LabelGoal label="Keep" count={labels.keepCount} tone="keep" />
        <LabelGoal label="Cut" count={labels.cutCount} tone="cut" />
        <div className="review-summary">
          <span>{labels.reviewedCount}/{job.samples?.length || 0} reviewed</span>
          <span>{labels.unsureCount} unsure</span>
        </div>
        {loadingMoreSamples ? (
          <div className="sample-loading" role="status">
            <RefreshCcw size={14} className="spin" />
            Adding more clips…
          </div>
        ) : null}
        {job.labelsUrl ? (
          <a className="label-download" href={job.labelsUrl} download>
            <Download size={14} />
            Download labels JSON
          </a>
        ) : null}
      </div>

      <button className="primary-button" disabled={!canTrain || Boolean(busy) || loadingMoreSamples || segmentsDirty} onClick={onTrain}>
        <GraduationCap size={18} />
        Train model
      </button>

      {job.model ? (
        <div className="training-results">
          <LossSparkline history={job.model.history} value={job.model.metrics.validationLoss} />
          <div className="metrics">
            <div>
              <span>Backend</span>
              <strong>{job.model.backend === 'videomae' ? 'VideoMAE' : 'Motion'}</strong>
            </div>
            <div>
              <span>Loss</span>
              <strong>{job.model.metrics.loss}</strong>
            </div>
            <div>
              <span>Train Acc</span>
              <strong>{Math.round(job.model.metrics.accuracy * 100)}%</strong>
            </div>
            {Number.isFinite(job.model.metrics.validationAccuracy) ? (
              <div>
                <span>Val Acc</span>
                <strong>{Math.round(job.model.metrics.validationAccuracy * 100)}%</strong>
              </div>
            ) : null}
            {job.model.hyperparameters?.bestEpoch ? (
              <div>
                <span>Best Step</span>
                <strong>{job.model.hyperparameters.bestEpoch}</strong>
              </div>
            ) : null}
          </div>
        </div>
      ) : null}

      <button className="primary-button" disabled={!job.model || Boolean(busy) || segmentsDirty} onClick={onAnalyze}>
        <SlidersHorizontal size={17} />
        Analyze
      </button>

      <AnalysisControls options={analysisOptions} onChange={setAnalysisOptions} disabled={Boolean(busy)} />

      <div className="export-row">
        <select value={exportFormat} onChange={event => setExportFormat(event.target.value)} disabled={Boolean(busy)}>
          <option value="mp4">MP4</option>
          <option value="mov">MOV</option>
        </select>
        <button
          className="primary-button"
          disabled={!job.analysis || Boolean(busy) || segmentsDirty}
          title={segmentsDirty ? 'Save segment changes before exporting.' : undefined}
          onClick={onExport}
        >
          <Download size={17} />
          Export
        </button>
      </div>

      {segmentsDirty ? <span className="segment-save-warning">Save segment changes before exporting.</span> : null}

      {job.export && !segmentsDirty ? (
        <a className="download-link" href={job.export.url} download>
          <Download size={16} />
          {formatSize(job.export.size)}
        </a>
      ) : null}
    </section>
  );
}

function LabelGoal({ label, count, tone }) {
  const percentage = Math.min(100, Math.round((count / LABEL_TARGET) * 100));
  return (
    <div className={`label-goal ${tone}`}>
      <div>
        <span>{label}</span>
        <strong>{count}/{LABEL_TARGET}</strong>
      </div>
      <div
        className="label-progress"
        role="progressbar"
        aria-label={`${label} labels`}
        aria-valuemin="0"
        aria-valuemax={LABEL_TARGET}
        aria-valuenow={Math.min(count, LABEL_TARGET)}
      >
        <span style={{ width: `${percentage}%` }} />
      </div>
    </div>
  );
}

function AnalysisControls({ options, onChange, disabled }) {
  const fields = [
    ['threshold', 'Threshold', 0.35, 0.8, 0.01],
    ['smoothingWindow', 'Smooth', 1, 11, 2],
    ['mergeGap', 'Merge', 0, 8, 0.5],
    ['paddingBefore', 'Pre', 0, 5, 0.25],
    ['paddingAfter', 'Post', 0, 6, 0.25]
  ];

  return (
    <div className="control-stack">
      {fields.map(([key, label, min, max, step]) => (
        <label key={key} className="range-row">
          <span>{label}</span>
          <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={options[key]}
            disabled={disabled}
            onChange={event =>
              onChange({
                ...options,
                [key]: Number(event.target.value)
              })
            }
          />
          <output>{options[key]}</output>
        </label>
      ))}
    </div>
  );
}

function EmptyStage() {
  return (
    <section className="empty-stage">
      <RefreshCcw size={42} className="spin" />
      <strong>Creating training samples</strong>
    </section>
  );
}

function TrainingStage({ job, labels, onLabel }) {
  const samples = job.samples;
  const sampleOrderRef = useRef([]);
  const lastReviewedModelRef = useRef(null);
  const [showModelCheck, setShowModelCheck] = useState(Boolean(job.model?.validationReview?.length));
  const randomizedSamples = useMemo(() => {
    const samplesById = new Map(samples.map(sample => [sample.id, sample]));
    const retainedIds = sampleOrderRef.current.filter(sampleId => samplesById.has(sampleId));
    const retainedIdSet = new Set(retainedIds);
    const addedSamples = shuffleSamples(samples.filter(sample => !retainedIdSet.has(sample.id)));
    const orderedIds = [...retainedIds, ...addedSamples.map(sample => sample.id)];
    sampleOrderRef.current = orderedIds;
    return orderedIds.map(sampleId => samplesById.get(sampleId));
  }, [samples]);
  const modelCheckSamples = useMemo(() => {
    const samplesById = new Map(samples.map(sample => [sample.id, sample]));
    return (job.model?.validationReview || [])
      .map(review => ({ sample: samplesById.get(review.id), review }))
      .filter(item => item.sample);
  }, [job.model?.validationReview, samples]);

  useEffect(() => {
    const modelId = job.model?.createdAt;
    if (!modelId || lastReviewedModelRef.current === modelId) return;
    lastReviewedModelRef.current = modelId;
    setShowModelCheck(Boolean(job.model.validationReview?.length));
  }, [job.model?.createdAt, job.model?.validationReview?.length]);

  const displayedSamples = showModelCheck
    ? modelCheckSamples
    : randomizedSamples.map(sample => ({ sample, review: null }));
  const matchingCount = modelCheckSamples.filter(({ sample, review }) => labels[sample.id] === review.predictedLabel).length;

  return (
    <div className="training-stage">
      <div className="stage-header">
        <div className="stage-title">
          <h2>{showModelCheck ? 'Model Check' : 'Training Samples'}</h2>
          <span>
            {showModelCheck
              ? `${modelCheckSamples.length} held-out clips · ${matchingCount}/${modelCheckSamples.length} predictions match your labels`
              : `${samples.length} one-second clips`}
          </span>
          {modelCheckSamples.length ? (
            <div className="stage-view-switch">
              <button className={showModelCheck ? 'active' : ''} onClick={() => setShowModelCheck(true)}>
                Model Check
              </button>
              <button className={!showModelCheck ? 'active' : ''} onClick={() => setShowModelCheck(false)}>
                All Samples
              </button>
            </div>
          ) : null}
        </div>
        <SourcePanel job={job} compact />
      </div>
      <div className="sample-grid">
        {displayedSamples.map(({ sample, review }) => {
          const modelMatches = review ? labels[sample.id] === review.predictedLabel : null;
          const confidence = review ? Math.round(review.confidence * 100) : null;
          return (
            <article key={sample.id} className={`sample-card ${labels[sample.id] || ''}${review ? ' validation-card' : ''}`}>
              <video src={sample.clipUrl} muted loop playsInline preload="metadata" onMouseEnter={event => event.currentTarget.play()} onMouseLeave={event => event.currentTarget.pause()} />
              {review ? (
                <div className="model-verdict">
                  <span className={`prediction ${review.predictedLabel}`}>
                    Model: {review.predictedLabel} · {confidence}%
                  </span>
                  <span className={modelMatches ? 'match' : 'mismatch'}>
                    {modelMatches ? 'Matches your label' : 'Disagrees with your label'}
                  </span>
                </div>
              ) : null}
              <div className="sample-meta">
                <span>{formatTime(sample.start)}</span>
                <div className="segmented">
                  <button className={labels[sample.id] === 'unsure' ? 'selected unsure' : 'unsure'} onClick={() => onLabel(sample.id, 'unsure')}>
                    <CircleHelp size={15} />
                    Unsure
                  </button>
                  <button className={labels[sample.id] === 'cut' ? 'selected cut' : 'cut'} onClick={() => onLabel(sample.id, 'cut')}>
                    <Scissors size={15} />
                    Cut
                  </button>
                  <button className={labels[sample.id] === 'keep' ? 'selected keep' : 'keep'} onClick={() => onLabel(sample.id, 'keep')}>
                    <Check size={15} />
                    Keep
                  </button>
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
}

function LossSparkline({ history, value }) {
  const values = history.map(point => point.validationLoss ?? point.loss);
  const displayedValue = Number.isFinite(value) ? value : values.at(-1);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const points = values
    .map((value, index) => {
      const x = values.length === 1 ? 0 : (index / (values.length - 1)) * 160;
      const y = 42 - ((value - min) / (max - min || 1)) * 36;
      return `${x},${y}`;
    })
    .join(' ');

  return (
    <div className="loss-widget">
      <div>
        <span>Validation curve</span>
        <strong>{displayedValue}</strong>
      </div>
      <svg viewBox="0 0 160 48" aria-hidden="true">
        <polyline points={points} fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  );
}

function PreviewStage({ job, dirty, busy, onDirtyChange, onSaveSegments }) {
  const analysis = job.analysis;
  const [segments, setSegments] = useState(analysis.keepSegments);
  const [selectedIndex, setSelectedIndex] = useState(null);
  const editedDuration = useMemo(
    () => segments.reduce((sum, segment) => sum + segment.end - segment.start, 0),
    [segments]
  );

  useEffect(() => {
    setSegments(analysis.keepSegments);
    setSelectedIndex(null);
    onDirtyChange(false);
  }, [analysis, onDirtyChange]);

  function updateBoundary(index, boundary, requestedValue) {
    setSegments(current => {
      const segment = current[index];
      if (!segment || !Number.isFinite(requestedValue)) return current;
      const previousEnd = index > 0 ? current[index - 1].end : 0;
      const nextStart = index < current.length - 1 ? current[index + 1].start : analysis.sourceDuration;
      const next = current.map(item => ({ ...item }));

      if (boundary === 'start') {
        next[index].start = Math.max(previousEnd, Math.min(segment.end - 0.1, requestedValue));
      } else {
        next[index].end = Math.min(nextStart, Math.max(segment.start + 0.1, requestedValue));
      }
      next[index].start = Math.round(next[index].start * 1000) / 1000;
      next[index].end = Math.round(next[index].end * 1000) / 1000;
      next[index].duration = Math.round((next[index].end - next[index].start) * 1000) / 1000;
      return next;
    });
    onDirtyChange(true);
  }

  return (
    <div className="preview-stage">
      <div className="stage-header">
        <div>
          <h2>Preview</h2>
          <span>
            {formatTime(editedDuration)} / {formatTime(analysis.sourceDuration)}
          </span>
        </div>
        <div className="metrics compact">
          <div>
            <span>Keep</span>
            <strong>{Math.round((editedDuration / analysis.sourceDuration) * 100)}%</strong>
          </div>
          <div>
            <span>Segments</span>
            <strong>{segments.length}</strong>
          </div>
        </div>
      </div>
      <VirtualPlayer
        sourceUrl={job.source.url}
        segments={segments}
        rows={analysis.rows}
        sourceDuration={analysis.sourceDuration}
        selectedIndex={selectedIndex}
        onSelectSegment={setSelectedIndex}
      />
      {selectedIndex !== null && segments[selectedIndex] ? (
        <SegmentEditor
          index={selectedIndex}
          segment={segments[selectedIndex]}
          dirty={dirty}
          busy={busy}
          onBoundaryChange={(boundary, value) => updateBoundary(selectedIndex, boundary, value)}
          onDiscard={() => {
            setSegments(analysis.keepSegments);
            onDirtyChange(false);
          }}
          onSave={() => onSaveSegments(segments.map(segment => ({ start: segment.start, end: segment.end })))}
        />
      ) : (
        <div className="segment-editor-hint">Select a green segment in the timeline or list to adjust its boundaries.</div>
      )}
      <SegmentTable segments={segments} selectedIndex={selectedIndex} onSelect={setSelectedIndex} />
    </div>
  );
}

function VirtualPlayer({ sourceUrl, segments, rows, sourceDuration, selectedIndex, onSelectSegment }) {
  const videoRef = useRef(null);
  const segmentIndexRef = useRef(0);
  const playingRef = useRef(false);
  const [playing, setPlaying] = useState(false);
  const [previewTime, setPreviewTime] = useState(0);

  const segmentStarts = useMemo(() => {
    const starts = [];
    let cursor = 0;
    for (const segment of segments) {
      starts.push(cursor);
      cursor += segment.end - segment.start;
    }
    return starts;
  }, [segments]);

  const editedDuration = useMemo(() => segments.reduce((sum, segment) => sum + segment.end - segment.start, 0), [segments]);

  function mapPreviewToOriginal(targetTime) {
    const clamped = Math.max(0, Math.min(editedDuration, targetTime));
    for (let index = 0; index < segments.length; index += 1) {
      const start = segmentStarts[index];
      const duration = segments[index].end - segments[index].start;
      if (clamped <= start + duration || index === segments.length - 1) {
        return {
          index,
          previewTime: clamped,
          originalTime: segments[index].start + Math.min(duration, Math.max(0, clamped - start))
        };
      }
    }
    return { index: 0, previewTime: 0, originalTime: segments[0]?.start || 0 };
  }

  function jumpToPreview(targetTime) {
    if (!videoRef.current || !segments.length) return;
    const mapped = mapPreviewToOriginal(targetTime);
    segmentIndexRef.current = mapped.index;
    videoRef.current.currentTime = mapped.originalTime;
    setPreviewTime(mapped.previewTime);
  }

  async function togglePlayback() {
    if (!videoRef.current || !segments.length) return;
    if (playingRef.current) {
      videoRef.current.pause();
      return;
    }
    if (previewTime >= editedDuration - 0.05) jumpToPreview(0);
    await videoRef.current.play();
  }

  function handleTimeUpdate() {
    if (!videoRef.current || !segments.length) return;
    const currentIndex = segmentIndexRef.current;
    const segment = segments[currentIndex];
    const originalTime = videoRef.current.currentTime;

    if (originalTime >= segment.end - 0.04) {
      const nextIndex = currentIndex + 1;
      if (nextIndex >= segments.length) {
        videoRef.current.pause();
        setPreviewTime(editedDuration);
        return;
      }
      segmentIndexRef.current = nextIndex;
      videoRef.current.currentTime = segments[nextIndex].start;
      setPreviewTime(segmentStarts[nextIndex]);
      return;
    }

    setPreviewTime(segmentStarts[currentIndex] + Math.max(0, originalTime - segment.start));
  }

  useEffect(() => {
    if (!segments.length) return;
    const targetIndex = selectedIndex === null ? 0 : Math.min(selectedIndex, segments.length - 1);
    jumpToPreview(segmentStarts[targetIndex] || 0);
  }, [sourceUrl, segments, selectedIndex]);

  return (
    <section className="player-shell">
      <video
        ref={videoRef}
        src={sourceUrl}
        playsInline
        preload="metadata"
        onTimeUpdate={handleTimeUpdate}
        onPlay={() => {
          playingRef.current = true;
          setPlaying(true);
        }}
        onPause={() => {
          playingRef.current = false;
          setPlaying(false);
        }}
      />
      <div className="transport">
        <button className="icon-button" onClick={togglePlayback} aria-label={playing ? 'Pause' : 'Play'}>
          {playing ? <Pause size={20} /> : <Play size={20} />}
        </button>
        <TimelineScrubber
          previewTime={previewTime}
          editedDuration={editedDuration}
          segments={segments}
          rows={rows}
          sourceDuration={sourceDuration}
          selectedIndex={selectedIndex}
          onSelectSegment={onSelectSegment}
          onSeek={jumpToPreview}
        />
        <span className="time-readout">
          {formatTime(previewTime)} / {formatTime(editedDuration)}
        </span>
      </div>
    </section>
  );
}

function TimelineScrubber({
  previewTime,
  editedDuration,
  segments,
  rows,
  sourceDuration,
  selectedIndex,
  onSelectSegment,
  onSeek
}) {
  const barRef = useRef(null);
  const percent = editedDuration ? (previewTime / editedDuration) * 100 : 0;

  function seekFromClientX(clientX) {
    const rect = barRef.current.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    onSeek(ratio * editedDuration);
  }

  return (
    <div
      ref={barRef}
      className="timeline"
      onPointerDown={event => {
        event.currentTarget.setPointerCapture(event.pointerId);
        seekFromClientX(event.clientX);
      }}
      onPointerMove={event => {
        if (event.buttons === 1) seekFromClientX(event.clientX);
      }}
    >
      <div className="source-ribbon">
        {rows.map(row => (
          <span
            key={row.start}
            className={row.decision}
            style={{
              left: `${(row.start / sourceDuration) * 100}%`,
              width: `${Math.max(0.1, ((row.end - row.start) / sourceDuration) * 100)}%`
            }}
          />
        ))}
      </div>
      <div className="edited-ribbon">
        {segments.map((segment, index) => {
          const totalBefore = segments.slice(0, index).reduce((sum, item) => sum + item.end - item.start, 0);
          return (
            <button
              type="button"
              key={`${segment.start}-${segment.end}`}
              className={selectedIndex === index ? 'selected' : ''}
              aria-label={`Select keep segment ${index + 1}`}
              onPointerDown={event => event.stopPropagation()}
              onClick={event => {
                event.stopPropagation();
                onSelectSegment(index);
              }}
              style={{
                left: `${(totalBefore / editedDuration) * 100}%`,
                width: `${((segment.end - segment.start) / editedDuration) * 100}%`
              }}
            />
          );
        })}
      </div>
      <div className="timeline-fill" style={{ width: `${percent}%` }} />
      <div className="timeline-thumb" style={{ left: `${percent}%` }} />
    </div>
  );
}

function SegmentEditor({ index, segment, dirty, busy, onBoundaryChange, onDiscard, onSave }) {
  function TimeControl({ label, boundary, value }) {
    return (
      <div className="boundary-control">
        <span>{label}</span>
        <div>
          <button type="button" onClick={() => onBoundaryChange(boundary, value - 1)} aria-label={`Move ${label.toLowerCase()} one second earlier`}>
            <Minus size={14} />1s
          </button>
          <button type="button" onClick={() => onBoundaryChange(boundary, value - 0.25)} aria-label={`Move ${label.toLowerCase()} a quarter-second earlier`}>
            <Minus size={14} />.25
          </button>
          <input
            key={`${boundary}-${value}`}
            type="number"
            step="0.1"
            defaultValue={value.toFixed(2)}
            aria-label={`${label} time in seconds`}
            onBlur={event => onBoundaryChange(boundary, Number(event.currentTarget.value))}
            onKeyDown={event => {
              if (event.key === 'Enter') event.currentTarget.blur();
            }}
          />
          <button type="button" onClick={() => onBoundaryChange(boundary, value + 0.25)} aria-label={`Move ${label.toLowerCase()} a quarter-second later`}>
            <Plus size={14} />.25
          </button>
          <button type="button" onClick={() => onBoundaryChange(boundary, value + 1)} aria-label={`Move ${label.toLowerCase()} one second later`}>
            <Plus size={14} />1s
          </button>
        </div>
      </div>
    );
  }

  return (
    <section className="segment-editor">
      <div className="segment-editor-heading">
        <div>
          <h2>Adjust segment {index + 1}</h2>
          <span>{formatTime(segment.end - segment.start)} kept</span>
        </div>
        <div className="segment-editor-actions">
          <button className="secondary-button" type="button" disabled={!dirty || busy} onClick={onDiscard}>
            <RotateCcw size={15} />
            Discard
          </button>
          <button className="primary-button" type="button" disabled={!dirty || busy} onClick={onSave}>
            <Save size={16} />
            Save changes
          </button>
        </div>
      </div>
      <div className="boundary-controls">
        <TimeControl label="Start" boundary="start" value={segment.start} />
        <TimeControl label="End" boundary="end" value={segment.end} />
      </div>
      <p>Move Start earlier to restore footage before the point, or move End later to restore footage after it.</p>
    </section>
  );
}

function SegmentTable({ segments, selectedIndex, onSelect }) {
  return (
    <section className="segment-list">
      <h2>Keep Segments</h2>
      <div className="segment-table">
        {segments.map((segment, index) => (
          <button
            type="button"
            key={`${segment.start}-${segment.end}`}
            className={`segment-row${selectedIndex === index ? ' selected' : ''}`}
            onClick={() => onSelect(index)}
          >
            <span>{String(index + 1).padStart(2, '0')}</span>
            <span>{formatTime(segment.start)}</span>
            <span>{formatTime(segment.end)}</span>
            <span>{formatTime(segment.duration)}</span>
            <span>{Math.round(metric(segment.averageProbability, 0) * 100)}%</span>
          </button>
        ))}
      </div>
    </section>
  );
}

export default App;
