async function readJson(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || 'Request failed.');
  }
  return payload;
}

export async function uploadVideo(file) {
  const form = new FormData();
  form.append('video', file);
  const response = await fetch('/api/jobs', {
    method: 'POST',
    body: form
  });
  return readJson(response);
}

export async function loadJob(jobId) {
  const response = await fetch(`/api/jobs/${jobId}`);
  return readJson(response);
}

export async function loadLatestJob() {
  const response = await fetch('/api/jobs/latest');
  return readJson(response);
}

export async function generateSamples(jobId, options) {
  const response = await fetch(`/api/jobs/${jobId}/samples`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(options)
  });
  return readJson(response);
}

export async function saveLabels(jobId, labels) {
  const response = await fetch(`/api/jobs/${jobId}/labels`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ labels })
  });
  return readJson(response);
}

export async function trainJob(jobId, labels) {
  const response = await fetch(`/api/jobs/${jobId}/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ labels })
  });
  return readJson(response);
}

export async function analyzeJob(jobId, options) {
  const response = await fetch(`/api/jobs/${jobId}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(options)
  });
  return readJson(response);
}

export async function saveSegments(jobId, segments) {
  const response = await fetch(`/api/jobs/${jobId}/segments`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ segments })
  });
  return readJson(response);
}

export async function exportJob(jobId, format) {
  const response = await fetch(`/api/jobs/${jobId}/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ format })
  });
  return readJson(response);
}
