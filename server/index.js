import express from 'express';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { ensureBaseDirs, DATA_DIR } from './lib/paths.js';
import jobsRouter from './routes/jobs.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PORT = Number(process.env.PORT || 3030);

await ensureBaseDirs();

const app = express();

app.use(express.json({ limit: '5mb' }));
app.use('/media', express.static(DATA_DIR, {
  immutable: false,
  maxAge: 0,
  acceptRanges: true
}));
app.use('/api/jobs', jobsRouter);

app.get('/api/health', (req, res) => {
  res.json({ ok: true });
});

const distPath = path.join(__dirname, '..', 'dist', 'client');
app.use(express.static(distPath));
app.get(/.*/, (req, res, next) => {
  if (req.path.startsWith('/api') || req.path.startsWith('/media')) {
    next();
    return;
  }
  res.sendFile(path.join(distPath, 'index.html'), error => {
    if (error) next();
  });
});

app.use((error, req, res, next) => {
  if (req.file?.path) {
    // Upload temp files are otherwise left behind when probing or validation fails.
    import('node:fs/promises').then(fs => fs.rm(req.file.path, { force: true })).catch(() => {});
  }

  const status = error.status || 500;
  const payload = {
    error: error.message || 'Unexpected server error.'
  };
  if (process.env.NODE_ENV !== 'production' && error.stderr) {
    payload.details = error.stderr.slice(-4000);
  }
  res.status(status).json(payload);
});

app.listen(PORT, '127.0.0.1', () => {
  console.log(`Clipper API listening on http://127.0.0.1:${PORT}`);
});
