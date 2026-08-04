#!/usr/bin/env python3
"""Extract cached, local VideoMAE embeddings for Clipper samples or a timeline."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import VideoMAEForPreTraining, __version__ as transformers_version


CACHE_VERSION = 1
IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("samples", "timeline"), required=True)
    parser.add_argument("--model-id", default="MCG-NJU/videomae-base")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--context-seconds", type=float, default=5.0)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--decode-workers", type=int, default=4)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def rounded(value):
    return round(float(value), 7)


def select_device(requested):
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("The requested MPS device is not available.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_config(args, job, video_path):
    stat = video_path.stat()
    return {
        "extractorVersion": "pretraining-encoder-v1",
        "transformersVersion": transformers_version,
        "modelId": args.model_id,
        "contextSeconds": args.context_seconds,
        "numFrames": args.num_frames,
        "imageSize": args.image_size,
        "sourceSize": stat.st_size,
        "sourceDuration": float(job["source"]["duration"]),
    }


def load_cache(output_path, config, mode, interval):
    if not output_path.exists():
        return {}
    try:
        payload = json.loads(output_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if payload.get("version") != CACHE_VERSION or payload.get("config") != config:
        return {}
    if payload.get("mode") != mode:
        return {}
    if mode == "timeline" and float(payload.get("interval", 0)) != interval:
        return {}
    return {row["id"]: row for row in payload.get("rows", [])}


def make_targets(args, job):
    duration = float(job["source"]["duration"])
    if args.mode == "samples":
        return [
            {
                "id": sample["id"],
                "start": float(sample["start"]),
                "end": float(sample["end"]),
                "center": (float(sample["start"]) + float(sample["end"])) / 2,
                "label": sample["label"],
            }
            for sample in job.get("samples", [])
            if sample.get("label") in ("keep", "cut")
        ]

    targets = []
    index = 0
    start = 0.0
    while start < duration:
        end = min(duration, start + args.interval)
        targets.append({
            "id": f"timeline-{index:06d}",
            "start": start,
            "end": end,
            "center": (start + end) / 2,
        })
        index += 1
        start = index * args.interval
    return targets


def context_bounds(center, context_seconds, duration):
    window = min(context_seconds, duration)
    start = max(0.0, min(duration - window, center - window / 2))
    return start, max(0.05, window)


def decode_frames(video_path, target, args, duration):
    start, window = context_bounds(target["center"], args.context_seconds, duration)
    fps = args.num_frames / window
    size = args.image_size
    filter_graph = (
        f"fps={fps:.8f},"
        f"scale={size}:{size}:force_original_aspect_ratio=decrease,"
        f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2:black"
    )
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.6f}", "-i", str(video_path),
        "-t", f"{window:.6f}", "-vf", filter_graph,
        "-frames:v", str(args.num_frames), "-an", "-sn",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    completed = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = size * size * 3
    count = len(completed.stdout) // frame_bytes
    if count == 0:
        raise RuntimeError(f"ffmpeg returned no frames for {target['id']}")
    frames = np.frombuffer(completed.stdout[:count * frame_bytes], dtype=np.uint8)
    frames = frames.reshape(count, size, size, 3)
    if count < args.num_frames:
        padding = np.repeat(frames[-1:], args.num_frames - count, axis=0)
        frames = np.concatenate((frames, padding), axis=0)
    elif count > args.num_frames:
        frames = frames[:args.num_frames]
    return frames.copy()


def frames_to_tensor(batch_frames):
    array = np.stack(batch_frames, axis=0)
    tensor = torch.from_numpy(array).permute(0, 1, 4, 2, 3).float().div_(255.0)
    return (tensor - IMAGE_MEAN) / IMAGE_STD


def save_payload(output_path, job, args, config, device, rows, started_at):
    payload = {
        "version": CACHE_VERSION,
        "jobId": job["id"],
        "mode": args.mode,
        "interval": args.interval if args.mode == "timeline" else None,
        "config": config,
        "device": str(device),
        "embeddingDimension": len(rows[0]["features"]) if rows else 0,
        "rows": rows,
        "elapsedSeconds": round(time.perf_counter() - started_at, 3),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")))
    temporary.replace(output_path)
    return payload


def ordered_rows(targets, cached):
    rows = []
    for target in targets:
        if target["id"] not in cached:
            continue
        row = cached[target["id"]]
        if "label" in target:
            row["label"] = target["label"]
        rows.append(row)
    return rows


def main():
    args = parse_args()
    started_at = time.perf_counter()
    job_path = Path(args.job).resolve()
    output_path = Path(args.output).resolve()
    job = json.loads(job_path.read_text())
    video_path = job_path.parent / job["source"]["relativePath"]
    config = make_config(args, job, video_path)
    targets = make_targets(args, job)
    cached = load_cache(output_path, config, args.mode, args.interval)
    missing = [target for target in targets if target["id"] not in cached]
    device = select_device(args.device)

    print(
        f"VideoMAE {args.mode}: {len(targets)} windows, {len(missing)} uncached, device={device}",
        file=sys.stderr,
        flush=True,
    )

    model = None
    if missing:
        pretrained_model = VideoMAEForPreTraining.from_pretrained(
            args.model_id,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )
        model = pretrained_model.videomae
        model.eval().to(device)

    for batch_start in range(0, len(missing), args.batch_size):
        batch_targets = missing[batch_start:batch_start + args.batch_size]
        with ThreadPoolExecutor(max_workers=min(args.decode_workers, len(batch_targets))) as executor:
            batch_frames = list(executor.map(
                lambda target: decode_frames(video_path, target, args, float(job["source"]["duration"])),
                batch_targets,
            ))
        pixel_values = frames_to_tensor(batch_frames).to(device)
        with torch.inference_mode():
            hidden = model(pixel_values=pixel_values).last_hidden_state
            embeddings = F.normalize(hidden.mean(dim=1), p=2, dim=1).cpu().numpy()
        for target, embedding in zip(batch_targets, embeddings):
            cached[target["id"]] = {
                "id": target["id"],
                "start": rounded(target["start"]),
                "end": rounded(target["end"]),
                "features": [rounded(value) for value in embedding],
            }

        completed_count = min(len(missing), batch_start + len(batch_targets))
        print(f"Embedded {completed_count}/{len(missing)} uncached windows", file=sys.stderr, flush=True)
        save_payload(output_path, job, args, config, device, ordered_rows(targets, cached), started_at)

    rows = ordered_rows(targets, cached)
    payload = save_payload(output_path, job, args, config, device, rows, started_at)
    print(json.dumps({
        "output": str(output_path),
        "rows": len(rows),
        "embeddingDimension": payload["embeddingDimension"],
        "device": str(device),
        "elapsedSeconds": payload["elapsedSeconds"],
    }))


if __name__ == "__main__":
    main()
