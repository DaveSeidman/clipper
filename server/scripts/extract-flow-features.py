#!/usr/bin/env python3
import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Extract movement-only optical-flow features.")
    parser.add_argument("--job", required=True, help="Path to job.json")
    parser.add_argument("--output", required=True, help="Feature cache JSON path")
    parser.add_argument("--method", choices=("farneback", "dis"), default="farneback")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument(
        "--context-seconds",
        type=float,
        default=1.0,
        help="Centered temporal window around each labeled second.",
    )
    parser.add_argument("--grid-width", type=int, default=12)
    parser.add_argument("--grid-height", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_sample_frames(capture, start, duration, frame_count, source_fps, size):
    start_frame = max(0, int(round(start * source_fps)))
    duration_frames = max(1, int(round(duration * source_fps)))
    targets = [
        start_frame + min(duration_frames - 1, int(((index + 0.5) / frame_count) * duration_frames))
        for index in range(frame_count)
    ]
    capture.set(cv2.CAP_PROP_POS_FRAMES, targets[0])
    cursor = targets[0]
    frames = []

    for target in targets:
        selected = None
        while cursor <= target:
            ok, frame = capture.read()
            if not ok:
                break
            if cursor == target:
                selected = frame
            cursor += 1
        if selected is None:
            if not frames:
                raise RuntimeError(f"Could not decode frame near {start:.3f}s")
            frames.append(frames[-1])
            continue
        gray = cv2.cvtColor(selected, cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(gray, size, interpolation=cv2.INTER_AREA))

    while len(frames) < frame_count:
        frames.append(frames[-1])
    return frames


def make_flow_calculator(method):
    if method == "dis":
        calculator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        calculator.setUseSpatialPropagation(True)
        return lambda previous, current: calculator.calc(previous, current, None)

    return lambda previous, current: cv2.calcOpticalFlowFarneback(
        previous,
        current,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )


def summarize_flow(frames, method, grid_width, grid_height):
    calculate = make_flow_calculator(method)
    pair_fields = []

    for previous, current in zip(frames, frames[1:]):
        flow = calculate(previous, current).astype(np.float32)
        global_motion = np.median(flow.reshape(-1, 2), axis=0)
        flow[:, :, 0] -= global_motion[0]
        flow[:, :, 1] -= global_motion[1]
        pair_fields.append(flow)

    height, width = frames[0].shape
    diagonal = math.sqrt(width * width + height * height)
    cell_features = []
    temporal_features = []
    direction_histogram = np.zeros(8, dtype=np.float64)
    magnitude_histogram = np.zeros(8, dtype=np.float64)
    magnitude_bins = np.array([0, 0.25, 0.5, 1, 2, 4, 8, 16, np.inf], dtype=np.float64)
    all_magnitudes = []

    cell_bounds = []
    for grid_y in range(grid_height):
        y0 = int((grid_y / grid_height) * height)
        y1 = max(y0 + 1, int(((grid_y + 1) / grid_height) * height))
        for grid_x in range(grid_width):
            x0 = int((grid_x / grid_width) * width)
            x1 = max(x0 + 1, int(((grid_x + 1) / grid_width) * width))
            cell_bounds.append((y0, y1, x0, x1))

    per_cell = [[] for _ in cell_bounds]
    for flow in pair_fields:
        dx = flow[:, :, 0]
        dy = flow[:, :, 1]
        magnitude = np.sqrt(dx * dx + dy * dy)
        angle = np.arctan2(dy, dx)
        all_magnitudes.append(magnitude.reshape(-1))

        temporal_features.extend([
            float(np.mean(magnitude) / diagonal),
            float(np.percentile(magnitude, 90) / diagonal),
            float(np.mean(magnitude >= 0.5)),
        ])

        direction_bins = np.floor(((angle + np.pi) / (2 * np.pi)) * 8).astype(np.int32)
        direction_bins = np.clip(direction_bins, 0, 7)
        for direction in range(8):
            direction_histogram[direction] += float(np.sum(magnitude[direction_bins == direction]))
        magnitude_histogram += np.histogram(magnitude, bins=magnitude_bins)[0]

        for index, (y0, y1, x0, x1) in enumerate(cell_bounds):
            cell_dx = dx[y0:y1, x0:x1]
            cell_dy = dy[y0:y1, x0:x1]
            cell_magnitude = magnitude[y0:y1, x0:x1]
            per_cell[index].append((
                float(np.mean(cell_dx) / width),
                float(np.mean(cell_dy) / height),
                float(np.mean(cell_magnitude) / diagonal),
                float(np.std(cell_magnitude) / diagonal),
                float(np.percentile(cell_magnitude, 90) / diagonal),
                float(np.mean(cell_magnitude >= 0.5)),
            ))

    for observations in per_cell:
        values = np.asarray(observations, dtype=np.float64)
        cell_features.extend(np.mean(values, axis=0).tolist())

    direction_total = float(np.sum(direction_histogram)) or 1.0
    magnitude_total = float(np.sum(magnitude_histogram)) or 1.0
    combined = np.concatenate(all_magnitudes)
    global_features = [
        float(np.mean(combined) / diagonal),
        float(np.std(combined) / diagonal),
        float(np.percentile(combined, 50) / diagonal),
        float(np.percentile(combined, 75) / diagonal),
        float(np.percentile(combined, 90) / diagonal),
        float(np.percentile(combined, 95) / diagonal),
        float(np.max(combined) / diagonal),
        float(np.mean(combined >= 0.5)),
    ]

    return (
        cell_features
        + (direction_histogram / direction_total).tolist()
        + (magnitude_histogram / magnitude_total).tolist()
        + temporal_features
        + global_features
    )


def main():
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        print(f"Using cached features: {output_path}", flush=True)
        return

    job_path = Path(args.job)
    job = json.loads(job_path.read_text())
    video_path = job_path.parent / job["source"]["relativePath"]
    labeled = [
        sample for sample in job["samples"]
        if sample.get("label") in ("keep", "cut")
    ]
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or job["source"].get("fps") or 30)
    started = time.perf_counter()
    rows = []

    for index, sample in enumerate(labeled, start=1):
        sample_start = float(sample["start"])
        sample_duration = float(sample.get("duration") or 1)
        context_duration = max(sample_duration, args.context_seconds)
        context_start = max(0.0, sample_start - ((context_duration - sample_duration) / 2))
        frame_count = max(2, int(round(args.fps * context_duration)))
        frames = read_sample_frames(
            capture,
            context_start,
            context_duration,
            frame_count,
            source_fps,
            (args.width, args.height),
        )
        rows.append({
            "id": sample["id"],
            "label": sample["label"],
            "features": summarize_flow(frames, args.method, args.grid_width, args.grid_height),
        })
        if index % 25 == 0 or index == len(labeled):
            elapsed = time.perf_counter() - started
            print(f"{index}/{len(labeled)} clips · {elapsed:.1f}s", flush=True)

    capture.release()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "jobId": job["id"],
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "extractionSeconds": round(time.perf_counter() - started, 3),
        "config": {
            "method": args.method,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "contextSeconds": args.context_seconds,
            "gridWidth": args.grid_width,
            "gridHeight": args.grid_height,
        },
        "featureDimensions": len(rows[0]["features"]) if rows else 0,
        "rows": rows,
    }
    output_path.write_text(json.dumps(payload))
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
