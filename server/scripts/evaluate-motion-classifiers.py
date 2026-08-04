#!/usr/bin/env python3
"""Compare nonlinear classifiers using time-blocked cross-validation.

The feature extractor intentionally emits motion-only data. This evaluator
keeps clips from the same span of video in the same fold so nearby samples do
not leak into both training and validation.
"""

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        required=True,
        nargs="+",
        help="One or more aligned feature caches; multiple caches are concatenated.",
    )
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model-output",
        help="Optional OpenCV model path; fits the selected classifier on every labeled row.",
    )
    parser.add_argument(
        "--selected-result",
        help="Reuse the selected classifier from a prior result instead of searching again.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--block-seconds", type=float, default=20.0)
    parser.add_argument(
        "--purge-seconds",
        type=float,
        default=None,
        help="Remove training rows this close to a validation row; defaults to the longest context.",
    )
    return parser.parse_args()


def fnv1a(text):
    value = 2166136261
    for character in text:
        value ^= ord(character)
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=np.int32)
    predicted = np.asarray(predicted, dtype=np.int32)
    accuracy = float(np.mean(actual == predicted))
    recalls = []
    for label in (0, 1):
        mask = actual == label
        recalls.append(float(np.mean(predicted[mask] == label)) if np.any(mask) else 0.0)
    return {
        "accuracy": accuracy,
        "balancedAccuracy": sum(recalls) / 2,
        "cutRecall": recalls[0],
        "keepRecall": recalls[1],
    }


def normalize_fit(features):
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std < 1e-8] = 1
    return mean, std


def balanced_training(features, labels):
    counts = [int(np.sum(labels == label)) for label in (0, 1)]
    target = max(counts)
    selected = []
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        selected.extend(np.resize(indices, target).tolist())
    selected = np.asarray(selected, dtype=np.int32)
    return features[selected], labels[selected]


def fit_model(kind, params, train_x, train_y):
    if params.get("balanced"):
        train_x, train_y = balanced_training(train_x, train_y)

    if kind in ("linear-svm", "rbf-svm"):
        mean, std = normalize_fit(train_x)
        fitted_x = ((train_x - mean) / std).astype(np.float32)
        model = cv2.ml.SVM_create()
        model.setType(cv2.ml.SVM_C_SVC)
        model.setKernel(cv2.ml.SVM_LINEAR if kind == "linear-svm" else cv2.ml.SVM_RBF)
        model.setC(float(params["c"]))
        if kind == "rbf-svm":
            model.setGamma(float(params["gammaScale"]) / train_x.shape[1])
        model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100000, 1e-7))
        model.train(fitted_x, cv2.ml.ROW_SAMPLE, train_y.astype(np.int32))
        return model, {"mean": mean, "std": std}

    model = cv2.ml.RTrees_create()
    model.setMaxDepth(int(params["maxDepth"]))
    model.setMinSampleCount(int(params["minSamples"]))
    model.setRegressionAccuracy(0)
    model.setUseSurrogates(False)
    model.setMaxCategories(2)
    model.setCalculateVarImportance(False)
    model.setActiveVarCount(int(params["activeVariables"]))
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, int(params["trees"]), 0))
    cv2.setRNGSeed(1337)
    model.train(train_x.astype(np.float32), cv2.ml.ROW_SAMPLE, train_y.astype(np.int32))
    return model, {}


def predict_model(model, preprocessing, test_x):
    fitted_test = test_x
    if "mean" in preprocessing:
        fitted_test = (test_x - preprocessing["mean"]) / preprocessing["std"]
    return model.predict(fitted_test.astype(np.float32))[1].reshape(-1).astype(np.int32)


def train_predict(kind, params, train_x, train_y, test_x):
    model, preprocessing = fit_model(kind, params, train_x, train_y)
    return predict_model(model, preprocessing, test_x)


def parameter_grid(feature_count):
    yield "linear-svm", [
        {"c": c, "balanced": balanced}
        for c, balanced in itertools.product((0.01, 0.1, 1, 10), (False, True))
    ]
    yield "rbf-svm", [
        {"c": c, "gammaScale": gamma, "balanced": balanced}
        for c, gamma, balanced in itertools.product(
            (0.1, 1, 10, 100),
            (0.1, 0.3, 1, 3),
            (False, True),
        )
    ]
    root = max(1, int(round(math.sqrt(feature_count))))
    yield "random-forest", [
        {
            "maxDepth": depth,
            "minSamples": minimum,
            "activeVariables": active,
            "trees": 300,
            "balanced": balanced,
        }
        for depth, minimum, active, balanced in itertools.product(
            (5, 10, 20),
            (2, 5),
            (root, root * 2),
            (False, True),
        )
    ]


def select_audit_indices(ids, labels):
    chosen = []
    for label in (0, 1):
        candidates = [index for index, value in enumerate(labels) if value == label]
        candidates.sort(key=lambda index: (fnv1a(f"{ids[index]}:nonlinear-audit"), ids[index]))
        chosen.extend(candidates[:5])
    return sorted(chosen)


def make_blocked_folds(starts, labels, available, fold_count, block_seconds):
    blocks = {}
    for index in available:
        block = int(math.floor(starts[index] / block_seconds))
        blocks.setdefault(block, []).append(index)

    # Greedily balance fold size and class counts while keeping each time block intact.
    fold_rows = [[] for _ in range(fold_count)]
    fold_counts = [{"total": 0, "keep": 0} for _ in range(fold_count)]
    ordered_blocks = sorted(
        blocks.items(),
        key=lambda item: (-len(item[1]), fnv1a(f"motion-block:{item[0]}")),
    )
    for _, indices in ordered_blocks:
        block_keep = sum(labels[index] == 1 for index in indices)
        target = min(
            range(fold_count),
            key=lambda fold: (
                fold_counts[fold]["total"],
                fold_counts[fold]["keep"],
                fold,
            ),
        )
        fold_rows[target].extend(indices)
        fold_counts[target]["total"] += len(indices)
        fold_counts[target]["keep"] += block_keep
    return fold_rows


def evaluate_candidate(
    kind,
    params,
    features,
    labels,
    ids,
    starts,
    folds,
    available,
    purge_seconds,
    training_fraction=1.0,
):
    fold_metrics = []
    training_sizes = []
    available_set = set(available)
    for validation in folds:
        validation_set = set(validation)
        training = sorted(available_set - validation_set)
        if purge_seconds > 0:
            validation_starts = starts[validation]
            training = [
                index for index in training
                if float(np.min(np.abs(validation_starts - starts[index]))) >= purge_seconds
            ]
        if training_fraction < 1:
            reduced = []
            for label in (0, 1):
                group = [index for index in training if labels[index] == label]
                group.sort(key=lambda index: (fnv1a(f"{ids[index]}:learning-curve"), ids[index]))
                count = max(1, int(round(len(group) * training_fraction)))
                reduced.extend(group[:count])
            training = sorted(reduced)
        training_sizes.append(len(training))
        predicted = train_predict(
            kind,
            params,
            features[training],
            labels[training],
            features[validation],
        )
        fold_metrics.append(metrics(labels[validation], predicted))
    return {
        "balancedAccuracy": float(np.mean([row["balancedAccuracy"] for row in fold_metrics])),
        "accuracy": float(np.mean([row["accuracy"] for row in fold_metrics])),
        "cutRecall": float(np.mean([row["cutRecall"] for row in fold_metrics])),
        "keepRecall": float(np.mean([row["keepRecall"] for row in fold_metrics])),
        "meanTrainingRows": float(np.mean(training_sizes)),
        "folds": fold_metrics,
    }


def rounded(value):
    if isinstance(value, float):
        return round(value, 5)
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item) for item in value]
    return value


def main():
    args = parse_args()
    started = time.perf_counter()
    feature_payloads = [json.loads(Path(feature_path).read_text()) for feature_path in args.features]
    feature_payload = feature_payloads[0]
    job = json.loads(Path(args.job).read_text())
    start_by_id = {sample["id"]: float(sample["start"]) for sample in job["samples"]}
    rows = feature_payload["rows"]
    aligned_caches = [{row["id"]: row for row in payload["rows"]} for payload in feature_payloads]
    ids = [row["id"] for row in rows]
    labels = np.asarray([1 if row["label"] == "keep" else 0 for row in rows], dtype=np.int32)
    for cache in aligned_caches[1:]:
        for row in rows:
            if row["id"] not in cache or cache[row["id"]]["label"] != row["label"]:
                raise RuntimeError(f"Feature caches are not aligned at {row['id']}")
    features = np.asarray([
        list(itertools.chain.from_iterable(cache[row_id]["features"] for cache in aligned_caches))
        for row_id in ids
    ], dtype=np.float32)
    starts = np.asarray([start_by_id[row_id] for row_id in ids], dtype=np.float64)
    purge_seconds = args.purge_seconds
    if purge_seconds is None:
        purge_seconds = max(
            float(payload.get("config", {}).get("contextSeconds", 1))
            for payload in feature_payloads
        )

    audit = select_audit_indices(ids, labels)
    audit_set = set(audit)
    development = [index for index in range(len(rows)) if index not in audit_set]
    folds = make_blocked_folds(starts, labels, development, args.folds, args.block_seconds)
    print(
        f"{len(rows)} rows · {features.shape[1]} features · "
        f"{len(development)} development · {len(audit)} audit",
        flush=True,
    )

    searches = []
    best = None
    if args.selected_result:
        prior = json.loads(Path(args.selected_result).read_text())["selected"]
        result = evaluate_candidate(
            prior["kind"],
            prior["params"],
            features,
            labels,
            ids,
            starts,
            folds,
            development,
            purge_seconds,
        )
        best = {"kind": prior["kind"], "params": prior["params"], "crossValidation": result}
        searches.append(best)
        print(
            f"Reused {best['kind']} · balanced {result['balancedAccuracy']:.3f}",
            flush=True,
        )
    else:
        for kind, candidates in parameter_grid(features.shape[1]):
            kind_best = None
            print(f"Searching {kind} ({len(candidates)} candidates)", flush=True)
            for candidate_index, params in enumerate(candidates, start=1):
                result = evaluate_candidate(
                    kind,
                    params,
                    features,
                    labels,
                    ids,
                    starts,
                    folds,
                    development,
                    purge_seconds,
                )
                entry = {"kind": kind, "params": params, "crossValidation": result}
                if kind_best is None or (
                    result["balancedAccuracy"],
                    result["accuracy"],
                ) > (
                    kind_best["crossValidation"]["balancedAccuracy"],
                    kind_best["crossValidation"]["accuracy"],
                ):
                    kind_best = entry
                if candidate_index % 8 == 0 or candidate_index == len(candidates):
                    print(
                        f"  {candidate_index}/{len(candidates)} · "
                        f"best balanced {kind_best['crossValidation']['balancedAccuracy']:.3f}",
                        flush=True,
                    )
            searches.append(kind_best)
            if best is None or (
                kind_best["crossValidation"]["balancedAccuracy"],
                kind_best["crossValidation"]["accuracy"],
            ) > (
                best["crossValidation"]["balancedAccuracy"],
                best["crossValidation"]["accuracy"],
            ):
                best = kind_best

    learning_curve = []
    for fraction in (0.25, 0.5, 0.75, 1.0):
        curve_metrics = evaluate_candidate(
            best["kind"],
            best["params"],
            features,
            labels,
            ids,
            starts,
            folds,
            development,
            purge_seconds,
            training_fraction=fraction,
        )
        learning_curve.append({
            "trainingFraction": fraction,
            **curve_metrics,
        })

    audit_prediction = train_predict(
        best["kind"],
        best["params"],
        features[development],
        labels[development],
        features[audit],
    )
    audit_metrics = metrics(labels[audit], audit_prediction)
    audit_rows = [
        {
            "id": ids[index],
            "start": float(starts[index]),
            "actualLabel": "keep" if labels[index] else "cut",
            "predictedLabel": "keep" if prediction else "cut",
            "correct": bool(prediction == labels[index]),
        }
        for index, prediction in zip(audit, audit_prediction)
    ]
    payload = {
        "version": 1,
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "featureCaches": [str(Path(feature_path)) for feature_path in args.features],
        "featureConfigs": [payload["config"] for payload in feature_payloads],
        "featureDimensions": int(features.shape[1]),
        "evaluation": {
            "strategy": "time-blocked-cross-validation",
            "folds": args.folds,
            "blockSeconds": args.block_seconds,
            "purgeSeconds": purge_seconds,
            "developmentRows": len(development),
            "auditRows": len(audit),
        },
        "bestByClassifier": searches,
        "selected": best,
        "learningCurve": learning_curve,
        "audit": {
            "metrics": audit_metrics,
            "rows": audit_rows,
        },
        "elapsedSeconds": time.perf_counter() - started,
    }
    if args.model_output:
        model_path = Path(args.model_output)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        final_model, preprocessing = fit_model(
            best["kind"],
            best["params"],
            features,
            labels,
        )
        final_model.save(str(model_path))
        preprocessing_path = model_path.with_suffix(".preprocessing.json")
        preprocessing_payload = {
            "version": 1,
            "classifier": best["kind"],
            "params": best["params"],
            "featureCaches": [str(Path(feature_path)) for feature_path in args.features],
            "featureConfigs": [item["config"] for item in feature_payloads],
            "featureDimensions": int(features.shape[1]),
            "trainingRows": len(rows),
            "classes": {"0": "cut", "1": "keep"},
        }
        if "mean" in preprocessing:
            preprocessing_payload["mean"] = preprocessing["mean"].tolist()
            preprocessing_payload["std"] = preprocessing["std"].tolist()
        preprocessing_path.write_text(json.dumps(preprocessing_payload))
        payload["modelArtifact"] = {
            "model": str(model_path),
            "preprocessing": str(preprocessing_path),
            "trainedRows": len(rows),
        }
    payload = rounded(payload)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "selected": payload["selected"],
        "audit": payload["audit"]["metrics"],
        "elapsedSeconds": payload["elapsedSeconds"],
        "output": str(output),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
