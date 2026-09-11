"""Reproducible training, validation selection and untouched test evaluation."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import FEATURES, MAX_LEN, arrays, build_windows, load_sample, normalize
from .models import AttentionClassifier, Autoencoder, transfer_encoder


def loader(x, lengths, y, batch_size=256, shuffle=False):
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(lengths),
                            torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def fit_autoencoder(model, x, lengths, y, device, epochs=3):
    normal = y == 0
    batches = loader(x[normal], lengths[normal], y[normal], shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    model.to(device).train()
    history = []
    for _ in range(epochs):
        losses = []
        for xb, lb, _ in batches:
            xb, lb = xb.to(device), lb.to(device)
            loss = model.errors(xb, lb)[0].mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return history


def fit_classifier(model, x, lengths, y, device, epochs=4):
    batches = loader(x, lengths, y, shuffle=True)
    ratio = float((y == 0).sum() / max(1, (y == 1).sum()))
    # Square-root weighting retains a rare-event signal without an unstable 100x loss.
    pos_weight = min(40.0, ratio ** 0.5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.to(device).train()
    history = []
    for _ in range(epochs):
        losses = []
        for xb, lb, yb in batches:
            xb, lb, yb = xb.to(device), lb.to(device), yb.to(device)
            logits, _ = model(xb, lb)
            loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return history, pos_weight


@torch.no_grad()
def predict(model, x, lengths, device, anomaly=False):
    model.to(device).eval()
    scores, token_scores = [], []
    for xb, lb, _ in loader(x, lengths, np.zeros(len(x), dtype="float32"),
                            batch_size=512):
        xb, lb = xb.to(device), lb.to(device)
        score, token = model.errors(xb, lb) if anomaly else model(xb, lb)
        scores.extend(score.cpu().numpy().tolist())
        token_scores.extend(token.cpu().numpy().tolist())
    return np.asarray(scores), np.asarray(token_scores)


def calibrated_probability(validation_logits, validation_y, logits):
    if len(np.unique(validation_y)) < 2:
        raise ValueError("Validation must contain both classes for calibration")
    calibration = LogisticRegression(max_iter=200, random_state=42)
    calibration.fit(validation_logits.reshape(-1, 1), validation_y)
    return calibration.predict_proba(logits.reshape(-1, 1))[:, 1], calibration


def best_f2_threshold(y, scores):
    candidates = np.unique(np.quantile(scores, np.linspace(0, 1, 201)))
    best = (-1.0, 0.5)
    for threshold in candidates:
        predicted = scores >= threshold
        tp = np.sum((y == 1) & predicted)
        fp = np.sum((y == 0) & predicted)
        fn = np.sum((y == 1) & ~predicted)
        f2 = 5 * tp / max(1, 5 * tp + 4 * fn + fp)
        if f2 > best[0]:
            best = (float(f2), float(threshold))
    return best[1]


def metrics(y, scores, threshold):
    p, r, f2, _ = precision_recall_fscore_support(
        y, scores >= threshold, average="binary", beta=2, zero_division=0)
    return {"average_precision": float(average_precision_score(y, scores)),
            "precision": float(p), "recall": float(r), "f2": float(f2),
            "threshold": float(threshold), "alerts": int((scores >= threshold).sum())}


def main(raw_path: str, output_path: str, sample_percent=10, epochs_ae=3, epochs_cls=4):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frame, raw_audit = load_sample(raw_path, sample_percent)
    windows, window_audit = build_windows(frame)
    windows, scaler = normalize(windows)
    splits = {name: arrays(windows, name) for name in ("train", "val", "test")}
    for name, (_, _, y, _) in splits.items():
        if len(np.unique(y)) < 2:
            raise ValueError(f"{name} has only one class; increase sample_percent")
    xtr, ltr, ytr, _ = splits["train"]
    xv, lv, yv, _ = splits["val"]
    xt, lt, yt, test_windows = splits["test"]

    ae = Autoencoder(len(FEATURES))
    ae_history = fit_autoencoder(ae, xtr, ltr, ytr, device, epochs_ae)
    transfer = AttentionClassifier(len(FEATURES))
    transfer_encoder(ae, transfer)
    transfer_history, pos_weight = fit_classifier(transfer, xtr, ltr, ytr,
                                                  device, epochs_cls)
    torch.manual_seed(seed)  # Same starting seed for the from-scratch control.
    baseline = AttentionClassifier(len(FEATURES))
    baseline_history, _ = fit_classifier(baseline, xtr, ltr, ytr,
                                          device, epochs_cls)

    ae_tr, _ = predict(ae, xtr, ltr, device, anomaly=True)
    ae_v, _ = predict(ae, xv, lv, device, anomaly=True)
    ae_t, token_error = predict(ae, xt, lt, device, anomaly=True)
    # Empirical normal-training CDF: comparable 0..1 anomaly scale.
    normal_error = np.sort(ae_tr[ytr == 0])
    anomaly_v = np.searchsorted(normal_error, ae_v, side="right") / len(normal_error)
    anomaly_t = np.searchsorted(normal_error, ae_t, side="right") / len(normal_error)
    transferred_v, _ = predict(transfer, xv, lv, device)
    transferred_t, attention_t = predict(transfer, xt, lt, device)
    base_v, _ = predict(baseline, xv, lv, device)
    base_t, _ = predict(baseline, xt, lt, device)
    prob_v, calibration = calibrated_probability(transferred_v, yv, transferred_v)
    prob_t = calibration.predict_proba(transferred_t.reshape(-1, 1))[:, 1]
    base_prob_v, base_calibration = calibrated_probability(base_v, yv, base_v)
    base_prob_t = base_calibration.predict_proba(base_t.reshape(-1, 1))[:, 1]

    threshold_ae = best_f2_threshold(yv, anomaly_v)
    threshold_base = best_f2_threshold(yv, base_prob_v)
    threshold_transfer = best_f2_threshold(yv, prob_v)
    # Alpha and alert threshold use validation only; test remains untouched.
    best = (-1.0, 0.0, 0.5)
    for alpha in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        score = alpha * anomaly_v + (1 - alpha) * prob_v
        threshold = best_f2_threshold(yv, score)
        f2 = metrics(yv, score, threshold)["f2"]
        if f2 > best[0]:
            best = (f2, alpha, threshold)
    _, alpha, threshold_combo = best
    combined_t = alpha * anomaly_t + (1 - alpha) * prob_t
    results = {
        "device": device, "seed": seed, "sample_percent": sample_percent,
        "raw_audit": raw_audit,
        "window_audit": {k: v for k, v in window_audit.items() if k != "lengths"},
        "length_distribution": np.histogram(window_audit["lengths"], bins=[2, 5, 9, 13, 17, 21, 25])[0].tolist(),
        "length_bins": ["2-4", "5-8", "9-12", "13-16", "17-20", "21-24"],
        "training": {"autoencoder_loss": ae_history, "transfer_loss": transfer_history,
                     "baseline_loss": baseline_history, "pos_weight": pos_weight},
        "selection": {"alpha": alpha, "metric": "F2 on validation",
                      "validation_positive_rate": float(yv.mean())},
        "test_positive_rate": float(yt.mean()),
        "test": {
            "stage_a": metrics(yt, anomaly_t, threshold_ae),
            "stage_b_transfer": metrics(yt, prob_t, threshold_transfer),
            "baseline_from_scratch": metrics(yt, base_prob_t, threshold_base),
            "combined": metrics(yt, combined_t, threshold_combo),
        },
    }
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    # Save all held-out windows for interactive lookup, capped to reduce deploy size.
    positives = [i for i, y in enumerate(yt) if y]
    negatives = [i for i, y in enumerate(yt) if not y]
    selected = positives + negatives[:1000]
    cases = []
    for i in selected:
        w = test_windows[i]
        cases.append({"sender": w.sender, "window": int(i), "label": int(yt[i]),
                      "transactions": w.transactions, "anomaly_score": float(ae_t[i]),
                      "anomaly_rank": float(anomaly_t[i]), "stage_b_probability": float(prob_t[i]),
                      "combined_score": float(combined_t[i]),
                      "alert": bool(combined_t[i] >= threshold_combo),
                      "attention": attention_t[i, :int(lt[i])].tolist(),
                      "reconstruction_error": token_error[i, :int(lt[i])].tolist()})
    with (out / "test_cases.json").open("w", encoding="utf-8") as file:
        json.dump(cases, file, ensure_ascii=False)
    torch.save({"autoencoder": ae.cpu().state_dict(),
                "classifier": transfer.cpu().state_dict(),
                "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                "features": FEATURES, "max_len": MAX_LEN,
                "calibration_coef": calibration.coef_.tolist(),
                "calibration_intercept": calibration.intercept_.tolist(),
                "normal_error": normal_error.tolist(), "alpha": alpha,
                "threshold": threshold_combo}, out / "model.pt")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/HI-Small_Trans.csv")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--sample-percent", type=int, default=10)
    parser.add_argument("--epochs-ae", type=int, default=3)
    parser.add_argument("--epochs-cls", type=int, default=4)
    args = parser.parse_args()
    main(args.raw, args.out, args.sample_percent, args.epochs_ae, args.epochs_cls)
