"""Re-evaluate a nonzero A/B fusion using validation only for selection."""
import json
from pathlib import Path

import numpy as np
import torch

from src.data import FEATURES, arrays, build_windows, load_sample, normalize
from src.models import AttentionClassifier, Autoencoder
from src.train import best_f2_threshold, metrics, predict

root = Path(__file__).resolve().parents[1]
art = root / "artifacts"
checkpoint = torch.load(art / "model.pt", map_location="cpu", weights_only=False)
frame, _ = load_sample(root / "data/raw/HI-Small_Trans.csv", 10)
windows, _ = build_windows(frame)
windows, scaler = normalize(windows)
assert np.allclose(scaler.mean_, checkpoint["scaler_mean"])
xv, lv, yv, _ = arrays(windows, "val")
xt, lt, yt, test_windows = arrays(windows, "test")

ae = Autoencoder(len(FEATURES))
ae.load_state_dict(checkpoint["autoencoder"])
classifier = AttentionClassifier(len(FEATURES))
classifier.load_state_dict(checkpoint["classifier"])
normal_errors = np.asarray(checkpoint["normal_error"])
coef = checkpoint["calibration_coef"][0][0]
intercept = checkpoint["calibration_intercept"][0]


def component_scores(x, lengths):
    anomaly_raw, token_errors = predict(ae, x, lengths, "cpu", anomaly=True)
    anomaly_rank = np.searchsorted(normal_errors, anomaly_raw, side="right") / len(normal_errors)
    logits, attention = predict(classifier, x, lengths, "cpu")
    probability = 1 / (1 + np.exp(-np.clip(coef * logits + intercept, -50, 50)))
    return anomaly_raw, anomaly_rank, probability, token_errors, attention


_, av, pv, _, _ = component_scores(xv, lv)
options = []
for alpha in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
    score = alpha * av + (1 - alpha) * pv
    threshold = best_f2_threshold(yv, score)
    options.append((metrics(yv, score, threshold)["f2"], alpha, threshold))
_, alpha, threshold = max(options, key=lambda item: (item[0], -item[1]))
print("Validation candidates", options)
print("Selected A weight", alpha, "threshold", threshold)

ae_raw, at, pt, token_error, attention = component_scores(xt, lt)
score = alpha * at + (1 - alpha) * pt
results = json.loads((art / "results.json").read_text(encoding="utf-8"))
results["selection"]["alpha"] = alpha
results["selection"]["fusion_candidates"] = [{"alpha": a, "validation_f2": f2}
                                                for f2, a, _ in options]
results["test"]["combined"] = metrics(yt, score, threshold)
(art / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

positive = np.flatnonzero(yt == 1).tolist()
negative = np.flatnonzero(yt == 0)
ranked_negative = negative[np.argsort(score[negative])[::-1]]
selected = positive + ranked_negative[:1000].tolist() + ranked_negative[-100:].tolist()
cases = []
for i in selected:
    w = test_windows[i]
    cases.append({"sender": w.sender, "window": int(i), "label": int(yt[i]),
                  "transactions": w.transactions, "anomaly_score": float(ae_raw[i]),
                  "anomaly_rank": float(at[i]), "stage_b_probability": float(pt[i]),
                  "combined_score": float(score[i]), "alert": bool(score[i] >= threshold),
                  "attention": attention[i, :int(lt[i])].tolist(),
                  "reconstruction_error": token_error[i, :int(lt[i])].tolist()})
(art / "test_cases.json").write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
checkpoint["alpha"] = alpha
checkpoint["threshold"] = threshold
torch.save(checkpoint, art / "model.pt")
print("Test combined:", results["test"]["combined"])
