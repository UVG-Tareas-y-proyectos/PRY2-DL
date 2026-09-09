"""Sender-level sampling and chronological transaction windows.

IBM HI-Small is synthetic banking data, not Guatemalan remittance data. No
post-transaction balance, transaction label or sender ID enters the features.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FORMATS = ("ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire")
FEATURES = ("log_amount", "log_gap_hours", "hour_sin", "hour_cos", "same_bank",
            "currency_mismatch", "new_destination") + tuple("fmt_" + x for x in FORMATS)
MAX_LEN = 24


def bucket(sender: str, salt: str) -> int:
    digest = hashlib.blake2b((salt + sender).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 100


def load_sample(path: str | Path, sample_percent: int = 10) -> tuple[pd.DataFrame, dict]:
    """Read IBM data in chunks; sample whole senders before constructing windows."""
    if not 1 <= sample_percent <= 100:
        raise ValueError("sample_percent must be in [1, 100]")
    chunks = []
    audit = {"raw_rows": 0, "raw_positive_transactions": 0, "sample_rows": 0}
    for chunk in pd.read_csv(path, chunksize=250_000,
                             dtype={"From Bank": str, "Account": str,
                                    "To Bank": str, "Account.1": str}):
        required = {"Timestamp", "From Bank", "Account", "To Bank", "Account.1",
                    "Amount Paid", "Payment Currency", "Receiving Currency",
                    "Payment Format", "Is Laundering"}
        if not required.issubset(chunk):
            raise ValueError(f"Missing IBM columns: {sorted(required - set(chunk))}")
        audit["raw_rows"] += len(chunk)
        audit["raw_positive_transactions"] += int(chunk["Is Laundering"].sum())
        chunk["sender"] = chunk["From Bank"].fillna("") + ":" + chunk["Account"].fillna("")
        keep = chunk["sender"].map(lambda s: bucket(s, "sample-v1") < sample_percent)
        selected = chunk.loc[keep].copy()
        audit["sample_rows"] += len(selected)
        chunks.append(selected)
    if not chunks:
        raise ValueError("No rows selected")
    frame = pd.concat(chunks, ignore_index=True)
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], format="%Y/%m/%d %H:%M", errors="coerce")
    frame = frame.dropna(subset=["Timestamp", "Amount Paid", "sender"])
    frame["split"] = frame["sender"].map(lambda s: "train" if bucket(s, "split-v1") < 70
                                         else ("val" if bucket(s, "split-v1") < 85 else "test"))
    frame = frame.sort_values(["sender", "Timestamp"], kind="stable").reset_index(drop=True)
    audit["sample_senders"] = int(frame["sender"].nunique())
    audit["sample_positive_transactions"] = int(frame["Is Laundering"].sum())
    return frame, audit


@dataclass
class Window:
    sender: str
    split: str
    features: np.ndarray
    label: int
    transactions: list[dict]


def build_windows(frame: pd.DataFrame, max_len: int = MAX_LEN) -> tuple[list[Window], dict]:
    """Create disjoint chronological windows, retaining only length >= 2."""
    windows: list[Window] = []
    dropped_short = 0
    for sender, group in frame.groupby("sender", sort=False):
        group = group.sort_values("Timestamp", kind="stable").copy()
        timestamp = group["Timestamp"]
        gap = timestamp.diff().dt.total_seconds().div(3600).fillna(0).clip(lower=0, upper=24 * 30)
        destination = group["To Bank"].fillna("") + ":" + group["Account.1"].fillna("")
        seen: set[str] = set()
        novel = []
        for item in destination:
            novel.append(float(item not in seen))
            seen.add(item)
        hour = timestamp.dt.hour + timestamp.dt.minute / 60
        matrix = np.column_stack([
            np.log1p(group["Amount Paid"].clip(lower=0).to_numpy(float)),
            np.log1p(gap.to_numpy(float)),
            np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
            (group["From Bank"] == group["To Bank"]).to_numpy(float),
            (group["Payment Currency"] != group["Receiving Currency"]).to_numpy(float),
            np.asarray(novel),
            *[(group["Payment Format"] == kind).to_numpy(float) for kind in FORMATS],
        ]).astype("float32")
        for start in range(0, len(group), max_len):
            part = group.iloc[start:start + max_len]
            if len(part) < 2:
                dropped_short += 1
                continue
            details = [{"timestamp": str(row["Timestamp"]),
                        "amount": float(row["Amount Paid"]),
                        "currency": str(row["Payment Currency"]),
                        "format": str(row["Payment Format"]),
                        "destination": str(row["To Bank"]) + ":" + str(row["Account.1"]),
                        "transaction_label": int(row["Is Laundering"])}
                       for _, row in part.iterrows()]
            windows.append(Window(sender, str(part["split"].iloc[0]),
                                  matrix[start:start + len(part)],
                                  int(part["Is Laundering"].max()), details))
    audit = {"windows": len(windows), "dropped_singleton_windows": dropped_short,
             "lengths": [len(w.features) for w in windows],
             "split_counts": {split: {"total": sum(w.split == split for w in windows),
                                      "positive": sum(w.split == split and w.label for w in windows)}
                              for split in ("train", "val", "test")}}
    assert not ({w.sender for w in windows if w.split == "train"} &
                {w.sender for w in windows if w.split == "test"})
    return windows, audit


def normalize(windows: list[Window]) -> tuple[list[Window], StandardScaler]:
    """Fit scaler only on train sender windows; padding is added afterward."""
    train = [w.features for w in windows if w.split == "train"]
    if not train:
        raise ValueError("No training windows")
    scaler = StandardScaler().fit(np.concatenate(train))
    for window in windows:
        window.features = scaler.transform(window.features).astype("float32")
    return windows, scaler


def arrays(windows: list[Window], split: str, max_len: int = MAX_LEN):
    selected = [w for w in windows if w.split == split]
    x = np.zeros((len(selected), max_len, len(FEATURES)), dtype="float32")
    lengths = np.zeros(len(selected), dtype="int64")
    y = np.zeros(len(selected), dtype="float32")
    for i, window in enumerate(selected):
        length = min(len(window.features), max_len)
        x[i, :length] = window.features[:length]
        lengths[i] = length
        y[i] = window.label
    return x, lengths, y, selected
