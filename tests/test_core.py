"""Behavioral checks for sender isolation and padded loss."""
import unittest

import numpy as np
import pandas as pd
import torch

from src.data import FEATURES, build_windows, normalize, arrays
from src.models import Autoencoder, AttentionClassifier


class PipelineTests(unittest.TestCase):
    def test_windows_and_scaler_do_not_mix_senders(self):
        rows = []
        for sender, split, amount in [("A", "train", 2), ("B", "val", 50),
                                      ("C", "test", 500)]:
            for hour in range(3):
                rows.append({"sender": sender, "split": split,
                             "Timestamp": pd.Timestamp("2022-09-01") + pd.Timedelta(hours=hour),
                             "Amount Paid": amount, "From Bank": "1", "To Bank": "2",
                             "Account.1": "9", "Payment Currency": "USD",
                             "Receiving Currency": "USD", "Payment Format": "Wire",
                             "Is Laundering": int(sender == "C" and hour == 2)})
        windows, audit = build_windows(pd.DataFrame(rows), max_len=24)
        self.assertEqual(audit["split_counts"]["test"]["positive"], 1)
        windows, scaler = normalize(windows)
        self.assertAlmostEqual(float(scaler.mean_[0]), float(np.log1p(2)))
        self.assertEqual({w.sender for w in windows if w.split == "train"}, {"A"})
        x, lengths, y, _ = arrays(windows, "test")
        self.assertEqual(x.shape, (1, 24, len(FEATURES)))
        self.assertEqual(lengths.tolist(), [3])
        self.assertEqual(y.tolist(), [1.0])
        self.assertTrue(np.all(x[0, 3:] == 0))

    def test_padding_cannot_change_anomaly_score(self):
        torch.manual_seed(1)
        model = Autoencoder(len(FEATURES), hidden=8)
        x = torch.randn(1, 5, len(FEATURES))
        length = torch.tensor([3])
        first, _ = model.errors(x, length)
        x[:, 3:] = 999
        second, _ = model.errors(x, length)
        self.assertTrue(torch.allclose(first, second, atol=1e-6))
        classifier = AttentionClassifier(len(FEATURES), hidden=8)
        _, weights = classifier(x, length)
        self.assertAlmostEqual(float(weights[0, 3:].detach().sum()), 0.0)
        self.assertAlmostEqual(float(weights[0, :3].detach().sum()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
