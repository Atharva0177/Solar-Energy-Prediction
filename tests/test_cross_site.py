"""Phase 9 tests: cross-site split semantics (PRD §12).

The chronological split's temporal-order guarantees live in
``tests/test_data_modules.py``; here we pin the cross-site protocol:
dynamic site counts, disjointness, held-out sites absent from training,
full-history eval frames for unseen sites, seed determinism.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.splits import cross_site_split


def synth_sites(n_sites: int = 20, days: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for sid in range(1, n_sites + 1):
        # stagger site start dates so "full history" is observable
        start = pd.Timestamp("2021-01-01") + pd.Timedelta(days=int(sid % 4))
        ts = pd.date_range(start, periods=96 * days, freq="15min")
        frames.append(pd.DataFrame({
            "site_id": sid, "timestamp": ts,
            "power": rng.random(len(ts)) * (5 + sid),
        }))
    return pd.concat(frames, ignore_index=True)


class TestCrossSiteSplit:
    def test_dynamic_counts_match_fractions(self):
        df = synth_sites(n_sites=20)
        out = cross_site_split(df, val_site_frac=0.15, test_site_frac=0.15, seed=1)
        assert len(out["sites"]["val"]) == max(1, round(20 * 0.15))   # 3
        assert len(out["sites"]["test"]) == 3
        assert len(out["sites"]["train"]) == 14

    def test_prd_example_shape_on_42_sites(self):
        # 42 sites → PRD §12 example shape: ~30 train / 6 val / 6 test
        df = synth_sites(n_sites=42, days=1)
        out = cross_site_split(df)
        assert (len(out["sites"]["train"]),
                len(out["sites"]["val"]),
                len(out["sites"]["test"])) == (30, 6, 6)

    def test_site_sets_disjoint_and_exhaustive(self):
        df = synth_sites(12)
        out = cross_site_split(df, seed=2)
        s = out["sites"]
        assert not set(s["val"]) & set(s["test"])
        assert not set(s["val"]) & set(s["train"])
        assert not set(s["test"]) & set(s["train"])
        assert sorted(s["train"] + s["val"] + s["test"]) == list(range(1, 13))

    def test_heldout_rows_never_in_train_frames(self):
        df = synth_sites(12)
        out = cross_site_split(df, seed=3)
        fr = out["frames"]
        for name in ("train", "val_seen", "test_seen"):
            assert not fr[name]["site_id"].isin(
                out["sites"]["val"] + out["sites"]["test"]).any(), name

    def test_unseen_frames_are_full_history_of_heldout_sites(self):
        df = synth_sites(10, days=4)
        out = cross_site_split(df, seed=4)
        s = out["sites"]
        n_val = df[df.site_id.isin(s["val"])].groupby("site_id").size()
        got = out["frames"]["val_unseen"].groupby("site_id").size()
        assert got.sort_index().tolist() == n_val.sort_index().tolist()

    def test_seen_frames_temporal_order_per_site(self):
        df = synth_sites(8)
        out = cross_site_split(df, seed=5)
        fr = out["frames"]
        tr_va = pd.concat([fr["train"], fr["val_seen"], fr["test_seen"]])
        for sid, sub in tr_va.groupby("site_id", observed=True):
            sub = sub.sort_values("timestamp")
            n = len(sub)
            n_tr = len(fr["train"][fr["train"].site_id == sid])
            # first n_tr rows of the site are exactly its training rows
            np.testing.assert_array_equal(
                sub.timestamp.to_numpy()[:n_tr],
                fr["train"][fr["train"].site_id == sid]
                .sort_values("timestamp").timestamp.to_numpy())

    def test_deterministic_given_seed(self):
        df = synth_sites(16)
        a = cross_site_split(df, seed=11)
        b = cross_site_split(df, seed=11)
        c = cross_site_split(df, seed=12)
        assert a["sites"] == b["sites"]
        assert a["sites"] != c["sites"]

    def test_fraction_validation(self):
        df = synth_sites(6, days=1)
        with pytest.raises(ValueError):
            cross_site_split(df, val_site_frac=0.6, test_site_frac=0.6)
        with pytest.raises(ValueError):
            cross_site_split(df, val_site_frac=1.5, test_site_frac=0.1)
