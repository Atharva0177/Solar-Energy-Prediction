"""Chronological + cross-site splitting (PRD §11–12).

No random row splits — temporal ordering is preserved *per site*, so every
site contributes its own early history to train and its own late history to
val/test. A single global timestamp cutoff would be wrong here because sites
come online at different dates.

Phase 9 adds the cross-site protocol: held-out SITES for val/test (counts
sized dynamically from the dataset), whose rows never touch training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIME_COL = "timestamp"
GROUP_COL = "site_id"


def chronological_split(
    df: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ``df`` chronologically inside each site into train/val/test.

    Ratios apply to each site's row count (rounded), so sites with different
    coverage windows still get a pure-temporal 70/15/15 cut.
    """
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must be 3 positive fractions summing to 1, got {ratios}")

    parts: list[list[pd.DataFrame]] = [[], [], []]
    for _, sub in df.sort_values([GROUP_COL, TIME_COL]).groupby(GROUP_COL, observed=True):
        n = len(sub)
        cuts = (int(round(n * ratios[0])), int(round(n * (ratios[0] + ratios[1]))))
        parts[0].append(sub.iloc[: cuts[0]])
        parts[1].append(sub.iloc[cuts[0] : cuts[1]])
        parts[2].append(sub.iloc[cuts[1] :])

    return (
        pd.concat(parts[0], ignore_index=True),
        pd.concat(parts[1], ignore_index=True),
        pd.concat(parts[2], ignore_index=True),
    )


def cross_site_split(
    df: pd.DataFrame,
    val_site_frac: float = 0.15,
    test_site_frac: float = 0.15,
    seed: int = 42,
    train_time_ratio: float = 0.70,
) -> dict:
    """Cross-site protocol (PRD §12): held-out sites for val/test.

    Site counts are sized dynamically: ``round(n_sites * frac)``, min 1 —
    42 sites at the defaults gives the PRD's example shape (30/6/6). Sites
    are chosen by a seeded permutation (site ids cluster by campus, so
    ordered slices would bias held-out sets toward one campus).

    Frames returned:

    * ``train`` / ``val_seen`` / ``test_seen`` — train-sites only, cut
      chronologically per site (train_time_ratio, rest split in half), so
      seen-site evaluation mirrors the D-011 protocol.
    * ``val_unseen`` / ``test_unseen`` — the FULL observed history of each
      held-out site (they contribute nothing to training, so no temporal
      restriction is needed and every eval row is genuinely unseen-site).

    Returns ``{"frames": {name: DataFrame}, "sites": {"train"/"val"/"test":
    sorted ids}}``.
    """
    if not 0 < val_site_frac < 1 or not 0 < test_site_frac < 1:
        raise ValueError("site fractions must be in (0, 1)")
    if val_site_frac + test_site_frac >= 1.0:
        raise ValueError("val+test site fractions must leave training sites")

    sites = pd.Series(df[GROUP_COL].unique()).sort_values().tolist()
    n = len(sites)
    n_test = max(1, int(round(n * test_site_frac)))
    n_val = max(1, int(round(n * val_site_frac)))
    if n_val + n_test >= n:
        raise ValueError(f"held-out sites ({n_val}+{n_test}) >= total sites ({n})")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    test_sites = sorted(sites[i] for i in perm[:n_test])
    val_sites = sorted(sites[i] for i in perm[n_test:n_test + n_val])
    train_set = set(sites) - set(test_sites) - set(val_sites)
    train_sites = sorted(train_set)

    is_train = df[GROUP_COL].isin(train_set)
    tr, va_seen, te_seen = chronological_split(
        df[is_train],
        ratios=(train_time_ratio, (1 - train_time_ratio) / 2, (1 - train_time_ratio) / 2),
    )
    frames = {
        "train": tr.reset_index(drop=True),
        "val_seen": va_seen.reset_index(drop=True),
        "test_seen": te_seen.reset_index(drop=True),
        "val_unseen": df[df[GROUP_COL].isin(val_sites)].reset_index(drop=True),
        "test_unseen": df[df[GROUP_COL].isin(test_sites)].reset_index(drop=True),
    }
    return {"frames": frames,
            "sites": {"train": train_sites, "val": val_sites, "test": test_sites}}
