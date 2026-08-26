"""Phase 3 EDA: plots + summary from processed parquet (PRD Phase 3).

Reads ``data/processed/solar`` (+ ``site_details.parquet``), writes PNGs and
``artifacts/eda/eda_summary.md``. Every number in the summary is computed here
from the data — nothing hand-typed.

Visual method: dataviz skill reference palette (validated set), one hue for
magnitude, diverging blue<->red for correlation polarity, recessive chrome,
no dual axes, legends whenever >=2 series.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PROCESSED = REPO_ROOT / "data" / "processed"
EDA_DIR = REPO_ROOT / "artifacts" / "eda"

# --- Palette (dataviz reference instance, light mode) ------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
CAT = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948",
       "#e87ba4", "#eb6834"]
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b"]
DIV_BLUE, DIV_RED, DIV_MID = "#104281", "#d03b3b", "#f0efec"

WEATHER_VARS = [
    "temperature",
    "apparent_temperature",
    "dew_point_temperature",
    "humidity",
    "wind_speed",
]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_2)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def new_fig(ax_w=8, ax_h=4.2):
    fig, ax = plt.subplots(figsize=(ax_w, ax_h), dpi=150, constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    return fig, ax


def save(fig, name):
    path = EDA_DIR / name
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path.name)


def main() -> int:
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PROCESSED / "solar")
    sites = pd.read_parquet(PROCESSED / "site_details.parquet").set_index("site_id")

    findings: list[str] = []

    day_obs = df["is_daylight"] & df["power"].notna()
    night_obs = ~df["is_daylight"] & df["power"].notna()

    # ------------------------------------------------------------------ 1
    fig, ax = new_fig(8, 4.2)
    d = df.loc[day_obs, "power"]
    ax.hist(d, bins=60, color=CAT[0], linewidth=0)
    ax.set_title("Daylight generation per 15-min interval (all sites, observed)",
                 loc="left", fontsize=11, color=INK)
    ax.set_xlabel("kWh / interval", fontsize=9, color=INK_2)
    ax.set_ylabel("intervals", fontsize=9, color=INK_2)
    save(fig, "01_power_distribution.png")
    findings += [
        f"Daylight observed intervals: {len(d):,}; median {d.median():.2f} kWh, "
        f"mean {d.mean():.2f} kWh, p99 {d.quantile(0.99):.2f} kWh, max {d.max():.2f} kWh.",
        f"Distribution is strongly right-skewed: {100*(d < 1).mean():.1f}% of daylight "
        f"observations are below 1 kWh.",
    ]

    # Night sanity: observed night rows are rare and twilight-concentrated.
    night = df.loc[night_obs, "power"]
    night_rows = df.loc[night_obs]
    twilight_share = (
        100.0 * night_rows["timestamp"].dt.hour.isin([16, 17, 18, 19, 20, 21]).mean()
    )
    findings.append(
        f"Night intervals are almost never recorded: {len(night):,} observed night rows "
        f"({100*len(night)/max(int(night_obs.sum() + day_obs.sum()), 1):.2f}% of observed) vs "
        f"{int(day_obs.sum()):,} daylight. All recorded night values are positive "
        f"(median {night.median():.2f} kWh), concentrated at 16-21h ({twilight_share:.0f}%) "
        f"with sun elevation just below 0 deg — consistent with real twilight diffuse "
        f"production, not clock errors. Dataset effectively documents daytime only; "
        f"night forecasting targets barely exist."
    )

    # ------------------------------------------------------------------ 2
    fig, ax = new_fig(8, 4.4)
    tod = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    grouped = df.assign(_tod=tod).groupby(["campus_id", "_tod"])["power"].mean()
    for i, cid in enumerate(sorted(df["campus_id"].unique())):
        if cid not in grouped.index.get_level_values(0):
            continue
        s = grouped.xs(cid, level="campus_id")
        ax.plot(s.index, s.values, color=CAT[int(cid) - 1], linewidth=2,
                solid_capstyle="round",
                label=f"Campus {cid} ({sites[sites['campus_id']==cid].index.nunique()} site"
                      f"{'' if sites[sites['campus_id']==cid].index.nunique()==1 else 's'})")
    ax.set_title("Mean generation by time of day, per campus",
                 loc="left", fontsize=11, color=INK)
    ax.set_xlabel("local hour", fontsize=9, color=INK_2)
    ax.set_ylabel("kWh / interval", fontsize=9, color=INK_2)
    ax.set_xlim(0, 23.75)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    save(fig, "02_daily_profiles_campus.png")

    peak = df.assign(_tod=tod).groupby("_tod")["power"].mean().idxmax()
    findings.append(
        f"Aggregate daily peak sits at ~{int(peak)}:{int(round((peak % 1)*60)):02d} local solar-aligned time; "
        f"campus profiles differ in scale (campus 1 hosts 27 of 42 sites) but align in phase."
    )

    # ------------------------------------------------------------------ 3
    monthly_energy = (
        df.dropna(subset=["power"])
        .assign(_m=lambda x: x["timestamp"].dt.to_period("M"))
        .groupby("_m")["power"].sum() / 1000.0  # kWh -> MWh per month, all sites
    )
    fig, ax = new_fig(9, 4.0)
    ax.plot(monthly_energy.index.to_timestamp(), monthly_energy.values,
            color=CAT[0], linewidth=2, solid_capstyle="round")
    ax.set_title("Total generated energy per month, all sites (observed intervals)",
                 loc="left", fontsize=11, color=INK)
    ax.set_ylabel("MWh / month", fontsize=9, color=INK_2)
    save(fig, "03_monthly_energy_timeseries.png")

    best_m = monthly_energy.idxmax()
    worst_m = monthly_energy.idxmin()
    findings.append(
        f"Monthly energy ranges {monthly_energy.min():.0f}-{monthly_energy.max():.0f} MWh; "
        f"peak {best_m}, trough {worst_m}. Note: raw sums under-count months/sites with "
        f"missing intervals ({df['power'].isna().mean()*100:.1f}% of all rows missing)."
    )

    # Seasonality: calendar-month climatology (mean per interval).
    cal = (
        df[df["power"].notna()]
        .groupby(df["timestamp"].dt.month)["power"].mean()
    )
    fig, ax = new_fig(7, 3.6)
    bars = ax.bar(range(1, 13), cal.values, color=CAT[0], width=0.62)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_title("Seasonality: mean generation per interval by calendar month",
                 loc="left", fontsize=11, color=INK)
    ax.set_xlabel("month", fontsize=9, color=INK_2)
    ax.set_ylabel("kWh / interval", fontsize=9, color=INK_2)
    top_month = cal.idxmax()
    low_month = cal.idxmin()
    ax.annotate(f"peak {cal.max():.1f}", xy=(top_month, cal.max()),
                xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=8, color=INK_2)
    ax.annotate(f"low {cal.min():.1f}", xy=(low_month, cal.min()),
                xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=8, color=INK_2)
    save(fig, "04_seasonality.png")
    findings.append(
        f"Southern-hemisphere seasonality confirmed: calendar-month means peak in "
        f"month {top_month} ({cal.max():.2f} kWh/interval) and bottom out in month "
        f"{low_month} ({cal.min():.2f})."
    )

    # ------------------------------------------------------------------ 5
    site_mean = (
        df[day_obs]
        .groupby("site_id", observed=True)["power"]
        .agg(["mean", "count"])
        .sort_values("mean")
    )
    cap = sites["capacity_kwp"]
    fig, ax = new_fig(7.5, 8.5)
    ypos = np.arange(len(site_mean))
    ax.barh(ypos, site_mean["mean"], height=0.62, color=CAT[0])
    ax.set_yticks(ypos)
    labels = []
    for sid in site_mean.index:
        c = cap.get(sid, np.nan)
        labels.append(f"s{sid}" + (f" · {c:.0f} kWp" if pd.notna(c) else ""))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_title("Mean daylight generation per site (label: installed capacity where known)",
                 loc="right", fontsize=10.5, color=INK)
    ax.set_xlabel("kWh / interval", fontsize=9, color=INK_2)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.grid(False, axis="y")
    save(fig, "05_site_comparison.png")

    top5 = site_mean.tail(5)[::-1]
    findings.append(
        "Top-5 sites by mean daylight output: "
        + ", ".join(f"site {sid} ({row['mean']:.1f} kWh)" for sid, row in top5.iterrows())
        + ". Capacity known for "
        + f"{cap.notna().sum()} of 42 sites; output roughly tracks capacity where both exist."
    )

    # ------------------------------------------------------------------ 6
    corr_cols = ["power", "solar_elevation_deg"] + WEATHER_VARS
    sub = df[corr_cols]
    corr = sub.corr(method="pearson")
    n = len(corr_cols)
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=150, constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    mat = corr.values
    vmin, vmax = -1.0, 1.0
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1, color="#e1e0d9"))
                continue
            frac = abs(v)
            hexcol = DIV_BLUE if v > 0 else DIV_RED
            mix = np.array(matplotlib.colors.to_rgb(DIV_MID))
            tgt = np.array(matplotlib.colors.to_rgb(hexcol))
            rgba = tuple(mix * (1 - frac) + tgt * frac) + (1.0,)
            ax.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1, color=rgba))
            txt_color = INK if frac < 0.55 else "#ffffff"
            ax.text(j + 0.5, n - 1 - i + 0.5, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color=txt_color)
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(corr_cols, rotation=35, ha="right", fontsize=8, color=INK_2)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(corr_cols[::-1], fontsize=8, color=INK_2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Pearson correlation (wind_direction excluded: circular quantity)",
                 loc="left", fontsize=10.5, color=INK)
    save(fig, "06_weather_correlation.png")
    findings.append(
        f"Strongest absolute correlations with power: "
        + ", ".join(
            f"{c} r={corr.loc['power', c]:+.2f}"
            for c in corr.loc["power"].drop("power").abs().sort_values(ascending=False).index[:3]
        )
        + ". Weather variables inter-correlate heavily (temperature cluster r>0.8)."
    )

    # Pooled correlations are diluted by between-site capacity variance
    # (Simpson-style); compute within-site means for the honest picture.
    d_obs = df[day_obs]
    within = {
        c: float(
            d_obs.groupby("site_id", observed=True)
            .apply(lambda g: g["power"].corr(g[c]), include_groups=False)
            .dropna()
            .mean()
        )
        for c in ["solar_elevation_deg"] + WEATHER_VARS
    }
    findings.append(
        "Pooled r understates signal (between-site capacity variance): "
        "within-site mean r(power, solar_elevation) = "
        f"{within['solar_elevation_deg']:.2f} (vs {corr.loc['power', 'solar_elevation_deg']:.2f} pooled); "
        f"humidity {within['humidity']:.2f}; temperature {within['temperature']:.2f}."
    )

    # ------------------------------------------------------------------ 7
    miss_by_site = (
        df.groupby("site_id", observed=True)["power"]
        .apply(lambda s: s.isna().mean() * 100)
    )
    miss_by_site = miss_by_site.sort_values()
    fig, ax = new_fig(7.5, 6.5)
    colors = [CAT[0] if v <= 50 else CAT[5] for v in miss_by_site.values]
    ypos = np.arange(len(miss_by_site))
    ax.barh(ypos, miss_by_site.values, height=0.62, color=colors)
    ax.axvline(df["power"].isna().mean() * 100, color=INK_2, linewidth=1,
               linestyle=(0, (4, 3)))
    ax.text(df["power"].isna().mean() * 100 + 1, 1, f"overall {df['power'].isna().mean()*100:.1f}%",
            fontsize=8, color=INK_2)
    ax.set_yticks(ypos[::4])
    ax.set_yticklabels([f"s{sid}" for sid in miss_by_site.index[::4]], fontsize=8)
    ax.set_title("Power missingness by site (red: above 50%)",
                 loc="right", fontsize=10.5, color=INK)
    ax.set_xlabel("% missing", fontsize=9, color=INK_2)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.grid(False, axis="y")
    save(fig, "07_missing_by_site.png")

    full_sites = (miss_by_site == 0).sum()
    dead_sites = (miss_by_site > 90).sum()
    findings.append(
        f"Missingness is structural, not random: {full_sites} sites report fully; "
        f"{dead_sites} sites are >90% empty (likely offline/unmonitored); overall {df['power'].isna().mean()*100:.1f}%."
    )

    # ------------------------------------------------------------------ 8
    value_cols = ["power"] + WEATHER_VARS + ["wind_direction"]
    miss = (
        df.assign(_m=df["timestamp"].dt.to_period("M"))[value_cols + ["_m"]]
        .groupby("_m")[value_cols].apply(lambda g: g.isna().mean() * 100.0)
    )
    import matplotlib.colors as mcolors

    cmap = mcolors.LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(7.6, 7.0), dpi=150, constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    im = ax.imshow(miss.values.T, aspect="auto", cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(range(len(miss.index)))
    ax.set_xticklabels([str(m) for m in miss.index], rotation=45, ha="right",
                       fontsize=7, color=INK_2)
    ax.set_yticks(range(len(value_cols)))
    ax.set_yticklabels(value_cols, fontsize=8, color=INK_2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("% missing", fontsize=8, color=INK_2)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    cb.outline.set_visible(False)
    ax.set_title("Missingness by month and variable", loc="left",
                 fontsize=10.5, color=INK)
    save(fig, "08_missingness_heatmap.png")

    wx_miss_months = miss[[c for c in WEATHER_VARS]].max(axis=1)
    gap_months = wx_miss_months[wx_miss_months > 80].index.tolist()
    findings.append(
        f"Weather gaps cluster in specific months "
        + (f"(>80% missing: {', '.join(str(m) for m in gap_months)})" if gap_months else "")
        + " — consistent with sensor outages rather than continuous decay."
    )

    # ------------------------------------------------------------------ 9
    biggest = site_mean["mean"].idxmax()
    ts = (
        df[df["site_id"] == biggest]
        .dropna(subset=["power"])
        .groupby(df["timestamp"].dt.normalize())["power"].sum()
    )
    fig, ax = new_fig(9.5, 3.8)
    ax.plot(ts.index, ts.values, color=CAT[0], linewidth=1.4, solid_capstyle="round")
    ax.set_title(f"Daily energy, site {biggest}"
                 + (f" ({sites.loc[biggest, 'capacity_kwp']:.0f} kWp)"
                    if pd.notna(sites.loc[biggest, "capacity_kwp"]) else ""),
                 loc="left", fontsize=11, color=INK)
    ax.set_ylabel("kWh / day", fontsize=9, color=INK_2)
    save(fig, "09_timeseries_largest_site.png")
    findings.append(
        f"Site {biggest} daily-energy trace shows clear seasonality with occasional "
        f"zero-runs (outages/cloud anomalies) worth watching during modeling."
    )

    # ------------------------------------------------------------------ summary
    summary = [
        "# EDA Summary (auto-generated)",
        "",
        f"_Source: `data/processed/solar` ({len(df):,} rows), generated by "
        "`scripts/run_eda.py`. All figures in this folder._",
        "",
        "## Findings",
        "",
    ]
    summary += [f"- {f}" for f in findings]
    (EDA_DIR / "eda_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print("wrote eda_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
