"""Statistical analysis of planet detection probability .npy files.

Covers:
  - Summary statistics (peak / mean det_prob, observable fraction)
  - Distributions of det_prob, integration time, flux contrast
  - Correlation between det_prob and physical quantities
  - Band-to-band comparison per planet
  - Time-series plots of detection probability
  - Faked vs. original planet comparisons

Usage:
    python tools/planet_detection_stats.py              # interactive plots
    python tools/planet_detection_stats.py --out plots/ # save to directory
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# File catalogue
# ---------------------------------------------------------------------------

PLANET_DATA_DIR      = Path(__file__).resolve().parents[2] / "for_rifah" / "for_rifah"
PLANET_DATA_DIR_FAKE = Path(__file__).resolve().parents[2] / "for_rifah" / "for_rifah_fake"

PLANET_FILES = {
    ("47 UMa",    "b", 1): "for_rifah_47_UMa_b_band1",
    ("47 UMa",    "b", 3): "for_rifah_47_UMa_b_band3",
    ("47 UMa",    "b", 4): "for_rifah_47_UMa_b_band4",
    ("47 UMa",    "c", 1): "for_rifah_47_UMa_c_band1",
    ("47 UMa",    "c", 3): "for_rifah_47_UMa_c_band3",
    ("47 UMa",    "c", 4): "for_rifah_47_UMa_c_band4",
    ("47 UMa",    "d", 1): "for_rifah_47_UMa_d_band1",
    ("47 UMa",    "d", 3): "for_rifah_47_UMa_d_band3",
    ("47 UMa",    "d", 4): "for_rifah_47_UMa_d_band4",
    ("55 Cnc",    "d", 1): "for_rifah_55_Cnc_d_band1",
    ("55 Cnc",    "d", 3): "for_rifah_55_Cnc_d_band3",
    ("55 Cnc",    "d", 4): "for_rifah_55_Cnc_d_band4",
    ("HD 114783", "c", 1): "for_rifah_HD_114783_c_band1",
    ("HD 114783", "c", 3): "for_rifah_HD_114783_c_band3",
    ("HD 114783", "c", 4): "for_rifah_HD_114783_c_band4",
    ("HD 134987", "c", 1): "for_rifah_HD_134987_c_band1",
    ("HD 134987", "c", 3): "for_rifah_HD_134987_c_band3",
    ("HD 134987", "c", 4): "for_rifah_HD_134987_c_band4",
    ("HD 154345", "b", 1): "for_rifah_HD_154345_b_band1",
    ("HD 154345", "b", 3): "for_rifah_HD_154345_b_band3",
    ("HD 154345", "b", 4): "for_rifah_HD_154345_b_band4",
    ("HD 160691", "c", 1): "for_rifah_HD_160691_c_band1",
    ("HD 160691", "c", 3): "for_rifah_HD_160691_c_band3",
    ("HD 160691", "c", 4): "for_rifah_HD_160691_c_band4",
    ("HD 190360", "b", 1): "for_rifah_HD_190360_b_band1",
    ("HD 190360", "b", 3): "for_rifah_HD_190360_b_band3",
    ("HD 190360", "b", 4): "for_rifah_HD_190360_b_band4",
    ("HD 192310", "c", 1): "for_rifah_HD_192310_c_band1",
    ("HD 192310", "c", 3): "for_rifah_HD_192310_c_band3",
    ("HD 192310", "c", 4): "for_rifah_HD_192310_c_band4",
    ("HD 217107", "c", 1): "for_rifah_HD_217107_c_band1",
    ("HD 217107", "c", 3): "for_rifah_HD_217107_c_band3",
    ("HD 217107", "c", 4): "for_rifah_HD_217107_c_band4",
    ("HD 87883",  "b", 1): "for_rifah_HD_87883_b_band1",
    ("HD 87883",  "b", 3): "for_rifah_HD_87883_b_band3",
    ("HD 87883",  "b", 4): "for_rifah_HD_87883_b_band4",
    ("eps Eri",   "b", 1): "for_rifah_eps_Eri_b_band1",
    ("eps Eri",   "b", 4): "for_rifah_eps_Eri_b_band4",
    ("pi Men",    "b", 1): "for_rifah_pi_Men_b_band1",
    ("pi Men",    "b", 3): "for_rifah_pi_Men_b_band3",
    ("pi Men",    "b", 4): "for_rifah_pi_Men_b_band4",
    ("ups And",   "d", 1): "for_rifah_ups_And_d_band1",
    ("ups And",   "d", 3): "for_rifah_ups_And_d_band3",
    ("ups And",   "d", 4): "for_rifah_ups_And_d_band4",
}

# Planets whose .npy files were faked or tweaked
FAKED_PLANETS = {
    ("47 UMa",    "b"),   # faked into FoV
    ("47 UMa",    "d"),   # faked into FoV
    ("55 Cnc",    "d"),   # tweaked brighter
    ("HD 87883",  "b"),   # tweaked brighter
    ("HD 134987", "c"),   # tweaked brighter
    ("HD 160691", "c"),   # tweaked brighter
    ("pi Men",    "b"),   # tweaked brighter
}

BAND_LABEL = {1: "Band 1 (575 nm)", 3: "Band 3 (730 nm)", 4: "Band 4 (825 nm)"}
BAND_COLOR = {1: "#1f77b4", 3: "#2ca02c", 4: "#d62728"}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _extract_median(d: dict, key: str, n: int) -> np.ndarray:
    sub = d.get(key, {})
    if isinstance(sub, dict) and "median" in sub:
        return np.array(sub["median"], dtype=float)
    return np.full(n, np.nan)


def build_records() -> pd.DataFrame:
    """Load all .npy files; return a flat DataFrame with one row per (planet, band, timestep)."""
    rows = []
    for (star, planet, band), stem in PLANET_FILES.items():
        faked = (star, planet) in FAKED_PLANETS
        data_dir = PLANET_DATA_DIR_FAKE if faked else PLANET_DATA_DIR
        path = data_dir / f"{stem}.npy"
        if not path.exists():
            warnings.warn(f"Missing: {path.name} (looked in {data_dir.name})")
            continue
        d = np.load(path, allow_pickle=True).item()
        n = len(d["decimal_year"])
        rows.append(pd.DataFrame({
            "star":            star,
            "planet":          planet,
            "band":            band,
            "pl_name":         f"{star} {planet}",
            "faked":           (star, planet) in FAKED_PLANETS,
            "decimal_year":    d["decimal_year"],
            "det_prob":        d["det_probability"],
            "solar_ko":        d["solar_keepout"],
            "gb_ko":           d["GB_keepout"],
            "observable":      ~d["solar_keepout"] & ~d["GB_keepout"],
            "flux_contrast":   _extract_median(d, "flux_contrast", n),
            "phase_angle_deg": _extract_median(d, "phase_angle_deg", n),
            "int_time_hours":  _extract_median(d, "integration_time_hours_opt", n),
        }))
    return pd.concat(rows, ignore_index=True)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (planet, band) with aggregate statistics."""
    grp = df.groupby(["pl_name", "star", "planet", "band", "faked"])

    summary = grp["det_prob"].agg(
        det_prob_max="max",
        det_prob_mean="mean",
        det_prob_std="std",
        det_prob_median="median",
    ).reset_index()

    summary["observable_frac"] = grp["observable"].mean().reset_index()["observable"]
    summary["solar_ko_frac"]   = grp["solar_ko"].mean().reset_index()["solar_ko"]
    summary["gb_ko_frac"]      = grp["gb_ko"].mean().reset_index()["gb_ko"]

    obs = df[df["observable"]]
    obs_det = obs.groupby(["pl_name", "band"])["det_prob"].mean().reset_index()
    obs_det.rename(columns={"det_prob": "det_prob_mean_obs"}, inplace=True)
    summary = summary.merge(obs_det, on=["pl_name", "band"], how="left")

    obs_int = obs.groupby(["pl_name", "band"])["int_time_hours"].median().reset_index()
    obs_int.rename(columns={"int_time_hours": "int_time_median_obs"}, inplace=True)
    summary = summary.merge(obs_int, on=["pl_name", "band"], how="left")

    return summary.sort_values(["pl_name", "band"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_det_prob_timeseries(df: pd.DataFrame, out_dir: Path | None = None):
    planets = df["pl_name"].unique()
    ncols = 3
    nrows = (len(planets) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    for ax, pl in zip(axes, planets):
        sub = df[df["pl_name"] == pl]
        faked = sub["faked"].iloc[0]
        for band, grp in sub.groupby("band"):
            ax.plot(grp["decimal_year"], grp["det_prob"],
                    color=BAND_COLOR[band], label=BAND_LABEL[band], lw=1.4)
            ko = grp["solar_ko"] | grp["gb_ko"]
            ax.fill_between(grp["decimal_year"], 0, 1,
                            where=ko, alpha=0.12, color="grey")
        ax.set_title(pl, fontsize=9, fontweight="bold")
        ax.set_xlabel("Year", fontsize=8)
        ax.set_ylabel("Det. probability", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right")

    for ax in axes[len(planets):]:
        ax.set_visible(False)

    handles = [plt.Line2D([0], [0], color=BAND_COLOR[b], lw=1.4, label=BAND_LABEL[b])
               for b in [1, 3, 4]]
    handles.append(plt.Rectangle((0, 0), 1, 1, fc="grey", alpha=0.3, label="Keepout"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Detection Probability Time Series", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_or_show(fig, out_dir, "timeseries_det_prob.png")


def plot_det_prob_distribution(df: pd.DataFrame, out_dir: Path | None = None):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, band in zip(axes, [1, 3, 4]):
        sub = df[df["band"] == band]
        for label, grp, color, ls in [
            ("Original", sub[~sub["faked"]], "#1f77b4", "-"),
            ("Faked",    sub[ sub["faked"]], "#ff7f0e", "--"),
        ]:
            vals = grp["det_prob"].dropna()
            if vals.empty:
                continue
            ax.hist(vals, bins=30, density=True, alpha=0.45, color=color, label=label)
            kde_x = np.linspace(0, 1, 300)
            try:
                ax.plot(kde_x, stats.gaussian_kde(vals)(kde_x), color=color, lw=2, ls=ls)
            except np.linalg.LinAlgError:
                pass
        ax.set_title(BAND_LABEL[band], fontsize=10)
        ax.set_xlabel("Detection probability", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle("Detection Probability Distribution by Band", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, out_dir, "distribution_det_prob.png")


def plot_correlation_matrix(df: pd.DataFrame, out_dir: Path | None = None):
    cols = {
        "det_prob":        "Det. probability",
        "flux_contrast":   "Flux contrast",
        "phase_angle_deg": "Phase angle (°)",
        "int_time_hours":  "Int. time (h)",
    }
    sub = df[list(cols.keys())].dropna()
    keys   = list(cols.keys())
    labels = list(cols.values())
    n = len(keys)

    fig, axes = plt.subplots(n, n, figsize=(3.5 * n, 3.5 * n))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            ax = axes[i][j]
            if i == j:
                ax.hist(sub[ki], bins=30, color="#4c72b0", alpha=0.7)
            else:
                r, _ = stats.pearsonr(sub[ki], sub[kj])
                ax.scatter(sub[kj], sub[ki], s=2, alpha=0.3, color="#4c72b0")
                ax.text(0.05, 0.92, f"r={r:.2f}", transform=ax.transAxes,
                        fontsize=8, color="crimson")
            if i == n - 1:
                ax.set_xlabel(labels[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=8)
            ax.tick_params(labelsize=6)

    fig.suptitle("Pairwise Correlations", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, out_dir, "correlation_matrix.png")


def plot_band_comparison(summary: pd.DataFrame, out_dir: Path | None = None):
    planets = summary["pl_name"].unique()
    x = np.arange(len(planets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(planets) * 0.9), 5))
    for i, band in enumerate([1, 3, 4]):
        sub = summary[summary["band"] == band].set_index("pl_name")
        heights = [sub.loc[p, "det_prob_max"] if p in sub.index else 0.0 for p in planets]
        ax.bar(x + i * width, heights, width, label=BAND_LABEL[band],
               color=BAND_COLOR[band], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(planets, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Peak detection probability", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)
    ax.set_title("Peak Detection Probability per Planet and Band", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, out_dir, "band_comparison_peak_det.png")


def plot_observable_fraction(summary: pd.DataFrame, out_dir: Path | None = None):
    sub = summary[summary["band"] == 1].sort_values("observable_frac", ascending=False)
    x = np.arange(len(sub))

    fig, ax = plt.subplots(figsize=(max(10, len(sub) * 0.85), 4.5))
    ax.bar(x, sub["observable_frac"], color="#2ca02c", alpha=0.85, label="Observable")
    ax.bar(x, sub["solar_ko_frac"], bottom=sub["observable_frac"].values,
           color="#ff7f0e", alpha=0.75, label="Solar keepout")
    ax.bar(x, sub["gb_ko_frac"],
           bottom=(sub["observable_frac"] + sub["solar_ko_frac"]).values,
           color="#d62728", alpha=0.75, label="GB keepout")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["pl_name"].values, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of time", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.set_title("Observable vs. Keepout Fractions (Band 1)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, out_dir, "observable_fraction.png")


def plot_int_time_vs_det_prob(df: pd.DataFrame, out_dir: Path | None = None):
    obs = df[df["observable"]].copy()
    planets = obs["pl_name"].unique()
    cmap = plt.colormaps["tab20"].resampled(len(planets))
    color_map = {p: cmap(i) for i, p in enumerate(planets)}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, band in zip(axes, [1, 3, 4]):
        sub = obs[obs["band"] == band].dropna(subset=["int_time_hours", "det_prob"])
        for pl, grp in sub.groupby("pl_name"):
            ax.scatter(grp["int_time_hours"], grp["det_prob"],
                       s=10, alpha=0.5, color=color_map[pl], label=pl)
        ax.set_xlabel("Integration time (h)", fontsize=9)
        ax.set_ylabel("Detection probability", fontsize=9)
        ax.set_title(BAND_LABEL[band], fontsize=10)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.05)

    handles = [plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=color_map[p], markersize=6, label=p)
               for p in planets]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Integration Time vs. Detection Probability (observable epochs only)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, out_dir, "int_time_vs_det_prob.png")


def plot_faked_vs_original(df: pd.DataFrame, out_dir: Path | None = None):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
    for ax, band in zip(axes, [1, 3, 4]):
        sub = df[df["band"] == band]
        data_orig  = sub[~sub["faked"]]["det_prob"].dropna().values
        data_faked = sub[ sub["faked"]]["det_prob"].dropna().values
        bp = ax.boxplot([data_orig, data_faked], tick_labels=["Original", "Faked"],
                        patch_artist=True,
                        boxprops=dict(facecolor="#aec7e8", alpha=0.7),
                        medianprops=dict(color="crimson", lw=2))
        bp["boxes"][1].set_facecolor("#ffbb78")
        if len(data_orig) > 1 and len(data_faked) > 1:
            _, pval = stats.mannwhitneyu(data_orig, data_faked, alternative="two-sided")
            ax.set_title(f"{BAND_LABEL[band]}\nMann-Whitney p={pval:.3f}", fontsize=9)
        else:
            ax.set_title(BAND_LABEL[band], fontsize=9)
        ax.set_ylabel("Detection probability", fontsize=9)
        ax.set_ylim(0, 1.1)
    fig.suptitle("Faked vs. Original Planet Detection Probability", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, out_dir, "faked_vs_original.png")

# ---------------------------------------------------------------------------
# Correlation & regression
# ---------------------------------------------------------------------------

PREDICTORS = ["flux_contrast", "phase_angle_deg", "int_time_hours"]
PREDICTOR_LABELS = {
    "flux_contrast":   "Flux contrast",
    "phase_angle_deg": "Phase angle (°)",
    "int_time_hours":  "Int. time (h)",
}


def _ols(X: np.ndarray, y: np.ndarray):
    """OLS via normal equations. Returns (coeffs, fitted, residuals, R2, se, t_stats, p_values).
    X must already include a leading column of ones for the intercept.
    """
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted    = X @ coeffs
    residuals = y - fitted
    n, k      = X.shape
    ss_res    = float(residuals @ residuals)
    ss_tot    = float(((y - y.mean()) ** 2).sum())
    r2        = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    dof       = n - k
    mse       = ss_res / dof
    cov       = mse * np.linalg.pinv(X.T @ X)
    se        = np.sqrt(np.diag(cov))
    t_stats   = coeffs / se
    p_values  = 2 * stats.t.sf(np.abs(t_stats), df=dof)
    return coeffs, fitted, residuals, r2, se, t_stats, p_values


def print_correlation_tables(df: pd.DataFrame):
    """Print Pearson and Spearman correlation tables for det_prob vs predictors, per band."""
    target = "det_prob"
    cols   = [target] + PREDICTORS

    for band in [1, 3, 4]:
        sub = df[df["band"] == band][cols].dropna()
        if sub.empty:
            continue
        print(f"\n=== {BAND_LABEL[band]} — correlation with det_prob (n={len(sub)}) ===")
        header = f"{'Variable':<20} {'Pearson r':>10} {'p-value':>10} {'Spearman rho':>12} {'p-value':>10}"
        print(header)
        print("-" * len(header))
        for pred in PREDICTORS:
            pr, pp = stats.pearsonr(sub[target], sub[pred])
            sr, sp = stats.spearmanr(sub[target], sub[pred])
            print(f"{PREDICTOR_LABELS[pred]:<20} {pr:>10.4f} {pp:>10.4f} {sr:>12.4f} {sp:>10.4f}")


def print_regression_tables(df: pd.DataFrame):
    """OLS regression of det_prob on all predictors, per band. Prints coefficient table."""
    for band in [1, 3, 4]:
        sub = df[df["band"] == band][["det_prob"] + PREDICTORS].dropna()
        if len(sub) < len(PREDICTORS) + 2:
            continue
        y = np.asarray(sub["det_prob"], dtype=float)
        X_raw = np.asarray(sub[PREDICTORS], dtype=float)
        # standardise predictors so coefficients are comparable
        X_mean, X_std = X_raw.mean(axis=0), X_raw.std(axis=0)
        X_std[X_std == 0] = 1.0
        X_scaled = (X_raw - X_mean) / X_std
        X = np.column_stack([np.ones(len(y)), X_scaled])

        coeffs, _, _, r2, se, t_stats, p_values = _ols(X, y)

        print(f"\n=== OLS regression: det_prob ~ predictors  |  {BAND_LABEL[band]}  |  n={len(sub)}  R²={r2:.4f} ===")
        names = ["Intercept"] + [PREDICTOR_LABELS[p] for p in PREDICTORS]
        header = f"{'Variable':<22} {'Coeff':>9} {'Std Err':>9} {'t':>8} {'p-value':>10} {'Sig':>4}"
        print(header)
        print("-" * len(header))
        for name, c, s, t, p in zip(names, coeffs, se, t_stats, p_values):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "."  if p < 0.1 else ""
            print(f"{name:<22} {c:>9.4f} {s:>9.4f} {t:>8.3f} {p:>10.4f} {sig:>4}")
        print("(Predictors are standardised — coefficients show effect per 1 SD change)")


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path | None = None):
    """Side-by-side Pearson r and Spearman rho heatmaps, one column per band."""
    cols = ["det_prob"] + PREDICTORS
    labels = ["Det. prob."] + [PREDICTOR_LABELS[p] for p in PREDICTORS]
    bands = [1, 3, 4]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for col, band in enumerate(bands):
        sub = df[df["band"] == band][cols].dropna()
        for row, (method, fn) in enumerate([
            ("Pearson r",   lambda a, b: stats.pearsonr(a, b)[0]),
            ("Spearman rho",  lambda a, b: stats.spearmanr(a, b)[0]),
        ]):
            n = len(cols)
            mat = np.zeros((n, n))
            for i, ci in enumerate(cols):
                for j, cj in enumerate(cols):
                    mat[i, j] = fn(sub[ci], sub[cj]) if i != j else 1.0

            ax = axes[row][col]
            im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
            ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
            ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                            fontsize=7, color="black" if abs(mat[i,j]) < 0.7 else "white")
            ax.set_title(f"{method}\n{BAND_LABEL[band]}", fontsize=9, fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    fig.suptitle("Correlation Heatmaps by Band", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, out_dir, "correlation_heatmaps.png")


def plot_regression_diagnostics(df: pd.DataFrame, out_dir: Path | None = None):
    """Per-band OLS diagnostics: predicted vs actual, residuals vs fitted, Q-Q plot."""
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))

    for col, band in enumerate([1, 3, 4]):
        sub = df[df["band"] == band][["det_prob"] + PREDICTORS].dropna()
        y     = np.asarray(sub["det_prob"], dtype=float)
        X_raw = np.asarray(sub[PREDICTORS], dtype=float)
        X_mean, X_std = X_raw.mean(axis=0), X_raw.std(axis=0)
        X_std[X_std == 0] = 1.0
        X = np.column_stack([np.ones(len(y)), (X_raw - X_mean) / X_std])

        _, fitted, residuals, r2, _, _, _ = _ols(X, y)

        # row 0: predicted vs actual
        ax = axes[0][col]
        ax.scatter(y, fitted, s=6, alpha=0.3, color=BAND_COLOR[band])
        lim = [min(y.min(), fitted.min()) - 0.02, max(y.max(), fitted.max()) + 0.02]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel("Actual det_prob", fontsize=8)
        ax.set_ylabel("Predicted", fontsize=8)
        ax.set_title(f"{BAND_LABEL[band]}\nPredicted vs Actual  R²={r2:.3f}", fontsize=8, fontweight="bold")

        # row 1: residuals vs fitted
        ax = axes[1][col]
        ax.scatter(fitted, residuals, s=6, alpha=0.3, color=BAND_COLOR[band])
        ax.axhline(0, color="k", lw=1, ls="--")
        ax.set_xlabel("Fitted values", fontsize=8)
        ax.set_ylabel("Residuals", fontsize=8)
        ax.set_title("Residuals vs Fitted", fontsize=8)

        # row 2: Q-Q plot of residuals
        ax = axes[2][col]
        res_std = (residuals - residuals.mean()) / residuals.std()
        (osm, osr), (slope, intercept, _) = stats.probplot(res_std, dist="norm")
        ax.scatter(osm, osr, s=6, alpha=0.4, color=BAND_COLOR[band])
        ql, qh = min(osm), max(osm)
        ax.plot([ql, qh], [slope * ql + intercept, slope * qh + intercept], "k--", lw=1)
        ax.set_xlabel("Theoretical quantiles", fontsize=8)
        ax.set_ylabel("Sample quantiles", fontsize=8)
        ax.set_title("Q-Q Plot of Residuals", fontsize=8)

    fig.suptitle("OLS Regression Diagnostics  (det_prob ~ standardised predictors)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, out_dir, "regression_diagnostics.png")


def plot_regression_coefficients(df: pd.DataFrame, out_dir: Path | None = None):
    """Coefficient plot (standardised) with 95 % CI, one panel per band."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    for ax, band in zip(axes, [1, 3, 4]):
        sub = df[df["band"] == band][["det_prob"] + PREDICTORS].dropna()
        y     = np.asarray(sub["det_prob"], dtype=float)
        X_raw = np.asarray(sub[PREDICTORS], dtype=float)
        X_mean, X_std = X_raw.mean(axis=0), X_raw.std(axis=0)
        X_std[X_std == 0] = 1.0
        X = np.column_stack([np.ones(len(y)), (X_raw - X_mean) / X_std])

        coeffs, _, _, r2, se, _, p_values = _ols(X, y)

        # skip intercept (index 0)
        names  = [PREDICTOR_LABELS[p] for p in PREDICTORS]
        c_plot = coeffs[1:]
        se_plot = se[1:]
        p_plot  = p_values[1:]
        ci95    = 1.96 * se_plot
        y_pos   = np.arange(len(names))

        colors = ["#d62728" if p < 0.05 else "#aec7e8" for p in p_plot]
        ax.barh(y_pos, c_plot, xerr=ci95, color=colors, alpha=0.8,
                error_kw=dict(ecolor="black", capsize=4, lw=1))
        ax.axvline(0, color="black", lw=1, ls="--")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Standardised coefficient", fontsize=9)
        ax.set_title(f"{BAND_LABEL[band]}\nR²={r2:.3f}", fontsize=9, fontweight="bold")

    red_patch  = mpatches.Patch(color="#d62728", alpha=0.8, label="p < 0.05")
    blue_patch = mpatches.Patch(color="#aec7e8", alpha=0.8, label="p ≥ 0.05")
    fig.legend(handles=[red_patch, blue_patch], loc="lower center", ncol=2,
               fontsize=9, frameon=False)
    fig.suptitle("OLS Regression Coefficients ± 95% CI  (standardised predictors)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, out_dir, "regression_coefficients.png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_or_show(fig, out_dir, fname):
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_dir / fname}")
        plt.close(fig)
    else:
        plt.show()


def print_summary_table(summary: pd.DataFrame):
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.float_format", "{:.3f}".format)
    cols = ["pl_name", "band", "faked",
            "det_prob_max", "det_prob_mean", "det_prob_median",
            "observable_frac", "solar_ko_frac", "int_time_median_obs"]
    print(summary[cols].to_string(index=False))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Planet detection statistical analysis")
    parser.add_argument("--out", type=str, default=None,
                        help="Directory to save plots (omit to show interactively)")
    args = parser.parse_args()
    out = Path(args.out) if args.out else None

    print("Loading data …")
    df = build_records()
    summary = build_summary(df)

    print("\n=== Summary statistics ===")
    print_summary_table(summary)

    print("\n=== Correlation tables ===")
    print_correlation_tables(df)

    print("\n=== Regression tables ===")
    print_regression_tables(df)

    print("\nGenerating plots …")
    plot_det_prob_timeseries(df, out)
    plot_det_prob_distribution(df, out)
    plot_correlation_matrix(df, out)
    plot_correlation_heatmap(df, out)
    plot_band_comparison(summary, out)
    plot_observable_fraction(summary, out)
    plot_int_time_vs_det_prob(df, out)
    plot_faked_vs_original(df, out)
    plot_regression_diagnostics(df, out)
    plot_regression_coefficients(df, out)
    print("Done.")
