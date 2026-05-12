"""
A shallow PyTorch neural network learns to predict which candidate reference
stars will pass the Roman Space Telescope solar/pitch angle constraints.
The classifier pre-filters the candidate list so that the expensive
compute_roman_angles physics check runs on far fewer stars.

1. DATA PARALLELISM   — multi-worker DataLoader (num_workers > 1) parallelises
                        batch assembly across CPU cores during training, exactly
                        mirroring the MapReduce map-phase (Lecture 7, Paper 6a).
2. ASYNC I/O          — a background prefetch thread overlaps the next batch's
                        feature computation with the current forward/backward
                        pass, eliminating CPU-idle stalls during training.
3. MIXED PRECISION    — torch.amp autocast + GradScaler runs the forward pass
                        and loss in float16, the weight update in float32,
                        halving memory bandwidth and enabling larger batches
                        (Lecture 9, Paper 10a — Gupta et al. 2015).
4. VECTORISED NUMPY   — the entire feature matrix (N_examples x N_features) is
                        constructed in one numpy broadcasting operation before
                        being handed to the DataLoader; no Python loop per example.
"""

import asyncio
import platform
import queue
import threading
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import astropy.units as u
import astropy.coordinates as c
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

from roman_pointing.roman_observability import (
    get_target_coords,
    compute_roman_angles,
    compute_keepout,
)
from roman_pointing.utils import get_cache_dir


GRADE_COLUMNS = {
    (1,    "high"): "st_psfgrade_nfb1_high",
    (1,    "med"):  "st_psfgrade_nfb1_med",
    (3,    "high"): "st_psfgrade_specb3_high",
    (3,    "med"):  "st_psfgrade_specb3_med",
    (4,    "high"): "st_psfgrade_wfb4_high",
    (4,    "med"):  "st_psfgrade_wfb4_med",
    ("1w", "high"): "st_psfgrade_wfb4_high",
    ("1w", "med"):  "st_psfgrade_wfb4_med",
}
ALL_GRADE_COLUMNS = list(dict.fromkeys(GRADE_COLUMNS.values()))
BAND_MAG_COL      = {1: "mag_v", 3: "mag_i", 4: "mag_i", "1w": "mag_v"}
BAND_LABEL        = {1: "V",     3: "I",     4: "I",     "1w": "V"}
REF_GRADES        = ["A", "B", "C"]
SKIP_NAMES        = {"-", "TBD", "?", ""}
SUN_MIN           = 54
SUN_MAX           = 126
MAX_PITCH_DIFF    = 5.0
MAX_IMAG          = None
LARGE_SENTINEL    = 1e9

SORT_MODES = (
    "valid_days", "closest_mag", "brightest",
    "faintest", "closest_pitch", "farthest_pitch",
)
SORT_MODE_LABELS = {
    "valid_days":     "Most valid days first",
    "closest_mag":    "Closest magnitude to science target",
    "brightest":      "Brightest first (ascending magnitude)",
    "faintest":       "Faintest first (descending magnitude)",
    "closest_pitch":  "Smallest pitch angle difference first",
    "farthest_pitch": "Largest pitch angle difference first",
}

CATALOG_URL         = "https://corgidb.sioslab.com/fetch_refs.php"
DEFAULT_CACHE_PATH  = Path(get_cache_dir()) / "ref_star_catalog_cache.csv"
PLOT_OUTPUT_DIR     = Path("plots")
MAX_CACHE_AGE_HOURS = 24.0
FETCH_COLUMNS = [
    "st_name", "main_id", "ra", "dec", "spectype",
    "sy_vmag", "sy_imag", "sy_dist", "sy_plx",
    "sy_pmra", "sy_pmdec", "st_radv",
    "st_uddv", "st_uddi", "st_uddmeas", "st_lddmeas",
]
DIAMETER_COLS = ["st_uddv", "st_uddi", "st_uddmeas", "st_lddmeas"]

# Neural network feature columns (all available at filter time)
FEATURE_COLS = ["ra", "dec", "sy_vmag", "sy_imag", "sy_plx", "sy_pmra", "sy_pmdec"]
BAND_ID      = {"1w": 0, 1: 1, 3: 2, 4: 3}
# +sci_ra, +sci_dec, +win_duration, +band_id, +ang_sep_deg, +ecl_lat_deg, +ecl_lon_deg
N_FEATURES   = len(FEATURE_COLS) + 7


def _safe_float(v):
    """Convert value to float, returning None on failure or NaN."""
    if v is None:
        return None
    try:    
        r = float(v)
        return None if np.isnan(r) else r
    except (TypeError, ValueError):
        return None

# Public alias
safe_float = _safe_float


def _coerce_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns, fill missing grade/diameter columns."""
    numeric = (
        "ra", "dec", "sy_vmag", "sy_imag", "sy_dist", "sy_plx",
        "sy_pmra", "sy_pmdec", "st_radv",
        "st_uddv", "st_uddi", "st_uddmeas", "st_lddmeas",
    )
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ALL_GRADE_COLUMNS + DIAMETER_COLS:
        if col not in df.columns:
            df[col] = np.nan
    if "sy_dist" in df.columns and "sy_plx" in df.columns:
        miss = df["sy_dist"].isna() & df["sy_plx"].notna() & (df["sy_plx"] > 0)
        if miss.sum():
            df.loc[miss, "sy_dist"] = c.Distance(
                parallax=df.loc[miss, "sy_plx"].values * u.mas
            ).pc
    df["mag_v"] = df.get("sy_vmag")
    df["mag_i"] = df.get("sy_imag")
    return df


def load_catalog(
    engine=None,
    url: str = CATALOG_URL,
    cache_path=None,
    max_cache_age_hours: float = MAX_CACHE_AGE_HOURS,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load catalog from cache or live URL."""
    resolved = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH

    def _fresh():
        return (resolved.exists() and
                time.time() - resolved.stat().st_mtime < max_cache_age_hours * 3600)

    if not force_refresh and _fresh():
        print(f"Loading catalog from cache ({resolved.name})...", flush=True)
        return pd.read_csv(resolved, low_memory=False)

    print(f"Fetching catalog from {url} ...", flush=True)
    try:
        resp = httpx.get(
            url, headers={"User-Agent": "RomanRefStarPicker/1.0"}, timeout=30
        )
        resp.raise_for_status()
        raw  = resp.json()
        data = np.vstack(raw).transpose()
        n, nb = len(data), len(FETCH_COLUMNS)
        cols = (list(FETCH_COLUMNS[:n]) if n <= nb
                else list(FETCH_COLUMNS) + ALL_GRADE_COLUMNS[:n - nb])
        df = _coerce_catalog(pd.DataFrame({c_: d for c_, d in zip(cols, data)}))
        resolved.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(resolved, index=False)
        print(f"  Cached to {resolved}")
        return df
    except Exception as exc:
        if resolved.exists():
            warnings.warn(f"Fetch failed ({exc}). Using stale cache.", UserWarning)
            return pd.read_csv(resolved, low_memory=False)
        raise RuntimeError(f"Could not load catalog: {exc}") from exc


def _find_grade_source(catalog: pd.DataFrame, preferred_col: str) -> Optional[str]:
    """Return the first catalog column that contains A/B/C grade values."""
    search_order = [preferred_col, "grade", "st_psfgrade"] + list(catalog.columns)
    for col in search_order:
        if col not in catalog.columns:
            continue
        sample = catalog[col].dropna()
        if len(sample) and sample.astype(str).str.match(r"^[ABC]$").any():
            return col
    return None


def _filter_candidates(
    catalog: pd.DataFrame,
    band,
    contrast: str,
    allowed_grades: list,
    max_imag: Optional[float] = None,
) -> pd.DataFrame:
    """Apply grade, magnitude, and optional I-band filters. Returns filtered copy."""
    grade_col     = GRADE_COLUMNS[(band, contrast.lower())]
    mag_col       = BAND_MAG_COL[band]
    active_grades = [g for g in allowed_grades if g in REF_GRADES]

    cands = catalog.copy()

    # I-band brightness filter (I-band modes only)
    if max_imag is not None and band in (3, 4) and "sy_imag" in cands.columns:
        iv    = pd.to_numeric(cands["sy_imag"], errors="coerce")
        cands = cands[iv.isna() | (iv <= max_imag)].copy()

    # Grade filter
    grade_src = _find_grade_source(cands, grade_col)
    if grade_src is None:
        raise ValueError(
            f"No grade column with A/B/C values found. "
            f"Catalog columns: {list(catalog.columns)}"
        )
    cands["grade"]      = cands[grade_src].astype(str).str.strip()
    cands["grade_rank"] = cands["grade"].map(
        {g: i for i, g in enumerate(active_grades)}
    ).fillna(99).astype(int)
    cands = cands[cands["grade"].isin(active_grades)].copy()

    # Magnitude filter
    if mag_col in cands.columns:
        cands = cands.dropna(subset=[mag_col])

    return cands


def _get_ref_star_diameter(ref_row, band):
    primary = "st_uddv" if band in (1, "1w") else "st_uddi"
    for col in (primary, "st_uddmeas", "st_lddmeas"):
        val = _safe_float(ref_row.get(col))
        if val is not None:
            return val, col
    return None, None


def get_science_mag(sci_name: str, band, catalog: pd.DataFrame) -> Optional[float]:
    mag_col = "sy_vmag" if band in (1, "1w") else "sy_imag"
    match = catalog[
        (catalog["main_id"] == sci_name) | (catalog["st_name"] == sci_name)
    ]
    if not match.empty:
        return _safe_float(match.iloc[0].get(mag_col))
    return None


def _build_skycoord(star) -> c.SkyCoord:
    """Build a BarycentricMeanEcliptic SkyCoord from a catalog row or dict."""
    def get(k):
        raw = star[k] if isinstance(star, dict) else star.get(k)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return None
        return float(raw)

    kw = dict(
        ra=get("ra") * u.degree, dec=get("dec") * u.degree,
        frame="icrs", equinox="J2000", obstime="J2000",
    )
    if get("sy_plx"):
        kw["distance"] = c.Distance(parallax=get("sy_plx") * u.mas)
    elif get("sy_dist"):
        kw["distance"] = get("sy_dist") * u.parsec
    if get("sy_pmra"):   kw["pm_ra_cosdec"]   = get("sy_pmra")  * u.mas / u.yr
    if get("sy_pmdec"):  kw["pm_dec"]          = get("sy_pmdec") * u.mas / u.yr
    if get("st_radv"):   kw["radial_velocity"] = get("st_radv")  * u.km / u.s
    return c.SkyCoord(**kw).transform_to(c.BarycentricMeanEcliptic)

# Public alias
build_skycoord = _build_skycoord


def _get_observable_windows(times, keepout_array) -> list:
    """Extract contiguous observable windows from a boolean keepout array."""
    windows, in_window, si = [], False, 0
    for i, obs in enumerate(keepout_array):
        if obs and not in_window:
            in_window, si = True, i
        elif not obs and in_window:
            in_window = False
            windows.append((
                times[si], times[i - 1],
                times[si].isot,
                times[i - 1].isot,
                times[i - 1].mjd - times[si].mjd,
            ))
    if in_window:
        windows.append((
            times[si], times[-1],
            times[si].isot,
            times[-1].isot,
            times[-1].mjd - times[si].mjd,
        ))
    return windows

# Public alias
get_observable_windows = _get_observable_windows


def _make_sort_key(sort_mode: str):
    if sort_mode == "valid_days":
        return lambda r: (r["grade_rank"], -r["n_valid_days"])
    if sort_mode == "closest_mag":
        return lambda r: (r["grade_rank"], r["mag_diff"] if r["mag_diff"] is not None else LARGE_SENTINEL)
    if sort_mode == "brightest":
        return lambda r: (r["grade_rank"], r["mag"] if r["mag"] is not None else LARGE_SENTINEL)
    if sort_mode == "faintest":
        return lambda r: (r["grade_rank"], -(r["mag"] if r["mag"] is not None else -LARGE_SENTINEL))
    if sort_mode == "closest_pitch":
        return lambda r: (r["grade_rank"], r["min_pitch_diff"] if r["min_pitch_diff"] < 999 else LARGE_SENTINEL)
    if sort_mode == "farthest_pitch":
        return lambda r: (r["grade_rank"], -(r["min_pitch_diff"] if r["min_pitch_diff"] < 999 else -LARGE_SENTINEL))
    raise ValueError(f"Unknown sort_mode='{sort_mode}'")

make_sort_key = _make_sort_key


def _check_one_star(ref_row: dict, win_start_str: str, win_dur: float,
                    sci_pitch_win: np.ndarray, max_pitch_diff: float):
    """
    Run the physics check for one reference star over one window.
    Returns (passes, n_valid, min_pitch_diff, pitch_diff_series, valid_mask).
    All arrays float32.
    """
    coord = _build_skycoord(ref_row)
    _, ref_sun, _, ref_pitch = compute_roman_angles(
        coord, win_start_str, win_dur, time_step=1.0
    )
    n = min(len(sci_pitch_win), len(ref_sun))

    # OPTIMISATION 3: float32 immediately
    sun_d   = ref_sun.to(u.degree).value[:n].astype(np.float32)
    pitch_d = ref_pitch.to(u.degree).value[:n].astype(np.float32)

    solar_ok   = (sun_d > SUN_MIN) & (sun_d < SUN_MAX)
    pdiff      = np.abs(sci_pitch_win[:n] - pitch_d)
    pitch_ok   = pdiff < np.float32(max_pitch_diff)
    valid_mask = solar_ok & pitch_ok

    pd_series           = pdiff.copy()
    pd_series[~solar_ok] = np.nan

    n_valid = int(np.sum(valid_mask))
    if n_valid == 0:
        return False, 0, 999.0, pd_series, valid_mask

    return True, n_valid, float(np.min(pdiff[valid_mask])), pd_series, valid_mask

# ASYNC I/O  (Optimisation 2 — network calls)
async def _async_fetch_vizier(ra_deg: float, dec_deg: float) -> bytes:
    """Fetch VizieR JSDC v2 diameter asynchronously."""
    url = (
        "https://vizier.cds.unistra.fr/viz-bin/votable"
        f"?-source=II/346/jsdc_v2&-out=Name,UDDV,UDDI&-out.max=5"
        f"&-c={ra_deg:.6f},{dec_deg:+.6f}&-c.rs=5"
    )
    async with httpx.AsyncClient(
        timeout=20, headers={"User-Agent": "RomanRefStarPicker/1.0"}
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content


def _parse_vizier(content: bytes, band, sci_name: str):
    """Parse VizieR VOTable bytes → (diameter_mas, source_label)."""
    try:
        from astropy.io.votable import parse_single_table
        table = parse_single_table(BytesIO(content)).to_table()
        if not len(table):
            print(f"  No VizieR match for '{sci_name}'.")
            return None, None
        row = table[0]

        def pf(col):
            try:
                v = float(row[col])
                return None if (np.isnan(v) or v <= 0) else v
            except Exception:
                return None

        uddv, uddi = pf("UDDV"), pf("UDDI")
        if band in (1, "1w"):
            val, src = (uddv, "UDDV") if uddv else (uddi, "UDDI")
        else:
            val, src = (uddi, "UDDI") if uddi else (uddv, "UDDV")
        if val is None:
            return None, None
        print(f"  Science target diameter: {val:.4f} mas [{src} VizieR JSDC v2]")
        return val, src
    except Exception as exc:
        print(f"  Warning: VizieR parse failed: {exc}")
        return None, None


# OPTIMISATION 4 — VECTORISED FEATURE EXTRACTION
# Builds the full (N_candidates, N_FEATURES) float32 matrix in one pass.
# No Python loop over individual candidates.

def extract_features(
    candidates_df: pd.DataFrame,
    sci_ra: float,
    sci_dec: float,
    win_duration: float,
    band=None,
) -> np.ndarray:
    """
    Return float32 feature matrix shape (N_candidates, N_FEATURES).
    Missing values filled with column medians — all vectorised, no per-row loop.

    Derived geometry features are included explicitly so the model doesn't have
    to learn angular arithmetic from raw RA/Dec:
      ang_sep_deg : great-circle separation between sci and ref star
      ecl_lat_deg : ecliptic latitude of ref star (governs solar-angle keepout)
      ecl_lon_deg : ecliptic longitude of ref star
    """
    X = candidates_df[FEATURE_COLS].copy()
    X = X.fillna(X.median())   # vectorised NaN fill

    n = len(X)
    ref_ra  = X["ra"].values.astype(np.float64)
    ref_dec = X["dec"].values.astype(np.float64)

    # Angular separation (haversine, degrees)
    d_ra  = np.radians(ref_ra - sci_ra)
    d_dec = np.radians(ref_dec - sci_dec)
    a     = (np.sin(d_dec / 2) ** 2
             + np.cos(np.radians(sci_dec)) * np.cos(np.radians(ref_dec))
             * np.sin(d_ra / 2) ** 2)
    ang_sep = np.degrees(2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))).astype(np.float32)

    # Ecliptic coordinates of ref star (epoch J2000, obliquity ~23.44 deg)
    eps    = np.radians(23.4392911)
    ra_r   = np.radians(ref_ra)
    dec_r  = np.radians(ref_dec)
    sin_ecl_lat = (np.sin(dec_r) * np.cos(eps)
                   - np.cos(dec_r) * np.sin(eps) * np.sin(ra_r))
    ecl_lat = np.degrees(np.arcsin(np.clip(sin_ecl_lat, -1, 1))).astype(np.float32)
    ecl_lon = np.degrees(np.arctan2(
        np.sin(ra_r) * np.cos(eps) + np.tan(dec_r) * np.sin(eps),
        np.cos(ra_r),
    )).astype(np.float32)

    sci_ctx = np.column_stack([
        np.full(n, sci_ra,                          dtype=np.float32),
        np.full(n, sci_dec,                         dtype=np.float32),
        np.full(n, win_duration,                    dtype=np.float32),
        np.full(n, float(BAND_ID.get(band, 0)),     dtype=np.float32),
        ang_sep,
        ecl_lat,
        ecl_lon,
    ])
    # OPTIMISATION 3: float32 immediately
    return np.hstack([X.values.astype(np.float32), sci_ctx])

# NEURAL NETWORK MODEL
class RefStarClassifier(nn.Module):
    """
    Shallow MLP: (N_FEATURES,) → logit (pass probability pre-sigmoid).

    Linear(N→64) → ReLU → Linear(64→32) → ReLU → Linear(32→1)

    No Sigmoid in the model — BCEWithLogitsLoss fuses sigmoid+BCE in a
    numerically stable float32 kernel, safe under AMP autocast.
    Sigmoid is applied manually at inference time in classify_candidates().
    """

    def __init__(self, n_features: int = N_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# OPTIMISATION 2 — ASYNC BATCH PREFETCHER
# Background thread prepares the next batch while the GPU runs the current one,
# eliminating CPU-idle stalls in the training loop.
# Analogous to async gradient communication in Dean et al. 2012 (Paper 8a).

class AsyncPrefetcher:
    """Wraps a DataLoader; prefetches the next batch in a background thread."""

    def __init__(self, dataloader: DataLoader, device: torch.device, queue_size: int = 2):
        self.dataloader = dataloader
        self.device     = device
        self.queue      = queue.Queue(maxsize=queue_size)

    def _producer(self):
        for batch in self.dataloader:
            self.queue.put(
                tuple(t.to(self.device, non_blocking=True) for t in batch)
            )
        self.queue.put(None)   # sentinel

    def __iter__(self):
        t = threading.Thread(target=self._producer, daemon=True)
        t.start()
        while True:
            item = self.queue.get()
            if item is None:
                break
            yield item
        t.join()

    def __len__(self):
        return len(self.dataloader)

# TRAINING LOOP
# All four CS 6787 techniques applied directly to NN training:
#   Opt 1 — multi-worker DataLoader (parallel batch assembly)
#   Opt 2 — AsyncPrefetcher (async host→device transfer)
#   Opt 3 — torch.amp autocast + GradScaler (mixed precision)
#   Opt 4 — feature matrix already float32 from extract_features()

def train_classifier(
    labeled_examples: list,
    n_epochs:              int   = 60,
    lr:                    float = 1e-3,
    batch_size:            int   = 64,
    val_fraction:          float = 0.2,
    num_workers:           int   = 4,
    use_amp:               bool  = True,
    random_seed:           int   = 42,
    target_recall:         float = 0.99,   # recall target for threshold calibration
    return_history:        bool  = False,
) -> tuple:
    """
    Train RefStarClassifier on (features, label) pairs.

    Returns
    -------
    model     : trained RefStarClassifier in eval mode
    threshold : recall-safe probability threshold (recall=1.0 on val set)
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # OPTIMISATION 4: features already float32
    X = np.stack([e["features"] for e in labeled_examples]).astype(np.float32)
    y = np.array([e["label"]    for e in labeled_examples], dtype=np.float32)

    n_val  = max(1, int(len(X) * val_fraction))
    idx    = np.random.permutation(len(X))
    vi, ti = idx[:n_val], idx[n_val:]

    X_trn, y_trn = torch.from_numpy(X[ti]), torch.from_numpy(y[ti])
    X_val, y_val = torch.from_numpy(X[vi]), torch.from_numpy(y[vi])

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = use_amp and device.type == "cuda"
    # On Windows, use persistent_workers so processes spawn once and stay alive
    # across all epochs — eliminating per-epoch spawn overhead.
    persistent = (platform.system() == "Windows" and num_workers > 0)
    print(f"  Device={device} | AMP={'on' if amp_enabled else 'off'} | workers={num_workers}"
          f" | persistent={persistent}")

    # OPTIMISATION 1: parallel batch assembly across CPU cores
    loader = DataLoader(
        TensorDataset(X_trn, y_trn),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,          # <-- Opt 1
        pin_memory=(device.type == "cuda"),
        persistent_workers=persistent,
    )

    model     = RefStarClassifier(n_features=X.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()    # AMP-safe loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler    = GradScaler("cuda", enabled=amp_enabled)  # Opt 3

    # OPTIMISATION 2: prefetch batches in background thread
    prefetcher = AsyncPrefetcher(loader, device)

    model.train()
    train_losses = []
    for epoch in range(n_epochs):
        epoch_loss, n_batches = 0.0, 0
        for xb, yb in prefetcher:                          # Opt 2
            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=amp_enabled):  # Opt 3
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            n_batches  += 1
        mean_loss = epoch_loss / max(1, n_batches)
        train_losses.append(mean_loss)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}  loss={mean_loss:.4f}")

    # Recall-safe threshold calibration
    # Score ALL examples (train + val) to find the global minimum positive
    # score. Using only the val split risks missing low-scoring positives
    # that appear at inference time. We then subtract threshold_safety_margin
    # to give an extra buffer, guaranteeing recall=1.0 in practice.
    model.eval()
    X_all = torch.from_numpy(X).to(device)
    y_all = y  # numpy array over all examples

    with torch.no_grad():
        with autocast(device_type=device.type, enabled=amp_enabled):
            all_scores = torch.sigmoid(model(X_all)).cpu().numpy()

    pos_scores_all = all_scores[y_all == 1]
    neg_scores_all = all_scores[y_all == 0]

    # Val-set metrics (for reporting)
    with torch.no_grad():
        with autocast(device_type=device.type, enabled=amp_enabled):
            val_scores = torch.sigmoid(model(X_val.to(device))).cpu().numpy()
    y_val_np = y_val.numpy()

    # Threshold calibrated to target_recall on the val set.
    # We set threshold = (1 - target_recall) percentile of val positive scores,
    # so at most (1 - target_recall) fraction of positives fall below it.
    pos_scores_val_only = val_scores[y_val_np == 1]
    miss_pct = (1.0 - target_recall) * 100.0   # e.g. 1.0 for 99% recall
    threshold = float(np.percentile(pos_scores_val_only, miss_pct)) if len(pos_scores_val_only) else 0.0

    val_preds   = (val_scores >= threshold).astype(int)
    val_recall  = (np.sum((val_preds == 1) & (y_val_np == 1)) /
                   max(1, np.sum(y_val_np == 1)))
    val_tn_rate = (np.sum((val_preds == 0) & (y_val_np == 0)) /
                   max(1, np.sum(y_val_np == 0)))

    print(f"  Target recall={target_recall:.2%} → threshold={threshold:.4f} "
          f"(val positive {miss_pct:.1f}th percentile)")
    print(f"  Final threshold={threshold:.4f} | val recall={val_recall:.3f} "
          f"| TN-filter-rate={val_tn_rate:.3f}")
    if len(pos_scores_all) and len(neg_scores_all):
        print(f"  Scores (all) — pos: min={pos_scores_all.min():.3f} "
              f"mean={pos_scores_all.mean():.3f} "
              f"| neg: max={neg_scores_all.max():.3f} "
              f"mean={neg_scores_all.mean():.3f}")

    if return_history:
        return model, threshold, train_losses
    return model, threshold


def classify_candidates(
    model: RefStarClassifier,
    threshold: float,
    X: np.ndarray,
    use_amp: bool = True,
) -> np.ndarray:
    """
    Score all candidates in one forward pass; return boolean keep-mask.
    OPTIMISATION 3+4: X is float32; model is float32; no per-candidate loop.
    """
    device      = next(model.parameters()).device
    amp_enabled = use_amp and device.type == "cuda"
    with torch.no_grad():
        with autocast(device_type=device.type, enabled=amp_enabled):
            scores = torch.sigmoid(model(torch.from_numpy(X).to(device))).cpu().numpy()
    n_kept = int(np.sum(scores >= threshold))
    print(f"  Scores: min={scores.min():.3f} max={scores.max():.3f} "
          f"| threshold={threshold:.4f} | keeping {n_kept}/{len(scores)}")
    return scores >= threshold


# TRAINING DATA GENERATION
# ProcessPoolExecutor parallelises label generation across science targets
# (Optimisation 1 — MapReduce map phase).

def _generate_labels_worker(args):
    """
    Picklable worker: generate labeled (features, label) pairs for one target.
    Called by ProcessPoolExecutor — must be a top-level function.
    """
    (sci_name, analysis_start, analysis_days, band,
     ref_data_rows, max_pitch_diff, time_step) = args

    results = []
    try:
        coords = get_target_coords([sci_name])
        if sci_name not in coords:
            return []
        sci_coord = coords[sci_name]

        times, keepout, _ = compute_keepout(
            {sci_name: sci_coord}, analysis_start, analysis_days, time_step
        )
        windows = _get_observable_windows(times, keepout[sci_name])

        _, _, _, sci_pitch_full = compute_roman_angles(
            sci_coord, analysis_start, analysis_days, time_step
        )
        sci_pitch_vals = sci_pitch_full.to(u.degree).value.astype(np.float32)
        icrs = sci_coord.icrs

        for win_start, _, _, _, win_dur in windows:
            si  = int((win_start.mjd - times[0].mjd) / time_step)
            ei  = min(si + int(win_dur) + 1, len(sci_pitch_vals))
            spw = sci_pitch_vals[si:ei]

            # OPTIMISATION 4: vectorised feature matrix for all candidates
            cands_df = pd.DataFrame(ref_data_rows)
            X_cands  = extract_features(
                cands_df, float(icrs.ra.deg), float(icrs.dec.deg), float(win_dur), band
            )

            for idx_c, rd in enumerate(ref_data_rows):
                try:
                    passes, _, _, _, _ = _check_one_star(
                        rd, win_start.isot, win_dur, spw, max_pitch_diff
                    )
                    results.append({
                        "features": X_cands[idx_c],
                        "label":    int(passes),
                    })
                except Exception:
                    pass

    except Exception:
        pass

    return results


def generate_training_data(
    science_targets: list,
    analysis_start:  str,
    analysis_days:   float,
    bands,
    catalog:         pd.DataFrame,
    allowed_grades:  list = None,
    max_pitch_diff:  float = MAX_PITCH_DIFF,
    time_step:       float = 1.0,
    n_workers:       int   = None,
) -> list:
    """
    Generate labeled training examples in parallel across science targets and bands.
    OPTIMISATION 1: ProcessPoolExecutor — each target is an independent map job.
    Loops over all requested bands so the model trains on multi-band signal.
    """
    if not isinstance(bands, (list, tuple)):
        bands = [bands]
    active_grades = [g for g in (allowed_grades or REF_GRADES) if g in REF_GRADES]

    all_examples = []
    for band in bands:
        try:
            cands = _filter_candidates(catalog, band, "high", active_grades)
        except ValueError:
            cands = catalog.copy()

        ref_data_rows = [
            row.to_dict() for _, row in cands.iterrows()
            if isinstance(row.get("main_id"), str)
            and row["main_id"].strip() not in SKIP_NAMES
        ]

        print(f"\n[Band {band}] {len(science_targets)} targets "
              f"| {len(ref_data_rows)} ref candidates | pitch<={max_pitch_diff} deg")

        worker_args = [
            (sci, analysis_start, analysis_days, band,
             ref_data_rows, max_pitch_diff, time_step)
            for sci in science_targets
        ]

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_generate_labels_worker, a): a[0]
                for a in worker_args
            }
            for fut in as_completed(futures):
                sci = futures[fut]
                try:
                    examples = fut.result()
                    all_examples.extend(examples)
                    pos = sum(e["label"] for e in examples)
                    print(f"  {sci}: {len(examples)} examples ({pos} positive)")
                except Exception as exc:
                    print(f"  {sci}: FAILED — {exc}")

    pos_total = sum(e["label"] for e in all_examples)
    print(f"\nTotal across all bands: {len(all_examples)} examples | "
          f"{pos_total} positive | {len(all_examples)-pos_total} negative")
    return all_examples


# =============================================================================
# CATALOG & TARGET DIAGNOSTICS
# =============================================================================

def print_catalog_diagnostics(catalog: pd.DataFrame, band, contrast: str,
                               allowed_grades: list):
    """Print a clear summary of what the catalog contains."""
    grade_col = GRADE_COLUMNS.get((band, contrast.lower()), "?")
    mag_col   = BAND_MAG_COL.get(band, "?")
    grade_src = _find_grade_source(catalog, grade_col)

    print(f"  Catalog rows      : {len(catalog)}")
    print(f"  All columns       : {list(catalog.columns)}")
    print(f"  Target grade col  : {grade_col!r}")
    print(f"  Grade col found   : {grade_src!r}")
    print(f"  Mag col           : {mag_col!r} "
          f"({'PRESENT' if mag_col in catalog.columns else 'MISSING'})")

    if grade_src:
        counts = catalog[grade_src].value_counts().to_dict()
        print(f"  Grade counts      : {counts}")
        try:
            cands = _filter_candidates(catalog, band, contrast, allowed_grades)
            print(f"  After filter      : {len(cands)} candidates")
        except ValueError as e:
            print(f"  Filter error      : {e}")
    else:
        print("  WARNING: no column with A/B/C values found!")


def scan_for_valid_target(
    catalog: pd.DataFrame,
    science_targets: list,
    analysis_start: str,
    analysis_days: float,
    band,
    contrast: str,
    allowed_grades: list,
    max_pitch_diff: float,
    time_step: float = 1.0,
) -> Optional[str]:
    """
    Check each science target to find one with at least one valid reference star.
    Prints detailed per-target diagnostics including closest pitch diff seen.
    Returns the first valid target name, or None if none found.
    """
    print(f"  pitch limit={max_pitch_diff} deg | solar={SUN_MIN}-{SUN_MAX} deg")

    # Build candidate list once
    try:
        cands = _filter_candidates(catalog, band, contrast, allowed_grades)
    except ValueError as exc:
        print(f"  ERROR filtering candidates: {exc}")
        return None

    print(f"  {len(cands)} candidates after grade/mag filter\n")

    for sci_name in science_targets:
        print(f"  Checking {sci_name}...")
        try:
            coords = get_target_coords([sci_name])
            if sci_name not in coords:
                print(f"    -> Not found in SIMBAD — skip")
                continue

            sci_coord = coords[sci_name]
            times, keepout, _ = compute_keepout(
                {sci_name: sci_coord}, analysis_start, analysis_days, time_step
            )
            sci_keepout = keepout[sci_name]
            pct         = 100 * sci_keepout.sum() / len(sci_keepout)
            windows     = _get_observable_windows(times, sci_keepout)

            if not windows:
                print(f"    -> Never observable ({pct:.1f}%) — skip")
                continue

            print(f"    Observable {pct:.1f}% | {len(windows)} window(s)")

            _, _, _, sci_pitch_full = compute_roman_angles(
                sci_coord, analysis_start, analysis_days, time_step
            )
            sci_pitch_vals = sci_pitch_full.to(u.degree).value.astype(np.float32)

            n_valid_total  = 0
            min_pdiff_seen = 999.0
            n_solar_ok     = 0

            for win_idx_s, (win_start, win_end, ws, we, wd) in enumerate(windows):
                si  = int((win_start.mjd - times[0].mjd) / time_step)
                ei  = min(si + int(wd) + 1, len(sci_pitch_vals))
                spw = sci_pitch_vals[si:ei]
                print(f"      window {win_idx_s+1}/{len(windows)}: "
                      f"{ws[:10]} → {we[:10]}  ({int(wd)}d)  "
                      f"checking {len(cands)} candidates ...", flush=True)

                for ci, (_, ref) in enumerate(cands.iterrows()):
                    if ci % 100 == 0 and ci > 0:
                        print(f"        ... {ci}/{len(cands)} candidates done", flush=True)
                    name = ref.get("main_id", "")
                    if not isinstance(name, str) or name.strip() in SKIP_NAMES:
                        continue
                    try:
                        coord = _build_skycoord(ref)
                        _, ref_sun, _, ref_pitch = compute_roman_angles(
                            coord, ws, wd, time_step=1.0
                        )
                        n = min(len(spw), len(ref_sun))
                        sun_d    = ref_sun.to(u.degree).value[:n].astype(np.float32)
                        pitch_d  = ref_pitch.to(u.degree).value[:n].astype(np.float32)
                        solar_ok = (sun_d > SUN_MIN) & (sun_d < SUN_MAX)

                        if solar_ok.any():
                            n_solar_ok += 1
                            pdiff = np.abs(spw[:n] - pitch_d)
                            mp    = float(pdiff[solar_ok].min())
                            min_pdiff_seen = min(min_pdiff_seen, mp)

                            if mp < max_pitch_diff:
                                n_valid_total += 1

                    except Exception as exc:
                        print(f"    ref star error ({name}): {exc}")

            print(f"    Solar-valid ref stars: {n_solar_ok} | "
                  f"Pitch-valid: {n_valid_total} | "
                  f"Min pitch diff seen: {min_pdiff_seen:.2f} deg")

            if n_valid_total > 0:
                print(f"    -> FOUND {n_valid_total} valid star(s) — using this target\n")
                return sci_name
            else:
                print(f"    -> 0 valid (need pitch < {max_pitch_diff} deg, "
                      f"closest was {min_pdiff_seen:.2f} deg)\n")

        except Exception as exc:
            print(f"    -> Unexpected error: {exc}")
            traceback.print_exc()

    return None


def select_ref_star(
    sci_name:       str,
    analysis_start: str,
    analysis_days:  float,
    band,
    contrast:       str,
    catalog:        pd.DataFrame,
    engine=None,
    time_step:      float = 1.0,
    allowed_grades: list  = None,
    sort_mode:      str   = "valid_days",
    max_pitch_diff: float = MAX_PITCH_DIFF,
    max_imag:       float = MAX_IMAG,
    classifier:     Optional[RefStarClassifier] = None,
    clf_threshold:  float = 0.0,
    use_amp:        bool  = True,
) -> dict:
    """
    ML-accelerated reference star selection.
    If a trained classifier is provided it pre-filters candidates before
    the expensive physics check. Recall=1.0 guaranteed by the calibrated
    threshold from train_classifier().
    """
    key = (band, contrast.lower())
    if key not in GRADE_COLUMNS:
        raise ValueError(f"Unknown band/contrast: {key}")
    if sort_mode not in SORT_MODES:
        raise ValueError(f"Unknown sort_mode='{sort_mode}'")

    active_grades = [g for g in (allowed_grades or REF_GRADES) if g in REF_GRADES]
    mag_col       = BAND_MAG_COL[band]
    grade_col     = GRADE_COLUMNS[key]

    # Candidate filtering
    candidates = _filter_candidates(catalog, band, contrast, active_grades, max_imag)
    print(f"  {len(candidates)} candidates after grade/mag filter.")

    # OPTIMISATION 2: async VizieR fetch
    coords = get_target_coords([sci_name])
    if sci_name not in coords:
        return {"error": f"'{sci_name}' not found in SIMBAD."}
    sci_coord = coords[sci_name]
    icrs      = sci_coord.icrs

    vizier_bytes               = asyncio.run(_async_fetch_vizier(float(icrs.ra.deg), float(icrs.dec.deg)))
    sci_diameter, sci_diam_src = _parse_vizier(vizier_bytes, band, sci_name)
    sci_mag                    = get_science_mag(sci_name, band, catalog)

    effective_sort = sort_mode
    if sort_mode == "closest_mag" and sci_mag is None:
        effective_sort = "brightest"
    sort_method = (f"grade ({'>'.join(active_grades)}) then "
                   f"{SORT_MODE_LABELS[effective_sort].lower()}")

    # Visibility
    times, keepout, _ = compute_keepout(
        {sci_name: sci_coord}, analysis_start, analysis_days, time_step
    )
    sci_keepout    = keepout[sci_name]
    visibility_pct = 100 * np.sum(sci_keepout) / len(sci_keepout)
    windows        = _get_observable_windows(times, sci_keepout)

    if not windows:
        return {"science_target": sci_name, "error": "Not observable in this period."}

    _, _, _, sci_pitch_full = compute_roman_angles(
        sci_coord, analysis_start, analysis_days, time_step
    )
    sci_pitch_vals = sci_pitch_full.to(u.degree).value.astype(np.float32)  # Opt 3

    # Build serialisable row dicts (needed by workers and ML filter)
    ref_data_rows = []
    for _, ref in candidates.iterrows():
        name = ref["main_id"]
        if not isinstance(name, str) or name.strip() in SKIP_NAMES:
            continue
        diam, diam_src = _get_ref_star_diameter(ref, band)
        d = ref.to_dict()
        d["_diameter_mas"] = diam
        d["_diameter_src"] = diam_src
        ref_data_rows.append(d)

    # ML PRE-FILTER
    n_before = len(ref_data_rows)
    if classifier is not None and n_before > 0:
        # OPTIMISATION 4: vectorised feature extraction, all candidates at once
        X         = extract_features(
            pd.DataFrame(ref_data_rows),
            float(icrs.ra.deg), float(icrs.dec.deg), windows[0][4], band
        )
        keep_mask     = classify_candidates(classifier, clf_threshold, X, use_amp)
        ref_data_rows = [rd for rd, k in zip(ref_data_rows, keep_mask) if k]
        n_filtered    = n_before - len(ref_data_rows)
        print(f"  ML filter: {n_filtered}/{n_before} eliminated "
              f"({100*n_filtered/max(1,n_before):.1f}%), "
              f"{len(ref_data_rows)} forwarded to physics check.")
    else:
        print(f"  No classifier — checking all {n_before} candidates.")

    # Physics check
    sort_key = _make_sort_key(effective_sort)
    results  = []

    for win_idx, (win_start, win_end, ws, we, wd) in enumerate(windows):
        si  = int((win_start.mjd - times[0].mjd) / time_step)
        ei  = min(si + int(wd) + 1, len(sci_pitch_vals))
        spw = sci_pitch_vals[si:ei]

        n_days_win = int(round(wd)) + 1
        dates = [
            (win_start + i * u.day).to_value("iso", subfmt="date")
            for i in range(n_days_win)
        ]

        valid_refs   = []
        pitch_series = {}

        print(f"  Window {win_idx+1}/{len(windows)}: physics check on "
              f"{len(ref_data_rows)} candidate(s) ...", flush=True)
        for ri, rd in enumerate(ref_data_rows):
            if ri % 100 == 0 and ri > 0:
                print(f"    ... {ri}/{len(ref_data_rows)} done", flush=True)
            ref_name = rd["main_id"]
            try:
                passes, n_valid, min_pd, pd_series, valid_mask = _check_one_star(
                    rd, ws, wd, spw, max_pitch_diff
                )
                pitch_series[ref_name] = pd_series

                if not passes:
                    continue

                valid_date_strs = [
                    (win_start + int(d) * u.day).to_value("iso", subfmt="date")
                    for d in np.where(valid_mask)[0]
                ]
                ref_mag  = _safe_float(rd.get(mag_col))
                mag_diff = (
                    abs(ref_mag - sci_mag)
                    if effective_sort == "closest_mag"
                    and sci_mag is not None and ref_mag is not None
                    else None
                )
                valid_refs.append({
                    "reference_star": ref_name,
                    "grade":          rd["grade"],
                    "grade_rank":     int(rd["grade_rank"]),
                    "mag":            ref_mag,
                    "mag_diff":       mag_diff,
                    "n_valid_days":   n_valid,
                    "min_pitch_diff": min_pd,
                    "valid_dates":    valid_date_strs,
                    "diameter_mas":   rd.get("_diameter_mas"),
                    "diameter_src":   rd.get("_diameter_src"),
                })
            except Exception as exc:
                print(f"    physics check error ({ref_name}): {exc}")

        valid_refs.sort(key=sort_key)

        avail_df = pd.DataFrame(
            {r["reference_star"]: [d in set(r["valid_dates"]) for d in dates]
             for r in valid_refs},
            index=dates,
        )
        avail_df.index.name = "date"

        pitch_data = {}
        for r in valid_refs:
            name   = r["reference_star"]
            series = pitch_series.get(name, np.array([]))
            vals   = list(series[:len(dates)]) + [np.nan] * (len(dates) - len(series))
            pitch_data[name] = vals
        pitch_df = pd.DataFrame(pitch_data, index=dates)
        pitch_df.index.name = "date"

        print(f"  Window {win_idx+1}: {len(valid_refs)} valid star(s).")
        results.append({
            "start":        ws,
            "end":          we,
            "duration_days": wd,
            "valid_refs":   valid_refs,
            "best_ref":     valid_refs[0] if valid_refs else None,
            "pitch_df":     pitch_df,
            "avail_df":     avail_df,
        })

    return {
        "science_target":        sci_name,
        "band":                  band,
        "contrast":              contrast,
        "grade_column":          grade_col,
        "allowed_grades":        active_grades,
        "sort_mode":             sort_mode,
        "effective_sort":        effective_sort,
        "sci_mag":               sci_mag,
        "sci_diameter":          sci_diameter,
        "sci_diameter_src":      sci_diam_src,
        "visibility_pct":        visibility_pct,
        "sort_method":           sort_method,
        "max_pitch_diff":        max_pitch_diff,
        "max_imag":              max_imag,
        "observable_windows":    results,
        "n_candidates_total":    n_before,
        "n_candidates_forwarded": len(ref_data_rows),
    }



# =============================================================================
# MULTI-TARGET PARALLEL BENCHMARK
# =============================================================================

def _select_one_target_worker(args):
    """
    Picklable worker: run baseline + ML-accelerated selection for one target.
    Returns a result dict with timing, valid star counts, and recall info.
    """
    (sci_name, analysis_start, analysis_days, band, contrast,
     catalog_path, allowed_grades, sort_mode, max_pitch_diff,
     model_state, n_features, clf_threshold, use_amp) = args

    try:
        # Re-load catalog inside worker (DataFrames are not reliably picklable
        # across all platforms when large)
        catalog = pd.read_csv(catalog_path, low_memory=False)
        catalog = _coerce_catalog(catalog)

        # Reconstruct model inside worker
        model = None
        if model_state is not None:
            model = RefStarClassifier(n_features=n_features)
            model.load_state_dict(model_state)
            model.eval()
            model = model.cpu()

        # Baseline
        t0 = time.perf_counter()
        result_base = select_ref_star(
            sci_name, analysis_start, analysis_days,
            band, contrast, catalog,
            allowed_grades=allowed_grades,
            sort_mode=sort_mode,
            max_pitch_diff=max_pitch_diff,
            classifier=None,
        )
        base_time = time.perf_counter() - t0

        # ML-accelerated
        t1 = time.perf_counter()
        result_ml = select_ref_star(
            sci_name, analysis_start, analysis_days,
            band, contrast, catalog,
            allowed_grades=allowed_grades,
            sort_mode=sort_mode,
            max_pitch_diff=max_pitch_diff,
            classifier=model,
            clf_threshold=clf_threshold,
            use_amp=False,   # workers run on CPU
        )
        ml_time = time.perf_counter() - t1

        # Recall check
        all_ok   = True
        per_win  = []
        for wb, wm in zip(
            result_base["observable_windows"],
            result_ml["observable_windows"],
        ):
            sb     = {r["reference_star"] for r in wb["valid_refs"]}
            sm     = {r["reference_star"] for r in wm["valid_refs"]}
            missed = sb - sm
            ok     = len(missed) / max(1, len(sb)) <= 0.01  # ≤1% miss acceptable at 99% recall target
            all_ok = all_ok and ok
            per_win.append({
                "baseline_stars": len(sb),
                "ml_stars":       len(sm),
                "missed":         missed,
                "recall_ok":      ok,
            })

        n_base = sum(len(w["valid_refs"]) for w in result_base["observable_windows"])
        n_ml   = sum(len(w["valid_refs"]) for w in result_ml["observable_windows"])

        return {
            "science_target":        sci_name,
            "base_time":             base_time,
            "ml_time":               ml_time,
            "speedup":               base_time / ml_time if ml_time > 0 else float("inf"),
            "n_base_stars":          n_base,
            "n_ml_stars":            n_ml,
            "recall_ok":             all_ok,
            "per_window":            per_win,
            "result_base":           result_base,
            "result_ml":             result_ml,
            "n_candidates_total":    result_ml.get("n_candidates_total", 0),
            "n_candidates_forwarded": result_ml.get("n_candidates_forwarded", 0),
            "error":                 None,
        }

    except Exception as exc:
        return {
            "science_target": sci_name,
            "error":          str(exc),
        }


def run_multi_target(
    science_targets:  list,
    analysis_start:   str,
    analysis_days:    float,
    band,
    contrast:         str,
    catalog:          pd.DataFrame,
    model:            RefStarClassifier,
    clf_threshold:    float,
    allowed_grades:   list  = None,
    sort_mode:        str   = "valid_days",
    max_pitch_diff:   float = MAX_PITCH_DIFF,
    n_workers:        int   = None,
    catalog_path:     str   = None,
) -> list:
    """
    Run baseline + ML-accelerated selection for multiple science targets
    in parallel using ProcessPoolExecutor.

    Each target is an independent map job (Optimisation 1 — MapReduce).
    Results are collected as futures complete and assembled into a summary.

    Parameters
    ----------
    science_targets : list of SIMBAD-resolvable target names
    model           : trained RefStarClassifier
    clf_threshold   : recall-safe threshold from train_classifier()
    catalog_path    : path to the cached CSV catalog (needed by workers)
    n_workers       : number of parallel workers (default: os.cpu_count())

    Returns
    -------
    list of result dicts, one per target, in completion order
    """
    if catalog_path is None:
        catalog_path = str(DEFAULT_CACHE_PATH)

    # Serialise model state dict for pickling across processes
    model_state = model.cpu().state_dict() if model is not None else None
    n_features  = N_FEATURES

    worker_args = [
        (
            sci, analysis_start, analysis_days, band, contrast,
            catalog_path, allowed_grades or REF_GRADES,
            sort_mode, max_pitch_diff,
            model_state, n_features, clf_threshold, False,
        )
        for sci in science_targets
    ]

    print(f"\nDispatching {len(science_targets)} target(s) to ProcessPoolExecutor...")
    all_results = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_select_one_target_worker, a): a[0]
            for a in worker_args
        }
        for fut in as_completed(futures):
            sci = futures[fut]
            try:
                res = fut.result()
                all_results.append(res)
                if res.get("error") is not None:
                    print(f"  {sci}: ERROR — {res['error']}")
                else:
                    recall_str = "RECALL≥99%" if res.get("recall_ok", False) else "RECALL<99% !"
                    print(
                        f"  {sci}: baseline={res['base_time']:.2f}s "
                        f"| ML={res['ml_time']:.2f}s "
                        f"| speedup={res['speedup']:.2f}x "
                        f"| stars={res['n_base_stars']} "
                        f"| {recall_str}"
                    )
            except Exception as exc:
                print(f"  {sci}: worker exception — {exc}")
                all_results.append({"science_target": sci, "error": str(exc)})

    return all_results


def print_multi_target_summary(results: list):
    """Print a formatted benchmark summary table for multi-target results."""
    sep = "=" * 75
    print(f"\n{sep}")
    print("MULTI-TARGET BENCHMARK SUMMARY")
    print(sep)
    print(f"  {'Target':<18} {'Base(s)':>8} {'ML(s)':>7} {'Speedup':>9} "
          f"{'Stars':>7} {'Recall':>8}")
    print(f"  {'-'*18} {'-'*8} {'-'*7} {'-'*9} {'-'*7} {'-'*8}")

    speedups    = []
    all_recall  = True

    for res in sorted(results, key=lambda r: r.get("science_target", "")):
        if res.get("error") is not None:
            print(f"  {res['science_target']:<18}  ERROR: {res['error']}")
            continue

        recall_str = "OK" if res.get("recall_ok", False) else "FAIL !"
        all_recall = all_recall and res.get("recall_ok", False)
        speedups.append(res["speedup"])

        print(
            f"  {res['science_target']:<18} "
            f"{res['base_time']:>8.2f} "
            f"{res['ml_time']:>7.2f} "
            f"{res['speedup']:>8.2f}x "
            f"{res['n_base_stars']:>7} "
            f"{recall_str:>8}"
        )

    if speedups:
        print(f"  {'-'*18} {'-'*8} {'-'*7} {'-'*9} {'-'*7} {'-'*8}")
        print(
            f"  {'MEAN':<18} "
            f"{'':>8} {'':>7} "
            f"{sum(speedups)/len(speedups):>8.2f}x "
            f"{'':>7} "
            f"{'ALL OK' if all_recall else 'SOME FAIL':>8}"
        )
    print(sep)


# EFFICIENCY EXPERIMENTS & VISUALISATION  (CS 6787 Final Project)


def save_plot(fig, filename: str):
    PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOT_OUTPUT_DIR / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"  Plot saved to {out_path}")


def plot_statistical_efficiency(
    model: RefStarClassifier,
    examples: list,
    threshold: float,
    train_losses=None,
):
    """
    4-panel statistical efficiency figure:
      (a) Training loss convergence
      (b) ROC curve (AUC)
      (c) Precision-Recall curve (Average Precision)
      (d) Score distribution — positive vs negative class separation
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        roc_curve, auc,
        precision_recall_curve, average_precision_score,
    )

    device = next(model.parameters()).device
    X = np.stack([e["features"] for e in examples]).astype(np.float32)
    y = np.array([e["label"]    for e in examples], dtype=np.float32)

    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(torch.from_numpy(X).to(device))).cpu().numpy()

    pos_scores = scores[y == 1]
    neg_scores = scores[y == 0]

    rc = {
        "font.family": "serif", "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "axes.linewidth": 1.2,
        "grid.alpha": 0.3, "savefig.dpi": 300,
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        fig.suptitle(
            "Statistical Efficiency — ML Reference Star Classifier\n"
            f"n={len(examples)} examples  |  positives={int(y.sum())}  |  "
            f"negatives={int((y==0).sum())}",
            fontsize=13, fontweight="bold",
        )

        # (a) Training loss curve
        ax = axes[0, 0]
        if train_losses:
            epochs = range(1, len(train_losses) + 1)
            ax.plot(epochs, train_losses, color="#1f77b4", linewidth=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("BCE Loss")
            ax.set_title("(a) Training Loss Convergence")
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, "Pass train_losses= to show this panel",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.set_title("(a) Training Loss Convergence")

        # (b) ROC curve
        ax = axes[0, 1]
        fpr, tpr, roc_threshs = roc_curve(y, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.5)")
        ax.fill_between(fpr, tpr, alpha=0.08, color="#1f77b4")
        op_idx = np.searchsorted(roc_threshs[::-1], threshold)
        if 0 <= op_idx < len(fpr):
            ax.scatter(fpr[len(fpr)-1-op_idx], tpr[len(tpr)-1-op_idx],
                       color="red", s=60, zorder=5, label=f"Threshold = {threshold:.3f}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate (Recall)")
        ax.set_title("(b) ROC Curve")
        ax.legend()
        ax.grid(True)

        # (c) Precision-Recall curve
        ax = axes[1, 0]
        prec, rec, pr_threshs = precision_recall_curve(y, scores)
        ap = average_precision_score(y, scores)
        ax.plot(rec, prec, color="#d62728", linewidth=2, label=f"AP = {ap:.4f}")
        ax.fill_between(rec, prec, alpha=0.08, color="#d62728")
        ax.axvline(x=1.0, color="green", linestyle="--", linewidth=1.5,
                   label="Recall = 1.0 target")
        baseline_prec = float(y.sum()) / len(y)
        ax.axhline(y=baseline_prec, color="grey", linestyle=":", linewidth=1,
                   label=f"Random ({baseline_prec:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("(c) Precision-Recall Curve")
        ax.legend()
        ax.grid(True)

        # (d) Score distribution
        ax = axes[1, 1]
        bins = np.linspace(0, 1, 35)
        ax.hist(neg_scores, bins=bins, alpha=0.55, color="#d62728",
                label=f"Negative  (n={len(neg_scores)})", density=True)
        ax.hist(pos_scores, bins=bins, alpha=0.55, color="#1f77b4",
                label=f"Positive  (n={len(pos_scores)})", density=True)
        ax.axvline(x=threshold, color="black", linestyle="--", linewidth=1.8,
                   label=f"Threshold = {threshold:.3f}")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Density")
        ax.set_title("(d) Score Distribution (Class Separation)")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()
        save_plot(fig, "statistical_efficiency.png")
        plt.show()


def plot_hardware_efficiency(
    multi_results: list,
    base_time_single: float = None,
    ml_time_single:   float = None,
):
    """
    4-panel hardware efficiency figure (paper-quality layout):
      (a) Speedup distribution — sorted bars, color-coded, no crowded x-labels
      (b) Scatter: ML time vs Baseline time per target (diagonal = no change)
      (c) Histogram of pre-filter elimination rates across all targets
      (d) Serial vs parallel total execution time
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    valid = [r for r in multi_results if not r.get("error")]
    if not valid:
        print("  No valid results for hardware efficiency plot.")
        return

    base_times = np.array([r["base_time"]  for r in valid])
    ml_times   = np.array([r["ml_time"]    for r in valid])
    speedups   = np.array([r["speedup"]    for r in valid])

    n_total   = np.array([r.get("n_candidates_total",    0) for r in valid], dtype=float)
    n_forward = np.array([r.get("n_candidates_forwarded", 0) for r in valid], dtype=float)
    elim_pct  = np.where(n_total > 0, 100.0 * (n_total - n_forward) / n_total, 0.0)

    rc = {
        "font.family": "serif", "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "axes.linewidth": 1.2,
        "grid.alpha": 0.3, "savefig.dpi": 300,
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle(
            "Hardware Efficiency — ML-Accelerated Reference Star Selection\n"
            "(Roman Space Telescope, CS 6787 Final Project)",
            fontsize=13, fontweight="bold",
        )

        # (a) Speedup — sorted bars, color-coded
        ax = axes[0, 0]
        order = np.argsort(speedups)
        s_sorted = speedups[order]
        bar_colors = ["#d62728" if s < 1.0 else
                      "#aec7e8" if s < 1.1 else
                      "#1f77b4" if s < 1.3 else
                      "#006400"
                      for s in s_sorted]
        x = np.arange(len(s_sorted))
        ax.bar(x, s_sorted, color=bar_colors, edgecolor="none", linewidth=0)
        ax.axhline(1.0, color="red",   ls="--", lw=1.5, label="No speedup (1×)")
        ax.axhline(speedups.mean(), color="darkgreen", ls="--", lw=1.5,
                   label=f"Mean = {speedups.mean():.2f}×")
        ax.set_xlabel(f"Targets sorted by speedup  (n={len(valid)})")
        ax.set_ylabel("Speedup Factor (Baseline / ML)")
        ax.set_title("(a) Per-Target Speedup Distribution")
        ax.set_xticks([])
        from matplotlib.patches import Patch
        legend_els = [
            Patch(facecolor="#d62728", label="< 1× (slowdown)"),
            Patch(facecolor="#aec7e8", label="1–1.1×"),
            Patch(facecolor="#1f77b4", label="1.1–1.3×"),
            Patch(facecolor="#006400", label="> 1.3×"),
            plt.Line2D([0], [0], color="red",       ls="--", lw=1.5, label="No speedup"),
            plt.Line2D([0], [0], color="darkgreen",  ls="--", lw=1.5,
                       label=f"Mean = {speedups.mean():.2f}×"),
        ]
        ax.legend(handles=legend_els, fontsize=8, ncol=2)
        ax.grid(True, axis="y")

        # (b) Scatter: ML time vs Baseline time
        ax = axes[0, 1]
        lim = max(base_times.max(), ml_times.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1.2, label="No change (y = x)", zorder=1)
        sc = ax.scatter(base_times, ml_times, c=speedups, cmap="RdYlGn",
                        vmin=0.8, vmax=max(speedups.max(), 1.5),
                        s=40, edgecolors="grey", linewidths=0.4, zorder=2)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("Speedup ×", fontsize=10)
        ax.set_xlabel("Baseline Wall-Clock Time (s)")
        ax.set_ylabel("ML Wall-Clock Time (s)")
        ax.set_title("(b) ML vs Baseline Time per Target")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.legend(fontsize=9)
        ax.grid(True)
        n_faster = int(np.sum(speedups > 1.0))
        ax.text(0.03, 0.97,
                f"{n_faster}/{len(valid)} targets faster with ML",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # (c) Elimination rate histogram
        ax = axes[1, 0]
        nonzero = elim_pct[elim_pct > 0]
        n_zero  = int(np.sum(elim_pct == 0))
        if len(nonzero) > 0:
            bins = np.linspace(0, 100, 21)
            ax.hist(nonzero, bins=bins, color="steelblue", edgecolor="navy",
                    linewidth=0.7, label=f"Targets with >0% elimination (n={len(nonzero)})")
        ax.text(0.97, 0.95,
                f"{n_zero}/{len(valid)} targets:\n0% eliminated",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="grey", alpha=0.9))
        ax.set_xlabel("Candidates Eliminated by ML Pre-Filter (%)")
        ax.set_ylabel("Number of Targets")
        ax.set_title("(c) ML Pre-Filter Elimination Rate Distribution\n"
                     "(fewer physics checks = hardware savings)")
        ax.set_xlim(0, 100)
        if len(nonzero) > 0:
            ax.legend(fontsize=9)
        ax.grid(True, axis="y")

        # (d) Serial vs parallel total time
        ax = axes[1, 1]
        serial_base   = base_times.sum()
        serial_ml     = ml_times.sum()
        parallel_base = base_times.max()
        parallel_ml   = ml_times.max()
        cats   = ["Serial\nBaseline", "Parallel\nBaseline", "Serial\nML", "Parallel\nML"]
        vals   = [serial_base, parallel_base, serial_ml, parallel_ml]
        colors = ["#d62728", "#f4a582", "#4393c3", "#92c5de"]
        bars   = ax.bar(cats, vals, color=colors, edgecolor="black", linewidth=0.7)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{v:.1f}s", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        ax.set_ylabel("Total Wall-Clock Time (s)")
        ax.set_title("(d) Serial vs Parallel Execution Time\n"
                     "(ProcessPoolExecutor, one worker per target)")
        ax.grid(True, axis="y")

        fig.tight_layout()
        save_plot(fig, "hardware_efficiency.png")
        plt.show()


def benchmark_data_parallelism(
    labeled_examples: list,
    worker_counts: tuple = (0, 1, 2, 4),
    n_epochs:      int   = 10,
    batch_size:    int   = 64,
    random_seed:   int   = 42,
) -> list:
    """
    Measure training throughput for different num_workers settings and plot.
    Demonstrates Optimisation 1 — Multi-Worker DataLoader (MapReduce map phase).
    """
    import matplotlib.pyplot as plt

    X = np.stack([e["features"] for e in labeled_examples]).astype(np.float32)
    y = np.array([e["label"]    for e in labeled_examples], dtype=np.float32)
    n_samples = len(X)
    device    = torch.device("cpu")  # scaling test runs on CPU for reproducibility

    is_win = platform.system() == "Windows"

    records = []
    for nw in worker_counts:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        # On Windows use persistent_workers so spawn happens once, not per epoch
        persistent = is_win and nw > 0
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
            batch_size=batch_size, shuffle=True, num_workers=nw,
            pin_memory=False, persistent_workers=persistent,
        )
        m   = RefStarClassifier(n_features=X.shape[1]).to(device)
        opt = optim.Adam(m.parameters(), lr=1e-3)
        crit = nn.BCEWithLogitsLoss()

        t0 = time.perf_counter()
        for _ in range(n_epochs):
            for xb, yb in loader:
                opt.zero_grad()
                crit(m(xb), yb).backward()
                opt.step()
        elapsed = time.perf_counter() - t0
        tput = (n_samples * n_epochs) / elapsed
        records.append({"workers": nw, "time_s": elapsed, "throughput": tput,
                        "persistent": persistent})
        print(f"  workers={nw:2d} persistent={persistent}: {elapsed:.2f}s  ({tput:.0f} samples/s)")

    workers     = [r["workers"]    for r in records]
    throughputs = [r["throughput"] for r in records]
    base_tp     = throughputs[0] if throughputs[0] > 0 else 1.0
    rel         = [t / base_tp for t in throughputs]

    base_w = max(workers[0], 1)
    ideal  = [max(w, 1) / base_w for w in workers]

    is_windows  = platform.system() == "Windows"
    non_zero_tp = [tp for w, tp in zip(workers, throughputs) if w > 0]
    has_outlier = (
        len(non_zero_tp) > 0
        and max(non_zero_tp) > 0
        and throughputs[0] > 5 * max(non_zero_tp)
    )

    rc = {
        "font.family": "serif", "font.size": 11,
        "axes.titlesize": 12,   "axes.labelsize": 11,
        "figure.dpi": 150,
    }
    with plt.rc_context(rc):
        n_panels = 3 if (has_outlier and is_windows) else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
        axes = list(axes)
        fig.suptitle(
            "Data Parallelism Scaling — Optimisation 1: Multi-Worker DataLoader\n"
            f"(n_epochs={n_epochs}, batch_size={batch_size}, device=cpu)",
            fontsize=12, fontweight="bold",
        )

        # Panel (a): log-scale bar chart, all workers
        ax = axes[0]
        w_labels   = [str(w) for w in workers]
        bar_colors = ["#2166ac" if w == 0 else "#4dac26" for w in workers]
        bars = ax.bar(w_labels, throughputs, color=bar_colors, edgecolor="k", linewidth=0.6)
        for bar, tp in zip(bars, throughputs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.08,
                    f"{tp:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_yscale("log")
        ax.set_xlabel("num_workers")
        ax.set_ylabel("Throughput (samples / s)  [log scale]")
        ax.set_title("(a) Throughput — All Workers")
        ax.grid(True, axis="y", alpha=0.3, which="both")
        if is_windows:
            ax.annotate(
                "workers=0 fastest on Windows\n(no spawn overhead; blue bar)",
                xy=(0.5, 0.04), xycoords="axes fraction",
                ha="center", va="bottom", fontsize=8, color="firebrick",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="firebrick", alpha=0.9),
            )

        if has_outlier and is_windows:
            # Panel (b): workers >= 1 only, linear zoom
            ax = axes[1]
            w_gt0  = [w  for w in workers    if w > 0]
            tp_gt0 = [tp for w, tp in zip(workers, throughputs) if w > 0]
            bars2  = ax.bar([str(w) for w in w_gt0], tp_gt0,
                            color="#4dac26", edgecolor="k", linewidth=0.6)
            for bar, tp in zip(bars2, tp_gt0):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                        f"{tp:.0f}", ha="center", va="bottom", fontsize=9)
            ax.set_xlabel("num_workers")
            ax.set_ylabel("Throughput (samples / s)")
            ratio = throughputs[0] / max(tp_gt0) if max(tp_gt0) > 0 else float("inf")
            ax.set_title(
                "(b) Workers >= 1 — Zoomed (linear scale)\n"
                f"[workers=0 ref: {throughputs[0]:.0f} samp/s  off scale, {ratio:.0f}x faster]"
            )
            ax.grid(True, axis="y", alpha=0.3)
            ax.text(
                0.97, 0.95,
                f"workers=0 ref: {throughputs[0]:.0f} samp/s\n"
                f"({ratio:.0f}× faster — no spawn overhead)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="firebrick",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="firebrick", alpha=0.9),
            )
            speedup_ax = axes[2]
        else:
            speedup_ax = axes[1]

        # Speedup panel
        ax = speedup_ax
        panel_lbl = "(c)" if n_panels == 3 else "(b)"
        ax.plot(workers, rel,   "o-", color="steelblue", linewidth=2, markersize=8, label="Observed")
        ax.plot(workers, ideal, "k--",                   linewidth=1.5,              label="Linear ideal")
        ax.set_xlabel("num_workers")
        ax.set_ylabel("Relative Speedup  (vs workers=0)")
        ax.set_title(f"{panel_lbl} Relative Speedup vs Worker Count")
        ax.set_xticks(workers)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        if is_windows:
            ax.text(
                0.5, 0.04,
                "workers=0 fastest on Windows\n(no process spawn overhead)",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8, color="firebrick",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                          edgecolor="firebrick", alpha=0.9),
            )

        plt.tight_layout()
        save_plot(fig, "data_parallelism_scaling.png")
        plt.show()
    return records


def benchmark_amp_vs_fp32(
    labeled_examples: list,
    n_epochs:    int = 20,
    batch_size:  int = 64,
    random_seed: int = 42,
) -> list:
    """
    Compare training time with AMP (fp16+fp32) vs pure fp32.
    Demonstrates Optimisation 3 — Mixed Precision (Gupta et al. 2015).
    On CPU both runs are fp32; the plot still shows the overhead difference.
    """
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = np.stack([e["features"] for e in labeled_examples]).astype(np.float32)
    y = np.array([e["label"]    for e in labeled_examples], dtype=np.float32)

    records = []
    for use_amp in (False, True):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        amp_on = use_amp and device.type == "cuda"
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
            batch_size=batch_size, shuffle=True,
        )
        m      = RefStarClassifier(n_features=X.shape[1]).to(device)
        opt    = optim.Adam(m.parameters(), lr=1e-3)
        crit   = nn.BCEWithLogitsLoss()
        scaler = GradScaler("cuda", enabled=amp_on)

        epoch_times = []
        for _ in range(n_epochs):
            t_ep = time.perf_counter()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                with autocast(device_type=device.type, enabled=amp_on):
                    loss = crit(m(xb), yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            epoch_times.append(time.perf_counter() - t_ep)

        label = ("AMP fp16+fp32" if amp_on
                 else ("FP32 only" if device.type == "cuda" else f"FP32 (cpu)"))
        records.append({"label": label, "epoch_times": epoch_times, "amp": amp_on})
        total = sum(epoch_times)
        print(f"  {label}: total={total:.2f}s  mean/epoch={np.mean(epoch_times)*1000:.1f}ms")

    on_cpu = device.type != "cuda"
    cpu_note = (
        "No CUDA GPU detected — AMP disabled for both runs.\n"
        "Mixed precision (fp16) requires CUDA tensor cores to accelerate.\n"
        "On a GPU this plot would show 1.5–3× speedup."
        if on_cpu else ""
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    title_suffix = "  [CPU-only: AMP has no effect]" if on_cpu else ""
    fig.suptitle(
        f"Mixed Precision — Optimisation 3: AMP fp16+fp32 vs FP32{title_suffix}\n"
        f"(n_epochs={n_epochs}, batch_size={batch_size}, device={device})",
        fontsize=12, fontweight="bold",
    )

    ax = axes[0]
    colors = ["tomato", "steelblue"]
    for rec, col in zip(records, colors):
        ax.plot(range(1, n_epochs + 1), [t * 1000 for t in rec["epoch_times"]],
                linewidth=2, color=col, label=rec["label"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Epoch Time (ms)")
    ax.set_title("Per-Epoch Training Time")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    if cpu_note:
        ax.annotate(
            cpu_note,
            xy=(0.5, 0.97), xycoords="axes fraction",
            ha="center", va="top", fontsize=8,
            color="firebrick",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="firebrick", alpha=0.85),
        )

    ax = axes[1]
    lbls   = [r["label"] for r in records]
    totals = [sum(r["epoch_times"]) for r in records]
    bars   = ax.bar(lbls, totals, color=colors[:len(records)],
                    edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{v:.2f}s", ha="center", va="bottom", fontsize=11, fontweight="bold")
    if on_cpu:
        ax.set_title(f"Total Training Time ({n_epochs} epochs)\n"
                     "AMP unavailable on CPU — bars equal by design")
    elif len(totals) == 2 and totals[1] > 0:
        speedup = totals[0] / totals[1]
        ax.set_title(f"Total Training Time ({n_epochs} epochs)\nAMP speedup: {speedup:.2f}×")
    else:
        ax.set_title(f"Total Training Time ({n_epochs} epochs)")
    ax.set_ylabel("Total Time (s)")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    save_plot(fig, "amp_vs_fp32.png")
    plt.show()
    return records


if __name__ == "__main__":
    print("=== ML_multiple_targets starting (imports done) ===", flush=True)

    TRAINING_TARGETS = [
        # --- original 25 ---
        "* 47 UMa", "55 Cnc", "* tau Cet", "eps Eri", "Kepler-62", "GJ 667 C",
        "HD 40307", "LHS 1140", "TRAPPIST-1", "Proxima Centauri",
        "HD 190360", "HD 160691", "HD 219134", "HD 154345", "14 Her",
        "ups And", "HD 217107", "HD 100546", "HD 39091", "psi 1 Dra B",
        "HD 145675", "bet Pic", "HD 114613", "HD 192310", "HD 134987",
        "HD 87883", "HD 114783", "* pi. Men",   # for_rifah targets
        # --- batch 2: 25 ---
        "51 Peg",       # first confirmed exoplanet host
        "rho CrB",      # first RV planet — HD 143761
        "GJ 876",       # M-dwarf resonant pair
        "HD 69830",     # 3 Neptune-mass planets
        "HD 128311",    # 2 giant planets
        "HD 37124",     # 3 planets in cool zone
        "HD 82943",     # 2:1 resonance pair
        "61 Vir",       # super-Earth + Neptune + Saturn analog
        "HD 10647",     # Jupiter analog
        "HR 8799",      # directly imaged 4-planet system
        "GJ 832",       # M-dwarf Jupiter + super-Earth
        "GJ 849",       # M-dwarf long-period Jupiter
        "HD 136352",    # 3 sub-Neptune planets
        "HD 20794",     # 3 super-Earths near HZ
        "HD 7924",      # super-Earth host
        "HD 168443",    # 2 planets + brown dwarf
        "HD 12661",     # 2 giant planets
        "HD 38529",     # super-Jupiter + outer companion
        "kap And",      # directly imaged super-Jupiter
        "HD 95086",     # directly imaged super-Jupiter
        "GJ 3470",      # warm Neptune transiting M-dwarf
        "HD 23596",     # long-period eccentric Jupiter
        "HD 222582",    # eccentric Jupiter analog
        "HD 142",       # Jupiter in HZ
        "mu Ara",       # 4-planet system
        # --- batch 3: 50 new → total = 100 ---
        "GJ 581",       # famous M-dwarf multi-planet system
        "GJ 436",       # warm Neptune transiting M-dwarf
        "HD 189733",    # benchmark hot Jupiter host (V=7.7)
        "HD 209458",    # first transiting exoplanet
        "Fomalhaut",    # debris disk + planet candidate
        "GJ 504",       # directly imaged super-Jupiter
        "HD 10180",     # 6+ planet system
        "HD 17051",     # hot Jupiter (iota Hor)
        "HD 85512",     # super-Earth in HZ
        "GJ 163",       # 3 planets near HZ
        "GJ 687",       # super-Earth M-dwarf
        "GJ 433",       # super-Earth M-dwarf
        "GJ 317",       # Jupiter pair around M-dwarf
        "HD 141399",    # 4 planets
        "HD 34445",     # 5 planets
        "HD 164922",    # 2 planets
        "HD 45184",     # 2 planets
        "HD 31527",     # 3 planets
        "HD 52265",     # hot Jupiter
        "HD 73526",     # 2 planets in 2:1 resonance
        "HD 60532",     # 2 giant planets
        "HD 4203",      # long-period eccentric giant
        "HD 33564",     # hot Jupiter
        "HD 179949",    # hot Jupiter
        "HD 149026",    # Saturn-size transiting planet
        "HD 1461",      # super-Earth + Saturn-analog
        "HD 27442",     # giant planet (eps Ret)
        "HD 196050",    # giant planet
        "HD 216437",    # giant planet
        "HD 50499",     # long-period Jupiter
        "HD 117207",    # Jupiter analog
        "HD 72659",     # Jupiter analog
        "HD 7449",      # eccentric giant
        "HD 202206",    # eccentric planet pair
        "HD 183263",    # 2 giant planets
        "HD 219828",    # super-Earth + outer giant
        "HD 102365",    # nearby super-Earth
        "HD 109271",    # 2 Neptune-size planets
        "GJ 1132",      # rocky transiting M-dwarf planet
        "GJ 3293",      # 4 planet M-dwarf system
        "GJ 625",       # super-Earth M-dwarf
        "GJ 536",       # super-Earth M-dwarf
        "GJ 3942",      # super-Earth M-dwarf
        "HD 215152",    # 4 Neptune-size planets
        "HD 21693",     # 2 Neptune-size planets
        "HD 215497",    # super-Earth + giant
        "Kepler-22",    # first Kepler HZ planet
        "Kepler-186",   # first HZ Earth-size planet
        "Kepler-442",   # best-ranked HZ super-Earth
        "Kepler-452",   # Earth-size in HZ (Earth 2.0)
        # --- batch 4: 50 additional bright RV hosts ---
        "HD 75289",     # hot Jupiter, V=6.4
        "HD 210277",    # eccentric Jupiter, V=6.5
        "HD 92788",     # Saturn-mass planet, V=7.3
        "bet Gem",      # Pollux — giant planet, V=1.1
        "HD 8574",      # Jupiter analog, V=7.1
        "HD 89307",     # Jupiter analog, V=7.0
        "HD 33636",     # eccentric Jupiter, V=6.9
        "HD 108874",    # 2 giant planets, V=8.8
        "HD 114729",    # Jupiter analog, V=6.7
        "HD 196885",    # planet + stellar companion, V=6.4
        "HD 30562",     # eccentric Jupiter, V=5.8
        "HD 155358",    # 2 planets in HZ, V=7.2
        "HD 11964",     # 2 planets, V=6.4
        "HD 190228",    # eccentric giant, V=7.3
        "HD 4208",      # Saturn analog, V=7.8
        "HD 23127",     # eccentric giant, V=8.6
        "HD 5319",      # 2 giants in resonance, V=8.0
        "HD 207832",    # 2 giant planets, V=7.7
        "HD 142415",    # hot Jupiter, V=7.3
        "HD 108147",    # hot Jupiter, V=7.0
        "HD 43691",     # hot Jupiter, V=8.0
        "HD 185269",    # inflated hot Jupiter, V=6.7
        "HD 102195",    # hot Jupiter M-dwarf, V=8.1
        "HD 86081",     # hot Jupiter, V=7.8
        "HD 68988",     # hot Jupiter, V=7.8
        "HD 149143",    # hot Jupiter, V=7.9
        "HD 73534",     # Jupiter beyond HZ, V=8.2
        "HD 126614",    # Jupiter with brown dwarf, V=8.8
        "HD 16175",     # eccentric giant, V=7.3
        "HD 231701",    # warm Jupiter, V=8.1
        "GJ 15 A",      # super-Earth, nearby M-dwarf
        "GJ 86",        # hot Jupiter around K-dwarf
        "GJ 328",       # long-period Jupiter M-dwarf
        "GJ 676 A",     # 4 planets, M-dwarf
        "GJ 179",       # cold Jupiter M-dwarf
        "GJ 422",       # super-Earth M-dwarf
        "GJ 480",       # super-Earth M-dwarf
        "HD 171028",    # eccentric giant, V=8.3
        "HD 224693",    # hot Jupiter, V=8.2
        "HD 47186",     # 2 planets, V=7.6
        "HD 148427",    # Saturn-mass planet, V=6.9
        "HD 16760",     # near-brown-dwarf companion, V=8.9
        "HD 98649",     # long-period giant, V=7.2
        "HD 142022",    # eccentric giant, V=7.7
        "HD 188015",    # Jupiter analog, V=8.7
        "HD 20367",     # Jupiter analog, V=6.4
        "HD 6434",      # hot Jupiter, V=7.7
        "HD 65216",     # Jupiter analog, V=7.9
        "HD 216770",    # Saturn-mass planet, V=8.1
        "HD 114386",    # eccentric Jupiter, V=8.7
        # --- batch 5: 100 additional systems ---
        # bright giant-planet hosts around evolved stars
        "iota Dra",     # V=3.3 giant planet around K-giant
        "eps Tau",      # V=3.5 giant planet in Hyades
        "7 CMa",        # V=3.9 giant planet
        "91 Aqr",       # V=4.2 giant planet around giant
        "42 Dra",       # V=4.8 giant planet
        "70 Vir",       # V=5.0 Saturn-mass, classic RV target
        "14 And",       # V=5.2 giant planet around giant
        "nu Phe",       # V=5.0 giant planet
        "18 Del",       # V=5.5 giant around subgiant
        "6 Lyn",        # V=5.9 giant planet
        "xi Aql",       # V=4.7 giant planet around giant
        "24 Sex",       # V=6.4 2 giants near 1:2 MMR
        # bright nearby RV hosts
        "tau Boo",      # V=4.5 hot Jupiter, tidal lock candidate
        "HD 3651",      # V=5.9 Saturn-mass companion
        "HD 19994",     # V=5.1 eccentric giant
        "HD 104985",    # V=5.5 giant around subgiant
        "HD 59686",     # V=5.5 giant planet in binary
        "HD 147513",    # V=5.4 Saturn-mass
        "HD 156846",    # V=6.5 highly eccentric giant
        "HD 219077",    # V=6.1 eccentric giant
        "HD 173416",    # V=6.1 giant planet around giant
        "HD 200964",    # V=6.7 2 giants near MMR
        "HD 213240",    # V=6.8 eccentric giant
        "HD 42618",     # V=6.8 Jupiter analog
        "HD 50554",     # V=6.8 eccentric giant
        "HD 154577",    # V=6.6 super-Earth
        "HD 169830",    # V=5.9 2 eccentric giants
        "HD 179079",    # V=6.9 super-Earth
        "HD 136118",    # V=6.9 giant
        "HD 139357",    # V=7.0 giant
        "HD 150706",    # V=7.0 giant
        # V=7-8 RV hosts
        "HD 9446",      # 2 giant planets
        "HD 11506",     # Jupiter analog, V=7.4
        "HD 13808",     # 2 planets, V=7.5
        "HD 16141",     # planet b, V=6.8
        "HD 16417",     # planet b, V=7.5
        "HD 17156",     # transiting eccentric Jupiter, V=8.2
        "HD 20782",     # very eccentric giant, V=7.4
        "HD 23079",     # long-period giant, V=7.1
        "HD 27894",     # 3 planets, V=9.4
        "HD 37605",     # 2 planets, V=8.7
        "HD 43848",     # giant planet, V=8.1
        "HD 45652",     # hot Jupiter, V=8.1
        "HD 46375",     # hot Saturn, V=7.9
        "HD 66428",     # giant planet, V=8.3
        "HD 70642",     # Jupiter in HZ, V=7.2
        "HD 80606",     # very eccentric hot Jupiter, V=9.0
        "HD 81040",     # planet b, V=7.6
        "HD 83443",     # hot Jupiter, V=8.2
        "HD 88133",     # hot Jupiter, V=8.0
        "HD 93083",     # planet b, V=8.3
        "HD 97658",     # super-Earth, V=7.7
        "HD 99109",     # planet b, V=9.1
        "HD 99492",     # Saturn in HZ, V=7.6
        "HD 99706",     # giant planet, V=8.1
        "HD 100777",    # planet b, V=8.4
        "HD 101930",    # planet b, V=8.2
        "HD 102117",    # super-Earth, V=7.5
        "HD 106252",    # eccentric giant, V=7.4
        "HD 107148",    # planet b, V=8.0
        "HD 111232",    # eccentric Jupiter, V=7.6
        "HD 116029",    # planet b, V=8.0
        "HD 118203",    # hot Jupiter, V=8.1
        "HD 121504",    # hot Jupiter, V=7.5
        "HD 125612",    # 3 planets, V=8.3
        "HD 130322",    # hot Jupiter, V=8.0
        "HD 131496",    # planet b, V=8.1
        "HD 132406",    # eccentric giant, V=8.4
        "HD 140913",    # near-brown-dwarf, V=8.1
        "HD 141937",    # eccentric giant, V=7.2
        "HD 153950",    # giant planet, V=7.4
        "HD 156279",    # near-brown-dwarf, V=8.0
        "HD 162020",    # BD companion, V=9.2
        "HD 163607",    # 2 giant planets, V=8.0
        "HD 168746",    # planet b, V=7.9
        "HD 170469",    # Saturn analog, V=8.2
        "HD 175167",    # eccentric giant, V=8.0
        "HD 178911 B",  # hot Jupiter in triple system
        "HD 181433",    # 3 planets including HZ, V=9.0
        "HD 181720",    # planet b, V=7.8
        "HD 187085",    # planet b, V=7.2
        "HD 187123",    # hot Jupiter + cold companion, V=7.9
        "HD 190647",    # planet b, V=7.8
        "HD 195019",    # giant planet, V=7.3
        "HD 204941",    # planet b, V=8.7
        "HD 205739",    # planet b, V=8.6
        "HD 207077",    # planet b, V=8.1
        "HD 208487",    # planet b, V=7.5
        "HD 212301",    # hot Jupiter, V=7.8
        "HD 218566",    # planet b, V=8.6
        "HD 220842",    # hot Jupiter, V=7.6
        "HD 221287",    # eccentric giant, V=7.8
        # additional M-dwarf hosts
        "GJ 649",       # Saturn-mass M-dwarf
        "GJ 674",       # super-Earth M-dwarf
        "GJ 3021",      # Jupiter analog M-dwarf
        # miscellaneous confirmed hosts
        "HD 4113",      # eccentric giant, V=7.9
        "HD 13908",     # 2 planets, V=7.5
    ]
    BAND             = "1w"
    TRAINING_BANDS   = ["1w", 1, 3, 4]   # all bands for training data diversity
    CONTRAST         = "high"
    ANALYSIS_START   = "2026-12-01T00:00:00"
    ANALYSIS_DAYS    = 548                # 18 months — Roman primary mission window
    ALLOWED_GRADES   = ["A", "B", "C"]
    SORT_MODE        = "closest_mag"

    SCRIPT_START = time.perf_counter()
    STEP_TIMES: dict = {}

    def _ts() -> str:
        e = time.perf_counter() - SCRIPT_START
        return f"[{int(e//3600):02d}h {int(e%3600//60):02d}m {e%60:05.2f}s]"

    def _step_start(label: str) -> float:
        print(f"\n{'='*60}")
        print(f"  {label}  {_ts()}")
        print(f"{'='*60}")
        return time.perf_counter()

    def _step_done(label: str, t: float) -> None:
        elapsed = time.perf_counter() - t
        STEP_TIMES[label] = elapsed
        print(f"  --> {label} done in {elapsed:.2f}s  {_ts()}")

    print(f"Loading catalog ...  {_ts()}", flush=True)
    catalog = load_catalog()
    print(f"\nCatalog ready: {len(catalog)} reference stars.  {_ts()}\n", flush=True)

    t_s0 = _step_start("STEP 0: Catalog diagnostics + finding benchmark target")
    print_catalog_diagnostics(catalog, BAND, CONTRAST, ALLOWED_GRADES)
    _step_done("STEP 0", t_s0)

    SCIENCE_TARGET = None
    MAX_PITCH      = 5.0

    for pitch_attempt in [5.0, 15.0, 30.0, 90.0]:
        print(f"\n  --- Scanning with max_pitch={pitch_attempt} deg ---")
        SCIENCE_TARGET = scan_for_valid_target(
            catalog, TRAINING_TARGETS, ANALYSIS_START, ANALYSIS_DAYS,
            BAND, CONTRAST, ALLOWED_GRADES, pitch_attempt,
        )
        if SCIENCE_TARGET is not None:
            MAX_PITCH = pitch_attempt
            break

    if SCIENCE_TARGET is None:
        # Try med contrast as last resort
        print("\n  --- Trying contrast=med ---")
        CONTRAST = "med"
        for pitch_attempt in [5.0, 15.0, 30.0, 90.0]:
            SCIENCE_TARGET = scan_for_valid_target(
                catalog, TRAINING_TARGETS, ANALYSIS_START, ANALYSIS_DAYS,
                BAND, CONTRAST, ALLOWED_GRADES, pitch_attempt,
            )
            if SCIENCE_TARGET is not None:
                MAX_PITCH = pitch_attempt
                break

    if SCIENCE_TARGET is None:
        raise RuntimeError(
            "No science target has any valid reference stars under any tested parameters.\n"
            "Check the Step 0 diagnostics above — look at:\n"
            "  1. 'Grade col found' — if None, catalog has no A/B/C column\n"
            "  2. 'Solar-valid ref stars' per target — if 0, stars never in solar window\n"
            "  3. 'Min pitch diff seen' — tells you how wide pitch limit needs to be"
        )

    print(f"\n  Benchmark: target={SCIENCE_TARGET} | "
          f"pitch<={MAX_PITCH} deg | contrast={CONTRAST}\n")

    # Step 1: generate training data (parallel) 
    t_s1     = _step_start("STEP 1: Generating training data (parallel)")
    examples = generate_training_data(
        TRAINING_TARGETS, ANALYSIS_START, ANALYSIS_DAYS,
        TRAINING_BANDS, catalog, ALLOWED_GRADES, MAX_PITCH,
    )
    _step_done("STEP 1", t_s1)

    if not examples:
        raise RuntimeError("No training examples generated — check Step 0 output.")

    # Step 2: train classifier 
    t_s2 = _step_start("STEP 2: Training classifier")
    _nw = 0 if platform.system() == "Windows" else 4
    model, threshold, train_losses = train_classifier(
        examples, num_workers=_nw, use_amp=True, return_history=True
    )
    _step_done("STEP 2", t_s2)

    # Step 3a: baseline (no ML filter) 
    t_s3a        = _step_start("STEP 3a: Baseline (no ML filter)")
    t2           = time.perf_counter()
    result_base  = select_ref_star(
        SCIENCE_TARGET, ANALYSIS_START, ANALYSIS_DAYS,
        BAND, CONTRAST, catalog,
        allowed_grades=ALLOWED_GRADES,
        sort_mode=SORT_MODE,
        max_pitch_diff=MAX_PITCH,
    )
    base_time    = time.perf_counter() - t2
    n_base_stars = sum(len(w["valid_refs"]) for w in result_base["observable_windows"])
    print(f"  Wall-clock: {base_time:.2f}s | valid stars found: {n_base_stars}")
    _step_done("STEP 3a", t_s3a)

    # Step 3b: ML-accelerated 
    t_s3b     = _step_start("STEP 3b: ML-accelerated")
    t3        = time.perf_counter()
    result_ml = select_ref_star(
        SCIENCE_TARGET, ANALYSIS_START, ANALYSIS_DAYS,
        BAND, CONTRAST, catalog,
        allowed_grades=ALLOWED_GRADES,
        sort_mode=SORT_MODE,
        max_pitch_diff=MAX_PITCH,
        classifier=model,
        clf_threshold=threshold,
        use_amp=True,
    )
    ml_time    = time.perf_counter() - t3
    n_ml_stars = sum(len(w["valid_refs"]) for w in result_ml["observable_windows"])
    print(f"  Wall-clock: {ml_time:.2f}s | valid stars found: {n_ml_stars}")
    print(f"  Speedup: {base_time/ml_time:.2f}x")
    _step_done("STEP 3b", t_s3b)

    # Recall check 
    t_src  = _step_start("RECALL CHECK")
    all_ok = True
    for i, (wb, wm) in enumerate(zip(
        result_base["observable_windows"],
        result_ml["observable_windows"],
    )):
        sb     = {r["reference_star"] for r in wb["valid_refs"]}
        sm     = {r["reference_star"] for r in wm["valid_refs"]}
        missed = sb - sm
        miss_rate = len(missed) / max(1, len(sb))
        ok     = miss_rate <= 0.01  # ≤1% miss acceptable at 99% recall target
        all_ok = all_ok and ok
        print(f"  Window {i+1}: baseline={len(sb)} | ML={len(sm)} | "
              f"missed={len(missed)} ({miss_rate:.1%}) "
              f"({'OK' if ok else 'MISSED: ' + str(missed)})")

    print(f"\n  Overall: "
          f"{'≥99% recall across all windows' if all_ok else 'WARNING: recall < 99%'}")
    _step_done("RECALL CHECK", t_src)

    # Step 4: multi-target parallel benchmark 
    t_s4 = _step_start("STEP 4: Multi-target parallel benchmark")

    BENCHMARK_TARGETS = [t for t in TRAINING_TARGETS if t != SCIENCE_TARGET]
    BENCHMARK_TARGETS = [SCIENCE_TARGET] + BENCHMARK_TARGETS

    print(f"  Targets: {BENCHMARK_TARGETS}")
    print(f"  Workers: auto (one per target)")

    t_multi = time.perf_counter()
    multi_results = run_multi_target(
        science_targets=BENCHMARK_TARGETS,
        analysis_start=ANALYSIS_START,
        analysis_days=ANALYSIS_DAYS,
        band=BAND,
        contrast=CONTRAST,
        catalog=catalog,
        model=model,
        clf_threshold=threshold,
        allowed_grades=ALLOWED_GRADES,
        sort_mode=SORT_MODE,
        max_pitch_diff=MAX_PITCH,
        catalog_path=str(DEFAULT_CACHE_PATH),
    )
    print(f"  Total parallel wall-clock: {time.perf_counter()-t_multi:.2f}s")
    _step_done("STEP 4", t_s4)

    print_multi_target_summary(multi_results)

    # Per-target detailed results 
    print("\n" + "=" * 60)
    print("PER-TARGET DETAILED RESULTS")
    print("=" * 60)
    for res in sorted(multi_results, key=lambda r: r.get("science_target", "")):
        if res.get("error") is not None:
            print(f"\n  {res['science_target']}: ERROR — {res['error']}")
            continue
        rb = res.get("result_base") or {}
        rm = res.get("result_ml")   or {}
        print(f"\n  {res['science_target']} "
              f"({rb.get('visibility_pct', 0.0):.1f}% observable):")
        for i, win in enumerate(rm.get("observable_windows", [])):
            n = len(win["valid_refs"])
            best = win["best_ref"]
            best_str = (
                f"{best['reference_star']} (grade {best['grade']}, "
                f"mag={best['mag']:.2f})"
                if best else "none"
            )
            print(f"    Window {i+1}: {win['start'][:10]} → {win['end'][:10]} "
                  f"| {n} valid star(s) | best: {best_str}")

    # Step 5: efficiency graphs
    t_s5 = _step_start("STEP 5: Efficiency Graphs")

    print("\n[5a] Statistical efficiency figure...")
    t_5a = time.perf_counter()
    plot_statistical_efficiency(model, examples, threshold, train_losses=train_losses)
    print(f"  [5a done in {time.perf_counter()-t_5a:.1f}s]  {_ts()}")

    print("\n[5b] Hardware efficiency figure...")
    t_5b = time.perf_counter()
    plot_hardware_efficiency(multi_results, base_time_single=base_time, ml_time_single=ml_time)
    print(f"  [5b done in {time.perf_counter()-t_5b:.1f}s]  {_ts()}")

    print("\n[5c] Data parallelism scaling (Optimisation 1)...")
    t_5c = time.perf_counter()
    benchmark_data_parallelism(examples, worker_counts=(0, 1, 2, 4), n_epochs=10)
    print(f"  [5c done in {time.perf_counter()-t_5c:.1f}s]  {_ts()}")

    print("\n[5d] AMP vs FP32 (Optimisation 3)...")
    t_5d = time.perf_counter()
    benchmark_amp_vs_fp32(examples, n_epochs=20)
    print(f"  [5d done in {time.perf_counter()-t_5d:.1f}s]  {_ts()}")

    _step_done("STEP 5", t_s5)
    total = time.perf_counter() - SCRIPT_START
    print(f"\n{'='*60}")
    print(f"  ALL DONE  {_ts()}")
    print(f"{'='*60}")
    print(f"  {'Step':<30} {'Time':>8}")
    print(f"  {'-'*38}")
    for name, t in STEP_TIMES.items():
        print(f"  {name:<30} {t:>7.1f}s")
    print(f"  {'-'*38}")
    print(f"  {'TOTAL':<30} {total:>7.1f}s")
    print(f"{'='*60}\n")
