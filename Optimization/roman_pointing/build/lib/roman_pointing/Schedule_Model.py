import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import permutations, combinations as iter_combinations, product as cart_product

import astropy.units as u
from astropy.time import Time

from roman_pointing.roman_pointing import calcRomanAngles, getL2Positions
from roman_pointing.roman_observability import (
    get_target_coords,
    compute_keepout,
    compute_visibility_fraction,
    print_visibility_windows,
    get_target_availability_windows,
)
from roman_pointing.Reference_Star_Selection_Tool import load_catalog, select_ref_star
from roman_pointing.Method import HLC_B1, SPEC, WFOV, PLANET_DATA_FILES
from roman_pointing.utils import compute_slew_time
from sqlalchemy import text

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

ANALYSIS_START = "2027-01-01T00:00:00"
ANALYSIS_DAYS  = 365
TIME_STEP_DAYS = 1
MAX_RETRIES    = 20  # max random target-pair attempts before giving up

# ADI observation parameters (standard Roll A–B–A–B sequence).
#
#   N              : half-sequence length; the full sequence contains 2*N target visits
#   R              : ratio of reference-to-target visit duration (default 1/4)
#   N_slew         : number of slews per single-target ADI sequence — 1 initial slew
#                    to the reference star plus transitions per the sequence diagram
#   N_rollchange   : number of roll-angle changes = 2*(N-1) for the default sequence
#   t_rollchange_s : time per roll change (s); 30-deg movement at +/-15 deg default
#   t_howfsc_s     : HOWFSC calibration overhead per science target (s)
#   t_slew_default_s : default per-slew time assuming a 90-deg target→reference
#                      separation (used as a fallback; actual slew times are computed)
ADI_PARAMS = {
    "N":                4,
    "R":                0.25,
    "N_slew":           5,
    "N_rollchange":     6,     # 2*(N-1) = 6 for N=4
    "t_rollchange_s":   600,   # 10 min per roll change
    "t_howfsc_s":       900,   # HOWFSC overhead per science target
    "t_slew_default_s": 1800,  # 30 min default (90-deg separation assumption)
}

# ---------------------------------------------------------------------------
# REFERENCE STAR INTEGRATION TIME
# ---------------------------------------------------------------------------

def get_ref_star_vmag(ref_star_name: str) -> float:
    """Get the V magnitude of a reference star from SIMBAD.

    Args:
        ref_star_name (str): Reference star name

    Returns:
        float: V magnitude, or 2.26 (corgietc default) if not found
    """
    from astroquery.simbad import Simbad

    try:
        simbad = Simbad()
        simbad.add_votable_fields("V")
        result = simbad.query_object(ref_star_name)
        if result is not None and len(result) > 0:
            val = result["V"][0]
            if not hasattr(val, 'mask') and val is not None:
                return float(val)
    except Exception:
        pass

    return 2.26


def compute_inttime_with_ref_star(
    pl_name: str,
    band: int,
    engine,
    ref_results: dict,
    sci_name: str,
    months_at_l2: float = 21,
) -> dict:
    """Compute integration time for a planet using the actual selected reference star.

    Looks up the best reference star from ref_results for sci_name, fetches its
    vmag from the database, and passes it into corgietc so the noise penalty
    reflects the real reference star rather than the default.

    Args:
        pl_name (str): Planet name in Orbits table (e.g. '47 UMa b')
        band (int): Band number — 1, 3, or 4
        engine: SQLAlchemy engine connected to plandb
        ref_results (dict): Output of select_ref_star keyed by science target name
        sci_name (str): Science target name (key into ref_results)
        months_at_l2 (float): Mission elapsed time for detector degradation (default 21)

    Returns:
        dict with keys:
            't'             : decimal year array
            'WA_arcsec'     : working angle array (arcsec)
            'dMag'          : delta magnitude array
            'intTime_hours' : integration time (hours), NaN where not observable
            'ref_star'      : reference star name used
            'ref_star_vmag' : reference star V magnitude used
    """
    import os, json, copy, shutil
    import EXOSIMS.Prototypes.TargetList
    import EXOSIMS.Prototypes.TimeKeeping

    # ── Get reference star name and vmag ─────────────────────────────────────
    ref_star_name = None
    result = ref_results.get(sci_name, {})
    for window in result.get("observable_windows", []):
        if window.get("best_ref") is not None:
            ref_star_name = window["best_ref"]["reference_star"]
            break

    if ref_star_name is None:
        ref_star_vmag    = 2.26
        ref_star_spectype = "a0v"
    else:
        ref_star_vmag    = get_ref_star_vmag(ref_star_name)
        ref_star_spectype = "a0v"   # spectral type lookup can be added if needed

    tref_per_ttar = ADI_PARAMS["R"]

    # ── Setup corgietc ────────────────────────────────────────────────────────
    os.environ["CGI_NOISE_DATA_DIR"] = r"C:\Users\tasni\roman_repos\cgi_noise\cgi_noise\data"
    os.environ["CORGIETC_DATA_DIR"]  = r"C:\Users\tasni\roman_repos\corgietc\corgietc\data"

    cache_dir = os.path.expanduser(r"~\.EXOSIMS\cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    scriptfile = os.path.join(os.environ["CORGIETC_DATA_DIR"], "scripts", "CGI_Noise.json")
    with open(scriptfile) as f:
        specs = json.loads(f.read())

    TL = EXOSIMS.Prototypes.TargetList.TargetList(**copy.deepcopy(specs))
    OS = TL.OpticalSystem
    vmag_ref = float(TL.Vmag[0])

    # match mode by wavelength
    band_wavelength = {1: 575, 3: 730, 4: 825}
    target_nm = band_wavelength[band]
    mode = next(
        (m for m in OS.observingModes if abs(m["lam"].to_value(u.nm) - target_nm) < 20),
        OS.observingModes[0],
    )

    # apply reference star settings
    mode["RefStar_V_mag"]               = ref_star_vmag
    mode["RefStar_SpectralType"]        = ref_star_spectype
    mode["TimeonRefStar_tRef_per_tTar"] = tref_per_ttar

    TK = EXOSIMS.Prototypes.TimeKeeping.TimeKeeping(missionLife=5.25)
    TK.allocate_time(months_at_l2 * 30.4375 * u.d)
    fZ = np.repeat(TL.ZodiacalLight.fZ0, 1)

    # ── Pull orbit data from DB ───────────────────────────────────────────────
    import pandas as pd
    wav      = band_wavelength[band]
    dmag_col = f"dMag_med_{wav}NM"
    query = text(f"""
        SELECT o.t, o.WA, o.{dmag_col} AS dMag, s.sy_vmag
        FROM Orbits o
        JOIN Planets p ON o.pl_id = p.pl_id
        JOIN Stars   s ON p.st_id = s.st_id
        WHERE o.pl_name = :pl_name
          AND o.default_orb = 1
        ORDER BY o.t
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"pl_name": pl_name})

    if df.empty:
        raise ValueError(f"No orbit data found for '{pl_name}' band {band}")

    sy_vmag = float(df["sy_vmag"].iloc[0])

    # scale star flux to target vmag
    flux_scale = 10.0 ** ((vmag_ref - sy_vmag) / 2.5)
    original_fluxes = {}
    for m in OS.observingModes:
        hk = m["hex"]
        if hk in TL.star_fluxes:
            original_fluxes[hk] = TL.star_fluxes[hk].copy()
            TL.star_fluxes[hk]  = original_fluxes[hk] * flux_scale
    original_vmag = TL.Vmag.copy()
    TL.Vmag[0]    = sy_vmag

    # ── Compute integration time at each epoch ────────────────────────────────
    intTimes_hours = np.full(len(df), np.nan)

    for i, row in df.iterrows():
        WA   = np.array([row["WA"]]) * u.arcsec
        dMag = np.array([row["dMag"]])

        if np.isnan(row["dMag"]) or np.isnan(row["WA"]):
            continue
        if (WA[0] < mode["IWA"]) or (WA[0] > mode["OWA"]):
            continue

        JEZ = TL.JEZ0[mode["hex"]] / (row["WA"] ** 2)
        t   = OS.calc_intTime(TL, [0], fZ, JEZ, dMag, WA, mode, TK=TK)
        t_hr = t[0].to(u.hour).value
        if not np.isnan(t_hr) and t_hr > 0:
            intTimes_hours[i] = t_hr

    # restore
    for hk, orig in original_fluxes.items():
        TL.star_fluxes[hk] = orig
    TL.Vmag[0] = original_vmag[0]

    return {
        "t":              df["t"].values,
        "WA_arcsec":      df["WA"].values,
        "dMag":           df["dMag"].values,
        "intTime_hours":  intTimes_hours,
        "ref_star":       ref_star_name,
        "ref_star_vmag":  ref_star_vmag,
        "tref_per_ttar":  tref_per_ttar,
    }



def compare_ref_star_impact(
    pl_name: str,
    band: int,
    engine,
    ref_results: dict,
    sci_name: str,
):
    """Show how much the actual reference star changes integration time vs the default.

    Runs corgietc twice — once with the corgietc default ref star (V=2.26, a0v)
    and once with the actual selected ref star — and prints a comparison.

    Args:
        pl_name (str): Planet name in Orbits table (e.g. '47 UMa b')
        band (int): Band number — 1, 3, or 4
        engine: SQLAlchemy engine connected to plandb
        ref_results (dict): Output of select_ref_star keyed by science target name
        sci_name (str): Science target name
    """
    import os, json, copy, shutil
    import EXOSIMS.Prototypes.TargetList
    import EXOSIMS.Prototypes.TimeKeeping
    import pandas as pd

    band_wavelength = {1: 575, 3: 730, 4: 825}

    # get actual ref star info
    ref_star_name = None
    for window in ref_results.get(sci_name, {}).get("observable_windows", []):
        if window.get("best_ref") is not None:
            ref_star_name = window["best_ref"]["reference_star"]
            break
    actual_vmag = get_ref_star_vmag(ref_star_name) if ref_star_name else 2.26

    print(f"\n{'='*55}")
    print(f"Ref star impact: {pl_name}  band={band}")
    print(f"  Science target : {sci_name}")
    print(f"  Ref star       : {ref_star_name or 'none found'}")
    print(f"  Default vmag   : 2.26  (corgietc default)")
    print(f"  Actual vmag    : {actual_vmag:.2f}  (from SIMBAD)")
    print(f"{'='*55}")

    os.environ["CGI_NOISE_DATA_DIR"] = r"C:\Users\tasni\roman_repos\cgi_noise\cgi_noise\data"
    os.environ["CORGIETC_DATA_DIR"]  = r"C:\Users\tasni\roman_repos\corgietc\corgietc\data"
    cache_dir = os.path.expanduser(r"~\.EXOSIMS\cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    scriptfile = os.path.join(os.environ["CORGIETC_DATA_DIR"], "scripts", "CGI_Noise.json")
    with open(scriptfile) as f:
        specs = json.loads(f.read())

    TL = EXOSIMS.Prototypes.TargetList.TargetList(**copy.deepcopy(specs))
    OS = TL.OpticalSystem
    vmag_ref = float(TL.Vmag[0])

    target_nm = band_wavelength[band]
    mode = next(
        (m for m in OS.observingModes if abs(m["lam"].to_value(u.nm) - target_nm) < 20),
        OS.observingModes[0],
    )

    TK = EXOSIMS.Prototypes.TimeKeeping.TimeKeeping(missionLife=5.25)
    TK.allocate_time(21 * 30.4375 * u.d)
    fZ = np.repeat(TL.ZodiacalLight.fZ0, 1)

    wav      = band_wavelength[band]
    dmag_col = f"dMag_med_{wav}NM"
    query    = text(f"""
        SELECT o.t, o.WA, o.{dmag_col} AS dMag, s.sy_vmag
        FROM Orbits o
        JOIN Planets p ON o.pl_id = p.pl_id
        JOIN Stars   s ON p.st_id = s.st_id
        WHERE o.pl_name = :pl_name AND o.default_orb = 1
        ORDER BY o.t
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"pl_name": pl_name})

    if df.empty:
        print("  No orbit data found.")
        return

    sy_vmag    = float(df["sy_vmag"].iloc[0])
    flux_scale = 10.0 ** ((vmag_ref - sy_vmag) / 2.5)
    original_fluxes = {}
    for m in OS.observingModes:
        hk = m["hex"]
        if hk in TL.star_fluxes:
            original_fluxes[hk] = TL.star_fluxes[hk].copy()
            TL.star_fluxes[hk]  = original_fluxes[hk] * flux_scale
    original_vmag = TL.Vmag.copy()
    TL.Vmag[0]    = sy_vmag

    def _run(ref_vmag):
        mode["RefStar_V_mag"]               = ref_vmag
        mode["TimeonRefStar_tRef_per_tTar"] = ADI_PARAMS["R"]
        results = []
        for _, row in df.iterrows():
            WA   = np.array([row["WA"]]) * u.arcsec
            dMag = np.array([row["dMag"]])
            if np.isnan(row["dMag"]) or np.isnan(row["WA"]):
                results.append(np.nan)
                continue
            if (WA[0] < mode["IWA"]) or (WA[0] > mode["OWA"]):
                results.append(np.nan)
                continue
            JEZ   = TL.JEZ0[mode["hex"]] / (row["WA"] ** 2)
            t_hr  = OS.calc_intTime(TL, [0], fZ, JEZ, dMag, WA, mode, TK=TK)[0].to(u.hour).value
            results.append(t_hr if (not np.isnan(t_hr) and t_hr > 0) else np.nan)
        return np.array(results)

    t_default = _run(2.26)
    t_actual  = _run(actual_vmag)

    for hk, orig in original_fluxes.items():
        TL.star_fluxes[hk] = orig
    TL.Vmag[0] = original_vmag[0]

    valid = np.isfinite(t_default) & np.isfinite(t_actual)
    if not np.any(valid):
        print("  No observable epochs to compare.")
        return

    med_default = np.nanmedian(t_default[valid])
    med_actual  = np.nanmedian(t_actual[valid])
    pct_change  = 100 * (med_actual - med_default) / med_default

    print(f"  Observable epochs : {valid.sum()}")
    print(f"  Median intTime (default ref V=2.26) : {med_default:.3f} hrs")
    print(f"  Median intTime (actual  ref V={actual_vmag:.2f}) : {med_actual:.3f} hrs")
    print(f"  Change                               : {pct_change:+.1f}%")
    print(f"{'='*55}\n")


def load_planet_data() -> dict:
    """Load all .npy planet detection files into a dict keyed by (target, planet, band)."""
    return {
        key: np.load(path, allow_pickle=True).item()
        for key, path in PLANET_DATA_FILES.items()
    }


def ref_visit_time_s(t_int_science_s: float, adi: dict, tref_per_ttar: float = None) -> float:
    """Total time (s) for the 3 reference-star visits in one ADI sequence.

    If tref_per_ttar is provided (the ratio passed into corgietc), the wall clock
    uses that directly: t_ref = tref_per_ttar * t_int_science.
    Otherwise falls back to the ADI sequence formula: 3 * R * T_int / (2*N).
    """
    if tref_per_ttar is not None:
        return tref_per_ttar * t_int_science_s
    return 3.0 * adi["R"] * t_int_science_s / (2.0 * adi["N"])


def get_integration_time(planet_data: dict, key: tuple, mode: str = "con"):
    """Return (decimal_year, t_int_hours) for observable time steps only."""
    d     = planet_data[key]
    t     = d["decimal_year"]
    t_int = d[f"integration_time_hours_{mode}"]["median"]
    mask  = ~d["solar_keepout"] & ~d["GB_keepout"] & np.isfinite(t_int)
    return t[mask], t_int[mask]


def get_completeness(planet_data: dict, key: tuple):
    """Return (decimal_year, det_probability) for a planet/band key."""
    d = planet_data[key]
    return d["decimal_year"], d["det_probability"]


def target_median_completeness(planet_data: dict, target_name: str) -> float | None:
    """
    Median detection probability across all planet/band keys for a target.
    Returns None if no data exists; includes zero-probability points.
    """
    keys = [k for k in planet_data if k[0] == target_name]
    if not keys:
        return None
    _, prob = get_completeness(planet_data, keys[0])
    finite  = prob[np.isfinite(prob)]
    return float(np.median(finite)) if len(finite) > 0 else 0.0


def get_target_tint_s(planet_data: dict, target_name: str) -> float:
    """Return median observable integration time (seconds) for a science target.

    Collects all (target, planet, band) keys for target_name, pools the
    observable (non-keepout, finite) integration times across all of them,
    and returns their median in seconds.  Falls back to 3600.0 when no
    valid data exists.
    """
    keys = [k for k in planet_data if k[0] == target_name]
    if not keys:
        return 3600.0

    all_t_int_hrs = []
    for key in keys:
        _, t_int_hrs = get_integration_time(planet_data, key, mode="con")
        finite = t_int_hrs[np.isfinite(t_int_hrs)]
        all_t_int_hrs.extend(finite.tolist())

    if not all_t_int_hrs:
        return 3600.0

    return float(np.median(all_t_int_hrs)) * 3600.0


def find_overlapping_windows(windows1: list, windows2: list):
    """Return the first (w1, w2) pair whose date ranges overlap, or (None, None)."""
    for w1 in windows1:
        if w1.get("best_ref") is None:
            continue
        t1_start = Time(w1["start"], format="iso", scale="utc")
        t1_end   = Time(w1["end"],   format="iso", scale="utc")
        for w2 in windows2:
            if w2.get("best_ref") is None:
                continue
            t2_start = Time(w2["start"], format="iso", scale="utc")
            t2_end   = Time(w2["end"],   format="iso", scale="utc")
            if min(t1_end, t2_end) > max(t1_start, t2_start):
                return w1, w2
    return None, None


def find_multi_target_overlap(ref_results: dict, target_names: list) -> dict | None:
    """Find a set of windows (one per target) that all overlap simultaneously.

    For each target, collects only windows where best_ref is not None.
    Tries every Cartesian product of window choices — one window per target —
    and returns the first combination whose date ranges all intersect.

    Returns {target_name: window_dict} or None if no overlap exists or any
    target has no valid windows.
    """
    valid_windows = {}
    for name in target_names:
        result = ref_results.get(name, {})
        wins = [
            w for w in result.get("observable_windows", [])
            if w.get("best_ref") is not None
        ]
        if not wins:
            return None
        valid_windows[name] = wins

    window_lists = [valid_windows[name] for name in target_names]
    for combo in cart_product(*window_lists):
        starts = [Time(w["start"], format="iso", scale="utc") for w in combo]
        ends   = [Time(w["end"],   format="iso", scale="utc") for w in combo]
        if min(ends) > max(starts):
            ref_names = [w["best_ref"]["reference_star"] for w in combo]
            if len(set(ref_names)) == len(ref_names):  # require distinct ref stars
                return {name: w for name, w in zip(target_names, combo)}

    return None


def overlap_time_array(w1: dict, w2: dict):
    """
    Given two window dicts (with 'start'/'end' ISO strings),
    return (overlap_start, overlap_end, t_obs array).
    """
    from datetime import datetime
    # Use string-max/min on ISO dates (YYYY-MM-DD lexicographic order is correct)
    start_iso = max(w1["start"], w2["start"])
    end_iso   = min(w1["end"],   w2["end"])
    start     = Time(start_iso, format="iso", scale="utc")
    end       = Time(end_iso,   format="iso", scale="utc")
    days      = (
        datetime.fromisoformat(end_iso) - datetime.fromisoformat(start_iso)
    ).days
    t_obs = start + np.arange(days) * u.d
    return start, end, t_obs


def build_slew_matrix(obj_keys: list, obj_meta: dict, all_coords: dict, t_mid: Time) -> dict:
    """
    Compute slew time (seconds) between every ordered pair of objects at t_mid.
    Returns a dict keyed by (k1, k2) with scalar float values.
    """
    slew_matrix = {}
    for i, k1 in enumerate(obj_keys):
        for k2 in obj_keys[i + 1:]:
            coord1 = all_coords[obj_meta[k1]["name"]]
            coord2 = all_coords[obj_meta[k2]["name"]]
            # compute_slew_time requires an array; wrap scalar t_mid in a 1-element array
            t_arr = Time([t_mid.jd], format="jd", scale=t_mid.scale)
            _, slew_s = compute_slew_time(coord1, coord2, t_arr)
            val = float(np.mean(slew_s))
            slew_matrix[(k1, k2)] = val
            slew_matrix[(k2, k1)] = val
    return slew_matrix


def build_time_resolved_slew_matrix(
    obj_keys: list, obj_meta: dict, all_coords: dict, t_obs
) -> dict:
    """
    Compute slew time arrays (seconds) for every ordered pair over all of t_obs.
    Returns a dict keyed by (k1, k2) with 1-D arrays of length len(t_obs).
    """
    slew_matrix = {}
    for i, k1 in enumerate(obj_keys):
        for k2 in obj_keys[i + 1:]:
            coord1 = all_coords[obj_meta[k1]["name"]]
            coord2 = all_coords[obj_meta[k2]["name"]]
            _, slew_s = compute_slew_time(coord1, coord2, t_obs)
            slew_matrix[(k1, k2)] = slew_s
            slew_matrix[(k2, k1)] = slew_s
    return slew_matrix


def interpolate_target_data(
    planet_data: dict, target_name: str, t_obs_yr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For a science target, return integration-time (hrs) and completeness arrays
    interpolated at each decimal-year value in t_obs_yr.

    Integration time is NaN at dates where the target is in keepout or has no data.
    Completeness is summed across all planet/band keys for the target (detecting
    any planet).  Returns (t_int_hrs, completeness) both of length len(t_obs_yr).
    """
    keys = [k for k in planet_data if k[0] == target_name]
    if not keys:
        return (
            np.full(len(t_obs_yr), np.nan),
            np.zeros(len(t_obs_yr)),
            np.ones(len(t_obs_yr), dtype=bool),
        )

    t_int_accum = np.zeros(len(t_obs_yr))
    t_int_count = np.zeros(len(t_obs_yr))
    comp_accum  = np.zeros(len(t_obs_yr))

    for key in keys:
        d         = planet_data[key]
        t_full    = d["decimal_year"]
        t_int_raw = d["integration_time_hours_con"]["median"].copy().astype(float)
        prob_raw  = d["det_probability"].copy().astype(float)

        # Mark keepout and non-finite integration times as NaN
        bad = d["solar_keepout"] | d["GB_keepout"] | ~np.isfinite(t_int_raw)
        t_int_raw[bad] = np.nan

        # Interpolate integration time only over valid (non-NaN) points
        valid = ~bad
        if np.sum(valid) > 1:
            t_int_interp = np.interp(
                t_obs_yr, t_full[valid], t_int_raw[valid],
                left=np.nan, right=np.nan,
            )
            # np.interp doesn't accept nan for left/right — fix clipping manually
            t_int_interp[t_obs_yr < t_full[valid].min()] = np.nan
            t_int_interp[t_obs_yr > t_full[valid].max()] = np.nan
        else:
            t_int_interp = np.full(len(t_obs_yr), np.nan)

        finite_mask = np.isfinite(t_int_interp)
        t_int_accum[finite_mask] += t_int_interp[finite_mask]
        t_int_count[finite_mask] += 1

        # Completeness: interpolate over all dates (0 outside range)
        comp_accum += np.interp(t_obs_yr, t_full, prob_raw, left=0.0, right=0.0)

    # Average integration time across keys; NaN where no key had valid data
    with np.errstate(invalid="ignore", divide="ignore"):
        t_int_hrs = np.where(t_int_count > 0, t_int_accum / t_int_count, np.nan)

    # keepout: a date is in keepout when no key has valid (finite) integration
    # time data for it.  This is consistent with get_target_tint_s, which
    # collects dates where t_int_hrs is finite.  The old ko_accum > 0 approach
    # incorrectly flagged dates that are simply outside some keys' date ranges
    # (np.interp defaults left/right=1.0 → treated as keepout), causing the
    # Pareto analysis to see zero observable dates even for valid combinations.
    in_keepout = ~np.isfinite(t_int_hrs)

    return t_int_hrs, comp_accum, in_keepout


def score_sequences_pareto(
    t_obs,
    obj_keys: list,
    obj_meta: dict,
    slew_matrix_ts: dict,
    adi: dict,
    planet_data: dict,
) -> list[dict]:
    """
    For every permutation of obj_keys, compute wall-clock time (hrs) and combined
    detection completeness at each date in t_obs.

    Wall-clock formula per permutation:
        T_wall = sum(visit_times) + sum(slew_times) + n_science * fixed_per_science

    where visit_time for science keys is T_int (the full ETC estimate), visit_time
    for reference keys is derived as  3 * R * T_int_paired / (2*N),  and
    fixed_per_science = N_rollchange * T_rollchange + T_HOWFSC.

    Reference entries in obj_meta must carry a "paired_with" key naming their
    science counterpart so that the reference visit time can be scaled correctly.

    Returns a list of dicts (one per sequence) each containing:
      sequence, label, wall_clock_hrs (array), completeness (array).
    """
    t_obs_yr = t_obs.decimalyear

    science_keys = [k for k in obj_keys if obj_meta[k]["type"] == "science"]
    ref_keys     = [k for k in obj_keys if obj_meta[k]["type"] == "reference"]
    n_science    = len(science_keys)

    tint_s     = {}
    comp_arr   = {}
    in_keepout = {}

    for k in science_keys:
        t_int_hrs, comp, ko = interpolate_target_data(
            planet_data, obj_meta[k]["name"], t_obs_yr
        )
        # Fallback: use median observable t_int on keepout / no-data dates
        median_s = (
            float(np.nanmedian(t_int_hrs)) * 3600.0
            if np.any(np.isfinite(t_int_hrs))
            else 3600.0
        )
        tint_s[k]     = np.where(np.isfinite(t_int_hrs), t_int_hrs * 3600.0, median_s)
        comp_arr[k]   = comp
        in_keepout[k] = ko

    for k in ref_keys:
        paired_k = obj_meta[k].get("paired_with")
        if paired_k and paired_k in tint_s:
            # 3 reference visits each lasting R * T_int_science / (2*N)
            tint_s[k] = 3.0 * adi["R"] * tint_s[paired_k] / (2.0 * adi["N"])
        else:
            # fallback: one HOWFSC worth of time
            tint_s[k] = np.full(len(t_obs), float(adi["t_howfsc_s"]))

    # Fixed overhead per science target: roll changes + HOWFSC
    fixed_s = n_science * (
        adi["N_rollchange"] * adi["t_rollchange_s"]
        + adi["t_howfsc_s"]
    )

    combined_comp = sum(comp_arr[k] for k in science_keys)

    # Dates where any science target uses the fallback (in keepout)
    any_keepout = np.zeros(len(t_obs), dtype=bool)
    for k in science_keys:
        any_keepout |= in_keepout[k]

    results = []
    for seq in permutations(obj_keys):
        slew_s         = sum(slew_matrix_ts[(seq[i], seq[i + 1])] for i in range(len(seq) - 1))
        visit_s        = sum(tint_s[k] for k in seq)
        wall_clock_hrs = (visit_s + slew_s + fixed_s) / 3600.0

        results.append({
            "sequence":       seq,
            "label":          " -> ".join(seq),
            "wall_clock_hrs": wall_clock_hrs,
            "completeness":   combined_comp.copy(),
            "any_keepout":    any_keepout.copy(),
        })

    return results


def pareto_front_indices(wall_clock: np.ndarray, completeness: np.ndarray) -> np.ndarray:
    """
    Return indices of Pareto-optimal points (minimize wall_clock, maximize completeness).
    Uses an O(N log N) sweep: sort by wall_clock ascending, keep points that
    strictly improve completeness.
    """
    valid = np.isfinite(wall_clock) & np.isfinite(completeness)
    if not np.any(valid):
        return np.array([], dtype=int)

    idx_valid   = np.where(valid)[0]
    order       = idx_valid[np.argsort(wall_clock[idx_valid])]
    pareto_idx  = []
    max_comp    = -np.inf
    for i in order:
        if completeness[i] > max_comp:
            pareto_idx.append(i)
            max_comp = completeness[i]
    return np.array(pareto_idx)


def plot_pareto(
    targ1_name: str,
    targ2_name: str,
    sequence_results: list[dict],
    t_obs,
    obj_meta: dict,
    planet_data: dict,
):
    """Three-panel per-combination Pareto analysis.

    Panel 1 — Pareto front (wall-clock vs completeness)
        One representative point per sequence, positioned at its median
        wall-clock and median completeness over observable dates.
        The red step-line is the Pareto front across those medians.
        Because completeness is sequence-independent (only slew timing
        differs), each sequence sits at the same y-level; the Pareto
        front simply picks the fastest sequence and shows how much faster
        it is.  Background grey dots show every individual date's scatter.

    Panel 2 — Wall-clock over time
        Line plot (x = observation date, y = wall-clock hrs) with one
        coloured line per sequence.  Shows which sequence ordering is
        cheapest and *when* the observation window is tightest.

    Panel 3 — Science yield over time
        Left axis: combined completeness (same for all sequences).
        Right axis: total integration time summed across science targets.
        High completeness + low integration time = high-efficiency date.
    """
    import matplotlib.cm as cm

    n_seq    = len(sequence_results)
    colors   = cm.tab10(np.linspace(0, 1, min(n_seq, 10)))
    t_obs_yr = t_obs.decimalyear

    # ── Pre-compute integration time once (shared across sequences) ──────────
    science_keys_all = [k for k in sequence_results[0]["sequence"]
                        if obj_meta[k]["type"] == "science"]
    t_int_total = np.zeros(len(t_obs))
    for k in science_keys_all:
        t_int_hrs, _, _ = interpolate_target_data(
            planet_data, obj_meta[k]["name"], t_obs_yr
        )
        t_int_total += np.where(np.isfinite(t_int_hrs), t_int_hrs, 0.0)

    # Shared completeness and keepout (same for all sequences)
    comp_shared = sequence_results[0]["completeness"]
    ko_shared   = sequence_results[0]["any_keepout"]
    obs_mask    = np.isfinite(comp_shared) & ~ko_shared

    if not np.any(obs_mask):
        print(f"plot_pareto: no observable dates for {targ1_name} + {targ2_name} "
              f"— skipping plot.")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))

    # ── Panel 1: Pareto front ────────────────────────────────────────────────
    # Background: all (wall_clock[t], completeness[t]) over observable dates
    for i, res in enumerate(sequence_results):
        mask = np.isfinite(res["wall_clock_hrs"]) & ~res["any_keepout"]
        if np.any(mask):
            ax1.scatter(res["wall_clock_hrs"][mask], comp_shared[mask],
                        color=colors[i % len(colors)], alpha=0.12, s=10, zorder=2)

    # Median per sequence → representative Pareto points
    med_wc   = []
    med_comp = []
    seq_labels = []
    for i, res in enumerate(sequence_results):
        mask = np.isfinite(res["wall_clock_hrs"]) & ~res["any_keepout"]
        if not np.any(mask):
            med_wc.append(np.nan)
            med_comp.append(np.nan)
        else:
            med_wc.append(float(np.median(res["wall_clock_hrs"][mask])))
            med_comp.append(float(np.median(comp_shared[mask])))
        seq_labels.append(" → ".join(obj_meta[k]["name"] for k in res["sequence"]))

    med_wc   = np.array(med_wc)
    med_comp = np.array(med_comp)
    valid    = np.isfinite(med_wc) & np.isfinite(med_comp)

    for i in np.where(valid)[0]:
        ax1.scatter(med_wc[i], med_comp[i],
                    color=colors[i % len(colors)], s=110, zorder=5,
                    edgecolors="black", linewidths=0.8, label=seq_labels[i])

    # Pareto front across median points
    pf_idx = pareto_front_indices(med_wc[valid], med_comp[valid])
    if len(pf_idx) > 0:
        # pf_idx is relative to the valid subset — map back to full arrays
        valid_positions = np.where(valid)[0]
        pf_wc   = med_wc[valid][pf_idx]
        pf_comp = med_comp[valid][pf_idx]
        so      = np.argsort(pf_wc)
        ax1.step(pf_wc[so], pf_comp[so], where="post",
                 color="red", linewidth=2.5, zorder=6, label="Pareto front")
        ax1.scatter(pf_wc, pf_comp, color="red", s=80, zorder=7,
                    edgecolors="darkred", linewidths=1.0)
        # Annotate Pareto-optimal sequences
        for idx in pf_idx:
            orig_i = valid_positions[idx]
            ax1.annotate(seq_labels[orig_i],
                         (med_wc[valid][idx], med_comp[valid][idx]),
                         fontsize=7, xytext=(6, 4), textcoords="offset points",
                         zorder=8)

    ax1.set_xlabel("Wall-clock time (hrs)  [median over observable dates]")
    ax1.set_ylabel("Combined completeness  [median]")
    ax1.set_title("Pareto front — sequences\n"
                  "Large dots = median per sequence; grey = individual dates")
    if np.any(valid):
        ax1.legend(fontsize=7, loc="lower right", framealpha=0.9,
                   title="Sequence", title_fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # ── Panel 2: Wall-clock over time ────────────────────────────────────────
    for i, res in enumerate(sequence_results):
        wc   = res["wall_clock_hrs"].copy().astype(float)
        mask = np.isfinite(wc) & ~res["any_keepout"]
        wc[~mask] = np.nan          # break line at non-observable dates
        color = colors[i % len(colors)]
        ax2.plot(t_obs_yr, wc, color=color, linewidth=1.4, alpha=0.85,
                 label=seq_labels[i])
        # Mark the minimum wall-clock date for this sequence
        if np.any(mask):
            best_t = t_obs_yr[mask][np.argmin(wc[mask])]
            best_v = np.nanmin(wc[mask])
            ax2.scatter(best_t, best_v, color=color, s=55, zorder=5,
                        edgecolors="black", linewidths=0.7)

    ax2.set_xlabel("Observation date (decimal year)")
    ax2.set_ylabel("Wall-clock time (hrs)")
    ax2.set_title("Wall-clock over the observation window\n"
                  "Dots mark each sequence's minimum-cost date")
    ax2.legend(fontsize=7, loc="best", title="Sequence", title_fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.4)

    # ── Panel 3: Science yield over time ─────────────────────────────────────
    comp_plot = comp_shared.copy().astype(float)
    comp_plot[~obs_mask] = np.nan

    t_int_plot = t_int_total.copy()
    t_int_plot[~obs_mask] = np.nan

    ax3.plot(t_obs_yr, comp_plot, color="steelblue", linewidth=2.0,
             label="Combined completeness", zorder=4)
    ax3.set_xlabel("Observation date (decimal year)")
    ax3.set_ylabel("Combined completeness", color="steelblue")
    ax3.tick_params(axis="y", labelcolor="steelblue")

    ax3r = ax3.twinx()
    ax3r.plot(t_obs_yr, t_int_plot, color="darkorange", linewidth=1.6,
              linestyle="--", label="Total integration time", zorder=3)
    ax3r.set_ylabel("Total integration time (hrs)", color="darkorange")
    ax3r.tick_params(axis="y", labelcolor="darkorange")

    ax3.set_title("Science yield over time\n"
                  "High completeness + low integration = efficient window")
    # Combined legend
    lines1, labs1 = ax3.get_legend_handles_labels()
    lines2, labs2 = ax3r.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="best")
    ax3.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        f"{targ1_name}  +  {targ2_name}  —  observable dates only",
        fontsize=13,
    )
    plt.tight_layout()
    plt.show(block=False)


def plot_sequence_window(
    pareto_results: list[dict],
    t_obs,
    obj_meta: dict,
    planet_data: dict,
    title: str = "",
):
    """Single-panel plot: all sequences' wall-clock time + integration time over the window.

    Wall-clock lines (one per sequence, left axis) are overlaid with the
    shared total science integration time (right axis, dashed black).
    Grey bands mark keepout periods.  The fastest sequence at each date
    stands out naturally as the lowest line.
    """
    import matplotlib.cm as cm
    from matplotlib.lines import Line2D

    n_seq    = len(pareto_results)
    t_yr     = t_obs.decimalyear
    cmap     = cm.tab20(np.linspace(0, 1, n_seq))

    # Integration time is sequence-independent — compute once
    sci_keys = [k for k in pareto_results[0]["sequence"]
                if obj_meta[k]["type"] == "science"]
    t_int_total = np.zeros(len(t_obs))
    any_ko      = np.zeros(len(t_obs), dtype=bool)
    for k in sci_keys:
        t_int, _, ko = interpolate_target_data(
            planet_data, obj_meta[k]["name"], t_yr
        )
        t_int_total += np.where(np.isfinite(t_int), t_int, 0.0)
        any_ko |= ko

    obs_mask = ~any_ko

    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax2 = ax1.twinx()

    # Grey keepout shading
    ax1.fill_between(t_yr, 0, 1, where=any_ko,
                     transform=ax1.get_xaxis_transform(),
                     color="lightgrey", alpha=0.55, label="Keepout", zorder=1)

    # Sort sequences by median wall clock (best first) so the legend is ordered
    def _med(res):
        mask = np.isfinite(res["wall_clock_hrs"]) & obs_mask
        return float(np.median(res["wall_clock_hrs"][mask])) if np.any(mask) else np.inf

    sorted_results = sorted(pareto_results, key=_med)

    legend_handles = []
    for i, res in enumerate(sorted_results):
        wc   = res["wall_clock_hrs"].copy().astype(float)
        mask = np.isfinite(wc) & obs_mask
        wc[~mask] = np.nan
        lw    = 2.2 if i == 0 else 1.0          # thickest = fastest
        alpha = 1.0 if i == 0 else 0.55
        label = " → ".join(obj_meta[k]["name"] for k in res["sequence"])
        line, = ax1.plot(t_yr, wc, color=cmap[i], lw=lw, alpha=alpha,
                         zorder=3 if i == 0 else 2)
        legend_handles.append(Line2D([0], [0], color=cmap[i], lw=lw,
                                     alpha=alpha, label=label))

    # Integration time on right axis
    t_int_plot = t_int_total.copy()
    t_int_plot[~obs_mask] = np.nan
    ax2.plot(t_yr, t_int_plot, color="black", lw=2, ls="--",
             label="Total int. time", zorder=5)
    ax2.set_ylabel("Total science integration time (h)", color="black")
    ax2.tick_params(axis="y", labelcolor="black")

    ax1.set_xlabel("Observation date (decimal year)")
    ax1.set_ylabel("Wall-clock time (h)")
    ax1.set_title(
        (title + "\n" if title else "") +
        "All sequences — wall-clock (coloured) vs integration time (dashed)\n"
        "Bold line = fastest sequence; grey = keepout"
    )
    ax1.grid(True, ls="--", alpha=0.35)

    # Legend: sequences + int-time handle
    legend_handles.append(Line2D([0], [0], color="black", lw=2, ls="--",
                                 label="Total int. time"))
    ax1.legend(handles=legend_handles, fontsize=7, loc="upper left",
               ncol=2, framealpha=0.9, title="Sequence (fastest → slowest)",
               title_fontsize=8)

    plt.tight_layout()
    plt.show(block=False)


def print_optimal_sequence(
    sequence_results: list[dict],
    pareto_results: list[dict],
    obj_meta: dict,
    adi: dict,
    t_obs,
):
    """
    Print the optimal sequence — the one on the Pareto front with the lowest
    wall clock time among observable-only dates — and break down where the
    time goes.
    """
    # Find the sequence with lowest median wall clock on observable dates
    best = None
    best_median = np.inf

    for res in pareto_results:
        mask = np.isfinite(res["wall_clock_hrs"]) & ~res["any_keepout"]
        if not np.any(mask):
            continue
        median_wc = float(np.median(res["wall_clock_hrs"][mask]))
        if median_wc < best_median:
            best_median = median_wc
            best = res

    if best is None:
        print("No fully observable sequence found.")
        return

    seq = best["sequence"]
    mask = np.isfinite(best["wall_clock_hrs"]) & ~best["any_keepout"]

    # Match to score_all_sequences result for the breakdown
    matched = next(
        (r for r in sequence_results if r["sequence"] == seq), None
    )

    print("\n" + "=" * 55)
    print("OPTIMAL SEQUENCE")
    print("=" * 55)
    print(f"  Order:  {' → '.join(seq)}")
    print(f"  Dates:  {np.sum(mask)} observable days in window")
    print()

    wc_hrs   = best["wall_clock_hrs"][mask]
    print(f"  Wall clock time (median): {np.median(wc_hrs):.2f} hrs")
    print(f"  Wall clock time (min):    {np.min(wc_hrs):.2f} hrs")
    print(f"  Wall clock time (max):    {np.max(wc_hrs):.2f} hrs")
    print()

    # Time breakdown
    if matched:
        science_s  = matched["total_science_s"]
        ref_s      = matched["total_ref_s"]
        slew_s     = matched["total_slew_s"]
        howfsc_s   = matched["total_howfsc_s"]
        n_sci      = sum(1 for k in seq if obj_meta[k]["type"] == "science")
        roll_s     = adi["N_rollchange"] * adi["t_rollchange_s"] * n_sci
        total_s    = matched["total_wall_clock_s"]

        print("  Time breakdown:")
        for label, val_s in [
            ("Science integration", science_s),
            ("Reference visits",    ref_s),
            ("Slew",                slew_s),
            ("HOWFSC",              howfsc_s),
            ("Roll changes",        roll_s),
        ]:
            print(f"    {label:<22} {val_s/3600:.2f} hrs  "
                  f"({100*val_s/total_s:.1f}%)")
        print(f"    {'TOTAL':<22} {total_s/3600:.2f} hrs")

    print()
    print(f"  Combined completeness score (median): "
          f"{float(np.median(best['completeness'][mask])):.3f}")
    print("=" * 55)

def print_sequence_time_table(sequence_results: list[dict], obj_meta: dict) -> None:
    """Print slew time and wall-clock time for every visit-order permutation.

    Integration time is identical for all sequences (same targets observed
    regardless of order), so it is printed once as a header note.
    What actually differs is the slew overhead.
    """
    int_hrs = sequence_results[0]["total_science_s"] / 3600.0 if sequence_results else 0.0
    print(f"\n  Science integration time (all sequences): {int_hrs:.2f} h")
    hdr = f"  {'Sequence':<56}  {'Slew (h)':>9}  {'Wall clock (h)':>14}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in sequence_results:
        label    = " → ".join(obj_meta[k]["name"] for k in r["sequence"])
        slew_hrs = r["total_slew_s"] / 3600.0
        wc_hrs   = r["total_wall_clock_hrs"]
        print(f"  {label:<56}  {slew_hrs:>9.2f}  {wc_hrs:>14.2f}")


def get_or_reverse_slew(slew_results: dict, targ1: str, targ2: str) -> dict | None:
    """Look up slew result for a pair in either order; return None if missing."""
    return slew_results.get((targ1, targ2)) or slew_results.get((targ2, targ1))

def calc_wall_clock_time(
    coord1, coord2, t_obs,
    t_int_s: float,
    adi: dict,
    tref_per_ttar: float = None,
):
    """
    Wall-clock time (seconds) for a single science-target ADI observation.

    If tref_per_ttar is provided (the ratio passed into corgietc), the reference
    star contribution is t_int_s * tref_per_ttar — consistent with the noise
    penalty model.  Otherwise falls back to the ADI formula (3*R)/(2*N).

    Formula:
        T_wallclock = T_int * (1 + tref_per_ttar)
                    + N_slew * T_slew
                    + N_rollchange * T_rollchange
                    + T_HOWFSC

    Returns (wall_clock_s, slew_time_s) as arrays over t_obs.
    """
    N              = adi["N"]
    R              = adi["R"]
    N_slew         = adi["N_slew"]
    N_rollchange   = adi["N_rollchange"]
    t_rollchange_s = adi["t_rollchange_s"]
    t_howfsc_s     = adi["t_howfsc_s"]

    ref_fraction = tref_per_ttar if tref_per_ttar is not None else (3.0 * R) / (2.0 * N)

    _, slew_time_s = compute_slew_time(coord1, coord2, t_obs)

    wall_clock_s = (
        t_int_s * (1.0 + ref_fraction)   # science + reference visits
        + N_slew * slew_time_s            # total slew overhead
        + N_rollchange * t_rollchange_s   # roll-angle changes
        + t_howfsc_s                      # HOWFSC calibration
    )
    return wall_clock_s, slew_time_s


def score_all_sequences(
    obj_keys: list,
    obj_meta: dict,
    visit_time: dict,
    slew_matrix: dict,
    adi: dict,
    planet_data: dict,
) -> list[dict]:
    """
    Enumerate all permutations of obj_keys and compute wall-clock cost for each.

    visit_time must be pre-populated:
      - science keys : full ETC integration time (T_int, seconds)
      - reference keys : use ref_visit_time_s(T_int_paired, adi) — i.e.
                         3 * R * T_int_science / (2*N) for 3 reference visits

    Wall-clock formula:
        T_wall = T_science + T_ref + T_slew
               + n_science * (N_rollchange * T_rollchange + T_HOWFSC)

    Returns a list of result dicts sorted by total_wall_clock_s ascending.
    """
    results = []
    for seq in permutations(obj_keys):
        total_slew_s    = sum(slew_matrix[(seq[i], seq[i + 1])] for i in range(len(seq) - 1))
        science_steps   = [s for s in seq if obj_meta[s]["type"] == "science"]
        n_science       = len(science_steps)
        total_science_s = sum(visit_time[s] for s in science_steps)
        total_ref_s     = sum(visit_time[s] for s in seq if obj_meta[s]["type"] == "reference")
        total_howfsc_s  = adi["t_howfsc_s"] * n_science
        roll_s          = adi["N_rollchange"] * adi["t_rollchange_s"] * n_science
        total_wall_s    = (
            total_science_s + total_ref_s + total_slew_s
            + roll_s
            + total_howfsc_s
        )
        completeness = {
            obj_meta[s]["name"]: target_median_completeness(planet_data, obj_meta[s]["name"])
            for s in science_steps
        }
        results.append({
            "sequence":             seq,
            "total_wall_clock_s":   total_wall_s,
            "total_wall_clock_hrs": total_wall_s / 3600.0,
            "total_science_s":      total_science_s,
            "total_ref_s":          total_ref_s,
            "total_howfsc_s":       total_howfsc_s,
            "total_roll_s":         roll_s,
            "total_slew_s":         total_slew_s,
            "completeness":         completeness,
        })
    results.sort(key=lambda x: x["total_wall_clock_s"])
    return results




def score_combination(
    target_names: list,
    ref_results: dict,
    planet_data: dict,
    target_coords: dict,
    adi: dict,
) -> dict | None:
    """Score a combination of N science targets for joint scheduling.

    Finds the first multi-target overlapping observable window, builds
    obj_meta / visit_time, fetches any missing reference star coordinates,
    computes a scalar slew matrix at the overlap midpoint, and scores all
    permutations via score_all_sequences.

    Returns a result dict or None if no overlapping window with valid
    reference stars exists.

    Keys in the result dict:
        targets, refs, overlap_start, overlap_end, overlap_days,
        t_obs, obj_meta, obj_keys, visit_time, slew_matrix,
        all_sequences, best_sequence, best_wall_clock_hrs,
        combined_completeness, efficiency.
    """
    # 1. Find overlapping window for all targets
    overlap_windows = find_multi_target_overlap(ref_results, target_names)
    if overlap_windows is None:
        return None

    # 2. Build obj_meta and visit_time
    obj_meta   = {}
    visit_time = {}
    refs       = []

    for i, sci_name in enumerate(target_names):
        sci_key  = f"sci_{i}"
        ref_key  = f"ref_{i}"
        ref_name = overlap_windows[sci_name]["best_ref"]["reference_star"]
        refs.append(ref_name)

        obj_meta[sci_key] = {"name": sci_name,  "type": "science"}
        obj_meta[ref_key] = {"name": ref_name,  "type": "reference",
                             "paired_with": sci_key}

        t_int_s = get_target_tint_s(planet_data, sci_name)
        visit_time[sci_key] = t_int_s
        visit_time[ref_key] = ref_visit_time_s(t_int_s, adi)

    obj_keys = list(obj_meta.keys())

    # Warn for very large permutation spaces
    n_perms = math.factorial(len(obj_keys))
    if n_perms > 40320:
        print(f"  Warning: {len(obj_keys)} objects => {n_perms:,} permutations. "
              "This may be slow.")

    # 3. Compute overlap time array from the intersection of all windows
    all_starts_iso = [overlap_windows[n]["start"] for n in target_names]
    all_ends_iso   = [overlap_windows[n]["end"]   for n in target_names]
    overlap_start_iso = max(all_starts_iso)
    overlap_end_iso   = min(all_ends_iso)

    # overlap_time_array expects two window dicts with start/end keys
    dummy_window = {"start": overlap_start_iso, "end": overlap_end_iso}
    overlap_start, overlap_end, t_obs = overlap_time_array(dummy_window, dummy_window)
    overlap_days = float((overlap_end - overlap_start).to(u.day).value)
    if overlap_days <= 0:
        return None

    # 4. Fetch missing reference star coords (mutates target_coords in place)
    new_names = [n for n in refs if n not in target_coords]
    if new_names:
        target_coords.update(get_target_coords(new_names))

    # 5. Build scalar slew matrix at overlap midpoint
    t_mid = Time(
        (overlap_start.mjd + overlap_end.mjd) / 2.0,
        format="mjd", scale="utc",
    )
    slew_matrix = build_slew_matrix(obj_keys, obj_meta, target_coords, t_mid)

    # 6. Score all permutations
    all_sequences = score_all_sequences(
        obj_keys, obj_meta, visit_time, slew_matrix, adi, planet_data
    )
    if not all_sequences:
        return None

    best = all_sequences[0]  # sorted ascending by total_wall_clock_s

    # 7. Combined completeness: sum of per-target median detection probabilities
    combined_completeness = sum(
        (target_median_completeness(planet_data, n) or 0.0)
        for n in target_names
    )
    efficiency = (
        combined_completeness / best["total_wall_clock_hrs"]
        if best["total_wall_clock_hrs"] > 0 else 0.0
    )

    return {
        "targets":               target_names,
        "refs":                  refs,
        "overlap_start":         overlap_start_iso,
        "overlap_end":           overlap_end_iso,
        "overlap_days":          overlap_days,
        "t_obs":                 t_obs,
        "obj_meta":              obj_meta,
        "obj_keys":              obj_keys,
        "visit_time":            visit_time,
        "slew_matrix":           slew_matrix,
        "all_sequences":         all_sequences,
        "best_sequence":         best,
        "best_wall_clock_hrs":   best["total_wall_clock_hrs"],
        "combined_completeness": combined_completeness,
        "efficiency":            efficiency,
    }


def plot_campaign_summary(combo_results: list, top_k: int = 20):
    """Two-figure campaign summary.

    Figure 1 — Sequence landscape (wall-clock vs completeness)
        Every sequence permutation for the top `top_k` combinations is plotted
        as a small dot, coloured by combination.  Since combined_completeness
        is fixed per combination (median detection prob is sequence-independent),
        points cluster horizontally — the spread along x shows how much visit
        ordering matters.  Large outlined dots mark each combination's best
        (fastest) sequence; the red step-line is the global Pareto front across
        those best sequences.

    Figure 2 — Pareto-front breakdown
        Only combinations that lie on the global Pareto front are shown.
        Left panel:  stacked horizontal bars (Science / Reference / Slew / OH)
                     for each combination's best sequence, sorted by
                     completeness ascending.  Science fraction is annotated.
        Right panel: completeness bar for the same ordering, so you can
                     directly compare time budget against science yield.
    """
    if not combo_results:
        print("plot_campaign_summary: no results to plot.")
        return

    # ── Figure 1: Campaign Pareto front ──────────────────────────────────────
    # One point per combination (best sequence), coloured by whether it is on
    # the Pareto front.  Non-Pareto points are grey; Pareto points are red and
    # annotated.  All combinations from combo_results are used so the front is
    # global, not limited to top_k.
    all_wc   = np.array([r["best_wall_clock_hrs"]   for r in combo_results])
    all_comp = np.array([r["combined_completeness"] for r in combo_results])
    pf_idx   = pareto_front_indices(all_wc, all_comp)
    pf_set   = set(pf_idx.tolist())

    fig1, ax1 = plt.subplots(figsize=(11, 7))

    # Non-Pareto combinations — grey
    non_pf = [i for i in range(len(combo_results)) if i not in pf_set]
    if non_pf:
        ax1.scatter(all_wc[non_pf], all_comp[non_pf],
                    color="tab:blue", alpha=0.45, s=28, zorder=3,
                    label="All combinations")

    # Pareto-front combinations — red, annotated
    if len(pf_idx) > 0:
        pf_wc   = all_wc[pf_idx]
        pf_comp = all_comp[pf_idx]
        so      = np.argsort(pf_wc)
        ax1.step(pf_wc[so], pf_comp[so], where="post",
                 color="red", linewidth=2.5, zorder=5)
        ax1.scatter(pf_wc, pf_comp, color="red", s=90, zorder=6,
                    edgecolors="darkred", linewidths=1.0,
                    label="Pareto front")
        for i in pf_idx:
            ax1.annotate(
                " + ".join(combo_results[i]["targets"]),
                (all_wc[i], all_comp[i]),
                fontsize=7, xytext=(6, 4), textcoords="offset points",
                zorder=7,
            )

    ax1.set_xlabel("Best wall-clock time (hrs)  [fastest sequence per combination]")
    ax1.set_ylabel("Combined completeness  [sum of per-target median detection prob]")
    ax1.set_title(
        f"Campaign Pareto front — {len(combo_results)} combinations\n"
        "Red = Pareto-optimal (min wall-clock for their completeness level)"
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show(block=False)

    # ── Figure 2: Pareto-front combination breakdown ──────────────────────────
    # Reuse all_wc / all_comp / pf_idx computed for Figure 1
    pf_combos = [combo_results[i] for i in pf_idx]
    # Sort by completeness ascending so bars read low → high completeness
    pf_combos = sorted(pf_combos, key=lambda r: r["combined_completeness"])

    if not pf_combos:
        return

    labels = [" + ".join(r["targets"]) for r in pf_combos]
    sci_h  = np.array([r["best_sequence"]["total_science_s"]  / 3600.0 for r in pf_combos])
    ref_h  = np.array([r["best_sequence"]["total_ref_s"]      / 3600.0 for r in pf_combos])
    slew_h = np.array([r["best_sequence"]["total_slew_s"]     / 3600.0 for r in pf_combos])
    oh_h   = np.array([
        (r["best_sequence"]["total_howfsc_s"]
         + r["best_sequence"].get("total_roll_s", 0.0)) / 3600.0
        for r in pf_combos
    ])
    comp   = np.array([r["combined_completeness"] for r in pf_combos])
    totals = sci_h + ref_h + slew_h + oh_h

    n = len(pf_combos)
    fig2, (ax_bar, ax_comp) = plt.subplots(
        1, 2,
        figsize=(14, max(4, 0.55 * n + 2.5)),
        gridspec_kw={"width_ratios": [3, 1]},
    )

    y      = np.arange(n)
    bar_kw = dict(height=0.6, edgecolor="white", linewidth=0.5)

    ax_bar.barh(y, sci_h,                                  color="#4c78a8", label="Science",          **bar_kw)
    ax_bar.barh(y, ref_h,  left=sci_h,                    color="#f58518", label="Reference",         **bar_kw)
    ax_bar.barh(y, slew_h, left=sci_h + ref_h,            color="#54a24b", label="Slew",              **bar_kw)
    ax_bar.barh(y, oh_h,   left=sci_h + ref_h + slew_h,   color="#b279a2", label="OH (HOWFSC+Roll)",  **bar_kw)

    for i, (tot, sci) in enumerate(zip(totals, sci_h)):
        sci_frac = 100.0 * sci / tot if tot > 0 else 0.0
        ax_bar.text(tot + 0.01 * totals.max(), i,
                    f"{tot:.2f} h  ({sci_frac:.0f}% sci)",
                    va="center", ha="left", fontsize=8)

    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(labels, fontsize=9)
    ax_bar.set_xlabel("Wall-clock time (hrs)")
    ax_bar.set_title("Wall-clock breakdown — Pareto-front combinations\n"
                     "(best sequence each; annotation shows science %)")
    ax_bar.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax_bar.set_xlim(0, totals.max() * 1.25)
    ax_bar.grid(axis="x", linestyle="--", alpha=0.4)

    ax_comp.barh(y, comp, height=0.6, color="#e45756", edgecolor="white")
    ax_comp.set_yticks(y)
    ax_comp.set_yticklabels([])
    ax_comp.set_xlabel("Combined completeness")
    ax_comp.set_title("Completeness")
    ax_comp.set_xlim(0, comp.max() * 1.3)
    ax_comp.grid(axis="x", linestyle="--", alpha=0.4)
    for i, c in enumerate(comp):
        ax_comp.text(c + 0.005 * comp.max(), i, f"{c:.4f}",
                     va="center", ha="left", fontsize=8)

    fig2.suptitle(
        "Pareto-front combinations: best sequence per combo\n"
        "Sorted by completeness — use this to pick the highest-science option",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    plt.show(block=False)


def print_campaign_summary(combo_results: list, top_k: int = 10):
    """Print a ranked table of the best target combinations.

    Sorted by efficiency (combined_completeness / best_wall_clock_hrs).
    Each row shows the total wall-clock broken down into:
      Sci (science integration), Ref (reference visits),
      Slew (slew overhead), OH (HOWFSC + roll-change overheads).
    """
    if not combo_results:
        print("print_campaign_summary: no results to display.")
        return

    subset = combo_results[:top_k]

    col_w = 38
    header = (
        f"{'Rank':>4}  {'Targets':<{col_w}}  "
        f"{'Total':>7}  {'Sci':>6}  {'Ref':>6}  {'Slew':>6}  {'OH':>6}  "
        f"{'Comp':>6}  {'combined_completeness / best_wall_clock_hrs':>9}  {'Days':>5}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}")
    print(f"CAMPAIGN SUMMARY  ({len(combo_results)} valid combinations, "
          f"showing top {len(subset)})")
    print(f"  All times in hours.  OH = HOWFSC + roll-change overhead.")
    print(sep)
    print(header)
    print("-" * len(header))

    for rank, r in enumerate(subset, start=1):
        tstr = " + ".join(r["targets"])
        bs   = r["best_sequence"]
        sci_h  = bs["total_science_s"]  / 3600.0
        ref_h  = bs["total_ref_s"]      / 3600.0
        slew_h = bs["total_slew_s"]     / 3600.0
        oh_h   = (bs["total_howfsc_s"] + bs.get("total_roll_s", 0.0)) / 3600.0
        total_h = sci_h + ref_h + slew_h + oh_h
        print(
            f"{rank:>4}  {tstr:<{col_w}}  "
            f"{total_h:>7.2f}  {sci_h:>6.2f}  {ref_h:>6.2f}  "
            f"{slew_h:>6.2f}  {oh_h:>6.2f}  "
            f"{r['combined_completeness']:>6.4f}  "
            f"{r['efficiency']:>9.5f}  "
            f"{r['overlap_days']:>5.0f}"
        )
        seq  = bs["sequence"]
        meta = r["obj_meta"]
        seq_names = " → ".join(meta[k]["name"] for k in seq)
        print(f"       {seq_names}")

    print(sep)


def run_campaign(
    start: str = ANALYSIS_START,
    days: float = ANALYSIS_DAYS,
    combo_size: int = 2,
    top_k: int = 10,
    top_k_pareto: int = 3,
    plot: bool = True,
) -> dict:
    """Run a full campaign over a given time window.

    For every unique target in HLC_B1 + SPEC + WFOV:
      1. Select a reference star over [start, start+days].
      2. Keep only targets that have at least one observable window with a
         valid reference star.
      3. Generate all C(valid_targets, combo_size) target combinations.
      4. For each combination: find the overlapping window, build the
         observation model, and score all visit-order permutations with
         score_all_sequences (fast, median-based).
      5. Rank combinations by efficiency = completeness / wall_clock_hrs.
      6. For the top top_k_pareto combinations, run the full time-resolved
         Pareto analysis and print / plot detailed results.

    Args:
        start        : ISO analysis start datetime string.
        days         : Analysis window duration in days.
        combo_size   : Science targets per combination (default 2).
        top_k        : Rows to display in the summary table.
        top_k_pareto : Combinations for full Pareto analysis (slow).
        plot         : Whether to produce plots.

    Returns:
        dict with keys: valid_targets, combo_results, ref_results,
        planet_data, target_coords.
    """
    n_obj_per_combo = combo_size * 2  # one ref star per science target
    n_perms = math.factorial(n_obj_per_combo)
    if n_perms > 40320:
        print(
            f"Warning: combo_size={combo_size} yields {n_obj_per_combo} objects "
            f"and {n_perms:,} permutations per combination. "
            "Consider using combo_size <= 3 for speed."
        )

    print(f"\n{'='*62}")
    print(f"CAMPAIGN  start={start[:10]}  days={days}  combo_size={combo_size}")
    print(f"{'='*62}")

    # ------------------------------------------------------------------
    # Step 1 — Load data, then restrict to targets with planet data files
    # ------------------------------------------------------------------
    print("\nLoading planet data and reference star catalog...")
    planet_data = load_planet_data()
    catalog     = load_catalog()

    targets_with_data = {k[0] for k in planet_data}

    # Build band/contrast lookup from program definitions, but only for
    # targets that actually have .npy files
    target_band_contrast = {}
    for program in (HLC_B1, SPEC, WFOV):
        for name, (_, band, contrast) in program.items():
            if name in targets_with_data and name not in target_band_contrast:
                target_band_contrast[name] = (band, contrast)

    # Targets in planet_data but not in any program get a default entry
    for name in targets_with_data:
        if name not in target_band_contrast:
            target_band_contrast[name] = (1, "high")

    n_candidates = len(target_band_contrast)
    print(f"{n_candidates} targets have planet data files.")

    print(f"Fetching coordinates for {n_candidates} targets...")
    target_coords = get_target_coords(list(target_band_contrast.keys()))

    # ------------------------------------------------------------------
    # Step 2 — Select reference stars only for planet-data targets
    # ------------------------------------------------------------------
    ref_results = {}
    print(f"\nSelecting reference stars for {n_candidates} targets...")
    for i, (name, (band, contrast)) in enumerate(target_band_contrast.items(), start=1):
        print(f"  [{i:>3}/{n_candidates}] {name}  (band={band}, contrast={contrast})")
        ref_results[name] = select_ref_star(
            sci_name=name,
            analysis_start=start,
            analysis_days=days,
            band=band,
            contrast=contrast,
            catalog=catalog,
        )

    # ------------------------------------------------------------------
    # Step 3 — Filter to targets with an observable window + valid ref star
    # ------------------------------------------------------------------
    valid_targets = [
        name for name, result in ref_results.items()
        if "observable_windows" in result
        and any(w.get("best_ref") is not None
                for w in result["observable_windows"])
    ]
    print(f"\n{len(valid_targets)} / {n_candidates} targets are observable with "
          "a valid reference star in this window.")

    if len(valid_targets) < combo_size:
        print(f"Not enough valid targets ({len(valid_targets)}) for "
              f"combo_size={combo_size}. Returning early.")
        return {
            "valid_targets": valid_targets,
            "combo_results": [],
            "ref_results":   ref_results,
            "planet_data":   planet_data,
            "target_coords": target_coords,
        }

    # ------------------------------------------------------------------
    # Step 4 — Score all combinations
    # ------------------------------------------------------------------
    all_combos = list(iter_combinations(valid_targets, combo_size))
    print(f"\nScoring {len(all_combos)} combinations "
          f"(combo_size={combo_size}, {n_perms} permutations each)...")

    combo_results = []
    n_skipped = 0
    for idx, targets in enumerate(all_combos, start=1):
        result = score_combination(
            list(targets), ref_results, planet_data, target_coords, ADI_PARAMS
        )
        if result is None:
            n_skipped += 1
        else:
            combo_results.append(result)
        if idx % 25 == 0 or idx == len(all_combos):
            print(f"  [{idx:>5}/{len(all_combos)}]  "
                  f"{len(combo_results)} valid  |  {n_skipped} skipped")

    print(f"\n{len(combo_results)} / {len(all_combos)} combinations have "
          "overlapping observable windows.")

    if not combo_results:
        print("No schedulable combinations found.")
        return {
            "valid_targets": valid_targets,
            "combo_results": [],
            "ref_results":   ref_results,
            "planet_data":   planet_data,
            "target_coords": target_coords,
        }

    # ------------------------------------------------------------------
    # Step 5 — Rank and summarise
    # ------------------------------------------------------------------
    combo_results.sort(key=lambda r: r["efficiency"], reverse=True)

    print_campaign_summary(combo_results, top_k=top_k)
    if plot:
        plot_campaign_summary(combo_results)

    # ------------------------------------------------------------------
    # Step 6 — Full Pareto analysis for top combinations
    # ------------------------------------------------------------------
    for rank, result in enumerate(combo_results[:top_k_pareto], start=1):
        tnames = result["targets"]
        print(f"\n{'─'*60}")
        print(f"Pareto analysis [{rank}/{top_k_pareto}]: {' + '.join(tnames)}")
        print(f"{'─'*60}")

        t_obs    = result["t_obs"]
        obj_keys = result["obj_keys"]
        obj_meta = result["obj_meta"]

        slew_matrix_ts = build_time_resolved_slew_matrix(
            obj_keys, obj_meta, target_coords, t_obs
        )
        pareto_results = score_sequences_pareto(
            t_obs, obj_keys, obj_meta, slew_matrix_ts, ADI_PARAMS, planet_data
        )

        # Diagnostic: show how many observable dates exist for this combination
        if pareto_results:
            ko = pareto_results[0]["any_keepout"]
            wc = pareto_results[0]["wall_clock_hrs"]
            n_obs = int(np.sum(~ko & np.isfinite(wc)))
            print(f"  t_obs range: {t_obs.decimalyear.min():.4f} – {t_obs.decimalyear.max():.4f}  "
                  f"({len(t_obs)} days)   observable: {n_obs}")
            for k in [k for k in obj_keys if obj_meta[k]["type"] == "science"]:
                name = obj_meta[k]["name"]
                t_int, _, ko_k = interpolate_target_data(planet_data, name, t_obs.decimalyear)
                print(f"    {name}: finite t_int={np.isfinite(t_int).sum()}/{len(t_int)}  "
                      f"in_keepout={ko_k.sum()}/{len(ko_k)}")

        if plot:
            sci_names = [obj_meta[k]["name"]
                         for k in obj_keys if obj_meta[k]["type"] == "science"]
            plot_pareto(
                sci_names[0],
                " + ".join(sci_names[1:]) if len(sci_names) > 1 else sci_names[0],
                pareto_results, t_obs, obj_meta, planet_data,
            )

        print_optimal_sequence(
            result["all_sequences"], pareto_results, obj_meta, ADI_PARAMS, t_obs
        )
        print_sequence_time_table(result["all_sequences"], obj_meta)
        if plot:
            plot_sequence_window(
                pareto_results, t_obs, obj_meta, planet_data,
                title=" + ".join(tnames),
            )

    if plot:
        plt.pause(0.001)  # keep all non-blocking figures alive until user closes them
        input("Press Enter to close all plots and exit...")
        plt.close("all")

    return {
        "valid_targets": valid_targets,
        "combo_results": combo_results,
        "ref_results":   ref_results,
        "planet_data":   planet_data,
        "target_coords": target_coords,
    }


def plot_campaign(targ1_name: str, targ2_name: str, planet_data: dict, campaign: dict):
    adi = ADI_PARAMS
    # Fixed overhead per science target: HOWFSC + roll changes
    per_sci_overhead_hrs = (
        adi["t_howfsc_s"] + adi["N_rollchange"] * adi["t_rollchange_s"]
    ) / 3600.0
    median_slew_hrs = float(np.median(campaign["wall_clock"]["slew_time_s"])) / 3600.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["tab:blue", "tab:orange"]

    for target_name, color in zip([targ1_name, targ2_name], colors):
        planet_keys = [k for k in PLANET_DATA_FILES if k[0] == target_name]
        for key in planet_keys:
            _, planet, band = key
            label = f"{target_name} {planet} band{band}"

            t_yr, t_int_hrs = get_integration_time(planet_data, key, mode="con")
            all_t, prob_all = get_completeness(planet_data, key)
            obs_mask        = np.isin(all_t, t_yr)
            prob_obs        = prob_all[obs_mask]

            # T_wallclock = T_int*(1 + (3*R)/(2*N)) + N_slew*T_slew + N_rollchange*T_rollchange + T_HOWFSC
            t_int_s = t_int_hrs * 3600.0
            wall_clock_hrs = (
                t_int_s * (1.0 + (3.0 * adi["R"]) / (2.0 * adi["N"]))
                + adi["N_slew"] * median_slew_hrs * 3600.0
                + adi["N_rollchange"] * adi["t_rollchange_s"]
                + adi["t_howfsc_s"]
            ) / 3600.0

            ax1.scatter(t_int_hrs, prob_obs,       label=label, color=color, alpha=0.7, s=20)
            ax2.scatter(t_int_hrs, wall_clock_hrs, label=label, color=color, alpha=0.7, s=20)

    for ax, ylabel, title in [
        (ax1, "Detection Probability",       "Integration Time vs Completeness"),
        (ax2, "Total Wall Clock Time (hrs)", "Integration Time vs Wall Clock Time"),
    ]:
        ax.set_xlabel("Integration Time (hours)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(f"Campaign: {targ1_name}  +  {targ2_name}", fontsize=13)
    plt.tight_layout()
    plt.show(block=False)


def main():
    rng = np.random.default_rng()

    # --- Step 1: Load all data ---

    all_targets = list({*HLC_B1.keys(), *SPEC.keys(), *WFOV.keys()})

    # Fetch coords once upfront; the cache in roman_observability means subsequent
    # calls from select_ref_star and get_target_availability_windows are free.
    target_coords = get_target_coords(all_targets)

    target_windows = get_target_availability_windows(
        target_names=all_targets,
        start_date=ANALYSIS_START,
        days=ANALYSIS_DAYS,
        time_step=TIME_STEP_DAYS,
        coords=target_coords,
    )

    catalog = load_catalog()
    ref_star_results = {}
    for program in (HLC_B1, SPEC, WFOV):
        for target_name, (priority, band, contrast) in program.items():
            if target_name not in ref_star_results:  # avoid duplicate lookups
                ref_star_results[target_name] = select_ref_star(
                    sci_name=target_name,
                    analysis_start=ANALYSIS_START,
                    analysis_days=ANALYSIS_DAYS,
                    band=band,
                    contrast=contrast,
                    catalog=catalog,
                )

    planet_data = load_planet_data()

    # Build slew results for all target pairs (one call per pair)
    t0      = Time(ANALYSIS_START, format="isot", scale="utc")
    ts_slew = t0 + np.arange(ANALYSIS_DAYS) * u.d

    target_list  = list(target_coords.keys())
    slew_results = {}
    for i, targ1 in enumerate(target_list):
        for targ2 in target_list[i + 1:]:
            _, slew_time_s = compute_slew_time(
                target_coords[targ1], target_coords[targ2], ts_slew
            )
            slew_results[(targ1, targ2)] = {
                "slew_angle_deg": _,       # kept for completeness
                "slew_time_s":    slew_time_s,
            }

    # --- Step 2: Pick a valid target pair (with retry) ---

    valid_targets = [
        t for t, result in ref_star_results.items()
        if "observable_windows" in result
        and any(w.get("best_ref") is not None for w in result["observable_windows"])
    ]

    campaign = None
    for attempt in range(MAX_RETRIES):
        targ1_name, targ2_name = rng.choice(valid_targets, size=2, replace=False)
        w1, w2 = find_overlapping_windows(
            ref_star_results[targ1_name]["observable_windows"],
            ref_star_results[targ2_name]["observable_windows"],
        )
        if w1 is not None:
            campaign = {
                "targ1": targ1_name,
                "targ2": targ2_name,
                "ref1":  w1["best_ref"]["reference_star"],
                "ref2":  w2["best_ref"]["reference_star"],
                "targ1_window": {"start": w1["start"], "end": w1["end"],
                                 "duration_days": w1["duration_days"]},
                "targ2_window": {"start": w2["start"], "end": w2["end"],
                                 "duration_days": w2["duration_days"]},
                "slew": get_or_reverse_slew(slew_results, targ1_name, targ2_name),
            }
            break
        print(f"Attempt {attempt + 1}: no overlapping window for "
              f"{targ1_name} / {targ2_name}, retrying…")
    else:
        raise RuntimeError(
            f"Could not find a valid target pair after {MAX_RETRIES} attempts."
        )

    targ1_name = campaign["targ1"]
    targ2_name = campaign["targ2"]

    if campaign["slew"] is None:
        raise RuntimeError(
            f"Slew data missing for pair ({targ1_name}, {targ2_name}). "
            "This should not happen — check slew_results construction."
        )

    # --- Step 3: Wall clock ---

    overlap_start, overlap_end, t_obs = overlap_time_array(
        campaign["targ1_window"], campaign["targ2_window"]
    )

    targ1_planet_keys = [k for k in planet_data if k[0] == targ1_name]
    if targ1_planet_keys:
        _, t_int_hrs       = get_integration_time(planet_data, targ1_planet_keys[0])
        time_science_visit = float(np.nanmedian(t_int_hrs)) * 3600
    else:
        time_science_visit = 3600.0

    wall_clock_s, slew_time_s = calc_wall_clock_time(
        target_coords[targ1_name], target_coords[targ2_name], t_obs,
        t_int_s=time_science_visit,
        adi=ADI_PARAMS,
    )

    campaign["wall_clock"] = {
        "t_obs":                t_obs,
        "wall_clock_s":         wall_clock_s,
        "wall_clock_hrs":       wall_clock_s / 3600.0,
        "slew_time_s":          slew_time_s,
        "time_science_visit_s": time_science_visit,
    }

    # --- Step 4: Sequence scoring ---

    obj_meta = {
        "targ1": {"name": targ1_name,        "type": "science"},
        "ref1":  {"name": campaign["ref1"],  "type": "reference", "paired_with": "targ1"},
        "targ2": {"name": targ2_name,        "type": "science"},
        "ref2":  {"name": campaign["ref2"],  "type": "reference", "paired_with": "targ2"},
    }
    visit_time = {
        "targ1": time_science_visit,
        "ref1":  ref_visit_time_s(time_science_visit, ADI_PARAMS),
        "targ2": time_science_visit,
        "ref2":  ref_visit_time_s(time_science_visit, ADI_PARAMS),
    }

    obj_keys   = list(obj_meta.keys())
    t_mid      = Time((overlap_start.mjd + overlap_end.mjd).item() / 2, format="mjd", scale="utc")

    # Fetch coords for all 4 objects in one call (reuses target_coords where possible)
    new_names  = [n for n in [campaign["ref1"], campaign["ref2"]]
                  if n not in target_coords]
    if new_names:
        target_coords.update(get_target_coords(new_names))

    slew_matrix = build_slew_matrix(obj_keys, obj_meta, target_coords, t_mid)

    sequence_results = score_all_sequences(
        obj_keys, obj_meta, visit_time, slew_matrix, ADI_PARAMS, planet_data
    )

    # --- Step 5: Time-resolved Pareto scoring & plot ---

    slew_matrix_ts = build_time_resolved_slew_matrix(
        obj_keys, obj_meta, target_coords, t_obs
    )
    pareto_results = score_sequences_pareto(
        t_obs, obj_keys, obj_meta, slew_matrix_ts, ADI_PARAMS, planet_data
    )
    plot_pareto(targ1_name, targ2_name, pareto_results, t_obs, obj_meta, planet_data)
    print_optimal_sequence(sequence_results, pareto_results, obj_meta, ADI_PARAMS, t_obs)

    print_sequence_time_table(sequence_results, obj_meta)

    return campaign, sequence_results, pareto_results


def toy_campaign(
    targ1_name: str = "* eps Eri",
    targ2_name: str = "ups And",
    ref1_name:  str = "* bet Leo",
    ref2_name:  str = "* tau Cet",
):
    """Run a quick end-to-end campaign with hardcoded targets — no SIMBAD target
    search loop and no select_ref_star calls, so it finishes in seconds.

    Useful for verifying that the pipeline mechanics work before committing to
    a full run.
    """
    print(f"=== TOY CAMPAIGN: {targ1_name} + {targ2_name} ===\n")

    planet_data   = load_planet_data()
    target_coords = get_target_coords([targ1_name, targ2_name, ref1_name, ref2_name])

    t0     = Time(ANALYSIS_START, format="isot", scale="utc")
    window = {
        "start":         ANALYSIS_START[:10],
        "end":           (t0 + ANALYSIS_DAYS * u.d).iso[:10],
        "duration_days": float(ANALYSIS_DAYS),
    }

    overlap_start, overlap_end, t_obs = overlap_time_array(window, window)

    obj_meta = {
        "targ1": {"name": targ1_name, "type": "science"},
        "ref1":  {"name": ref1_name,  "type": "reference", "paired_with": "targ1"},
        "targ2": {"name": targ2_name, "type": "science"},
        "ref2":  {"name": ref2_name,  "type": "reference", "paired_with": "targ2"},
    }
    obj_keys = list(obj_meta.keys())

    # Median-based summary (for the printed table)
    targ1_planet_keys = [k for k in planet_data if k[0] == targ1_name]
    if targ1_planet_keys:
        _, t_int_hrs       = get_integration_time(planet_data, targ1_planet_keys[0])
        time_science_visit = float(np.nanmedian(t_int_hrs)) * 3600
    else:
        time_science_visit = 3600.0

    visit_time = {
        "targ1": time_science_visit,
        "ref1":  ref_visit_time_s(time_science_visit, ADI_PARAMS),
        "targ2": time_science_visit,
        "ref2":  ref_visit_time_s(time_science_visit, ADI_PARAMS),
    }
    t_mid        = Time((overlap_start.mjd + overlap_end.mjd).item() / 2, format="mjd", scale="utc")
    slew_matrix  = build_slew_matrix(obj_keys, obj_meta, target_coords, t_mid)
    sequence_results = score_all_sequences(
        obj_keys, obj_meta, visit_time, slew_matrix, ADI_PARAMS, planet_data
    )

    # Time-resolved Pareto analysis
    slew_matrix_ts = build_time_resolved_slew_matrix(
        obj_keys, obj_meta, target_coords, t_obs
    )
    pareto_results = score_sequences_pareto(
        t_obs, obj_keys, obj_meta, slew_matrix_ts, ADI_PARAMS, planet_data
    )
    plot_pareto(targ1_name, targ2_name, pareto_results, t_obs, obj_meta, planet_data)
    print_optimal_sequence(sequence_results, pareto_results, obj_meta, ADI_PARAMS, t_obs)

    print_sequence_time_table(sequence_results, obj_meta)

    return sequence_results, pareto_results


if __name__ == "__main__":
    campaign_output = run_campaign()
