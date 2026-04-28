"""
Mission Planning Optimization Tool
====================================
Roman Coronagraph Instrument (CGI) observation scheduling.

Wall-clock model (from corgietc notebook 07_Reference_Star_Settings):

    T_wall = (1 + R_i) × T_science_i              per science target i
           + T_slew_inter                          inter-target slews between science targets
           + Σ_i  N_slew × T_slew_sci_to_ref[i]   intra-ADI sci↔ref transitions
           + n_science × N_rollchange × T_rollchange
           + n_science × T_HOWFSC

KEY DESIGN DECISIONS
---------------------
1. R (tRef_per_tTar) is PER SCIENCE TARGET, not a global constant.
   - R is passed into corgietc's calc_intTime() — it affects T_science
     through the noise model (ref star noise contributes to total noise).
   - T_ref = R × T_science is then added on top for actual ref star pointing time.
   - Different science targets may have different R values depending on their
     reference star brightness and separation.
   - R is stored per science target in tint_s[sci_key]["R"].

2. Reference stars are NOT schedulable objects.
   - They do not appear in the visit-order permutation.
   - Their time (R × T_science) and intra-ADI slew cost are computed directly
     from the science target's data.

3. T_science comes from compute_inttime_with_ref_star() which re-runs
   calc_intTime() with the actual reference star V magnitude set in
   mode["RefStar_V_mag"]. This corrects T_science for the true noise penalty.
   The wall-clock is then (1 + R) × T_science_corrected.

4. The .npy planet data files contain T_science pre-computed with R=0.25
   as a default. When compute_inttime_with_ref_star() is available, its
   corrected T_science replaces the .npy value. Otherwise the .npy value
   is used with a fallback R from ADI_PARAMS["R_default"].

Slew model (from roman_pointing SlewTime notebook):
    Slew angle = max(|Δpitch|, |Δyaw|) in observatory frame.
    Slew time interpolated from GSFC SlewSettle.ecsv table.
    compute_slew_time() in roman_pointing.utils handles this correctly.
"""

from __future__ import annotations

import math
from itertools import permutations, combinations as iter_combinations, product as cart_product
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import astropy.units as u
from astropy.time import Time
from sqlalchemy import text

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

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

ANALYSIS_START       = "2027-01-01T00:00:00"
ANALYSIS_DAYS        = 365
TIME_STEP_DAYS       = 1
MAX_RETRIES          = 20
MISSION_BUDGET_DAYS  = 60.0

# ADI observation parameters (standard Roll A-B-A-B sequence, N=4).
#
_R_FALLBACK = 0.25   # used only when optimise_R has not been run for a target

ADI_PARAMS = {
    "N":                4,
    "R_default":        _R_FALLBACK,
    "N_slew":           4,      # intra-ADI sci↔ref transitions per science target
    "N_rollchange":     6,      # 2*(N-1) = 6 for N=4
    "t_rollchange_s":   600,    # 10 min per roll change
    "t_howfsc_s":       900,    # HOWFSC overhead per science target
    "t_slew_default_s": 1800,   # fallback slew time (90-deg separation assumed)
}


# ===========================================================================
# SECTION 1 — TINT_S DATA STRUCTURE
# ===========================================================================
#
# tint_s is a dict keyed by sci_key. Each value is a dict:
#   {
#     "t_science_s" : float   — T_science in seconds (corrected for actual ref star)
#     "R"           : float   — tRef_per_tTar used in corgietc for this target
#     "ref_star"    : str     — name of the reference star used
#     "ref_star_vmag": float  — V mag of the reference star
#     "source"      : str     — "corgietc" | "npy_fallback"
#   }
#
# Wall-clock for target i = (1 + R_i) × T_science_i + slew terms + overheads
# ===========================================================================


def make_tint_entry(
    t_science_s: float,
    R: float,
    ref_star: str | None = None,
    ref_star_vmag: float = 2.26,
    source: str = "npy_fallback",
) -> dict:
    """Create a standardised tint_s entry for one science target."""
    return {
        "t_science_s":   t_science_s,
        "R":             R,
        "ref_star":      ref_star,
        "ref_star_vmag": ref_star_vmag,
        "source":        source,
    }


# ===========================================================================
# SECTION 2 — WALL-CLOCK FORMULA (single source of truth)
# ===========================================================================

def compute_wall_clock_s(
    tint_s: dict[str, dict],
    slew_inter_s: float | np.ndarray,
    intra_slew: dict[str, float | np.ndarray],
    sci_keys: list[str],
    adi: dict,
) -> float | np.ndarray:
    """
    Authoritative wall-clock formula used by every code path.

    T_wall = Σ_i (1 + R_i) × T_science_i          (sci + actual ref pointing)
           + T_slew_inter                           (inter-target slews)
           + Σ_i N_slew × T_slew_sci_to_ref_i      (intra-ADI transitions)
           + n_science × N_rollchange × T_rollchange
           + n_science × T_HOWFSC

    R_i comes from tint_s[sci_key]["R"] — the per-target value from corgietc.
    This means different science targets can have different R values.

    Args:
        tint_s       : {sci_key: {"t_science_s": float, "R": float, ...}}
        slew_inter_s : total inter-target slew time for this permutation (s)
        intra_slew   : {sci_key: slew time to paired ref star (s)}
        sci_keys     : list of science target keys
        adi          : ADI_PARAMS dict

    Returns:
        Wall-clock time in seconds (scalar or array matching t_obs shape).
    """
    n_science = len(sci_keys)

    # Science + reference time: Σ (1 + R_i) × T_science_i
    sci_and_ref_s = sum(
        (1.0 + tint_s[k]["R"]) * tint_s[k]["t_science_s"]
        for k in sci_keys
    )

    # Intra-ADI slews: N_slew × T_slew_sci_to_ref per science target
    intra_slew_s = sum(
        adi["N_slew"] * intra_slew[k]
        for k in sci_keys
    )

    # Fixed overheads per science target
    overhead_s = n_science * (
        adi["N_rollchange"] * adi["t_rollchange_s"]
        + adi["t_howfsc_s"]
    )

    return sci_and_ref_s + slew_inter_s + intra_slew_s + overhead_s


# ===========================================================================
# SECTION 3 — REFERENCE STAR UTILITIES
# ===========================================================================

def get_ref_star_vmag(ref_star_name: str) -> float:
    """Fetch V magnitude of a reference star from SIMBAD. Returns 2.26 on failure."""
    from astroquery.simbad import Simbad
    try:
        simbad = Simbad()
        simbad.add_votable_fields("V")
        result = simbad.query_object(ref_star_name)
        if result is not None and len(result) > 0:
            val = result["V"][0]
            if not hasattr(val, "mask") and val is not None:
                return float(val)
    except Exception:
        pass
    return 2.26


def compute_inttime_with_ref_star(
    pl_name: str,
    band: int,
    engine,
    ref_star_name: str | None,
    R: float,
    months_at_l2: float = 21,
    ref_star_spectype: str | None = None,
) -> dict:
    """
    Compute T_science using the actual reference star and the specified R value.

    Runs calc_intTime() with:
        mode["RefStar_V_mag"]               = actual ref star V mag
        mode["TimeonRefStar_tRef_per_tTar"] = R   ← this is the key parameter

    T_science returned is corrected for the true noise penalty of the actual
    reference star at this R value. The wall-clock is (1 + R) × T_science.

    Args:
        pl_name       : planet name in Orbits table (e.g. '47 UMa b')
        band          : instrument band — 1, 3, or 4
        engine        : SQLAlchemy engine connected to plandb
        ref_star_name : SIMBAD-resolvable name of the reference star, or None
        R             : tRef_per_tTar value to use in corgietc
        months_at_l2  : mission elapsed time for detector degradation

    Returns:
        dict with keys:
            t              : decimal year array
            WA_arcsec      : working angle array
            dMag           : delta-magnitude array
            intTime_hours  : T_science per epoch (NaN outside IWA/OWA or keepout)
            t_wall_hours   : (1 + R) × intTime_hours
            ref_star       : reference star name used
            ref_star_vmag  : V magnitude used
            R              : tRef_per_tTar used
            source         : "corgietc"
    """
    import os, json, copy, shutil
    import EXOSIMS.Prototypes.TargetList
    import EXOSIMS.Prototypes.TimeKeeping
    import pandas as pd

    ref_star_vmag = get_ref_star_vmag(ref_star_name) if ref_star_name else 2.26
    if ref_star_spectype is not None:
        ref_star_spectype = _normalize_spectype(ref_star_spectype)
    else:
        ref_star_spectype = "a0v"

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
    vmag_ref_default = float(TL.Vmag[0])

    band_wavelength = {1: 575, 3: 730, 4: 825}
    target_nm = band_wavelength[band]
    mode = next(
        (m for m in OS.observingModes if abs(m["lam"].to_value(u.nm) - target_nm) < 20),
        OS.observingModes[0],
    )

    # ── The two parameters that matter for T_science ──────────────────────────
    mode["RefStar_V_mag"]               = ref_star_vmag      # actual ref star brightness
    mode["RefStar_SpectralType"]        = ref_star_spectype
    mode["TimeonRefStar_tRef_per_tTar"] = R                   # per-target R value
    # ─────────────────────────────────────────────────────────────────────────

    TK = EXOSIMS.Prototypes.TimeKeeping.TimeKeeping(missionLife=5.25)
    TK.allocate_time(months_at_l2 * 30.4375 * u.d)
    fZ = np.repeat(TL.ZodiacalLight.fZ0, 1)

    wav      = band_wavelength[band]
    dmag_col = f"dMag_med_{wav}NM"
    query = text(f"""
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
        raise ValueError(f"No orbit data for '{pl_name}' band {band}")

    sy_vmag    = float(df["sy_vmag"].iloc[0])
    flux_scale = 10.0 ** ((vmag_ref_default - sy_vmag) / 2.5)

    original_fluxes = {}
    for m in OS.observingModes:
        hk = m["hex"]
        if hk in TL.star_fluxes:
            original_fluxes[hk] = TL.star_fluxes[hk].copy()
            TL.star_fluxes[hk]  = original_fluxes[hk] * flux_scale
    original_vmag = TL.Vmag.copy()
    TL.Vmag[0]   = sy_vmag

    intTimes_hours = np.full(len(df), np.nan)
    for i, row in df.iterrows():
        WA   = np.array([row["WA"]]) * u.arcsec
        dMag = np.array([row["dMag"]])
        if np.isnan(row["dMag"]) or np.isnan(row["WA"]):
            continue
        if (WA[0] < mode["IWA"]) or (WA[0] > mode["OWA"]):
            continue
        JEZ  = TL.JEZ0[mode["hex"]] / (row["WA"] ** 2)
        t_hr = OS.calc_intTime(TL, [0], fZ, JEZ, dMag, WA, mode, TK=TK)[0].to(u.hour).value
        if not np.isnan(t_hr) and t_hr > 0:
            intTimes_hours[i] = t_hr

    for hk, orig in original_fluxes.items():
        TL.star_fluxes[hk] = orig
    TL.Vmag[0] = original_vmag[0]

    return {
        "t":             df["t"].values,
        "WA_arcsec":     df["WA"].values,
        "dMag":          df["dMag"].values,
        "intTime_hours": intTimes_hours,
        "t_wall_hours":  (1.0 + R) * intTimes_hours,
        "ref_star":      ref_star_name,
        "ref_star_vmag": ref_star_vmag,
        "R":             R,
        "source":        "corgietc",
    }


def compare_ref_star_impact(
    pl_name: str,
    band: int,
    engine,
    ref_star_name: str | None,
    sci_name: str,
    R: float = None,
) -> None:
    """
    Show how the actual reference star and R value change T_science vs defaults.

    Runs corgietc twice:
        1. Default: V=2.26, R=ADI_PARAMS["R_default"]
        2. Actual : V=from SIMBAD, R=provided R (or R_default if None)

    Prints T_science and T_wall = (1+R) × T_science for both.
    """
    R_actual  = R if R is not None else ADI_PARAMS["R_default"]
    R_default = ADI_PARAMS["R_default"]
    actual_vmag = get_ref_star_vmag(ref_star_name) if ref_star_name else 2.26

    print(f"\n{'='*65}")
    print(f"Ref star impact: {pl_name}  band={band}")
    print(f"  Science target : {sci_name}")
    print(f"  Ref star       : {ref_star_name or 'none'}")
    print(f"  Default vmag   : 2.26  (corgietc default)   R_default={R_default}")
    print(f"  Actual  vmag   : {actual_vmag:.2f}  (from SIMBAD)      R_actual={R_actual}")
    print(f"{'='*65}")

    import os, json, copy, shutil
    import EXOSIMS.Prototypes.TargetList
    import EXOSIMS.Prototypes.TimeKeeping
    import pandas as pd

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
    vmag_ref_default = float(TL.Vmag[0])

    band_wavelength = {1: 575, 3: 730, 4: 825}
    mode = next(
        (m for m in OS.observingModes
         if abs(m["lam"].to_value(u.nm) - band_wavelength[band]) < 20),
        OS.observingModes[0],
    )

    TK = EXOSIMS.Prototypes.TimeKeeping.TimeKeeping(missionLife=5.25)
    TK.allocate_time(21 * 30.4375 * u.d)
    fZ = np.repeat(TL.ZodiacalLight.fZ0, 1)

    wav      = band_wavelength[band]
    dmag_col = f"dMag_med_{wav}NM"
    query = text(f"""
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
    flux_scale = 10.0 ** ((vmag_ref_default - sy_vmag) / 2.5)
    original_fluxes = {}
    for m in OS.observingModes:
        hk = m["hex"]
        if hk in TL.star_fluxes:
            original_fluxes[hk] = TL.star_fluxes[hk].copy()
            TL.star_fluxes[hk]  = original_fluxes[hk] * flux_scale
    original_vmag = TL.Vmag.copy()
    TL.Vmag[0]   = sy_vmag

    def _run(ref_vmag: float, R_val: float) -> np.ndarray:
        mode["RefStar_V_mag"]               = ref_vmag
        mode["TimeonRefStar_tRef_per_tTar"] = R_val
        results = []
        for _, row in df.iterrows():
            WA   = np.array([row["WA"]]) * u.arcsec
            dMag = np.array([row["dMag"]])
            if np.isnan(row["dMag"]) or np.isnan(row["WA"]):
                results.append(np.nan); continue
            if (WA[0] < mode["IWA"]) or (WA[0] > mode["OWA"]):
                results.append(np.nan); continue
            JEZ  = TL.JEZ0[mode["hex"]] / (row["WA"] ** 2)
            t_hr = OS.calc_intTime(TL, [0], fZ, JEZ, dMag, WA, mode, TK=TK)[0].to(u.hour).value
            results.append(t_hr if (not np.isnan(t_hr) and t_hr > 0) else np.nan)
        return np.array(results)

    t_default = _run(2.26, R_default)
    t_actual  = _run(actual_vmag, R_actual)

    for hk, orig in original_fluxes.items():
        TL.star_fluxes[hk] = orig
    TL.Vmag[0] = original_vmag[0]

    valid = np.isfinite(t_default) & np.isfinite(t_actual)
    if not np.any(valid):
        print("  No observable epochs to compare.")
        return

    med_sci_def  = np.nanmedian(t_default[valid])
    med_sci_act  = np.nanmedian(t_actual[valid])
    med_wall_def = (1 + R_default) * med_sci_def
    med_wall_act = (1 + R_actual)  * med_sci_act
    pct_sci      = 100 * (med_sci_act  - med_sci_def)  / med_sci_def
    pct_wall     = 100 * (med_wall_act - med_wall_def) / med_wall_def

    print(f"  Observable epochs                   : {valid.sum()}")
    print(f"  --- T_science (from calc_intTime) ---")
    print(f"  Median T_sci  default (V=2.26, R={R_default}) : {med_sci_def:.3f} hrs")
    print(f"  Median T_sci  actual  (V={actual_vmag:.2f}, R={R_actual}) : {med_sci_act:.3f} hrs")
    print(f"  T_sci change                        : {pct_sci:+.1f}%")
    print(f"  --- T_wall = (1 + R) × T_science ---")
    print(f"  Median T_wall default               : {med_wall_def:.3f} hrs")
    print(f"  Median T_wall actual                : {med_wall_act:.3f} hrs")
    print(f"  T_wall change                       : {pct_wall:+.1f}%")
    print(f"{'='*65}\n")


# ===========================================================================
# SECTION 4 — PLANET / TARGET DATA UTILITIES
# ===========================================================================

def load_planet_data() -> dict:
    """Load all .npy planet detection files keyed by (target, planet, band)."""
    return {
        key: np.load(path, allow_pickle=True).item()
        for key, path in PLANET_DATA_FILES.items()
    }


def get_integration_time(planet_data: dict, key: tuple, mode: str = "con"):
    """Return (decimal_year, t_int_hours) for observable (non-keepout) epochs."""
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
    """Median detection probability across all planet/band keys for a target."""
    keys = [k for k in planet_data if k[0] == target_name]
    if not keys:
        return None
    _, prob = get_completeness(planet_data, keys[0])
    finite  = prob[np.isfinite(prob)]
    return float(np.median(finite)) if len(finite) > 0 else 0.0


def get_target_tint_entry(
    planet_data: dict,
    target_name: str,
    ref_star_name: str | None = None,
    ref_star_spectype: str | None = None,
    R: float = None,
    band: int = 1,
    engine=None,
    pl_name: str | None = None,
) -> dict:
    """
    Build a tint_s entry for one science target.

    Priority:
        1. If engine and pl_name provided → call compute_inttime_with_ref_star()
           with the actual R and ref star. Uses corrected T_science.
        2. Otherwise → use median from .npy files with R_default as fallback.

    Args:
        planet_data   : loaded planet data dict
        target_name   : science target name
        ref_star_name : reference star name (for corgietc run)
        R             : tRef_per_tTar to use. If None, uses ADI_PARAMS["R_default"]
        band          : instrument band (1, 3, or 4)
        engine        : SQLAlchemy engine (optional, needed for corgietc)
        pl_name       : planet name in DB (optional, needed for corgietc)

    Returns:
        tint_s entry dict: {t_science_s, R, ref_star, ref_star_vmag, source}
    """
    R_val = R if R is not None else ADI_PARAMS["R_default"]

    # Path 1: corgietc with actual ref star and R
    if engine is not None and pl_name is not None:
        try:
            result = compute_inttime_with_ref_star(
                pl_name=pl_name,
                band=band,
                engine=engine,
                ref_star_name=ref_star_name,
                R=R_val,
                ref_star_spectype=ref_star_spectype,
            )
            # Use median of finite integration times as scalar representative
            finite = result["intTime_hours"][np.isfinite(result["intTime_hours"])]
            t_sci_s = float(np.median(finite)) * 3600.0 if len(finite) > 0 else 3600.0
            return make_tint_entry(
                t_science_s   = t_sci_s,
                R             = R_val,
                ref_star      = ref_star_name,
                ref_star_vmag = result["ref_star_vmag"],
                source        = "corgietc",
            )
        except Exception as e:
            print(f"  Warning: corgietc failed for {target_name}: {e}. "
                  f"Falling back to .npy data with R={R_val}.")

    # Path 2: .npy fallback
    keys = [k for k in planet_data if k[0] == target_name]
    all_hrs: list[float] = []
    for key in keys:
        _, t_int_hrs = get_integration_time(planet_data, key, mode="con")
        all_hrs.extend(t_int_hrs[np.isfinite(t_int_hrs)].tolist())

    t_sci_s = float(np.median(all_hrs)) * 3600.0 if all_hrs else 3600.0
    ref_vmag = get_ref_star_vmag(ref_star_name) if ref_star_name else 2.26

    return make_tint_entry(
        t_science_s   = t_sci_s,
        R             = R_val,
        ref_star      = ref_star_name,
        ref_star_vmag = ref_vmag,
        source        = "npy_fallback",
    )


def interpolate_target_data(
    planet_data: dict,
    target_name: str,
    t_obs_yr: np.ndarray,
    R: float = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate T_science, T_wall, and completeness at requested dates.

    Args:
        planet_data : planet data dict
        target_name : science target name
        t_obs_yr    : decimal year array
        R           : tRef_per_tTar for this target. If None, uses R_default.

    Returns:
        t_sci_hrs   : T_science (hrs) — NaN in keepout / outside data range
        t_wall_hrs  : (1 + R) × T_science (hrs)
        completeness: detection probability summed across all planet/band keys
        in_keepout  : True where T_science is NaN
    """
    R_val = R if R is not None else ADI_PARAMS["R_default"]

    keys = [k for k in planet_data if k[0] == target_name]
    if not keys:
        nan_arr = np.full(len(t_obs_yr), np.nan)
        return nan_arr, nan_arr, np.zeros(len(t_obs_yr)), np.ones(len(t_obs_yr), dtype=bool)

    t_int_accum = np.zeros(len(t_obs_yr))
    t_int_count = np.zeros(len(t_obs_yr))
    comp_accum  = np.zeros(len(t_obs_yr))

    for key in keys:
        d         = planet_data[key]
        t_full    = d["decimal_year"]
        t_int_raw = d["integration_time_hours_con"]["median"].copy().astype(float)
        prob_raw  = d["det_probability"].copy().astype(float)

        bad = d["solar_keepout"] | d["GB_keepout"] | ~np.isfinite(t_int_raw)
        t_int_raw[bad] = np.nan
        valid = ~bad

        if np.sum(valid) > 1:
            t_int_interp = np.interp(
                t_obs_yr, t_full[valid], t_int_raw[valid],
                left=np.nan, right=np.nan,
            )
            t_int_interp[t_obs_yr < t_full[valid].min()] = np.nan
            t_int_interp[t_obs_yr > t_full[valid].max()] = np.nan
        else:
            t_int_interp = np.full(len(t_obs_yr), np.nan)

        finite_mask = np.isfinite(t_int_interp)
        t_int_accum[finite_mask] += t_int_interp[finite_mask]
        t_int_count[finite_mask] += 1
        comp_accum += np.interp(t_obs_yr, t_full, prob_raw, left=0.0, right=0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        t_sci_hrs = np.where(t_int_count > 0, t_int_accum / t_int_count, np.nan)

    t_wall_hrs = (1.0 + R_val) * t_sci_hrs      # ← R applied per target
    in_keepout = ~np.isfinite(t_sci_hrs)

    return t_sci_hrs, t_wall_hrs, comp_accum, in_keepout


# ===========================================================================
# SECTION 5 — OBSERVABILITY AND WINDOW UTILITIES
# ===========================================================================

def find_multi_target_overlap(ref_results: dict, target_names: list) -> dict | None:
    """
    Find windows (one per target) that all overlap simultaneously
    with distinct reference stars.

    Returns {target_name: window_dict} or None.
    """
    valid_windows: dict[str, list] = {}
    for name in target_names:
        wins = [
            w for w in ref_results.get(name, {}).get("observable_windows", [])
            if w.get("best_ref") is not None
        ]
        if not wins:
            return None
        valid_windows[name] = wins

    for combo in cart_product(*[valid_windows[n] for n in target_names]):
        starts = [Time(w["start"], format="iso", scale="utc") for w in combo]
        ends   = [Time(w["end"],   format="iso", scale="utc") for w in combo]
        if min(ends) > max(starts):
            ref_names = [w["best_ref"]["reference_star"] for w in combo]
            if len(set(ref_names)) == len(ref_names):
                return {name: w for name, w in zip(target_names, combo)}
    return None


def overlap_time_array(w1: dict, w2: dict):
    """Build t_obs array over the intersection of two windows."""
    from datetime import datetime
    start_iso = max(w1["start"], w2["start"])
    end_iso   = min(w1["end"],   w2["end"])
    start     = Time(start_iso, format="iso", scale="utc")
    end       = Time(end_iso,   format="iso", scale="utc")
    days      = (
        datetime.fromisoformat(end_iso) - datetime.fromisoformat(start_iso)
    ).days
    t_obs = start + np.arange(days) * u.d
    return start, end, t_obs


# ===========================================================================
# SECTION 6 — SLEW MATRICES
# ===========================================================================

def build_slew_matrix(
    sci_keys: list,
    sci_to_ref: dict[str, str],
    obj_meta: dict,
    all_coords: dict,
    t_mid: Time,
) -> tuple[dict, dict]:
    """
    Scalar slew-time matrices at t_mid.

    Returns:
        inter_slew : {(ki, kj): seconds} for every ordered sci→sci pair
        intra_slew : {sci_key: seconds} to paired ref star for each science target
    """
    inter_slew: dict = {}
    for i, k1 in enumerate(sci_keys):
        for k2 in sci_keys[i + 1:]:
            coord1 = all_coords[obj_meta[k1]["name"]]
            coord2 = all_coords[obj_meta[k2]["name"]]
            t_arr  = Time([t_mid.jd], format="jd", scale=t_mid.scale)
            _, slew_s = compute_slew_time(coord1, coord2, t_arr)
            val = float(np.mean(slew_s))
            inter_slew[(k1, k2)] = val
            inter_slew[(k2, k1)] = val

    intra_slew: dict = {}
    for sci_k in sci_keys:
        ref_name = sci_to_ref.get(sci_k)
        if ref_name and ref_name in all_coords:
            t_arr = Time([t_mid.jd], format="jd", scale=t_mid.scale)
            _, slew_s = compute_slew_time(
                all_coords[obj_meta[sci_k]["name"]],
                all_coords[ref_name],
                t_arr,
            )
            intra_slew[sci_k] = float(np.mean(slew_s))
        else:
            intra_slew[sci_k] = float(ADI_PARAMS["t_slew_default_s"])

    return inter_slew, intra_slew


def build_time_resolved_slew_matrix(
    sci_keys: list,
    sci_to_ref: dict[str, str],
    obj_meta: dict,
    all_coords: dict,
    t_obs,
) -> tuple[dict, dict]:
    """
    Time-resolved slew-time arrays over t_obs.

    Returns:
        inter_slew_ts : {(ki, kj): array} for every ordered sci→sci pair
        intra_slew_ts : {sci_key: array} to paired ref star
    """
    inter_slew_ts: dict = {}
    for i, k1 in enumerate(sci_keys):
        for k2 in sci_keys[i + 1:]:
            coord1 = all_coords[obj_meta[k1]["name"]]
            coord2 = all_coords[obj_meta[k2]["name"]]
            _, slew_s = compute_slew_time(coord1, coord2, t_obs)
            inter_slew_ts[(k1, k2)] = slew_s
            inter_slew_ts[(k2, k1)] = slew_s

    intra_slew_ts: dict = {}
    for sci_k in sci_keys:
        ref_name = sci_to_ref.get(sci_k)
        if ref_name and ref_name in all_coords:
            _, slew_s = compute_slew_time(
                all_coords[obj_meta[sci_k]["name"]],
                all_coords[ref_name],
                t_obs,
            )
            intra_slew_ts[sci_k] = slew_s
        else:
            intra_slew_ts[sci_k] = np.full(len(t_obs), float(ADI_PARAMS["t_slew_default_s"]))

    return inter_slew_ts, intra_slew_ts


# ===========================================================================
# SECTION 7 — LEVEL 1: SEQUENCE SCORING
# ===========================================================================

def score_all_sequences(
    sci_keys: list,
    obj_meta: dict,
    tint_s: dict[str, dict],
    inter_slew: dict,
    intra_slew: dict,
    adi: dict,
    planet_data: dict,
) -> list[dict]:
    """
    Score every visit-order permutation of science targets.

    Reference stars are NOT in the permutation. Their time (R_i × T_science_i)
    is added per target via compute_wall_clock_s().

    R_i is read from tint_s[sci_key]["R"] — the per-target value from corgietc
    or the fallback. Different targets can have different R values.

    Returns list sorted by total_wall_clock_s ascending.
    """
    n_science = len(sci_keys)

    total_science_s = sum(tint_s[k]["t_science_s"] for k in sci_keys)
    total_ref_s     = sum(tint_s[k]["R"] * tint_s[k]["t_science_s"] for k in sci_keys)
    total_howfsc_s  = adi["t_howfsc_s"] * n_science
    total_roll_s    = adi["N_rollchange"] * adi["t_rollchange_s"] * n_science
    total_intra_s   = sum(adi["N_slew"] * intra_slew[k] for k in sci_keys)

    completeness = {
        obj_meta[k]["name"]: target_median_completeness(planet_data, obj_meta[k]["name"])
        for k in sci_keys
    }

    results = []
    for seq in permutations(sci_keys):
        slew_inter_s = sum(
            inter_slew[(seq[i], seq[i + 1])]
            for i in range(len(seq) - 1)
        )
        total_wall_s = compute_wall_clock_s(
            tint_s       = tint_s,
            slew_inter_s = slew_inter_s,
            intra_slew   = intra_slew,
            sci_keys     = sci_keys,
            adi          = adi,
        )
        results.append({
            "sequence":             seq,
            "total_wall_clock_s":   total_wall_s,
            "total_wall_clock_hrs": total_wall_s / 3600.0,
            "total_science_s":      total_science_s,
            "total_ref_s":          total_ref_s,
            "total_slew_inter_s":   slew_inter_s,
            "total_slew_intra_s":   total_intra_s,
            "total_slew_s":         slew_inter_s + total_intra_s,
            "total_howfsc_s":       total_howfsc_s,
            "total_roll_s":         total_roll_s,
            "completeness":         completeness,
            "R_per_target":         {k: tint_s[k]["R"] for k in sci_keys},
        })

    results.sort(key=lambda x: x["total_wall_clock_s"])
    return results


def score_sequences_pareto(
    t_obs,
    sci_keys: list,
    obj_meta: dict,
    tint_s_scalar: dict[str, dict],
    inter_slew_ts: dict,
    intra_slew_ts: dict,
    adi: dict,
    planet_data: dict,
) -> list[dict]:
    """
    Time-resolved Pareto scoring of all permutations.

    Uses per-target R from tint_s_scalar[sci_key]["R"] when computing
    T_wall over time. R_i is constant per target over time (same ref star
    used throughout the window).

    Returns list of dicts with arrays of length len(t_obs).
    """
    t_obs_yr  = t_obs.decimalyear
    n_science = len(sci_keys)

    t_sci_ts:   dict[str, np.ndarray] = {}
    t_wall_ts:  dict[str, np.ndarray] = {}
    comp_arr:   dict[str, np.ndarray] = {}
    in_keepout: dict[str, np.ndarray] = {}

    for k in sci_keys:
        R_k = tint_s_scalar[k]["R"]    # ← per-target R
        t_sci_hrs, t_wall_hrs, comp, ko = interpolate_target_data(
            planet_data, obj_meta[k]["name"], t_obs_yr, R=R_k,
        )
        # Fill keepout with median for array arithmetic; track keepout separately
        med_sci_s = (
            float(np.nanmedian(t_sci_hrs)) * 3600.0
            if np.any(np.isfinite(t_sci_hrs)) else 3600.0
        )
        t_sci_ts[k]   = np.where(np.isfinite(t_sci_hrs), t_sci_hrs * 3600.0, med_sci_s)
        # T_wall_ts already accounts for R_k via interpolate_target_data
        t_wall_ts[k]  = np.where(
            np.isfinite(t_wall_hrs),
            t_wall_hrs * 3600.0,
            (1.0 + R_k) * med_sci_s,
        )
        comp_arr[k]   = comp
        in_keepout[k] = ko

    # Fixed overheads (constant over time)
    fixed_s = n_science * (
        adi["N_rollchange"] * adi["t_rollchange_s"] + adi["t_howfsc_s"]
    )
    # Intra-ADI slew (time-resolved)
    total_intra_slew_s = sum(adi["N_slew"] * intra_slew_ts[k] for k in sci_keys)

    # Combined science+ref time over time: Σ_i (1+R_i) × T_science_i(t)
    total_sci_ref_s = sum(t_wall_ts[k] for k in sci_keys)

    combined_comp = sum(comp_arr[k] for k in sci_keys)
    any_keepout   = np.zeros(len(t_obs), dtype=bool)
    for k in sci_keys:
        any_keepout |= in_keepout[k]

    results = []
    for seq in permutations(sci_keys):
        slew_inter_s = sum(
            inter_slew_ts[(seq[i], seq[i + 1])]
            for i in range(len(seq) - 1)
        )
        wall_clock_s = total_sci_ref_s + slew_inter_s + total_intra_slew_s + fixed_s
        results.append({
            "sequence":       seq,
            "label":          " -> ".join(seq),
            "wall_clock_hrs": wall_clock_s / 3600.0,
            "completeness":   combined_comp.copy(),
            "any_keepout":    any_keepout.copy(),
        })

    return results


def pareto_front_indices(
    wall_clock: np.ndarray,
    completeness: np.ndarray,
) -> np.ndarray:
    """Pareto-optimal indices: minimize wall_clock, maximize completeness."""
    valid = np.isfinite(wall_clock) & np.isfinite(completeness)
    if not np.any(valid):
        return np.array([], dtype=int)
    idx_valid  = np.where(valid)[0]
    order      = idx_valid[np.argsort(wall_clock[idx_valid])]
    pareto_idx = []
    max_comp   = -np.inf
    for i in order:
        if completeness[i] > max_comp:
            pareto_idx.append(i)
            max_comp = completeness[i]
    return np.array(pareto_idx)


# ===========================================================================
# SECTION 8 — COMBINATION SCORING
# ===========================================================================

def score_combination(
    target_names: list,
    ref_results: dict,
    planet_data: dict,
    target_coords: dict,
    adi: dict,
    R_map: dict[str, float] | None = None,
    engine=None,
    pl_name_map: dict[str, str] | None = None,
    band_map: dict[str, int] | None = None,
) -> dict | None:
    """
    Score all visit-order permutations for a campaign combination.

    Reference stars are NOT schedulable objects. Their cost is computed
    via R_i × T_science_i per target using the per-target R value.

    Args:
        target_names : list of science target names
        ref_results  : output of select_ref_star() per target
        planet_data  : loaded planet data
        target_coords: {name: SkyCoord} for all targets
        adi          : ADI_PARAMS
        R_map        : {target_name: R} — per-target tRef_per_tTar values.
                       If None, uses ADI_PARAMS["R_default"] for all targets.
        engine       : SQLAlchemy engine for corgietc (optional)
        pl_name_map  : {target_name: planet_name} for corgietc (optional)
        band_map     : {target_name: band} for corgietc (optional)

    Returns result dict or None if no valid overlapping window exists.
    """
    overlap_windows = find_multi_target_overlap(ref_results, target_names)
    if overlap_windows is None:
        return None

    obj_meta:   dict = {}
    tint_s:     dict = {}
    sci_to_ref: dict = {}
    refs:       list = []

    for i, sci_name in enumerate(target_names):
        sci_key  = f"sci_{i}"
        best_ref  = overlap_windows[sci_name]["best_ref"]
        ref_name  = best_ref["reference_star"]
        ref_stype = best_ref.get("spectype")
        refs.append(ref_name)

        obj_meta[sci_key]   = {"name": sci_name, "type": "science"}
        sci_to_ref[sci_key] = ref_name

        # ── Per-target R value ─────────────────────────────────────────────
        R_val = (R_map or {}).get(sci_name, adi["R_default"])
        # ──────────────────────────────────────────────────────────────────

        pl_name = (pl_name_map or {}).get(sci_name)
        band    = (band_map    or {}).get(sci_name, 1)

        tint_s[sci_key] = get_target_tint_entry(
            planet_data      = planet_data,
            target_name      = sci_name,
            ref_star_name    = ref_name,
            ref_star_spectype = ref_stype,
            R                = R_val,
            band             = band,
            engine           = engine,
            pl_name          = pl_name,
        )

    sci_keys = list(obj_meta.keys())
    n_perms  = math.factorial(len(sci_keys))
    if n_perms > 40320:
        print(f"  Warning: {len(sci_keys)} targets => {n_perms:,} permutations.")

    # Overlap time array
    overlap_start_iso = max(overlap_windows[n]["start"] for n in target_names)
    overlap_end_iso   = min(overlap_windows[n]["end"]   for n in target_names)
    dummy = {"start": overlap_start_iso, "end": overlap_end_iso}
    overlap_start, overlap_end, t_obs = overlap_time_array(dummy, dummy)
    overlap_days = float((overlap_end - overlap_start).to(u.day).value)
    if overlap_days <= 0:
        return None

    # Fetch missing coordinates for reference stars
    new_names = [n for n in refs if n not in target_coords]
    if new_names:
        target_coords.update(get_target_coords(new_names))

    t_mid = Time(
        (overlap_start.mjd + overlap_end.mjd) / 2.0,
        format="mjd", scale="utc",
    )
    inter_slew, intra_slew = build_slew_matrix(
        sci_keys, sci_to_ref, obj_meta, target_coords, t_mid,
    )

    all_sequences = score_all_sequences(
        sci_keys, obj_meta, tint_s, inter_slew, intra_slew, adi, planet_data,
    )
    if not all_sequences:
        return None

    best = all_sequences[0]
    combined_completeness = sum(
        target_median_completeness(planet_data, n) or 0.0 for n in target_names
    )
    efficiency = (
        combined_completeness / best["total_wall_clock_hrs"]
        if best["total_wall_clock_hrs"] > 0 else 0.0
    )

    return {
        "targets":               target_names,
        "refs":                  refs,
        "sci_to_ref":            sci_to_ref,
        "overlap_start":         overlap_start_iso,
        "overlap_end":           overlap_end_iso,
        "overlap_days":          overlap_days,
        "t_obs":                 t_obs,
        "obj_meta":              obj_meta,
        "sci_keys":              sci_keys,
        "tint_s":                tint_s,
        "inter_slew":            inter_slew,
        "intra_slew":            intra_slew,
        "all_sequences":         all_sequences,
        "best_sequence":         best,
        "best_wall_clock_hrs":   best["total_wall_clock_hrs"],
        "combined_completeness": combined_completeness,
        "efficiency":            efficiency,
        "R_per_target":          {sci_to_ref[k]: tint_s[k]["R"]
                                  for k in sci_keys},
    }


# ===========================================================================
# SECTION 9 — LEVEL 2: MISSION KNAPSACK SCHEDULER
# ===========================================================================

@dataclass
class ScheduledCampaign:
    rank:           int
    targets:        list
    refs:           list
    best_sequence:  list
    obj_meta:       dict
    overlap_start:  str
    overlap_end:    str
    overlap_days:   float
    wall_clock_hrs: float
    completeness:   float
    efficiency:     float
    priority_tier:  int
    cumulative_hrs: float
    R_per_target:   dict     # {sci_key: R} for audit


@dataclass
class MissionSchedule:
    campaigns:            List[ScheduledCampaign] = field(default_factory=list)
    budget_days:          float = MISSION_BUDGET_DAYS
    total_wall_clock_hrs: float = 0.0
    total_completeness:   float = 0.0
    n_targets_covered:    int   = 0
    skipped:              List[dict] = field(default_factory=list)


def build_priority_map() -> dict[str, int]:
    """Build {target_name: tier} from HLC_B1/SPEC/WFOV program definitions."""
    pm: dict[str, int] = {}
    for program in (HLC_B1, SPEC, WFOV):
        for name, (tier, band, contrast) in program.items():
            if name not in pm or tier < pm[name]:
                pm[name] = tier
    return pm


def _iso_to_dt(iso: str) -> datetime:
    return datetime.strptime(iso[:10], "%Y-%m-%d")


def _time_clash(sc: ScheduledCampaign, candidate: dict) -> bool:
    s1, e1 = _iso_to_dt(sc.overlap_start), _iso_to_dt(sc.overlap_end)
    s2, e2 = _iso_to_dt(candidate["overlap_start"]), _iso_to_dt(candidate["overlap_end"])
    return min(e1, e2) > max(s1, s2)


def _get_tier(targets: list, priority_map: dict) -> int:
    return min(priority_map.get(t, 4) for t in targets)


def schedule_greedy_knapsack(
    combo_results: list,
    budget_days: float = MISSION_BUDGET_DAYS,
    priority_map: dict = None,
) -> MissionSchedule:
    """
    Greedy Level 2 mission scheduler.

    Ranking: score = (C / T_wall) × (1 / tier)
    Constraints enforced:
        - Budget: cumulative wall-clock ≤ budget_days × 24 hrs
        - Once-per-target: each science target scheduled at most once
        - No time overlap: observable windows must not clash
    """
    if priority_map is None:
        priority_map = build_priority_map()

    budget_hrs    = budget_days * 24.0
    remaining_hrs = budget_hrs
    schedule      = MissionSchedule(budget_days=budget_days)

    def score(c: dict) -> float:
        tier = _get_tier(c["targets"], priority_map)
        return c["efficiency"] * (1.0 / tier) + c["combined_completeness"] * 0.001

    sorted_combos = sorted(combo_results, key=score, reverse=True)
    covered_targets: set[str] = set()
    rank = 0

    for combo in sorted_combos:
        wc_hrs = combo["best_wall_clock_hrs"]

        if wc_hrs > remaining_hrs:
            schedule.skipped.append({**combo, "_skip_reason": "budget exceeded"})
            continue
        if covered_targets & set(combo["targets"]):
            schedule.skipped.append({**combo, "_skip_reason": "target already scheduled"})
            continue
        if any(_time_clash(sc, combo) for sc in schedule.campaigns):
            schedule.skipped.append({**combo, "_skip_reason": "window clash"})
            continue

        rank += 1
        remaining_hrs                 -= wc_hrs
        schedule.total_wall_clock_hrs += wc_hrs
        schedule.total_completeness   += combo["combined_completeness"]
        covered_targets               |= set(combo["targets"])

        tier = _get_tier(combo["targets"], priority_map)
        sc   = ScheduledCampaign(
            rank=rank,
            targets=combo["targets"],
            refs=combo["refs"],
            best_sequence=list(combo["best_sequence"]["sequence"]),
            obj_meta=combo["obj_meta"],
            overlap_start=combo["overlap_start"],
            overlap_end=combo["overlap_end"],
            overlap_days=combo["overlap_days"],
            wall_clock_hrs=wc_hrs,
            completeness=combo["combined_completeness"],
            efficiency=combo["efficiency"],
            priority_tier=tier,
            cumulative_hrs=schedule.total_wall_clock_hrs,
            R_per_target=combo.get("R_per_target", {}),
        )
        schedule.campaigns.append(sc)

    schedule.n_targets_covered = len(covered_targets)
    return schedule


def schedule_exact_knapsack(
    combo_results: list,
    budget_days: float = MISSION_BUDGET_DAYS,
    priority_map: dict = None,
    resolution_hrs: float = 0.5,
) -> MissionSchedule:
    """
    Exact 0/1 knapsack via DP. Maximises total completeness within budget.
    Does not enforce time-window non-overlap.
    """
    if priority_map is None:
        priority_map = build_priority_map()

    budget_hrs = budget_days * 24.0
    best_per_pair: dict[frozenset, dict] = {}
    for combo in combo_results:
        key = frozenset(combo["targets"])
        if key not in best_per_pair or combo["efficiency"] > best_per_pair[key]["efficiency"]:
            best_per_pair[key] = combo
    items = list(best_per_pair.values())
    W     = int(math.ceil(budget_hrs / resolution_hrs))

    dp     = np.zeros(W + 1)
    chosen = [[] for _ in range(W + 1)]

    for combo in items:
        w_i = int(math.ceil(combo["best_wall_clock_hrs"] / resolution_hrs))
        c_i = combo["combined_completeness"]
        for w in range(W, w_i - 1, -1):
            if dp[w - w_i] + c_i > dp[w]:
                dp[w]     = dp[w - w_i] + c_i
                chosen[w] = chosen[w - w_i] + [combo]

    selected = sorted(
        chosen[W],
        key=lambda c: (_get_tier(c["targets"], priority_map), -c["combined_completeness"]),
    )
    schedule = MissionSchedule(budget_days=budget_days)

    for rank, combo in enumerate(selected, 1):
        wc_hrs = combo["best_wall_clock_hrs"]
        schedule.total_wall_clock_hrs += wc_hrs
        schedule.total_completeness   += combo["combined_completeness"]
        tier = _get_tier(combo["targets"], priority_map)
        sc   = ScheduledCampaign(
            rank=rank,
            targets=combo["targets"],
            refs=combo["refs"],
            best_sequence=list(combo["best_sequence"]["sequence"]),
            obj_meta=combo["obj_meta"],
            overlap_start=combo["overlap_start"],
            overlap_end=combo["overlap_end"],
            overlap_days=combo["overlap_days"],
            wall_clock_hrs=wc_hrs,
            completeness=combo["combined_completeness"],
            efficiency=combo["efficiency"],
            priority_tier=tier,
            cumulative_hrs=schedule.total_wall_clock_hrs,
            R_per_target=combo.get("R_per_target", {}),
        )
        schedule.campaigns.append(sc)

    schedule.n_targets_covered = len({t for sc in schedule.campaigns for t in sc.targets})
    return schedule


# ===========================================================================
# SECTION 10 — PRINTING
# ===========================================================================

def print_optimal_sequence(
    sequence_results: list[dict],
    pareto_results: list[dict],
    obj_meta: dict,
    adi: dict,
    t_obs,
) -> None:
    """Print the Pareto-optimal sequence with full time breakdown including per-target R."""
    best        = None
    best_median = np.inf

    for res in pareto_results:
        mask = np.isfinite(res["wall_clock_hrs"]) & ~res["any_keepout"]
        if not np.any(mask):
            continue
        median_wc = float(np.median(res["wall_clock_hrs"][mask]))
        if median_wc < best_median:
            best_median = median_wc
            best        = res

    if best is None:
        print("No fully observable sequence found.")
        return

    seq     = best["sequence"]
    mask    = np.isfinite(best["wall_clock_hrs"]) & ~best["any_keepout"]
    matched = next((r for r in sequence_results if r["sequence"] == seq), None)

    print("\n" + "=" * 65)
    print("OPTIMAL SEQUENCE")
    print("=" * 65)
    print(f"  Order : {' → '.join(obj_meta[k]['name'] for k in seq)}")
    print(f"  Dates : {np.sum(mask)} observable days in window")
    wc_hrs = best["wall_clock_hrs"][mask]
    print(f"  Wall clock (median): {np.median(wc_hrs):.2f} hrs")
    print(f"  Wall clock (min):    {np.min(wc_hrs):.2f} hrs")

    if matched:
        sci_s    = matched["total_science_s"]
        ref_s    = matched["total_ref_s"]
        roll_s   = matched["total_roll_s"]
        howfsc_s = matched["total_howfsc_s"]
        total_s  = matched["total_wall_clock_s"]

        print(f"\n  Time breakdown:")
        print(f"    {'Science integration':<38} {sci_s/3600:.2f} hrs  ({100*sci_s/total_s:.1f}%)")

        # Show R per target
        R_per = matched.get("R_per_target", {})
        for k in seq:
            R_k = R_per.get(k, adi["R_default"])
            ref_k = R_k * matched["total_science_s"] / len(seq)   # approximate
            print(f"    Reference ({obj_meta[k]['name']}, R={R_k:.3f})"
                  f"{'':>10} R×T_sci = {R_k * matched['total_science_s'] / 3600 / len(seq):.2f} hrs")

        print(f"    {'Reference total':<38} {ref_s/3600:.2f} hrs  ({100*ref_s/total_s:.1f}%)")
        print(f"    {'Slew inter-target':<38} {matched['total_slew_inter_s']/3600:.2f} hrs  "
              f"({100*matched['total_slew_inter_s']/total_s:.1f}%)")
        print(f"    {'Slew intra-ADI':<38} {matched['total_slew_intra_s']/3600:.2f} hrs  "
              f"({100*matched['total_slew_intra_s']/total_s:.1f}%)")
        print(f"    {'Roll changes':<38} {roll_s/3600:.2f} hrs  ({100*roll_s/total_s:.1f}%)")
        print(f"    {'HOWFSC':<38} {howfsc_s/3600:.2f} hrs  ({100*howfsc_s/total_s:.1f}%)")
        print(f"    {'TOTAL':<38} {total_s/3600:.2f} hrs")

    print(f"\n  Combined completeness (median): "
          f"{float(np.median(best['completeness'][mask])):.4f}")
    print("=" * 65)


def print_sequence_time_table(
    sequence_results: list[dict],
    obj_meta: dict,
) -> None:
    """Print wall-clock breakdown for every permutation, showing per-target R."""
    if not sequence_results:
        return

    sci_hrs = sequence_results[0]["total_science_s"] / 3600.0
    ref_hrs = sequence_results[0]["total_ref_s"]     / 3600.0
    R_per   = sequence_results[0].get("R_per_target", {})

    print(f"\n  Science integration (all sequences): {sci_hrs:.2f} h")
    print(f"  Reference total     (all sequences): {ref_hrs:.2f} h")
    print(f"  Per-target R values: "
          + ", ".join(f"{obj_meta[k]['name']}=R{v:.3f}" for k, v in R_per.items()))
    print(f"  [Reference star is NOT a schedulable object. "
          f"T_ref = R_i × T_science_i per target.]")

    hdr = (f"  {'Sequence (science targets)':<50}  "
           f"{'Slew inter':>10}  {'Slew intra':>10}  {'Wall clock':>11}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in sequence_results:
        label     = " → ".join(obj_meta[k]["name"] for k in r["sequence"])
        inter_hrs = r["total_slew_inter_s"] / 3600.0
        intra_hrs = r["total_slew_intra_s"] / 3600.0
        wc_hrs    = r["total_wall_clock_hrs"]
        print(f"  {label:<50}  {inter_hrs:>10.2f}  {intra_hrs:>10.2f}  {wc_hrs:>11.2f}")


def print_mission_schedule(schedule: MissionSchedule) -> None:
    """Print the full mission timeline table with R values per campaign."""
    sep = "=" * 115
    print(f"\n{sep}")
    print(f"MISSION SCHEDULE — {len(schedule.campaigns)} campaigns | "
          f"Budget: {schedule.budget_days}d | "
          f"Used: {schedule.total_wall_clock_hrs:.1f}/{schedule.budget_days*24:.0f} hrs | "
          f"Targets: {schedule.n_targets_covered} | "
          f"Total C: {schedule.total_completeness:.4f}")
    print(sep)
    hdr = (f"{'#':>3}  {'Targets':<30}  {'Window':<22}  "
           f"{'T_wall':>7}  {'C':>6}  {'Tier':>4}  {'Cum hrs':>8}  R values")
    print(hdr)
    print("-" * len(hdr))
    for sc in schedule.campaigns:
        tstr  = " + ".join(sc.targets)
        wstr  = f"{sc.overlap_start[:10]} → {sc.overlap_end[:10]}"
        r_str = ", ".join(f"{k}:{v:.3f}" for k, v in sc.R_per_target.items())
        seq   = " → ".join(sc.obj_meta[k]["name"] for k in sc.best_sequence)
        print(f"{sc.rank:>3}  {tstr:<30}  {wstr:<22}  "
              f"{sc.wall_clock_hrs:>7.2f}  {sc.completeness:>6.4f}  "
              f"{sc.priority_tier:>4}  {sc.cumulative_hrs:>8.1f}  {r_str}")
        print(f"       Sequence: {seq}")
        print(f"       Refs:     {' + '.join(sc.refs)}")
    print(sep)
    if schedule.skipped:
        print(f"\n{len(schedule.skipped)} campaigns skipped:")
        reasons: dict[str, int] = {}
        for s in schedule.skipped:
            r = s.get("_skip_reason", "unknown")
            reasons[r] = reasons.get(r, 0) + 1
        for reason, count in reasons.items():
            print(f"  {count:>4}  {reason}")


def print_campaign_summary(combo_results: list, top_k: int = 10) -> None:
    """Print ranked campaign combinations. Shows R per target."""
    if not combo_results:
        return
    subset = combo_results[:top_k]
    col_w  = 38
    header = (
        f"{'Rank':>4}  {'Targets':<{col_w}}  "
        f"{'Total':>7}  {'Sci':>6}  {'Ref':>6}  {'Slew':>6}  {'OH':>6}  "
        f"{'Comp':>7}  {'Eff':>9}  {'Days':>5}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}")
    print(f"CAMPAIGN SUMMARY  ({len(combo_results)} combinations, top {len(subset)})")
    print(f"  Ref = Σ R_i × T_sci_i  (per-target R from corgietc or fallback R_default)")
    print(sep)
    print(header)
    print("-" * len(header))
    for rank, r in enumerate(subset, 1):
        bs     = r["best_sequence"]
        sci_h  = bs["total_science_s"]  / 3600.0
        ref_h  = bs["total_ref_s"]      / 3600.0
        slew_h = bs["total_slew_s"]     / 3600.0
        oh_h   = (bs["total_howfsc_s"] + bs.get("total_roll_s", 0.0)) / 3600.0
        tstr   = " + ".join(r["targets"])
        print(
            f"{rank:>4}  {tstr:<{col_w}}  "
            f"{sci_h+ref_h+slew_h+oh_h:>7.2f}  {sci_h:>6.2f}  {ref_h:>6.2f}  "
            f"{slew_h:>6.2f}  {oh_h:>6.2f}  "
            f"{r['combined_completeness']:>7.4f}  "
            f"{r['efficiency']:>9.5f}  {r['overlap_days']:>5.0f}"
        )
        seq_names = " → ".join(r["obj_meta"][k]["name"] for k in bs["sequence"])
        R_str     = ", ".join(
            f"{r['obj_meta'][k]['name']}=R{r['tint_s'][k]['R']:.3f}"
            for k in r["sci_keys"]
        )
        print(f"       {seq_names}   [{R_str}]")
    print(sep)


# ===========================================================================
# SECTION 11 — PLOTTING
# ===========================================================================

def plot_pareto(
    targ1_name: str,
    targ2_name: str,
    sequence_results: list[dict],
    t_obs,
    obj_meta: dict,
    planet_data: dict,
    tint_s: dict[str, dict] = None,
) -> None:
    """Three-panel Pareto analysis. Uses per-target R for T_wall display."""
    import matplotlib.cm as cm

    colors   = cm.tab10(np.linspace(0, 1, min(len(sequence_results), 10)))
    t_obs_yr = t_obs.decimalyear

    sci_keys    = list(sequence_results[0]["sequence"])
    t_int_total = np.zeros(len(t_obs))
    for k in sci_keys:
        R_k = (tint_s or {}).get(k, {}).get("R", ADI_PARAMS["R_default"])
        t_sci_hrs, _, _, _ = interpolate_target_data(
            planet_data, obj_meta[k]["name"], t_obs_yr, R=R_k,
        )
        t_int_total += np.where(np.isfinite(t_sci_hrs), t_sci_hrs, 0.0)

    comp_shared = sequence_results[0]["completeness"]
    ko_shared   = sequence_results[0]["any_keepout"]
    obs_mask    = np.isfinite(comp_shared) & ~ko_shared

    if not np.any(obs_mask):
        print(f"plot_pareto: no observable dates — skipping.")
        return

    seq_labels = [
        " → ".join(obj_meta[k]["name"] for k in res["sequence"])
        for res in sequence_results
    ]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))

    # Panel 1: Pareto front
    med_wc, med_comp = [], []
    for i, res in enumerate(sequence_results):
        mask = np.isfinite(res["wall_clock_hrs"]) & ~res["any_keepout"]
        if np.any(mask):
            ax1.scatter(res["wall_clock_hrs"][mask], comp_shared[mask],
                        color=colors[i % len(colors)], alpha=0.12, s=10, zorder=2)
        med_wc.append(float(np.median(res["wall_clock_hrs"][mask])) if np.any(mask) else np.nan)
        med_comp.append(float(np.median(comp_shared[mask])) if np.any(mask) else np.nan)

    med_wc   = np.array(med_wc)
    med_comp = np.array(med_comp)
    valid    = np.isfinite(med_wc) & np.isfinite(med_comp)

    for i in np.where(valid)[0]:
        ax1.scatter(med_wc[i], med_comp[i], color=colors[i % len(colors)],
                    s=110, zorder=5, edgecolors="black", linewidths=0.8,
                    label=seq_labels[i])

    pf_idx = pareto_front_indices(med_wc[valid], med_comp[valid])
    if len(pf_idx) > 0:
        valid_positions = np.where(valid)[0]
        pf_wc   = med_wc[valid][pf_idx]
        pf_comp = med_comp[valid][pf_idx]
        so      = np.argsort(pf_wc)
        ax1.step(pf_wc[so], pf_comp[so], where="post",
                 color="red", linewidth=2.5, zorder=6, label="Pareto front")
        ax1.scatter(pf_wc, pf_comp, color="red", s=80, zorder=7,
                    edgecolors="darkred", linewidths=1.0)
        for idx in pf_idx:
            ax1.annotate(seq_labels[valid_positions[idx]],
                         (med_wc[valid][idx], med_comp[valid][idx]),
                         fontsize=7, xytext=(6, 4), textcoords="offset points")

    ax1.set_xlabel("Wall-clock (hrs)  [median, observable dates]")
    ax1.set_ylabel("Combined completeness  [median]")
    ax1.set_title("Pareto front — sequences")
    if np.any(valid):
        ax1.legend(fontsize=7, loc="lower right", framealpha=0.9,
                   title="Sequence", title_fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Panel 2: Wall-clock over time
    for i, res in enumerate(sequence_results):
        wc   = res["wall_clock_hrs"].copy().astype(float)
        mask = np.isfinite(wc) & ~res["any_keepout"]
        wc[~mask] = np.nan
        ax2.plot(t_obs_yr, wc, color=colors[i % len(colors)],
                 linewidth=1.4, alpha=0.85, label=seq_labels[i])
        if np.any(mask):
            ax2.scatter(t_obs_yr[mask][np.argmin(wc[mask])], np.nanmin(wc[mask]),
                        color=colors[i % len(colors)], s=55, zorder=5,
                        edgecolors="black", linewidths=0.7)
    ax2.set_xlabel("Observation date (decimal year)")
    ax2.set_ylabel("Wall-clock (hrs)")
    ax2.set_title("Wall-clock over window\nDots = minimum-cost date per sequence")
    ax2.legend(fontsize=7, loc="best", title="Sequence", title_fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.4)

    # Panel 3: Science yield over time
    comp_plot  = comp_shared.copy().astype(float)
    t_int_plot = t_int_total.copy()
    comp_plot[~obs_mask]  = np.nan
    t_int_plot[~obs_mask] = np.nan

    ax3.plot(t_obs_yr, comp_plot, color="steelblue", linewidth=2.0,
             label="Combined completeness", zorder=4)
    ax3.set_xlabel("Observation date (decimal year)")
    ax3.set_ylabel("Combined completeness", color="steelblue")
    ax3.tick_params(axis="y", labelcolor="steelblue")

    ax3r = ax3.twinx()
    ax3r.plot(t_obs_yr, t_int_plot, color="darkorange", linewidth=1.6,
              linestyle="--", label="Total T_science", zorder=3)
    ax3r.set_ylabel("Total T_science (hrs)", color="darkorange")
    ax3r.tick_params(axis="y", labelcolor="darkorange")

    lines1, labs1 = ax3.get_legend_handles_labels()
    lines2, labs2 = ax3r.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="best")
    ax3.set_title("Science yield over time\nHigh C + low T_sci = efficient window")
    ax3.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(f"{targ1_name}  +  {targ2_name}  —  observable dates only", fontsize=13)
    plt.tight_layout()
    plt.show(block=False)


def plot_mission_timeline(schedule: MissionSchedule,
                          analysis_start: str = ANALYSIS_START) -> None:
    """Gantt-style mission timeline. Colour = priority tier. Width = wall-clock."""
    tier_colors = {1: "#e45756", 2: "#f58518", 3: "#4c78a8", 4: "#54a24b"}
    t0 = datetime.strptime(analysis_start[:10], "%Y-%m-%d")

    fig, (ax_main, ax_budget) = plt.subplots(
        2, 1, figsize=(14, 7),
        gridspec_kw={"height_ratios": [4, 1]},
    )

    yticks, ylabels = [], []
    for sc in schedule.campaigns:
        y        = sc.rank - 1
        s_days   = (_iso_to_dt(sc.overlap_start) - t0).days
        e_days   = (_iso_to_dt(sc.overlap_end)   - t0).days
        dur_days = e_days - s_days
        color    = tier_colors.get(sc.priority_tier, "#9e9e9e")
        wc_days  = sc.wall_clock_hrs / 24.0

        ax_main.barh(y, dur_days, left=s_days, height=0.5,
                     color=color, alpha=0.2, edgecolor=color, linewidth=0.5)
        ax_main.barh(y, wc_days, left=s_days, height=0.5,
                     color=color, alpha=0.85, edgecolor="white", linewidth=0.8)

        r_str = " ".join(f"R{v:.2f}" for v in sc.R_per_target.values())
        ax_main.text(s_days + wc_days + 0.5, y,
                     f"C={sc.completeness:.3f} [{r_str}]", va="center", fontsize=7)

        yticks.append(y)
        ylabels.append(f"#{sc.rank} {' + '.join(sc.targets)}")

    ax_main.set_yticks(yticks)
    ax_main.set_yticklabels(ylabels, fontsize=8)
    ax_main.set_xlabel("Days from mission start")
    ax_main.set_title(
        f"Mission Timeline — {len(schedule.campaigns)} campaigns  |  "
        f"Total C={schedule.total_completeness:.4f}  |  "
        f"{schedule.total_wall_clock_hrs:.1f} hrs used",
        fontsize=10,
    )
    handles = [
        mpatches.Patch(color=tier_colors[t], label=f"Priority tier {t}")
        for t in sorted(tier_colors)
        if any(sc.priority_tier == t for sc in schedule.campaigns)
    ]
    ax_main.legend(handles=handles, fontsize=8, loc="lower right")
    ax_main.grid(axis="x", linestyle="--", alpha=0.4)
    ax_main.set_xlim(left=0)

    cum_hrs = [0.0] + [sc.cumulative_hrs for sc in schedule.campaigns]
    ax_budget.step(range(len(cum_hrs)), cum_hrs, where="post",
                   color="#4c78a8", linewidth=2)
    ax_budget.axhline(schedule.budget_days * 24, color="red",
                      linewidth=1.5, linestyle="--", label="Budget limit")
    ax_budget.set_xlabel("Campaigns scheduled")
    ax_budget.set_ylabel("Cumulative hrs")
    ax_budget.set_ylim(0, schedule.budget_days * 24 * 1.1)
    ax_budget.legend(fontsize=8)
    ax_budget.grid(linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show(block=False)


def plot_campaign_summary(combo_results: list) -> None:
    """Two-figure campaign summary: Pareto front + breakdown bars."""
    if not combo_results:
        return

    all_wc   = np.array([r["best_wall_clock_hrs"]   for r in combo_results])
    all_comp = np.array([r["combined_completeness"] for r in combo_results])
    pf_idx   = pareto_front_indices(all_wc, all_comp)
    pf_set   = set(pf_idx.tolist())

    fig1, ax1 = plt.subplots(figsize=(11, 7))
    non_pf = [i for i in range(len(combo_results)) if i not in pf_set]
    if non_pf:
        ax1.scatter(all_wc[non_pf], all_comp[non_pf],
                    color="tab:blue", alpha=0.45, s=28, zorder=3, label="All")
    if len(pf_idx) > 0:
        pf_wc   = all_wc[pf_idx]
        pf_comp = all_comp[pf_idx]
        so      = np.argsort(pf_wc)
        ax1.step(pf_wc[so], pf_comp[so], where="post", color="red", linewidth=2.5, zorder=5)
        ax1.scatter(pf_wc, pf_comp, color="red", s=90, zorder=6,
                    edgecolors="darkred", linewidths=1.0, label="Pareto front")
        for i in pf_idx:
            ax1.annotate(
                " + ".join(combo_results[i]["targets"]),
                (all_wc[i], all_comp[i]),
                fontsize=7, xytext=(6, 4), textcoords="offset points",
            )
    ax1.set_xlabel("Best wall-clock (hrs)"); ax1.set_ylabel("Combined completeness")
    ax1.set_title(f"Campaign Pareto front — {len(combo_results)} combinations")
    ax1.legend(fontsize=9); ax1.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout(); plt.show(block=False)

    pf_combos = sorted([combo_results[i] for i in pf_idx],
                        key=lambda r: r["combined_completeness"])
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

    n      = len(pf_combos)
    fig2, (ax_bar, ax_comp) = plt.subplots(
        1, 2, figsize=(14, max(4, 0.55 * n + 2.5)),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    y      = np.arange(n)
    bar_kw = dict(height=0.6, edgecolor="white", linewidth=0.5)
    ax_bar.barh(y, sci_h,                                color="#4c78a8", label="Science",              **bar_kw)
    ax_bar.barh(y, ref_h,  left=sci_h,                  color="#f58518", label="Reference (Σ R_i×T_i)",**bar_kw)
    ax_bar.barh(y, slew_h, left=sci_h + ref_h,          color="#54a24b", label="Slew",                 **bar_kw)
    ax_bar.barh(y, oh_h,   left=sci_h + ref_h + slew_h, color="#b279a2", label="OH",                   **bar_kw)
    for i, (tot, sci) in enumerate(zip(totals, sci_h)):
        sci_frac = 100.0 * sci / tot if tot > 0 else 0.0
        ax_bar.text(tot + 0.01 * totals.max(), i,
                    f"{tot:.2f} h  ({sci_frac:.0f}% sci)", va="center", ha="left", fontsize=8)
    ax_bar.set_yticks(y); ax_bar.set_yticklabels(labels, fontsize=9)
    ax_bar.set_xlabel("Wall-clock (hrs)")
    ax_bar.set_title("Wall-clock breakdown — Pareto-front combinations")
    ax_bar.legend(loc="lower right", fontsize=8); ax_bar.set_xlim(0, totals.max() * 1.25)
    ax_bar.grid(axis="x", linestyle="--", alpha=0.4)

    ax_comp.barh(y, comp, height=0.6, color="#e45756", edgecolor="white")
    ax_comp.set_yticks(y); ax_comp.set_yticklabels([])
    ax_comp.set_xlabel("Combined completeness"); ax_comp.set_title("Completeness")
    ax_comp.set_xlim(0, comp.max() * 1.3); ax_comp.grid(axis="x", linestyle="--", alpha=0.4)
    for i, c in enumerate(comp):
        ax_comp.text(c + 0.005 * comp.max(), i, f"{c:.4f}",
                     va="center", ha="left", fontsize=8)

    fig2.suptitle("Pareto-front combinations", fontsize=10, y=1.02)
    plt.tight_layout(); plt.show(block=False)


# ===========================================================================
# SECTION 12 — MAIN ENTRY POINT
# ===========================================================================

def run_campaign(
    start: str         = ANALYSIS_START,
    days: float        = ANALYSIS_DAYS,
    combo_size: int    = 2,
    top_k: int         = 10,
    top_k_pareto: int  = 3,
    budget_days: float = MISSION_BUDGET_DAYS,
    solver: str        = "greedy",
    R_map: dict        = None,
    engine             = None,
    pl_name_map: dict  = None,
    band_map: dict     = None,
    plot: bool         = True,
) -> dict:
    """
    Full two-level mission scheduling pipeline.

    Level 1: Score all C(valid_targets, combo_size) campaign combinations.
             Each combination scores all n! permutations of science targets.
             R_i per target comes from R_map (or R_default if not provided).

    Level 2: Greedy or exact knapsack selects which campaigns to schedule
             across the mission window subject to budget, uniqueness,
             and time-window non-overlap constraints.

    Args:
        start        : ISO analysis start datetime
        days         : analysis window in days
        combo_size   : science targets per campaign (default 2)
        top_k        : rows in summary table
        top_k_pareto : combinations for full time-resolved Pareto analysis
        budget_days  : mission budget in days (default 60)
        solver       : "greedy" (fast) or "exact" (DP, guaranteed optimal)
        R_map        : {target_name: R} — explicit per-target tRef_per_tTar override.
                       If None and engine provided, R is auto-computed via optimise_R().
                       If None and no engine, all targets fall back to _R_FALLBACK=0.25.
        engine       : SQLAlchemy engine for corgietc — enables R optimization and T_science correction
        pl_name_map  : {target_name: planet_name} for corgietc (optional)
        band_map     : {target_name: band} for corgietc (optional)
        plot         : produce plots

    Returns:
        dict: valid_targets, combo_results, ref_results, planet_data,
              target_coords, mission_schedule
    """
    n_perms = math.factorial(combo_size)

    print(f"\n{'='*65}")
    print(f"MISSION SCHEDULING PIPELINE")
    print(f"  Start: {start[:10]}  Window: {days}d  Combo: {combo_size} targets")
    print(f"  Permutations per campaign: {n_perms}")
    print(f"  Budget: {budget_days}d  Solver: {solver}")
    print(f"  R_map: {'provided' if R_map else ('auto-computed via optimise_R' if engine else f'none — fallback R={_R_FALLBACK}')}")
    print(f"{'='*65}")

    # Step 1: Load data
    print("\n[1] Loading planet data...")
    planet_data  = load_planet_data()
    catalog      = load_catalog()
    priority_map = build_priority_map()

    targets_with_data    = {k[0] for k in planet_data}
    target_band_contrast = {}
    for program in (HLC_B1, SPEC, WFOV):
        for name, (tier, band, contrast) in program.items():
            if name in targets_with_data and name not in target_band_contrast:
                target_band_contrast[name] = (band, contrast)
    for name in targets_with_data:
        if name not in target_band_contrast:
            target_band_contrast[name] = (1, "high")

    n_candidates = len(target_band_contrast)
    print(f"   {n_candidates} targets have planet data.")

    # Step 2: Coordinates
    print(f"\n[2] Fetching coordinates...")
    target_coords = get_target_coords(list(target_band_contrast.keys()))

    # Step 3: Reference stars
    ref_results: dict = {}
    print(f"\n[3] Selecting reference stars for {n_candidates} targets...")
    for i, (name, (band, contrast)) in enumerate(target_band_contrast.items(), 1):
        print(f"  [{i:>3}/{n_candidates}] {name}  (band={band})")
        ref_results[name] = select_ref_star(
            sci_name=name, analysis_start=start, analysis_days=days,
            band=band, contrast=contrast, catalog=catalog,
        )

    # Step 4: Filter
    valid_targets = [
        name for name, result in ref_results.items()
        if "observable_windows" in result
        and any(w.get("best_ref") is not None for w in result["observable_windows"])
    ]
    print(f"\n[4] {len(valid_targets)} / {n_candidates} targets observable.")

    if len(valid_targets) < combo_size:
        return {
            "valid_targets": valid_targets, "combo_results": [],
            "ref_results": ref_results, "planet_data": planet_data,
            "target_coords": target_coords, "mission_schedule": None,
        }

    # Step 4b: Compute optimal R for each valid target
    if engine is not None and R_map is None:
        print(f"\n[4b] Computing optimal R for {len(valid_targets)} targets...")
        R_map = {}
        for name in valid_targets:
            ref_name    = None
            ref_spectype = None
            for w in ref_results[name].get("observable_windows", []):
                if w.get("best_ref") is not None:
                    ref_name     = w["best_ref"]["reference_star"]
                    ref_spectype = w["best_ref"].get("spectype")
                    break
            pl = (pl_name_map or {}).get(name)
            if pl is None:
                keys = [k for k in planet_data if k[0] == name]
                pl = f"{keys[0][0]} {keys[0][1]}" if keys else None
            band = (band_map or {}).get(name, target_band_contrast.get(name, (1,))[0])
            if pl is None:
                print(f"  {name}: no pl_name — using R={_R_FALLBACK}")
                R_map[name] = _R_FALLBACK
                continue
            try:
                res = optimise_R(pl_name=pl, band=band, engine=engine,
                                 ref_star_name=ref_name,
                                 ref_star_spectype=ref_spectype)
                R_map[name] = res["R_opt"]
            except Exception as e:
                print(f"  {name}: optimise_R failed ({e}) — using R={_R_FALLBACK}")
                R_map[name] = _R_FALLBACK
        print(f"  R_map: { {k: round(v,3) for k,v in R_map.items()} }")
    elif R_map is not None:
        print(f"\n[4b] Using provided R_map: {R_map}")
    else:
        print(f"\n[4b] No engine provided — using R={_R_FALLBACK} for all targets")

    # Step 5: Level 1 — score all combinations
    all_combos = list(iter_combinations(valid_targets, combo_size))
    print(f"\n[5] Level 1 — scoring {len(all_combos)} combinations...")

    combo_results: list = []
    n_skipped = 0
    for idx, targets in enumerate(all_combos, 1):
        result = score_combination(
            list(targets), ref_results, planet_data, target_coords, ADI_PARAMS,
            R_map=R_map, engine=engine, pl_name_map=pl_name_map, band_map=band_map,
        )
        if result is None:
            n_skipped += 1
        else:
            combo_results.append(result)
        if idx % 25 == 0 or idx == len(all_combos):
            print(f"  [{idx:>5}/{len(all_combos)}]  {len(combo_results)} valid  |  {n_skipped} skipped")

    if not combo_results:
        print("  No schedulable combinations found.")
        return {
            "valid_targets": valid_targets, "combo_results": [],
            "ref_results": ref_results, "planet_data": planet_data,
            "target_coords": target_coords, "mission_schedule": None,
        }

    combo_results.sort(key=lambda r: r["efficiency"], reverse=True)
    print_campaign_summary(combo_results, top_k=top_k)
    if plot:
        plot_campaign_summary(combo_results)

    # Step 6: Level 2 — mission knapsack
    print(f"\n[6] Level 2 — mission knapsack ({solver})...")
    if solver == "exact":
        mission_schedule = schedule_exact_knapsack(
            combo_results, budget_days=budget_days, priority_map=priority_map,
        )
    else:
        mission_schedule = schedule_greedy_knapsack(
            combo_results, budget_days=budget_days, priority_map=priority_map,
        )
    print_mission_schedule(mission_schedule)
    if plot:
        plot_mission_timeline(mission_schedule, analysis_start=start)

    # Step 7: Full Pareto for top combinations
    print(f"\n[7] Full Pareto for top {top_k_pareto} combinations...")
    for rank, result in enumerate(combo_results[:top_k_pareto], 1):
        tnames   = result["targets"]
        sci_keys = result["sci_keys"]
        obj_meta = result["obj_meta"]
        sci_to_ref = result["sci_to_ref"]
        tint_s   = result["tint_s"]

        print(f"\n{'─'*65}")
        print(f"Pareto [{rank}/{top_k_pareto}]: {' + '.join(tnames)}")
        R_str = ", ".join(
            f"{obj_meta[k]['name']}=R{tint_s[k]['R']:.3f} ({tint_s[k]['source']})"
            for k in sci_keys
        )
        print(f"  R values: {R_str}")
        print(f"{'─'*65}")

        t_obs = result["t_obs"]
        inter_slew_ts, intra_slew_ts = build_time_resolved_slew_matrix(
            sci_keys, sci_to_ref, obj_meta, target_coords, t_obs,
        )
        pareto_results = score_sequences_pareto(
            t_obs, sci_keys, obj_meta, tint_s,
            inter_slew_ts, intra_slew_ts, ADI_PARAMS, planet_data,
        )

        print_optimal_sequence(
            result["all_sequences"], pareto_results, obj_meta, ADI_PARAMS, t_obs,
        )
        print_sequence_time_table(result["all_sequences"], obj_meta)

        if plot:
            sci_names = [obj_meta[k]["name"] for k in sci_keys]
            plot_pareto(
                sci_names[0],
                " + ".join(sci_names[1:]) if len(sci_names) > 1 else sci_names[0],
                pareto_results, t_obs, obj_meta, planet_data, tint_s=tint_s,
            )

    if plot:
        plt.pause(0.001)
        input("\nPress Enter to close all plots...")
        plt.close("all")

    return {
        "valid_targets":    valid_targets,
        "combo_results":    combo_results,
        "ref_results":      ref_results,
        "planet_data":      planet_data,
        "target_coords":    target_coords,
        "mission_schedule": mission_schedule,
    }


# ===========================================================================
# TOY CAMPAIGN (fast sanity check)
# ===========================================================================

def toy_campaign(
    targ1_name: str = "* eps Eri",
    targ2_name: str = "ups And",
    ref1_name:  str = "* bet Leo",
    ref2_name:  str = "* tau Cet",
    R1: float = None,
    R2: float = None,
) -> tuple:
    """
    Quick sanity check with hardcoded targets.

    R1, R2 are the tRef_per_tTar values for targ1 and targ2 respectively.
    If None, uses ADI_PARAMS["R_default"] for both.
    """
    R1_val = R1 if R1 is not None else ADI_PARAMS["R_default"]
    R2_val = R2 if R2 is not None else ADI_PARAMS["R_default"]

    print(f"=== TOY CAMPAIGN: {targ1_name} (R={R1_val}) + {targ2_name} (R={R2_val}) ===\n")

    planet_data   = load_planet_data()
    target_coords = get_target_coords([targ1_name, targ2_name, ref1_name, ref2_name])

    t0     = Time(ANALYSIS_START, format="isot", scale="utc")
    window = {
        "start":         ANALYSIS_START[:10],
        "end":           (t0 + ANALYSIS_DAYS * u.d).iso[:10],
        "duration_days": float(ANALYSIS_DAYS),
    }
    overlap_start, overlap_end, t_obs = overlap_time_array(window, window)

    obj_meta   = {
        "sci_0": {"name": targ1_name, "type": "science"},
        "sci_1": {"name": targ2_name, "type": "science"},
    }
    sci_keys   = ["sci_0", "sci_1"]
    sci_to_ref = {"sci_0": ref1_name, "sci_1": ref2_name}

    # Build tint_s with per-target R values
    tint_s: dict = {}
    for k, name, ref_name, R_val in [
        ("sci_0", targ1_name, ref1_name, R1_val),
        ("sci_1", targ2_name, ref2_name, R2_val),
    ]:
        keys = [pk for pk in planet_data if pk[0] == name]
        if keys:
            _, t_int_hrs = get_integration_time(planet_data, keys[0])
            t_sci_s = float(np.nanmedian(t_int_hrs)) * 3600.0
        else:
            t_sci_s = 3600.0
        tint_s[k] = make_tint_entry(
            t_science_s=t_sci_s, R=R_val,
            ref_star=ref_name, source="npy_fallback",
        )

    t_mid = Time(
        (overlap_start.mjd + overlap_end.mjd).item() / 2,
        format="mjd", scale="utc",
    )
    inter_slew, intra_slew = build_slew_matrix(
        sci_keys, sci_to_ref, obj_meta, target_coords, t_mid,
    )
    sequence_results = score_all_sequences(
        sci_keys, obj_meta, tint_s, inter_slew, intra_slew, ADI_PARAMS, planet_data,
    )

    inter_slew_ts, intra_slew_ts = build_time_resolved_slew_matrix(
        sci_keys, sci_to_ref, obj_meta, target_coords, t_obs,
    )
    pareto_results = score_sequences_pareto(
        t_obs, sci_keys, obj_meta, tint_s,
        inter_slew_ts, intra_slew_ts, ADI_PARAMS, planet_data,
    )

    plot_pareto(targ1_name, targ2_name, pareto_results, t_obs, obj_meta, planet_data, tint_s=tint_s)
    print_optimal_sequence(sequence_results, pareto_results, obj_meta, ADI_PARAMS, t_obs)
    print_sequence_time_table(sequence_results, obj_meta)

    return sequence_results, pareto_results

REF_FRACS = np.linspace(0.01, 1.0, 100)   # notebook: ref_fracs


def _load_corgietc(sy_vmag: float, ref_star_vmag: float, band: int, months_at_l2: float,
                   ref_star_spectype: str = "a0v"):
    import EXOSIMS.Prototypes.TargetList, EXOSIMS.Prototypes.TimeKeeping
    cache = os.path.expanduser(r"~\.EXOSIMS\cache")
    if os.path.exists(cache):
        shutil.rmtree(cache)
    with open(os.path.join(os.environ["CORGIETC_DATA_DIR"], "scripts", "CGI_Noise.json")) as f:
        specs = json.loads(f.read())
    TL = EXOSIMS.Prototypes.TargetList.TargetList(**copy.deepcopy(specs))
    OS = TL.OpticalSystem
    vmag_default = float(TL.Vmag[0])
    bwl = {1: 575, 3: 730, 4: 825}
    mode = next((m for m in OS.observingModes if abs(m["lam"].to_value(u.nm) - bwl[band]) < 20),
                OS.observingModes[0])
    flux_scale = 10.0 ** ((vmag_default - sy_vmag) / 2.5)
    orig_fluxes = {}
    for m in OS.observingModes:
        hk = m["hex"]
        if hk in TL.star_fluxes:
            orig_fluxes[hk] = TL.star_fluxes[hk].copy()
            TL.star_fluxes[hk] = orig_fluxes[hk] * flux_scale
    orig_vmag  = TL.Vmag.copy()
    TL.Vmag[0] = sy_vmag
    mode["RefStar_V_mag"]        = ref_star_vmag
    mode["RefStar_SpectralType"] = ref_star_spectype
    TK = EXOSIMS.Prototypes.TimeKeeping.TimeKeeping(missionLife=5.25)
    TK.allocate_time(months_at_l2 * 30.4375 * u.d)
    fZ = np.repeat(TL.ZodiacalLight.fZ0, 1)
    return TL, OS, TK, fZ, mode, orig_fluxes, orig_vmag


def _restore(TL, orig_fluxes, orig_vmag):
    for hk, orig in orig_fluxes.items():
        TL.star_fluxes[hk] = orig
    TL.Vmag[0] = orig_vmag[0]


def _sweep_one_epoch(OS, TL, TK, fZ, mode, WA_arcsec: float, dMag: float) -> np.ndarray:
    """
    Notebook loop for one (WA, dMag) epoch.
    Returns totTimes in seconds (NaN if outside IWA/OWA).
    """
    WA = np.array([WA_arcsec]) * u.arcsec
    if (WA[0] < mode["IWA"]) or (WA[0] > mode["OWA"]):
        return np.full(len(REF_FRACS), np.nan)

    JEZ      = TL.JEZ0[mode["hex"]] / (WA_arcsec ** 2)
    dMag_arr = np.array([dMag])

    # ── Notebook In[7] loop ──────────────────────────────────────────────
    intTimes = np.zeros(len(REF_FRACS)) * u.d
    for j, ref_frac in enumerate(REF_FRACS):
        mode["TimeonRefStar_tRef_per_tTar"] = ref_frac
        intTimes[j] = OS.calc_intTime(TL, [0], fZ, JEZ, dMag_arr, WA, mode, TK=TK)[0]

    totTimes = intTimes * (1.0 + REF_FRACS)   # notebook: totTimes = intTimes*(1+ref_fracs)
    # ────────────────────────────────────────────────────────────────────

    totTimes_s = totTimes.to(u.s).value
    totTimes_s[~np.isfinite(totTimes_s) | (totTimes_s <= 0)] = np.nan
    return totTimes_s


def _get_ref_vmag(ref_star_name: str | None) -> float:
    if ref_star_name is None:
        return 2.26
    try:
        from astroquery.simbad import Simbad
        s = Simbad()
        s.add_votable_fields("V")
        r = s.query_object(ref_star_name)
        if r is not None and len(r) > 0:
            val = r["V"][0]
            if not hasattr(val, "mask") and val is not None:
                return float(val)
    except Exception:
        pass
    return 2.26


_BPGS_TYPES = ["b3v", "a0v", "a5v", "f5v", "g0v", "g5v", "k0v", "k5v", "m0v", "m5v"]

def _normalize_spectype(simbad_sptype: str) -> str:
    """Map a raw SIMBAD spectral type string to the nearest BPGS grid type."""
    s = simbad_sptype.strip().lower().replace(" ", "")
    if not s:
        return "a0v"
    letter = s[0]
    # Extract numeric subclass (first digit after letter)
    import re
    m = re.search(r'(\d+(?:\.\d+)?)', s)
    subclass = float(m.group(1)) if m else 5.0
    if letter in ("o",):
        return "b3v"
    if letter == "b":
        return "b3v"
    if letter == "a":
        return "a0v" if subclass < 3 else "a5v"
    if letter == "f":
        return "f5v"
    if letter == "g":
        return "g0v" if subclass < 3 else "g5v"
    if letter == "k":
        return "k0v" if subclass < 3 else "k5v"
    if letter == "m":
        return "m0v" if subclass < 3 else "m5v"
    return "a0v"


def _get_ref_spectype(ref_star_name: str | None) -> str:
    """Fetch spectral type of a reference star from SIMBAD, mapped to nearest BPGS type."""
    if ref_star_name is None:
        return "a0v"
    try:
        from astroquery.simbad import Simbad
        s = Simbad()
        s.add_votable_fields("sptype")
        r = s.query_object(ref_star_name)
        if r is not None and len(r) > 0:
            raw = r["SP_TYPE"][0] if "SP_TYPE" in r.colnames else None
            if raw is not None and not (hasattr(raw, "mask") and raw.mask):
                return _normalize_spectype(str(raw))
    except Exception:
        pass
    return "a0v"


def optimise_R(
    pl_name: str,
    band: int,
    engine,
    ref_star_name: str | None,
    months_at_l2: float = 21,
    verbose: bool = True,
    ref_star_spectype: str | None = None,
) -> dict:
    """
    Find R* that minimises median T_wall across all observable epochs.

    Runs the notebook sweep at every observable (WA, dMag) epoch, takes
    the median T_wall across epochs at each R, then:
        R* = REF_FRACS[nanargmin(median_T_wall)]

    T_science and T_wall at R* are returned for use in scheduling.

    Returns dict:
        R_opt              : optimal R*
        T_sci_opt_hrs      : median T_science at R* — use in tint_s
        T_wall_opt_hrs     : median T_wall at R* — use for wall-clock budget
        T_sci_default_hrs  : median T_science at R=0.25 (for comparison)
        T_wall_default_hrs : median T_wall at R=0.25 (for comparison)
        saving_pct         : % T_wall saving vs R=0.25
        n_epochs           : number of observable epochs
        T_wall_median_grid : median T_wall (hrs) at each R in REF_FRACS
        T_sci_median_grid  : median T_science (hrs) at each R in REF_FRACS
        ref_star_vmag, sy_vmag, pl_name, band
    """
    import pandas as pd

    ref_star_vmag = _get_ref_vmag(ref_star_name)
    if ref_star_spectype is not None:
        ref_star_spectype = _normalize_spectype(ref_star_spectype)
    else:
        ref_star_spectype = _get_ref_spectype(ref_star_name)

    if verbose:
        print(f"\n  optimise_R: {pl_name}  band={band}")
        print(f"    ref={ref_star_name or 'none'}  ref_vmag={ref_star_vmag:.2f}  spectype={ref_star_spectype}")

    # Fetch orbit data
    bwl      = {1: 575, 3: 730, 4: 825}
    dmag_col = f"dMag_med_{bwl[band]}NM"
    query = text(f"""
        SELECT o.t, o.WA, o.{dmag_col} AS dMag, s.sy_vmag,
               o.solar_keepout, o.GB_keepout
        FROM Orbits o
        JOIN Planets p ON o.pl_id = p.pl_id
        JOIN Stars   s ON p.st_id = s.st_id
        WHERE o.pl_name = :pl_name AND o.default_orb = 1
        ORDER BY o.t
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"pl_name": pl_name})
    if df.empty:
        raise ValueError(f"No orbit data for '{pl_name}' band {band}")

    sy_vmag  = float(df["sy_vmag"].iloc[0])
    obs_mask = (
        ~df["solar_keepout"].astype(bool)
        & ~df["GB_keepout"].astype(bool)
        & df["WA"].notna() & df["dMag"].notna()
        & np.isfinite(df["WA"].values)
        & np.isfinite(df["dMag"].values)
    )
    df_obs = df[obs_mask].reset_index(drop=True)

    if verbose:
        print(f"    {len(df_obs)} observable / {len(df)} total epochs")

    _fallback = {
        "R_opt": 0.25,
        "T_sci_opt_hrs": np.nan, "T_wall_opt_hrs": np.nan,
        "T_sci_default_hrs": np.nan, "T_wall_default_hrs": np.nan,
        "saving_pct": 0.0, "n_epochs": len(df_obs),
        "T_wall_median_grid": np.full(len(REF_FRACS), np.nan),
        "T_sci_median_grid":  np.full(len(REF_FRACS), np.nan),
        "ref_star_vmag": ref_star_vmag, "ref_star_spectype": ref_star_spectype,
        "sy_vmag": sy_vmag, "pl_name": pl_name, "band": band,
    }

    if len(df_obs) == 0:
        print("    Warning: no observable epochs. Returning R=0.25.")
        return _fallback

    # Set up corgietc once
    TL, OS, TK, fZ, mode, orig_fluxes, orig_vmag = _load_corgietc(
        sy_vmag, ref_star_vmag, band, months_at_l2, ref_star_spectype,
    )

    # Sweep R at every observable epoch
    n = len(df_obs)
    T_wall_all = np.full((n, len(REF_FRACS)), np.nan)

    for i, row in df_obs.iterrows():
        WA_arcsec = float(row["WA"])
        dMag      = float(row["dMag"])
        T_wall_all[i] = _sweep_one_epoch(OS, TL, TK, fZ, mode, WA_arcsec, dMag)
        if verbose and (i % max(1, n // 5) == 0):
            print(f"    epoch {i+1}/{n}  WA={WA_arcsec:.3f}\"  dMag={dMag:.1f}", end="\r")

    _restore(TL, orig_fluxes, orig_vmag)
    if verbose:
        print()

    # T_science = T_wall / (1 + R)
    T_sci_all = T_wall_all / (1.0 + REF_FRACS[np.newaxis, :])

    # Median across epochs
    T_wall_median = np.nanmedian(T_wall_all, axis=0)
    T_sci_median  = np.nanmedian(T_sci_all,  axis=0)

    valid = np.isfinite(T_wall_median) & (T_wall_median > 0)
    if not np.any(valid):
        print("    Warning: all T_wall NaN. Returning R=0.25.")
        return {**_fallback,
                "T_wall_median_grid": T_wall_median / 3600,
                "T_sci_median_grid":  T_sci_median  / 3600}

    # R* = REF_FRACS[nanargmin(totTimes)] — notebook line
    idx_opt      = np.nanargmin(np.where(valid, T_wall_median, np.inf))
    R_opt        = float(REF_FRACS[idx_opt])
    T_wall_opt_s = float(T_wall_median[idx_opt])
    T_sci_opt_s  = float(T_sci_median[idx_opt])

    # Comparison at default R=0.25
    idx_025          = np.argmin(np.abs(REF_FRACS - 0.25))
    T_wall_default_s = float(T_wall_median[idx_025]) if valid[idx_025] else np.nan
    T_sci_default_s  = float(T_sci_median[idx_025])  if valid[idx_025] else np.nan

    saving_pct = (
        100.0 * (T_wall_default_s - T_wall_opt_s) / T_wall_default_s
        if np.isfinite(T_wall_default_s) and T_wall_default_s > 0 else 0.0
    )

    if verbose:
        print(f"    R* = {R_opt:.3f}  "
              f"T_wall = {T_wall_opt_s/3600:.3f} hrs  "
              f"(R=0.25 → {T_wall_default_s/3600:.3f} hrs)  "
              f"saving = {saving_pct:.1f}%")

    return {
        "R_opt":              R_opt,
        "T_sci_opt_hrs":      T_sci_opt_s  / 3600,
        "T_wall_opt_hrs":     T_wall_opt_s / 3600,
        "T_sci_default_hrs":  T_sci_default_s  / 3600 if np.isfinite(T_sci_default_s)  else np.nan,
        "T_wall_default_hrs": T_wall_default_s / 3600 if np.isfinite(T_wall_default_s) else np.nan,
        "saving_pct":         saving_pct,
        "n_epochs":           n,
        "T_wall_median_grid": T_wall_median / 3600,
        "T_sci_median_grid":  T_sci_median  / 3600,
        "ref_star_vmag":      ref_star_vmag,
        "ref_star_spectype":  ref_star_spectype,
        "sy_vmag":            sy_vmag,
        "pl_name":            pl_name,
        "band":               band,
    }


def build_optimal_R_map(
    target_config: list[dict],
    engine,
    months_at_l2: float = 21,
    verbose: bool = True,
) -> tuple[dict[str, float], dict]:
    """
    Run optimise_R for every target. Returns (R_map, all_results).

    target_config: list of dicts with keys:
        target_name, pl_name, band, ref_star_name

    R_map = {target_name: R_opt} — pass directly to run_campaign(R_map=R_map)
    """
    R_map, all_results = {}, {}

    print(f"\n{'='*60}")
    print(f"Optimising R for {len(target_config)} targets")
    print(f"{'='*60}")

    for cfg in target_config:
        name = cfg["target_name"]
        try:
            res = optimise_R(
                pl_name       = cfg["pl_name"],
                band          = cfg.get("band", 1),
                engine        = engine,
                ref_star_name = cfg.get("ref_star_name"),
                months_at_l2  = months_at_l2,
                verbose       = verbose,
            )
            R_map[name]       = res["R_opt"]
            all_results[name] = res
        except Exception as e:
            print(f"  ERROR {name}: {e}. Falling back to R=0.25.")
            R_map[name]       = 0.25
            all_results[name] = {"R_opt": 0.25, "error": str(e)}

    print(f"\n{'='*60}  SUMMARY")
    hdr = f"  {'Target':<25} {'R_opt':>6} {'T_wall_opt':>11} {'T_wall_R025':>12} {'Saving':>8}"
    print(hdr); print("  " + "-"*(len(hdr)-2))
    for name, res in all_results.items():
        if "error" in res:
            print(f"  {name:<25} ERROR"); continue
        print(f"  {name:<25} {res['R_opt']:>6.3f}"
              f" {res['T_wall_opt_hrs']:>10.3f}h"
              f" {res['T_wall_default_hrs']:>11.3f}h"
              f" {res['saving_pct']:>7.1f}%")
    print(f"\n  R_map = {R_map}")
    return R_map, all_results


def build_R_map_from_ref_results(
    ref_results: dict,
    planet_data: dict,
    engine,
    band_map: dict = None,
    pl_name_map: dict = None,
    months_at_l2: float = 21,
    verbose: bool = True,
) -> tuple[dict, dict]:
    """
    Build R_map from select_ref_star() output directly.

    Most convenient entry point — takes ref_results from run_campaign() Step 3.
    Pass returned R_map to run_campaign(R_map=R_map, engine=engine).
    """
    target_config = []
    for target_name, result in ref_results.items():
        ref_star_name = None
        for window in result.get("observable_windows", []):
            if window.get("best_ref") is not None:
                ref_star_name = window["best_ref"]["reference_star"]
                break
        if ref_star_name is None:
            continue
        if pl_name_map and target_name in pl_name_map:
            pl_name = pl_name_map[target_name]
        else:
            keys = [k for k in planet_data if k[0] == target_name]
            if not keys:
                continue
            pl_name = f"{keys[0][0]} {keys[0][1]}"
        target_config.append({
            "target_name": target_name, "pl_name": pl_name,
            "band": (band_map or {}).get(target_name, 1),
            "ref_star_name": ref_star_name,
        })
    if not target_config:
        print("No valid targets found.")
        return {}, {}
    return build_optimal_R_map(target_config, engine, months_at_l2, verbose)


def plot_R_sweep(result: dict, title: str = "") -> None:
    """Plot T_science and T_wall vs R — matches notebook figure."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(REF_FRACS, result["T_sci_median_grid"],
              color="steelblue", lw=2, label="Science Time")
    ax.loglog(REF_FRACS, result["T_wall_median_grid"],
              color="darkorange", lw=2, label=r"Total Time $(1+R)\times T_{sci}$")
    if np.isfinite(result["R_opt"]):
        ax.axvline(result["R_opt"], color="green", lw=1.5, ls="--",
                   label=f"R* = {result['R_opt']:.3f}  "
                         f"(saves {result['saving_pct']:.1f}%)")
    ax.axvline(0.25, color="red", lw=1.2, ls=":", label="Default R=0.25")
    ax.set_xlabel("Fractional Time on Reference Star  R = $t_r / t$")
    ax.set_ylabel("Time (hours)")
    ax.set_title(title or (
        f"{result['pl_name']}  band={result['band']}  "
        f"ref vmag={result['ref_star_vmag']:.2f}\n"
        f"R* = {result['R_opt']:.3f}  "
        f"T_wall = {result['T_wall_opt_hrs']:.3f} hrs  "
        f"(default: {result['T_wall_default_hrs']:.3f} hrs)"
    ))
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show(block=False)

# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    # Example: pass per-target R values
    # R_map = {"47 UMa": 0.30, "eps Eri": 0.20}   ← different R per target
    # run_campaign(R_map=R_map)

    run_campaign()