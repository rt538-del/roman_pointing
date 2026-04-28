import os
import json
import copy
import shutil
import EXOSIMS.Prototypes.TargetList
import EXOSIMS.Prototypes.TimeKeeping
from sqlalchemy import text
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import scipy.optimize as optimize
import time
from roman_pointing.roman_pointing import calcRomanAngles, getL2Positions
import astropy.units as u
from astropy.time import Time
from astroquery.simbad import Simbad
from astropy.coordinates import (
    SkyCoord,
    Distance,
    get_body_barycentric,
    BarycentricMeanEcliptic,
)
from roman_pointing.roman_observability import (
    get_target_coords,
    compute_keepout,
)
from roman_pointing.Reference_Star_Selection_Tool import (
    load_catalog,
    select_ref_star,
)

#Get all of the information in order 

# These targets and priorities were taken from the TPC Master Targets Spreadsheet - January 2026
# Each program is a single dict: target (priority, band, contrast)
# Priority tiers: 1 = must do, 2 = high / enabling, 3 = medium, 4 = lower
# "enabling" observations are required prerequisites — short ref star snapshots
# to confirm no companions before subsequent science observations can proceed.
# They are scheduled at the same priority as "high" (tier 2), not as optional targets.
# band: 1=NFB Band1, 3=SPEC Band3, 4=WFOV Band4s
# contrast: "high" (deep) or "med" (shallow) — from "raw speckle field request" column

HLC_B1 = {
    # target:           (priority, band, contrast)
    # --- must do ---
    "ups And":          (1, 1, "high"),
    "HIP 71618":        (1, 1, "high"),
    "* eps Eri":        (1, 1, "high"),
    "* bet Leo":        (1, 1, "high"),
    "* w Her":          (1, 1, "high"),
    "* eta Crv":        (1, 1, "med"),   # raw speckle 1e-7
    "HR 4796A":         (1, 1, "med"),   # raw speckle 1e-5
    # --- medium ---
    "14 Her":           (3, 1, "high"),
    "* tau Cet":        (3, 1, "high"),
    "* bet Vir":        (3, 1, "high"),
    "* bet UMa":        (3, 1, "high"),
    # --- lower ---
    "* del UMa":        (4, 1, "high"),
    "* zet Lep":        (4, 1, "high"),
    "HD 10647":         (4, 1, "high"),
    "HD 172555":        (4, 1, "high"),
    # --- enabling (companion check snapshots, raw speckle 1e-7) ---
    "eta UMa":          (2, 1, "med"),
    "eps UMa":          (2, 1, "med"),
    "alf Cyg":          (2, 1, "med"),
    "alf Col":          (2, 1, "med"),
    "alf Cep":          (2, 1, "med"),
    "gam Peg":          (2, 1, "med"),
    "eta Cen":          (2, 1, "med"),
    "bet Lib":          (2, 1, "med"),
    "del Cas":          (2, 1, "med"),
    "eps CMa":          (2, 1, "med"),
    "alf Ara":          (2, 1, "med"),
    "bet Ori":          (2, 1, "med"),
    "eps Ori":          (2, 1, "med"),
    "bet Tau":          (2, 1, "med"),
    "bet TrA":          (2, 1, "med"),
    "zet Oph":          (2, 1, "med"),
    "eta Tau":          (2, 1, "med"),
    "kap Ori":          (2, 1, "med"),
    "alf Gru":          (2, 1, "med"),
    "bet CMi":          (2, 1, "med"),
    "* alf Aql":        (2, 1, "med"),
}

SPEC = {
    "HIP 71618":        (1, 3, "high"),  # must do
    "HIP 54515":        (3, 3, "high"),
    "HR 8799":          (3, 3, "high"),
}

WFOV = {
    # --- must do ---
    "HD 109573":        (1, 4, "med"),   # HR 4796A,  shallow B4
    "HD 109085":        (1, 4, "med"),   # eta Crv,   shallow B4 or B1
    # --- high ---
    "HD 172167":        (2, 4, "high"),  # alf Lyr,   deep B1+B4
    "HD 216956":        (2, 4, "high"),  # alf PsA,   deep B1+B4
    "HD 102647":        (2, 4, "high"),  # bet Leo,   deep B1+B4
    "HD 14055":         (2, 4, "high"),  # gam Tri,   deep B1+B4
    # --- medium ---
    "HD 222439":        (3, 4, "med"),   # kap And,   shallow B4
    "HD 50571":         (3, 4, "med"),   # HR 2562,   shallow B4
    "HD 218396":        (3, 4, "med"),   # HR 8799,   shallow B4
    # --- lower ---
    "HD 106036":        (4, 4, "med"),   # shallow B1 or B4
    "HD 106797":        (4, 4, "med"),
    "HD 113457":        (4, 4, "med"),
    "HD 10647":         (4, 4, "high"),  # deep B4
    "HD163296":         (4, 4, "med"),   # shallow B1+B4
    "HD191089":         (4, 4, "med"),
}

# Planet detection data (.npy) — detection probability over time and integration time
# Original (real) files live in for_rifah/for_rifah/
# Faked/tweaked files (modified to be in FoV or brighter) live in for_rifah/for_rifah_fake/
PLANET_DATA_DIR      = Path(__file__).resolve().parents[3] / "for_rifah" / "for_rifah"
PLANET_DATA_DIR_FAKE = Path(__file__).resolve().parents[3] / "for_rifah" / "for_rifah_fake"
PLANET_DATA_FILES = {
    # 47 UMa b — faked into FoV
    ('47 UMa', 'b', 1): PLANET_DATA_DIR_FAKE / 'for_rifah_47_UMa_b_band1.npy',
    ('47 UMa', 'b', 3): PLANET_DATA_DIR_FAKE / 'for_rifah_47_UMa_b_band3.npy',
    ('47 UMa', 'b', 4): PLANET_DATA_DIR_FAKE / 'for_rifah_47_UMa_b_band4.npy',
    # 47 UMa c — original
    ('47 UMa', 'c', 1): PLANET_DATA_DIR / 'for_rifah_47_UMa_c_band1.npy',
    ('47 UMa', 'c', 3): PLANET_DATA_DIR / 'for_rifah_47_UMa_c_band3.npy',
    ('47 UMa', 'c', 4): PLANET_DATA_DIR / 'for_rifah_47_UMa_c_band4.npy',
    # 47 UMa d — faked into FoV
    ('47 UMa', 'd', 1): PLANET_DATA_DIR_FAKE / 'for_rifah_47_UMa_d_band1.npy',
    ('47 UMa', 'd', 3): PLANET_DATA_DIR_FAKE / 'for_rifah_47_UMa_d_band3.npy',
    ('47 UMa', 'd', 4): PLANET_DATA_DIR_FAKE / 'for_rifah_47_UMa_d_band4.npy',
    # 55 Cnc d — tweaked brighter
    ('55 Cnc', 'd', 1): PLANET_DATA_DIR_FAKE / 'for_rifah_55_Cnc_d_band1.npy',
    ('55 Cnc', 'd', 3): PLANET_DATA_DIR_FAKE / 'for_rifah_55_Cnc_d_band3.npy',
    ('55 Cnc', 'd', 4): PLANET_DATA_DIR_FAKE / 'for_rifah_55_Cnc_d_band4.npy',
    # HD 114783 — planet c, original
    ('HD 114783', 'c', 1): PLANET_DATA_DIR / 'for_rifah_HD_114783_c_band1.npy',
    ('HD 114783', 'c', 3): PLANET_DATA_DIR / 'for_rifah_HD_114783_c_band3.npy',
    ('HD 114783', 'c', 4): PLANET_DATA_DIR / 'for_rifah_HD_114783_c_band4.npy',
    # HD 134987 c — tweaked brighter
    ('HD 134987', 'c', 1): PLANET_DATA_DIR_FAKE / 'for_rifah_HD_134987_c_band1.npy',
    ('HD 134987', 'c', 3): PLANET_DATA_DIR_FAKE / 'for_rifah_HD_134987_c_band3.npy',
    ('HD 134987', 'c', 4): PLANET_DATA_DIR_FAKE / 'for_rifah_HD_134987_c_band4.npy',
    # HD 154345 — planet b, original
    ('HD 154345', 'b', 1): PLANET_DATA_DIR / 'for_rifah_HD_154345_b_band1.npy',
    ('HD 154345', 'b', 3): PLANET_DATA_DIR / 'for_rifah_HD_154345_b_band3.npy',
    ('HD 154345', 'b', 4): PLANET_DATA_DIR / 'for_rifah_HD_154345_b_band4.npy',
    # HD 160691 c — tweaked brighter
    ('HD 160691', 'c', 1): PLANET_DATA_DIR_FAKE / 'for_rifah_HD_160691_c_band1.npy',
    ('HD 160691', 'c', 3): PLANET_DATA_DIR_FAKE / 'for_rifah_HD_160691_c_band3.npy',
    ('HD 160691', 'c', 4): PLANET_DATA_DIR_FAKE / 'for_rifah_HD_160691_c_band4.npy',
    # HD 190360 — planet b, original
    ('HD 190360', 'b', 1): PLANET_DATA_DIR / 'for_rifah_HD_190360_b_band1.npy',
    ('HD 190360', 'b', 3): PLANET_DATA_DIR / 'for_rifah_HD_190360_b_band3.npy',
    ('HD 190360', 'b', 4): PLANET_DATA_DIR / 'for_rifah_HD_190360_b_band4.npy',
    # HD 192310 — planet c, original
    ('HD 192310', 'c', 1): PLANET_DATA_DIR / 'for_rifah_HD_192310_c_band1.npy',
    ('HD 192310', 'c', 3): PLANET_DATA_DIR / 'for_rifah_HD_192310_c_band3.npy',
    ('HD 192310', 'c', 4): PLANET_DATA_DIR / 'for_rifah_HD_192310_c_band4.npy',
    # HD 217107 — planet c, original
    ('HD 217107', 'c', 1): PLANET_DATA_DIR / 'for_rifah_HD_217107_c_band1.npy',
    ('HD 217107', 'c', 3): PLANET_DATA_DIR / 'for_rifah_HD_217107_c_band3.npy',
    ('HD 217107', 'c', 4): PLANET_DATA_DIR / 'for_rifah_HD_217107_c_band4.npy',
    # HD 87883 b — tweaked brighter
    ('HD 87883', 'b', 1): PLANET_DATA_DIR_FAKE / 'for_rifah_HD_87883_b_band1.npy',
    ('HD 87883', 'b', 3): PLANET_DATA_DIR_FAKE / 'for_rifah_HD_87883_b_band3.npy',
    ('HD 87883', 'b', 4): PLANET_DATA_DIR_FAKE / 'for_rifah_HD_87883_b_band4.npy',
    # eps Eri — planet b, original (no band 3 file exists)
    ('* eps Eri', 'b', 1): PLANET_DATA_DIR / 'for_rifah_eps_Eri_b_band1.npy',
    ('* eps Eri', 'b', 4): PLANET_DATA_DIR / 'for_rifah_eps_Eri_b_band4.npy',
    # pi Men b — tweaked brighter
    ('* pi. Men', 'b', 1): PLANET_DATA_DIR_FAKE / 'for_rifah_pi_Men_b_band1.npy',
    ('* pi. Men', 'b', 3): PLANET_DATA_DIR_FAKE / 'for_rifah_pi_Men_b_band3.npy',
    ('* pi. Men', 'b', 4): PLANET_DATA_DIR_FAKE / 'for_rifah_pi_Men_b_band4.npy',
    # ups And — planet d
    ('ups And', 'd', 1): PLANET_DATA_DIR / 'for_rifah_ups_And_d_band1.npy',
    ('ups And', 'd', 3): PLANET_DATA_DIR / 'for_rifah_ups_And_d_band3.npy',
    ('ups And', 'd', 4): PLANET_DATA_DIR / 'for_rifah_ups_And_d_band4.npy',
}

# corgietc setup ===========================================================================


os.environ["CGI_NOISE_DATA_DIR"] = r"C:\Users\tasni\roman_repos\cgi_noise\cgi_noise\data"
os.environ["CORGIETC_DATA_DIR"]  = r"C:\Users\tasni\roman_repos\corgietc\corgietc\data"

# band number -> wavelength in nm (for matching observing mode and dMag column)
BAND_WAVELENGTH = {1: 575, 3: 730, 4: 825}

def _setup_corgietc():
    """Build TL and OS once. Returns (TL, OS, vmag_ref)."""
    cache_dir = os.path.expanduser(r"~\.EXOSIMS\cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    scriptfile = os.path.join(os.environ["CORGIETC_DATA_DIR"], "scripts", "CGI_Noise.json")
    with open(scriptfile) as f:
        specs = json.loads(f.read())

    TL = EXOSIMS.Prototypes.TargetList.TargetList(**copy.deepcopy(specs))
    OS = TL.OpticalSystem
    vmag_ref = float(TL.Vmag[0])
    return TL, OS, vmag_ref


def _get_mode_for_band(OS, band):
    """Return the observing mode whose wavelength matches the given band number."""
    target_nm = BAND_WAVELENGTH[band]
    for mode in OS.observingModes:
        if abs(mode["lam"].to_value(u.nm) - target_nm) < 20:
            return mode
    raise ValueError(f"No observing mode found near {target_nm} nm for band {band}")


def get_planet_orbit_data(engine, pl_name, band):
    """Query Orbits + Stars for a planet and return a DataFrame with t, WA, dMag, sy_vmag.

    Args:
        engine: SQLAlchemy engine connected to plandb
        pl_name (str): Planet name as stored in Orbits table (e.g. '47 UMa b')
        band (int): Band number — 1 (575 nm), 3 (730 nm), or 4 (825 nm)

    Returns:
        pandas.DataFrame with columns: t, WA, dMag, sy_vmag
    """
    wav = BAND_WAVELENGTH[band]
    dmag_col = f"dMag_med_{wav}NM"

    query = text(f"""
        SELECT o.t, o.WA, o.{dmag_col} AS dMag, s.sy_vmag
        FROM Orbits o
        JOIN Planets p ON o.pl_id = p.pl_id
        JOIN Stars s ON p.st_id = s.st_id
        WHERE o.pl_name = :pl_name
          AND o.default_orb = 1
        ORDER BY o.t
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"pl_name": pl_name})

    return df


def compute_integration_times(
    pl_name,
    band,
    engine,
    months_at_l2=21,
    ref_star_vmag=2.26,
    ref_star_spectype="a0v",
    tref_per_ttar=0.25,
):
    """Compute integration time at each epoch for a planet using corgietc.

    Pulls WA, dMag, and sy_vmag from the database, scales the star flux,
    applies reference star settings, and calls calc_intTime for each time step.

    Args:
        pl_name (str): Planet name as in Orbits table (e.g. '47 UMa b')
        band (int): Band number — 1, 3, or 4
        engine: SQLAlchemy engine connected to plandb
        months_at_l2 (float): Months into mission for detector degradation (default 21)
        ref_star_vmag (float): V magnitude of the reference star (default 2.26)
        ref_star_spectype (str): Spectral type of reference star (default 'a0v')
        tref_per_ttar (float): Ratio of time on reference star to science target (default 0.25)

    Returns:
        dict with keys:
            't'             : decimal year array
            'WA_arcsec'     : working angle array (arcsec)
            'dMag'          : delta magnitude array
            'intTime_hours' : integration time array (hours), NaN where not observable
    """
    TL, OS, vmag_ref = _setup_corgietc()
    mode = _get_mode_for_band(OS, band)

    # apply reference star settings to the mode
    mode["RefStar_V_mag"]              = ref_star_vmag
    mode["RefStar_SpectralType"]       = ref_star_spectype
    mode["TimeonRefStar_tRef_per_tTar"] = tref_per_ttar

    # TimeKeeping object at the specified mission elapsed time
    TK = EXOSIMS.Prototypes.TimeKeeping.TimeKeeping(missionLife=5.25)
    TK.allocate_time(months_at_l2 * 30.4375 * u.d)

    fZ = np.repeat(TL.ZodiacalLight.fZ0, 1)

    # pull orbit data from DB
    df = get_planet_orbit_data(engine, pl_name, band)
    if df.empty:
        raise ValueError(f"No orbit data found for '{pl_name}' band {band}")

    sy_vmag = float(df["sy_vmag"].iloc[0])

    # scale star flux to match this target's vmag
    flux_scale = 10.0 ** ((vmag_ref - sy_vmag) / 2.5)
    original_fluxes = {}
    for m in OS.observingModes:
        hk = m["hex"]
        if hk in TL.star_fluxes:
            original_fluxes[hk] = TL.star_fluxes[hk].copy()
            TL.star_fluxes[hk] = original_fluxes[hk] * flux_scale
    original_vmag = TL.Vmag.copy()
    TL.Vmag[0]    = sy_vmag

    intTimes_hours = np.full(len(df), np.nan)

    for i, row in df.iterrows():
        WA   = np.array([row["WA"]]) * u.arcsec
        dMag = np.array([row["dMag"]])

        if np.isnan(row["dMag"]) or np.isnan(row["WA"]):
            continue
        if (WA[0] < mode["IWA"]) or (WA[0] > mode["OWA"]):
            continue

        # JEZ scaled by physical separation in AU — use WA converted via distance
        # For now use WA (arcsec) as a proxy; replace with s_AU column if available
        JEZ = TL.JEZ0[mode["hex"]] / (row["WA"] ** 2)

        t = OS.calc_intTime(TL, [0], fZ, JEZ, dMag, WA, mode, TK=TK)
        t_hr = t[0].to(u.hour).value
        if not np.isnan(t_hr) and t_hr > 0:
            intTimes_hours[i] = t_hr

    # restore original fluxes and vmag
    for hk, orig in original_fluxes.items():
        TL.star_fluxes[hk] = orig
    TL.Vmag[0] = original_vmag[0]

    return {
        "t":             df["t"].values,
        "WA_arcsec":     df["WA"].values,
        "dMag":          df["dMag"].values,
        "intTime_hours": intTimes_hours,
    }