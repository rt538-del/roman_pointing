import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import milp, LinearConstraint, Bounds
from roman_pointing.roman_observability import (
    get_target_coords,
    compute_keepout,
    compute_visibility_fraction,
    print_visibility_windows,
)
from roman_pointing.Reference_Star_Selection_Tool import (
    load_catalog,
    select_ref_star,
)


#SCHEDULING AND MAXIMIZNG TARGETS AND REF STARS
#GOAL SEE AS MUCH STUFF AS POSSIBLE AND MAKE SURE WE CAN ACTUALLY SEE IT

#FIRST WE ORGANIZ THE INFORMATION WE NEED
# These targets and priorities were taken from the TPC Master Targets Spreadsheet - January 2026
# Each program is a single dict: target -> (priority, band, contrast)
# Priority tiers: 1 = must do, 2 = high / enabling, 3 = medium, 4 = lower
# NOTE: "enabling" observations are required prerequisites — short ref star snapshots
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

# Derived lookups — single source of truth from the program dicts above
ALL_PROGRAMS        = {"HLC_B1": HLC_B1, "SPEC": SPEC, "WFOV": WFOV}
ALL_TARGET_PRIORITY = {t: v[0] for prog in ALL_PROGRAMS.values() for t, v in prog.items()}
all_targets         = list({t for prog in ALL_PROGRAMS.values() for t in prog})

ANALYSIS_START      = "2026-12-01T00:00:00"
ANALYSIS_DAYS       = 365
MISSION_BUDGET_DAYS = 60



#WITHIN THE SOLAR KEEPOUT
#WHEN ARE THE TARGETS ABOVE AVAILIBLE
#create a matrix with thee avalible targets and dates they are availnle

coords_dict = get_target_coords(all_targets)

ts, keepout, solar_angles = compute_keepout(
    coords_dict,
    ANALYSIS_START,
    ANALYSIS_DAYS,
    time_step=1,
)

# availability matrix: target name -> bool array (True = observable)
visibility = compute_visibility_fraction(keepout)


#WHAT REFERENCE STARS ARE AVAIBLE FOR THE TARGETS

catalog = load_catalog()

# Match ref stars for each target using the correct band and contrast per program,
# iterated in priority order (must-do first)
ref_star_results = {}

for program_params in ALL_PROGRAMS.values():
    for target in sorted(program_params, key=lambda t: ALL_TARGET_PRIORITY.get(t, 99)):
        _, band, contrast = program_params[target]
        result = select_ref_star(
            sci_name=target,
            analysis_start=ANALYSIS_START,
            analysis_days=ANALYSIS_DAYS,
            band=band,
            contrast=contrast,
            catalog=catalog,
            sort_mode="valid_days",
        )
        ref_star_results[(target, band)] = result


# SCHEDULING
# Standard ADI+RDI observing sequence:
#   1. Initial HOWFSC data collection
#   2. Dig dark hole on reference star
#   3. ADI sequence on target: Roll A - Roll B - Roll A - Roll B  (2*N target visits)
#   4. Second reference star visit
#   5. Second ADI sequence (repeat)
#   6. Third reference star visit
#   7. Final HOWFSC touch-up

# Standard sequence parameters (defaults from TPC guidance) 
N            = 4        # number of target visits prior to mid-cycle reference visit
                        # (standard Roll A-B-A-B sequence = 4 total target visits)
R            = 1/4      # ratio of reference to target visit duration (80/20 split)
N_slew       = 5        # slews: 1 initial to ref + 4 target<->ref transitions
N_rollchange = 2*(N-1)  # = 6 roll changes for standard N=4 sequence
T_slew       = 0.5      # hours; assumes 90-deg slew at 30 min per slew
T_rollchange = 1/6      # hours; 10 min per +/-15-deg roll change (30 deg movement)
T_HOWFSC     = 1.0      # hours; initial + final HOWFSC combined

# Per-target wall clock equations 
# Integration time comes from Corgi-ETC (optimistic / conservative)
# T_targetvisit  = t_etc / (2*N)
# T_referencevisit = R * T_targetvisit
# T_obs (science + ref) = t_etc * (1 + (3*R) / (2*N))
# T_wallclock = T_obs + N_slew*T_slew + N_rollchange*T_rollchange + T_HOWFSC

def load_planet_visibility(filepath):
    """Load reflected-light planet detection data and find optimal observing epochs.

    The .npy files give integration time and detection probability as a function
    of time (orbital phase). Because reflected light is highly phase-dependent,
    the best time to observe within each visibility window must be selected to
    maximize detection probability and minimize integration time.

    Args:
        filepath (str): Path to the .npy file (e.g. for_rifah_eps_Eri_b_band1.npy).

    Returns:
        dict with keys:
            decimal_year     - full time axis (array)
            t_opt            - optimistic integration time median in hours (array)
            t_con            - conservative integration time median in hours (array)
            det_probability  - detection probability at each epoch (array)
            observable       - True when not in solar or GB keepout (bool array)
            best_epoch_idx   - index of best observable epoch (highest det_prob)
            best_year        - decimal year of best observable epoch
            best_det_prob    - detection probability at best epoch
            best_t_opt       - optimistic integration time at best epoch (hours)
            best_t_con       - conservative integration time at best epoch (hours)
    """
    data = np.load(filepath, allow_pickle=True)
    d = data[()]

    decimal_year   = np.array(d["decimal_year"])
    t_opt          = np.array(d["integration_time_hours_opt"]["median"])
    t_con          = np.array(d["integration_time_hours_con"]["median"])
    det_prob       = np.array(d["det_probability"])
    solar_ko       = np.array(d["solar_keepout"], dtype=bool)
    gb_ko          = np.array(d["GB_keepout"],    dtype=bool)
    observable     = ~solar_ko & ~gb_ko

    # Find best epoch: highest detection probability while observable
    obs_det_prob = np.where(observable, det_prob, -np.inf)
    best_idx     = int(np.argmax(obs_det_prob))

    return {
        "decimal_year":    decimal_year,
        "t_opt":           t_opt,
        "t_con":           t_con,
        "det_probability": det_prob,
        "observable":      observable,
        "best_epoch_idx":  best_idx,
        "best_year":       decimal_year[best_idx],
        "best_det_prob":   det_prob[best_idx],
        "best_t_opt":      t_opt[best_idx],
        "best_t_con":      t_con[best_idx],
    }


def compute_wallclock(t_etc, N=N, R=R, N_slew=N_slew,
                      N_rollchange=N_rollchange, T_slew=T_slew,
                      T_rollchange=T_rollchange, T_HOWFSC=T_HOWFSC):
    """Compute total wall clock time (hours) for one ADI+RDI observing sequence.

    Args:
        t_etc (float): Total integration time from Corgi-ETC in hours
                       (optimistic or conservative).
        N (int): Number of target visits per half-sequence. Default 4.
        R (float): Reference-to-target visit duration ratio. Default 1/4.
        N_slew (int): Number of slews in the sequence. Default 5.
        N_rollchange (int): Number of roll changes. Default 2*(N-1).
        T_slew (float): Time per slew in hours. Default 0.5 (30 min).
        T_rollchange (float): Time per roll change in hours. Default 1/6 (10 min).
        T_HOWFSC (float): Total HOWFSC time in hours. Default 1.0.

    Returns:
        dict with keys:
            t_targetvisit   - time per individual target visit (hrs)
            t_referencevisit - time per reference star visit (hrs)
            t_obs           - science + reference time, no overheads (hrs)
            t_wallclock     - full wall clock including all overheads (hrs)
    """
    MISSION_BUDGET_HOURS = MISSION_BUDGET_DAYS * 24

    t_targetvisit    = t_etc / (2 * N)
    t_referencevisit = R * t_targetvisit
    t_obs            = t_etc * (1 + (3 * R) / (2 * N))
    t_wallclock      = t_obs + N_slew * T_slew + N_rollchange * T_rollchange + T_HOWFSC

    fits_in_budget = t_wallclock <= MISSION_BUDGET_HOURS
    if not fits_in_budget:
        print(f"  WARNING: wallclock {t_wallclock:.1f} hrs exceeds mission budget "
              f"({MISSION_BUDGET_HOURS:.0f} hrs). Target cannot be scheduled.")

    return {
        "t_targetvisit":    t_targetvisit,
        "t_referencevisit": t_referencevisit,
        "t_obs":            t_obs,
        "t_wallclock":      t_wallclock,
        "fits_in_budget":   fits_in_budget,
    }




# --- Planet visibility data files ---
# Maps (target, planet, band) -> path to reflected-light detection .npy file from planet tool.
# Add entries here as more files become available for other targets.
PLANET_DATA_DIR = Path(__file__).resolve().parents[3] / "for_rifah" / "for_rifah"
PLANET_DATA_FILES = {
    ('47 UMa', 'b', 1): PLANET_DATA_DIR / 'for_rifah_47_UMa_b_band1.npy',
    ('47 UMa', 'b', 3): PLANET_DATA_DIR / 'for_rifah_47_UMa_b_band3.npy',
    ('47 UMa', 'b', 4): PLANET_DATA_DIR / 'for_rifah_47_UMa_b_band4.npy',
    ('47 UMa', 'c', 1): PLANET_DATA_DIR / 'for_rifah_47_UMa_c_band1.npy',
    ('47 UMa', 'c', 3): PLANET_DATA_DIR / 'for_rifah_47_UMa_c_band3.npy',
    ('47 UMa', 'c', 4): PLANET_DATA_DIR / 'for_rifah_47_UMa_c_band4.npy',
    ('47 UMa', 'd', 1): PLANET_DATA_DIR / 'for_rifah_47_UMa_d_band1.npy',
    ('47 UMa', 'd', 3): PLANET_DATA_DIR / 'for_rifah_47_UMa_d_band3.npy',
    ('47 UMa', 'd', 4): PLANET_DATA_DIR / 'for_rifah_47_UMa_d_band4.npy',
    ('55 Cnc', 'd', 1): PLANET_DATA_DIR / 'for_rifah_55_Cnc_d_band1.npy',
    ('55 Cnc', 'd', 3): PLANET_DATA_DIR / 'for_rifah_55_Cnc_d_band3.npy',
    ('55 Cnc', 'd', 4): PLANET_DATA_DIR / 'for_rifah_55_Cnc_d_band4.npy',
    ('HD 114783', 'c', 1): PLANET_DATA_DIR / 'for_rifah_HD_114783_c_band1.npy',
    ('HD 114783', 'c', 3): PLANET_DATA_DIR / 'for_rifah_HD_114783_c_band3.npy',
    ('HD 114783', 'c', 4): PLANET_DATA_DIR / 'for_rifah_HD_114783_c_band4.npy',
    ('HD 134987', 'c', 1): PLANET_DATA_DIR / 'for_rifah_HD_134987_c_band1.npy',
    ('HD 134987', 'c', 3): PLANET_DATA_DIR / 'for_rifah_HD_134987_c_band3.npy',
    ('HD 134987', 'c', 4): PLANET_DATA_DIR / 'for_rifah_HD_134987_c_band4.npy',
    ('HD 154345', 'b', 1): PLANET_DATA_DIR / 'for_rifah_HD_154345_b_band1.npy',
    ('HD 154345', 'b', 3): PLANET_DATA_DIR / 'for_rifah_HD_154345_b_band3.npy',
    ('HD 154345', 'b', 4): PLANET_DATA_DIR / 'for_rifah_HD_154345_b_band4.npy',
    ('HD 160691', 'c', 1): PLANET_DATA_DIR / 'for_rifah_HD_160691_c_band1.npy',
    ('HD 160691', 'c', 3): PLANET_DATA_DIR / 'for_rifah_HD_160691_c_band3.npy',
    ('HD 160691', 'c', 4): PLANET_DATA_DIR / 'for_rifah_HD_160691_c_band4.npy',
    ('HD 190360', 'b', 1): PLANET_DATA_DIR / 'for_rifah_HD_190360_b_band1.npy',
    ('HD 190360', 'b', 3): PLANET_DATA_DIR / 'for_rifah_HD_190360_b_band3.npy',
    ('HD 190360', 'b', 4): PLANET_DATA_DIR / 'for_rifah_HD_190360_b_band4.npy',
    ('HD 192310', 'c', 1): PLANET_DATA_DIR / 'for_rifah_HD_192310_c_band1.npy',
    ('HD 192310', 'c', 3): PLANET_DATA_DIR / 'for_rifah_HD_192310_c_band3.npy',
    ('HD 192310', 'c', 4): PLANET_DATA_DIR / 'for_rifah_HD_192310_c_band4.npy',
    ('HD 217107', 'c', 1): PLANET_DATA_DIR / 'for_rifah_HD_217107_c_band1.npy',
    ('HD 217107', 'c', 3): PLANET_DATA_DIR / 'for_rifah_HD_217107_c_band3.npy',
    ('HD 217107', 'c', 4): PLANET_DATA_DIR / 'for_rifah_HD_217107_c_band4.npy',
    ('HD 87883', 'b', 1): PLANET_DATA_DIR / 'for_rifah_HD_87883_b_band1.npy',
    ('HD 87883', 'b', 3): PLANET_DATA_DIR / 'for_rifah_HD_87883_b_band3.npy',
    ('HD 87883', 'b', 4): PLANET_DATA_DIR / 'for_rifah_HD_87883_b_band4.npy',
    ('* eps Eri', 'b', 1): PLANET_DATA_DIR / 'for_rifah_eps_Eri_b_band1.npy',
    ('* eps Eri', 'b', 4): PLANET_DATA_DIR / 'for_rifah_eps_Eri_b_band4.npy',
    ('pi Men', 'b', 1): PLANET_DATA_DIR / 'for_rifah_pi_Men_b_band1.npy',
    ('pi Men', 'b', 3): PLANET_DATA_DIR / 'for_rifah_pi_Men_b_band3.npy',
    ('pi Men', 'b', 4): PLANET_DATA_DIR / 'for_rifah_pi_Men_b_band4.npy',
    ('ups And', 'd', 1): PLANET_DATA_DIR / 'for_rifah_ups_And_d_band1.npy',
    ('ups And', 'd', 3): PLANET_DATA_DIR / 'for_rifah_ups_And_d_band3.npy',
    ('ups And', 'd', 4): PLANET_DATA_DIR / 'for_rifah_ups_And_d_band4.npy',
}

# Load all available planet data and compute wallclock at the best observable epoch
planet_timing = {}
for (target, planet, band), filepath in PLANET_DATA_FILES.items():
    pdata = load_planet_visibility(filepath)
    key = (target, band)
    if key not in planet_timing or pdata["best_det_prob"] > planet_timing[key]["best_det_prob"]:
        planet_timing[key] = {
            "planet_data":   pdata,
            "planet":        planet,
            "opt":           compute_wallclock(pdata["best_t_opt"]),
            "con":           compute_wallclock(pdata["best_t_con"]),
            "best_year":     pdata["best_year"],
            "best_det_prob": pdata["best_det_prob"],
        }

# Targets without planet data files will use Corgi-ETC integration times directly
# when those become available. For now, priority is used as a completeness proxy.

# Priority -> proxy completeness score (used when no .npy file is available)
PRIORITY_COMPLETENESS_PROXY = {1: 0.80, 2: 0.60, 3: 0.40, 4: 0.20}

# Conservative wallclock multiplier from notes (T_wallclock_conservative = 1.15 * T_wallclock)
CONSERVATIVE_MULTIPLIER = 1.15


def build_campaign_table(programs, keepout, ts, planet_timing, all_target_priority):
    """Build a flat table of per-target scheduling inputs for the optimizer.

    For each target assembles:
      - t_wc_opt / t_wc_con : optimistic and conservative wall clock (hours)
      - completeness         : detection probability if .npy available, else priority proxy
      - obs_window_hrs       : total observable hours within the analysis period
      - priority             : TPC priority tier
    """
    time_step_days = (ts[1].mjd - ts[0].mjd) if len(ts) > 1 else 1.0
    rows = []

    for prog_name, prog in programs.items():
        for target, (priority, band, contrast) in prog.items():
            key = (target, band)

            if key in planet_timing:
                t_opt = planet_timing[key]["opt"]["t_wallclock"]
                t_con = planet_timing[key]["con"]["t_wallclock"]
                completeness = planet_timing[key]["best_det_prob"]
            else:
                t_opt = None
                t_con = None
                completeness = PRIORITY_COMPLETENESS_PROXY.get(priority, 0.2)

            if target in keepout:
                obs_days = float(np.sum(keepout[target])) * time_step_days
            else:
                obs_days = 0.0

            rows.append({
                "target":         target,
                "program":        prog_name,
                "band":           band,
                "priority":       priority,
                "completeness":   completeness,
                "t_wc_opt":       t_opt,
                "t_wc_con":       t_con,
                "obs_window_hrs": obs_days * 24.0,
            })

    return rows


def solve_campaign(rows, lam, use_conservative=False, mission_budget_days=MISSION_BUDGET_DAYS):
    """Solve one scalarised MILP point on the Pareto front.

    Objective:
        minimize  -lam * sum(c_i * x_i)  +  (1-lam) * sum(T_wc_i / T_mission * x_i)

    x_i in {0,1}: 1 = target is scheduled.

    Constraints:
        sum(T_wc_i * x_i) <= T_mission   (total mission budget)
        T_wc_i * x_i      <= W_i         (must fit in observability window)
    """
    T_mission = mission_budget_days * 24.0

    valid = [r for r in rows
             if (r["t_wc_con"] if use_conservative else r["t_wc_opt"]) is not None]
    if not valid:
        return None

    n     = len(valid)
    t_wc  = np.array([r["t_wc_con"] if use_conservative else r["t_wc_opt"] for r in valid])
    compl = np.array([r["completeness"] for r in valid])
    t_norm = t_wc / T_mission

    c_obj = -lam * compl + (1 - lam) * t_norm

    budget_constraint = LinearConstraint(t_wc.reshape(1, -1), lb=-np.inf, ub=T_mission)
    obs_windows = np.array([r["obs_window_hrs"] for r in valid])
    window_constraint = LinearConstraint(np.diag(t_wc), lb=-np.inf, ub=obs_windows)

    result = milp(
        c=c_obj,
        constraints=[budget_constraint, window_constraint],
        integrality=np.ones(n),
        bounds=Bounds(lb=0, ub=1),
    )

    if result.status != 0:
        return None

    x = np.round(result.x).astype(int)
    scheduled = [valid[i] for i in range(n) if x[i] == 1]

    return {
        "lam":               lam,
        "x":                 x,
        "completeness":      float(np.dot(compl, x)),
        "t_wc_hrs":          float(np.dot(t_wc, x)),
        "n_scheduled":       int(x.sum()),
        "scheduled_targets": scheduled,
    }


def run_pareto(rows, n_points=50, use_conservative=False):
    """Sweep lambda to trace the Pareto front between completeness and total time."""
    solutions, seen = [], set()
    for lam in np.linspace(0, 1, n_points):
        sol = solve_campaign(rows, lam, use_conservative=use_conservative)
        if sol is None:
            continue
        key = (round(sol["completeness"], 4), round(sol["t_wc_hrs"], 2))
        if key not in seen:
            seen.add(key)
            solutions.append(sol)
    return solutions


def plot_pareto(pareto_opt, pareto_con=None):
    """Plot Pareto front: completeness vs total wall clock time, coloured by n_scheduled."""
    fig, ax = plt.subplots(figsize=(9, 6))

    def _plot(solutions, label, color, marker):
        comp = [s["completeness"] for s in solutions]
        days = [s["t_wc_hrs"] / 24.0 for s in solutions]
        n    = [s["n_scheduled"] for s in solutions]
        sc = ax.scatter(days, comp, c=n, cmap="viridis", s=80,
                        marker=marker, label=label, zorder=3)
        ax.plot(days, comp, color=color, alpha=0.4, linewidth=1)
        return sc

    sc = _plot(pareto_opt, "Optimistic", "steelblue", "o")
    if pareto_con:
        _plot(pareto_con, "Conservative", "tomato", "s")

    plt.colorbar(sc, ax=ax, label="Number of targets scheduled")
    ax.axvline(MISSION_BUDGET_DAYS, color="black", linestyle="--", linewidth=1.2,
               label=f"Mission budget ({MISSION_BUDGET_DAYS} days)")
    ax.set_xlabel("Total Wall Clock Time (days)")
    ax.set_ylabel("Total Completeness (sum of detection probabilities)")
    ax.set_title("Campaign Design Pareto Front\nCompleteness vs. Total Observation Time")
    ax.legend()
    plt.tight_layout()
    return fig, ax


def print_campaign(solution, ref_star_results, label="Campaign"):
    """Print a formatted schedule for one Pareto solution.

    Shows each scheduled target paired with its best available reference star,
    with per-target wall clock time and completeness, sorted by TPC priority.
    """
    scheduled = solution["scheduled_targets"]
    if not scheduled:
        print(f"\n[{label}] No targets scheduled.")
        return

    scheduled = sorted(scheduled, key=lambda r: (r["priority"], r["target"]))

    print(f"\n{'='*88}")
    print(f"  {label}  |  lam={solution['lam']:.2f}  |  "
          f"{solution['n_scheduled']} targets  |  "
          f"Completeness={solution['completeness']:.3f}  |  "
          f"Total={solution['t_wc_hrs']/24:.1f} days")
    print(f"{'='*88}")
    print(f"{'Pri':>3}  {'Target':<18}  {'Program':<8}  {'Ref Star':<22}  "
          f"{'Gr':>2}  {'RefDays':>7}  {'WC(hrs)':>8}  {'Compl':>6}")
    print(f"{'-'*3}  {'-'*18}  {'-'*8}  {'-'*22}  {'-'*2}  {'-'*7}  {'-'*8}  {'-'*6}")

    for row in scheduled:
        target = row["target"]
        band   = row["band"]
        t_wc   = row.get("t_wc_opt") or row.get("t_wc_con") or 0.0
        compl  = row["completeness"]

        # Pull best ref star across observable windows
        ref_name  = "N/A"
        ref_grade = "-"
        ref_days  = 0
        res = ref_star_results.get((target, band))
        if res and res.get("observable_windows"):
            for win in res["observable_windows"]:
                best = win.get("best_ref")
                if best:
                    ref_name  = best["reference_star"]
                    ref_grade = best["grade"]
                    ref_days  = best["n_valid_days"]
                    break

        print(f"{row['priority']:>3}  {target:<18}  {row['program']:<8}  "
              f"{ref_name:<22}  {ref_grade:>2}  {ref_days:>7}  {t_wc:>8.2f}  {compl:>6.3f}")

    print(f"{'-'*88}")
    print(f"{'':>3}  {'TOTAL':>18}  {'':>8}  {'':>22}  {'':>2}  {'':>7}  "
          f"{solution['t_wc_hrs']:>8.2f}  {solution['completeness']:>6.3f}")
    print()


def compare_pareto_campaigns(pareto_solutions, ref_star_results, n_compare=4):
    """Print schedules at representative points across the Pareto front.

    Selects n_compare solutions evenly spaced by completeness (high → low) so
    you can see how the campaign composition changes as you trade completeness
    for total integration time.
    """
    if not pareto_solutions:
        print("No Pareto solutions to compare.")
        return

    sorted_sols = sorted(pareto_solutions, key=lambda s: -s["completeness"])
    if len(sorted_sols) <= n_compare:
        chosen = sorted_sols
    else:
        indices = np.round(np.linspace(0, len(sorted_sols) - 1, n_compare)).astype(int)
        chosen  = [sorted_sols[i] for i in indices]

    for i, sol in enumerate(chosen):
        print_campaign(sol, ref_star_results,
                       label=f"Pareto Point {i+1}/{len(chosen)}")


def expand_obs_sequence(target, ref_name, t_wc_hrs,
                        N=N, R=R, N_slew=N_slew, N_rollchange=N_rollchange,
                        T_slew=T_slew, T_rollchange=T_rollchange, T_HOWFSC=T_HOWFSC):
    """Expand one target observation into its detailed ADI+RDI micro-steps.

    Standard sequence (Roll A-B-A-B, N=4 science visits per half):
      HOWFSC start
      Slew -> ref star  |  Ref visit 1 (dark hole dig)
      Slew -> target    |  Sci Roll A [1]  RC  Sci Roll B [2]  RC  Sci Roll A [3]  RC  Sci Roll B [4]
      Slew -> ref star  |  Ref visit 2
      Slew -> target    |  Sci Roll A [5]  RC  Sci Roll B [6]  RC  Sci Roll A [7]  RC  Sci Roll B [8]
      Slew -> ref star  |  Ref visit 3
      HOWFSC final

    Returns list of dicts: {label, actor, duration_hrs}
    """
    t_etc         = t_wc_hrs  # approximate: use wallclock as proxy for scaling
    # Recompute from wallclock inverse
    overhead      = N_slew * T_slew + N_rollchange * T_rollchange + T_HOWFSC
    t_obs         = t_wc_hrs - overhead
    t_etc_sci     = t_obs / (1 + (3 * R) / (2 * N))
    T_tv          = t_etc_sci / (2 * N)   # per science visit
    T_rv          = R * T_tv              # per ref visit

    rolls = ["Roll A", "Roll B", "Roll A", "Roll B"]

    steps = []

    steps.append({"label": "HOWFSC start",              "actor": "telescope",  "duration_hrs": T_HOWFSC / 2})
    steps.append({"label": f"Slew -> {ref_name}",        "actor": ref_name,     "duration_hrs": T_slew})
    steps.append({"label": f"Ref visit 1  (dark hole)",  "actor": ref_name,     "duration_hrs": T_rv})

    # First half — N science visits
    steps.append({"label": f"Slew -> {target}",          "actor": target,       "duration_hrs": T_slew})
    for vi in range(N):
        steps.append({"label": f"Sci {rolls[vi % 4]}  [{vi+1}/{2*N}]",
                      "actor": target, "duration_hrs": T_tv})
        if vi < N - 1:
            steps.append({"label": "Roll change", "actor": "telescope", "duration_hrs": T_rollchange})

    # Mid-cycle ref visit
    steps.append({"label": f"Slew -> {ref_name}",        "actor": ref_name,     "duration_hrs": T_slew})
    steps.append({"label": f"Ref visit 2",                "actor": ref_name,     "duration_hrs": T_rv})

    # Second half — N science visits
    steps.append({"label": f"Slew -> {target}",          "actor": target,       "duration_hrs": T_slew})
    for vi in range(N):
        steps.append({"label": f"Sci {rolls[vi % 4]}  [{N+vi+1}/{2*N}]",
                      "actor": target, "duration_hrs": T_tv})
        if vi < N - 1:
            steps.append({"label": "Roll change", "actor": "telescope", "duration_hrs": T_rollchange})

    # Final ref + HOWFSC
    steps.append({"label": f"Slew -> {ref_name}",        "actor": ref_name,     "duration_hrs": T_slew})
    steps.append({"label": f"Ref visit 3",                "actor": ref_name,     "duration_hrs": T_rv})
    steps.append({"label": "HOWFSC final",                "actor": "telescope",  "duration_hrs": T_HOWFSC / 2})

    return steps


def analyze_target_pairs(programs, ref_star_results, coords_dict,
                         separation_threshold_deg=15.0):
    """Analyze the geometry of (science, ref) pairs across all programs.

    Identifies:
      1. Shared reference stars — targets that use the same ref star and could
         be grouped into a single continuous observation block, saving slew time.
      2. Close target pairs — science targets within separation_threshold_deg
         of each other, regardless of ref star.
      3. For each close pair: whether they share a ref star (best case for
         co-scheduling) or have different ref stars (still saves inter-obs slew).

    Prints a report and returns a dict with the findings.
    """
    # Build target -> best ref star mapping from ref_star_results
    target_ref = {}
    for prog in programs.values():
        for target, (priority, band, contrast) in prog.items():
            res = ref_star_results.get((target, band))
            if res and res.get("observable_windows"):
                for win in res["observable_windows"]:
                    best = win.get("best_ref")
                    if best:
                        target_ref[target] = best["reference_star"]
                        break

    all_sci = list(target_ref.keys())

    # 1. Shared reference stars
    ref_to_targets = {}
    for target, ref in target_ref.items():
        ref_to_targets.setdefault(ref, []).append(target)

    shared = {ref: tgts for ref, tgts in ref_to_targets.items() if len(tgts) > 1}

    print(f"\n{'='*70}")
    print("  TARGET PAIR ANALYSIS")
    print(f"{'='*70}")

    print(f"\n--- Shared reference stars ({len(shared)} ref stars used by >1 target) ---")
    if shared:
        for ref, tgts in sorted(shared.items(), key=lambda x: -len(x[1])):
            print(f"  {ref:<25}  <-  {', '.join(tgts)}")
    else:
        print("  None found.")

    # 2. Close target pairs by angular separation
    print(f"\n--- Science target pairs within {separation_threshold_deg} deg ---")
    close_pairs = []
    for i, t1 in enumerate(all_sci):
        for t2 in all_sci[i + 1:]:
            c1 = coords_dict.get(t1)
            c2 = coords_dict.get(t2)
            if c1 is None or c2 is None:
                continue
            sep = float(c1.separation(c2).deg)
            if sep <= separation_threshold_deg:
                same_ref = target_ref.get(t1) == target_ref.get(t2)
                close_pairs.append({
                    "t1": t1, "t2": t2,
                    "sep_deg": sep,
                    "ref_t1": target_ref.get(t1, "N/A"),
                    "ref_t2": target_ref.get(t2, "N/A"),
                    "shared_ref": same_ref,
                })

    close_pairs.sort(key=lambda x: x["sep_deg"])

    if close_pairs:
        print(f"  {'Target 1':<20}  {'Target 2':<20}  {'Sep(deg)':>8}  "
              f"{'Shared ref?':>11}  Ref stars")
        print(f"  {'-'*20}  {'-'*20}  {'-'*8}  {'-'*11}  {'-'*30}")
        for p in close_pairs:
            shared_str = "YES" if p["shared_ref"] else "no"
            if p["shared_ref"]:
                ref_str = p["ref_t1"]
            else:
                ref_str = f"{p['ref_t1']} / {p['ref_t2']}"
            print(f"  {p['t1']:<20}  {p['t2']:<20}  {p['sep_deg']:>8.1f}  "
                  f"{shared_str:>11}  {ref_str}")
    else:
        print(f"  No pairs within {separation_threshold_deg} deg.")

    # 3. Co-schedulable groups: shared ref AND close on sky
    coschedulable = [p for p in close_pairs if p["shared_ref"]]
    print(f"\n--- Co-schedulable groups (shared ref + close on sky): "
          f"{len(coschedulable)} pair(s) ---")
    if coschedulable:
        for p in coschedulable:
            slew_hrs = p["sep_deg"] / SLEW_RATE_DEG_PER_HR
            print(f"  {p['t1']} + {p['t2']}  |  ref: {p['ref_t1']}  |  "
                  f"sep: {p['sep_deg']:.1f} deg  |  slew: {slew_hrs*60:.0f} min")
    else:
        print("  None — no targets share both a ref star and sky proximity.")

    print()
    return {
        "target_ref":      target_ref,
        "shared_refs":     shared,
        "close_pairs":     close_pairs,
        "coschedulable":   coschedulable,
    }


# Roman slew rate: 90 deg in 30 min = 180 deg/hr
SLEW_RATE_DEG_PER_HR = 180.0


def _inter_slew_hrs(sci_a, sci_b):
    """Slew time (hrs) between two science targets based on sky separation."""
    c1 = coords_dict.get(sci_a)
    c2 = coords_dict.get(sci_b)
    if c1 is None or c2 is None:
        return 1.0   # fallback 1 hr if coords missing
    return float(c1.separation(c2).deg) / SLEW_RATE_DEG_PER_HR


def build_full_timeline(solution, ref_star_results, planet_timing,
                        use_conservative=False, mission_budget_days=MISSION_BUDGET_DAYS,
                        pair_analysis=None):
    """Stage 2: order (science, ref) pairs by sky proximity, then expand each pair
    into its fixed ADI+RDI micro-step sequence.

    The (science target, reference star) pairs are already fixed by ref_star_results —
    this function just picks the best ref star for each target from there.

    Ordering uses nearest-neighbor greedy + 2-opt to minimise total inter-observation
    slew time (from the end of one observation to the start of the next).
    Priority is NOT used for ordering.

    Each observation unit expands to:
      HOWFSC start
      Slew -> ref star  |  Ref visit 1 (dark hole)
      Slew -> science   |  Roll A / Roll B  x N
      Slew -> ref star  |  Ref visit 2
      Slew -> science   |  Roll A / Roll B  x N
      Slew -> ref star  |  Ref visit 3
      HOWFSC final
    """
    T_mission = mission_budget_days * 24.0
    n_days    = len(ts)
    selected  = list(solution["scheduled_targets"])

    if not selected:
        return []

    def _wc(row):
        return ((row.get("t_wc_con") if use_conservative else row.get("t_wc_opt"))
                or row.get("t_wc_opt") or row.get("t_wc_con") or 0.0)

    # Build (science, ref_star) pairs from ref_star_results
    pairs = []
    for row in selected:
        target   = row["target"]
        band     = row["band"]
        ref_name = "N/A"
        res = ref_star_results.get((target, band))
        if res and res.get("observable_windows"):
            for win in res["observable_windows"]:
                best = win.get("best_ref")
                if best:
                    ref_name = best["reference_star"]
                    break
        pairs.append({
            "row":      row,
            "target":   target,
            "ref_star": ref_name,
            "wc":       _wc(row),
        })

    def observable_at(target, t_start_hr, dur_hr):
        d0   = int(t_start_hr / 24.0)
        d1   = int(np.ceil((t_start_hr + dur_hr) / 24.0))
        if d1 >= n_days:
            return False
        mask = keepout.get(target, np.zeros(n_days, dtype=bool))
        return bool(mask[d0:d1 + 1].all())

    # ------------------------------------------------------------------ #
    # Nearest-neighbor greedy over science target positions                #
    # ------------------------------------------------------------------ #
    unvisited = list(range(len(pairs)))
    order     = []        # list of (pair_idx, t_start, t_end)
    t_cursor  = 0.0
    prev_sci  = None

    # Build co-schedulable group lookup: target -> set of targets it should stay near
    coschedulable_neighbors = {}
    if pair_analysis:
        for cp in pair_analysis.get("coschedulable", []):
            coschedulable_neighbors.setdefault(cp["t1"], set()).add(cp["t2"])
            coschedulable_neighbors.setdefault(cp["t2"], set()).add(cp["t1"])

    while unvisited:
        candidates = []
        for idx in unvisited:
            p        = pairs[idx]
            sl       = _inter_slew_hrs(prev_sci, p["target"]) if prev_sci else 0.0
            t_arrive = t_cursor + sl
            if t_arrive + p["wc"] > T_mission:
                continue
            if not observable_at(p["target"], t_arrive, p["wc"]):
                continue
            # Prefer co-schedulable neighbors of current target (same ref, nearby)
            is_coschedulable = (prev_sci is not None and
                                p["target"] in coschedulable_neighbors.get(prev_sci, set()))
            candidates.append((0.0 if is_coschedulable else sl, sl, idx))

        if not candidates:
            break

        candidates.sort()
        best_idx = candidates[0][2]
        p        = pairs[best_idx]
        sl       = _inter_slew_hrs(prev_sci, p["target"]) if prev_sci else 0.0
        t_arrive = t_cursor + sl

        order.append((best_idx, t_arrive, t_arrive + p["wc"]))
        t_cursor = t_arrive + p["wc"]
        prev_sci = p["target"]
        unvisited.remove(best_idx)

    # ------------------------------------------------------------------ #
    # 2-opt improvement                                                    #
    # ------------------------------------------------------------------ #
    improved = True
    while improved:
        improved = False
        for i in range(len(order) - 1):
            for j in range(i + 2, len(order)):
                new_order = (order[:i + 1]
                             + list(reversed(order[i + 1:j + 1]))
                             + order[j + 1:])
                valid = True
                t_cur = 0.0
                prev  = None
                timed = []
                for (idx, _, _) in new_order:
                    p  = pairs[idx]
                    sl = _inter_slew_hrs(prev, p["target"]) if prev else 0.0
                    t_arr = t_cur + sl
                    if t_arr + p["wc"] > T_mission or not observable_at(p["target"], t_arr, p["wc"]):
                        valid = False
                        break
                    timed.append((idx, t_arr, t_arr + p["wc"]))
                    t_cur = t_arr + p["wc"]
                    prev  = p["target"]
                if valid and t_cur < order[-1][2]:
                    order    = timed
                    improved = True

    # ------------------------------------------------------------------ #
    # Expand each (science, ref) pair into its ADI+RDI micro-steps        #
    # ------------------------------------------------------------------ #
    timeline    = []
    prev_target = None

    for (idx, t_start, _) in order:
        p      = pairs[idx]
        target = p["target"]
        ref    = p["ref_star"]
        row    = p["row"]
        wc     = p["wc"]

        # Inter-observation slew between consecutive science targets
        if prev_target is not None:
            sl = _inter_slew_hrs(prev_target, target)
            if sl > 0:
                timeline.append({
                    "target": target, "program": row["program"],
                    "priority": row["priority"], "ref_star": ref,
                    "label":        f"Slew  {prev_target} -> {target}",
                    "actor":        "telescope",
                    "duration_hrs": sl,
                    "t_start_hrs":  t_start - sl,
                    "t_end_hrs":    t_start,
                    "t_start_days": (t_start - sl) / 24.0,
                    "t_end_days":   t_start / 24.0,
                })

        # Fixed ADI+RDI sequence for this (science, ref) pair
        steps  = expand_obs_sequence(target, ref, wc)
        t_step = t_start
        for step in steps:
            t_end = t_step + step["duration_hrs"]
            timeline.append({
                "target":       target,
                "program":      row["program"],
                "priority":     row["priority"],
                "ref_star":     ref,
                "label":        step["label"],
                "actor":        step["actor"],
                "duration_hrs": step["duration_hrs"],
                "t_start_hrs":  t_step,
                "t_end_hrs":    t_end,
                "t_start_days": t_step / 24.0,
                "t_end_days":   t_end / 24.0,
            })
            t_step = t_end

        prev_target = target

    return timeline


def print_full_timeline(timeline, label="Full Observation Timeline"):
    """Print the complete micro-step sequence with running clock."""
    print(f"\n{'='*108}")
    print(f"  {label}")
    print(f"{'='*108}")
    print(f"{'#':>4}  {'T_start(d)':>10}  {'T_end(d)':>9}  {'Dur(hr)':>8}  "
          f"{'Pri':>3}  {'Target':<18}  {'Step':<40}  {'Actor'}")
    print("-" * 108)

    current_target = None
    for i, s in enumerate(timeline):
        # Print a blank separator between targets
        if s["target"] != current_target and current_target is not None:
            print()
        current_target = s["target"]

        print(f"{i+1:>4}  {s['t_start_days']:>10.3f}  {s['t_end_days']:>9.3f}  "
              f"{s['duration_hrs']:>8.3f}  {s['priority']:>3}  "
              f"{s['target']:<18}  {s['label']:<40}  {s['actor']}")

    print("-" * 108)
    n_targets = len({s["target"] for s in timeline})
    total_hrs = timeline[-1]["t_end_hrs"] if timeline else 0.0
    print(f"  Targets: {n_targets}  |  Total: {total_hrs:.2f} hrs = {total_hrs/24:.2f} days")
    print()


def plot_full_timeline(timeline, label="Full Observation Timeline"):
    """Gantt chart of the full micro-step sequence, coloured by actor type."""
    import matplotlib.patches as mpatches

    if not timeline:
        print("Nothing to plot.")
        return None, None

    # Assign y-position per target (grouped blocks)
    targets_ordered = list(dict.fromkeys(s["target"] for s in timeline))
    y_map = {t: i for i, t in enumerate(targets_ordered)}

    actor_colors = {
        "telescope": "silver",    # HOWFSC + roll changes
        "ref":       "darkorange",
        "sci":       "steelblue",
    }

    def _actor_color(actor, target):
        if actor == "telescope":
            return "silver"
        if actor == target:
            return "steelblue"
        return "darkorange"

    fig, ax = plt.subplots(figsize=(16, max(4, len(targets_ordered) * 0.7)))

    for s in timeline:
        y     = y_map[s["target"]]
        color = _actor_color(s["actor"], s["target"])
        width = s["t_end_days"] - s["t_start_days"]
        ax.barh(y, width, left=s["t_start_days"], height=0.6,
                color=color, edgecolor="black", linewidth=0.3)

    ax.set_yticks(range(len(targets_ordered)))
    ax.set_yticklabels(
        [f"P{timeline[[s['target'] for s in timeline].index(t)]['priority'] if False else '?'}  {t}"
         for t in targets_ordered],
        fontsize=8,
    )
    # Simpler y-labels: priority from first occurrence
    prio_map = {}
    for s in timeline:
        prio_map.setdefault(s["target"], s["priority"])
    ax.set_yticklabels([f"P{prio_map[t]}  {t}" for t in targets_ordered], fontsize=8)

    ax.set_xlabel("Mission Days")
    ax.set_title(label)
    ax.axvline(MISSION_BUDGET_DAYS, color="red", linestyle="--",
               linewidth=1.4, label=f"Budget ({MISSION_BUDGET_DAYS} days)")

    patches = [
        mpatches.Patch(color="steelblue",  label="Science (target)"),
        mpatches.Patch(color="darkorange", label="Reference star"),
        mpatches.Patch(color="silver",     label="Telescope overhead (HOWFSC / roll)"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8)
    plt.tight_layout()
    return fig, ax


def run_pareto_with_timelines(rows, ref_star_results, planet_timing,
                              n_points=50, use_conservative=False,
                              mission_budget_days=MISSION_BUDGET_DAYS,
                              pair_analysis=None):
    """Combined Stage 1 + Stage 2: sweep lambda, sequence every candidate, return
    enriched Pareto points built from actual timeline metrics.

    For each lambda:
      1. Solve MILP to get candidate target set (Stage 1)
      2. Build full micro-step timeline for that set (Stage 2)
      3. Record what actually fits within the budget after sequencing

    Deduplicates by (actual_completeness, actual_t_wc_hrs) so only distinct
    campaigns appear on the front.

    Returns list of dicts:
        lam, timeline, targets, n_scheduled, completeness, t_wc_hrs, t_wc_days
    """
    results, seen = [], set()

    for lam in np.linspace(0, 1, n_points):
        sol = solve_campaign(rows, lam,
                             use_conservative=use_conservative,
                             mission_budget_days=mission_budget_days)
        if sol is None or not sol["scheduled_targets"]:
            continue

        timeline = build_full_timeline(sol, ref_star_results, planet_timing,
                                       use_conservative=use_conservative,
                                       mission_budget_days=mission_budget_days,
                                       pair_analysis=pair_analysis)
        if not timeline:
            continue

        # Metrics from the actual sequenced timeline
        targets_done  = list(dict.fromkeys(s["target"] for s in timeline))
        total_wc_hrs  = timeline[-1]["t_end_hrs"]
        target_compl  = {r["target"]: r["completeness"] for r in sol["scheduled_targets"]}
        actual_compl  = sum(target_compl.get(t, 0.0) for t in targets_done)

        key = (round(actual_compl, 4), round(total_wc_hrs, 2))
        if key in seen:
            continue
        seen.add(key)

        results.append({
            "lam":         lam,
            "timeline":    timeline,
            "targets":     targets_done,
            "n_scheduled": len(targets_done),
            "completeness": actual_compl,
            "t_wc_hrs":    total_wc_hrs,
            "t_wc_days":   total_wc_hrs / 24.0,
        })

    return results


def plot_pareto_timelines(front_opt, front_con=None):
    """Plot Pareto front from actual sequenced timeline metrics.

    Each point is a real, executable campaign. Colour = number of targets
    that fit within the budget after full micro-step sequencing.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    def _plot(front, label, color, marker):
        if not front:
            return None
        comp = [f["completeness"] for f in front]
        days = [f["t_wc_days"]    for f in front]
        n    = [f["n_scheduled"]  for f in front]
        sc = ax.scatter(days, comp, c=n, cmap="viridis", s=80,
                        marker=marker, label=label, zorder=3)
        ax.plot(days, comp, color=color, alpha=0.4, linewidth=1)
        return sc

    sc = _plot(front_opt, "Optimistic",   "steelblue", "o")
    if front_con:
        _plot(front_con, "Conservative", "tomato",    "s")

    if sc is not None:
        plt.colorbar(sc, ax=ax, label="Targets sequenced within budget")
    ax.axvline(MISSION_BUDGET_DAYS, color="black", linestyle="--", linewidth=1.2,
               label=f"Mission budget ({MISSION_BUDGET_DAYS} days)")
    ax.set_xlabel("Total Wall Clock Time — actual sequence (days)")
    ax.set_ylabel("Total Completeness — targets sequenced")
    ax.set_title("Campaign Design Pareto Front\n"
                 "Each point = a fully sequenced, executable campaign")
    ax.legend()
    plt.tight_layout()
    return fig, ax


def compare_timeline_front(front, n_compare=4, label="Campaigns"):
    """Print detailed micro-step timelines at representative points on the front.

    Selects n_compare points evenly spaced from max-completeness to min-time
    so you can see how the campaign sequence changes across the front.
    """
    if not front:
        print("No campaigns to compare.")
        return

    ordered = sorted(front, key=lambda f: -f["completeness"])
    if len(ordered) <= n_compare:
        chosen = ordered
    else:
        idx    = np.round(np.linspace(0, len(ordered) - 1, n_compare)).astype(int)
        chosen = [ordered[i] for i in idx]

    print(f"\n{'#'*10}  {label}  {'#'*10}")
    for i, f in enumerate(chosen):
        print_full_timeline(
            f["timeline"],
            label=f"Pareto Point {i+1}/{len(chosen)}  "
                  f"(lam={f['lam']:.2f}  |  {f['n_scheduled']} targets  |  "
                  f"compl={f['completeness']:.3f}  |  {f['t_wc_days']:.1f} days)"
        )


def diff_timeline_campaigns(front_a, front_b, label_a="A", label_b="B"):
    """Diff the extreme campaigns (max-completeness vs min-time) on each front."""
    for front, label in [(front_a, label_a), (front_b, label_b)]:
        if len(front) < 2:
            continue
        ordered = sorted(front, key=lambda f: -f["completeness"])
        hi, lo  = ordered[0], ordered[-1]
        kept    = set(hi["targets"]) & set(lo["targets"])
        dropped = set(hi["targets"]) - set(lo["targets"])
        added   = set(lo["targets"]) - set(hi["targets"])
        print(f"\nDiff [{label}]  Max-completeness -> Min-time")
        print(f"  Kept    ({len(kept):>2}): {', '.join(sorted(kept)) or 'none'}")
        print(f"  Dropped ({len(dropped):>2}): {', '.join(sorted(dropped)) or 'none'}")
        print(f"  Added   ({len(added):>2}): {', '.join(sorted(added)) or 'none'}")
        print(f"  Completeness : {hi['completeness']:.3f} -> {lo['completeness']:.3f}")
        print(f"  Time (days)  : {hi['t_wc_days']:.1f} -> {lo['t_wc_days']:.1f}")
        print()


def diff_campaigns(sol_a, sol_b, label_a="A", label_b="B"):
    """Show what changes between two Pareto solutions.

    Prints which targets are kept, dropped, and added as you move from
    sol_a to sol_b along the front (e.g., from max-completeness to min-time).
    """
    names_a = {r["target"] for r in sol_a["scheduled_targets"]}
    names_b = {r["target"] for r in sol_b["scheduled_targets"]}
    kept    = names_a & names_b
    dropped = names_a - names_b
    added   = names_b - names_a

    print(f"\nDiff  [{label_a}] -> [{label_b}]")
    print(f"  Kept    ({len(kept):>2}): {', '.join(sorted(kept)) or 'none'}")
    print(f"  Dropped ({len(dropped):>2}): {', '.join(sorted(dropped)) or 'none'}")
    print(f"  Added   ({len(added):>2}): {', '.join(sorted(added)) or 'none'}")
    print(f"  Completeness : {sol_a['completeness']:.3f} -> {sol_b['completeness']:.3f}")
    print(f"  Time (days)  : {sol_a['t_wc_hrs']/24:.1f} -> {sol_b['t_wc_hrs']/24:.1f}")
    print()


# =============================================================================
# RUN: Stage 1 (MILP selection) + Stage 2 (sequencing) for every Pareto point
# =============================================================================

# --- Analyze (science, ref) pair geometry before optimizing ---
pair_analysis = analyze_target_pairs(
    ALL_PROGRAMS, ref_star_results, coords_dict, separation_threshold_deg=15.0
)

campaign_rows = build_campaign_table(
    ALL_PROGRAMS, keepout, ts, planet_timing, ALL_TARGET_PRIORITY
)

# Check enough targets have real wall clock times to make optimization meaningful
n_with_timing = sum(1 for r in campaign_rows if r["t_wc_opt"] is not None)
if n_with_timing < 2:
    print(f"Only {n_with_timing} target(s) have integration time data. "
          f"Add more .npy files to PLANET_DATA_FILES before running the Pareto sweep.")
else:
    # --- Stage 1 + Stage 2 combined: every Pareto candidate gets fully sequenced ---
    print("Running optimistic Pareto sweep + sequencing...")
    front_opt = run_pareto_with_timelines(
        campaign_rows, ref_star_results, planet_timing,
        n_points=50, use_conservative=False, pair_analysis=pair_analysis
    )

    print("Running conservative Pareto sweep + sequencing...")
    front_con = run_pareto_with_timelines(
        campaign_rows, ref_star_results, planet_timing,
        n_points=50, use_conservative=True, pair_analysis=pair_analysis
    )

    # --- Pareto front: each point = a real, fully sequenced campaign ---
    fig, ax = plot_pareto_timelines(front_opt, front_con)
    plt.show()

    # --- Print detailed micro-step timelines at 4 points across the front ---
    compare_timeline_front(front_opt, n_compare=4, label="OPTIMISTIC CAMPAIGNS")
    compare_timeline_front(front_con, n_compare=4, label="CONSERVATIVE CAMPAIGNS")

    # --- Show what changes between max-completeness and min-time extremes ---
    diff_timeline_campaigns(front_opt, front_con,
                            label_a="Optimistic", label_b="Conservative")

    # --- Gantt chart for the best completeness campaign ---
    best = sorted(front_opt, key=lambda f: -f["completeness"])[0]
    fig2, ax2 = plot_full_timeline(
        best["timeline"],
        label=f"Best Completeness Campaign — {best['n_scheduled']} targets, "
              f"{best['t_wc_days']:.1f} days"
    )
    plt.show()
