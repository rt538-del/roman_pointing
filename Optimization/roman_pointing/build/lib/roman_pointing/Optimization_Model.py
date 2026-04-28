import os
import json
import copy
import numpy as np
import astropy.units as u
import EXOSIMS.Prototypes.TargetList
import matplotlib.pyplot as plt
import shutil
from astroquery.simbad import Simbad as SimbadQuery

# ── Fix environment issues ────────────────────────────────────────────────────
os.environ["CGI_NOISE_DATA_DIR"] = r"C:\Users\tasni\roman_repos\cgi_noise\cgi_noise\data"
os.environ["CORGIETC_DATA_DIR"]  = r"C:\Users\tasni\roman_repos\corgietc\corgietc\data"

import EXOSIMS
if not hasattr(EXOSIMS, "__version__") or EXOSIMS.__version__ == "3.7.0alpha":
    EXOSIMS.__version__ = "3.7.0a0"

from MeanStars.MeanStars import MeanStars as MeanStarsClass
import EXOSIMS.Prototypes.TargetList as TL_module
TL_module.MeanStars = MeanStarsClass

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination

from roman_pointing.Reference_Star_Selection_Tool import load_catalog, select_ref_star

# ── Clear EXOSIMS cache ───────────────────────────────────────────────────────
cache_dir = os.path.expanduser(r"~\.EXOSIMS\cache")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"Cleared EXOSIMS cache at {cache_dir}")

# ── Parameters ────────────────────────────────────────────────────────────────
TARGETS = [
    "14 Her", "23 Lib", "47 UMa", "alf Cen A", "bet Gem",
    "bet Pic", "eps Eri", "gam Cep", "mu. Ara",
    "pi. Men", "rho01 Cnc", "tau Cet", "ups And",
    "HD 100546", "HD 114613", "HD 142", "HD 154345", "HD 190360",
    "HD 192310", "HD 217107", "HD 219077", "HD 219134", "HD 30562"
]
ANALYSIS_START      = "2026-12-01T00:00:00"
ANALYSIS_DAYS       = 365
MISSION_BUDGET_DAYS = 60
PLANET_SEP_AU       = 4.0

# ── Reference star design alternatives ───────────────────────────────────────
DESIGN_POINTS = {
    "Grade A only\nClosest Mag": {
        "allowed_grades": ["A"],
        "sort_mode":      "closest_mag",
        "color":          "blue",
        "marker":         "o",
        "ls":             "-",
    },
    "Grade A+B\nClosest Mag": {
        "allowed_grades": ["A", "B"],
        "sort_mode":      "closest_mag",
        "color":          "green",
        "marker":         "s",
        "ls":             "--",
    },
    "Grade A+B+C\nClosest Mag": {
        "allowed_grades": ["A", "B", "C"],
        "sort_mode":      "closest_mag",
        "color":          "red",
        "marker":         "^",
        "ls":             "-.",
    },
    "Grade A+B\nMost Valid Days": {
        "allowed_grades": ["A", "B"],
        "sort_mode":      "valid_days",
        "color":          "orange",
        "marker":         "D",
        "ls":             ":",
    },
    "Grade A+B\nClosest Pitch": {
        "allowed_grades": ["A", "B"],
        "sort_mode":      "closest_pitch",
        "color":          "purple",
        "marker":         "v",
        "ls":             "-",
    },
}

# ── Vmag fallback via SIMBAD ──────────────────────────────────────────────────
def get_vmag_from_simbad(target_name):
    try:
        simbad = SimbadQuery()
        simbad.add_votable_fields("V")
        result = simbad.query_object(target_name)
        if result is not None and len(result) > 0:
            val = result["V"][0]
            if not np.ma.is_masked(val) and val is not None:
                return float(val)
    except Exception as e:
        print(f"    SIMBAD query failed: {e}")
    return None

# ── Science value function ────────────────────────────────────────────────────
def science_value(vmag):
    return 10.0 ** ((5.0 - vmag) / 2.5)

# ── Setup base ETC ────────────────────────────────────────────────────────────
scriptfile = os.path.join(
    os.environ["CORGIETC_DATA_DIR"], "scripts", "CGI_Noise.json"
)
with open(scriptfile, "r") as f:
    specs = json.loads(f.read())

TL_base   = EXOSIMS.Prototypes.TargetList.TargetList(**copy.deepcopy(specs))
OS_base   = TL_base.OpticalSystem
mode_base = OS_base.observingModes[0]  # OPT_IMG_NFB1_HLC

print("Observing modes available:")
for m in OS_base.observingModes:
    print(f"  {m['Scenario']}")

catalog = load_catalog()

# ── Shared ETC inputs ─────────────────────────────────────────────────────────
fZ_base  = np.repeat(TL_base.ZodiacalLight.fZ0, 1)
WA_base  = (np.array([7.5])
            * (mode_base["lam"] / OS_base.pupilDiam).to(
                u.arcsec, equivalencies=u.dimensionless_angles()))
JEZ_base = TL_base.JEZ0[mode_base["hex"]] / (PLANET_SEP_AU**2)
vmag_ref = float(TL_base.Vmag[0])
print(f"\nBase TL prototype Vmag: {vmag_ref:.2f}")

# ── Precompute Vmags for all targets once ─────────────────────────────────────
print("\nPrecomputing Vmags for all targets...")
vmag_cache = {}
for target in TARGETS:
    # try a quick picker call just for Vmag
    try:
        picker_tmp = select_ref_star(
            sci_name       = target,
            analysis_start = ANALYSIS_START,
            analysis_days  = ANALYSIS_DAYS,
            band           = 1,
            contrast       = "med",
            catalog        = catalog,
            sort_mode      = "closest_mag",
            allowed_grades = ["A", "B", "C"],
        )
        vmag = picker_tmp.get("sci_mag")
    except Exception:
        vmag = None

    if vmag is None:
        print(f"  {target}: not in corgidb, querying SIMBAD...")
        vmag = get_vmag_from_simbad(target)

    if vmag is None:
        print(f"  {target}: defaulting to 5.0")
        vmag = 5.0

    vmag_cache[target] = vmag
    print(f"  {target}: Vmag={vmag:.2f}")

print(f"\nVmag cache complete for {len(vmag_cache)} targets")

# ── Helper: compute intTime for a target given its vmag ──────────────────────
def compute_inttime(target, vmag):
    """Scale fluxes, compute integration time, restore fluxes."""
    flux_scale = 10.0 ** ((vmag_ref - vmag) / 2.5)

    original_fluxes = {}
    for mode in OS_base.observingModes:
        hex_key = mode["hex"]
        if hex_key in TL_base.star_fluxes:
            original_fluxes[hex_key] = TL_base.star_fluxes[hex_key].copy()
            TL_base.star_fluxes[hex_key] = original_fluxes[hex_key] * flux_scale

    original_vmag   = TL_base.Vmag.copy()
    TL_base.Vmag[0] = vmag

    achievable_dmag    = None
    achievable_inttime = None
    for dmag_test in [25, 24, 23, 22, 21, 20, 19, 18, 17, 16]:
        t = OS_base.calc_intTime(
            TL_base, [0], fZ_base, JEZ_base,
            np.array([float(dmag_test)]), WA_base, mode_base
        )
        t_days = t[0].to(u.day).value
        if not np.isnan(t_days) and t_days > 0:
            if achievable_dmag is None:
                achievable_dmag    = dmag_test
                achievable_inttime = t_days

    for hex_key, orig in original_fluxes.items():
        TL_base.star_fluxes[hex_key] = orig
    TL_base.Vmag[0] = original_vmag[0]

    return achievable_dmag, achievable_inttime

# ── Precompute integration times for all targets once ────────────────────────
print("\nPrecomputing integration times for all targets...")
inttime_cache = {}
for target in TARGETS:
    vmag = vmag_cache[target]
    dmag, inttime = compute_inttime(target, vmag)
    if dmag is not None:
        inttime_cache[target] = {
            "intTime_days"    : inttime,
            "dmag_achievable" : dmag,
            "vmag"            : vmag,
            "science_value"   : science_value(vmag),
        }
        print(f"  {target}: intTime={inttime:.4f}d  "
              f"dMag={dmag}  sci_val={science_value(vmag):.3f}")
    else:
        print(f"  {target}: no achievable dMag — skipping")

print(f"\nIntegration time cache complete for {len(inttime_cache)} targets")


# =============================================================================
# NSGA-II Problem Class
# =============================================================================
class RomanSchedulingProblem(Problem):
    """
    Decision variables per target (3 per target):
        x[i*3+0] = window index   (discrete int)
        x[i*3+1] = ref star index (discrete int)
        x[i*3+2] = time fraction  (continuous [0,1])

    Objectives (minimized):
        f1 = -n_detections
        f2 = -mean_science_value_per_detection

    Constraints:
        g1 = sum(time_fractions) - 1.0 <= 0
    """

    def __init__(self, results, mission_budget_days):
        self.results             = results
        self.targets             = list(results.keys())
        self.n_targets           = len(self.targets)
        self.mission_budget_days = mission_budget_days

        self.max_windows = []
        self.max_refs    = []
        for target in self.targets:
            windows = results[target]["picker"]["observable_windows"]
            self.max_windows.append(max(1, len(windows)))
            max_refs = max(
                (len(w["valid_refs"]) for w in windows if w["valid_refs"]),
                default=1,
            )
            self.max_refs.append(max(1, max_refs))

        xl = np.zeros(self.n_targets * 3)
        xu = np.zeros(self.n_targets * 3)
        for i in range(self.n_targets):
            xl[i*3+0] = 0
            xu[i*3+0] = self.max_windows[i] - 1
            xl[i*3+1] = 0
            xu[i*3+1] = self.max_refs[i] - 1
            xl[i*3+2] = 0.0
            xu[i*3+2] = 1.0

        super().__init__(
            n_var    = self.n_targets * 3,
            n_obj    = 2,
            n_constr = 1,
            xl       = xl,
            xu       = xu,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        f1_vals, f2_vals, g1_vals = [], [], []

        for x in X:
            n_detections   = 0
            total_sci_val  = 0.0
            time_fractions = []

            for i, target in enumerate(self.targets):
                win_idx = int(round(x[i*3+0]))
                ref_idx = int(round(x[i*3+1]))
                t_frac  = x[i*3+2]
                time_fractions.append(t_frac)

                windows = self.results[target]["picker"]["observable_windows"]
                if not windows:
                    continue

                win_idx = min(win_idx, len(windows) - 1)
                window  = windows[win_idx]
                if not window["valid_refs"]:
                    continue

                t_alloc      = t_frac * self.mission_budget_days
                intTime_days = self.results[target]["intTime_days"]
                sv           = self.results[target]["science_value"]

                if (not np.isnan(intTime_days)
                        and intTime_days > 0
                        and t_alloc >= intTime_days):
                    n_detections  += 1
                    total_sci_val += sv

            f1 = -float(n_detections)
            if n_detections > 0:
                f2 = -float(total_sci_val / n_detections)
            else:
                f2 = 0.0
            g1 = float(np.sum(time_fractions)) - 1.0

            f1_vals.append(f1)
            f2_vals.append(f2)
            g1_vals.append(g1)

        out["F"] = np.column_stack([f1_vals, f2_vals])
        out["G"] = np.array(g1_vals).reshape(-1, 1)


# =============================================================================
# STAGE 1+2+3: Run full optimization for each design point
# =============================================================================
print("\n" + "="*60)
print("RUNNING OPTIMIZATION FOR EACH DESIGN ALTERNATIVE")
print("="*60)

all_pareto  = {}
all_results = {}

for dp_name, dp in DESIGN_POINTS.items():
    dp_label = dp_name.replace("\n", " ")
    print(f"\n{'='*60}")
    print(f"Design Point: {dp_label}")
    print(f"  Grades:    {dp['allowed_grades']}")
    print(f"  Sort mode: {dp['sort_mode']}")
    print(f"{'='*60}")

    results_dp = {}

    for target in TARGETS:
        # skip if no integration time computed
        if target not in inttime_cache:
            continue

        # skip if integration time exceeds budget
        if inttime_cache[target]["intTime_days"] > MISSION_BUDGET_DAYS:
            print(f"  {target}: intTime > budget, skipping")
            continue

        # run ref star picker with this design point's parameters
        try:
            picker_result = select_ref_star(
                sci_name       = target,
                analysis_start = ANALYSIS_START,
                analysis_days  = ANALYSIS_DAYS,
                band           = 1,
                contrast       = "med",
                catalog        = catalog,
                sort_mode      = dp["sort_mode"],
                allowed_grades = dp["allowed_grades"],
            )
        except Exception as e:
            print(f"  {target}: picker failed ({e}), skipping")
            continue

        if "error" in picker_result:
            print(f"  {target}: {picker_result['error']}")
            continue

        # check at least one window has valid ref stars
        has_valid_refs = any(
            len(w["valid_refs"]) > 0
            for w in picker_result["observable_windows"]
        )
        if not has_valid_refs:
            print(f"  {target}: no valid ref stars with grades "
                  f"{dp['allowed_grades']}, skipping")
            continue

        results_dp[target] = {
            "picker"          : picker_result,
            "intTime_days"    : inttime_cache[target]["intTime_days"],
            "dmag_achievable" : inttime_cache[target]["dmag_achievable"],
            "vmag"            : inttime_cache[target]["vmag"],
            "science_value"   : inttime_cache[target]["science_value"],
        }

    print(f"\n  Feasible targets: {len(results_dp)}")
    if not results_dp:
        print(f"  No feasible targets — skipping this design point")
        continue

    # print target summary for this design point
    total_needed = sum(r["intTime_days"] for r in results_dp.values())
    print(f"  Total time needed: {total_needed:.1f}d  "
          f"Budget: {MISSION_BUDGET_DAYS}d  "
          f"Coverage: {MISSION_BUDGET_DAYS/total_needed*100:.0f}%")

    for target, r in sorted(results_dp.items(),
                            key=lambda x: x[1]["science_value"],
                            reverse=True):
        win = r["picker"]["observable_windows"]
        best_ref = win[0]["best_ref"] if win and win[0]["best_ref"] else None
        ref_str  = (f"{best_ref['reference_star']} "
                    f"grade={best_ref['grade']}"
                    if best_ref else "N/A")
        print(f"    {target:20s}: intTime={r['intTime_days']:.2f}d  "
              f"sci_val={r['science_value']:.3f}  "
              f"best_ref={ref_str}")

    # build and run NSGA-II
    problem_dp = RomanSchedulingProblem(
        results             = results_dp,
        mission_budget_days = MISSION_BUDGET_DAYS,
    )

    algorithm   = NSGA2(pop_size=100, eliminate_duplicates=True)
    termination = get_termination("n_gen", 200)

    print(f"\n  Running NSGA-II...")
    res_dp = pymoo_minimize(
        problem_dp,
        algorithm,
        termination,
        seed    = 42,
        verbose = False,
    )

    F_dp       = res_dp.F
    n_det_dp   = -F_dp[:, 0]
    sci_val_dp = -F_dp[:, 1]

    all_pareto[dp_name] = {
        "n_detections" : n_det_dp,
        "sci_values"   : sci_val_dp,
        "n_targets"    : len(results_dp),
        "color"        : dp["color"],
        "marker"       : dp["marker"],
        "ls"           : dp["ls"],
    }
    all_results[dp_name] = results_dp

    print(f"  Detections range: {n_det_dp.min():.0f} to {n_det_dp.max():.0f}")
    print(f"  Sci val range   : {sci_val_dp.min():.2f} to {sci_val_dp.max():.2f}")


# =============================================================================
# PLOTTING
# =============================================================================
print("\n" + "="*60)
print("PLOTTING RESULTS")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax_idx, (ax, zoom) in enumerate(zip(axes, [False, True])):
    for dp_name, dp_res in all_pareto.items():
        sort_idx     = np.argsort(dp_res["n_detections"])
        n_det_sorted = dp_res["n_detections"][sort_idx]
        sv_sorted    = dp_res["sci_values"][sort_idx]

        if zoom:
            mask = (n_det_sorted >= 2) & (n_det_sorted <= 12)
            if not mask.any():
                continue
            n_det_sorted = n_det_sorted[mask]
            sv_sorted    = sv_sorted[mask]

        label = dp_name.replace("\n", " — ")
        ax.plot(n_det_sorted, sv_sorted,
                color=dp_res["color"],
                marker=dp_res["marker"],
                linestyle=dp_res["ls"],
                linewidth=2.5,
                markersize=9,
                label=label)

        if zoom:
            for nd, sv in zip(n_det_sorted, sv_sorted):
                ax.annotate(f"{sv:.1f}",
                           xy=(nd, sv),
                           xytext=(4, 4),
                           textcoords="offset points",
                           fontsize=7,
                           color=dp_res["color"])

    ax.set_xlabel("Number of Detections (f1)", fontsize=12)
    ax.set_ylabel("Mean Science Value per Detection (f2)", fontsize=12)
    ax.set_title(
        "Full Pareto Front Comparison" if not zoom
        else "Zoomed View: 2-12 Detections\n(Region of Interest)",
        fontsize=12
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

fig.suptitle(
    "Pareto Front Comparison: Reference Star Selection Strategies\n"
    "Roman CGI Band 1 HLC, 23 Exoplanet Host Targets, "
    f"Budget={MISSION_BUDGET_DAYS}d",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig("pareto_front_ref_star_strategies.png", dpi=150)
plt.show()
print("Plot saved to pareto_front_ref_star_strategies.png")


# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "="*70)
print("DESIGN ALTERNATIVE SUMMARY")
print("="*70)
print(f"{'Strategy':<35} {'Targets':>8} {'Max Det':>8} "
      f"{'Best SV':>10} {'@ 6 Det':>10}")
print("-"*70)

for dp_name, dp_res in all_pareto.items():
    label    = dp_name.replace("\n", " ")
    n_det    = dp_res["n_detections"]
    sv       = dp_res["sci_values"]
    sort_idx = np.argsort(n_det)
    n_det_s  = n_det[sort_idx]
    sv_s     = sv[sort_idx]
    at_6     = sv_s[n_det_s == 6]
    at_6_str = f"{at_6[0]:.2f}" if len(at_6) > 0 else "N/A"

    print(f"  {label:<33} "
          f"{dp_res['n_targets']:>8} "
          f"{n_det.max():>8.0f} "
          f"{sv.max():>10.2f} "
          f"{at_6_str:>10}")

print("="*70)

# ── Print best schedule for recommended design point ──────────────────────────
recommended = "Grade A+B\nClosest Mag"
if recommended in all_pareto and recommended in all_results:
    print(f"\nRecommended Design: {recommended.replace(chr(10), ' ')}")
    dp_res   = all_pareto[recommended]
    res_dp   = all_results[recommended]
    n_det    = dp_res["n_detections"]
    sv       = dp_res["sci_values"]
    sort_idx = np.argsort(n_det)
    n_det_s  = n_det[sort_idx]
    sv_s     = sv[sort_idx]

    # find the knee — where marginal sci val drop is largest
    if len(n_det_s) > 2:
        diffs    = np.diff(sv_s)
        knee_idx = np.argmin(diffs) + 1
        knee_det = int(n_det_s[knee_idx])
        knee_sv  = sv_s[knee_idx]
        print(f"  Knee of curve at {knee_det} detections "
              f"(sci_val={knee_sv:.2f})")
        print(f"  Recommended operating point: {knee_det} detections")

print("\nDone.")