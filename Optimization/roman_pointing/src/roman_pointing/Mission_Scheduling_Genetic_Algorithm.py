#Mission_Scheduling_Genetic_Algorithm.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from roman_pointing.roman_observability import (
    get_target_coords,
    compute_keepout,
)
from roman_pointing.Reference_Star_Selection_Tool import (
    load_catalog,
    select_ref_star,
)


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

# Priority weights: higher tier = higher weight in the objective
PRIORITY_WEIGHT = {1: 4.0, 2: 3.0, 3: 2.0, 4: 1.0}

# Detection probability proxies (no .npy file)
PRIORITY_COMPLETENESS = {1: 0.80, 2: 0.60, 3: 0.40, 4: 0.20}

# Proxy integration times (hrs) when no .npy file is available
PRIORITY_T_ETC_FALLBACK = {1: 5.0, 2: 1.0, 3: 8.0, 4: 12.0}

# ADI+RDI sequence parameters
N            = 4
R            = 1 / 4
N_slew       = 5
N_rollchange = 2 * (N - 1)
T_slew       = 0.5
T_rollchange = 1 / 6
T_HOWFSC     = 1.0


def compute_wallclock(t_etc):
    """Compute wall clock time (hours) from integration time."""
    return (
        t_etc * (1 + (3 * R) / (2 * N))
        + N_slew * T_slew
        + N_rollchange * T_rollchange
        + T_HOWFSC
    )



#
# Step 1: Build and rank the target sequence
#
# For each target:
#   - Compute solar keepout → obs_mask (is it observable at all?)
#   - Assign score = det_prob × priority_weight
#   - Targets with obs_days == 0 are excluded
# Output: target_sequence — candidates sorted by score (highest first)
#

coords_dict = get_target_coords(all_targets)

ts, keepout, solar_angles = compute_keepout(
    coords_dict,
    ANALYSIS_START,
    ANALYSIS_DAYS,
    time_step=1,
)

# Planet detection data (.npy) — detection probability over time
PLANET_DATA_DIR = Path(__file__).resolve().parents[3] / "for_rifah" / "for_rifah"
PLANET_DATA_FILES = {
    ('47 UMa', 'c', 1): PLANET_DATA_DIR / 'for_rifah_47_UMa_c_band1.npy',
    ('HD 114783', 'c', 1): PLANET_DATA_DIR / 'for_rifah_HD_114783_c_band1.npy',
    ('HD 154345', 'b', 1): PLANET_DATA_DIR / 'for_rifah_HD_154345_b_band1.npy',
    ('HD 160691', 'c', 3): PLANET_DATA_DIR / 'for_rifah_HD_160691_c_band3.npy',
    ('HD 190360', 'b', 1): PLANET_DATA_DIR / 'for_rifah_HD_190360_b_band1.npy',
    ('HD 190360', 'b', 3): PLANET_DATA_DIR / 'for_rifah_HD_190360_b_band3.npy',
    ('HD 192310', 'c', 1): PLANET_DATA_DIR / 'for_rifah_HD_192310_c_band1.npy',
    ('HD 217107', 'c', 1): PLANET_DATA_DIR / 'for_rifah_HD_217107_c_band1.npy',
    ('HD 217107', 'c', 3): PLANET_DATA_DIR / 'for_rifah_HD_217107_c_band3.npy',
    ('* eps Eri', 'b', 1): PLANET_DATA_DIR / 'for_rifah_eps_Eri_b_band1.npy',
    ('* eps Eri', 'b', 4): PLANET_DATA_DIR / 'for_rifah_eps_Eri_b_band4.npy',
    ('ups And', 'd', 1): PLANET_DATA_DIR / 'for_rifah_ups_And_d_band1.npy',
}

def load_planet_visibility(filepath):
    data     = np.load(filepath, allow_pickle=True)[()]
    t_opt    = np.array(data["integration_time_hours_opt"]["median"])
    t_con    = np.array(data["integration_time_hours_con"]["median"])
    det_prob = np.array(data["det_probability"])
    obs      = ~np.array(data["solar_keepout"], dtype=bool) & \
               ~np.array(data["GB_keepout"],    dtype=bool)
    best_idx = int(np.argmax(np.where(obs, det_prob, -np.inf)))
    return {
        "best_t_opt":    t_opt[best_idx],
        "best_t_con":    t_con[best_idx],
        "best_det_prob": det_prob[best_idx],
    }

planet_timing = {}
for (target, planet, band), filepath in PLANET_DATA_FILES.items():
    pdata = load_planet_visibility(filepath)
    key = (target, band)
    if key not in planet_timing or pdata["best_det_prob"] > planet_timing[key]["best_det_prob"]:
        planet_timing[key] = {
            "planet":       planet,
            "opt":          compute_wallclock(pdata["best_t_opt"]),
            "con":          compute_wallclock(pdata["best_t_con"]),
            "best_t_opt":   pdata["best_t_opt"],
            "best_t_con":   pdata["best_t_con"],
            "best_det_prob": pdata["best_det_prob"],
        }

target_sequence = []
for prog_name, prog in ALL_PROGRAMS.items():
    for target, (priority, band, contrast) in prog.items():
        key      = (target, band)
        obs_mask = keepout.get(target, np.zeros(len(ts), dtype=bool))
        obs_days = int(np.sum(obs_mask))

        if obs_days == 0:
            continue   # not observable — exclude

        if key in planet_timing:
            det_prob  = planet_timing[key]["best_det_prob"]
            t_etc_opt = planet_timing[key]["best_t_opt"]
            t_etc_con = planet_timing[key]["best_t_con"]
        else:
            det_prob  = PRIORITY_COMPLETENESS.get(priority, 0.2)
            t_etc_opt = PRIORITY_T_ETC_FALLBACK.get(priority, 8.0)
            t_etc_con = t_etc_opt * 1.15

        score = det_prob * PRIORITY_WEIGHT.get(priority, 1.0)

        target_sequence.append({
            "target":    target,
            "program":   prog_name,
            "band":      band,
            "priority":  priority,
            "contrast":  contrast,
            "det_prob":  det_prob,
            "score":     score,
            "t_etc_opt": t_etc_opt,
            "t_etc_con": t_etc_con,
            "obs_mask":  obs_mask,
            "obs_days":  obs_days,
            "coord":     coords_dict.get(target),
        })

# Sort by score descending — highest priority/completeness targets come first
target_sequence.sort(key=lambda x: -x["score"])

print(f"\nStep 1 — Target sequence ({len(target_sequence)} observable targets):")
print(f"  {'#':<3}  {'Target':<20}  {'Pri':>3}  {'DetProb':>7}  {'Score':>6}  {'ObsDays':>7}")
print(f"  {'-'*3}  {'-'*20}  {'-'*3}  {'-'*7}  {'-'*6}  {'-'*7}")
for rank, c in enumerate(target_sequence, 1):
    print(f"  {rank:<3}  {c['target']:<20}  {c['priority']:>3}  "
          f"{c['det_prob']:>7.3f}  {c['score']:>6.3f}  {c['obs_days']:>7}")


# 
# Step 2: Assign 1 ref star per target
#
# Planet .npy files supply planet timing and detection probability data only.
# Reference star assignments are generated separately by the reference-star
# selection pipeline using the catalog loaded by load_catalog().
# For each target in the sequence, call select_ref_star() and take the
# single best ref star from the highest-quality window.
# Output: obs_units — one entry per target, each with its ref_star name
# 


#HLC-supporting stars  are  > than SPC-supporting star

catalog = load_catalog()

def pick_best_ref(target, band, contrast):
    """Return the name of the best single ref star for this target, or None."""
    res = select_ref_star(
        sci_name=target,
        analysis_start=ANALYSIS_START,
        analysis_days=ANALYSIS_DAYS,
        band=band,
        contrast=contrast,
        catalog=catalog,
        sort_mode="valid_days",
    )
    if not isinstance(res, dict):
        return None
    windows = res.get("observable_windows", [])
    if not isinstance(windows, list):
        return None
    for win in windows:
        if not isinstance(win, dict):
            continue
        best_ref = win.get("best_ref")
        if isinstance(best_ref, dict):
            return best_ref.get("reference_star")
    return None

obs_units = []
for c in target_sequence:
    ref_star = pick_best_ref(c["target"], c["band"], c["contrast"])
    obs_units.append({**c, "ref_star": ref_star})

print(f"\nStep 2 — Observation units (target + 1 ref star):")
print(f"  {'Target':<20}  {'Ref Star':<22}  {'Score':>6}")
print(f"  {'-'*20}  {'-'*22}  {'-'*6}")
for u in obs_units:
    ref_label = u["ref_star"] if u["ref_star"] else "NO REF"
    print(f"  {u['target']:<20}  {ref_label:<22}  {u['score']:>6.3f}")

# Targets with no ref star available are excluded from the GA
obs_units = [u for u in obs_units if u["ref_star"] is not None]
print(f"\n  {len(obs_units)} units with a valid ref star (excluded "
      f"{len(target_sequence) - len(obs_units)} with no ref star)")


# 
# Step 3: Build campaigns and slew costs
#
# A campaign = 2 observation units (2 science targets + 2 ref stars).
# All unique unordered pairs of obs_units form the candidate campaign pool.
#
# slew_matrix[i,j] = slew time (hrs) from obs_unit i to obs_unit j.
#   Used for:
#     - within a campaign  : unit_a to unit_b
#     - between campaigns  : last unit of campaign k → first unit of campaign k+1
# 

#CHANGE THIS TO PICK OUT THE CALCULTOR (SLEW CALCULTOR USIGN SCIPY OPT)

SLEW_RATE_DEG_PER_HR = 180.0   # 90 deg in 30 min
n_units = len(obs_units)

# Unit-to-unit slew time matrix
slew_matrix = np.zeros((n_units, n_units))
for i in range(n_units):
    for j in range(n_units):
        if i == j:
            continue
        c1 = obs_units[i]["coord"]
        c2 = obs_units[j]["coord"]
        if c1 is None or c2 is None:
            slew_matrix[i, j] = 1.0
        else:
            slew_matrix[i, j] = float(c1.separation(c2).deg) / SLEW_RATE_DEG_PER_HR

# Form all campaigns = unique unordered pairs of obs_units
campaigns = []
for i in range(n_units):
    for j in range(i + 1, n_units):
        campaigns.append({"unit_a": i, "unit_b": j})

n_campaigns = len(campaigns)

print(f"\nStep 3 — Campaigns (2 targets + 2 ref stars each):")
print(f"  Obs units   : {n_units}")
print(f"  Campaigns   : {n_campaigns}  ({n_units} choose 2)")
print(f"  Mean slew   : {slew_matrix[slew_matrix > 0].mean():.2f} hrs")


# 
# Step 4: Genetic Algorithm — evaluate campaign sequences
#
# Chromosome: ordered permutation of campaign indices (0..n_campaigns-1).
#   Walk campaigns greedily until the mission budget is exhausted.
#   Each campaign visits unit_a then unit_b (2 targets + 2 ref stars).
#
# Pareto objectives:
#   1. completeness = Σ det_prob_i  for all scheduled units
#                     det_prob_i is the detection probability achieved by
#                     integrating for t_etc_i — so completeness is a direct
#                     function of integration time
#   2. t_wallclock  = total wall clock (hrs) across all scheduled campaigns
#
# Wall clock formula (per scheduled unit across all campaigns):
#   t_etc_total × (1 + 3R/2N)          ADI+RDI overhead on integration time
#   + n_units × N_slew × T_slew        slew sequences per target
#   + n_units × N_rollchange × T_rollchange
#   + T_HOWFSC                          one-time WFC sensing overhead
#   + slew_total                        inter-unit slews (entry + intra-campaign)
#
# Operators:
#   Selection : tournament (size TOURN_SIZE)
#   Elitism   : top ELITE_SIZE chromosomes survive unchanged each generation
#   Crossover : Order Crossover (OX) — preserves relative campaign ordering
#   Mutation  : swap two random campaign positions
# 

POP_SIZE   = 100
N_GEN      = 200
TOURN_SIZE = 3
CX_PROB    = 0.85
MUT_PROB   = 0.15
ELITE_SIZE = 5
SEED       = 42

rng = np.random.default_rng(SEED)


def evaluate(chromosome, use_conservative=False):
    """Walk the chromosome (campaign order), schedule campaigns until budget exhausted.

    Each campaign contributes 2 obs units: unit_a then unit_b.
    Slew costs:
      - entry slew: end of previous campaign (unit_b) → current campaign unit_a
      - intra-campaign slew: unit_a → unit_b

    Detection score for each unit = det_prob_i, the detection probability
    achieved by integrating for t_etc_i hours. The returned completeness value
    is the summed detection score across all scheduled units, so it can exceed
    1.0 when multiple targets are scheduled.

    Returns:
        completeness : sum of det_prob for all scheduled units
        t_wallclock  : total wall clock (hrs)
        scheduled    : list of campaign indices that were scheduled
    """
    T_mission = MISSION_BUDGET_DAYS * 24.0
    t_etc_key = "t_etc_con" if use_conservative else "t_etc_opt"

    t_etc_total  = 0.0
    slew_total   = 0.0
    completeness = 0.0
    scheduled    = []
    prev_b       = None   # unit_b index of the last scheduled campaign

    for camp_idx in chromosome:
        camp  = campaigns[camp_idx]
        ua    = obs_units[camp["unit_a"]]
        ub    = obs_units[camp["unit_b"]]
        a_idx = camp["unit_a"]
        b_idx = camp["unit_b"]

        t_etc_camp = ua[t_etc_key] + ub[t_etc_key]
        intra_slew = slew_matrix[a_idx, b_idx]
        entry_slew = slew_matrix[prev_b, a_idx] if prev_b is not None else 0.0

        n_units_next = (len(scheduled) + 1) * 2   # 2 units per campaign
        t_wc_next = (
            (t_etc_total + t_etc_camp) * (1 + (3 * R) / (2 * N))
            + n_units_next * N_slew * T_slew
            + n_units_next * N_rollchange * T_rollchange
            + T_HOWFSC
            + slew_total + entry_slew + intra_slew
        )

        if t_wc_next > T_mission:
            break

        t_etc_total  += t_etc_camp
        slew_total   += entry_slew + intra_slew
        completeness += ua["det_prob"] + ub["det_prob"]
        scheduled.append(camp_idx)
        prev_b = b_idx

    if not scheduled:
        return 0.0, 0.0, []

    n_units_sched = len(scheduled) * 2
    t_wallclock = (
        t_etc_total * (1 + (3 * R) / (2 * N))
        + n_units_sched * N_slew * T_slew
        + n_units_sched * N_rollchange * T_rollchange
        + T_HOWFSC
        + slew_total
    )
    return completeness, t_wallclock, scheduled


def tournament_select(population, fitnesses):
    contestants = rng.choice(len(population), size=TOURN_SIZE, replace=False)
    best = max(contestants, key=lambda i: fitnesses[i][0])
    return population[best].copy()


def order_crossover(p1, p2):
    """OX crossover: copy a slice from p1, fill remainder in p2 order."""
    n    = len(p1)
    a, b = sorted(rng.choice(n, size=2, replace=False))
    child = np.full(n, -1, dtype=int)
    child[a:b] = p1[a:b]
    fill  = [x for x in p2 if x not in child[a:b]]
    pos   = list(range(b, n)) + list(range(0, a))
    for i, p in enumerate(pos):
        child[p] = fill[i]
    return child


def mutate(chromosome):
    c    = chromosome.copy()
    i, j = rng.choice(len(c), size=2, replace=False)
    c[i], c[j] = c[j], c[i]
    return c


def is_dominated(a, b):
    """True if b dominates a (b has >= completeness AND <= wall clock)."""
    return b[0] >= a[0] and b[1] <= a[1] and (b[0] > a[0] or b[1] < a[1])


def update_pareto(pareto_pts, pareto_sols, new_pt, new_sol):
    """Add new solution to Pareto front if non-dominated."""
    for p in pareto_pts:
        if is_dominated(new_pt, p):
            return pareto_pts, pareto_sols
    kept_pts  = [p for p in pareto_pts  if not is_dominated(p, new_pt)]
    kept_sols = [s for s in pareto_sols if not is_dominated(
                     (s["completeness"], s["t_wallclock"]), new_pt)]
    return kept_pts + [new_pt], kept_sols + [new_sol]


def run_ga(use_conservative=False, verbose=True):
    """Run the GA and return the Pareto front of campaign sequences.

    Each chromosome is a permutation of campaign indices (0..n_campaigns-1).
    """
    population  = [rng.permutation(n_campaigns) for _ in range(POP_SIZE)]
    pareto_pts  = []
    pareto_sols = []

    for gen in range(N_GEN):
        fitnesses = [evaluate(chrom, use_conservative) for chrom in population]

        for _, (comp, t_wc, sched) in enumerate(fitnesses):
            if comp == 0:
                continue
            sol = {
                "completeness":     comp,
                "t_wallclock":      t_wc,
                "t_wallclock_days": t_wc / 24.0,
                "n_campaigns":      len(sched),
                "n_scheduled":      len(sched) * 2,
                "campaigns": [
                    {
                        "target_a": obs_units[campaigns[k]["unit_a"]]["target"],
                        "ref_a":    obs_units[campaigns[k]["unit_a"]]["ref_star"],
                        "t_etc_a":  obs_units[campaigns[k]["unit_a"]]["t_etc_opt"],
                        "target_b": obs_units[campaigns[k]["unit_b"]]["target"],
                        "ref_b":    obs_units[campaigns[k]["unit_b"]]["ref_star"],
                        "t_etc_b":  obs_units[campaigns[k]["unit_b"]]["t_etc_opt"],
                    }
                    for k in sched
                ],
                "sequence": [
                    (
                        obs_units[campaigns[k]["unit_a"]]["target"],
                        obs_units[campaigns[k]["unit_a"]]["ref_star"],
                        obs_units[campaigns[k]["unit_a"]]["t_etc_opt"],
                    )
                    for k in sched
                ] + [
                    (
                        obs_units[campaigns[k]["unit_b"]]["target"],
                        obs_units[campaigns[k]["unit_b"]]["ref_star"],
                        obs_units[campaigns[k]["unit_b"]]["t_etc_opt"],
                    )
                    for k in sched
                ],
            }
            pareto_pts, pareto_sols = update_pareto(
                pareto_pts, pareto_sols, (comp, t_wc), sol
            )

        if verbose and gen % 50 == 0:
            best_comp   = max((f[0] for f in fitnesses), default=0)
            valid_times = [f[1] for f in fitnesses if f[0] > 0]
            best_time   = min(valid_times) if valid_times else 0
            print(f"  Gen {gen:>4}  |  best total detection score={best_comp:.3f}  "
                  f"|  min wall clock={best_time/24:.1f} days  "
                  f"|  Pareto size={len(pareto_pts)}")

        # Elitism: carry top ELITE_SIZE chromosomes unchanged into next generation
        elite_idx = sorted(range(len(fitnesses)), key=lambda i: -fitnesses[i][0])[:ELITE_SIZE]
        next_pop  = [population[i].copy() for i in elite_idx]

        while len(next_pop) < POP_SIZE:
            p1    = tournament_select(population, fitnesses)
            p2    = tournament_select(population, fitnesses)
            child = order_crossover(p1, p2) if rng.random() < CX_PROB else p1.copy()
            if rng.random() < MUT_PROB:
                child = mutate(child)
            next_pop.append(child)
        population = next_pop

    pareto_sols.sort(key=lambda s: -s["completeness"])
    return pareto_sols


def plot_pareto(solutions, label="GA Pareto Front"):
    fig, ax = plt.subplots(figsize=(9, 6))
    comp = [s["completeness"]     for s in solutions]
    days = [s["t_wallclock_days"] for s in solutions]
    n    = [s["n_campaigns"]      for s in solutions]
    sc   = ax.scatter(days, comp, c=n, cmap="viridis", s=80, zorder=3)
    ax.plot(days, comp, color="steelblue", alpha=0.4, linewidth=1)
    plt.colorbar(sc, ax=ax, label="Campaigns scheduled")
    ax.axvline(MISSION_BUDGET_DAYS, color="red", linestyle="--", linewidth=1.2,
               label=f"Mission budget ({MISSION_BUDGET_DAYS} days)")
    ax.set_xlabel("Total Wall Clock (days)")
    ax.set_ylabel("Total detection score (sum of det_prob for scheduled units)")
    ax.set_title(label)
    ax.legend()
    plt.tight_layout()
    return fig, ax


# --- Run ---
print(f"\nStep 4 — Running GA ({POP_SIZE} population, {N_GEN} generations, "
      f"elitism={ELITE_SIZE}, {n_campaigns} candidate campaigns)...")
pareto_opt = run_ga(use_conservative=False, verbose=True)
pareto_con = run_ga(use_conservative=True,  verbose=True)

print(f"\nOptimistic front  : {len(pareto_opt)} solutions")
print(f"Conservative front: {len(pareto_con)} solutions")

plot_pareto(pareto_opt, "GA Pareto Front — Optimistic")
plot_pareto(pareto_con, "GA Pareto Front — Conservative")
plt.show()

best = pareto_opt[0]
print(f"\nBest campaign sequence ({best['n_campaigns']} campaigns, "
      f"{best['n_scheduled']} units):")
print(f"  {'#':<3}  {'Target A':<22}  {'Ref A':<22}  {'t_etc_A':>8}  {'Target B':<22}  {'Ref B':<22}  {'t_etc_B':>8}")
for k, c in enumerate(best["campaigns"], 1):
    print(f"  {k:<3}  {c['target_a']:<22}  {c['ref_a']:<22}  {c['t_etc_a']:>8.2f}  "
          f"{c['target_b']:<22}  {c['ref_b']:<22}  {c['t_etc_b']:>8.2f}")
print(f"\n  Total detection score : {best['completeness']:.3f}")
print(f"  Wall clock           : {best['t_wallclock_days']:.1f} days")
print(f"  HOWFSC overhead included in total: {T_HOWFSC:.1f} hr")
print(f"  ADI/RDI overhead factor: 1 + 3R/2N = {1 + (3 * R) / (2 * N):.3f}")

print(f"\nBest unit sequence ({best['n_scheduled']} units):")
for target, ref, t_etc in best["sequence"]:
    print(f"  Target: {target:<20}  Ref: {ref:<22}  t_etc_opt: {t_etc:.2f} hr")
print(f"  Total detection score : {best['completeness']:.3f}")
print(f"  Wall clock           : {best['t_wallclock_days']:.1f} days")
print(f"  HOWFSC overhead included in total: {T_HOWFSC:.1f} hr")
