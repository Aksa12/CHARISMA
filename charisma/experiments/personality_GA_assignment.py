import json
import os
import itertools
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from charisma.config import config


BIN_TARGETS = {"Similar": 0.40, "Medium": 0.20, "Different": 0.40}
L1_THRESHOLDS = dict(
    L1_SIMILAR_MAX=1.00, L1_MEDIUM_MIN=1.25, L1_MEDIUM_MAX=2.00, L1_DIFFERENT_MIN=2.25
)
GLOBAL_SEED = 42

# ============================================================
# CORE FUNCTIONS
# ============================================================


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _parse_percentage_label(label: str) -> float:
    try:
        pct_str = label.split()[-1].strip().replace("%", "")
        pct = float(pct_str) / 100.0
        return round(pct, 2)
    except Exception:
        return np.nan


def select_bfi_from_votes(bfi_json_obj: Dict) -> Dict[str, float]:
    dims = [
        "Openness",
        "Conscientiousness",
        "Extraversion",
        "Agreeableness",
        "Neuroticism",
    ]
    out = {}
    for d in dims:
        entries = bfi_json_obj.get(d, [])
        if not entries:
            out[d] = np.nan
            continue
        max_count = max(e.get("theCount", -1) for e in entries)
        tied = [e for e in entries if e.get("theCount", -1) == max_count]
        numeric = []
        for e in tied:
            lab = e.get("myValue") or e.get("personality_type") or ""
            numeric.append(_parse_percentage_label(lab))
        if len(numeric) > 1:
            diffs = [abs(v - 0.5) for v in numeric]
            min_diff = min(diffs)
            candidates = [v for v, dff in zip(numeric, diffs) if dff == min_diff]
            chosen = max(candidates)
        else:
            chosen = numeric[0]
        snapped = round(chosen * 4) / 4.0
        out[d] = snapped
    return out


def load_characters_bfi(characters_csv_path: str) -> pd.DataFrame:
    df_chars = pd.read_csv(characters_csv_path)

    bfi_rows = []
    for _, row in df_chars.iterrows():
        raw = row["bfi_dimensions_json"]
        obj = json.loads(raw) if isinstance(raw, str) else raw
        vec = select_bfi_from_votes(obj)
        rec = {
            "id": row["id"],
            "mbti_profile": row["mbti_profile"],
            "sloan_top_label": row["sloan_top_label"],
        }
        rec.update(vec)
        bfi_rows.append(rec)

    return pd.DataFrame(bfi_rows)


def l1_distance(dims, vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    return float(sum(abs(vec_a[d] - vec_b[d]) for d in dims))


def cosine_sim_zscore(df, dims, a_row, b_row):
    """
    Compute cosine similarity after z-scoring each dimension across all agents.
    df: full character dataframe with numeric BFI columns
    a_row, b_row: Series rows for the two agents
    """
    # Z-score across all agents (same for both)
    z_df = (df[dims] - df[dims].mean()) / df[dims].std(ddof=0)

    # Extract z-scored vectors for the two agents
    va = z_df.loc[a_row.name].values.astype(float)
    vb = z_df.loc[b_row.name].values.astype(float)

    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(va, vb) / (na * nb))

def cosine_sim(dims, vec_a: dict, vec_b: dict) -> float:
        va = np.array([vec_a[d] for d in dims], dtype=float)
        vb = np.array([vec_b[d] for d in dims], dtype=float)
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        denom = na * nb
        if denom == 0.0:
            return float("nan")
        return float(np.dot(va, vb) / denom)

def make_pair_table(df_bfi: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    dims = [
        "Openness",
        "Conscientiousness",
        "Extraversion",
        "Agreeableness",
        "Neuroticism",
    ]

    def bin_by_l1(l1: float) -> str:
        if l1 <= thresholds["L1_SIMILAR_MAX"]:
            return "Similar"
        if thresholds["L1_MEDIUM_MIN"] <= l1 <= thresholds["L1_MEDIUM_MAX"]:
            return "Medium"
        if l1 >= thresholds["L1_DIFFERENT_MIN"]:
            return "Different"
        gaps = [
            (abs(l1 - thresholds["L1_SIMILAR_MAX"]), "Similar"),
            (abs(l1 - thresholds["L1_MEDIUM_MIN"]), "Medium"),
            (abs(l1 - thresholds["L1_MEDIUM_MAX"]), "Medium"),
            (abs(l1 - thresholds["L1_DIFFERENT_MIN"]), "Different"),
        ]
        return sorted(gaps, key=lambda x: x[0])[0][1]

    rows = []
    for a, b in itertools.combinations(df_bfi["id"], 2):
        va = df_bfi[df_bfi["id"] == a].iloc[0]
        vb = df_bfi[df_bfi["id"] == b].iloc[0]
        vec_a = {
            k: va[k]
            for k in dims
        }
        vec_b = {
            k: vb[k]
            for k in dims
        }
        l1 = l1_distance(dims, vec_a, vec_b)
        cosine_sim_zscored = cosine_sim_zscore(df_bfi, dims, va, vb)
        cosine_sim_val = cosine_sim(dims, va, vb)

        rec = {
            "agent_a_id": a,
            "agent_b_id": b,
            "agent_a_name": va["mbti_profile"],
            "agent_b_name": vb["mbti_profile"],
            "agent_a_sloan": va["sloan_top_label"],
            "agent_b_sloan": vb["sloan_top_label"],
            "pair_L1": round(l1, 3),
            "pair_L1_bin": bin_by_l1(l1),
            "pair_cosine_zscore": round(cosine_sim_zscored, 6),
            "pair_cosine": round(cosine_sim_val, 6),
        }

        # add per-agent components
        for d in dims:
            rec[f"a_{d}"] = vec_a[d]
            rec[f"b_{d}"] = vec_b[d]

        rows.append(rec)
    return pd.DataFrame(rows)


def load_and_filter_scenarios(
    easy_path: str, hard_path: str, include_categories: List[str], n_reps: int
) -> pd.DataFrame:
    easy = pd.read_csv(easy_path)
    hard = pd.read_csv(hard_path)
    easy["difficulty"] = "Easy"
    hard["difficulty"] = "Hard"
    # easy["scenario_idx"] = np.arange(len(easy))
    # hard["scenario_idx"] = np.arange(len(hard))
    easy["scenario_idx"] = [i for i in range(len(easy))]
    hard["scenario_idx"] = [i for i in range(len(hard))]
    include_set = set([c.strip() for c in include_categories])

    easy = easy[easy["social_goal_category"].isin(include_set)].copy()
    hard = hard[hard["social_goal_category"].isin(include_set)].copy()


    # slots = pd.concat([easy, hard], ignore_index=True)
    slots = hard.copy()
    slots = slots.loc[slots.index.repeat(n_reps)].copy()
    slots["replicate_id"] = slots.groupby(["difficulty", "scenario_idx"]).cumcount() + 1
    slots["subcategory"] = slots["social_goal_category"]
    slots["slot_id"] = (
        slots["difficulty"].astype(str)
        + "_"
        + slots["subcategory"].astype(str).str.replace(" ", "")
        + "_"
        + "S"
        + slots["scenario_idx"].astype(int).astype(str).str.zfill(3)
        + "_"
        + "R"
        + slots["replicate_id"].astype(int).astype(str).str.zfill(2)
    )
    return slots


def sample_pairs_for_slots(
    slots,
    df_pairs,
    df_bfi,
    bin_targets,
    max_pair_overall,
    max_pair_per_cell,
    random_state,
):
    rng = np.random.default_rng(random_state)
    total = len(slots)
    target_counts = {k: int(round(v * total)) for k, v in bin_targets.items()}
    deficit = total - sum(target_counts.values())
    order = ["Medium", "Similar", "Different"]
    i = 0
    while deficit != 0:
        k = order[i % len(order)]
        target_counts[k] += 1 if deficit > 0 else -1
        deficit += -1 if deficit > 0 else 1
        i += 1

    used_counts, cell_counts = {}, {}
    agents = pd.unique(df_bfi["id"])
    exposure = {a: 0 for a in agents}
    bin_usage = {k: 0 for k in bin_targets.keys()}
    slots = slots.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    pairs_by_bin = {
        k: df_pairs[df_pairs["pair_L1_bin"] == k].copy() for k in bin_targets.keys()
    }

    assignments = []
    for _, slot in slots.iterrows():
        cell = (slot["subcategory"], slot["difficulty"])
        deficits = {k: target_counts[k] - bin_usage[k] for k in bin_targets.keys()}
        chosen_bin = max(deficits.items(), key=lambda kv: kv[1])[0]
        candidates = pairs_by_bin[chosen_bin]

        def ok_pair(row):
            a, b = row["agent_a_id"], row["agent_b_id"]
            if used_counts.get((a, b), 0) >= max_pair_overall:
                return False
            if cell_counts.get((cell, a, b), 0) >= max_pair_per_cell:
                return False
            return True

        cand = candidates[candidates.apply(ok_pair, axis=1)].copy()
        if cand.empty:
            for alt_bin, _ in sorted(deficits.items(), key=lambda kv: -kv[1]):
                alt = pairs_by_bin[alt_bin][
                    pairs_by_bin[alt_bin].apply(ok_pair, axis=1)
                ]
                if not alt.empty:
                    chosen_bin = alt_bin
                    cand = alt
                    break

        if cand.empty:

            def ok_relaxed(row):
                a, b = row["agent_a_id"], row["agent_b_id"]
                if used_counts.get((a, b), 0) >= max_pair_overall:
                    return False
                if cell_counts.get((cell, a, b), 0) >= 2:
                    return False
                return True

            cand = df_pairs[df_pairs.apply(ok_relaxed, axis=1)].copy()

        cand = cand.assign(
            load=cand.apply(
                lambda r: exposure[r["agent_a_id"]] + exposure[r["agent_b_id"]], axis=1
            )
        )
        min_load = cand["load"].min()
        cand = cand[cand["load"] == min_load]
        chosen = cand.sample(n=1, random_state=rng.integers(0, 2**32 - 1)).iloc[0]
        a, b = chosen["agent_a_id"], chosen["agent_b_id"]
        a_name, b_name = chosen["agent_a_name"], chosen["agent_b_name"]
        # if rng.random() < 0.5:
        #     a, b = b, a
        #     a_name, b_name = b_name, a_name
        
        assignments.append(
            {
                "slot_id": slot["slot_id"],
                "agent_a_id": a,
                "agent_b_id": b,
                "agent_a_name": a_name,
                "agent_b_name": b_name,
                "pair_L1": chosen["pair_L1"],
                "pair_L1_bin": chosen["pair_L1_bin"],
            }
        )
        used_counts[(a, b)] = used_counts.get((a, b), 0) + 1
        used_counts[(b, a)] = used_counts.get((b, a), 0) + 1
        cell_counts[(cell, a, b)] = cell_counts.get((cell, a, b), 0) + 1
        cell_counts[(cell, b, a)] = cell_counts.get((cell, b, a), 0) + 1
        exposure[a] += 1
        exposure[b] += 1
        bin_usage[chosen["pair_L1_bin"]] += 1

    df_assign = pd.DataFrame(assignments)
    # Merge once, then keep only the minimal columns you want to store
    merged = slots.merge(df_assign, on="slot_id", how="left")

    minimal_cols = [
        "scenario_idx",       # from slots
        "scenario",           # from slots
        "replicate_id",       # from slots
        "subcategory",        # from slots
        "difficulty",         # from slots
        "slot_id",            # from slots
        "agent_a_id",         # from df_assign
        "agent_b_id",         # from df_assign
        "agent_a_name",         # from df_assign
        "agent_b_name",         # from df_assign
        "pair_L1",            # from df_assign
        "pair_L1_bin",        # from df_assign
    ]
    return merged[minimal_cols].copy()


def run_personality_ga_assignment(
    characters_csv: str,
    easy_scenario_csv: str,
    hard_scenario_csv: str,
    include_categories: List[str],
    n_scenario_reps: int,
    max_pair_overall: int,
    max_pair_per_cell: int,
    pairs_output_csv: str,
    assignment_output_csv: str,
):
    print("Loading characters and building BFI table...")
    df_bfi = load_characters_bfi(characters_csv)

    print("Building pair table...")
    df_pairs = make_pair_table(df_bfi, L1_THRESHOLDS)
    df_pairs.to_csv(pairs_output_csv, index=False)
    print(f"Saved pair table to: {pairs_output_csv}")

    print("Loading and filtering scenarios...")
    slots = load_and_filter_scenarios(
        easy_scenario_csv, hard_scenario_csv, include_categories, n_scenario_reps
    )
    print(f"Total scenario slots: {len(slots)}")

    print("Sampling pairs for each slot...")
    schedule = sample_pairs_for_slots(
        slots,
        df_pairs,
        df_bfi,
        BIN_TARGETS,
        max_pair_overall,
        max_pair_per_cell,
        GLOBAL_SEED,
    )

    schedule.to_csv(assignment_output_csv, index=False)
    print(f"✅ Saved final schedule to: {assignment_output_csv}")

    # Diagnostics
    print("\n=== Bin counts ===")
    print(schedule["pair_L1_bin"].value_counts())
    print("\n=== Agent appearances ===")
    counts = (
        pd.concat(
            [
                schedule["agent_a_id"].value_counts(),
                schedule["agent_b_id"].value_counts(),
            ],
            axis=1,
            keys=["as_A", "as_B"],
        )
        .fillna(0)
        .sum(axis=1)
        .astype(int)
        .sort_values(ascending=False)
    )
    print(counts)


if __name__ == "__main__":

    cfg = config.pipeline.experiments.personality_ga
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    run_personality_ga_assignment(
        characters_csv=os.path.join(root, cfg.characters_filepath),
        easy_scenario_csv=os.path.join(root, cfg.easy_scenario_csv),
        hard_scenario_csv=os.path.join(root, cfg.hard_scenario_csv),
        include_categories=cfg.include_categories,
        n_scenario_reps=cfg.n_scenario_reps,
        max_pair_overall=cfg.max_pair_overall,
        max_pair_per_cell=cfg.max_pair_per_cell,
        pairs_output_csv=os.path.join(root, cfg.pairs_output_csv),
        assignment_output_csv=os.path.join(root, cfg.assignment_output_csv),
    )
