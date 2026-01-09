#!/usr/bin/env python3
"""
BFI (raw) + L1-distance pairing + scenario scheduling.

- Within each L1 bin (similar / medium / different / buffer), candidate pairs are
  selected RANDOMLY (seeded). No preference for most-similar/most-different.

- L1 thresholds:
    similar:   L1 <= 1.00
    medium:    1.25 <= L1 <= 2.00
    different: L1 >= 2.25
    buffer:    in-between gaps (1.00–1.25, 2.00–2.25)

- Ensures UNIQUE, UNDIRECTED pairs, and blocks self-pairs robustly.
- Stores personality tags (MBTI) and the pair’s L1 distance & bin in the final CSV.

Usage (adapt paths/params in main()):
  python schedule_bfi_l1_random_bins.py
"""

import os
import itertools
from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np

from charisma.config import config

GLOBAL_SEED = 42
# ---------------------------
# L1 thresholds
# ---------------------------
L1_THRESHOLDS = dict(
    L1_SIMILAR_MAX=1.00,
    L1_MEDIUM_MIN=1.25,
    L1_MEDIUM_MAX=2.00,
    L1_DIFFERENT_MIN=2.25,
)


# ---------------------------
# Helpers
# ---------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _get_col(df: pd.DataFrame, name: str) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    return low.get(name.lower(), None)


def _check_degree_feasibility(n_agents: int, d: int):
    if n_agents <= 1:
        raise ValueError("Need at least 2 agents.")
    if d < 0 or d >= n_agents:
        raise ValueError(f"n_partners must be in [0, {n_agents-1}]. Got {d}.")
    if (n_agents * d) % 2 != 0:
        raise ValueError(f"n_agents * n_partners must be even. Got {n_agents} * {d} = {n_agents*d} (odd).")


def _l1_distance(u: np.ndarray, v: np.ndarray) -> float:
    # Manhattan distance on RAW BFI (no standardization)
    return float(np.abs(u - v).sum())


def _load_bfi_matrix(df: pd.DataFrame) -> np.ndarray:
    # Required BFI columns (case-insensitive)
    req = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    cols = []
    for n in req:
        col = _get_col(df, n)
        if col is None:
            raise ValueError(f"characters.csv missing BFI column '{n}'. Found columns: {list(df.columns)}")
        cols.append(col)
    X = df[cols].astype(float).to_numpy()
    return X  # raw values in [0, 1]


def _bin_label_for_l1(d: float) -> str:
    t = L1_THRESHOLDS
    if d <= t["L1_SIMILAR_MAX"]:
        return "similar"
    if t["L1_MEDIUM_MIN"] <= d <= t["L1_MEDIUM_MAX"]:
        return "medium"
    if d >= t["L1_DIFFERENT_MIN"]:
        return "different"
    return "buffer"  # gaps: (1.00–1.25) or (2.00–2.25)


def _build_l1_bins(agent_ids: List[str], X: np.ndarray) -> Dict[Tuple[str, str], Tuple[float, str]]:
    """
    Map all unordered pairs to (L1 distance, bin). Keys use the agent ID values.
    """
    pairs = list(itertools.combinations(range(len(agent_ids)), 2))
    out = {}
    for i, j in pairs:
        d = _l1_distance(X[i], X[j])
        lab = _bin_label_for_l1(d)
        out[(agent_ids[i], agent_ids[j])] = (d, lab)
    return out


def _random_degree_constrained_pairs(agent_ids: List[str], d: int, rng: np.random.Generator,
                                     max_restarts: int = 300) -> List[Tuple[str, str]]:
    """Degree-constrained random simple graph (each node has degree d)."""
    _check_degree_feasibility(len(agent_ids), d)

    def canon(a, b):
        return (a, b) if a < b else (b, a)

    for _ in range(max_restarts):
        deg = {a: d for a in agent_ids}
        edges = set()
        success = True

        while True:
            need = [a for a in agent_ids if deg[a] > 0]
            if not need:
                break
            # Most constrained first; random tie-break to diversify search
            need.sort(key=lambda a: (-deg[a], rng.random()))
            progressed = False
            for a in need:
                cands = [b for b in agent_ids if b != a and deg[b] > 0 and canon(a, b) not in edges]
                if not cands:
                    success = False
                    break
                rng.shuffle(cands)
                b = cands[0]
                if a == b:
                    continue
                edges.add(canon(a, b))
                deg[a] -= 1
                deg[b] -= 1
                progressed = True
            if not success or not progressed:
                success = False
                break

        if success:
            return sorted(list(edges))

    raise RuntimeError("Failed to construct degree-constrained pairs. Try reducing n_partners or change --seed.")


def _targets_for(d: int) -> Dict[str, int]:
    """
    Per-agent desired mix across bins following approximately:
      40% similar, 20% medium, 40% different.
    Rounds to integers that sum to d (largest-remainder style).
    """
    # ideal (continuous) counts
    ideal = {
        "similar":   0.40 * d,
        "medium":    0.20 * d,
        "different": 0.40 * d,
    }

    # floors + remainders
    out = {k: int(np.floor(v)) for k, v in ideal.items()}
    rem = d - sum(out.values())
    if rem > 0:
        fracs = {k: ideal[k] - out[k] for k in ideal.keys()}
        # give the leftover slots to the largest fractional parts
        for k in sorted(fracs.keys(), key=lambda x: -fracs[x]):
            if rem <= 0:
                break
            out[k] += 1
            rem -= 1
    return out


def _distance_mixed_pairs_l1(agent_ids: List[str], d: int, X: np.ndarray,
                             rng: np.random.Generator) -> List[Tuple[str, str]]:
    """
    New pairing:
      1) Build ALL unique pairs with L1 + bin tag.
      2) For each agent, enforce per-agent bin quotas from _targets_for(d) (~40/20/40).
      3) Assign edges bin-by-bin (scarcest bin first), avoiding self/dupes, respecting quotas and degrees.
      4) Fill residual degrees with any remaining feasible edges.
      Prints per-agent achieved distribution.

    Returns sorted UNIQUE undirected edges (a<b).
    """
    # ---------- helpers ----------
    def canon(a, b): return (a, b) if a < b else (b, a)

    # ---------- 1) all pairs with L1 + bin ----------
    pair_info = _build_l1_bins(agent_ids, X)  # (a,b) -> (dist, lab)
    # compress to canonical (a<b) rows
    all_pairs = []
    for (a, b), (dist, lab) in pair_info.items():
        if a == b: 
            continue
        a0, b0 = canon(a, b)
        all_pairs.append((a0, b0, float(dist), lab))
    # group by bin
    pairs_by_bin = {"similar": [], "medium": [], "different": [], "buffer": []}
    for a, b, dval, lab in all_pairs:
        pairs_by_bin[lab].append((a, b, dval))

    # ---------- 2) per-agent quotas & degrees ----------
    quotas = {a: _targets_for(d).copy() for a in agent_ids}             # {'similar':x,'medium':y,'different':z}
    deg_left = {a: d for a in agent_ids}
    used = set()
    chosen = []

    # availability per agent per bin (neighbors)
    avail = {a: {"similar": set(), "medium": set(), "different": set(), "buffer": set()} for a in agent_ids}
    for lab in ["similar","medium","different","buffer"]:
        for a, b, _ in pairs_by_bin[lab]:
            avail[a][lab].add(b); avail[b][lab].add(a)

    # randomize candidate orders (repeatable)
    for lab in ["similar","medium","different","buffer"]:
        rng.shuffle(pairs_by_bin[lab])
    for a in agent_ids:
        for lab in ["similar","medium","different","buffer"]:
            L = list(avail[a][lab]); rng.shuffle(L); avail[a][lab] = L

    # choose bin order by global scarcity: required units vs available edges
    total_quota = {"similar": 0, "medium": 0, "different": 0}
    for a in agent_ids:
        for lab in total_quota.keys():
            total_quota[lab] += quotas[a].get(lab, 0)
    global_avail = {lab: len(pairs_by_bin[lab]) for lab in total_quota.keys()}
    # smaller (avail / required) means scarcer; put first
    def scarcity_score(lab):
        req = max(1, total_quota[lab])
        return global_avail[lab] / req
    bin_order = sorted(["different","similar","medium"], key=lambda b: scarcity_score(b))

    # track achieved per-agent per-bin
    got = {a: {"similar": 0, "medium": 0, "different": 0} for a in agent_ids}

    def can_add(a, b):
        if a == b:
            return False
        if deg_left[a] <= 0 or deg_left[b] <= 0:
            return False
        return canon(a, b) not in used

    # ---------- 3) assign bin-by-bin (scarcest first) ----------
    for lab in bin_order:
        # per-agent remaining need in this bin
        need = {a: max(0, quotas[a].get(lab, 0)) for a in agent_ids}
        if sum(need.values()) == 0 or not pairs_by_bin[lab]:
            continue

        # Build neighbor lists limited to this bin
        neigh = {a: [b for b in avail[a][lab]] for a in agent_ids}

        # Agents with demand in this bin, sorted by (need ascending, then fewest neighbors)
        agents_list = [a for a in agent_ids if need[a] > 0]
        agents_list.sort(key=lambda a: (need[a], len(neigh[a]), rng.random()))

        progressed = True
        while progressed:
            progressed = False
            # re-sort each loop as needs change
            agents_list = [a for a in agent_ids if need[a] > 0 and deg_left[a] > 0]
            if not agents_list:
                break
            agents_list.sort(key=lambda a: (need[a], len(neigh[a]), rng.random()))
            for a in agents_list:
                if need[a] <= 0 or deg_left[a] <= 0:
                    continue
                # feasible partners in this bin who also need it (or at least have degree left)
                feas_need = [b for b in neigh[a] if can_add(a, b) and need.get(b, 0) > 0]
                if not feas_need:
                    # if no mutual-need partner, allow any with degree left in this bin
                    feas_any = [b for b in neigh[a] if can_add(a, b)]
                    if not feas_any:
                        continue
                    # prefer partner with higher remaining need first, then higher deg_left
                    feas_any.sort(key=lambda b: (need.get(b, 0), deg_left[b], rng.random()), reverse=True)
                    b = feas_any[0]
                else:
                    # both need this bin: pick the partner with highest need then highest deg_left
                    feas_need.sort(key=lambda b: (need[b], deg_left[b], rng.random()), reverse=True)
                    b = feas_need[0]

                # assign
                used.add(canon(a, b))
                chosen.append(canon(a, b))
                deg_left[a] -= 1; deg_left[b] -= 1
                need[a] -= 1; need[b] = max(0, need[b] - 1)
                got[a][lab] += 1; got[b][lab] += 1
                # drop from local views to reduce future checks
                try: neigh[a].remove(b)
                except ValueError: pass
                try: neigh[b].remove(a)
                except ValueError: pass
                progressed = True

    # ---------- 4) fill remaining degrees with any feasible edges ----------
    if any(deg_left[a] > 0 for a in agent_ids):
        # recompute deficits to prioritize filling that helps remaining bin quotas
        def bin_deficit(a, lab): return max(0, quotas[a].get(lab, 0) - got[a][lab])
        # union of all leftover candidates not used yet
        leftover = []
        for lab in ["similar","different","medium","buffer"]:
            for a, b, _ in pairs_by_bin[lab]:
                k = canon(a, b)
                if k in used:
                    continue
                leftover.append((a, b, lab))
        rng.shuffle(leftover)

        # greedy fill
        for a, b, lab in sorted(
            leftover,
            key=lambda t: (
                # prefer edges that satisfy *both* parties' remaining bin deficits
                (bin_deficit(t[0], t[2]) > 0) + (bin_deficit(t[1], t[2]) > 0),
                deg_left[t[0]] + deg_left[t[1]],
                rng.random()
            ),
            reverse=True
        ):
            if all(v == 0 for v in deg_left.values()):
                break
            if not can_add(a, b):
                continue
            used.add(canon(a, b))
            chosen.append(canon(a, b))
            deg_left[a] -= 1; deg_left[b] -= 1
            if lab in got[a]: got[a][lab] += 1
            if lab in got[b]: got[b][lab] += 1

    # ---------- finalize ----------
    edges = sorted(list({canon(a, b) for (a, b) in chosen if a != b}))

    # per-agent achieved + print
    print("\n=== Pair distribution per agent (achieved) ===")
    deg_realized = {a: 0 for a in agent_ids}
    per_agent_bins = {a: {"similar": 0, "medium": 0, "different": 0} for a in agent_ids}
    def pair_lab(a,b):
        return (_build_l1_bins([a,b], X[[agent_ids.index(a), agent_ids.index(b)]])  # tiny reuse
                .get((a,b)) or (0.0,"buffer"))[1]
    # faster: use pair_info we already have
    def lab_of(a,b):
        di = pair_info.get((a,b)) or pair_info.get((b,a))
        return di[1] if di else "buffer"

    for a, b in edges:
        lab = lab_of(a, b)
        if lab in per_agent_bins[a]: per_agent_bins[a][lab] += 1
        if lab in per_agent_bins[b]: per_agent_bins[b][lab] += 1
        deg_realized[a] += 1; deg_realized[b] += 1

    for a in sorted(agent_ids, key=lambda x: str(x)):
        tgt = _targets_for(d)
        s, m, df = per_agent_bins[a]["similar"], per_agent_bins[a]["medium"], per_agent_bins[a]["different"]
        unmet = {"similar": max(0, tgt["similar"]-s),
                 "medium": max(0, tgt["medium"]-m),
                 "different": max(0, tgt["different"]-df)}
        unmet_str = "" if sum(unmet.values())==0 else f" | unmet={unmet}"
        print(f"Agent {a}: total={deg_realized[a]} | similar={s}  medium={m}  different={df} | targets={tgt}{unmet_str}")

    # global mix
    from collections import Counter
    cnt = Counter(lab_of(a, b) for (a, b) in edges)
    total = len(edges) if edges else 1
    print("\n=== Global bin mix (pairs) ===")
    print({k: f"{cnt.get(k,0)} ({cnt.get(k,0)/total:.1%})" for k in ["similar","medium","different"]})

    # safety: exact degree
    expected = (len(agent_ids) * d) // 2
    if len(edges) != expected or any(deg_realized[a] != d for a in agent_ids):
        print("[WARN] Degree unmet; falling back to randomized degree-only.")
        return _random_degree_constrained_pairs(agent_ids, d, rng)

    return edges




# ---------------------------
# Main
# ---------------------------
def main():
    # ---- CONFIGURE HERE ----
    cfg = config.pipeline.experiments.robustness_2
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    seed = GLOBAL_SEED
    characters_csv = os.path.join(root, cfg.characters_filepath)
    scenarios_csv = os.path.join(root, cfg.scenario_csv)
    n_partners = cfg.n_partners
    include_categories = cfg.include_categories  # scenarios.social_goal_category
    n_scenarios = cfg.n_scenarios
    assignment_output_csv = os.path.join(root, cfg.assignment_output_csv)
    # ------------------------

    rng = np.random.default_rng(seed)

    # Load characters
    chars = pd.read_csv(characters_csv)
    id_col = _get_col(chars, "id")
    disp_col = _get_col(chars, "mbti_profile")


    # Normalize IDs; fail on duplicates
    chars["__id__"] = chars[id_col].astype(str).str.strip()
    dup_counts = chars["__id__"].value_counts()
    dups = dup_counts[dup_counts > 1]
    if len(dups) > 0:
        raise ValueError(f"Duplicate agent IDs detected: {list(dups.index)}")
    chars["__display__"] = chars[disp_col].astype(str)

    # BFI (raw)
    X = _load_bfi_matrix(chars)
    agent_ids = chars["__id__"].tolist()
    agent_name = dict(zip(chars["__id__"], chars["__display__"]))

    # Feasibility
    _check_degree_feasibility(len(agent_ids), n_partners)

    # Build L1-based pairs (random selection inside bins)
    edges = _distance_mixed_pairs_l1(agent_ids, n_partners, X, rng)

    # Build a lookup for pair distance/bin so we can save them
    # Use canonical (min_id, max_id) keys
    pair_info_raw = _build_l1_bins(agent_ids, X)  # (idA,idB)->(dist,bin)
    def canon(a, b): return (a, b) if a < b else (b, a)
    pair_info = {canon(a, b): pair_info_raw.get((a, b), pair_info_raw.get((b, a))) for (a, b) in pair_info_raw.keys()}

    # Load scenarios
    sc = pd.read_csv(scenarios_csv)
    sc_cat = _get_col(sc, "social_goal_category")
    sc_title = _get_col(sc, "title")
    sc_id = _get_col(sc, "id")

    # If 'id' missing, auto-create from row order starting at 1 (deterministic)
    if sc_id is None:
        sc = sc.reset_index(drop=True)
        sc.insert(0, "id", np.arange(1, len(sc) + 1))
        sc_id = "id"

    # Filter categories if requested
    if include_categories:
        sc = sc[sc[sc_cat].astype(str).isin(include_categories)].copy()
        if sc.empty:
            raise ValueError("No scenarios left after include-categories filter.")

    # Sample scenarios (without replacement), deterministic w.r.t. seed
    if len(sc) < n_scenarios:
        raise ValueError(f"Requested n_scenario={n_scenarios} but only {len(sc)} scenarios available after filtering.")
    idxs = rng.choice(sc.index.values, size=n_scenarios, replace=False)
    sc_sel = sc.loc[idxs].copy()
    sc_sel = sc_sel.sort_values(by=sc_id).reset_index(drop=True)

    # Build assignments: each pair × each scenario
    def dname(a): return agent_name.get(a, a)
    edges_sorted = sorted(edges, key=lambda e: (dname(e[0]), dname(e[1])))

    rows = []
    for pair_idx, (a, b) in enumerate(edges_sorted, start=1):
        if a == b:
            continue
        pair_id = f"pair_{pair_idx:03d}"

        # Lookup pair distance & bin
        info = pair_info.get(canon(a, b))
        if info is None:
            # compute on the fly if not present (shouldn't happen)
            i = agent_ids.index(a); j = agent_ids.index(b)
            d = _l1_distance(X[i], X[j])
            lab = _bin_label_for_l1(d)
            info = (d, lab)
        pair_L1, pair_bin = float(info[0]), str(info[1])

        for idx, srow in sc_sel.iterrows():
            sid = str(srow[sc_id])
            scat = str(srow[sc_cat]) if sc_cat and not pd.isna(srow[sc_cat]) else ""
            rows.append({
                "condition_idx": len(rows),
                "pair_id": pair_id,
                "agent_a_id": a,
                "agent_a_name": dname(a),
                "agent_b_id": b,
                "agent_b_name": dname(b),
                "pair_L1": round(pair_L1, 3),          # <-- distance
                "pair_L1_bin": pair_bin,               # <-- similar/medium/different/buffer
                "scenario_idx": sid,
                "scenario": srow["scenario"],
                "subcategory": scat
            })

    out = pd.DataFrame(rows).sort_values(["pair_id", "scenario_idx"]).reset_index(drop=True)

    # Final sanity: ensure no self-pairs
    if (out["agent_a_id"] == out["agent_b_id"]).any():
        raise RuntimeError("Self-pairs found in final output; investigate duplicates in agent IDs.")

    # Ensure output directory exists
    ensure_dir(os.path.dirname(assignment_output_csv))
    out.to_csv(assignment_output_csv, index=False)

    # Diagnostics
    n_pairs = len(edges_sorted)
    print(f"Wrote {len(out)} rows to {assignment_output_csv}")
    print(f"Unique pairs: {n_pairs}")
    print(f"Scenarios per pair: {n_scenarios}")
    print(f"Total conversations: {n_pairs * n_scenarios}")

    # Degree check (each agent must appear exactly n_partners times)
    deg = {}
    for a, b in edges_sorted:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    bad = {k: v for k, v in deg.items() if v != n_partners}
    if bad:
        print("WARNING: degree mismatch for these agent IDs:", bad)


if __name__ == "__main__":
    main()
