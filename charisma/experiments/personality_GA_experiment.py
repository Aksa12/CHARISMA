#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
import time
import json
import ast
from typing import Any, Dict, List, Optional, Callable

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from charisma.interaction_generation.generation import run_interaction_pipeline
from charisma.evaluation.generation import evaluate_conversation_app
from charisma.config import config

# =========================
# USER CONFIG
# =========================
GLOBAL_SEED = 42                                     # per-slot seed = GLOBAL_SEED + schedule index
load_dotenv()
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1", api_key=OPEN_ROUTER_API_KEY
)

# =========================
# CSV helpers
# =========================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def append_row_csv(path: str, row: Dict[str, Any], header: List[str]):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})

def upsert_row_csv(path: str, key_col: str, key_val: str, row: Dict[str, Any], header: List[str]):
    """Upsert a row by key to keep resume behavior simple and robust."""
    if not os.path.exists(path):
        append_row_csv(path, row, header)
        return
    df = pd.read_csv(path, dtype=str)
    if key_col not in df.columns:
        append_row_csv(path, row, header)
        return
    mask = (df[key_col].astype(str) == str(key_val))
    row_df = pd.DataFrame([{k: str(row.get(k, "")) for k in header}])
    if mask.any():
        df.loc[mask, header] = row_df.values
    else:
        df = pd.concat([df, row_df], ignore_index=True)
    df.to_csv(path, index=False)

def read_done_slot_ids(results_csv: str) -> set:
    if not os.path.exists(results_csv):
        return set()
    df = pd.read_csv(results_csv, dtype=str)
    if "row_id" not in df.columns or "status" not in df.columns:
        return set()
    return set(df.loc[df["status"] == "ok", "row_id"].astype(str))

# =========================
# Loaders (scenarios/characters)
# =========================

def load_characters(characters_csv: str) -> pd.DataFrame:
    """Load characters; you can enrich this if your generator needs more fields."""
    return pd.read_csv(characters_csv)

def build_combined_scenarios(easy_csv: str, hard_csv: str) -> pd.DataFrame:
    """
    Combine Easy and Hard into one table with a stable scenario_idx:
      - easy indices: 0 .. len(easy)-1
      - hard indices: len(easy) .. len(easy)+len(hard)-1
    This must match how the schedule was created.
    """
    easy = pd.read_csv(easy_csv)
    hard = pd.read_csv(hard_csv)

    easy = easy.reset_index(drop=True).copy()
    hard = hard.reset_index(drop=True).copy()
    easy["difficulty"] = "Easy"
    hard["difficulty"] = "Hard"

    easy["scenario_idx"] = range(len(easy))
    hard["scenario_idx"] = range(len(easy), len(easy) + len(hard))

    combined = pd.concat([easy, hard], ignore_index=True)
    # Normalize column names we’ll pass to the generator/evaluator
    # Expecting at least:
    # base_shared_goal, social_goal_category, explanation, shared_goal,
    # first_agent_goal, second_agent_goal, agent1_role, agent2_role, scenario
    return combined

# def lookup_scenario_row(combined_df: pd.DataFrame, scenario_idx: int, scenario_text: str) -> pd.Series:
#     """Find scenario row by scenario_idx; if missing, try fallback by scenario text."""
#     df = combined_df
#     if "scenario_idx" in df.columns:
#         m = df[df["scenario_idx"] == scenario_idx]
#         if len(m) == 1:
#             return m.iloc[0]
#     # Fallback: match by scenario text (if unique)
#     if "scenario" in df.columns:
#         m2 = df[df["scenario"].astype(str) == str(scenario_text)]
#         if len(m2) >= 1:
#             return m2.iloc[0]
#     raise KeyError(f"Scenario not found for scenario_idx={scenario_idx} (and fallback by text failed).")

def lookup_scenario_row(scenario_idx: int, scenarios_df: pd.DataFrame) -> pd.Series:
    row = scenarios_df.iloc[scenario_idx]
    return row


# =========================
# Results schema
# =========================

RESULTS_COLS = [
    # schedule identifiers
    "row_id", "scenario_idx",
    "agent_a_id", "agent_b_id", "agent_a_name", "agent_b_name", "pair_id", "pair_type_index", "pair_type_label", "pair_local_index",
    # scenario details loaded on demand
    "base_shared_goal", "social_goal_category", "explanation",
    "shared_goal", "first_agent_goal", "second_agent_goal", "agent1_role", "agent2_role", "scenario", "average_entailment_score",
    "difficulty", "model_name",
    # run metadata
    "seed", "status", "created_at",
    # outputs
    "interaction_history", "evaluation_json",
]


def ensure_results_header(results_csv: str):
    if os.path.exists(results_csv):
        return
    # create file with just the header, no empty row
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_COLS)
        w.writeheader()

# =========================
# Runner
# =========================

def run_with_schedule(
    schedule_csv: str,
    characters_csv: str,
    scenarios_csv: str,
    behavioral_coding_csv: str,
    results_csv: str,
    model: str,
    provider: str = None,
):
    ensure_results_header(results_csv)

    # Load static inputs
    schedule = pd.read_csv(schedule_csv, dtype=str)
    # schedule = schedule.sort_values("row_id").reset_index(drop=True)
    chars = load_characters(characters_csv)
    scenarios_df = pd.read_csv(scenarios_csv)

    # Resume: skip done slot_ids
    done_ids = read_done_slot_ids(results_csv)
    total = len(schedule)
    pending_idxs = [i for i in range(total) if schedule.loc[i, "row_id"] not in done_ids]

    print(f"[Runner] schedule={total} done={len(done_ids)} pending={len(pending_idxs)}")

    processed = 0
    errors = 0

    for i in pending_idxs:

        srow = schedule.loc[i]
        # Parse minimal fields
        row_id = str(srow["row_id"])
        scenario_idx = int(srow["scenario_idx"])
        pair_id = int(srow["pair_id"])
        pair_type_index = int(srow["pair_type_index"])
        pair_type_label = srow["pair_type_label"]
        pair_local_index = int(srow["pair_local_index"])
        agent_a_id = str(srow["agent_a_id"])
        agent_b_id = str(srow["agent_b_id"])
        difficulty = str(srow["difficulty"])
        model_name = str(srow["model_name"])

        # Reconstruct scenario row (full details)
        scen = lookup_scenario_row(scenario_idx, scenarios_df)
        scenario_data = scen.to_dict()
        # scenario_data = {
        #     "base_shared_goal": scen.get("base_shared_goal", ""),
        #     "social_goal_category": scen.get("social_goal_category", ""),
        #     "explanation": scen.get("explanation", ""),
        #     "shared_goal": scen.get("shared_goal", ""),
        #     "first_agent_goal": scen.get("first_agent_goal", ""),
        #     "second_agent_goal": scen.get("second_agent_goal", ""),
        #     "agent1_role": scen.get("agent1_role", ""),
        #     "agent2_role": scen.get("agent2_role", ""),
        #     "scenario": scen.get("scenario", ""),
        # }

        # Pull character rows (customize fields your generator needs)
        # Assumes characters.csv has an ID column named 'agent_id' (adjust if different)
        def row_by_id(df: pd.DataFrame, col: str, val: str) -> Dict[str, Any]:
            m = df[df[col].astype(str) == str(val)]
            if len(m) == 0:
                raise KeyError(f"Character not found: {col}={val}")
            return m.iloc[0].to_dict()

        id_col = "agent_id" if "agent_id" in chars.columns else "id"
        agent_a = row_by_id(chars, id_col, agent_a_id)
        agent_b = row_by_id(chars, id_col, agent_b_id)
        agent_a_name = str(agent_a["mbti_profile"])
        agent_b_name = str(agent_b["mbti_profile"])
        subcategory_a = str(agent_a["subcategory"])
        subcategory_b = str(agent_b["subcategory"])
        personality_a = {
            'Openness': agent_a['Openness'],
            'Conscientiousness': agent_a['Conscientiousness'],
            'Extraversion': agent_a['Extraversion'],
            'Agreeableness': agent_a['Agreeableness'],
            'Neuroticism': agent_a['Neuroticism'],
        }
        personality_b = {
            'Openness': agent_b['Openness'],
            'Conscientiousness': agent_b['Conscientiousness'],
            'Extraversion': agent_b['Extraversion'],
            'Agreeableness': agent_b['Agreeableness'],
            'Neuroticism': agent_b['Neuroticism'],
        }


        # Deterministic per-slot seed
        seed = int(GLOBAL_SEED + i)

        # Prepare base row for results
        row_out = {
            "row_id": row_id,
            "scenario_idx": scenario_idx,
            "pair_id": pair_id,
            "pair_type_index": pair_type_index,
            "pair_type_label": pair_type_label,
            "pair_local_index": pair_local_index,
            "agent_a_id": agent_a_id,
            "agent_b_id": agent_b_id,
            "agent_a_name": agent_a_name,
            "agent_b_name": agent_b_name,
            "base_shared_goal": scenario_data["base_shared_goal"],
            "social_goal_category": scenario_data["social_goal_category"],
            "explanation": scenario_data["explanation"],
            "shared_goal": scenario_data["shared_goal"],
            "first_agent_goal": scenario_data["first_agent_goal"],
            "second_agent_goal": scenario_data["second_agent_goal"],
            "agent1_role": scenario_data["agent1_role"],
            "agent2_role": scenario_data["agent2_role"],
            "scenario": scenario_data["scenario"],
            "difficulty": difficulty,
            "average_entailment_score": scenario_data.get("average_entailment_score", ""),
            "model_name": model_name,
            "seed": seed,
            "status": "started",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "interaction_history": "",
            "evaluation_json": "",
        }

        status = "ok"
        try:
            interaction_output = run_interaction_pipeline(
                    behavioral_coding_filename=behavioral_coding_csv,
                    agent1_name=agent_a_name,
                    agent2_name=agent_b_name,
                    subcategory_a=subcategory_a,
                    subcategory_b=subcategory_b,
                    personality_a=personality_a,
                    personality_b=personality_b,
                    scenario_data=scenario_data,
                    model=model,
                    provider=provider,
                    max_turns=20,
                    num_samples=None
                )
        except Exception as e:
                interaction_output = []
                status = "interaction_failed"
                raise e
        print(f"\ninteraction_output: {interaction_output}")
        
        evaluation_output = {}
        if status == "ok":
            try:
                scenario_setting = {
                    "shared_goal": scenario_data["shared_goal"],
                    "chosen_goal_category": scenario_data["social_goal_category"],
                    "first_agent_goal": scenario_data["first_agent_goal"],
                    "second_agent_goal": scenario_data["second_agent_goal"],
                    "first_agent_role": scenario_data["agent1_role"],
                    "second_agent_role": scenario_data["agent2_role"],
                }
                scenario = ast.literal_eval(scenario_data["scenario"])['scenario_context']
                interaction = [{"agent": d["agent"], "response": d["response"]} for d in interaction_output]
                evaluation_output = evaluate_conversation_app(scenario_setting, scenario, interaction, client, model, provider)
            except Exception as e:
                evaluation_output = {"error": str(e)}
                status = "evaluation_failed"
                raise e
        
        print(f"evaluation_output: {evaluation_output}")

        row_out.update({
            "status": status,
            "interaction_history": json.dumps(interaction_output),
            "evaluation_json": json.dumps(evaluation_output, ensure_ascii=False),
        })

        # append row
        append_row_csv(results_csv, row_out, RESULTS_COLS)


        processed += 1
        pct = 100.0 * (len(done_ids) + processed) / total
        print(f"[Runner] [{len(done_ids)+processed}/{total}] {row_id} -> {row_out['status']} ({pct:.1f}%)")


    print(f"[Runner] Finished. newly_processed={processed}, errors={errors}, total_done={len(done_ids)+processed}/{total}")

if __name__ == "__main__":

    cfg = config.pipeline.experiments.personality_ga
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    run_with_schedule(
        schedule_csv=os.path.join(root, cfg.assignment_output_csv),
        characters_csv=os.path.join(root, cfg.characters_filepath),
        scenarios_csv=os.path.join(root, cfg.scenarios_csv),
        behavioral_coding_csv=os.path.join(root,  config.pipeline.behavioral_coding_csv),
        results_csv=os.path.join(root, cfg.results_csv),
        model=config.pipeline.model,
        provider=config.pipeline.provider if config.pipeline.provider else None,
    )
