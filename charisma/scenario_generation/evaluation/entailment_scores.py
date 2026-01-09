from typing import Dict, Tuple
from dataclasses import dataclass
from transformers import pipeline
import pandas as pd
import ast
from logging import getLogger
log = getLogger(__name__)
@dataclass
class AlignmentScore:
    label: str
    score: float

def align_schema_NLI(
    scenario: str,
    schema: Dict[str, str],
    model_name: str = "facebook/bart-large-mnli",
    multi_label: bool = False
) -> Tuple[Dict[str, AlignmentScore], float]:
    """
    Runs NLI entailment to test alignment of scenario with each schema slot.

    :param scenario: the generated scenario text
    :param schema: dict with keys matching built hypotheses
    :param threshold: min entailment score to consider aligned
    :param multi_label: if True, analyze each slot independently
    :return: per-slot AlignmentScore and overall average entailment score
    """
    nli_model = pipeline(
        task="zero-shot-classification",
        model=model_name,
        multi_label=multi_label
    )

    hypotheses = schema
    slot_scores = {}
    total = 0.0
    count = 0

    scopes = list(hypotheses.items())
    for slot_key, hypothesis in scopes:
        result = nli_model(scenario, candidate_labels=[hypothesis], hypothesis_template="{}")
        # HuggingFace returns `labels` and `scores`
        label = result["labels"][0]
        score = result["scores"][0]
        slot_scores[slot_key] = AlignmentScore(label=label, score=score)
        total += score
        count += 1

    average_entailment = total / count if count > 0 else 0.0
    return slot_scores, average_entailment

def calculate_entailment_scores(
    path_to_csv_file: str,
    output_csv_file: str
):
    # Load the CSV file
    df = pd.read_csv(path_to_csv_file)
    # Iterate through each row and get the scenario and scenario setting
    results = []
    for idx, row in df.iterrows():
        # scenario = row['scenario']
        scenario = ast.literal_eval(row["scenario"])['scenario_context']
        schema = {
            "shared_goal": row['shared_goal'],
            "social_goal_category": row['social_goal_category'],
            "first_agent_goal": row['first_agent_goal'],
            "second_agent_goal": row['second_agent_goal'],
            "first_agent_role": row['agent1_role'],
            "second_agent_role": row['agent2_role']
        }
        
        # Get the alignment scores
        slot_scores, average_entailment = align_schema_NLI(scenario, schema)
        log.info(f"Processed row {idx + 1}/{len(df)}: {average_entailment:.4f} average entailment score")
        output_record = row.to_dict()
        output_record.update({
            "average_entailment_score": average_entailment,
            "slot_entailment_scores": {f"{key}_score": score.score for key, score in slot_scores.items()}
        })
        results.append(output_record)
        # Save all results to the output CSV file
        pd.DataFrame(results).to_csv(output_csv_file, index=False)
        
if __name__ == "__main__":
    calculate_entailment_scores("outputs/scenarios_database.csv", "outputs/scenario_evaluation/entailment/entailment_scores.csv")