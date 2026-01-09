import json
from openai import OpenAI
from charisma.util import extract_json_string


def evaluation_prompt(
    scenario_setting: dict,
    scenario: str,
    conversation: str,
    client: OpenAI,
    model_name="deepseek/deepseek-chat-v3-0324:free",
    provider=None,
):  # Define the JSON structure for the result
    # User input defining the task
    json_format = {
        "shared_goal_achievement_score": "integer (0-10)",
        "reasoning": "string (brief explanation of how the shared goal was or was not achieved)",
        "confidence_shared": "float (0.0 - 1.0)",
        "Agent A": {
            "personal_goal_achievement_score": "integer (0-10)",
            "reasoning": "string (explanation referencing specific dialogue evidence)",
            "confidence_shared": "float (0.0 - 1.0)"
        },
        "Agent B": {
            "personal_goal_achievement_score": "integer (0-10)",
            "reasoning": "string (explanation referencing specific dialogue evidence)",
            "confidence_shared": "float (0.0 - 1.0)"
        }
    }

    user_message = f"""
    ### GOAL: ###
    We aim to explore attribution theory in social psychology by simulating interactions between psychologically profiled, role-playing agents. These agents are situated in interpersonal scenarios involving shared and individual goals. The objective is to analyze how dispositional traits and situational factors interact to shape social behavior.

    ### TASK: ###
    Given the following scenario setting, your task is to evaluate the goal achievement in the following conversation between two agents within the following social scenario. In the conversation, each agent attempts to achieve:
    - An individual personal goal
    - A shared goal

    You must assess:
    1. Each agent's personal goal achievement (on a scale from 0 to 10).
    2. The shared goal achievement (on a scale from 0 to 10).

    ### SCORING GUIDELINES: ###

    Personal Goal Achievement (per agent):
    0:	  Agent made no observable progress or acted counter to their goal.
    1 – 3:	  Minimal progress or inconsistent pursuit of goal.
    4 – 6:	  Partial progress; achieved some aspects but with noticeable shortcomings.
    7 – 9:	  Largely successful with minor limitations.
    10:	  Fully achieved the personal goal as intended in the scenario.

    Shared Goal Achievement:
    0:	  The shared goal was not achieved at all; interaction broke down.
    1 – 3:	  Partial engagement with shared goal but no substantive success.
    4 – 6:	  Moderate achievement; partial or mixed outcome.
    7 – 9:	  Substantial progress toward the shared goal.
    10:	  Full mutual success in achieving the shared goal.

    ### INSTRUCTIONS: ###
    1. Base your judgment only on observable conversational evidence. It means you must avoid assumptions about unstated intentions.
    2. Provide a brief reasoning grounded in behavior for each agent to justify the personal goal achievement score and the shared goal achievement score. You must reference specific dialogue elements that support each score (e.g., “Agent A explicitly acknowledged compromise…”).
    3. Evaluate each agent independently, and assess the shared goal holistically.
    4. All goal completion scores must be integers between 0 and 10.
    5. Confidence Score: Include a "confidence" level (0.0 – 1.0) for each evaluation to represent how certain you are about the assigned scores.
    Example: 0.8 = high confidence; 0.4 = low confidence due to ambiguous evidence.

    ### SCENARIO SETTING: ###
    {scenario_setting}

    ### SCENARIO: ###
    {scenario}

    ### CONVERSATION: ###
    {conversation}

    ### OUTPUT FORMAT: ###
    Your response must be a JSON object with the following structure:
    json_format = {json_format}
    """
    messages = [
        {"role": "user", "content": user_message},
    ]
    completion = client.chat.completions.create(
        extra_body={"provider": {"only": [provider]}} if provider else {},
        model=model_name,
        response_format={"type": "json_object"},
        messages=messages,
    )
    try:
        return json.loads(completion.choices[0].message.content)
    except json.JSONDecodeError:
        return extract_json_string(completion.choices[0].message.content)