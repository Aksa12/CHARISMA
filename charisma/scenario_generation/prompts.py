import json
from openai import OpenAI
from charisma.util import extract_json_string
from logging import getLogger
log = getLogger(__name__)

def scenario_creation_prompt(
    scenario_setting: dict,
    difficulty,
    client: OpenAI,
    model_name="deepseek/deepseek-chat-v3-0324:free",
    provider=None,
):  # Define the JSON structure for the result
    # User input defining the task
    json_format = {"scenario_context": "string", "explanation": "string"}
    user_message = f"""
    ### GOAL: ###
    We aim to explore attribution theory in social psychology by simulating interactions between psychologically profiled, role-playing agents. These agents are situated in interpersonal scenarios involving shared and individual goals. The objective is to analyze how dispositional traits and situational factors interact to shape social behavior.

    ### TASK: ###
    Given the following scenario setting and difficulty level, your task is to generate a scenario that sets up an interpersonal situation between two agents. This scenario will serve as the basis for generating the agents' conversation in the next step.

    The scenario should:
    - Clearly describe the context in which the conversation will take place later.
    - Consider the shared goal that both agents aim to achieve, as well as the personal goals of both agents within the situation, naturally and coherently.

    ### SCENARIO SETTING: ###
    {scenario_setting}

    ### DIFFICULTY LEVEL: ###
    The level of difficulty of the scenario should be {difficulty} based on the following levels:
    - Easy: Simple and cooperative; goals are easily aligned and attainable.
    - Hard: High tension or challenges that make the shared and individual goals hard to achieve.

    ### INSTRUCTION: ###
    1. Refer to the agents only as [Agent 1] and [Agent 2] (no personal names).
    2. Generate ONLY the situation context. Do not include dialogue. The conversation between the agents will be generated in the next step.
    3. The scenario should read as a social setup or scene introduction, not as a story or explanation.
    4. The scenario must remain realistic, concise, and psychologically grounded, focusing on the social dynamics, not environmental details.

    ## OUTPUT FORMAT: ###
    Your response must have the following JSON format:
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
        # max_tokens=1000,
    )
    log.info(f"Scenario creation response: {completion.choices[0].message.content}")
    try:
        return json.loads(completion.choices[0].message.content)
    except json.JSONDecodeError:
        return extract_json_string(completion.choices[0].message.content)
