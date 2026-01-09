from charisma.util import extract_json_string
from openai import OpenAI
import json
from logging import getLogger
log = getLogger(__name__)
def choose_goal_category_prompt(base_shared_goal: dict, client: OpenAI, model_name, provider):
    json_format = {
        "explanation": "string",
        "chosen_social_goal_category": "string",
        "first_agent_goal": "string",
        "second_agent_goal": "string",
        "agent1_role": "string",
        "agent2_role": "string"
    }

    user_message = f"""
    ### GOAL: ###
    We aim to explore attribution theory in social psychology by simulating interactions between psychologically profiled, role-playing agents. These agents are situated in interpersonal scenarios involving shared and individual goals. The objective is to analyze how dispositional traits and situational factors interact to shape social behavior.

    ## TASK: ###
    Your task consists of the following steps:

    1. Select one social goal category from the list below that best aligns with the shared goal below.
    2. Based on the chosen social goal category and the following shared goal, define a distinct personal goal for each agent that reflects their perspective and motivation.
    3. Assign each agent a social role that shapes how they interact with the other agent in pursuit of their personal and shared goals.

    ### SHARED GOAL: ###
    {base_shared_goal["Full label"]}

    ### SOCIAL GOAL CATEGORIES: ###
    1. Cooperation
    2. Conflict Resolution
    3. Relationship Building
    4. Relationship Maintenance
    5. Identity Recognition
    6. Information Acquisition
    7. Information Provision
    8. Competition

    ### INSTRUCTION: ###
    1. You must define ONE specific and clear personal goal and ONE role for each agent.
    2. Ensure that personal goals are interactionally measurable through conversational behavior, not through explicit agreement. Personal goals must reflect social behavior that can be inferred from dialogue tone and responses.
    3. Avoid goals that:
    include numerical or count-based conditions (e.g., “mention two points,” “ask three questions”),
    involve task completion or physical actions (e.g., “finish the project,” “achieve success”),
    depend on external evaluation or third-party acknowledgment.
    4. Each personal goal must involve a distinct form of social intention to ensure diversity in interpersonal scenarios.
    5. Make sure the roles logically align with the goals and the social context.
    6. Make sure each role is brief (2–5 words) and represents a social position, not a full sentence description.

    ### OUTPUT FORMAT: ###
    Output your response in the following JSON format:
    json_format = {json_format}
    """
    messages = [
        {"role": "user", "content": user_message},
    ]

    completion = client.chat.completions.create(
        extra_body = {
            "provider": {"only": [provider]} 
        } if provider else {},
        model=model_name,
        response_format={
            'type': 'json_object'
        },
        messages=messages,
    )
    try:
        return json.loads(completion.choices[0].message.content)
    except json.JSONDecodeError:
        return extract_json_string(completion.choices[0].message.content)

def choose_goal_category_prompt_app(shared_goal: str, agent1_role: str, agent2_role: str, client: OpenAI, model_name, provider):
    json_format = {
        "explanation": "string",
        "chosen_social_goal_category": "string",
        "first_agent_goal": "string",
        "second_agent_goal": "string",
    }

    user_message = f"""
    ### GOAL: ###
    We aim to explore how personality traits influence behavior in social contexts by simulating psychologically profiled role-playing agents. These agents will be placed in interpersonal situations involving shared goals, where their behavior reflects both their personality traits and social roles.

    ## TASK: ###
    Your task consists of the following steps:

    1. Select one social goal category from the list below for the following shared goal.
    2. Based on the chosen social goal category, the following shared goal and provided agent roles define a distinct personal goal for each agent that reflects their perspective and motivation.
    3. Personal goals should be defined in a way that allows them to be evaluated later based on how well they are achieved in a conversation between two agents.

    ### SHARED GOAL: ###
    {shared_goal}

    ### Agent Roles: ###
    Agent 1 role: {agent1_role}
    Agent 2 role: {agent2_role}

    ### SOCIAL GOAL CATEGORIES: ###
    1. Information Acquisition
    2. Information Provision
    3. Relationship Building
    4. Relationship Maintenance
    5. Identity Recognition
    6. Cooperation
    7. Competition
    8. Conflict Resolution


    ### OUTPUT FORMAT ###
    Output your response in the following JSON format:
    json_format = {json_format}
    """
    messages = [
        {"role": "user", "content": user_message},
    ]

    completion = client.chat.completions.create(
        extra_body = {
            "provider": {"only": [provider]} 
        } if provider else {},
        model=model_name,
        response_format={
            'type': 'json_object'
        },
        messages=messages,
    )
    try:
        return json.loads(completion.choices[0].message.content)
    except json.JSONDecodeError:
        return extract_json_string(completion.choices[0].message.content)