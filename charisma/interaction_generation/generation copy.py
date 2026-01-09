import os
import csv
from dotenv import load_dotenv
import logging
import datetime
import pytz
from itertools import combinations
import pandas as pd
from charisma.util import extract_json_string
import re
import json
import ast
import random
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.runnables.config import RunnableConfig
import time
from charisma.interaction_generation.templates import system_templ, human_templ
from charisma.config import config


# Current working directory
current_dir = os.getcwd()
# Get absolute path
LOG_PATH = os.path.join(current_dir, "logs")
TIMESTAMP = str(datetime.datetime.now(pytz.timezone("Europe/Berlin"))).replace(" ", "_").replace(".", ":")


BFI_SCHEMA = {
  "name": "bfi_output",
  "strict": True,
  "schema": {
    "type": "object",
    "additionalProperties": False,
    "required": [
      "bfi_personality_reflection",
      "bfi_scores",
      "response",
      "behavioral_code",
      "explanation"
    ],
    "properties": {
      "bfi_personality_reflection": {"type": "string"},
      "bfi_scores": {
        "type": "object",
        "additionalProperties": False,
        "required": [
          "openness",
          "conscientiousness",
          "extraversion",
          "agreeableness",
          "neuroticism"
        ],
        "properties": {
          "openness": {"type": "number"},
          "conscientiousness": {"type": "number"},
          "extraversion": {"type": "number"},
          "agreeableness": {"type": "number"},
          "neuroticism": {"type": "number"}
        }
      },
      "response": {"type": "string"},
      "behavioral_code": {"type": "string"},
      "explanation": {"type": "string"}
    }
  }
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # logging.FileHandler(f"{LOG_PATH}/{TIMESTAMP}.log"), # TODO: this causes issues in some environments
        logging.StreamHandler()
    ]
)
load_dotenv()
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
STORE = {}

def get_session_history(agent_name: str, session_id: str) -> BaseChatMessageHistory:
    if (agent_name, session_id) not in STORE:
        STORE[(agent_name, session_id)] = AgentHistory(session_id=session_id, agent_name=agent_name)
    return STORE[(agent_name, session_id)]

class AgentHistory(BaseChatMessageHistory, BaseModel):
    session_id: str
    agent_name: str
    messages: List[BaseMessage] = Field(default_factory=list)

    def __init__(self, session_id: str, agent_name: str, **data):
        super().__init__(session_id=session_id, agent_name=agent_name, **data)
        self.agent_name = agent_name

    def add_messages(self, new_messages: List[BaseMessage]) -> None:
        for m in new_messages:
            if isinstance(m, AIMessage):
                print(f'\n####### {parse_json_response(m.content)}\n##### {m.content}\n')
                response = parse_json_response(m.content)["response"]
                behavioral_code = parse_json_response(m.content).get("behavioral_code", "")
                new_content = {
                    "agent": self.agent_name,
                    "response": response,
                    "behavioral_code": behavioral_code,
                }
                message = AIMessage(content=f'{json.dumps(new_content)}')
                logging.info(f"****************ADDED MESSAGE 1 {m}************************")
                logging.info(f"****************ADDED MESSAGE 2 {message}************************")
                self.messages.append(message)

    def add_reply(self, new_message: BaseMessage) -> None:
        logging.info(f"****************ADDED REPLY {new_message}************************")
        self.messages.append(new_message)


    def clear(self) -> None:
        self.messages = []

def parse_json_response(json_text: str):
    try:
        # Extract the JSON object (fallback if any extra text)
        match = re.search(r'\{.*\}', json_text, re.DOTALL)
        if not match:
            return {"response": json_text}
        json_block = match.group()

        # Parse the JSON block as is
        try:
            data = json.loads(json_block)
            logging.info(f"RESPONSE JSON: {data}")
            return data
        except json.JSONDecodeError as e:
            logging.info(f"JSONDecodeError — {e}. Attempting to re-escape inner content…")
        
        # Fallback: extract everything inside the response string
        inner_match = re.search(
            r'"response"\s*:\s*"(.*)"\s*}$',
            json_text,
            re.DOTALL
        )

        if inner_match:
            raw_content = inner_match.group(1)
            
            # Rebuild a safe JSON block by using json.dumps to escape raw_content
            safe_block = '{ "response": ' + json.dumps(raw_content) + ' }'
            logging.info(f"SAFE JSON BLOCK:{safe_block}")
            try:
                return json.loads(safe_block)
            except json.JSONDecodeError as e2:
                logging.info(f"Still invalid after re-escaping — {e2}. Falling back to raw text.")
        

        # If all else fails, return the raw text
        logging.info(f"RESPONSE JSON TEXT: {json_text}")
        return {"response": json_text}
    
    except Exception as e:
        logging.info(f"ERROR - {e}")
        raise e


# TODO: need to test it before replacing with parse_json_response
def get_json_response(json_text: str):
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        return extract_json_string(json_text)

def get_character_list(character_file):
    try:
        character_list = []
        with open(character_file, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            
            # Expect columns: character, subcategory
            for row in reader:
                name = row.get("mbti_profile")
                movie = row.get("subcategory")
                if name and movie:
                    character_list.append({"character": name, "movie": movie})

        return character_list
    except FileNotFoundError:
        logging.info(f"File '{character_file}' not found.")
        return []
    except csv.Error as e:
        logging.info(f"Error reading CSV file '{character_file}': {e}")
        return []
    except Exception as e:
        logging.info(f"An error occurred: {e}")
        return []

def format_behavioral_codes(df, act_type):
    """
    Filter behavioral codes by type of act and return formatted string.
    """
    subset = df[(df["Type of Act"] == act_type) | (df["Type of Act"] == "Universal")]
    formatted = [
        f'- name: "{row["Behaviour Code"]}", description: "{row["Definition"]}."'
        for _, row in subset.iterrows()
    ]
    # Add default "None" option
    formatted.append('- name: "None", description: "No code from the list applies to this response."')
    return "\n".join(formatted)


class RolePlayEngine:
    def __init__(self, charaction_filename, scenarios_filename, behavioral_coding_filename, output_csv, model, max_turns=20, num_samples=None):
        if charaction_filename:
            self.characters = get_character_list(charaction_filename)
        # if no scenario sample limit, use all
        if not num_samples and scenarios_filename:
            self.scenarios = pd.read_csv(scenarios_filename)
        elif scenarios_filename:  
            self.scenarios = pd.read_csv(scenarios_filename)[:num_samples]

        self.behavioral_coding = pd.read_csv(behavioral_coding_filename)
        self.max_turns = max_turns
        self.output_csv = output_csv

        provider = None
        if len(model.get("providers", [])) > 0:
            provider = {
                "only": model["providers"],
            }

        self.model = ChatOpenAI(
            model_name= model["name"],
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=OPEN_ROUTER_API_KEY,
            temperature=0.5,
            # max_tokens=1000,
            streaming=False,
            verbose=False,
            extra_body={"provider": provider, "structured_outputs": True, } if provider else {},
            # model_kwargs={
            #     "response_format": {"type": "json_object"},
            # },
            model_kwargs={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": BFI_SCHEMA,
                },
                # Optional but often helpful with OpenRouter routing:
                # "require_parameters": True,
            },

        )

        # base prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_templ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template(human_templ),
        ])

    def role_play(self, agent_a, agent_b, subcategory_a, subcategory_b, personality_a, personality_b, history_agenta_key, history_agentb_key, session_id, row, streaming=False):
        shared_goal = row["shared_goal"]
        social_goal_category = row["social_goal_category"]
        first_agent_goal = row["first_agent_goal"]
        second_agent_goal = row["second_agent_goal"]
        agent1_role = row["agent1_role"]
        agent2_role = row["agent2_role"]
        scenario = ast.literal_eval(row["scenario"])['scenario_context']

        behavioral_code_str = format_behavioral_codes(self.behavioral_coding, social_goal_category)

        # Partially bind all static variables on the prompt
        prompt1 = self.prompt.partial(
            agent_number=1,
            agent_name=agent_a,
            sub_category=subcategory_a,
            personality=personality_a,
            social_role=agent1_role,
            scenario=scenario,
            shared_goal=shared_goal,
            agent_goal=first_agent_goal,
            other_agent_name=agent_b,
            other_agent_number=2,
            behavioral_code_str=behavioral_code_str,
        )
        prompt2 = self.prompt.partial(
            agent_number=2,
            agent_name=agent_b,
            sub_category=subcategory_b,
            personality=personality_b,
            social_role=agent2_role,
            scenario=scenario,
            shared_goal=shared_goal,
            agent_goal=second_agent_goal,
            other_agent_name=agent_a,
            other_agent_number=1,
            behavioral_code_str=behavioral_code_str,
        )

        # Compose pipelines via pipe (Runnable) and wrap in memory
        pipeline1 = prompt1 | self.model
        pipeline2 = prompt2 | self.model

        # Create RunnableWithMessageHistory for each agent
        chain1 = RunnableWithMessageHistory(
            runnable=pipeline1,
            get_session_history=get_session_history,
            input_messages_key="user_message",
            history_messages_key="history",
            history_factory_config=[
                ConfigurableFieldSpec(
                    id="agent_name",
                    annotation=str,
                    name="Agent Name",
                    description="Name of the agent for which the history is being retrieved.",
                    default="",
                    is_shared=True,
                ),
                ConfigurableFieldSpec(
                    id="session_id",
                    annotation=str,
                    name="Session ID",
                    description="Session ID for the conversation history.",
                    default="",
                    is_shared=True,
                ),
            ],
        )
        chain2 = RunnableWithMessageHistory(
            runnable=pipeline2,
            get_session_history=get_session_history,
            input_messages_key="user_message",
            history_messages_key="history",
            history_factory_config=[
                ConfigurableFieldSpec(
                    id="agent_name",
                    annotation=str,
                    name="Agent Name",
                    description="Name of the agent for which the history is being retrieved.",
                    default="",
                    is_shared=True,
                ),
                ConfigurableFieldSpec(
                    id="session_id",
                    annotation=str,
                    name="Session ID",
                    description="Session ID for the conversation history.",
                    default="",
                    is_shared=True,
                ),
            ],
        )

        user_message = None
        dialogues = []
        
        for turn in range(self.max_turns):
            control_str = ""
            if turn >= 14 and turn <= 15:
                control_str = "You are nearly at the end of the conversation."
            elif turn >= 16 and turn <= 17:
                control_str = "Only a couple of turns remain in the conversation"
            if turn >= 18:
                control_str = "This is your last response in the conversation."
            turn_str = str(turn)

            # pick the chain & agent
            is_agent_1 = (turn % 2 == 0)
            chain = chain1 if is_agent_1 else chain2
            agent = history_agenta_key if is_agent_1 else history_agentb_key
            other_agent = history_agentb_key if is_agent_1 else history_agenta_key
            speaker_name = agent_a if is_agent_1 else agent_b
            prompt_i = prompt1 if is_agent_1 else prompt2

            history_msgs = get_session_history(agent, session_id).messages
            rendered = prompt_i.format_messages(
                history=history_msgs,
                turn=turn_str,
                control_str=control_str,
            )
            logging.info("=== Messages sent to LLM ===")
            logging.info(f"{rendered}")

            ai_msg = chain.invoke(
                {"turn": turn_str, "control_str": control_str, "user_message": turn_str},
                config={"configurable": {"session_id": session_id, "agent_name": agent}},
            )

            # extract the raw text
            raw = ai_msg.content
            response = parse_json_response(raw)
            reply = response["response"]

            # add the current agent reply to the other agent's history
            user_message = {
                "agent": agent,
                "response": reply,
                "behavioral_code": response.get("behavioral_code", ""),
            }
            user_message_str = f'{json.dumps(user_message)}' if user_message else ''
            get_session_history(other_agent, session_id).add_reply(HumanMessage(content=user_message_str))

            logging.info(f"Turn {turn_str} -> {agent} : {reply}\n")
            dialogues.append({
                "agent": agent,
                "response": reply,
                "behavioral_code": response.get("behavioral_code", ""),
                "explanation": response.get("explanation", ""),
                "bfi_personality_reflection": response.get("bfi_personality_reflection", ""),
                "bfi_scores": response.get("bfi_scores", ""),
            })
            if streaming:
                user_message["behavioral_code"] = response.get("behavioral_code", "")
                user_message["explanation"] = response.get("explanation", "")
                yield user_message
        
        if not streaming:
            yield dialogues

    def run(self) -> None:
        # prepare CSV writer
        output_csv_file = open(self.output_csv, "w", newline="", encoding="utf-8")
        fieldnames = [
            "agents",
            "base_shared_goal",
            "social_goal_category",
            "explanation",
            "shared_goal",
            "first_agent_goal",
            "second_agent_goal",
            "agent1_role",
            "agent2_role",
            "scenario",
            "interaction_history",
        ]
        writer = csv.DictWriter(output_csv_file, fieldnames=fieldnames)
        writer.writeheader()

        scenarios_data = self.scenarios

        character_list = [c["character"] for c in self.characters]

        for idx, row in scenarios_data.iterrows():
            scenario=ast.literal_eval(row["scenario"])['scenario_context']
            logging.info(f"\n_______Scenario {idx}: {scenario}_______\n")

            all_pairs = list(combinations(character_list, 2))
            random_pairs = random.sample(all_pairs, 5)
            logging.info(f"PAIRS {random_pairs}_______\n")

            # track which unordered pairs have been executed
            for character_1, character_2 in random_pairs:

                for swap in (False, True): 
                    STORE.clear() # clear the global store for each new turn
                    agent_a = character_1 if not swap else character_2
                    agent_b = character_2 if not swap else character_1
                    history_agenta_key = agent_a
                    history_agentb_key = agent_b
                    agents_key = f"{history_agenta_key}-{history_agentb_key}"
                    logging.info(f"_______Role-playing: {agents_key}_______\n\n")
                    session_id = f"scenario{idx}_pair{agents_key}"

                    dialogues_gen = self.role_play(agent_a, agent_b, history_agenta_key, history_agentb_key, session_id, row)
                    for m in dialogues_gen:
                        dialogues = m
                    
                    self._print_history(session_id, history_agenta_key, history_agentb_key)
                    agents_list = [history_agenta_key, history_agentb_key]

                    # after all turns, save a record
                    writer.writerow({
                        "agents":               f"{agents_list}",
                        "base_shared_goal":     row["base_shared_goal"],
                        "social_goal_category": row["social_goal_category"],
                        "explanation":          row["explanation"],
                        "shared_goal":          row["shared_goal"],
                        "first_agent_goal":     row["first_agent_goal"],
                        "second_agent_goal":    row["second_agent_goal"],
                        "agent1_role":          row["agent1_role"],
                        "agent2_role":          row["agent2_role"],
                        "scenario":             row["scenario"],
                        "interaction_history":  json.dumps(dialogues),
                    })
                    output_csv_file.flush()  # ensure it’s on disk
                
                # logging.info("sleep for 30s\n")
                # sleep to avoid rate limits on OpenRouter
                time.sleep(30)


        # close the CSV file
        output_csv_file.close()
        return "Done! Interaction generation completed and saved to CSV."

    def _print_history(self, session_id: str, history_agenta_key: str, history_agentb_key: str) -> None:
        logging.info("---- Raw STORE ---")
        for i, turn in enumerate(STORE[(history_agenta_key, session_id)].messages):
            logging.info(f"{i:>2}: {turn.content}")

        for char in (history_agenta_key, history_agentb_key):
            logging.info(f"---- {char}'s view ----")
            hist = get_session_history(char, session_id).messages
            for i, msg in enumerate(hist):
                role = "HUMAN" if isinstance(msg, HumanMessage) else "AI"
                logging.info(f"{i:>2}: {role} ▶ {msg.content}") 


def interaction_generation(charaction_filename: str, scenarios_filename: str, behavioral_coding_filename:str, output_csv: str, model: str, provider = None, max_turns: int = 20,num_samples=None) -> None:
    model_obj = {
        # "name": "deepseek/deepseek-chat-v3-0324",
        # "name": "deepseek/deepseek-r1-0528:free",
        "name": model,
        "providers": [provider] if provider else [],
    }
    engine = RolePlayEngine(charaction_filename, scenarios_filename, behavioral_coding_filename, output_csv, model_obj, max_turns, num_samples)
    return engine.run()

# Demo app function for interaction generation
def interaction_generation_app(model, provider, agent1_name, agent2_name, scenario_data, scenario, max_turns=10):
    model_obj = {
        "name": model,
        "providers": [provider] if provider else [],
    }

    history_agenta_key = agent1_name
    history_agentb_key = agent2_name

    agents_key = f"{history_agenta_key}-{history_agentb_key}"

    session_id = f"scenario_pair{agents_key}"

    STORE.clear()

    engine = RolePlayEngine(None, None, config.pipeline.behavioral_coding_csv, '', model_obj, max_turns=max_turns)
    data = {
        "shared_goal": scenario_data.get("shared_goal", ""),
        "social_goal_category": scenario_data.get("social_goal_category", ""),
        "first_agent_goal": scenario_data.get("first_agent_goal", ""),
        "second_agent_goal": scenario_data.get("second_agent_goal", ""),
        "agent1_role": scenario_data.get("agent1_role", ""),
        "agent2_role": scenario_data.get("agent2_role", ""),
        "scenario": json.dumps(scenario),
    }
    # returns a generator for streaming
    return engine.role_play(agent1_name, agent2_name, history_agenta_key, history_agentb_key, session_id, data, True)


def run_interaction_pipeline(behavioral_coding_filename: str, agent1_name: str, agent2_name: str,subcategory_a:str,subcategory_b:str, personality_a,personality_b, scenario_data: dict, model: str, provider = None, max_turns: int = 20,num_samples=None):
    model_obj = {
        "name": model,
        "providers": [provider] if provider else [],
    }
    history_agenta_key = agent1_name
    history_agentb_key = agent2_name

    agents_key = f"{history_agenta_key}-{history_agentb_key}"

    session_id = f"scenario_pair{agents_key}"

    STORE.clear()
    engine = RolePlayEngine(None, None, behavioral_coding_filename, None, model_obj, max_turns, num_samples)
    dialogues_gen = engine.role_play(agent1_name, agent2_name, subcategory_a, subcategory_b, personality_a, personality_b, history_agenta_key, history_agentb_key, session_id, scenario_data)
    for m in dialogues_gen:
        dialogues = m
    return dialogues

if __name__ == "__main__":
    charaction_filename = config.pipeline.interaction_generation.characters_filepath
    scenarios_filename = config.pipeline.interaction_generation.scenarios_filepath
    behavioral_coding_filename = config.pipeline.behavioral_coding_csv
    output_csv = "interaction_generation_output.csv"
    model = config.pipeline.model
    provider = config.pipeline.provider
    interaction_generation(charaction_filename, scenarios_filename, behavioral_coding_filename, output_csv, model,  provider, 2)