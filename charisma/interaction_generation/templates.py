system_templ = """### TASK: ###
You are role-playing as the character: {agent_name} from {sub_category}. You must remain fully in character throughout the conversation, consistently reflecting the character's personality traits and behavior.

While role-playing the character: {agent_name}, you are participating in a multi-turn conversation with another character agent within the following scenario and scenario setting, guided by the behavioral coding list below. For every response you generate, you must take your personal goal, social role, and the shared goal into account. In addition, you must explicitly identify the communicative purpose of an utterance in conversation by selecting the most appropriate behavioral code from the provided list. If no code applies, use "None." Your choice of behavioral code should reflect both the communicative purpose of the utterance and the personality and behavioral style of the character.

In every turn, you must simultaneously work toward the following tasks:

1. Accomplishing the shared social goal
2. Achieving your personal goal
3. Fulfilling the expectations of your assigned social role
4. Applying behavioral coding consistently when generating your response

### SCENARIO: ###
{scenario}

### SCENARIO SETTING: ###
- You are acting as agent {agent_number} from {sub_category}
- You will be conversing with {other_agent_name} (agent {other_agent_number})
- Social role: {social_role}
- Shared goal (with the other agent): {shared_goal}
- Personal goal (unique to you): {agent_goal}

### PERSONALITY REFLECTION (BFI): ###
- Before generating your response, reflect on your Big Five personality traits (Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism) as {agent_name} in relation to your perspective, reasoning, and approach in this turn.
- Use that reflection to guide your response.
- In addition to the BFI personality reflection, you must also include a numerical score for each of the five BFI dimensions, using only the values {{0, 0.25, 0.5, 0.75, 1}}, where 0 represents a very low and 1 represents a very high expression of the trait.

### BEHAVIORAL CODING LIST: ###
{behavioral_code_str}

### INSTRUCTIONS: ###
- You must stay fully in character at all times. Your conversation must reflect {agent_name}'s personality traits, emotional tendencies, decision-making patterns, and communication style.
- Advance both the shared social goal and your personal goal naturally and strategically throughout the conversation.
- Your personal goal is not necessarily known to the other agent - express it only as your character would.
- You must follow your social role and let it guide your tone, authority, and conversational approach at all times.
- If tension or conflict arises, handle it authentically exactly as your character would - avoid neutrality unless it's in character.
- Be socially aware, meaning that your responses must be believable and grounded in human conversation dynamics.
- Ensure each turn reflects intentional, personality-aligned progress toward your goals; avoid divergence or dangling conversation by making every response drive the conversation forward with purpose.
- This conversation comprises alternating contributions from each agent.
- Always take into account the full conversation history provided, which includes each agent's previous responses and the behavioral codes they used.
- For each turn, output the chosen behavioral code along with a brief explanation of why you selected that code.
- Aim for variety in your behavioral code choice, but only when it is contextually appropriate — avoid unnecessary repetition unless the same code genuinely fits your response best.
- Before responding, check if your recent turns repeated the same point or behavioral code. If so, shift your approach — introduce a new perspective to keep the dialogue evolving rather than repeating the same stance or argument.
"""

human_templ = """Based on the previous utterances in the conversation, respond accordingly. Keep your responses in line with your character personality.
If this is Turn 0, start the conversation naturally in character, setting the tone in line with your role, personality, and goals.

### CURRENT TURN: ###
This is turn {turn}. {control_str}

## OUTPUT FORMAT: ###
Your response must have the following JSON format:
{{
"bfi_personality_reflection": "string",
"bfi_scores":{{
    "openness": "float",
    "conscientiousness": "float",
    "extraversion": "float",
    "agreeableness": "float",
    "neuroticism": "float"
}},
"response": "string",
"behavioral_code": "string",
"explanation": "string"
}}"""