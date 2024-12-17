import os
import openai
import random

# Set your OpenAI API Key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# A simple representation of prompt parameters or templates.
# In a real scenario, these might be embeddings or more complex structures.
prompt_templates = [
    "You are a helpful assistant. Answer the question succinctly:\n{USER_QUERY}",
    "You are a knowledgeable expert. Explain in detail:\n{USER_QUERY}",
    "Please provide a simple, friendly explanation:\n{USER_QUERY}"
]

# A dictionary to store performance scores (a stand-in for a Bayesian model).
# In practice, you'd store distributions and update them probabilistically.
template_scores = {i: 0.0 for i in range(len(prompt_templates))}

def choose_prompt_template():
    # In a real Bayesian approach, you'd sample from a posterior distribution.
    # Here, we just pick a template weighted by scores or at random.
    # Let's do a simple heuristic: add a bit of noise so it can explore.
    scored_templates = [(i, template_scores[i] + random.gauss(0, 1)) for i in range(len(prompt_templates))]
    scored_templates.sort(key=lambda x: x[1], reverse=True)
    # Top template after adding noise
    chosen = scored_templates[0][0]
    return chosen

def ask_chatgpt(prompt, role="user", system_message=None):
    """
    Sends a message to ChatGPT using the provided prompt.
    `system_message` can be used to set a system-level directive.
    """
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": role, "content": prompt})

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7
    )
    return response.choices[0].message["content"].strip()

def simulate_user_feedback(response):
    """
    Simulate a user rating.
    In a real scenario, user feedback could be collected interactively.
    """
    # A very naive simulation: if "simple" in response, user is happier.
    rating = 3
    if "simple" in response.lower():
        rating = 5
    elif "confusing" in response.lower():
        rating = 1
    return rating

def interpret_user_feedback(user_message):
    """
    Ask ChatGPT to interpret user's sentiment.
    In a real scenario, the user's actual feedback or rating would be used.
    """
    interpretation_prompt = f"User said: '{user_message}'. Rate their satisfaction (1 to 5) and explain."
    interpretation = ask_chatgpt(interpretation_prompt, role="user")
    # For simplicity, we just look for a number in the interpretation:
    # In a real system, parse the content more robustly.
    # Here we randomly pick a rating if we can't parse one.
    import re
    match = re.search(r"(\d)/5", interpretation)
    if match:
        return int(match.group(1))
    else:
        return random.randint(1,5)

def clarify_if_uncertain():
    """
    If uncertain about user satisfaction, ask either the user or the LLM for clarity.
    """
    clarification_prompt = "I want to ensure I understand your feedback. Could you clarify what you found helpful or unhelpful?"
    clarification_response = ask_chatgpt(clarification_prompt, role="user")
    # This is a stand-in: a real scenario would show this to the user or somehow integrate their answer.

    # Interpret the clarification
    clarification_rating = interpret_user_feedback(clarification_response)
    return clarification_rating

def update_scores(template_index, rating):
    """
    Update the template score based on the feedback rating.
    """
    # A simple update rule: add (rating - 3) to the score to shift it up/down.
    # In a Bayesian approach, you'd update a distribution, not just a score.
    template_scores[template_index] += (rating - 3)

def generate_training_prompt():
    """
    In training mode, we ask ChatGPT to produce a pseudo-user prompt.
    This simulates having a user prompt without real user input.
    """
    # Random topic request
    topic = random.choice(["quantum entanglement", "the French Revolution", "Python programming for loops", "photosynthesis in plants"])
    system_message = "You are a helpful prompt generator. Please produce a user query as if it came from a curious user."
    training_query = f"Generate a user-like query about {topic}, something that a beginner might ask."

    pseudo_user_prompt = ask_chatgpt(training_query, role="user", system_message=system_message)
    return pseudo_user_prompt

def main_interaction_cycle(real_user_input=None):
    """
    One cycle of:
    - If no real user input, generate a pseudo-user prompt (training scenario)
    - Choose a prompt template and fill in the query
    - Ask ChatGPT (the main LLM) for the response
    - Get user feedback (simulated or interpreted)
    - Possibly clarify if uncertain
    - Update template scores
    """

    # Step 1: Obtain user query
    if real_user_input is None:
        user_query = generate_training_prompt()
        print(f"Training scenario. Generated pseudo-user query: {user_query}")
    else:
        user_query = real_user_input

    # Step 2: Choose a prompt template
    chosen_index = choose_prompt_template()
    chosen_template = prompt_templates[chosen_index]
    prompt_for_llm = chosen_template.replace("{USER_QUERY}", user_query)

    # Step 3: Get LLM response
    llm_response = ask_chatgpt(prompt_for_llm)
    print(f"\nLLM Response:\n{llm_response}\n")

    # Step 4: Get feedback
    # Here we simulate user feedback. In a real scenario, you'd get actual user input or rating.
    # Alternatively, if you want to interpret from LLM or user message:
    # rating = interpret_user_feedback("The user feedback here") # if user was there
    rating = simulate_user_feedback(llm_response)
    print(f"Simulated Rating: {rating}")

    # If rating is uncertain (e.g., rating is neutral), ask for clarification
    if rating == 3:
        # Uncertain scenario
        clarification_rating = clarify_if_uncertain()
        print(f"Clarification Rating: {clarification_rating}")
        rating = clarification_rating

    # Step 5: Update parameters
    update_scores(chosen_index, rating)
    print(f"Updated template scores: {template_scores}")

    # Next cycle, hopefully the chosen template evolves to produce better responses.

if __name__ == "__main__":
    # Run a few cycles in training mode (no real user input)
    for _ in range(3):
        main_interaction_cycle()

    # In a real application, you'd replace main_interaction_cycle(None) with main_interaction_cycle(user_input)
    # once you have a real user prompt from outside.
