from dotenv import load_dotenv

load_dotenv()
from base_agents import DirectPromptAgent  # noqa: E402


if __name__ == "__main__":
    direct_agent = DirectPromptAgent("FranceAgent")

    user_prompt = "What is the Capital of France?"
    response = direct_agent.execute(user_prompt)

    print("Prompt:", user_prompt)
    print("Response:", response)
    print(
        "Knowledge source: this answer comes from the general knowledge "
        "of the selected LLM model used by the agent."
    )
