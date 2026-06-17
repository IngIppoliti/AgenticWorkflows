from dotenv import load_dotenv

load_dotenv()
from base_agents import DirectPromptAgent  # noqa: E402
from test_logger import log_test_run  # noqa: E402


if __name__ == "__main__":
    direct_agent = DirectPromptAgent("FranceAgent")

    user_prompt = "What is the Capital of France?"
    response = direct_agent.execute(user_prompt)

    print("Prompt:", user_prompt)
    print("Response:", response)

    log_test_run(
        test_file=__file__,
        input_data=user_prompt,
        output_data=response,
        extra="Knowledge source: general LLM knowledge.",
    )

    print(
        "Knowledge source: this answer comes from the general knowledge "
        "of the selected LLM model used by the agent."
    )
