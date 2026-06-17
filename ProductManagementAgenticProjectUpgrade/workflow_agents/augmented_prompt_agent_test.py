import os

from dotenv import load_dotenv

load_dotenv()

from base_agents import AugmentedPromptAgent  # noqa: E402
from test_logger import log_test_run  # noqa: E402


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file before running this test.")

    persona = "a helpful travel advisor"
    augmented_agent = AugmentedPromptAgent("TravelGuideAgent", persona)

    user_prompt = "Why is Paris a great city for first-time visitors?"
    augmented_agent_response = augmented_agent.execute(user_prompt)

    print("Prompt:", user_prompt)
    print("Response:", augmented_agent_response)

    log_test_run(
        test_file=__file__,
        input_data=user_prompt,
        output_data=augmented_agent_response,
        extra=(
            "Knowledge source: general LLM knowledge. "
            "Persona: a helpful travel advisor."
        ),
    )

    # The response is generated using the LLM's general knowledge base,
    # plus the role/persona instructions supplied to the agent.
    # The persona helps steer tone, style, and framing of the answer.
    print(
        "Knowledge source: this answer is based on the selected LLM model's "
        "general knowledge, with the persona influencing the presentation."
    )
    print(
        "Persona effect: specifying 'a helpful travel advisor' makes the output "
        "more oriented toward travel guidance and practical recommendations."
    )
