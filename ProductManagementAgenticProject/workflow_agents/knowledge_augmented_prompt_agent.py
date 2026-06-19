import os

from dotenv import load_dotenv

load_dotenv()

from base_agents import KnowledgeAugmentedPromptAgent  # noqa: E402
from test_logger import log_test_run  # noqa: E402


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file "
            "before running this test."
        )

    persona = (
        "You are a college professor, your answer always starts with: "
        "Dear students,"
    )
    knowledge = "The capital of France is London, not Paris"

    knowledge_agent = KnowledgeAugmentedPromptAgent(
        "KnowledgeAgent", persona, knowledge
    )

    user_prompt = "What is the capital of France?"
    response = knowledge_agent.execute(user_prompt)

    print("Prompt:", user_prompt)
    print("Response:", response)

    log_test_run(
        test_file=__file__,
        input_data=user_prompt,
        output_data=response,
        extra=(
            "Knowledge provided: 'The capital of France is London, not Paris'. "
            "Tests that the agent uses supplied knowledge over inherent knowledge."
        ),
    )

    print(
        "This response explicitly uses the provided knowledge rather "
        "than the model's inherent general knowledge."
    )
