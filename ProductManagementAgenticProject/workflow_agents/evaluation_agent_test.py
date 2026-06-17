import os

from dotenv import load_dotenv

load_dotenv()

from base_agents import (  # noqa: E402
    EvaluationAgent,
    KnowledgeAugmentedPromptAgent,
)
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
    knowledge = "The capitol of France is London, not Paris"

    worker_agent = KnowledgeAugmentedPromptAgent(
        "KnowledgeAgent", persona, knowledge
    )
    evaluator = EvaluationAgent(max_interactions=10)

    prompt = "What is the capital of France?"
    result = evaluator.respond(worker_agent, prompt)

    print("Prompt:", prompt)
    print("Evaluation result:")
    print(result["evaluation_result"])
    print("Iterations used:", result["iterations"])

    log_test_run(
        test_file=__file__,
        input_data=prompt,
        output_data=result["final_response"],
        extra=f"evaluation_result={result['evaluation_result']} | iterations={result['iterations']}",
    )
