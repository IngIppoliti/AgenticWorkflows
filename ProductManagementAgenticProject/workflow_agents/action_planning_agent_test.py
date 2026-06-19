import unittest
from unittest.mock import patch

from base_agents import ActionPlanningAgent
from test_logger import log_test_run


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeCompletions:
    def create(self, **kwargs):
        return type(
            "Response",
            (),
            {
                "choices": [
                    FakeChoice(
                        "Here are the steps:\n"
                        "- Crack the eggs\n"
                        "- Heat the pan\n"
                        "- Cook and serve"
                    )
                ]
            },
        )()


class FakeClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": FakeCompletions()})()


class ActionPlanningAgentTest(unittest.TestCase):
    def test_respond_returns_clean_action_steps(self):
        prompt = "One morning I wanted to have scrambled eggs"

        with patch("base_agents.client", None), patch("base_agents.OpenAI", return_value=FakeClient()):
            agent = ActionPlanningAgent(
                name="TestActionPlanner",
                knowledge="Simple cooking guidance."
            )
            steps = agent.execute(prompt)

        expected = ["Crack the eggs", "Heat the pan", "Cook and serve"]

        log_test_run(
            test_file=__file__,
            input_data=prompt,
            output_data=str(steps),
            extra=f"expected={expected} | passed={steps == expected}",
        )

        self.assertEqual(steps, expected)


if __name__ == "__main__":
    unittest.main()
