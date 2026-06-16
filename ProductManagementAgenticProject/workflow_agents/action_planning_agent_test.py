import unittest
from unittest.mock import patch

from base_agents import ActionPlanningAgent


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
        with patch("base_agents.OpenAI", return_value=FakeClient()):
            agent = ActionPlanningAgent(
                knowledge="Simple cooking guidance.",
                api_key="fake-key",
            )
            steps = agent.respond(
                "One morning I wanted to have scrambled eggs"
            )

        self.assertEqual(
            steps,
            ["Crack the eggs", "Heat the pan", "Cook and serve"],
        )


if __name__ == "__main__":
    unittest.main()
