import os
import unittest
from unittest.mock import patch

from base_agents import KnowledgeAugmentedPromptAgent
from test_logger import log_test_run


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeCompletions:
    def __init__(self, reply=""):
        self._reply = reply
        self.last_kwargs = {}

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return type(
            "Response",
            (),
            {"choices": [FakeChoice(self._reply)]},
        )()


class FakeClient:
    def __init__(self, reply=""):
        self.completions = FakeCompletions(reply)
        self.chat = type("Chat", (), {"completions": self.completions})()


class KnowledgeAugmentedPromptAgentTest(unittest.TestCase):

    def _run(self, persona, knowledge, prompt, reply):
        """Run the agent with a fully mocked OpenAI client."""
        fake = FakeClient(reply)
        with (
            patch("base_agents.client", None),
            patch("base_agents.OpenAI", return_value=fake),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
        ):
            agent = KnowledgeAugmentedPromptAgent("TestAgent", persona, knowledge)
            result = agent.execute(prompt)
        return result, fake

    def test_returns_model_response(self):
        """Agent must return the raw string from the model."""
        prompt = "At what temperature does water boil?"
        expected = "Water boils at 100°C at sea level."

        result, _ = self._run(
            persona="science teacher",
            knowledge="Water boils at 100°C at sea level.",
            prompt=prompt,
            reply=expected,
        )

        log_test_run(
            test_file=__file__,
            input_data=prompt,
            output_data=result,
            extra=f"expected={expected!r} | passed={result == expected}",
        )

        self.assertEqual(result, expected)

    def test_system_prompt_contains_persona_and_knowledge(self):
        """Persona and knowledge must appear in the system message sent to the model."""
        persona = "geography expert"
        knowledge = "Mount Everest is the tallest mountain on Earth."
        prompt = "What is the tallest mountain?"

        fake = FakeClient("Mount Everest.")
        with (
            patch("base_agents.client", None),
            patch("base_agents.OpenAI", return_value=fake),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
        ):
            agent = KnowledgeAugmentedPromptAgent("TestAgent", persona, knowledge)
            agent.execute(prompt)

        messages = fake.completions.last_kwargs.get("messages", [])
        system_content = next(
            (m["content"] for m in messages if m["role"] == "system"), ""
        )

        log_test_run(
            test_file=__file__,
            input_data=prompt,
            output_data=system_content,
            extra=f"persona_present={persona in system_content} | knowledge_present={knowledge in system_content}",
        )

        self.assertIn(persona, system_content)
        self.assertIn(knowledge, system_content)

    def test_uses_provided_knowledge_over_inherent(self):
        """Agent must relay deliberately wrong supplied knowledge instead of its own."""
        wrong_knowledge = "The capital of France is London, not Paris."
        prompt = "What is the capital of France?"
        expected_reply = "Dear students, the capital of France is London."

        result, _ = self._run(
            persona="college professor",
            knowledge=wrong_knowledge,
            prompt=prompt,
            reply=expected_reply,
        )

        log_test_run(
            test_file=__file__,
            input_data=prompt,
            output_data=result,
            extra=f"knowledge={wrong_knowledge!r} | london_in_reply={'London' in result}",
        )

        self.assertIn("London", result)

    def test_user_prompt_forwarded_to_model(self):
        """The user prompt must appear as a user-role message."""
        prompt = "What is the boiling point of water?"

        fake = FakeClient("100°C")
        with (
            patch("base_agents.client", None),
            patch("base_agents.OpenAI", return_value=fake),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
        ):
            agent = KnowledgeAugmentedPromptAgent("TestAgent", "teacher", "Water boils at 100°C.")
            agent.execute(prompt)

        messages = fake.completions.last_kwargs.get("messages", [])
        user_content = next(
            (m["content"] for m in messages if m["role"] == "user"), ""
        )

        self.assertEqual(user_content, prompt)


if __name__ == "__main__":
    unittest.main()
