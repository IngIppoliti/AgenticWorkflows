import unittest
from unittest.mock import patch

from base_agents import RoutingAgent


class FakeAgent:
    def __init__(self, name, description, response):
        self.name = name
        self.description = description
        self.response = response

    def execute(self, user_prompt):
        return f"{self.name}:{self.response}:{user_prompt}"


class RoutingAgentTest(unittest.TestCase):
    def test_route_returns_best_matching_agent_response(self):
        fake_travel = FakeAgent(
            "travel", "travel advice", "travel-response"
        )
        fake_weather = FakeAgent(
            "weather", "weather forecast", "weather-response"
        )

        embeddings = {
            "What is the weather?": [1.0, 0.0],
            "travel advice": [0.0, 1.0],
            "weather forecast": [1.0, 0.0],
        }

        def fake_embeddings_create(*args, **kwargs):
            text = kwargs["input"]
            response = type(
                "Response",
                (),
                {
                    "data": [
                        type("Item", (), {"embedding": embeddings[text]})()
                    ]
                },
            )
            return response

        fake_embeddings = type(
            "Embeddings",
            (),
            {"create": staticmethod(fake_embeddings_create)},
        )
        fake_client = type(
            "FakeClient",
            (),
            {"embeddings": fake_embeddings()},
        )()

        with patch("base_agents.get_client", return_value=fake_client):
            router = RoutingAgent([fake_travel, fake_weather])
            result = router.route("What is the weather?")

        self.assertEqual(
            result,
            "weather:weather-response:What is the weather?",
        )


if __name__ == "__main__":
    unittest.main()
