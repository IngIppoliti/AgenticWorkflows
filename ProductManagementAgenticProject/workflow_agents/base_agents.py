import math
import os

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables and initialize OpenAI client lazily.
load_dotenv()

client = None


def get_client():
    global client

    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=api_key,
        )

    return client


class DirectPromptAgent:
    """
    An agent specialized in fetching data.
    For this example, it will simulate fetching user data.
    """
    def __init__(self, name):
        self.name = name

    def execute(self, user_prompt):
        response = get_client().chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )

        return response.choices[0].message.content


class AugmentedPromptAgent:
    """
    An agent that can fetch data from external sources and use it to
    answer prompts. For this example, it will simulate fetching user
    data and using it to answer a prompt.
    """
    def __init__(self, name, personas):
        self.name = name
        self.personas = personas

    def execute(self, user_prompt):
        system_prompt = f"""You are a {self.personas}"""
        response = get_client().chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content


class KnowledgeAugmentedPromptAgent:
    """
    An agent that answers using only explicitly provided knowledge
    and persona instructions.
    """
    def __init__(self, name, persona, knowledge):
        self.name = name
        self.persona = persona
        self.knowledge = knowledge

    def respond(self, user_prompt):
        system_prompt = (
            f"You are {self.persona} knowledge-based assistant. "
            "Forget all previous context.\n"
            f"Use only the following knowledge to answer, do not use "
            f"your own knowledge: {self.knowledge}\n"
            "Answer the prompt based on this knowledge, not your own."
        )

        response = get_client().chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )

        return response.choices[0].message.content


class ActionPlanningAgent:
    """
    Extracts a clean list of action steps from a user prompt using
    the provided knowledge and OpenAI's chat model.
    """

    def __init__(self, knowledge="", api_key=None):
        self.knowledge = knowledge
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None

        if self.api_key:
            self.client = OpenAI(
                base_url="https://openai.vocareum.com/v1",
                api_key=self.api_key,
            )

    def _clean_steps(self, text):
        steps = []

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            line = line.lstrip("-•*0123456789. ").strip()
            if not line:
                continue

            lowered = line.lower()
            if lowered.startswith(("here are", "steps:", "action plan:")):
                continue

            steps.append(line)

        return steps

    def respond(self, user_prompt):
        if self.client is None:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        system_prompt = (
            "You are an Action Planning Agent. Extract the action steps "
            "needed to complete the task described in the user's prompt. "
            f"Use the following knowledge when helpful: {self.knowledge}. "
            "Return the steps as a concise list."
        )

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )

        raw_text = response.choices[0].message.content or ""
        return self._clean_steps(raw_text)


class RoutingAgent:
    """
    Routes a user prompt to the most relevant agent using embeddings
    and cosine similarity.
    """

    def __init__(self, agents=None):
        self.agents = agents or []

    def get_embedding(self, text):
        response = get_client().embeddings.create(
            model="text-embedding-3-large",
            input=text,
        )
        return response.data[0].embedding

    def _cosine_similarity(self, left, right):
        dot_product = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))

        if left_norm == 0 or right_norm == 0:
            return 0.0

        return dot_product / (left_norm * right_norm)

    def route(self, user_prompt):
        if not self.agents:
            raise ValueError("No agents are available for routing.")

        prompt_embedding = self.get_embedding(user_prompt)

        best_agent = None
        best_score = -1.0

        for entry in self.agents:
            if isinstance(entry, dict):
                description = entry.get("description", "")
                agent_callable = entry.get("agent")
            else:
                description = getattr(entry, "description", "")
                agent_callable = entry

            description_embedding = self.get_embedding(description)
            similarity = self._cosine_similarity(
                prompt_embedding,
                description_embedding,
            )

            if similarity > best_score:
                best_score = similarity
                best_agent = agent_callable

        if best_agent is None:
            raise ValueError("Unable to select a routing target.")

        if isinstance(best_agent, dict):
            callable_agent = best_agent.get("agent")
        else:
            callable_agent = best_agent

        if hasattr(callable_agent, "execute"):
            return callable_agent.execute(user_prompt)
        if hasattr(callable_agent, "respond"):
            return callable_agent.respond(user_prompt)
        if callable(callable_agent):
            return callable_agent(user_prompt)

        raise TypeError(
            "The selected agent does not provide a callable response method."
        )


class EvaluationAgent:
    """
    Evaluates a worker response against criteria and returns a concise
    assessment along with the number of interaction rounds used.
    """
    def __init__(self, max_interactions=5):
        self.max_interactions = max_interactions

    def respond(self, worker_agent, user_prompt, criteria=None):
        if criteria is None:
            criteria = (
                "accuracy, clarity, and whether the response follows "
                "the provided knowledge instructions"
            )

        worker_response = None
        evaluation_result = ""
        correction_instructions = ""
        iterations_performed = 0

        for _ in range(self.max_interactions):
            iterations_performed += 1
            if hasattr(worker_agent, "respond"):
                worker_response = worker_agent.respond(user_prompt)
            else:
                worker_response = worker_agent.execute(user_prompt)

            evaluation_prompt = (
                "Evaluate the worker response against these criteria: "
                f"{criteria}.\n\nWorker response:\n{worker_response}"
            )

            evaluation_response = get_client().chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert evaluator. Be concise, "
                            "truthful, and focus on accuracy, clarity, "
                            "and instruction-following."
                        ),
                    },
                    {"role": "user", "content": evaluation_prompt},
                ],
                temperature=0,
            )
            evaluation_result = evaluation_response.choices[0].message.content

            correction_prompt = (
                "Based on the evaluation, provide short correction "
                f"instructions for the worker response.\n\nEvaluation:\n"
                f"{evaluation_result}"
            )

            correction_response = get_client().chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a correction assistant. Return brief "
                            "and actionable guidance only."
                        ),
                    },
                    {"role": "user", "content": correction_prompt},
                ],
                temperature=0,
            )
            correction_instructions = (
                correction_response.choices[0].message.content
            )

            if (
                "no issues" in evaluation_result.lower()
                or "good" in evaluation_result.lower()
            ):
                break

        return {
            "final_response": worker_response,
            "evaluation_result": evaluation_result,
            "correction_instructions": correction_instructions,
            "iterations": iterations_performed,
        }
