import csv
import logging
import math
import os
import re
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables and initialize OpenAI client lazily.
load_dotenv()

client = None

logger = logging.getLogger(__name__)


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

    def execute(self, user_prompt):
        system_prompt = (
            f"You are a {self.persona} knowledge-based assistant. "
            "Forget all previous context.\n"
            f"Use only the following knowledge to answer: {self.knowledge}\n"
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

# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): Persona description for the agent.
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text):
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        separator = "\n"
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

        chunks, start, chunk_id = [], 0, 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)

            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "chunk_size": end - start,
                "start_char": start,
                "end_char": end
            })

            if end >= len(text):
                break
            start = end - self.chunk_overlap
            chunk_id += 1

        with open(f"chunks-{self.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

        return chunks

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['text'].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding='utf-8', index=False)
        return df

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(eval(x)))
        df['similarity'] = df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))

        best_chunk = df.loc[df['similarity'].idxmax(), 'text']

        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context."},
                {"role": "user", "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"}
            ],
            temperature=0
        )

        return response.choices[0].message.content
    
class ActionPlanningAgent:
    """
    Extracts a clean list of action steps from a user prompt using
    the provided knowledge and OpenAI's chat model.
    """

    def __init__(self, name, knowledge=""):
        self.name = name
        self.knowledge = knowledge


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

    def execute(self, user_prompt):

        system_prompt = (

            f"""
            You are an Action Planning Agent.

            Your task is to extract the action steps needed to complete the user's objective.

            Use this knowledge when helpful:
            {self.knowledge}

            Rules:
            - Return only a concise list of actionable steps.
            - Each step must be a single action.
            - Do not include explanations, introductions, or conclusions.
            - Do not add steps that are not implied by the user's request.
            """
        )

        response = get_client().chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )

        raw_text = response.choices[0].message.content or ""
        logger.debug("Raw response from ActionPlanningAgent: %s", raw_text)
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

    def route(self, task):
        if not self.agents:
            raise ValueError("No agents are available for routing.")

        task_embedding = self.get_embedding(task)

        best_agent = None
        best_agent_name = ""
        best_score = -1.0

        for entry in self.agents:
            if isinstance(entry, dict):
                name = entry.get("name", "")
                description = entry.get("description", "")
                agent_callable = entry.get("func")
            else:
                name = getattr(entry, "name", "")
                description = getattr(entry, "description", "")
                agent_callable = entry

            description_embedding = self.get_embedding(description)
            similarity = self._cosine_similarity(
                task_embedding,
                description_embedding,
            )

            if similarity > best_score:
                best_score = similarity
                best_agent = agent_callable
                best_agent_name = name
            
        logger.info(f"Task → {task} Routing → {best_agent_name} (score: {best_score:.3f})")


        if best_agent is None:
            raise ValueError("Unable to select a routing target.")

        if hasattr(best_agent, "execute"):
            output = best_agent.execute(task)
        elif callable(best_agent):
            output = best_agent(task)
        else:
            raise TypeError(
                "The selected agent does not provide a callable response method."
            )

        return output, best_agent_name


class EvaluationAgent:
    """
    Evaluates a worker agent's responses against role-specific criteria and
    iteratively refines them up to max_interactions rounds.
    """
    def __init__(
        self,
        persona: str,
        evaluation_criteria: str,
        agent_to_evaluate,
        max_interactions: int = 5,
    ):
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.agent_to_evaluate = agent_to_evaluate
        self.max_interactions = max_interactions

    def evaluate(self, worker_response: str) -> str:
        evaluation_prompt = (
            "Evaluate the worker response against these criteria: "
            f"{self.evaluation_criteria}.\n\nWorker response:\n{worker_response}"
        )
        response = get_client().chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.persona},
                {"role": "user", "content": evaluation_prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    def _generate_correction_instructions(self, evaluation_result: str) -> str:
        correction_prompt = (
            "Based on the evaluation, provide short correction "
            f"instructions for the worker response.\n\nEvaluation:\n"
            f"{evaluation_result}"
        )
        response = get_client().chat.completions.create(
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
        return response.choices[0].message.content

    def execute(self, user_prompt: str) -> dict:
        worker_response = None
        evaluation_result = ""
        correction_instructions = ""
        iterations_performed = 0

        revised_prompt = user_prompt
        for _ in range(self.max_interactions):
            iterations_performed += 1
            worker_response = self.agent_to_evaluate.execute(revised_prompt)
            evaluation_result = self.evaluate(worker_response)
            logger.debug("Iteration %d worker response: %s", iterations_performed, worker_response)
            logger.debug("Iteration %d evaluation: %s", iterations_performed, evaluation_result)
            lowered = evaluation_result.lower()
            if any(
                kw in lowered
                for kw in ("no issues", "good", "acceptable", "meets the criteria", "is accurate")
            ):
                break

            correction_instructions = self._generate_correction_instructions(evaluation_result)
            revised_prompt = (
                f"{user_prompt}\n\n"
                f"Revised instructions based on evaluation: {correction_instructions}"
            )

        return {
            "final_response": worker_response,
            "evaluation_result": evaluation_result,
            "correction_instructions": correction_instructions,
            "iterations": iterations_performed,
        }
