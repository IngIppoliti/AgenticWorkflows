import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from base_agents import RAGKnowledgePromptAgent
from test_logger import log_test_run


# ── Mock helpers ────────────────────────────────────────────────────────────

def make_embedding_response(vector):
    r = MagicMock()
    r.data[0].embedding = vector
    return r


def make_chat_response(content):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def make_mock_client(embedding_vectors=(), chat_reply=""):
    """Return a mock OpenAI client pre-wired with embedding and chat responses."""
    mock = MagicMock()
    if embedding_vectors:
        mock.embeddings.create.side_effect = [
            make_embedding_response(v) for v in embedding_vectors
        ]
    else:
        mock.embeddings.create.return_value = make_embedding_response([1.0, 0.0])
    mock.chat.completions.create.return_value = make_chat_response(chat_reply)
    return mock


# ── Test class ───────────────────────────────────────────────────────────────

class RAGKnowledgePromptAgentTest(unittest.TestCase):

    def _make_agent(self, persona="test professor", chunk_size=100, chunk_overlap=20):
        """Construct an agent with deterministic filenames (no real I/O in __init__)."""
        with patch("base_agents.datetime") as mock_dt, \
             patch("base_agents.uuid") as mock_uuid:
            mock_dt.now.return_value.strftime.return_value = "20240101_120000"
            mock_uuid.uuid4.return_value.hex = "abcdef12abcdef12"
            return RAGKnowledgePromptAgent("test-key", persona, chunk_size, chunk_overlap)

    # ── calculate_similarity ─────────────────────────────────────────────

    def test_similarity_identical_vectors_is_one(self):
        agent = self._make_agent()
        self.assertAlmostEqual(agent.calculate_similarity([1, 0, 0], [1, 0, 0]), 1.0)

    def test_similarity_orthogonal_vectors_is_zero(self):
        agent = self._make_agent()
        self.assertAlmostEqual(agent.calculate_similarity([1, 0], [0, 1]), 0.0)

    def test_similarity_opposite_vectors_is_minus_one(self):
        agent = self._make_agent()
        self.assertAlmostEqual(agent.calculate_similarity([1, 0], [-1, 0]), -1.0)

    # ── chunk_text ───────────────────────────────────────────────────────

    def test_chunk_text_short_text_returns_single_chunk(self):
        agent = self._make_agent(chunk_size=1000)
        result = agent.chunk_text("Hello world")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["chunk_id"], 0)
        self.assertIn("Hello world", result[0]["text"])

        log_test_run(
            test_file=__file__,
            input_data="Hello world",
            output_data=str(result),
            extra="chunk_size=1000 → single chunk expected",
        )

    def test_chunk_text_long_text_produces_multiple_chunks(self):
        agent = self._make_agent(chunk_size=20, chunk_overlap=5)
        long_text = "A" * 60
        with patch("builtins.open", unittest.mock.mock_open()), \
             patch("base_agents.csv") as mock_csv:
            mock_csv.DictWriter.return_value.writeheader = MagicMock()
            mock_csv.DictWriter.return_value.writerow = MagicMock()
            result = agent.chunk_text(long_text)
        self.assertGreater(len(result), 1)

    def test_chunk_text_no_infinite_loop_when_tail_smaller_than_overlap(self):
        # chunk_size=20, chunk_overlap=15 → step=5
        # text of 28 chars: chunk[0:20], then chunk[5:25], then chunk[10:28]
        # Without the break-fix this would loop forever on the last chunk.
        agent = self._make_agent(chunk_size=20, chunk_overlap=15)
        text = "X" * 28
        with patch("builtins.open", unittest.mock.mock_open()), \
             patch("base_agents.csv") as mock_csv:
            mock_csv.DictWriter.return_value.writeheader = MagicMock()
            mock_csv.DictWriter.return_value.writerow = MagicMock()
            result = agent.chunk_text(text)
        self.assertTrue(all(c["chunk_size"] > 0 for c in result))

    def test_chunk_text_each_chunk_has_required_keys(self):
        agent = self._make_agent(chunk_size=20, chunk_overlap=5)
        long_text = "B" * 50
        with patch("builtins.open", unittest.mock.mock_open()), \
             patch("base_agents.csv") as mock_csv:
            mock_csv.DictWriter.return_value.writeheader = MagicMock()
            mock_csv.DictWriter.return_value.writerow = MagicMock()
            result = agent.chunk_text(long_text)
        for chunk in result:
            self.assertIn("chunk_id", chunk)
            self.assertIn("text", chunk)
            self.assertIn("chunk_size", chunk)

    # ── calculate_embeddings ─────────────────────────────────────────────

    def test_calculate_embeddings_calls_api_once_per_chunk(self):
        agent = self._make_agent()
        mock_client = make_mock_client(
            embedding_vectors=[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        )
        chunks_df = pd.DataFrame({"text": ["a", "b", "c"], "chunk_size": [1, 1, 1]})

        with patch("base_agents.pd.read_csv", return_value=chunks_df), \
             patch("pandas.DataFrame.to_csv"), \
             patch("base_agents.OpenAI", return_value=mock_client):
            agent.calculate_embeddings()

        self.assertEqual(mock_client.embeddings.create.call_count, 3)

    def test_calculate_embeddings_returns_dataframe_with_embeddings_column(self):
        agent = self._make_agent()
        mock_client = make_mock_client(
            embedding_vectors=[[1.0, 0.0], [0.0, 1.0]]
        )
        chunks_df = pd.DataFrame({"text": ["chunk one", "chunk two"], "chunk_size": [9, 9]})

        with patch("base_agents.pd.read_csv", return_value=chunks_df), \
             patch("pandas.DataFrame.to_csv"), \
             patch("base_agents.OpenAI", return_value=mock_client):
            result = agent.calculate_embeddings()

        self.assertIn("embeddings", result.columns)
        self.assertEqual(len(result), 2)

    # ── find_prompt_in_knowledge ─────────────────────────────────────────

    def test_find_prompt_returns_model_response(self):
        agent = self._make_agent()
        expected = "Dear students, Clara hosts Crosscurrents."
        mock_client = make_mock_client(
            embedding_vectors=[[1.0, 0.0]],
            chat_reply=expected,
        )
        embeddings_df = pd.DataFrame({
            "text": ["chunk about Crosscurrents podcast"],
            "embeddings": ["[1.0, 0.0]"],
        })

        with patch("base_agents.pd.read_csv", return_value=embeddings_df), \
             patch("base_agents.OpenAI", return_value=mock_client):
            result = agent.find_prompt_in_knowledge("What podcast does Clara host?")

        log_test_run(
            test_file=__file__,
            input_data="What podcast does Clara host?",
            output_data=result,
            extra=f"expected={expected!r} | passed={result == expected}",
        )
        self.assertEqual(result, expected)

    def test_find_prompt_picks_most_similar_chunk(self):
        """Context sent to the LLM must come from the chunk with highest cosine similarity."""
        agent = self._make_agent()
        # Prompt embedding [0.9, 0.1] → closest to chunk 0 embedding [1.0, 0.0]
        mock_client = make_mock_client(
            embedding_vectors=[[0.9, 0.1]],
            chat_reply="answer",
        )
        embeddings_df = pd.DataFrame({
            "text": ["chunk about podcasts", "chunk about whales"],
            "embeddings": ["[1.0, 0.0]", "[0.0, 1.0]"],
        })

        with patch("base_agents.pd.read_csv", return_value=embeddings_df), \
             patch("base_agents.OpenAI", return_value=mock_client):
            agent.find_prompt_in_knowledge("What is the podcast about?")

        user_message = mock_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertIn("chunk about podcasts", user_message)
        self.assertNotIn("chunk about whales", user_message)

    def test_find_prompt_includes_persona_in_system_message(self):
        persona = "You are a college professor"
        agent = self._make_agent(persona=persona)
        mock_client = make_mock_client(
            embedding_vectors=[[1.0, 0.0]],
            chat_reply="ok",
        )
        embeddings_df = pd.DataFrame({
            "text": ["some context"],
            "embeddings": ["[1.0, 0.0]"],
        })

        with patch("base_agents.pd.read_csv", return_value=embeddings_df), \
             patch("base_agents.OpenAI", return_value=mock_client):
            agent.find_prompt_in_knowledge("any question")

        system_message = mock_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn(persona, system_message)


if __name__ == "__main__":
    unittest.main()
