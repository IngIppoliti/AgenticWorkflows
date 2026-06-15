import os
import base_agents.

# Load environment variables and initialize OpenAI client
load_dotenv()

client = OpenAI(
    base_url = "https://openai.vocareum.com/v1",
    api_key=os.getenv("OPENAI_API_KEY"))

