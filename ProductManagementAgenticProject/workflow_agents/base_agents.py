import os

# Load environment variables and initialize OpenAI client
load_dotenv()

client = OpenAI(
    base_url = "https://openai.vocareum.com/v1",
    api_key=os.getenv("OPENAI_API_KEY"))


class DirectPrompAgent(Agent):
    """
    An agent specialized in fetching data.
    For this example, it will simulate fetching user data.
    """
  
    def execute(self, user_prompt):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content
