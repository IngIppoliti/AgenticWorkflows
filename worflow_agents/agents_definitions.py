class Agent:
    """
    A base class for our agents.
    While not strictly necessary for simple cases,
    it can be useful for defining common interfaces or utilities.
    """
    def __init__(self, name):
        self.name = name
        print(f"Agent '{self.name}' initialized.")

    def execute(self, data=None):
        """
        A generic method to execute the agent's task.
        Specific agents will override this.
        """
        raise NotImplementedError("Each agent must implement the 'execute' method.")

class DataFetchingAgent(Agent):
    """
    An agent specialized in fetching data.
    For this example, it will simulate fetching user data.
    """
    def __init__(self, name, data_source):
        super().__init__(name)
        self.data_source = data_source
        print(f"DataFetchingAgent will fetch from: {self.data_source}")

    def execute(self, user_id):
        """
        Simulates fetching data for a given user_id.
        In a real scenario, this might involve an API call or database query.
        """
        print(f"'{self.name}' is fetching data for user_id: {user_id} from {self.data_source}...")
        # Simulate data fetching
        if user_id == "123":
            return {"user_id": "123", "name": "Alice Wonderland", "occupation": "Dreamer"}
        else:
            return {"user_id": user_id, "error": "User not found"}

class DataProcessingAgent(Agent):
    """
    An agent specialized in processing data.
    For this example, it will extract key information from the fetched data.
    """
    def __init__(self, name, fields_to_extract=None):
        super().__init__(name)
        self.fields_to_extract = fields_to_extract if fields_to_extract else ["name", "occupation"]
        print(f"DataProcessingAgent will extract fields: {self.fields_to_extract}")

    def execute(self, fetched_data):
        """
        Processes the fetched data to extract specified fields.
        """
        print(f"'{self.name}' is processing data: {fetched_data}")
        if "error" in fetched_data:
            return {"processed_info": None, "error": fetched_data["error"]}

        processed_info = {}
        for field in self.fields_to_extract:
            processed_info[field] = fetched_data.get(field, "N/A")

        return {"processed_info": processed_info, "original_data_keys": list(fetched_data.keys())}
