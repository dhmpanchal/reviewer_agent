# Make sure your environment has LANGCHAIN_API_KEY set

from langsmith import Client
from dotenv import load_dotenv
load_dotenv()

client = Client()

# Replace with your actual run ID from LangSmith
run_id = "019c2d7c-48a2-78b3-8c23-879f90bb3a6f"

# Update the run status to "aborted" (equivalent to stopping it)
client.update_run(run_id=run_id, status="aborted")