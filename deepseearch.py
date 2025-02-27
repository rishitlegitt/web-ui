import asyncio
import os
import logging
from dotenv import load_dotenv
from src.utils import utils
from src.utils.agent_state import AgentState
from src.utils.deep_research import deep_research  # Adjust this import path if needed

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    # Hard-code the research prompt here:
    research_task = """All Contract related Act Law with their Content (as they are written in the act itself) of Country Italy
Note : extract Entire Law"""
    
    # Create an instance of AgentState to manage stop requests during the research
    agent_state = AgentState()

    # Initialize the language model using your utility function
    llm = utils.get_llm_model(
        provider="google",
        model_name="gemini-2.0-flash-thinking-exp",
        num_ctx=32000,
        temperature=0,
        base_url="",
        api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Run the deep_research function with the provided task and parameters.
    report, file_path = await deep_research(
        task=research_task,
        llm=llm,
        agent_state=agent_state,
        max_search_iterations=10,  # Adjust as needed
        max_query_num=5,           # Adjust as needed
        headless=True,             # Run browser headless
        use_own_browser=False      # Do not use an existing browser instance
        # Additional keyword arguments (e.g., use_vision) can be added here.
    )

    # Output the final report to the console.
    print("=== Deep Research Report ===")
    print(report)
    if file_path:
        print(f"\nReport saved at: {file_path}")

if __name__ == "__main__":
    asyncio.run(main())
