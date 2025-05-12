from google.adk.agents.llm_agent import LlmAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import VertexAiSearchTool
from task_manager import AgentWithTaskManager

YOUR_DATASTORE_ID = "hr001_1693905991585"

vertex_search_tool = VertexAiSearchTool(data_store_id=YOUR_DATASTORE_ID)

class VertexAISearchAgent(AgentWithTaskManager):
    """An agent that helps answer Human Resources Management Regulations."""

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self):
        self._agent = self._build_agent()
        self._user_id = 'remote_agent'
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    def get_processing_message(self) -> str:
        return 'Processing the Human Resources Management Regulations request...'

    def _build_agent(self) -> LlmAgent:
        """Builds the LLM agent for the Human Resources Management Regulations agent."""
        return LlmAgent(
            model='gemini-2.0-flash-001',
            name='Human Resources Management Regulations_agent',
            description=(
                'This agent answers questions about the Human Resources Management Regulations for the employees'
            ),
            instruction="""
    You are an agent who answers questions about the Human Resources Management Regulations for employees based on information found in the document store: {YOUR_DATASTORE_ID}.

    Use the search tool to find relevant information before answering.
    If the answer isn't in the documents, say that you couldn't find the information.
    """,
            tools=[
                vertex_search_tool,
            ],
        )
