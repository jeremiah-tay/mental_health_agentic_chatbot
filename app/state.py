from typing import List, Dict, Any
from langgraph.graph import MessagesState

class SupervisorState(MessagesState):
    current_agent: str = ""
    task_assignments: Dict[str, List[str]] = {} # Track what each agent should do
    agent_outputs: Dict[str, Any] = {} # Store outputs from each agent
    workflow_stage: str = "initial" # Track workflow
    iteration_count: int = 0
    max_iterations: int = 10
    final_output: str = ""