from typing import Dict, Any
from langchain_core.messages import BaseMessage
from app.state import SupervisorState
from chains.supervisor_chain import create_supervisor_chain

# Assuming SupervisorState and create_supervisor_chain are defined as you provided

def supervisor_agent(state: SupervisorState) -> Dict[str, Any]:
    """
    Supervisor agent that orchestrates the workflow by deciding the next agent to act.

    Args:
        state: The current state of the workflow.

    Returns:
        A dictionary with the updated state.
    """
    print("---SUPERVISOR---")
    
    # 1. Get the supervisor chain (the decision-maker)
    supervisor_chain = create_supervisor_chain()
    
    # 2. Extract messages from the state
    messages: list[BaseMessage] = state.get("messages", [])
    user_message = messages[-1].content if messages else ""
    chat_history = "\n".join([f"{msg.type}: {msg.content}" for msg in messages[:-1]])
    
    # 3. Invoke the chain to get the next agent's name
    next_agent_name = supervisor_chain.invoke({
        "user_message": user_message,
        "chat_history": chat_history,
    }).content
    
    print(f"Supervisor decided next agent is: {next_agent_name}")
    
    # 4. Update the state with the decision and increment the loop counter
    state["current_agent"] = next_agent_name
    state["iteration_count"] = state.get("iteration_count", 0) + 1
    
    return state