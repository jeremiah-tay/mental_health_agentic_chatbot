import sys
import os
from typing import List, Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

# LangChain and LangGraph imports
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Add project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

# Import your actual calendar tools
from tools.calendar_tools import (
    list_calendars, list_events, insert_event, test_calendar_connection
)

# Import the booking agent
from agents.booking import create_booking_graph

# --- 1. Define Placeholder Tools for the Supervisor ---
@tool
def rag_tool(query: str) -> str:
    """Searches for mental health resources and articles based on the user's query."""
    print(f"--- SUPERVISOR: Calling RAG Tool with query: '{query}' ---")
    return "Based on your query, here is a helpful article on managing anxiety: [link to article]."

@tool
def guidance_tool(query: str) -> str:
    """Recommends Cognitive Behavioral Therapy (CBT) techniques based on the user's situation."""
    print(f"--- SUPERVISOR: Calling Guidance Tool with query: '{query}' ---")
    return "A recommended CBT technique for this situation is 'Cognitive Restructuring'."

llm = ChatOpenAI(model="gpt-5-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.2)

class SupervisorState(TypedDict):
    """
    Represents the state of the Supervisor agent's conversation.
    It holds a list of messages that are appended to across graph runs.
    """
    messages: Annotated[list[AnyMessage], add_messages]

def create_supervisor_graph(llm: ChatOpenAI):
    """
    Creates and compiles the supervisor agent's graph, which can delegate to a sub-graph.
    """
    # Define the tools specific to the booking agent
    supervisor_tools = [rag_tool, guidance_tool]
    supervisor_tool_node = ToolNode(supervisor_tools)

    # Create the booking sub-graph instance
    booking_graph = create_booking_graph(llm)

    # Define the prompt template and create the agent runnable
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         """You are a helpful and empathetic mental health assistant who acts as a supervisor.
        - Your primary role is to analyze the user's request and decide the best course of action.
        - If the request is about scheduling, booking, or managing calendar events, you MUST delegate the task to the booking agent. To delegate, respond with the exact phrase: "delegating_to_booking_agent".
        - For all other requests related to mental health support, resources, or guidance, handle the task yourself using your available tools (rag_tool, guidance_tool).
        - After the booking agent finishes its task, you will receive the full conversation. You must then summarize the outcome for the user and ask if there's anything else you can help with."""),
            MessagesPlaceholder(variable_name="messages"),
        ])
    supervisor_runnable = prompt_template | llm.bind_tools(supervisor_tools)

    # --- Define the Nodes for the Graph ---
    def start_node(state: SupervisorState) -> SupervisorState:
        """Initializes the conversation with a welcome message if it's the first turn."""
        print("--- SUPERVISOR: Entering start_node ---")
        if not state.get("messages"):
            return {"messages": [AIMessage(content="Hello! I'm your mental health assistant. I can help you find resources or book appointments. How may I help you today?")]}
        return state

    def supervisor_router_node(state: SupervisorState):
        """The main decision-making node for the supervisor."""
        print("--- SUPERVISOR: Routing request... ---")
        response = supervisor_runnable.invoke({"messages": state["messages"]})
        
        # Case 1: Supervisor decides to use its own tools
        if response.tool_calls:
            print("--- SUPERVISOR: Decided to use own tools. ---")
            return {"messages": [response]}

        # Case 2: Supervisor decides to delegate to the booking agent
        if "delegating_to_booking_agent" in response.content:
            print("--- SUPERVISOR: Delegating to booking agent. ---")
            # We don't add the "delegating..." message to the history
            return {"messages": [AIMessage(content="delegating_to_booking_agent")]}
            
        # Case 3: Supervisor responds directly
        print("--- SUPERVISOR: Responding directly. ---")
        return {"messages": [response]}

    def booking_agent_node(state: SupervisorState):
        """Invokes the booking agent sub-graph."""
        print("--- SUPERVISOR: Invoking booking agent sub-graph. ---")
        # Run the booking agent graph with the current conversation history
        booking_result = booking_graph.invoke({"messages": state["messages"]})
        # The booking agent's final message is what we want to continue with
        return {"messages": booking_result["messages"]}

    # --- Construct the Graph ---
    builder = StateGraph(SupervisorState)

    builder.add_node("start", start_node)
    builder.add_node("supervisor_router", supervisor_router_node)
    builder.add_node("supervisor_tools", supervisor_tool_node)
    builder.add_node("booking_agent", booking_agent_node)

    # --- Define the Edges ---
    builder.set_entry_point("start")
    builder.add_edge("start", "supervisor_router")

    # Define the conditional routing logic from the supervisor router
    def route_from_supervisor(state: SupervisorState):
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "supervisor_tools"
        if "delegating_to_booking_agent" in last_message.content:
            return "booking_agent"
        return END

    builder.add_conditional_edges("supervisor_router", route_from_supervisor)
    
    # After using its own tools, the supervisor re-evaluates
    builder.add_edge("supervisor_tools", "supervisor_router")
    # The supervisor will only re-engage after a new user prompt is submitted.
    builder.add_edge("booking_agent", END)

    return builder.compile()