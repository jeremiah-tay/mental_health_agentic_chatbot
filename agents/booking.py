import sys
import os
from typing import List, Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

# LangChain and LangGraph imports
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Add project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

# Import your actual calendar tools
from tools.calendar_tools import (
    list_calendar_list,
    list_calendar_events,
    insert_calendar_event,
    create_calendar
)

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.2)

class BookingAgentState(TypedDict):
    """
    Represents the state of the booking agent's conversation.
    It holds a list of messages that are appended to across graph runs.
    """
    messages: Annotated[list[AnyMessage], add_messages]

def create_booking_graph(llm: ChatOpenAI):
    """
    Creates and compiles a self-contained LangGraph for the booking agent.

    This function sets up the agent's prompt, binds its tools, defines the
    nodes and edges of the graph, and returns the compiled runnable graph.

    Args:
        llm (ChatOpenAI): The language model instance to be used by the agent.

    Returns:
        A compiled LangGraph application.
    """
    # Define the tools specific to the booking agent
    booking_tools = [list_calendar_list, list_calendar_events, insert_calendar_event, create_calendar]
    tool_node = ToolNode(booking_tools)

    # Define the prompt template and create the agent runnable
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         """You are a helpful and empathetic assistant who can also manage a user's Google Calendar.

        **Calendar Tool Instructions:**
        - Always use 'Asia/Singapore' timezone for events
        - Use the format: {{'dateTime': '2025-10-06T14:00:00', 'timeZone': 'Asia/Singapore'}}
        - Use 'primary' as the default calendar_id unless specified otherwise
        - When creating events, always provide both start and end times

        **Event Creation Format:**
        For appointments, use this exact format:
        - start: {{'dateTime': 'YYYY-MM-DDTHH:MM:SS', 'timeZone': 'Asia/Singapore'}}
        - end: {{'dateTime': 'YYYY-MM-DDTHH:MM:SS', 'timeZone': 'Asia/Singapore'}}

        **General Conversation:**
        For all other conversations, maintain a supportive and patient tone."""),
        MessagesPlaceholder(variable_name="messages"),
    ])
    agent_runnable = prompt_template | llm.bind_tools(booking_tools)

    # --- Define the Nodes for the Graph ---
    def agent_node(state: BookingAgentState):
        """
        The primary node that calls the LLM to decide on the next action.
        The response from the LLM is added to the state.
        """
        response = agent_runnable.invoke({"messages": state["messages"]})
        return {"messages": [response]}

    # --- Construct the Graph ---
    builder = StateGraph(BookingAgentState)

    # Add the agent node and the tool node to the graph
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    # Set the entry point for the graph
    builder.set_entry_point("agent")

    # Define the conditional logic for routing
    # If the agent's response contains tool calls, route to the 'tools' node.
    # Otherwise, the graph run ends.
    builder.add_conditional_edges(
        "agent",
        tools_condition,
    )
    
    # After the 'tools' node is executed, the flow always returns to the 'agent' node
    # to process the tool output.
    builder.add_edge("tools", "agent")

    # Compile the graph and return it
    return builder.compile()