import sys
import os
import pytz
from typing import List, Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from datetime import datetime, timedelta

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
    list_calendars, list_events, insert_event, test_calendar_connection
)

llm = ChatOpenAI(model="gpt-5-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.2)

class BookingAgentState(TypedDict):
    """
    Represents the state of the booking agent's conversation.
    It holds a list of messages that are appended to across graph runs.
    """
    messages: Annotated[list[AnyMessage], add_messages]

def get_current_singapore_time():
        """Get current time in Singapore timezone"""
        singapore_tz = pytz.timezone('Asia/Singapore')
        return datetime.now(singapore_tz)

def create_booking_graph(llm: ChatOpenAI):
    """
    Creates and compiles a self-contained LangGraph for the booking agent.
    """
    # Get current date and time
    current_time = get_current_singapore_time()
    current_date_str = current_time.strftime('%A, %B %d, %Y')
    current_time_str = current_time.strftime('%I:%M %p')
    tomorrow_str = (current_time + timedelta(days = 1)).strftime('%A, %B %d, %Y')
    day_after_tomorrow_str = (current_time + timedelta(days = 2)).strftime('%A, %B %d, %Y')
    
    # Define the tools specific to the booking agent
    booking_tools = [list_calendars, list_events, insert_event]
    tool_node = ToolNode(booking_tools)

    # Define the prompt template and create the agent runnable
    prompt = ChatPromptTemplate.from_messages([
    ("system", f"""You are a booking assistant. Your ONLY job is to call the insert_event tool when you have all the required information.

    **CURRENT DATE AND TIME:** Today is {current_date_str} at {current_time_str} Singapore Time.

    **IMPORTANT: Look at the ENTIRE conversation history to gather all information before calling insert_event.**

    **REQUIRED INFORMATION TO CALL insert_event:**
    1. summary (session title or "Appointment with [name]")
    2. start_datetime (in format YYYY-MM-DDTHH:MM:SS)
    3. end_datetime (in format YYYY-MM-DDTHH:MM:SS, usually start + 1 hour)

    **CONVERSATION ANALYSIS:**
    - Read ALL previous messages in the conversation
    - Extract date/time information from earlier messages
    - Extract name/title information from earlier messages
    - Combine all information to make the booking

    **EXAMPLES:**
    
    **Conversation 1:**
    User: "Book a session next week at 2pm"
    AI: "What's your name?"
    User: "John"
    AI: [Look at conversation - user wants next week at 2pm, name is John]
    AI: [Call insert_event with summary="Appointment with John", start_datetime="2024-10-14T14:00:00", end_datetime="2024-10-14T15:00:00"]

    **Conversation 2:**
    User: "Book a session for tomorrow"
    AI: "What time tomorrow and what's your name?"
    User: "2pm and my name is Sarah"
    AI: [Look at conversation - user wants tomorrow at 2pm, name is Sarah]
    AI: [Call insert_event with summary="Appointment with Sarah", start_datetime="2024-10-15T14:00:00", end_datetime="2024-10-15T15:00:00"]

    **CRITICAL: When you have ALL required information from the conversation, call insert_event immediately!**
    
    **INFORMATION EXTRACTION PROCESS:**
    1. Read the ENTIRE conversation from the beginning
    2. Look for:
    - Date/time requests (e.g., "tomorrow", "next week", "2pm")
    - Names provided by the user
    - Session titles provided by the user
    3. Calculate exact dates from relative references
    4. Determine the session title:
    - If user provided a specific title: use that
    - If user provided their name: use "Appointment with [name]"
    - If user provided both: use the specific title
    5. Once you have summary, start_datetime, end_datetime, call insert_event

    **INFORMATION GATHERING:**
    If the user doesn't provide complete information, ask for ALL missing information in ONE message:
    - Date and time
    - Name

    Example: "I'd be happy to book a session! I need to know:
    1. What day and time would you like? (weekdays, 9 AM - 6 PM)
    2. What's your name so that I can make a booking for this session?"

    Then when they provide the information, call insert_event immediately.
    """),
    MessagesPlaceholder(variable_name="messages"),
])
    agent_runnable = prompt | llm.bind_tools(booking_tools)

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