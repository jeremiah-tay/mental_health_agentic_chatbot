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

# Import the safety check
from risk_classifier.safetycheck import SafetyCheck
from risk_classifier.crisis_response import CrisisResponse

# Import the risk classifier
try:
    risk_classifier = SafetyCheck(base_dir="risk_classifier/saved_models")
    print("✅ Risk classifier loaded successfully")
except FileNotFoundError as e:
    print(f"⚠️ Risk classifier models not found: {e}")
    print("⚠️ Running without risk assessment - models need to be downloaded")
    risk_classifier = None
except Exception as e:
    print(f"⚠️ Risk classifier failed to load: {e}")
    risk_classifier = None

# Import your actual calendar tools
from tools.calendar_tools import (
    list_calendars, list_events, insert_event, test_calendar_connection
)

# Import the CBT tools
from tools.cbt_tools import (
    select_cbt_tool
)

# Import the RAG tools
from tools.rag_tools import (
    rag_tool
)

# Import the booking agent
from agents.booking import create_booking_graph


llm = ChatOpenAI(
    model = "gpt-5-mini",
    api_key = os.getenv("OPENAI_API_KEY"),
    temperature = 0.2,
    max_tokens = 250)

class SupervisorState(TypedDict):
    """
    Represents the state of the Supervisor agent's conversation.
    It holds a list of messages that are appended to across graph runs.
    """
    messages: Annotated[list[AnyMessage], add_messages]
    conversation_ended: bool

def create_supervisor_graph(llm: ChatOpenAI):
    """
    Creates and compiles the supervisor agent's graph, which can delegate to a sub-graph.
    """
    # Define the tools specific to the booking agent
    supervisor_tools = [rag_tool, select_cbt_tool]
    supervisor_tool_node = ToolNode(supervisor_tools)

    # Create the booking sub-graph instance
    booking_graph = create_booking_graph(llm)

    # Define the prompt template and create the agent runnable
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         """
        You are a helpful and empathetic mental health supervisor. 
        Your primary role is to analyze the user's request and determine the most appropriate action.

        DECISION LOGIC:
        1. **Booking/Scheduling**: If the user wants to book, schedule, reschedule, cancel, or check appointment availability → delegate to booking agent (respond with: "delegating_to_booking_agent")

        2. **Mental Health Support**: For all other mental health requests, use your available tools:
        - Use `rag_tool` for general information questions about mental health topics
        - Use `select_cbt_tool` for specific mental health concerns requiring therapeutic techniques
        - The tools themselves contain detailed guidance on when and how to use them appropriately

        3. **Casual Conversation**: For greetings, small talk, or non-mental health topics → respond directly and empathetically

        4. **Conversation End**: If user indicates they want to end the conversation → respond warmly and include "__END__" in your response

        Keep responses concise (1-3 sentences) unless the user specifically asks for detailed information.

        Example Conversation:

        Example 1: Booking
        User: Can you help me book an appointment?
        Your Thought: The user is asking for a booking, so I need to delegate that task to the booking agent.
        Your Response: delegating_to_booking_agent
        
        Example 2: Specific Clinical Technique (CBT)
        User: I can't stop worrying about my exams, I am so stressed out.
        Your Thought: The use is expressing a specific  problem ('worrying', 'stressed out') and needs help. This is a request for an actional technique, not general information. I must use the select_cbt_technique tool.
        Your Action: select_cbt_technique(user_mental_health_concern="I can't stop worrying about my exams, I am so stressed out.")

        Example 3: General RAG
        User: What is Cognitive Behavioral Therapy (CBT)?
        Your Thoguht: The user is asking for general information about CBT, so I must use the rag_tool.
        Your Action: rag_tool(query="What is Cognitive Behavioral Therapy (CBT)?")

        Example 4: Casual Conversation
        User: Hi, how are you?
        Your Thought: The user is just having a casual conversation, so I must respond directly to the user.
        Your Response: Hello! I'm your mental health assistant. I can help you find resources or book appointments. How may I help you today?

        Example 5: End Conversation
        User: Thank you, goodbye!
        Your Thought: The user is saying goodbye and wants to end the conversation.
        Your Response: Thank you for chatting with me today. Take care, and remember I'm here whenever you need support! __END__
        """),
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
    
    def risk_assessment_node(state: SupervisorState):
        """Assesses if the user's latest message indicates risk."""
        print("--- SUPERVISOR: Running risk assessment... ---")
        
        # Check if risk classifier is available
        if risk_classifier is None:
            print("--- SUPERVISOR: Risk classifier not available, proceeding normally. ---")
            return state
        
        # Get the latest user message
        latest_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                latest_message = msg.content
                break
        
        if not latest_message:
            print("--- SUPERVISOR: No user message found, proceeding normally. ---")
            return state
        
        # Run risk classification
        try:
            risk_score = risk_classifier(latest_message)
            print(f"--- SUPERVISOR: Risk assessment result: {risk_score} ---")
            
            if risk_score == 1:  # At risk
                print("--- SUPERVISOR: CRISIS DETECTED - Returning crisis response. ---")
                crisis_response = CrisisResponse(latest_message)
                crisis_response = AIMessage(content=crisis_response)
                return {
                    "messages": [crisis_response],
                    "conversation_ended": True
                }
            else:
                print("--- SUPERVISOR: No risk detected, proceeding with normal flow. ---")
                return state
                
        except Exception as e:
            print(f"--- SUPERVISOR: Risk assessment failed: {e}, proceeding normally. ---")
            return state
    
    def route_from_risk_assessment(state: SupervisorState):
        """Route from risk assessment based on whether conversation ended."""
        if state.get("conversation_ended"):
            return END
        else:
            return "supervisor_router"

    def supervisor_router_node(state: SupervisorState):
        """The main decision-making node for the supervisor."""
        print("--- SUPERVISOR: Routing request... ---")

        response = supervisor_runnable.invoke({"messages": state["messages"]})

        # Check if the supervisor decided to end the conversation
        if "__END__" in response.content:
            print("--- SUPERVISOR: LLM decided to end conversation. ---")
            
            # Clean up the message for the user (remove the signal)
            final_message_content = response.content.replace("__END__", "").strip()
            final_message = AIMessage(content=final_message_content)
            
            # Return the final goodbye message and end the graph
            return {
                "messages": [final_message],
                "conversation_ended": True
            }
        
        # Case 1: Supervisor decides to use its own tools
        if response.tool_calls:
            print("--- SUPERVISOR: Decided to use own tools. ---")
            print(f"--- SUPERVISOR: Response: {response} ---")
            return {"messages": [response]}

        # Case 2: Supervisor decides to delegate to the booking agent
        if "delegating_to_booking_agent" in response.content:
            print("--- SUPERVISOR: Delegating to booking agent. ---")
            print(f"--- SUPERVISOR: Response: {response} ---")
            # We don't add the "delegating..." message to the history
            return {"messages": [AIMessage(content="delegating_to_booking_agent")]}
            
        # Case 3: Supervisor responds directly
        print("--- SUPERVISOR: Responding directly. ---")
        print(f"--- SUPERVISOR: Response: {response} ---")
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
    builder.add_node("risk_assessment", risk_assessment_node)
    builder.add_node("supervisor_router", supervisor_router_node)
    builder.add_node("supervisor_tools", supervisor_tool_node)
    builder.add_node("booking_agent", booking_agent_node)

    # --- Define the Edges ---
    builder.set_entry_point("start")
    builder.add_edge("start", "risk_assessment")
    builder.add_conditional_edges("risk_assessment", route_from_risk_assessment)

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