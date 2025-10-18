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

# Import the CBT tools
from tools.cbt_tools import (
    select_cbt_technique
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
    supervisor_tools = [rag_tool, select_cbt_technique]
    supervisor_tool_node = ToolNode(supervisor_tools)

    # Create the booking sub-graph instance
    booking_graph = create_booking_graph(llm)

    # Define the prompt template and create the agent runnable
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         """You are a helpful and empathetic mental health supervisor. 
         Your primary role is to analyze the user's request and determine the most appropriate action to provide the best possible support.

        IMPORTANT: 
        - Keep all responses concise and to the point. Aim for 1-3 sentences maximum unless the user specifically asks for detailed information. 
        - Break your responsese into multiple paragraphs if needed.
        - Use bullet points if needed.

        You must follow this decision-making logic precisely:
        1. **Handle Booking and Scheduling**: If the user's request is at all related to booking, scheduling, rescheduling, cancelling, or checking availability for appointments, you MUST delegate the task to the booking agent.
        - To delegate, respond with the exact phrase: "delegating_to_booking_agent".
        2. **Select Clinical Techniques**: If the user describes a specific mental health concern, a negative feeling, or a problem and asks for actionable help, a specific exercise, or a technique to manage it, you MUST use the select_cbt_technique tool. 
        - This tool is designed to find the most clinically appropriate CBT technique for their problem.
        - Do NOT use the rag_tool for this. The rag_tool is for information, not for clinical application.
        3. **Provide General Information**: If the user asks for general information about mental health, resources, or guidance, you MUST use the rag_tool.
        - This tool is designed to provide general information about mental health, resources, or guidance.
        - Do NOT use the select_cbt_technique tool for this. The select_cbt_technique tool is for clinical application, not for general information.
        4. **Handle Conversation**: If the user is just having a casual conversation, you MUST respond directly to the user.
        - For simple conversational turns (greetings, goodbyes, thank-yous), respond empathetically and conversationally as a supervisor.
        5. **Summarize Outcome**: After the booking agent finishes its task, you must summarize the outcome for the user and ask if there's anything else you can help with.
        - To summarize the outcome, respond with the exact phrase: "summarizing_outcome".

        Remember: Be concise, empathetic, and helpful. Avoid lengthy explanations unless specifically requested.

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