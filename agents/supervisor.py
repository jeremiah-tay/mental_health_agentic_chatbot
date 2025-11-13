import sys
import os
from typing import List, Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

# Add project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from riskclassifier_v2.safetycheck import SafetyCheck
from riskclassifier_v2.crisis_response import CrisisResponse

# Import the risk classifier
try:
    risk_classifier = SafetyCheck(base_dir="riskclassifier_v2/saved_models")
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

#EDITED - add conversation tracking fields to state
class SupervisorState(TypedDict):
    """
    Represents the state of the Supervisor agent's conversation.
    It holds a list of messages that are appended to across graph runs.
    """
    messages: Annotated[list[AnyMessage], add_messages]
    conversation_ended: bool
    conversation_id: str  #NEW - track conversation id
    tools_called: List[str]  #NEW - track which tools were called
    tools_result: dict  #NEW - track tool results
    agents_used: List[str]  #NEW - track which agents were used
    risk_probability: float

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

        CRITICAL: You are speaking DIRECTLY to the user. Never describe what you should do - just do it and never say your thoughts out loud. 

        **LOCAL HEALTH-RELATED RESOURCES (Singapore) **
        When appropriate, provide these resources:
        - Suicidal thoughts, emotional crisis - Samaritans of Singapore: 1767
        - General mental health support- National Mindline: 1771 
        - medical emergencies - Singapore Civil Defence Force: 995
        - immediate danger - Singapore Police Force: 999
        - violence and sexual harrassment- National Anti-Violence and Sexual Harassment Helpline: 1800 777 0000
        
        Only mention these resources when:
        - User explicitly asks for help resources
        - Situation suggests they need professional support
        - User asks "who can I call" or "where can I get help"
        - Do not force it when it is not appropriate

        DECISION LOGIC:
        1. **Booking/Scheduling**: If the user wants to book, schedule, reschedule, cancel, or check appointment availability → delegate to booking agent (respond with: "delegating_to_booking_agent")

        2. **Mental Health Support**: For all other mental health requests, use your available tools:
        - Use `rag_tool` for general information questions about mental health topics
        - Use `select_cbt_tool` for specific mental health concerns requiring therapeutic techniques
        - The tools themselves contain detailed guidance on when and how to use them appropriately

        3. **Mental Health Support - After Tool Results**:
            When you receive results from select_cbt_tool:
            - You will see the selected CBT technique and guidance in the tool results
            - Generate a warm, empathetic therapeutic response that:
            a) Validates the user's feelings with empathy
            b) Do not go out of your way to mention the technique, especially if it makes the sentence unnatural
            c) Provides 1-2 concrete, actionable steps they can try right now
            d) Offers follow-up support if appropriate
            - Keep it conversational and supportive (2-4 sentences)
            - Speak directly to the user, not about what you should do
    
            When you receive results from rag_tool:
            - Use the retrieved information to answer the user's question
            - Present the information in a clear, empathetic way
            - Keep it concise unless more detail is requested

        4. **Casual Conversation**: For greetings, small talk, or non-mental health topics → respond directly and warmly

        5. **Conversation End**: If user indicates they want to end the conversation → respond warmly and include "__END__" in your response

        Keep responses concise (1-3 sentences) unless the user specifically asks for detailed information.

        Example Conversation:

        Example 1: Booking
        User: Can you help me book an appointment?
        Your Thought: The user is asking for a booking, so I need to delegate that task to the booking agent.
        Your Response: delegating_to_booking_agent
        
        Example 2: Specific Clinical Technique (CBT)
        User: I can't stop worrying about my exams, I am so stressed out.
        Your Thought: The user is expressing a specific  problem ('worrying', 'stressed out') and needs help. This is a request for an actional technique, not general information. I must use the select_cbt_technique tool.
        Your Action: select_cbt_technique(user_mental_health_concern="I can't stop worrying about my exams, I am so stressed out.")

        Example 3: After CBT Tool Returns (Second Pass - Generate Response)
        [Tool returned: mindfulness technique for chronic worry]
        Response: I hear you — constant worrying about exams can be exhausting. Let's try a quick exercise: pause for 2 minutes, breathe slowly, and when worries pop up just label them as "worrying" without judging, then return to your breath. Would you like me to guide you through a longer breathing exercise?

        Example 4: General RAG
        User: What is Cognitive Behavioral Therapy (CBT)?
        Your Thought: The user is asking for general information about CBT, so I must use the rag_tool.
        Your Action: rag_tool(query="What is Cognitive Behavioral Therapy (CBT)?")

        Example 5: Casual Conversation
        User: Hi, how are you?
        Your Thought: The user is just having a casual conversation, so I must respond directly to the user.
        Your Response: Hello! I'm your mental health assistant. I can help you find resources or book appointments. How may I help you today?

        Example 6: End Conversation
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
        
        #NEW - initialize tracking fields if not present
        if "tools_called" not in state:
            state["tools_called"] = []
        if "tools_result" not in state:
            state["tools_result"] = {}
        if "agents_used" not in state:
            state["agents_used"] = []
        
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
            # NEW - added risk_probability
            risk_prediction, risk_probability = risk_classifier(latest_message)
            print(f"--- SUPERVISOR: Risk assessment result: {risk_prediction} ---")
            print(f"--- SUPERVISOR: Risk probability: {risk_probability:.4f} ---")  #NEW - debug print
            
            if risk_prediction == 1:  # At risk
                print("--- SUPERVISOR: CRISIS DETECTED - Returning crisis response. ---")
                crisis_response = CrisisResponse(latest_message)
                crisis_response_msg = AIMessage(content=crisis_response)
                
                #NEW - track crisis response agent
                return {
                    "messages": [crisis_response_msg],
                    "risk_probability": risk_probability,
                    "conversation_ended": True,
                    "agents_used": state.get("agents_used", []) + ["crisis_response"],
                    "tools_called": state.get("tools_called", []),     #preserve
                    "tools_result": state.get("tools_result", {})      #preserve
                }
            else:
                print("--- SUPERVISOR: No risk detected, proceeding with normal flow. ---")
                #EDITED - always set risk_probability in state
                return {
                    "risk_probability": risk_probability,
                    "tools_called": state.get("tools_called", []),     #preserve
                    "tools_result": state.get("tools_result", {}),     #preserve
                    "agents_used": state.get("agents_used", [])        #preserve
                }
                
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
        """the main decision-making node for the supervisor."""
        print("--- SUPERVISOR: Routing request... ---")

        response = supervisor_runnable.invoke({"messages": state["messages"]})

        #NEW - always preserve existing tracking data
        agents_used = state.get("agents_used", [])
        if "supervisor" not in agents_used:
            agents_used.append("supervisor")

        #check if the supervisor decided to end the conversation
        if "__END__" in response.content:
            print("--- SUPERVISOR: LLM decided to end conversation. ---")
            
            final_message_content = response.content.replace("__END__", "").strip()
            final_message = AIMessage(content=final_message_content)
            
            return {
                "messages": [final_message],
                "conversation_ended": True,
                "agents_used": agents_used,
                "tools_called": state.get("tools_called", []),  #preserve
                "tools_result": state.get("tools_result", {})   #preserve
            }
        
        #case 1: supervisor decides to use its own tools
        if response.tool_calls:
            print("--- SUPERVISOR: Decided to use own tools. ---")
            print(f"--- SUPERVISOR: Response: {response} ---")
            
            tool_names = [tool_call["name"] for tool_call in response.tool_calls]
            existing_tools = state.get("tools_called", [])
            
            return {
                "messages": [response],
                "tools_called": existing_tools + tool_names,
                "agents_used": agents_used,
                "tools_result": state.get("tools_result", {})  #preserve
            }

        #case 2: supervisor decides to delegate to the booking agent
        if "delegating_to_booking_agent" in response.content:
            print("--- SUPERVISOR: Delegating to booking agent. ---")
            print(f"--- SUPERVISOR: Response: {response} ---")
            
            return {
                "messages": [AIMessage(content="delegating_to_booking_agent")],
                "agents_used": agents_used,
                "tools_called": state.get("tools_called", []),  #preserve
                "tools_result": state.get("tools_result", {})   #preserve
            }
            
        #case 3: supervisor responds directly
        print("--- SUPERVISOR: Responding directly. ---")
        print(f"--- SUPERVISOR: Response: {response} ---")
        return {
            "messages": [response],
            "agents_used": agents_used,
            "tools_called": state.get("tools_called", []),  #preserve
            "tools_result": state.get("tools_result", {})   #preserve
        }

    #EDITED - capture tool results in supervisor_tools_node
    def supervisor_tools_node(state: SupervisorState):
        """execute supervisor tools and capture results."""
        print("--- SUPERVISOR: Executing supervisor tools... ---")
        
        #initialize tracking
        tools_called = state.get("tools_called", []).copy()  #make a copy to avoid mutation
        tools_result = state.get("tools_result", {}).copy()
        
        #STEP 1: capture tool calls from the AIMessage BEFORE execution
        last_ai_message = None
        for msg in reversed(state["messages"]):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                last_ai_message = msg
                break
        
        if last_ai_message and last_ai_message.tool_calls:
            for tool_call in last_ai_message.tool_calls:
                tool_name = tool_call["name"]
                if tool_name not in tools_called:
                    tools_called.append(tool_name)
                    print(f"--- SUPERVISOR TOOLS: Registered tool call: {tool_name} ---")
        
        #STEP 2: execute the tools
        result = supervisor_tool_node.invoke(state)
        
        #STEP 3: capture tool results from ToolMessage objects
        for msg in result["messages"]:
            #check if this is a ToolMessage
            if hasattr(msg, "name") and hasattr(msg, "content"):
                tool_name = msg.name
                
                #ensure tool is in tools_called list
                if tool_name not in tools_called:
                    tools_called.append(tool_name)
                    print(f"--- SUPERVISOR TOOLS: Added tool from result: {tool_name} ---")
                
                #store tool result
                try:
                    import json
                    if isinstance(msg.content, str):
                        #try to parse as json
                        try:
                            parsed_content = json.loads(msg.content)
                            tools_result[tool_name] = parsed_content
                            print(f"--- SUPERVISOR TOOLS: Stored JSON result for {tool_name} ---")
                        except json.JSONDecodeError:
                            #if not json, store as string
                            tools_result[tool_name] = msg.content
                            print(f"--- SUPERVISOR TOOLS: Stored string result for {tool_name} ---")
                    else:
                        tools_result[tool_name] = msg.content
                        print(f"--- SUPERVISOR TOOLS: Stored non-string result for {tool_name} ---")
                except Exception as e:
                    print(f"--- SUPERVISOR TOOLS: Error storing result for {tool_name}: {e} ---")
                    tools_result[tool_name] = str(msg.content)
            else:
                #debug: log messages that don't match expected format
                print(f"--- SUPERVISOR TOOLS: Skipping message without name/content: {type(msg)} ---")
        
        print(f"--- SUPERVISOR TOOLS: Final tools_called: {tools_called} ---")
        print(f"--- SUPERVISOR TOOLS: Final tools_result keys: {list(tools_result.keys())} ---")
        
        return {
            "messages": result["messages"],
            "tools_called": tools_called,
            "tools_result": tools_result,
            "agents_used": state.get("agents_used", [])
        }

    def booking_agent_node(state: SupervisorState):
        """Invokes the booking agent sub-graph."""
        print("--- SUPERVISOR: Invoking booking agent sub-graph. ---")
        
        #NEW - track booking agent usage
        agents_used = state.get("agents_used", [])
        if "booking_agent" not in agents_used:
            agents_used.append("booking_agent")
        
        # Run the booking agent graph with the current conversation history
        booking_result = booking_graph.invoke({"messages": state["messages"]})
        
        #NEW - extract any tool calls from booking agent
        booking_tools_called = []
        booking_tools_result = {}
        
        for msg in booking_result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    booking_tools_called.append(tool_call["name"])
            if hasattr(msg, "name") and hasattr(msg, "content"):
                #tool result message
                try:
                    import json
                    if isinstance(msg.content, str):
                        booking_tools_result[msg.name] = json.loads(msg.content)
                    else:
                        booking_tools_result[msg.name] = msg.content
                except:
                    booking_tools_result[msg.name] = msg.content
        
        #merge with existing tracking
        existing_tools = state.get("tools_called", [])
        existing_results = state.get("tools_result", {})
        
        return {
            "messages": booking_result["messages"],
            "agents_used": agents_used,
            "tools_called": existing_tools + booking_tools_called,
            "tools_result": {**existing_results, **booking_tools_result},
            "conversation_ended": state.get("conversation_ended", False),  
            "risk_probability": state.get("risk_probability", 0.0)
        }

    # --- Construct the Graph ---
    builder = StateGraph(SupervisorState)

    builder.add_node("start", start_node)
    builder.add_node("risk_assessment", risk_assessment_node)
    builder.add_node("supervisor_router", supervisor_router_node)
    builder.add_node("supervisor_tools", supervisor_tools_node)  #EDITED - now uses custom node
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


