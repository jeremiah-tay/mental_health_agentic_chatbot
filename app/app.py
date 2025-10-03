import sys
import os
import streamlit as st
from typing import List, Annotated
from typing_extensions import TypedDict

# Add project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# LangChain and LangGraph imports
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Import your actual calendar tools
from tools.calendar_tools import (
    list_calendar_list,
    list_calendar_events,
    insert_calendar_event,
    create_calendar
)
from dotenv import load_dotenv
load_dotenv()

# --- LangGraph State Definition ---
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# --- System Prompt and LLM Setup ---
system_prompt = """
You are a helpful and empathetic assistant who can also manage a user's Google Calendar.

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
For all other conversations, maintain a supportive and patient tone.
"""
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Using gpt-4o-mini as gpt-5-mini is not available yet
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.2)

tools = [list_calendar_list, list_calendar_events, insert_calendar_event, create_calendar]
llm_with_tools = llm.bind_tools(tools)

# The chain combines the prompt and the LLM with tools
chain = prompt_template | llm_with_tools

# --- Node Definitions ---
def start_node(state: State) -> State:
    """Start node - initializes the conversation with a welcome message."""
    if not state.get("messages"):
        welcome_message = AIMessage(content="""Hello! I'm your personal assistant. I can help you with booking events on your Google Calendar. How can I help you today?""")
        return {"messages": [welcome_message]}
    return state # Return state unchanged if messages already exist

def tool_calling_llm_node(state: State):
    """The node that calls the LLM with the system prompt and tools."""
    response = chain.invoke({"messages": state["messages"]})
    return {"messages": [response]}

def should_continue(state: State) -> str:
    """Determine whether to continue to tools or end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls and len(last_message.tool_calls) > 0:
        return "tools"
    return END

# --- Graph Construction ---
builder = StateGraph(State)

builder.add_node("start", start_node)
builder.add_node("tool_calling_llm", tool_calling_llm_node)
builder.add_node("tools", ToolNode(tools))

builder.set_entry_point("start")
builder.add_edge("start", "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    should_continue,
    {"tools": "tools", END: END}
)
builder.add_edge("tools", "tool_calling_llm")

# Compile the graph
app = builder.compile()

# --- Streamlit UI ---
if __name__ == "__main__":
    st.title('Google Calendar AI Agent (LangGraph)')

    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Only add welcome message on first load
        initial_state = app.invoke({"messages": []})
        st.session_state.messages = initial_state.get('messages', [])

    for msg in st.session_state.messages:
        if isinstance(msg, AIMessage) and msg.content.strip():
            with st.chat_message("ai", avatar="🤖"):
                st.markdown(msg.content)
        elif isinstance(msg, HumanMessage):
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(msg.content)

    if prompt := st.chat_input("What would you like to do?"):
        # Add user message to session state and display it
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            with st.chat_message("ai", avatar="🤖"):
                try:
                    # Process through the graph WITHOUT going through start_node
                    final_state = app.invoke({"messages": st.session_state.messages})
                    final_message = final_state["messages"][-1]
                    st.session_state.messages = final_state["messages"]
                    st.markdown(final_message.content)
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(AIMessage(content=f"Sorry, I encountered an error: {str(e)}"))
        st.rerun()