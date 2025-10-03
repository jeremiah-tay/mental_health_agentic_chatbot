import sys
import os
import streamlit as st
from typing import List, Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

# Add project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# LangChain and LangGraph imports
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# Import the supervisor graph, which is now the main entry point to our application
from agents.supervisor import create_supervisor_graph

# Load environment variables from a .env file
load_dotenv()

# --- 1. Initialize the Language Model ---
# This single LLM instance will be passed to both the supervisor and booking agents.
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.2)

# --- 2. Create the Main Application Graph ---
# The supervisor graph is our main app. It contains all the routing logic,
# including when to call the booking agent sub-graph.
app = create_supervisor_graph(llm)

# --- 3. Streamlit User Interface ---
if __name__ == "__main__":
    st.title('Multi-Agent Mental Health Assistant')

    # Initialize session state for messages if it doesn't exist
    if "messages" not in st.session_state:
        # Start the conversation by invoking the graph's start node
        # We pass an empty state to trigger the welcome message.
        initial_state = app.invoke({})
        st.session_state.messages = initial_state.get('messages', [])
    
    # Display all chat messages from the history
    for msg in st.session_state.messages:
        # Ensure the message has content before displaying
        if isinstance(msg, AIMessage) and msg.content and "delegating_to_booking_agent" not in msg.content:
            with st.chat_message("ai", avatar="🤖"):
                st.markdown(msg.content)
        elif isinstance(msg, HumanMessage):
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(msg.content)

    # Accept user input from the chat interface
    if prompt := st.chat_input("How can I help you?"):
        # Add the user's message to the session state and display it
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        # Show a spinner while the agent is processing the request
        with st.spinner("Thinking..."):
            with st.chat_message("ai", avatar="🤖"):
                try:
                    # Invoke the main supervisor graph with the full conversation history
                    final_state = app.invoke({"messages": st.session_state.messages})
                    
                    # Update the session state with the final list of messages from the graph run
                    st.session_state.messages = final_state["messages"]
                    
                    # Display the last message from the AI, checking if it has content
                    final_message = final_state["messages"][-1]
                    if final_message.content:
                         st.markdown(final_message.content)

                except Exception as e:
                    # Handle any errors during the graph execution
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(AIMessage(content=error_msg))
        
        # Rerun the Streamlit script to update the UI immediately
        st.rerun()