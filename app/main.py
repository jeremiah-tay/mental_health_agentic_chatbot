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

# Import the authentication function
from config.auth import authenticate

# Load environment variables from a .env file
load_dotenv()


# --- 1. Initialize the Language Model ---
# This single LLM instance will be passed to both the supervisor and booking agents.
llm = ChatOpenAI(model="gpt-5-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.2)

# --- 2. Create the Main Application Graph ---
# The supervisor graph is our main app. It contains all the routing logic,
# including when to call the booking agent sub-graph.
app = create_supervisor_graph(llm)

# --- 3. Streamlit User Interface ---
if __name__ == "__main__":
    st.title('Multi-Agent Mental Health Chatbot')

    # Initialize authentication state - only run once per session
    if "auth_completed" not in st.session_state:
        with st.spinner("Setting up authentication..."):
            try:
                authenticate()
                st.session_state.auth_completed = True
                st.success("✅ Authentication completed successfully!")
            except Exception as e:
                st.error(f"❌ Authentication failed: {e}")
                st.stop()

    # Add a new chat button to the top right of the page
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 New Chat"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    with col2:
        st.write("")

    # Initialize session state for messages if it doesn't exist
    if "messages" not in st.session_state:
        # Just add the welcome message directly, don't invoke the graph
        st.session_state.messages = [AIMessage(
            content="""
            Hello! I'm your mental health assistant. I am here to support you.

            You can ask me to:
            - Find resources or answer questions on mental health
            - Guide you through a coping exercise
            - Book an appointment with a therapist

            How can I help you today?
            """)]
    
    # Initialize conversation_ended state
    if "conversation_ended" not in st.session_state:
        st.session_state.conversation_ended = False

    # Display all chat messages from the history
    for msg in st.session_state.messages:
        # Ensure the message has content before displaying
        if isinstance(msg, AIMessage) and msg.content and "delegating_to_booking_agent" not in msg.content:
            with st.chat_message("ai", avatar="🤖"):
                st.markdown(msg.content)
        elif isinstance(msg, HumanMessage):
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(msg.content)
    
    # Show conversation ended message if applicable
    if st.session_state.conversation_ended:
        st.info("💬 This conversation has ended. Please refresh the page to start a new conversation.")
        st.button("🔄 Start New Conversation", on_click=lambda: st.rerun())
        st.stop()

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