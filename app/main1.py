# app/main1.py
import sys
import os
import streamlit as st
import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage


# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.auth import authenticate

# Load env
load_dotenv()

# --- Streamlit UI ---
st.title("Multi-Agent Mental Health Chatbot")

# --- Authentication ---
if "auth_completed" not in st.session_state:
    with st.spinner("Setting up authentication..."):
        try:
            authenticate()
            st.session_state.auth_completed = True
            st.success("✅ Authentication completed successfully!")
        except Exception as e:
            st.error(f"❌ Authentication failed: {e}")
            st.stop()

# --- Chat setup ---
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(
        content="""Hello! I'm your mental health assistant.

You can ask me to:
- Find mental health resources
- Guide you through a coping exercise
- Book an appointment with a therapist

How can I help you today?
""")]

# Display chat history
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        with st.chat_message("ai", avatar="🤖"):
            st.markdown(msg.content)
    elif isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(msg.content)

# --- User Input ---
if prompt := st.chat_input("How can I help you?"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # Prepare message history for backend
    history = []
    for m in st.session_state.messages[:-1]:
        if isinstance(m, HumanMessage):
            history.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            history.append({"role": "assistant", "content": m.content})

    # Call backend FastAPI
    with st.chat_message("ai", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/chat",
                    json={"message": prompt, "history": history}
                )
                response.raise_for_status()
                reply = response.json()["response"]
                st.markdown(reply)
                st.session_state.messages.append(AIMessage(content=reply))
            except Exception as e:
                st.error(f"❌ Error contacting backend: {e}")


