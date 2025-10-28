# app/main.py
import sys
import os
import streamlit as st
import requests
from typing import List
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from config.auth import authenticate

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

API_URL = "http://127.0.0.1:8000/chat"  # FastAPI backend endpoint

st.set_page_config(page_title="Mental Health Chatbot", page_icon="💬")

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

st.title("💬 Multi-Agent Mental Health Chatbot")

# --- Initialize messages ---
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="""
        Hello! I'm your mental health assistant.  
        I can help you with:
        - Finding mental health resources  
        - Guiding coping exercises  
        - Booking therapy sessions  

        How can I support you today?
    """)]

# --- Display messages ---
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        with st.chat_message("ai", avatar="🤖"):
            st.markdown(msg.content)
    elif isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(msg.content)

# --- Handle user input ---
if prompt := st.chat_input("How can I help you?"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        try:
            response = requests.post(API_URL, json={"message": prompt})
            response.raise_for_status()
            bot_reply = response.json().get("response", "I'm sorry, something went wrong.")
        except Exception as e:
            bot_reply = f"❌ Error contacting backend: {e}"

    st.session_state.messages.append(AIMessage(content=bot_reply))
    with st.chat_message("ai", avatar="🤖"):
        st.markdown(bot_reply)

    st.rerun()
