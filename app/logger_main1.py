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

# NEW - import conversation logging functions
from conversation_history.logger import generate_conversation_id, log_conversation_turn

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

# --- NEW: Add "New Chat" button at the TOP (always visible) ---
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("🔄 New Chat"):
        # Clear all session state EXCEPT authentication
        keys_to_keep = ["auth_completed"]
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        st.rerun()
with col2:
    st.write("")

# --- NEW: Initialize conversation_id ---
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = generate_conversation_id()
    print(f"New conversation started with id: {st.session_state.conversation_id}")

# --- Chat setup ---
if "messages" not in st.session_state:
    welcome_message = """Hello! I'm your mental health assistant.

You can ask me to:
- Find mental health resources
- Guide you through a coping exercise
- Book an appointment with a therapist

How can I help you today?"""
    
    st.session_state.messages = [AIMessage(content=welcome_message)]
    
    # NEW - Log turn 0 (welcome message)
    log_conversation_turn(
        conversation_id=st.session_state.conversation_id,
        conversation_turn=0,
        human_message="",
        ai_message=welcome_message,
        tools_called=[],
        tools_result={},
        agents_used=[],
        conversation_ended=False,
        risk_probability=0.0
    )
    print(f"Logged turn 0 (welcome message) for conversation {st.session_state.conversation_id}")

# --- NEW: Initialize conversation state ---
if "conversation_ended" not in st.session_state:
    st.session_state.conversation_ended = False

if "conversation_turn" not in st.session_state:
    st.session_state.conversation_turn = 0

# --- Display chat history (with message filtering) ---
for msg in st.session_state.messages:
    # NEW - Filter out internal routing messages
    if isinstance(msg, AIMessage) and msg.content and "delegating_to_booking_agent" not in msg.content:
        with st.chat_message("ai", avatar="🤖"):
            st.markdown(msg.content)
    elif isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(msg.content)

# --- NEW: Show conversation ended message with button ---
if st.session_state.conversation_ended:
    st.info("💬 This conversation has ended. Please click 'New Chat' to start a new conversation.")
    
    # Add spacing
    st.write("")
    
    # Second button when conversation ends
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 Start New Conversation", key="new_chat_ended"):
            # Clear all session state EXCEPT authentication
            keys_to_keep = ["auth_completed"]
            for key in list(st.session_state.keys()):
                if key not in keys_to_keep:
                    del st.session_state[key]
            st.rerun()
    
    st.stop()  # Prevent chat input from showing

# --- User Input ---
if prompt := st.chat_input("How can I help you?"):
    # NEW - Increment turn counter
    st.session_state.conversation_turn += 1
    current_turn = st.session_state.conversation_turn
    
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    
    # Prepare message history for backend
    history = []
    for m in st.session_state.messages[1:-1]:  # Skip welcome message and current message
        if isinstance(m, HumanMessage):
            history.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            # NEW - Filter out internal messages from history sent to backend
            if "delegating_to_booking_agent" not in m.content:
                history.append({"role": "assistant", "content": m.content})
    
    # Call backend FastAPI
    with st.chat_message("ai", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/chat",
                    json={
                        "message": prompt,
                        "history": history,
                        "conversation_id": st.session_state.conversation_id  # NEW - Send conversation ID
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                # Extract all response data with None handling
                reply = data["response"]
                conversation_ended = data.get("conversation_ended") or False
                risk_probability = data.get("risk_probability") or 0.0
                tools_called = data.get("tools_called") or []
                tools_result = data.get("tools_result") or {}
                agents_used = data.get("agents_used") or []
                
                # NEW - Show risk warning if elevated
                if risk_probability > 0.5:
                    st.warning(f"⚠️ Elevated concern detected (confidence: {risk_probability:.1%})")
                
                # Display response
                st.markdown(reply)
                st.session_state.messages.append(AIMessage(content=reply))
                
                # NEW - Log this turn to database
                log_conversation_turn(
                    conversation_id=st.session_state.conversation_id,
                    conversation_turn=current_turn,
                    human_message=prompt,
                    ai_message=reply,
                    tools_called=tools_called,
                    tools_result=tools_result,
                    agents_used=agents_used,
                    conversation_ended=conversation_ended,
                    risk_probability=risk_probability
                )
                print(f"Logged turn {current_turn} for conversation {st.session_state.conversation_id}")
                
                # NEW - Update conversation ended state
                if conversation_ended:
                    st.session_state.conversation_ended = True
                
            except requests.exceptions.RequestException as e:
                error_msg = f"❌ Error contacting backend: {e}"
                st.error(error_msg)
                st.session_state.messages.append(AIMessage(content=error_msg))
                
                # NEW - Log error turn
                log_conversation_turn(
                    conversation_id=st.session_state.conversation_id,
                    conversation_turn=current_turn,
                    human_message=prompt,
                    ai_message=error_msg,
                    tools_called=[],
                    tools_result={"error": str(e)},
                    agents_used=[],
                    conversation_ended=False,
                    risk_probability=0.0
                )
            except Exception as e:
                error_msg = f"❌ Unexpected error: {e}"
                st.error(error_msg)
                st.session_state.messages.append(AIMessage(content=error_msg))
                
                # NEW - Log error turn
                log_conversation_turn(
                    conversation_id=st.session_state.conversation_id,
                    conversation_turn=current_turn,
                    human_message=prompt,
                    ai_message=error_msg,
                    tools_called=[],
                    tools_result={"error": str(e)},
                    agents_used=[],
                    conversation_ended=False,
                    risk_probability=0.0
                )
    
    # Rerun to update UI
    st.rerun()