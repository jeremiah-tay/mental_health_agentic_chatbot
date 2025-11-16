# app/main.py
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

# --- Custom CSS for Mental Health Theme ---
st.markdown("""
<style>
    /* Gradient background */
    .stApp {
        background: linear-gradient(135deg, #f0f9e8 0%, #e8f5e9 25%, #e3f2fd 50%, #e1f5fe 75%, #e0f7fa 100%);
        background-attachment: fixed;
    }
            
    /* Make all text black */
    .stApp, .stApp p, .stApp span, .stApp div, .stChatMessage p, .stMarkdown, .stTextInput, .stButton > button {
        color: #000000;
    }

    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Chat message styling */
    .stChatMessage {
        background-color: transparent;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        background: white;
        height: 48px;
        box-sizing: border-box;
        color: #000000 !important;
        caret-color: #000000 !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #81c784;
        box-shadow: 0 0 0 3px rgba(129, 199, 132, 0.1);
    }
    
    /* Ensure input container doesn't get cut off */
    .stTextInput {
        margin-bottom: 0;
    }
    
    /* Add padding to prevent cutoff at bottom */
    .main .block-container {
        padding-bottom: 2rem;
    }
    
    /* Align send button with input field */
    div[data-testid="column"] {
        vertical-align: bottom;
    }
    
    div[data-testid="column"]:nth-child(2) {
        display: flex;
        align-items: flex-end;
        padding-bottom: 0;
        vertical-align: bottom;
    }
    
    div[data-testid="column"]:nth-child(2) button {
        height: 48px !important;
        min-height: 48px !important;
        max-height: 48px !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding: 0 !important;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        box-sizing: border-box;
    }
    
    div[data-testid="column"]:nth-child(1) {
        display: flex;
        align-items: flex-end;
        padding-bottom: 0;
        vertical-align: bottom;
    }
    
    div[data-testid="column"]:nth-child(1) > div {
        width: 100%;
        margin-bottom: 0;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 20px;
        border: none;
        background: linear-gradient(135deg, #81c784 0%, #4caf50 100%);
        color: white;
        font-weight: 500;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    ###########################################
    
    /* Suggested prompt buttons - smaller size */
    button[key^="prompt_"] {
        font-size: 0.75rem !important;
        padding: 0.4rem 0.8rem !important;
        min-height: auto !important;
        color: white !important;
        background: linear-gradient(135deg, #81c784 0%, #4caf50 100%) !important;
        border-radius: 20px;
        border: none;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    /* Hover effect for suggested prompt buttons */
    button[key^="prompt_"]:hover {
        color: red !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
    }
    
    ##################################
    
    /* Title styling */
    .main-title {
        text-align: center;
        color: #2e7d32;
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #66bb6a;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Streamlit UI ---
# Icon and title (without white box) - heart beside title
st.markdown("""
<div style="text-align: center; margin-bottom: 1.5rem;">
    <h1 class="main-title">💚 Mental Health Assistant</h1>
    <p class="subtitle">Your supportive companion for mental wellness</p>
</div>
""", unsafe_allow_html=True)

# --- Authentication ---
if "auth_completed" not in st.session_state:
    with st.spinner("Setting up authentication..."):
        try:
            authenticate()
            st.session_state.auth_completed = True
            # Removed success message as requested
        except Exception as e:
            st.error(f"❌ Authentication failed: {e}")
            st.stop()

# --- NEW: Add "New Chat" button below subtitle ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔄 New Chat", use_container_width=True):
        # Clear all session state EXCEPT authentication
        keys_to_keep = ["auth_completed"]
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        st.rerun()

# --- Initialize conversation_id first ---
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

# --- Suggested Prompts (only show if no conversation yet or first message) - BELOW NEW CHAT BUTTON ---
if len(st.session_state.messages) <= 1:
    st.markdown("---")
    st.markdown("##### 💡 Suggested Questions")
    
    # Define suggested prompts - only 3 questions
    suggested_prompts = [
        ("What is Cognitive Behavioural Therapy?", "🧠"),
        ("Tell me about Mental Health", "📚"),
        ("Book an appointment with a therapist", "📅")
    ]
    
    # Display prompts in a grid - 3 columns for 3 buttons
    cols = st.columns(3)
    for idx, (prompt, icon) in enumerate(suggested_prompts):
        with cols[idx]:
            if st.button(f"{icon} {prompt}", key=f"prompt_{idx}", use_container_width=True):
                # Store prompt to be processed
                st.session_state.pending_prompt = prompt
                st.rerun()
    
    st.markdown("---")

# --- NEW: Initialize conversation state ---
if "conversation_ended" not in st.session_state:
    st.session_state.conversation_ended = False

if "conversation_turn" not in st.session_state:
    st.session_state.conversation_turn = 0

# --- Display chat history (with message filtering) ---
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        # NEW - Filter out internal routing messages
        if isinstance(msg, AIMessage) and msg.content and "delegating_to_booking_agent" not in msg.content:
            with st.chat_message("ai", avatar="💚"):
                st.markdown(msg.content)
        elif isinstance(msg, HumanMessage):
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg.content)

# --- Check for pending prompt first (before processing) ---
prompt = None
if not st.session_state.get("conversation_ended", False):
    if "pending_prompt" in st.session_state:
        prompt = st.session_state.pending_prompt
        del st.session_state.pending_prompt

# --- NEW: Show conversation ended message with button ---
if st.session_state.conversation_ended:
    st.info("💬 This conversation has ended. Please click 'New Chat' to start a new conversation.")
    
    # Add spacing
    st.write("")
    
    # Second button when conversation ends
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 New Chat", key="new_chat_ended"):
            # Clear all session state EXCEPT authentication
            keys_to_keep = ["auth_completed"]
            for key in list(st.session_state.keys()):
                if key not in keys_to_keep:
                    del st.session_state[key]
            st.rerun()
    
    st.stop()  # Prevent chat input from showing

if prompt:
    # Input field is automatically cleared by form's clear_on_submit=True
    # NEW - Increment turn counter
    st.session_state.conversation_turn += 1
    current_turn = st.session_state.conversation_turn
    
    # Add user message to chat history
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user", avatar="👤"):
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
    
    # Display AI response with spinner (this will appear below user message)
    with st.chat_message("ai", avatar="💚"):
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
                # Add AI response to chat history
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
    
    # After processing, rerun to show input bar below the new messages
    st.rerun()

# --- User Input (positioned directly below all messages - always at bottom) ---
# Only show input if conversation hasn't ended (shows even during loading)
if not st.session_state.get("conversation_ended", False):
    # Custom input positioned directly below messages
    st.markdown("---")
    # Use a form to allow Enter key submission
    with st.form(key="chat_form", clear_on_submit=True):
        # Create a nicer input layout with proper alignment
        input_col1, input_col2 = st.columns([9, 1], gap="small")
        with input_col1:
            user_input = st.text_input(
                "Ask a question...",
                key="user_input",
                label_visibility="collapsed",
                placeholder="Ask a question..."
            )
        with input_col2:
            # Align send button with input field - same row alignment
            # Note: form_submit_button doesn't accept 'key' parameter
            send_button = st.form_submit_button("➤", use_container_width=True)
        
        # Process input if form was submitted (Enter key or button click) and has input
        if send_button and user_input:
            st.session_state.pending_prompt = user_input
            st.rerun()

# Legal disclaimer at bottom
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.85rem; margin-top: 2rem;">
    <p>📄 <strong>Legal Disclaimer:</strong> This chatbot is for informational and supportive purposes only. 
    It is not a substitute for professional medical advice, diagnosis, or treatment. 
    If you are experiencing a mental health crisis, please contact emergency services immediately.</p>
</div>
""", unsafe_allow_html=True)

