#conversation_history/logger.py

import os
import uuid
from supabase import create_client, Client
from dotenv import load_dotenv
from typing import List, Dict, Optional

#load environment variables
load_dotenv()

#supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

#initialize supabase client for conversation logging
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("conversation logging supabase client initialized")

def generate_conversation_id():
    """generates a new unique conversation id"""
    return str(uuid.uuid4())

def log_conversation_turn(
    conversation_id: str,
    conversation_turn: int,
    human_message: str,
    ai_message: str,
    tools_called: List[str],
    tools_result: Dict,
    agents_used: List[str],
    conversation_ended: bool,
    risk_probability: float = 0.0
) -> bool:
    """
    logs a single conversation turn to the supabase conversation_log table
    
    args:
        conversation_id: unique uuid for this conversation
        conversation_turn: turn number (0 for welcome message, 1+ for user interactions)
        human_message: the user's message (empty string for turn 0)
        ai_message: the ai's response message
        tools_called: list of tool names that were called (empty list if none)
        tools_result: dict containing results from tools (empty dict if none)
        agents_used: list of agent names that were used (empty list if none)
        conversation_ended: boolean indicating if conversation ended on this turn
        risk_probability: float probability of user being at risk (0.0-1.0)
    
    returns:
        bool: true if logging successful, false otherwise
    """
    try:
        #NEW - convert risk_probability to native python float and round to 5 decimal places
        risk_probability = round(float(risk_probability), 5)
        
        #prepare the data to insert
        data = {
            "conversation_id": conversation_id,
            "conversation_turn": conversation_turn,
            "human_message": human_message,
            "ai_message": ai_message,
            "tools_called": tools_called,
            "tools_result": tools_result,
            "agents_used": agents_used,
            "conversation_ended": conversation_ended,
            "risk_probability": risk_probability
        }
        
        #insert into supabase
        result = supabase.table("conversation_log").insert(data).execute()
        
        #check if insertion was successful
        if result.data:
            print(f"successfully logged turn {conversation_turn} for conversation {conversation_id}")
            return True
        else:
            print(f"failed to log turn {conversation_turn}: no data returned")
            return False
            
    except Exception as e:
        print(f"error logging conversation turn {conversation_turn}: {e}")
        return False
