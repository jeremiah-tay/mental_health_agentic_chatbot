import os
import json
import sys
from datetime import datetime, timedelta
from typing import Optional, List

# Google Calendar specific imports
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Pydantic for data validation and LangChain tool definition
from langchain_core.tools import tool
from pydantic.v1 import BaseModel, Field

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from auth import authenticate

# --- Google Calendar Authentication & Service ---
SCOPES = ["https://www.googleapis.com/auth/calendar"]
CREDENTIALS_FILE = 'credentials/credential.json'
TOKEN_FILE = 'credentials/token.json'    

def get_calendar_service():
    """Authenticates with Google Calendar API and returns a service object."""
    try:
        # Check if token.json exists, if not, call auth.py to generate it
        if not os.path.exists(TOKEN_FILE):
            print(f"Token file '{TOKEN_FILE}' not found. Running authentication...")
            authenticate()  # This will generate the token.json file
        
        # Load credentials
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        
        # Check if credentials are valid
        if not creds.valid:
            if creds.expired and creds.refresh_token:
                try:
                    print("Refreshing expired credentials...")
                    creds.refresh(Request())
                except Exception as e:
                    print(f"Token refresh failed: {e}")
                    print("Re-authenticating...")
                    if os.path.exists(TOKEN_FILE):
                        os.remove(TOKEN_FILE)
                    authenticate()  # Re-generate token
                    creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
            else:
                print("No valid credentials found. Re-authenticating...")
                if os.path.exists(TOKEN_FILE):
                    os.remove(TOKEN_FILE)
                authenticate()
                creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        
        # Save refreshed credentials if needed
        if creds.valid:
            with open(TOKEN_FILE, "w") as token:
                token.write(creds.to_json())
        
        return build("calendar", "v3", credentials=creds)
    except Exception as e:
        raise Exception(f"Failed to authenticate with Google Calendar: {e}")

class EventDateTime(BaseModel):
    dateTime: str = Field(..., description="The start or end date-time for the event in ISO 8601 format, e.g., '2025-10-10T14:00:00'")
    timeZone: str = Field("Asia/Singapore", description="The time zone, e.g., 'Asia/Singapore'")

class ListEventsArgs(BaseModel):
    start_time: str = Field(..., description="The start of the time window to check for events, in ISO 8601 format.")
    end_time: str = Field(..., description="The end of the time window to check for events, in ISO 8601 format.")
    calendar_id: str = Field("primary", description="The ID of the calendar to check.")

class InsertEventArgs(BaseModel):
    summary: str = Field(..., description="The title or summary of the event (e.g., 'First Therapy Session', 'Anxiety Counseling'). Ask the user what they'd like to call their session.")
    start_datetime: str = Field(..., description="Start time in ISO 8601 format for Singapore Time, e.g., '2024-10-14T16:00:00' (this will be interpreted as 4:00 PM Singapore Time). Must be weekdays only, 9 AM - 6 PM. Convert all relative dates (tomorrow, next Tuesday) to exact dates.")
    end_datetime: str = Field(..., description="End time in ISO 8601 format for Singapore Time, e.g., '2024-10-14T17:00:00' (this will be interpreted as 5:00 PM Singapore Time). Must be weekdays only, 9 AM - 6 PM. Convert all relative dates to exact dates.")
    timezone: str = Field("Asia/Singapore", description="Timezone for the event - always use 'Asia/Singapore' for Singapore Time")
    calendar_id: str = Field("primary", description="The ID of the calendar to use.")
    description: Optional[str] = Field(None, description="A description for the event. For a therapy session, this could include notes like 'First session'.")
    
# (You can keep Create, Update, and Delete schemas if you want the bot to have those abilities)

# --- LangChain Tools ---

@tool
def list_calendars() -> str:
    """Lists all the calendars in the user's account to find the right one to use."""
    try:
        service = get_calendar_service()
        calendar_list = service.calendarList().list().execute().get('items', [])
        if not calendar_list:
            return json.dumps({"status": "success", "message": "No calendars found.", "calendars": []})
        calendars = [{"id": cal['id'], "summary": cal['summary']} for cal in calendar_list]
        return json.dumps({"status": "success", "calendars": calendars})
    except Exception as e:
        return json.dumps({"status": "error", "message": f"An error occurred: {e}"})

@tool("list_events", args_schema=ListEventsArgs)
def list_events(start_time: str, end_time: str, calendar_id: str = "primary") -> str:
    """
    Lists events within a specified time range to check for availability.
    
    Args:
        start_time: Start time in ISO 8601 format (e.g., '2025-10-14T16:00:00')
        end_time: End time in ISO 8601 format (e.g., '2025-10-14T17:00:00')
        calendar_id: Calendar ID to check (defaults to 'primary')
    
    Returns:
        JSON string with status and either events list or availability message
    """
    try:
        service = get_calendar_service()
        events_result = service.events().list(
            calendarId=calendar_id,
            timeMin=start_time,
            timeMax=end_time,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        events = events_result.get('items', [])
        if not events:
            return json.dumps({
                "status": "success", 
                "message": "No upcoming events found in this time range. The slot is available."
            })
        return json.dumps({
            "status": "success",
            "events": [{"summary": event['summary'], "start": event['start']['dateTime']} for event in events]
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"An error occurred: {e}"
        })

@tool("insert_event", args_schema=InsertEventArgs)
def insert_event(
    summary: str,
    start_datetime: str,
    end_datetime: str,
    timezone: str = "Asia/Singapore",
    calendar_id: str = "primary",
    description: Optional[str] = None,
) -> str:
    """
    Inserts a new event into a specified Google Calendar after checking for conflicts.
    All times are interpreted as Singapore Time (SGT).
    Enforces office hours: Monday-Friday, 9 AM - 6 PM.
    """
    try:
        # Validate timezone
        if timezone != "Asia/Singapore":
            return json.dumps({
                "status": "error",
                "message": f"Only Singapore Time (Asia/Singapore) is supported. Got: {timezone}"
            })
        
        # Parse the datetime to check office hours
        from datetime import datetime
        try:
            start_dt = datetime.fromisoformat(start_datetime)
        except ValueError:
            return json.dumps({
                "status": "error",
                "message": f"Invalid datetime format: {start_datetime}. Please use format: YYYY-MM-DDTHH:MM:SS"
            })
        
        # Check if it's a weekday (Monday = 0, Sunday = 6)
        if start_dt.weekday() > 4:  # Saturday = 5, Sunday = 6
            return json.dumps({
                "status": "error",
                "message": "Booking failed: Appointments can only be scheduled on weekdays (Monday to Friday). Please choose a weekday."
            })
        
        # Check if it's within office hours (9 AM to 6 PM)
        if not (9 <= start_dt.hour < 18):  # 9 AM to 6 PM (18:00)
            return json.dumps({
                "status": "error",
                "message": "Booking failed: Appointments can only be scheduled during office hours (9:00 AM to 6:00 PM Singapore Time). Please choose a time within these hours."
            })
        
        # Debug: Log what we received
        print(f"DEBUG: Received parameters:")
        print(f"  summary: {summary}")
        print(f"  start_datetime: {start_datetime} (Singapore Time)")
        print(f"  end_datetime: {end_datetime} (Singapore Time)")
        print(f"  timezone: {timezone}")
        print(f"  calendar_id: {calendar_id}")
        print(f"  day_of_week: {start_dt.strftime('%A')}")
        print(f"  hour: {start_dt.hour}")
        
        service = get_calendar_service()
        print(f"DEBUG: Service created successfully")

        # Convert strings to the required dict format with Singapore timezone
        start = {"dateTime": start_datetime, "timeZone": "Asia/Singapore"}
        end = {"dateTime": end_datetime, "timeZone": "Asia/Singapore"}
        
        print(f"DEBUG: Start dict: {start}")
        print(f"DEBUG: End dict: {end}")

        # 1. CONFLICT CHECKING: Verify the slot isn't already taken
        print(f"DEBUG: Checking for conflicts...")
        
        # Add Singapore timezone to datetime strings for Google Calendar API
        time_min_with_tz = start_datetime + '+08:00'  # Singapore is UTC+8
        time_max_with_tz = end_datetime + '+08:00'
        
        print(f"DEBUG: API timeMin: {time_min_with_tz}")
        print(f"DEBUG: API timeMax: {time_max_with_tz}")
        
        events_result = service.events().list(
            calendarId=calendar_id,
            timeMin=time_min_with_tz,
            timeMax=time_max_with_tz,
            singleEvents=True,
        ).execute()
        
        print(f"DEBUG: Found {len(events_result.get('items', []))} existing events")
        
        if events_result.get("items", []):
            return json.dumps({
                "status": "error",
                "message": "Booking failed: The requested time slot is already booked. Please ask the user to choose another time."
            })

        # 2. INSERT EVENT
        print(f"DEBUG: No conflicts found, creating event...")
        event_body = {
            "summary": summary,
            "description": description,
            "start": start,
            "end": end,
        }
        
        print(f"DEBUG: Event body: {event_body}")
        
        created_event = service.events().insert(calendarId=calendar_id, body=event_body).execute()
        
        print(f"DEBUG: Event created successfully: {created_event.get('id')}")
        
        return json.dumps({
            "status": "success",
            "message": f"Event '{summary}' created successfully for {start_dt.strftime('%A, %B %d at %I:%M %p')} Singapore Time.",
            "event_id": created_event['id'],
            "link": created_event.get('htmlLink')
        })
    except Exception as e:
        print(f"DEBUG: Exception occurred: {str(e)}")
        print(f"DEBUG: Exception type: {type(e)}")
        return json.dumps({
            "status": "error",
            "message": f"An error occurred: {e}"
        })
@tool
def test_calendar_connection() -> str:
    """Test function to verify Google Calendar connection"""
    try:
        service = get_calendar_service()
        # Try to list calendars
        calendar_list = service.calendarList().list().execute()
        return json.dumps({
            "status": "success",
            "message": f"Connection successful. Found {len(calendar_list.get('items', []))} calendars."
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Connection failed: {e}"
        })