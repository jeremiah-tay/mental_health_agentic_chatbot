import json
from google_apis import create_service
from langchain_core.tools import tool
from typing import Optional, List, Dict

client_secret = 'credential.json'

def construct_google_calendar_client(client_secret):
    """
    Constructs a Google Calendar API client.

    Parameters:
    - client_secret (str): The path to the client secret JSON file.

    Returns:
    - service: The Google Calendar API service instance.
    """
    API_NAME = 'calendar'
    API_VERSION = 'v3'
    SCOPES = ['https://www.googleapis.com/auth/calendar']
    service = create_service()
    return service

calendar_service = construct_google_calendar_client(client_secret)

@tool
def create_calendar(calendar_name: str) -> str:
    """
    Create a new calendar list

    Parameters:
    - calendar_name (str): The name of the new calendar list.

    Returns:
    - str: JSON string with the result of calendar creation.
    """
    try:
        calendar_list = {
            'summary': calendar_name
        }
        created_calendar_list = calendar_service.calendars().insert(body=calendar_list).execute()
        
        result = {
            'status': 'success',
            'calendar_id': created_calendar_list.get('id'),
            'calendar_name': created_calendar_list.get('summary'),
            'message': f'Calendar "{calendar_name}" created successfully!'
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_result = {
            'status': 'error',
            'message': f'Failed to create calendar: {str(e)}'
        }
        return json.dumps(error_result, indent=2)

@tool
def list_calendar_list(max_capacity = 200):
    """
    Lists calendar lists until the total number of items reaches max_capacity

    Parameters:
    - max capacity (int or str, optional): The maximum number of calendar lists to retrieve. Defaults to 200.
      If a string is provided, it will be converted to an integer.

    Returns:
    - list: A list of dictionaries containing cleaned calendar list information with 'id', 'name', and 'description'.
    """
    if isinstance(max_capacity, str):
        max_capacity = int(max_capacity)
    
    all_calendars = []
    all_calendars_cleaned = []
    next_page_token = None
    capacity_tracker = 0

    while True:
        calendar_list = calendar_service.calendarList().list(
            maxResults = min(200, max_capacity - capacity_tracker),
            pageToken = next_page_token
        ).execute()
        calendars = calendar_list.get('items', [])
        all_calendars.extend(calendars)
        capacity_tracker += len(calendars)
        if capacity_tracker >= max_capacity:
            break
        next_page_token = calendar_list.get('nextPageToken')
        if not next_page_token:
            break

    for calendar in all_calendars:
        all_calendars_cleaned.append(
            {
                'id': calendar['id'],
                'name': calendar['summary'],
                'description': calendar.get('description', [])
            })

    return all_calendars_cleaned

@tool
def list_calendar_events(calendar_id, max_capacity = 20):
    """
    Lists events from a specified calendar until the total number of events reaches max_capacity.

    Parameters:
    - calendar_id (str): The ID of the calendar from which to list events.
    - max_capacity ( int or str, optional): THe maximum number of events to retrieve. Defaults to 20.
      If a string is provided, it will be converted to an integer.
    
    Returns:
    - list: A list of events from the specified calendar.
    """
    if isinstance(max_capacity, str):
        max_capacity = int(max_capacity)
    
    all_events = []
    next_page_token = None
    capacity_tracker = 0
    while True:
        events_list = calendar_service.events().list(
            calendarId = calendar_id,
            maxResults = min(250, max_capacity - capacity_tracker),
            pageToken = next_page_token
        ).execute()
        events = events_list.get('items', [])
        all_events.extend(events)
        if capacity_tracker >= max_capacity:
            break
        next_page_token = events_list.get('nextPageToken')
        if not next_page_token:
            break

    return all_events

@tool
def insert_calendar_event(
    summary: str,
    start: Dict,
    end: Dict,
    calendar_id: str = "primary",
    description: Optional[str] = None,
    location: Optional[str] = None,
    attendees: Optional[List[Dict]] = None
) -> str:
    """
    Inserts an event into a Google Calendar.

    Args:
        summary (str): The title or summary of the event.
        start (Dict): The start time of the event, e.g., {'dateTime': '2025-10-05T14:00:00', 'timeZone': 'Asia/Singapore'}.
        end (Dict): The end time of the event, e.g., {'dateTime': '2025-10-05T15:00:00', 'timeZone': 'Asia/Singapore'}.
        calendar_id (str, optional): The ID of the calendar. Defaults to "primary".
        description (Optional[str], optional): A description for the event. Defaults to None.
        location (Optional[str], optional): The location of the event. Defaults to None.
        attendees (Optional[List[Dict]], optional): A list of attendees, e.g., [{'email': 'user@example.com'}]. Defaults to None.

    Returns:
        str: A JSON string with the confirmation details of the created event.
    """
    try:
        # FIX: Build the event body directly from the function arguments.
        # This is the correct way to handle input from the LLM.
        event_body = {
            'summary': summary,
            'start': start,
            'end': end,
        }
        if description:
            event_body['description'] = description
        if location:
            event_body['location'] = location
        if attendees:
            event_body['attendees'] = attendees

        created_event = calendar_service.events().insert(
            calendarId=calendar_id,
            body=event_body
        ).execute()

        result = {
            'status': 'success',
            'htmlLink': created_event.get('htmlLink'),
            'summary': created_event.get('summary'),
            'message': 'Event created successfully!'
        }
        return json.dumps(result, indent=2)

    except Exception as e:
        error_result = {
            'status': 'error',
            'message': f"Failed to create event: {str(e)}"
        }
        return json.dumps(error_result, indent=2)