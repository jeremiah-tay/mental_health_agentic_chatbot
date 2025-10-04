import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build


# All configuration is now in this file
API_NAME = 'calendar'
API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/calendar']
TOKEN_FILE = 'token.json'
CREDENTIALS_FILE = 'credential.json' # Make sure this file exists

def create_service():
    """
    Creates a Google Calendar service client by handling all authentication.
    This is the single source of truth for authentication.
    """
    creds = None

    # 1. The token file stores the user's access and refresh tokens.
    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except Exception as e:
            print(f"Error loading token file: {e}. Will re-authenticate.")
            creds = None # Force re-authentication

    # 2. If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing expired token...")
            creds.refresh(Request())
        else:
            print("No valid token found, starting new authentication flow...")
            # Use the reliable InstalledAppFlow for local development
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            # This will automatically open a browser for the user to log in
            creds = flow.run_local_server(port=0)
        
        # 3. Save the credentials for the next run
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
        print(f"Credentials saved to {TOKEN_FILE}")

    try:
        service = build(API_NAME, API_VERSION, credentials=creds)
        print(f"{API_NAME} service created successfully")
        return service
    except Exception as e:
        print(f"Failed to create service: {e}")
        # If service creation fails, it's often due to a bad token.
        # Deleting the token forces re-authentication on the next run.
        if os.path.exists(TOKEN_FILE):
            os.remove(TOKEN_FILE)
            print(f"Deleted invalid {TOKEN_FILE}. Please restart the application.")
        return None