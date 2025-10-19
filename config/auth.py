import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# --- CONFIGURATION ---
SCOPES = ['https://www.googleapis.com/auth/calendar']
CREDENTIALS_FILE = 'credentials/credential.json'
TOKEN_FILE = 'credentials/token.json'
# ---------------------

# Ensure credentials directory exists
os.makedirs('credentials', exist_ok=True)

def authenticate():
    """Handles Google authentication and creates token.json."""
    creds = None
    if os.path.exists(TOKEN_FILE):
        print(f"'{TOKEN_FILE}' already exists. Authentication seems to be complete.")
        # Optional: Verify it works
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
            service = build('calendar', 'v3', credentials=creds)
            print("Successfully created Google Calendar service. Token is valid!")
            return
        except Exception as e:
            print(f"Token file is invalid: {e}. Deleting it to re-authenticate.")
            os.remove(TOKEN_FILE)
            creds = None # Reset creds

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing expired credentials...")
            creds.refresh(Request())
        else:
            print(f"'{TOKEN_FILE}' not found or invalid. Starting new login flow...")
            # Use InstalledAppFlow for a clear console-based login process
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
        print(f"Authentication successful! Credentials saved to '{TOKEN_FILE}'.")

if __name__ == '__main__':
    # First, ensure the old, possibly broken token is gone.
    if os.path.exists(TOKEN_FILE):
        print(f"Deleting existing '{TOKEN_FILE}' to ensure a clean login.")
        os.remove(TOKEN_FILE)
        
    authenticate()
    print("\nYou can now run your main Streamlit application.")
