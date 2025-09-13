from supabase import create_client
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

def get_supabase():
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
