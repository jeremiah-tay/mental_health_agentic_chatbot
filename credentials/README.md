# Credentials Directory

This directory contains Google OAuth credentials required for calendar integration.

## Setup Instructions

1. **Place `credential.json` in this folder**

2. **Run the authentication script**
   - Execute: `python config/auth.py`
   - This will open a browser window for Google OAuth authentication
   - After successful authentication, `token.json` will be automatically created in this folder

## Files in this directory

- `credential.json` - Your Google OAuth 2.0 client credentials (you must provide this)
- `token.json` - Generated authentication token (created automatically after running `python config/auth.py`)

## Note
- The `token.json` file is generated automatically and should not be manually edited
- If authentication fails, delete `token.json` and run `python config/auth.py` again
- Keep these files secure and do not commit them to version control