# Mental Health Agentic Chatbot

An intelligent mental health assistant chatbot built with LangGraph, Streamlit, and FastAPI. The chatbot provides empathetic support, CBT (Cognitive Behavioral Therapy) techniques, appointment booking, and risk assessment capabilities.

## Features

- **Conversational AI**: Empathetic mental health support using GPT models
- **CBT Technique Selection**: Intelligent selection of appropriate therapeutic techniques
- **RAG (Retrieval-Augmented Generation)**: Access to mental health knowledge base
- **Appointment Booking**: Google Calendar integration for scheduling therapy sessions
- **Risk Assessment**: Automated detection of crisis situations with appropriate responses
- **Analytics Dashboard**: Comprehensive conversation analytics and insights
- **Conversation Logging**: Persistent storage of all interactions in Supabase

## Project Structure

```
mental_health_agentic_chatbot/
├── app/                    # Streamlit frontend application
│   └── main.py            # Main Streamlit UI
├── backend/               # FastAPI backend server
│   ├── api_server.py      # API endpoints
│   └── utils/             # Backend utilities
├── agents/                # LangGraph agents
│   ├── supervisor.py      # Main supervisor agent
│   └── booking.py         # Booking agent
├── tools/                 # LangChain tools
│   ├── cbt_tools.py       # CBT technique selection
│   ├── rag_tools.py       # RAG retrieval
│   └── calendar_tools.py  # Google Calendar integration
├── config/                # Configuration files
│   └── auth.py            # Google Calendar authentication
├── conversation_history/   # Conversation logging and analytics
│   ├── logger.py          # Conversation logging
│   └── analytics.py       # Analytics dashboard
├── riskclassifier_v2/     # Risk assessment models (archived)
├── credentials/           # Google Calendar credentials
├── pdf/                   # PDF documents for RAG
└── outputs/               # Generated outputs and logs
```

## Prerequisites

- Python 3.9 or higher
- Google Cloud Project with Calendar API enabled
- Supabase account with PostgreSQL and PGVector extension
- OpenAI API key

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd mental_health_agentic_chatbot
```

### 2. Create Virtual Environment

```bash
python -m venv chatbot_proj

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

#### Step 3.1: Install Main Dependencies

```bash
pip install -r requirements.txt
```

#### Step 3.2: Download Risk Classifier Model

```bash
python riskclassifier_v2/download_models.py
```

**Note**: If you encounter issues with PyPDF2, you may need to install `pypdf` instead:
```bash
pip install pypdf
```

### 4. Environment Variables Setup

In the `.env` file in the project root directory:

```bash
DATABASE_URL = ""
SUPABASE_URL = ""
SUPABASE_SERVICE_ROLE_KEY = ""
OPENAI_API_KEY = ""
ALLOWED_ORIGINS=
LLM_MODEL=
OPENAI_EMBEDDING_MODEL=
```

Add the following environment variables to `.env`:
The rest of the keys, please refer to the appendix section of our group report to obtain them
```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
```

**How to obtain credentials:**

- **OpenAI API Key**: 
  1. Go to https://platform.openai.com/api-keys
  2. Create a new API key
  3. Copy and paste into `.env`

- **Supabase Credentials**:
  1. Create a project at https://supabase.com
  2. Go to Project Settings > API
  3. Copy the Project URL and Service Role Key
  4. Ensure PGVector extension is enabled in your database

### 5. Google Calendar Setup

#### Step 5.1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google Calendar API:
   - Navigate to "APIs & Services" > "Library"
   - Search for "Google Calendar API"
   - Click "Enable"

#### Step 5.2: Create OAuth 2.0 Credentials

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth client ID"
3. Choose "Desktop app" as the application type
4. Download the credentials JSON file
5. Save it as `credentials/credential.json` in the project root

#### Step 5.3: Authenticate

Run the authentication script:

```bash
python config/auth.py
```

This will:
- Open a browser window for Google authentication
- Save the token to `credentials/token.json`
- Verify the connection

### 6. Supabase Database Setup

#### Step 6.1: Create Required Tables

You'll need to set up the following tables in your Supabase database:

1. **conversation_log** table:
```sql
CREATE TABLE conversation_log (
    id BIGSERIAL PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    conversation_turn INTEGER NOT NULL,
    human_message TEXT,
    ai_message TEXT NOT NULL,
    tools_called JSONB DEFAULT '[]',
    tools_result JSONB DEFAULT '{}',
    agents_used JSONB DEFAULT '[]',
    conversation_ended BOOLEAN DEFAULT FALSE,
    risk_probability FLOAT DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

2. **cbt_techniques** table (for CBT tool):
```sql
CREATE TABLE cbt_techniques (
    id SERIAL PRIMARY KEY,
    technique_name TEXT UNIQUE NOT NULL,
    description TEXT,
    example_phrases JSONB,
    indicators JSONB,
    emotional_states JSONB,
    when_to_use TEXT,
    when_not_to_use TEXT
);
```

3. **documents** table (for RAG):
```sql
CREATE TABLE documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT,
    metadata JSONB,
    embedding VECTOR(1536)  -- Adjust dimension based on your embedding model
);
```

#### Step 6.2: Set Up PGVector Extension

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

#### Step 6.3: Create Match Function (for RAG)

```sql
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding VECTOR(1536),
    match_limit INT DEFAULT 5
)
RETURNS TABLE (
    id BIGINT,
    content TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        documents.id,
        documents.content,
        documents.metadata,
        1 - (documents.embedding <=> query_embedding) AS similarity
    FROM documents
    ORDER BY documents.embedding <=> query_embedding
    LIMIT match_limit;
END;
$$;
```

### 7. Risk Classifier Models (Optional)

The risk classifier models are optional. If you want to enable risk assessment:

1. The models should be placed in `riskclassifier_v2/saved_models/`
2. The system will automatically load them if available
3. If models are not found, the chatbot will run without risk assessment

**Note**: The `risk_classifier` and `riskclassifier_v2` folders are archived and should not be modified.

### 8. Populate Knowledge Base (Optional)

To enable RAG functionality, you can populate the `documents` table in Supabase:

1. Place PDF files in the `pdf/` directory
2. Use the PDF loader utility to extract and embed content
3. Store embeddings in the Supabase `documents` table

## Running the Application

### Start the Backend Server

In the first terminal:

```bash
uvicorn backend.api_server:app --reload --host 127.0.0.1 --port 8000
```

The backend will be available at `http://127.0.0.1:8000`

### Start the Frontend Application

In a second terminal:

```bash
streamlit run app/main.py
```

The frontend will automatically open in your browser at `http://localhost:8501`

### Verify Installation

1. **Backend Health Check**: Visit `http://127.0.0.1:8000` - you should see:
   ```json
   {"message": "✅ LangGraph Backend is running"}
   ```

2. **Frontend**: The Streamlit app should load with the mental health assistant interface

3. **Google Calendar**: Try booking an appointment to verify calendar integration

## Usage

### Basic Chat

1. Open the Streamlit application
2. Type your message in the input field
3. The chatbot will respond using appropriate tools and agents

### Booking an Appointment

1. Ask to book an appointment (e.g., "Book a session for tomorrow at 2pm")
2. Provide your name when prompted
3. The system will create a calendar event

### CBT Techniques

The chatbot automatically selects appropriate CBT techniques based on your concerns:
- Cognitive Restructuring
- Behavioral Activation
- Grounding
- Problem Solving
- Mindfulness
- Emotion Regulation

### Analytics Dashboard

To view conversation analytics:

```bash
streamlit run conversation_history/analytics.py
```

This provides insights into:
- Conversation statistics
- Tool usage patterns
- Risk assessment metrics
- Agent performance

## Troubleshooting

### Common Issues

1. **Import Errors**:
   - Ensure virtual environment is activated
   - Reinstall dependencies: `pip install -r requirements.txt`

2. **Google Calendar Authentication Fails**:
   - Verify `credentials/credential.json` exists
   - Run `python config/auth.py` again
   - Check that Calendar API is enabled in Google Cloud Console

3. **Supabase Connection Errors**:
   - Verify `.env` file has correct credentials
   - Check Supabase project is active
   - Ensure database tables are created

4. **Risk Classifier Not Loading**:
   - This is expected if models are not present
   - The chatbot will run without risk assessment
   - Check console for specific error messages

5. **Port Already in Use**:
   - Backend: Change port with `--port 8001`
   - Frontend: Streamlit will automatically use next available port

### Debug Mode

Enable debug logging by setting environment variable:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Development

### Code Structure

- **Agents**: LangGraph state machines for conversation orchestration
- **Tools**: LangChain tools for specific functionalities
- **Backend**: FastAPI REST API for chat processing
- **Frontend**: Streamlit UI for user interaction

### Adding New Tools

1. Create tool function in `tools/` directory
2. Use `@tool` decorator from `langchain_core.tools`
3. Add tool to supervisor agent in `agents/supervisor.py`

### Testing

Run linting:
```bash
# Install development dependencies
pip install black flake8 mypy

# Format code
black .

# Check linting
flake8 .
```

## Environment Variables Reference

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for LLM | Yes |
| `SUPABASE_URL` | Supabase project URL | Yes |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key | Yes |

## Dependencies

Key dependencies:
- **FastAPI**: Backend API framework
- **Streamlit**: Frontend UI framework
- **LangChain/LangGraph**: Agent orchestration
- **OpenAI**: LLM provider
- **Supabase**: Database and vector search
- **Google Calendar API**: Appointment scheduling

See `requirements.txt` for complete list.

## License

[Add your license here]

## Support

For issues and questions:
- Check the troubleshooting section
- Review error messages in console
- Verify all environment variables are set correctly

## Contributing

[Add contribution guidelines if applicable]
