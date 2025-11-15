# DSA4213 Natural Language Processing in Data Science


### 🌍 Mental Health Multi-Agentic Chatbot 🌍
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

## Database Schema
Our team will be using Supabase to host it online @ [![Supabase](https://img.shields.io/badge/Supabase-Database-green?logo=supabase&style=flat-square)](https://supabase.io/). If you would like to access our Postgres database, please contact us for the login details.

### `rag_text_chunks`
Stores the raw text chunks extracted from the mental health PDF documents.

| Column     | Type     | Description                               |
|------------|----------|-------------------------------------------|
| `id`  | INT (PK) | Unique identifier for the text chunk                       |
| `metadata`     | JSONB      | Stores metadata about the chunk                    |
| `source`  | TEXT  | The original PDF document source     |
| `chunk_number` | INT  | The sequential index of the chunk within its source document           |
| `content` | TEXT     | The raw text content of the chunk                      |

### `documents`
Stores the vector embeddings for each text chunk, enabling RAG similarity search via `pgvector`.

| Column     | Type     | Description                               |
|------------|----------|-------------------------------------------|
| `id`  | INT (PK) | Unique identifier for the document embedding                       |
| `source`     | TEXT      | The original document source                    |
| `content`  | TEXT  | The raw text content of the chunk     |
| `chunk index` | INT  | The sequential index of the chunk           |
| `embedding` | vector     | The vector embedding for the `content`                      |
| `created_at` | TIMESTAMPTZ     | Timestamp of when the embedding was created                      |

### `risk_classifier_data`
Stores the labeled training data used to fine-tune the Risk Classifier model.

| Column     | Type     | Description                               |
|------------|----------|-------------------------------------------|
| `id`  | INT (PK)  | Unique identifier for the training sample                       |
| `text`     | VARCHAR      | The user input text sample                    |
| `label`  | VARCHAR  | The classification label (e.g., 1 for at-risk, 0 for not-at-risk).     |

constraint risk_classifier_data_pkey primary key (id)


### `cbt_techniques`
Stores the profiles for each Cognitive Behavioural Therapy (CBT) technique used by the CBT tool.

| Column     | Type     | Description                               |
|------------|----------|-------------------------------------------|
| `technique_name`  | TEXT (PK) | The unique name of the CBT technique                       |
| `description`     | TEXT      | A clinical description of the technique                    |
| `example_phrases`  | TEXT[]  | An array of example user phrases that match this technique     |
| `indicators` | TEXT  | Clinical indicators for when to use this technique           |
| `emotional_states` | TEXT[]     | An array of target emotional states (e.g., 'anxiety', 'sadness')    |
| `when_to_use` | TEXT     | Specific use cases or scenarios                      |
| `when_not_to_use` | TEXT     | Contraindications (when this technique is inappropriate)           |

constraint cbt_techniques_pkey

### `conversation_log`
Stores a detailed log of every conversation turn for analysis and monitoring on the analytics dashboard.

| Column     | Type     | Description                               |
|------------|----------|-------------------------------------------|
| `id`  | UUID (PK)  | Unique identifier for the log entry                       |
| `conversation_id`     | UUID      | Identifier linking all turns in a single conversation     |
| `conversation_turn`  | INT  | The sequence number of the turn in the conversation     |
| `human_message` | TEXT  | The user's message           |
| `ai_message` | TEXT     | The chatbot's response                      |
| `tools_called` | TEXT[]     | An array of tool names the agent decided to call                      |
| `tools_result` | JSONB     | The JSON data returned from the tools                      |
| `agents_used` | TEXT[]     | An array of agents that were activated (e.g., 'Supervisor')                      |
| `conversation_ended` | BOOLEAN     | true if the conversation was terminated (e.g., crisis path)                      |
| `created_at` | TIMESTAMPTZ     | Timestamp of when the conversation turn occurred                      |
| `risk_probability` | FLOAT     | The risk score (0.0 to 1.0) for the human_message                      |

constraint conversation_log_pkey primary key (id)
constraint conversation_log_conversation_id_conversation_turn_key unique (conversation_id, conversation_turn)
  
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
- You will have to use your own OpenAI API KEY
- The rest of the keys, please refer to the appendix section of our group report.
```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
```

**How to obtain credentials:**

- **OpenAI API Key**: 
  1. Go to https://platform.openai.com/api-keys
  2. Create a new API key
  3. Copy and paste into `.env`

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
- Refer to the appendix section to obtain the username and password for the google account (WHERE SHOULD I BEST PUT THIS)
- Save the token to `credentials/token.json`
- Verify the connection

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

## Support

For issues and questions:
- Check the troubleshooting section
- Review error messages in console
- Verify all environment variables are set correctly

## Contributing

[Add contribution guidelines if applicable]
