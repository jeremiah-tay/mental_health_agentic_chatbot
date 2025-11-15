<div align="center">
  
# DSA4213 Natural Language Processing in Data Science

[![](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)]()
[![](https://img.shields.io/badge/Agents-LangGraph-%231C3C3C?logo=langgraph&logoColor=white)]()
[![](https://img.shields.io/badge/Backend-FastAPI-green?logo=fastapi)]()
[![](https://img.shields.io/badge/Frontend-Streamlit-red?logo=streamlit)]()
[![](https://img.shields.io/badge/Database-Supabase-brightgreen?logo=supabase)]()

### 🤖 Mental Health Multi-Agentic Chatbot 🤖

![Chatbot Demo Animation](https://app.lottiefiles.com/share/32d3a87c-f097-420f-a56f-01d9ba158ce5)

</div>

## Table of Contents
- [About the Project](#about-the-project)
  - [Agentic Chatbot in Mental Health](#agentic-chatbot-in-mental-health)
  - [Features](#features)
  - [Repository Structure](#repository-structure)
  - [Database Schema](#database-schema)
- [Setup Instructions](#setup-instructions)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Create Virtual Environment](#2-create-virtual-environment)
  - [3. Install Dependencies](#3-install-dependencies)
  - [4. Environment Variables Setup](#4-environment-variables-setup)
  - [5. Google Calendar Setup](#5-google-calendar-setup)
- [Running the Application](#running-the-application)
  - [Start the Backend Server](#start-the-backend-server)
  - [Start the Frontend Application](#start-the-frontend-application)
  - [Verify Installation](#verify-installation)
- [Usage](#usage)
  - [Basic Chat](#basic-chat)
  - [Booking an Appointment](#booking-an-appointment)
  - [CBT Techniques](#cbt-techniques)
  - [Analytics Dashboard](#analytics-dashboard)
- [Troubleshooting](#troubleshooting)
  - [Common Issues](#common-issues)
  - [Support](#support)
- [Group Members](#group-members)

[🔼 Back to Top](#table-of-contents)

# About the Project
## Agentic Chatbot in Mental Health
An intelligent mental health assistant built with LangGraph, Streamlit, and FastAPI.

We created this project to address a critical gap in mental health care: a large portion of people struggling with mental health issues do not seek formal help. While many turn to AI chatbots for support, these tools are often failing them in the most critical moments.

Most existing chatbots rely on simple keyword detection for risk assessment, causing them to underestimate or miss serious suicide risk and offer only generic, unhelpful replies. Our goal was to build a safer, more effective assistant. By using a multi-agent architecture, our chatbot provides real, evidence-based support—like CBT and appointment booking—while a robust risk assessment system works to ensure users in crisis are safely guided toward the professional help they urgently need.


## Features

- **Conversational AI**: Empathetic mental health support using GPT models
- **CBT Technique Selection**: Intelligent selection of appropriate therapeutic techniques
- **RAG (Retrieval-Augmented Generation)**: Access to mental health knowledge base
- **Appointment Booking**: Google Calendar integration for scheduling therapy sessions
- **Risk Assessment**: Automated detection of crisis situations with appropriate responses
- **Analytics Dashboard**: Comprehensive conversation analytics and insights
- **Conversation Logging**: Persistent storage of all interactions in Supabase



[🔼 Back to Top](#table-of-contents)

## Repository Structure

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
| `chunk_index` | INT  | The sequential index of the chunk           |
| `embedding` | vector     | The vector embedding for the `content`                      |
| `created_at` | TIMESTAMPTZ     | Timestamp of when the embedding was created                      |


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
| `id`  | UUID   | Unique identifier for the log entry              |
| `conversation_id`     | UUID      | Identifier linking all turns in a single conversation   |
| `conversation_turn`  | INT  | The sequence number of the turn in the conversation    |
| `human_message` | TEXT  | The user's message           |
| `ai_message` | TEXT     | The chatbot's response    |
| `tools_called` | TEXT[]     | An array of tool names the agent decided to call   |
| `tools_result` | JSONB     | The JSON data returned from the tools                      |
| `agents_used` | TEXT[]     | An array of agents that were activated (e.g., 'Supervisor')         |
| `conversation_ended` | BOOLEAN     | true if the conversation was terminated (e.g., crisis path)        |
| `created_at` | TIMESTAMPTZ     | Timestamp of when the conversation turn occurred         |
| `risk_probability` | FLOAT     | The risk score (0.0 to 1.0) for the human_message          |
```sql
CONSTRAINT conversation_log_conversation_id_conversation_turn_key 
UNIQUE (conversation_id, conversation_turn)
```

[🔼 Back to Top](#table-of-contents)

# Setup Instructions

## 1. Clone the Repository

```bash
git clone <repository-url>
cd mental_health_agentic_chatbot
```

## 2. Create Virtual Environment

```bash
python -m venv chatbot_proj

# On macOS/Linux:
source chatbot_proj/bin/activate

# On Windows:
chatbot_proj\Scripts\activate
```

## 3. Install Dependencies

### Step 3.1: Install Main Dependencies

```bash
pip install -r requirements.txt
```

### Step 3.2: Download Risk Classifier Model

```bash
python riskclassifier_v2/download_models.py
```

**Note**: If you encounter issues with PyPDF2, you may need to install `pypdf` instead:
```bash
pip install pypdf
```

## 4. Environment Variables Setup

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
Before running the application, you must set up your environment variables.
1. Find the file named `.env` in the root of the project.
2. Copy and paste the the keys in the **appendix section of our group project** into the `.env` file.
3. You will have to provide your own OpenAI API KEY

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
```

**How to obtain credentials:**

- **OpenAI API Key**: 
  1. Go to https://platform.openai.com/api-keys
  2. Create a new API key
  3. Copy the key and paste it into `OPENAI_API_KEY`.

[🔼 Back to Top](#table-of-contents)

## 5. Google Calendar Setup
This is required for the Appointment Booking agent to function.

### Step 5.1: Obtain OAuth 2.0 Credentials

1. Go to **appendix section of our Final Group Report**, click on the link that brings you to a Google Drive
2. Download the credentials JSON file
3. Save it as `credentials/credential.json` in the project root

### Step 5.2: Authenticate

This final step links your project to your Google Calendar:
1. Make sure your `credentials/credential.json` file is saved in the correct folder.
2. Run the authentication script:
   ```bash
   python config/auth.py
   ```
3. The script will open a browser window for Google authentication.
4. Log in with the Google Account found in the **appendix section of our Final Group Report**
5. Once you log in and permission in granted, the script will create a `credentials/token.json` file. This token is what the chatbot uses to securely manage your calendar.

[🔼 Back to Top](#table-of-contents)

# Running the Application

## Start the Backend Server

In the first terminal:

```bash
uvicorn backend.api_server:app --reload --host 127.0.0.1 --port 8000
```

The backend will be available at `http://127.0.0.1:8000`

## Start the Frontend Application

In a second terminal:

```bash
streamlit run app/main.py
```

The frontend will automatically open in your browser at `http://localhost:8501`

## Verify Installation

Once both servers are running, follow these steps to verify:

1. **Check the Backend Terminal Log**
   At the terminal where you ran `uvicorn`. You should see success mssages confirming that all components have loaded.
   ```
   ✅ Risk classifier loaded successfully
   Supabase and OpenAI clients initialized
   Loaded 6 CBT technique profiles from Supabase
   Embedding model loaded and 6 technique embeddings computed
   CBT technique selection tool initialized
   INFO:     Started server process [33767]
   INFO:     Waiting for application startup.
   INFO:     Application startup complete.
   ```
   
2. **Frontend**: The Streamlit app should load with the mental health assistant interface

[🔼 Back to Top](#table-of-contents)

# Usage

## Basic Chat

1. Open the Streamlit application
2. Type your message in the input field
3. The chatbot will respond using appropriate tools and agents

## Booking an Appointment

1. Click on `Book an Appointment with a Therapist` or type "Book a session for tomorrow at 2pm"
2. Provide your name when prompted
3. The system will create a calendar event

## CBT Techniques

The chatbot automatically selects appropriate CBT techniques based on your concerns:
- Cognitive Restructuring
- Behavioral Activation
- Grounding
- Problem Solving
- Mindfulness
- Emotion Regulation

## Analytics Dashboard

To view conversation analytics:

```bash
streamlit run conversation_history/analytics.py
```

This provides insights into:
- Conversation statistics
- Tool usage patterns
- Risk assessment metrics
- Agent performance

[🔼 Back to Top](#table-of-contents)

# Troubleshooting

## Common Issues

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

## Support

For issues and questions:
- Check the troubleshooting section
- Review error messages in console
- Verify all environment variables are set correctly

[🔼 Back to Top](#table-of-contents)

# Group Members
| Name           | GitHub         | 
|----------------|-----------------|
| Jeremiah Tay    | [jeremiah-tay](https://github.com/jeremiah-tay) |
| Sim Zhi Sherng   | [sim-zhi-sherng](https://github.com/ZhiSherng) |
| Wynnona Pheeby | [wynpyy](https://github.com/wynpyy)  | 
| Rachel Chun | [rachel](https://github.com/Chxlz)  | 
