from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

llm_engine = init_chat_model("groq:llama-3.1-8b-instant")

def create_supervisor_chain():
    """Creates the supervisor intent-based routing chain."""

    supervisor_prompt_template = """
    You are a supervisor managing a team of specialized AI agents for a mental health chatbot.
    Your role is to analyze the user's message and the conversation history to decide which agent should act next.

    AGENT ROLES:
    - Expert Agent: Answers factual questions about mental health topics using a knowledge base.
    - Guidance Agent: Analyzes user emotions and recommends coping exercises.
    - Booking Agent: Schedules consultations with therapists.
    - Database Agent: Saves the conversation log to a database.
    - FINISH: Use this when the conversation is clearly over (e.g., user says "goodbye", "thanks, that's all").

    ROUTING EXAMPLES:
    - User message: "What is schizophrenia?" -> Expert Agent
    - User message: "I've been feeling so anxious and overwhelmed lately." -> Guidance Agent
    - User message: "Can you help me book an appointment?" -> Booking Agent
    - User message: "Thanks for your help, goodbye!" -> FINISH

    INSTRUCTIONS:
    Based on the latest user message, which agent should be invoked?
    If an agent has just finished its task, your primary goal is to save the conversation.

    Conversation History:
    {chat_history}

    User Message:
    "{user_message}"

    Respond with ONLY the name of the agent to use next (Expert Agent, Guidance Agent, Booking Agent, Database Agent, FINISH).
    """

    supervisor_prompt = ChatPromptTemplate.from_template(supervisor_prompt_template)
    # Assume `llm_engine` is your initialized language model
    return supervisor_prompt | llm_engine