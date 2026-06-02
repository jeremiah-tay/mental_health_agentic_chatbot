"""
Amazon Bedrock AgentCore Runtime entrypoint for the mental health supervisor agent.

Run locally:
    python agentcore_main.py

Or with the AgentCore CLI:
    agentcore configure --entrypoint agentcore_main.py
    agentcore deploy
"""
import os
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from bedrock_agentcore import BedrockAgentCoreApp

# Ensure project root is on the path (AgentCore may run from another cwd)
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

load_dotenv()

from agents.supervisor import create_supervisor_graph

app = BedrockAgentCoreApp()
log = app.logger


def _build_messages(
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> list:
    messages = []
    for item in history or []:
        role = item.get("role")
        content = item.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=user_message))
    return messages


def _invoke_supervisor(
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
    conversation_id: Optional[str] = None,
) -> Dict[str, Any]:
    messages = _build_messages(message, history)
    state = {
        "messages": messages,
        "conversation_ended": False,
        "tools_called": [],
        "tools_result": {},
        "agents_used": [],
        "risk_probability": 0.0,
    }
    if conversation_id:
        state["conversation_id"] = conversation_id

    final_state = _supervisor_graph.invoke(state)
    final_messages = final_state.get("messages", [])
    reply = final_messages[-1].content if final_messages else "No response."

    return {
        "response": reply,
        "messages": [m.content for m in final_messages],
        "conversation_ended": final_state.get("conversation_ended", False),
        "risk_probability": round(float(final_state.get("risk_probability", 0.0)), 5),
        "tools_called": final_state.get("tools_called", []),
        "tools_result": final_state.get("tools_result", {}),
        "agents_used": final_state.get("agents_used", []),
    }


def _create_supervisor_graph():
    log.info("Initialising mental health supervisor graph...")
    llm = ChatOpenAI(
        model="gpt-5-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.2,
    )
    graph = create_supervisor_graph(llm)
    log.info("Supervisor graph ready")
    return graph


try:
    _supervisor_graph = _create_supervisor_graph()
except Exception as exc:
    log.error("Critical failure during agent initialisation: %s", exc)
    raise


@app.entrypoint
def invoke(payload: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """
    AgentCore handler for each invocation.

    Accepts:
      - prompt or message: current user utterance
      - history: optional list of {role, content} prior turns
      - conversation_id: optional; falls back to context.session_id when present
    """
    try:
        user_message = (
            payload.get("message")
            or payload.get("prompt")
            or ""
        ).strip()
        if not user_message:
            return {"error": "message or prompt is required"}

        history = payload.get("history")
        conversation_id = payload.get("conversation_id")
        if not conversation_id and context is not None:
            conversation_id = getattr(context, "session_id", None)

        log.info("Invoke received (conversation_id=%s)", conversation_id)
        return _invoke_supervisor(user_message, history, conversation_id)
    except Exception as exc:
        log.exception("Invoke failed: %s", exc)
        return {"error": str(exc)}


if __name__ == "__main__":
    app.run()
