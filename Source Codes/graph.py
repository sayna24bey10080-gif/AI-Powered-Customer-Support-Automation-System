"""
graph.py – LangGraph workflow for AI-powered customer support.
Every agent uses Groq (Llama 3 70B) to generate responses from
retrieved RAG context. Intent classification also uses the LLM.
"""

import os
from groq import Groq
from langgraph.graph import StateGraph, END

from state import SupportState
from rag import retrieve_docs
from memory import save_conversation, get_last_issue, get_conversation_history
from human_review import requires_human_review
from supervisor import supervisor_review

# ── Groq client ───────────────────────────────────────────────────────────────
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
AGENT_MODEL = "llama-3.3-70b-versatile"


# ── Shared LLM call helper ────────────────────────────────────────────────────
def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.4) -> str:
    response = client.chat.completions.create(
        model=AGENT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()


# ── Intent Classification (LLM-powered) ──────────────────────────────────────
CLASSIFIER_SYSTEM = """You are a customer support routing assistant for ABC Technologies.
Classify the customer's query into exactly one of these departments:
- Sales (pricing, plans, subscriptions, upgrades, billing)
- Technical (installation, errors, crashes, login issues, app bugs, API, integrations)
- Billing (refunds, invoices, payment failures, charges, disputes)
- Account (profile changes, password, team management, 2FA, account closure)
- Memory (customer asking about their previous support history/issue)
- General (anything else)

Respond with ONLY the department name, nothing else."""

def classify_intent(state: SupportState) -> SupportState:
    query = state["query"]
    try:
        department = call_llm(CLASSIFIER_SYSTEM, query, temperature=0.0).strip()
        # Validate output
        valid = {"Sales", "Technical", "Billing", "Account", "Memory", "General"}
        if department not in valid:
            department = "General"
    except Exception:
        # Fallback to keyword matching if LLM fails
        q = query.lower()
        if "previous" in q and "issue" in q:
            department = "Memory"
        elif any(w in q for w in ["pricing", "plan", "subscription", "upgrade"]):
            department = "Sales"
        elif any(w in q for w in ["error", "crash", "install", "login", "bug", "api"]):
            department = "Technical"
        elif any(w in q for w in ["refund", "invoice", "charge", "payment", "dispute"]):
            department = "Billing"
        elif any(w in q for w in ["password", "profile", "account", "2fa", "team"]):
            department = "Account"
        else:
            department = "General"

    state["department"] = department
    return state


# ── Generic LLM Agent ─────────────────────────────────────────────────────────
AGENT_SYSTEM = """You are a helpful, professional customer support agent for ABC Technologies.
Use ONLY the provided knowledge base context to answer the customer's question.
Be specific, accurate, and empathetic. If the context doesn't cover the question,
say so honestly and suggest contacting support@abctech.com.
Do not make up information not present in the context."""

def run_agent(state: SupportState, agent_name: str) -> SupportState:
    query = state["query"]
    context = retrieve_docs(query)
    state["retrieved_context"] = context

    # Include brief conversation history for context
    history = get_conversation_history(state["customer_name"], limit=3)
    history_text = ""
    if history:
        history_text = "\n\nPREVIOUS INTERACTIONS WITH THIS CUSTOMER:\n"
        for h in history:
            history_text += f"- [{h['timestamp']}] {h['department']}: {h['issue'][:100]}...\n"

    user_prompt = f"""Customer Name: {state['customer_name']}
Department: {agent_name}
Customer Query: {query}
{history_text}
KNOWLEDGE BASE CONTEXT:
{context}

Please provide a helpful, accurate response."""

    try:
        response = call_llm(AGENT_SYSTEM, user_prompt)
    except Exception as e:
        response = (
            f"I apologize, but I'm experiencing a technical issue ({e}). "
            "Please contact support@abctech.com for immediate assistance."
        )

    state["response"] = response
    return state


def sales_agent(state: SupportState) -> SupportState:
    return run_agent(state, "Sales")

def technical_agent(state: SupportState) -> SupportState:
    return run_agent(state, "Technical")

def billing_agent(state: SupportState) -> SupportState:
    return run_agent(state, "Billing")

def account_agent(state: SupportState) -> SupportState:
    return run_agent(state, "Account")

def general_agent(state: SupportState) -> SupportState:
    return run_agent(state, "General")


# ── Memory Node ───────────────────────────────────────────────────────────────
def memory_node(state: SupportState) -> SupportState:
    query = state["query"].lower()
    if "previous" in query and ("issue" in query or "history" in query or "last" in query):
        # Customer asking about their history
        state["response"] = get_last_issue(state["customer_name"])
    else:
        # Save this conversation (response will be updated by supervisor)
        save_conversation(
            name=state["customer_name"],
            department=state["department"],
            issue=state["query"],
            response=state.get("response", ""),
        )
    return state


# ── Human Review Node ─────────────────────────────────────────────────────────
def human_review_node(state: SupportState) -> SupportState:
    if requires_human_review(state["query"]):
        state["approval_required"] = True
        escalation_note = (
            "\n\n⚠️ ESCALATION NOTICE: This query has been flagged for human supervisor review "
            "due to its sensitive nature (refund / cancellation / dispute). "
            "A senior support agent will follow up within 2 business hours."
        )
        state["response"] = state.get("response", "") + escalation_note
    return state


# ── Supervisor Node ───────────────────────────────────────────────────────────
def supervisor_node(state: SupportState) -> SupportState:
    result = supervisor_review(
        query=state["query"],
        agent_response=state.get("response", ""),
        context=state.get("retrieved_context", ""),
        department=state["department"],
    )
    state["final_response"] = result["final_response"]
    state["supervisor_feedback"] = result["feedback"]
    state["approved"] = result["approved"]

    # Update memory with the final polished response
    if state["department"] != "Memory":
        save_conversation(
            name=state["customer_name"],
            department=state["department"],
            issue=state["query"],
            response=result["final_response"][:500],  # store first 500 chars
        )

    return state


# ── Routing ───────────────────────────────────────────────────────────────────
def route_department(state: SupportState) -> str:
    return state["department"]


# ── Graph Assembly ────────────────────────────────────────────────────────────
builder = StateGraph(SupportState)

builder.add_node("classifier", classify_intent)
builder.add_node("sales", sales_agent)
builder.add_node("technical", technical_agent)
builder.add_node("billing", billing_agent)
builder.add_node("account", account_agent)
builder.add_node("general", general_agent)
builder.add_node("memory", memory_node)
builder.add_node("human", human_review_node)
builder.add_node("supervisor", supervisor_node)

builder.set_entry_point("classifier")

builder.add_conditional_edges(
    "classifier",
    route_department,
    {
        "Sales": "sales",
        "Technical": "technical",
        "Billing": "billing",
        "Account": "account",
        "Memory": "memory",
        "General": "general",
    },
)

# All agents → memory → human review → supervisor → END
for agent in ("sales", "technical", "billing", "account", "general"):
    builder.add_edge(agent, "memory")

builder.add_edge("memory", "human")
builder.add_edge("human", "supervisor")
builder.add_edge("supervisor", END)

graph = builder.compile()
