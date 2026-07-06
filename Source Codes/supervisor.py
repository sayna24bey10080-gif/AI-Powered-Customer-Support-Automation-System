"""
supervisor.py – LLM-powered supervisor that reviews and improves agent responses.
Uses Groq to validate quality, tone, accuracy, and completeness before
the response reaches the customer.
"""

import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
SUPERVISOR_MODEL = "llama-3.3-70b-versatile"


SUPERVISOR_SYSTEM = """You are a senior customer support quality supervisor at ABC Technologies.
Your job is to review agent-drafted responses before they are sent to customers.

Evaluate each response on:
1. ACCURACY – Is the information factually correct based on the provided context?
2. COMPLETENESS – Does it fully answer the customer's question?
3. TONE – Is it professional, empathetic, and on-brand?
4. ACTIONABILITY – Does it give the customer clear next steps?
5. COMPLIANCE – Does it follow company policies correctly?

If the response is good, approve it with minor polish.
If the response has significant issues, rewrite it to fix them.

Always end your response with:
---
Thank you for contacting ABC Technologies Support. Is there anything else I can help you with?"""


def supervisor_review(query: str, agent_response: str, context: str, department: str) -> dict:
    """
    Review the agent's draft response and return an improved final response
    along with supervisor feedback.

    Returns:
        dict with keys: 'final_response', 'feedback', 'approved'
    """
    prompt = f"""CUSTOMER QUERY (Department: {department}):
{query}

RETRIEVED KNOWLEDGE BASE CONTEXT:
{context}

AGENT'S DRAFT RESPONSE:
{agent_response}

Please review the draft response. If it is accurate and complete, approve it (with light polish).
If it has issues (missing info, wrong tone, inaccurate claims), rewrite it.
First give a brief one-sentence feedback on what you changed or confirmed, then provide the final response."""

    try:
        chat_completion = client.chat.completions.create(
            model=SUPERVISOR_MODEL,
            messages=[
                {"role": "system", "content": SUPERVISOR_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        full_output = chat_completion.choices[0].message.content.strip()

        # Split feedback line from the actual response
        lines = full_output.split("\n", 1)
        feedback = lines[0].strip() if len(lines) > 1 else "Response reviewed and approved."
        final_response = lines[1].strip() if len(lines) > 1 else full_output

        return {
            "final_response": final_response,
            "feedback": feedback,
            "approved": True,
        }

    except Exception as e:
        # Graceful fallback – pass through agent response if supervisor fails
        return {
            "final_response": agent_response + "\n\n---\nThank you for contacting ABC Technologies Support.",
            "feedback": f"Supervisor LLM unavailable ({e}); agent response passed through.",
            "approved": False,
        }
