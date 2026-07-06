"""
app.py – Streamlit UI for the AI Customer Support Automation System.
Run with: streamlit run app.py
"""

import sys
import os
from pathlib import Path

# Ensure the Source Code directory is on the path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ABC Technologies Support",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        color: white;
    }
    .department-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 4px;
    }
    .badge-sales { background: #e8f4fd; color: #1a6fa8; }
    .badge-technical { background: #fff3e0; color: #e65100; }
    .badge-billing { background: #fce4ec; color: #880e4f; }
    .badge-account { background: #e8f5e9; color: #1b5e20; }
    .badge-general { background: #f3e5f5; color: #4a148c; }
    .badge-memory { background: #e3f2fd; color: #0d47a1; }
    .response-box {
        background: #f8f9fa;
        border-left: 4px solid #0f3460;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin-top: 0.5rem;
        color: #1a1a1a !important;
    }
    }
    .supervisor-box {
        background: #fff8e1;
        border-left: 4px solid #f59e0b;
        padding: 0.7rem 1rem;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #78350f !important;
        margin-top: 0.5rem;
    }
    }
    .escalation-warning {
        background: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 0.7rem 1rem;
        border-radius: 6px;
        font-size: 0.9rem;
        color: #7f1d1d;
    }
    .history-item {
        background: #f1f5f9;
        border-radius: 8px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.4rem;
        font-size: 0.85rem;
        color: #1a1a1a !important;
    }
    }
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ── Lazy imports (so Streamlit loads before heavy deps) ───────────────────────
@st.cache_resource(show_spinner="Loading AI models and building knowledge index...")
def load_graph():
    from graph import graph
    return graph

@st.cache_resource(show_spinner=False)
def get_memory_functions():
    from memory import get_conversation_history
    return get_conversation_history


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x60/0f3460/ffffff?text=ABC+Technologies", width=200)
    st.markdown("---")

    st.subheader("🔑 Configuration")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get your free key at console.groq.com",
    )
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    st.markdown("---")
    st.subheader("👤 Customer Details")
    customer_name = st.text_input("Your Name", value="", placeholder="e.g. David")

    st.markdown("---")
    st.subheader("📋 Example Queries")
    examples = {
        "💰 Pricing": "What are the differences between the Basic and Pro plans?",
        "🔧 Technical": "My application keeps crashing on startup. How do I fix it?",
        "💳 Billing": "I was charged incorrectly last month. I need a refund.",
        "👤 Account": "How do I enable two-factor authentication on my account?",
        "🔙 History": "What was my previous support issue?",
        "❓ General": "Do you have a referral program?",
        "⚠️ Escalation": "I need to cancel my subscription and speak to a manager.",
    }
    for label, example_query in examples.items():
        if st.button(label, use_container_width=True):
            st.session_state["prefill_query"] = example_query

    st.markdown("---")
    st.caption("Powered by LangGraph + Groq + FAISS")


# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤖 AI Customer Support</h1>
    <p style="opacity:0.85; margin:0;">Powered by LangGraph · RAG · Groq Llama 3 70B</p>
</div>
""", unsafe_allow_html=True)

# ── API Key guard ─────────────────────────────────────────────────────────────
if not os.environ.get("GROQ_API_KEY"):
    st.warning("⬅️ Enter your Groq API key in the sidebar to get started.")
    st.stop()

# ── Load graph ────────────────────────────────────────────────────────────────
graph = load_graph()
get_history = get_memory_functions()

# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "prefill_query" not in st.session_state:
    st.session_state.prefill_query = ""

# ── Two-column layout ─────────────────────────────────────────────────────────
col_chat, col_info = st.columns([2, 1])

with col_chat:
    st.subheader("💬 Support Chat")

    # Chat history display
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(f"**{turn['customer']}:** {turn['query']}")
        with st.chat_message("assistant"):
            dept_colors = {
                "Sales": "badge-sales", "Technical": "badge-technical",
                "Billing": "badge-billing", "Account": "badge-account",
                "General": "badge-general", "Memory": "badge-memory",
            }
            badge_cls = dept_colors.get(turn["department"], "badge-general")
            st.markdown(
                f'<span class="department-badge {badge_cls}">🏷️ {turn["department"]}</span>',
                unsafe_allow_html=True,
            )
            if turn.get("escalated"):
                st.markdown(
                    '<div class="escalation-warning">⚠️ This query has been escalated for human review.</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<div class="response-box">{turn["response"]}</div>',
                unsafe_allow_html=True,
            )
            if turn.get("supervisor_feedback"):
                st.markdown(
                    f'<div class="supervisor-box">🔍 <strong>Supervisor:</strong> {turn["supervisor_feedback"]}</div>',
                    unsafe_allow_html=True,
                )

    # Query input
    prefill = st.session_state.pop("prefill_query", "") or ""
    query = st.chat_input(
        placeholder="Describe your issue or question...",
    )

    # Handle prefill via text area if chat_input can't be pre-filled
    if prefill:
        st.info(f"💡 Example loaded: *{prefill}*")
        query = query or prefill

    if query:
        if not customer_name.strip():
            st.error("Please enter your name in the sidebar first.")
        else:
            with st.spinner("🤖 Routing your query and generating a response..."):
                initial_state: dict = {
                    "customer_name": customer_name.strip(),
                    "query": query,
                    "department": "",
                    "retrieved_context": "",
                    "approval_required": False,
                    "approved": False,
                    "memory": [],
                    "response": "",
                    "final_response": "",
                    "supervisor_feedback": "",
                    "conversation_history": [],
                }
                try:
                    result = graph.invoke(initial_state)
                    st.session_state.chat_history.append({
                        "customer": customer_name.strip(),
                        "query": query,
                        "department": result.get("department", "General"),
                        "response": result.get("final_response") or result.get("response", ""),
                        "supervisor_feedback": result.get("supervisor_feedback", ""),
                        "escalated": result.get("approval_required", False),
                        "context_snippet": result.get("retrieved_context", "")[:300],
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Something went wrong: {e}")


with col_info:
    st.subheader("📊 Session Info")

    total = len(st.session_state.chat_history)
    escalated = sum(1 for t in st.session_state.chat_history if t.get("escalated"))
    depts = {}
    for t in st.session_state.chat_history:
        depts[t["department"]] = depts.get(t["department"], 0) + 1

    c1, c2 = st.columns(2)
    c1.metric("Queries", total)
    c2.metric("Escalations", escalated)

    if depts:
        st.markdown("**Departments Handled**")
        for dept, count in sorted(depts.items(), key=lambda x: -x[1]):
            st.progress(count / total, text=f"{dept}: {count}")

    st.markdown("---")
    st.subheader("🕓 Your Support History")
    if customer_name.strip():
        history = get_history(customer_name.strip(), limit=5)
        if history:
            for h in history:
                st.markdown(
                    f'<div class="history-item"><strong>{h["department"]}</strong> · {h["timestamp"][:16]}<br>'
                    f'<em>{h["issue"][:80]}...</em></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No previous interactions found.")
    else:
        st.caption("Enter your name in the sidebar to see history.")

    st.markdown("---")
    if st.session_state.chat_history:
        last = st.session_state.chat_history[-1]
        with st.expander("🔍 Last Retrieved Context (RAG)"):
            st.text(last.get("context_snippet", "N/A"))

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
