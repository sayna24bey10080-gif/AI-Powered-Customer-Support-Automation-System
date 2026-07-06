# 🤖 AI-Powered Customer Support Automation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-green?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-Llama3_70B-orange?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-red?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-pink?style=for-the-badge)
![SQLite](https://img.shields.io/badge/SQLite-Memory-lightgrey?style=for-the-badge)

A production-quality, multi-agent AI customer support system with real RAG, LLM-powered routing, supervisor review, and a full Streamlit chat interface.

</div>

---

## 📌 Project Overview

This system automates customer support for **ABC Technologies** using a multi-agent architecture built on **LangGraph**. Each customer query is classified by an LLM, routed to a specialist agent, answered using **real vector search (FAISS + sentence-transformers)** over a knowledge base, reviewed by an **LLM supervisor**, and delivered through a polished **Streamlit UI**.

Unlike toy implementations, every component uses a real LLM (Groq Llama 3.3 70B) — there is no keyword matching, no hardcoded responses, and no fake RAG.

---

## ✨ Key Features

| Feature | Implementation |
|---|---|
| 🔍 **Real RAG** | `sentence-transformers` embeddings + FAISS cosine similarity index, top-4 chunk retrieval |
| 🧠 **LLM Intent Classification** | Groq Llama 3.3 70B classifies every query into the correct department |
| 🤖 **LLM Specialist Agents** | 5 agents (Sales, Technical, Billing, Account, General) each powered by Groq |
| 🔎 **LLM Supervisor** | Groq reviews every agent response for accuracy, tone, completeness & compliance |
| 💾 **Conversation Memory** | SQLite stores full interaction history per customer, recalled across sessions |
| 🚨 **Human-in-the-Loop** | Automatic escalation detection for sensitive queries (refunds, disputes, cancellations) |
| 💬 **Streamlit Chat UI** | Full chat interface with department badges, supervisor feedback, RAG viewer & session stats |

---

## 🏗️ System Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│         Classifier Node             │
│   LLM Intent Classification         │
│   (Groq Llama 3.3 70B)             │
└─────────────────────────────────────┘
    │
    ├─── Sales ────────┐
    ├─── Technical ────┤
    ├─── Billing ──────┤──► Specialist Agent Node
    ├─── Account ──────┤     │  ① RAG retrieval (FAISS)
    ├─── General ──────┘     │  ② Groq LLM response generation
    └─── Memory ─────────────┘
                │
                ▼
    ┌───────────────────────┐
    │     Memory Node       │
    │  SQLite save / recall │
    └───────────────────────┘
                │
                ▼
    ┌───────────────────────┐
    │   Human Review Node   │
    │  Escalation detection │
    └───────────────────────┘
                │
                ▼
    ┌───────────────────────┐
    │    Supervisor Node    │
    │  LLM quality review   │
    │  & response rewrite   │
    └───────────────────────┘
                │
                ▼
         Final Response
          Streamlit UI
```

---

## 🔍 How Real RAG Works

This is **not** keyword matching. Here is the actual pipeline:

1. **Chunking** — All `.txt` files in `docs/` are split into 300-character overlapping chunks (50-char overlap to avoid cutting sentences).
2. **Embedding** — Each chunk is encoded into a 384-dimensional vector using `all-MiniLM-L6-v2` (runs locally, no API needed).
3. **Indexing** — Vectors are L2-normalized and stored in a FAISS `IndexFlatIP` (Inner Product = cosine similarity after normalization).
4. **Retrieval** — At query time, the query is embedded and the top-4 most semantically similar chunks are retrieved — "app won't open" correctly matches "application crash on startup" even though the words differ.
5. **Generation** — Retrieved chunks are injected into the LLM prompt as grounding context. The agent is instructed to use ONLY this context, preventing hallucination.

---

## 📁 Project Structure

```
AI-Powered Customer Support Automation System/
│
├── Source Code/
│   ├── app.py              # Streamlit chat UI (frontend)
│   ├── graph.py            # LangGraph workflow, all agent nodes, routing logic
│   ├── rag.py              # Real RAG: FAISS index, sentence-transformers, chunk retrieval
│   ├── supervisor.py       # LLM-powered response quality review (Groq)
│   ├── memory.py           # SQLite conversation history (save & recall)
│   ├── state.py            # Shared TypedDict state schema for LangGraph
│   └── human_review.py     # Escalation trigger detection
│
├── docs/
│   ├── pricing_guide.txt        # Full pricing: plans, billing, add-ons (~1.5KB)
│   ├── company_policy.txt       # Refund, cancellation, compensation policies (~2KB)
│   ├── technical_manual.txt     # Installation, troubleshooting, API docs (~3KB)
│   └── faq.txt                  # Comprehensive FAQ (~3KB)
│
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- A free Groq API key from [console.groq.com](https://console.groq.com)

### Step 1 — Clone the repository
```bash
git clone https://github.com/sayna24bey10080-gif/AI-Powered-Customer-Support-Automation-System.git
cd AI-Powered-Customer-Support-Automation-System
```

### Step 2 — Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the app
```bash
cd "Source Code"
streamlit run app.py
```

### Step 5 — Enter your Groq API key
Open `http://localhost:8501` in your browser and paste your Groq API key in the sidebar.

> **Note:** The FAISS vector index builds automatically on first launch (~15 seconds). It is cached for all subsequent runs.

---

## 🧰 Tech Stack

| Technology | Purpose |
|---|---|
| [LangGraph](https://github.com/langchain-ai/langgraph) | Multi-agent graph orchestration & state management |
| [Groq](https://console.groq.com) | Ultra-fast LLM inference (Llama 3.3 70B) |
| [FAISS](https://github.com/facebookresearch/faiss) | Facebook's vector similarity search library |
| [sentence-transformers](https://www.sbert.net/) | Local text embeddings (`all-MiniLM-L6-v2`) |
| [Streamlit](https://streamlit.io/) | Python-native web UI framework |
| SQLite | Lightweight local database for conversation memory |

---

## 📊 Agent Departments

| Department | Triggered By | Knowledge Source |
|---|---|---|
| 💰 Sales | Pricing, plans, subscriptions, upgrades | `pricing_guide.txt` |
| 🔧 Technical | Errors, crashes, installation, login, API | `technical_manual.txt` |
| 💳 Billing | Refunds, invoices, payment failures, disputes | `company_policy.txt` |
| 👤 Account | Password, 2FA, profile, team management | `faq.txt` |
| ❓ General | Everything else | All docs |
| 🔙 Memory | "What was my previous issue?" | SQLite database |

---

## ⚙️ Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key — set via sidebar UI or environment variable |

---

## 🙋 Example Queries to Try

- *"What are the differences between the Basic and Pro plans?"* → Sales agent
- *"My application keeps crashing on startup. How do I fix it?"* → Technical agent  
- *"I was charged incorrectly last month. I need a refund."* → Billing agent + Escalation flag
- *"How do I enable two-factor authentication?"* → Account agent
- *"What was my previous support issue?"* → Memory recall
- *"I need to cancel my subscription and speak to a manager."* → Escalation + Human review

---

## 👩‍💻 Author

**Sayna Patel**  
[GitHub](https://github.com/sayna24bey10080-gif)
