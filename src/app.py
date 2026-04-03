"""
app.py — HelpDesk Copilot
Streamlit dashboard for the support ticket triage agent.
"""

import os
import sys
import streamlit as st
import time
from db import log_ticket

# Fix imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(
    page_title="HelpDesk Copilot",
    page_icon="🎫",
    layout="wide",
)

# ── Load agent (cached so it only loads once) ──
@st.cache_resource
def load_agent():
    from agent import build_agent, load_resources, load_classifier
    from agent import bm25, collection, doc_ids, kb_df, llm, classifier
    agent = build_agent()
    return agent

with st.spinner("Loading models... (first time takes ~30 seconds)"):
    agent = load_agent()


# ── HEADER ──
st.title("🎫 HelpDesk Copilot")
st.markdown("AI-powered support ticket triage — classify, retrieve, draft, escalate")
st.divider()


# ── SIDEBAR ──
with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider(
        "Escalation Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.40,
        step=0.05,
        help="Tickets below this confidence get escalated for human review"
    )

    st.divider()
    st.header("📊 About")
    st.markdown("""
    **Stack:**
    - Classification: BGE + SVM
    - Retrieval: Hybrid (BM25 + MiniLM)
    - Drafting: GPT-4o-mini
    - Agent: LangGraph
    
    **Datasets:**
    - 20,581 KB articles (DevRev)
    - 23,742 support tickets
    """)

    st.divider()
    st.header("📈 Session Stats")
    if "ticket_count" not in st.session_state:
        st.session_state.ticket_count = 0
        st.session_state.escalated_count = 0
        st.session_state.auto_replied_count = 0

    col1, col2 = st.columns(2)
    col1.metric("Total", st.session_state.ticket_count)
    col2.metric("Escalated", st.session_state.escalated_count)


# ── SAMPLE TICKETS ──
sample_tickets = [
    "How do I connect my Bitbucket account to DevRev for syncing issues?",
    "I want to set up an automation workflow that auto-closes tickets after 48 hours of no customer response.",
    "How can I create a snap-in command in DevRev?",
    "What is the process to install an app from the DevRev marketplace?",
    "I was charged twice for my subscription this month. Please refund the duplicate charge.",
    "Our team needs help setting up SSO with Okta.",
]

st.subheader("✉️ Submit a Ticket")

# Sample ticket selector
selected_sample = st.selectbox(
    "Try a sample ticket:",
    ["-- Type your own --"] + sample_tickets,
)

if selected_sample != "-- Type your own --":
    ticket_text = st.text_area("Ticket Text", value=selected_sample, height=100)
else:
    ticket_text = st.text_area("Ticket Text", placeholder="Describe your issue here...", height=100)


# ── PROCESS TICKET ──
if st.button("🚀 Process Ticket", type="primary", use_container_width=True):
    if not ticket_text.strip():
        st.warning("Please enter a ticket description.")
    else:
        # Update session stats
        st.session_state.ticket_count += 1

        with st.spinner("Processing ticket..."):
            start_time = time.time()

            result = agent.invoke({
                "ticket_text": ticket_text,
                "queue": "",
                "queue_confidence": 0.0,
                "priority": "",
                "priority_confidence": 0.0,
                "retrieved_articles": [],
                "draft_reply": "",
                "escalated": False,
                "reasoning": "",
            })

            elapsed = time.time() - start_time

        # Update stats
         # Log to PostgreSQL for Grafana
        log_ticket(
            ticket_text=ticket_text,
            queue=result["queue"],
            queue_confidence=result["queue_confidence"],
            priority=result["priority"],
            priority_confidence=result["priority_confidence"],
            escalated=result["escalated"],
            response_time_ms=int(elapsed * 1000),
            num_articles=len(result["retrieved_articles"]),
            draft_reply=result["draft_reply"],
        )
        if result["escalated"]:
            st.session_state.escalated_count += 1
        else:
            st.session_state.auto_replied_count += 1

        # ── RESULTS ──
        st.divider()

        # Status banner
        if result["escalated"]:
            st.error("⚠️ ESCALATED — Needs human review before sending")
        else:
            st.success("✅ Auto-drafted — Ready for review")

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Queue", result["queue"])
        col2.metric("Queue Confidence", f"{result['queue_confidence']:.0%}")
        col3.metric("Priority", result["priority"])
        col4.metric("Response Time", f"{elapsed:.1f}s")

        st.divider()

        # Two columns: draft reply + retrieved articles
        left, right = st.columns([3, 2])

        with left:
            st.subheader("📝 Draft Reply")
            st.markdown(result["draft_reply"])

            # Action buttons
            st.divider()
            bcol1, bcol2, bcol3 = st.columns(3)
            bcol1.button("✅ Approve & Send", type="primary", use_container_width=True)
            bcol2.button("✏️ Edit", use_container_width=True)
            bcol3.button("🔺 Escalate", use_container_width=True)

        with right:
            st.subheader("📚 Retrieved Articles")
            for i, article in enumerate(result["retrieved_articles"]):
                with st.expander(f"[{i+1}] {article['title']} ({article['score']:.2f})"):
                    st.markdown(article["text"][:300] + "...")

            st.divider()
            st.subheader("🔍 Agent Reasoning")
            st.code(result["reasoning"], language=None)
            