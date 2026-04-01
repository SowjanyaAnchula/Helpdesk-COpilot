"""
agent.py — HelpDesk Copilot
LangGraph agent: classify → retrieve → draft → escalate
"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import TypedDict, Literal
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from hybrid_search import load_resources, hybrid_search

load_dotenv()

# ── PATHS ──
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR  = os.path.join(BASE_DIR, "data")

# ── CONFIG ──
CONFIDENCE_THRESHOLD = 0.40   # below this → escalate
LLM_MODEL = "gpt-4o-mini"


# ── STATE ──
class TicketState(TypedDict):
    ticket_text: str
    queue: str
    queue_confidence: float
    priority: str
    priority_confidence: float
    retrieved_articles: list
    draft_reply: str
    escalated: bool
    reasoning: str


# ── LOAD MODELS ──
def load_classifier():
    """Load the best classifier (BGE + SVM)."""
    from sentence_transformers import SentenceTransformer

    path = os.path.join(MODEL_DIR, "bge_svm")

    with open(os.path.join(path, "queue_classifier.pkl"), "rb") as f:
        queue_data = pickle.load(f)

    with open(os.path.join(path, "priority_classifier.pkl"), "rb") as f:
        priority_data = pickle.load(f)

    with open(os.path.join(path, "config.pkl"), "rb") as f:
        config = pickle.load(f)

    embedder = SentenceTransformer(config["embedder"])

    return {
        "embedder": embedder,
        "queue_model": queue_data["model"],
        "queue_labels": queue_data["labels"],
        "priority_model": priority_data["model"],
        "priority_labels": priority_data["labels"],
    }


# Load everything at module level
print("Loading models...")
classifier = load_classifier()
bm25, collection, doc_ids, kb_df = load_resources()
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.3)
print("All models loaded ✅")


# ── NODE: CLASSIFY ──
def classify_ticket(state: TicketState) -> dict:
    """Predict queue and priority with confidence scores."""
    text = state["ticket_text"]

    # Embed the ticket
    embedding = classifier["embedder"].encode([text], convert_to_numpy=True)

    # Queue prediction
    queue_probs = classifier["queue_model"].predict_proba(embedding)[0]
    queue_idx   = np.argmax(queue_probs)
    queue_label = classifier["queue_labels"][queue_idx]
    queue_conf  = float(queue_probs[queue_idx])

    # Priority prediction
    priority_probs = classifier["priority_model"].predict_proba(embedding)[0]
    priority_idx   = np.argmax(priority_probs)
    priority_label = classifier["priority_labels"][priority_idx]
    priority_conf  = float(priority_probs[priority_idx])

    return {
        "queue": queue_label,
        "queue_confidence": queue_conf,
        "priority": priority_label,
        "priority_confidence": priority_conf,
        "reasoning": f"Classified as {queue_label} ({queue_conf:.0%}) | Priority: {priority_label} ({priority_conf:.0%})",
    }


# ── NODE: RETRIEVE ──
def retrieve_articles(state: TicketState) -> dict:
    """Search KB using hybrid search."""
    query = state["ticket_text"]
    results = hybrid_search(bm25, collection, query, doc_ids, top_k=5)

    articles = []
    for doc_id, score in results:
        row = kb_df[kb_df["id"] == doc_id]
        if len(row) > 0:
            row = row.iloc[0]
            articles.append({
                "id": doc_id,
                "title": row["title"],
                "text": row["text"][:500],
                "score": score,
            })

    return {
        "retrieved_articles": articles,
        "reasoning": state["reasoning"] + f"\nRetrieved {len(articles)} KB articles",
    }


# ── NODE: CHECK CONFIDENCE ──
def check_confidence(state: TicketState) -> Literal["draft", "escalate"]:
    """Route based on classifier confidence."""
    if state["queue_confidence"] < CONFIDENCE_THRESHOLD:
        return "escalate"
    return "draft"


# ── NODE: DRAFT REPLY ──
def draft_reply(state: TicketState) -> dict:
    """Generate a reply using LLM grounded in retrieved articles."""
    # Build context from retrieved articles
    context = "\n\n".join([
        f"Article: {a['title']}\n{a['text']}"
        for a in state["retrieved_articles"]
    ])

    system_prompt = """You are a helpful support agent for a SaaS company.
Write a professional, concise reply to the customer's ticket.
ONLY use information from the provided knowledge base articles.
If the articles don't contain enough information, say so honestly.
Keep the reply under 200 words."""

    user_prompt = f"""Customer Ticket:
{state['ticket_text']}

Queue: {state['queue']}
Priority: {state['priority']}

Knowledge Base Articles:
{context}

Write a helpful reply:"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    return {
        "draft_reply": response.content,
        "escalated": False,
        "reasoning": state["reasoning"] + "\nDraft reply generated by GPT-4o-mini",
    }


# ── NODE: ESCALATE ──
# ── NODE: ESCALATE + DRAFT ──
def escalate_and_draft(state: TicketState) -> dict:
    """Flag for human review but still generate a draft reply."""
    # Build context from retrieved articles
    context = "\n\n".join([
        f"Article: {a['title']}\n{a['text']}"
        for a in state["retrieved_articles"]
    ])

    system_prompt = """You are a helpful support agent for a SaaS company.
Write a professional, concise reply to the customer's ticket.
ONLY use information from the provided knowledge base articles.
If the articles don't contain enough information, say so honestly.
Keep the reply under 200 words.
NOTE: This reply will be reviewed by a human agent before sending."""

    user_prompt = f"""Customer Ticket:
{state['ticket_text']}

Queue: {state['queue']}
Priority: {state['priority']}

Knowledge Base Articles:
{context}

Write a helpful draft reply:"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    return {
        "draft_reply": f"[NEEDS REVIEW] {response.content}",
        "escalated": True,
        "reasoning": state["reasoning"] + f"\n⚠ ESCALATED — queue confidence {state['queue_confidence']:.0%} < {CONFIDENCE_THRESHOLD:.0%} threshold\nDraft reply still generated for human review",
    }


# ── BUILD GRAPH ──
def build_agent():
    workflow = StateGraph(TicketState)

    # Add nodes
    workflow.add_node("classify", classify_ticket)
    workflow.add_node("retrieve", retrieve_articles)
    workflow.add_node("draft", draft_reply)
    workflow.add_node("escalate", escalate_and_draft)

    # Define flow
    workflow.set_entry_point("classify")
    workflow.add_edge("classify", "retrieve")
    workflow.add_conditional_edges("retrieve", check_confidence, {
        "draft": "draft",
        "escalate": "escalate",
    })
    workflow.add_edge("draft", END)
    workflow.add_edge("escalate", END)

    return workflow.compile()


# ── MAIN ──
if __name__ == "__main__":
    agent = build_agent()

    # Test tickets
    test_tickets = [
        "How do I connect my Bitbucket account to DevRev for syncing issues?",
        "I want to set up an automation workflow that auto-closes tickets after 48 hours of no customer response.",
        "How can I create a snap-in command in DevRev?",
        "What is the process to install an app from the DevRev marketplace?",
    ]

    for ticket in test_tickets:
        print("\n" + "=" * 60)
        print(f"TICKET: {ticket[:80]}...")
        print("=" * 60)

        result = agent.invoke({
            "ticket_text": ticket,
            "queue": "",
            "queue_confidence": 0.0,
            "priority": "",
            "priority_confidence": 0.0,
            "retrieved_articles": [],
            "draft_reply": "",
            "escalated": False,
            "reasoning": "",
        })

        print(f"\n📋 Queue:      {result['queue']} ({result['queue_confidence']:.0%})")
        print(f"🔺 Priority:   {result['priority']} ({result['priority_confidence']:.0%})")
        print(f"🚨 Escalated:  {result['escalated']}")
        print(f"\n📝 Draft Reply:\n{result['draft_reply']}")
        print(f"\n🔍 Reasoning:\n{result['reasoning']}")