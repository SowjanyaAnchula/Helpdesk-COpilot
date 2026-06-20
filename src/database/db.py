"""
db.py — HelpDesk Copilot
Logs ticket processing results to PostgreSQL for Grafana monitoring.
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5433)),
    "dbname": os.getenv("POSTGRES_DB", "helpdesk"),
    "user": os.getenv("POSTGRES_USER", "helpdesk"),
    "password": os.getenv("POSTGRES_PASSWORD", "helpdesk123"),
}


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def log_ticket(ticket_text, queue, queue_confidence, priority,
               priority_confidence, escalated, response_time_ms,
               num_articles, draft_reply):
    """Log a processed ticket to PostgreSQL."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO ticket_logs
            (ticket_text, queue, queue_confidence, priority,
             priority_confidence, escalated, response_time_ms,
             num_articles_retrieved, draft_reply)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (ticket_text, queue, queue_confidence, priority,
              priority_confidence, escalated, response_time_ms,
              num_articles, draft_reply))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"DB logging failed (non-fatal): {e}")


def get_stats():
    """Get summary stats for the dashboard."""
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN escalated THEN 1 ELSE 0 END) as escalated,
                ROUND(AVG(response_time_ms)) as avg_response_ms,
                ROUND(AVG(queue_confidence)::numeric, 2) as avg_confidence
            FROM ticket_logs
        """)
        stats = cur.fetchone()
        cur.close()
        conn.close()
        return stats
    except Exception as e:
        print(f"DB stats failed: {e}")
        return None