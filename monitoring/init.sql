-- HelpDesk Copilot monitoring tables

CREATE TABLE IF NOT EXISTS ticket_logs (
    id SERIAL PRIMARY KEY,
    ticket_text TEXT NOT NULL,
    queue VARCHAR(100),
    queue_confidence FLOAT,
    priority VARCHAR(50),
    priority_confidence FLOAT,
    escalated BOOLEAN DEFAULT FALSE,
    response_time_ms INTEGER,
    num_articles_retrieved INTEGER,
    draft_reply TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS daily_stats (
    id SERIAL PRIMARY KEY,
    date DATE DEFAULT CURRENT_DATE,
    total_tickets INTEGER DEFAULT 0,
    auto_replied INTEGER DEFAULT 0,
    escalated INTEGER DEFAULT 0,
    avg_response_time_ms FLOAT,
    avg_queue_confidence FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);