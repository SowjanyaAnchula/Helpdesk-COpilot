"""
data_prep.py — HelpDesk Copilot
Final cleaning for KB before ChromaDB ingestion.
Fixes: escaped newlines, hex escapes, markdown nav noise.
"""

import os
import re
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def fix_escaped_newlines(text):
    """Convert literal \\n and \\t to real whitespace."""
    if isinstance(text, str):
        text = text.replace("\\n", "\n").replace("\\t", "\t")
    return text


def decode_hex_escapes(text):
    """Convert \\xNN hex escapes left behind by bytes repr."""
    if isinstance(text, str):
        try:
            if re.search(r"\\x[0-9a-fA-F]{2}", text):
                text = re.sub(
                    r"((?:\\x[0-9a-fA-F]{2})+)",
                    lambda m: bytes.fromhex(
                        m.group(0).replace("\\x", "")
                    ).decode("utf-8", errors="replace"),
                    text,
                )
        except Exception:
            pass
    return text


def strip_markdown_nav(text):
    """Remove scraped site-navigation noise."""
    # Nav-style markdown links
    nav_words = (
        r"(Login|Sign [Uu]p|Book a [Dd]emo|Our Customers|Resources|"
        r"Get [Ss]tarted|Request a [Dd]emo|Contact [Uu]s|Privacy|Legal|"
        r"DPA|Subprocessors|Extensions)"
    )
    text = re.sub(rf"\[{nav_words}\]\([^)]*\)", "", text)

    # Bare URL markdown links
    text = re.sub(r"\[https?://[^\]]+\]\([^)]+\)", "", text)

    # Heading underlines (=== or ---)
    text = re.sub(r"^[=\-]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Sidebar/nav list patterns like "- [Link text](/path)"
    text = re.sub(r"^\s*-\s*\[[^\]]+\]\(/[^)]+\)\s*$", "", text, flags=re.MULTILINE)

    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def strip_table_noise(text):
    """Clean up messy markdown table formatting."""
    # Remove lines that are only pipe separators: | --- | --- |
    text = re.sub(r"^\s*\|[\s\-:]+\|\s*$", "", text, flags=re.MULTILINE)
    return text


def clean_kb(df):
    df = df.copy()
    print(f"KB before cleaning: {len(df):,} chunks")

    df["text"] = df["text"].apply(fix_escaped_newlines)
    df["text"] = df["text"].apply(decode_hex_escapes)
    df["text"] = df["text"].apply(strip_markdown_nav)
    df["text"] = df["text"].apply(strip_table_noise)

    # Strip leading/trailing whitespace
    df["text"] = df["text"].str.strip()

    # Drop chunks that are now too short (nav-only or empty)
    before = len(df)
    df = df[df["text"].str.len() >= 30]
    print(f"Dropped {before - len(df)} near-empty chunks (< 30 chars)")

    # Drop exact duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    print(f"Dropped {before - len(df)} duplicate chunks")

    print(f"KB after cleaning: {len(df):,} chunks")
    return df


if __name__ == "__main__":
    # Load
    kb_path = os.path.join(DATA_DIR, "kb_clean.csv")
    kb_df = pd.read_csv(kb_path)

    # Clean
    kb_final = clean_kb(kb_df)

    # Show sample before/after
    print("\n── Sample cleaned chunk ──")
    sample = kb_final.iloc[0]
    print(f"ID:    {sample['id']}")
    print(f"Title: {sample['title']}")
    print(f"Text:  {sample['text'][:200]}...")

    # Save
    out_path = os.path.join(DATA_DIR, "kb_final.csv")
    kb_final.to_csv(out_path, index=False)
    print(f"\n✅ Saved {out_path} — {kb_final.shape}")