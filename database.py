"""
database.py - SQLite database setup and utility functions.

Manages two tables:
  - raw_posts: stores original scraped Reddit data
  - cleaned_posts: stores preprocessed data with keywords, embeddings, and cluster IDs
"""

import sqlite3
import json
import numpy as np
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reddit.db")


def get_connection():
    """Return a new SQLite connection with Row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they do not exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS raw_posts (
            id TEXT PRIMARY KEY,
            subreddit TEXT,
            title TEXT,
            content TEXT,
            author TEXT,
            num_comments INTEGER,
            created_utc TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cleaned_posts (
            id TEXT PRIMARY KEY,
            subreddit TEXT,
            title TEXT,
            cleaned_content TEXT,
            masked_author TEXT,
            num_comments INTEGER,
            created_at TEXT,
            keywords TEXT,
            embedding BLOB,
            cluster_id INTEGER
        )
    """)

    conn.commit()
    conn.close()


def insert_raw_posts(posts):
    """Insert raw posts into the database. Skips duplicates.

    Args:
        posts: list of dicts with keys matching raw_posts columns.
    Returns:
        Number of rows inserted.
    """
    conn = get_connection()
    cursor = conn.cursor()
    count = 0
    for post in posts:
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO raw_posts
                (id, subreddit, title, content, author, num_comments, created_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                post["id"], post["subreddit"], post["title"],
                post["content"], post["author"],
                post["num_comments"], post["created_utc"]
            ))
            count += cursor.rowcount
        except Exception:
            pass
    conn.commit()
    conn.close()
    return count


def insert_cleaned_posts(posts):
    """Insert or replace cleaned posts into the database.

    Args:
        posts: list of dicts with cleaned post data.
    """
    conn = get_connection()
    cursor = conn.cursor()
    for post in posts:
        embedding_blob = None
        if post.get("embedding") is not None:
            embedding_blob = np.array(post["embedding"], dtype=np.float32).tobytes()
        cursor.execute("""
            INSERT OR REPLACE INTO cleaned_posts
            (id, subreddit, title, cleaned_content, masked_author,
             num_comments, created_at, keywords, embedding, cluster_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            post["id"], post["subreddit"], post["title"],
            post["cleaned_content"], post["masked_author"],
            post["num_comments"], post["created_at"],
            json.dumps(post.get("keywords", [])),
            embedding_blob,
            post.get("cluster_id")
        ))
    conn.commit()
    conn.close()


def update_embeddings_and_clusters(updates):
    """Batch update embedding and cluster_id for posts.

    Args:
        updates: list of (embedding_list, cluster_id, post_id) tuples.
    """
    conn = get_connection()
    cursor = conn.cursor()
    for emb, cluster_id, post_id in updates:
        emb_blob = np.array(emb, dtype=np.float32).tobytes()
        cursor.execute("""
            UPDATE cleaned_posts SET embedding = ?, cluster_id = ? WHERE id = ?
        """, (emb_blob, int(cluster_id), post_id))
    conn.commit()
    conn.close()


def get_raw_posts():
    """Return all raw posts as a list of dicts."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM raw_posts")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_cleaned_posts():
    """Return all cleaned posts as a list of dicts."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM cleaned_posts")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_cleaned_posts_with_embeddings():
    """Return cleaned posts that have embeddings and cluster assignments."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM cleaned_posts WHERE embedding IS NOT NULL AND cluster_id IS NOT NULL"
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_unprocessed_raw_posts():
    """Return raw posts that have not yet been cleaned."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT r.* FROM raw_posts r
        LEFT JOIN cleaned_posts c ON r.id = c.id
        WHERE c.id IS NULL
    """)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows
