"""
preprocess.py - Data preprocessing pipeline.

Steps:
  1. Remove HTML tags and special characters
  2. Filter out promoted/ad/irrelevant content
  3. Convert timestamps to readable format
  4. Mask usernames for privacy (SHA-256 hash)
  5. Extract keywords using TF-IDF

Usage:
  python preprocess.py
"""

import re
import json
import hashlib
import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from database import init_db, get_raw_posts, insert_cleaned_posts

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import warnings
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


def ensure_nltk_data():
    """Download required NLTK data if not already present."""
    for resource in ["punkt", "punkt_tab", "stopwords"]:
        try:
            if "punkt" in resource:
                nltk.data.find(f"tokenizers/{resource}")
            else:
                nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def remove_html_tags(text):
    """Strip HTML tags using BeautifulSoup."""
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ")


def remove_special_characters(text):
    """Remove URLs, markdown formatting, and non-standard characters."""
    if not text:
        return ""
    # Remove URLs
    text = re.sub(r"http[s]?://\S+", "", text)
    # Remove markdown links [text](url)
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)
    # Remove markdown formatting characters
    text = re.sub(r"[*_~`#>|]", "", text)
    # Keep only letters, numbers, basic punctuation, and spaces
    text = re.sub(r"[^\w\s.,!?;:'\-]", " ", text)
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_irrelevant(post):
    """Check if a post is promoted, deleted, or otherwise irrelevant.

    Returns True if the post should be filtered out.
    """
    content = (post.get("content", "") + " " + post.get("title", "")).lower()

    # Filter promoted / ad content
    irrelevant_patterns = [
        r"\bpromoted\b", r"\badvertisement\b", r"\bsponsored\b",
        r"^\[deleted\]$", r"^\[removed\]$",
    ]
    for pattern in irrelevant_patterns:
        if re.search(pattern, content):
            return True

    # Filter posts with very short or empty content
    if len(content.strip()) < 10:
        return True

    return False


def mask_username(username):
    """Replace username with a hashed identifier for privacy."""
    if not username or username in ("[deleted]", "[removed]", "AutoModerator"):
        return "anonymous"
    return "user_" + hashlib.sha256(username.encode()).hexdigest()[:8]


def convert_timestamp(utc_str):
    """Convert a Unix UTC timestamp string to a readable datetime string."""
    try:
        ts = float(utc_str)
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError, OSError):
        return utc_str


def extract_keywords(texts, top_n=10):
    """Extract top-N keywords for each document using TF-IDF scoring.

    Args:
        texts: list of document strings.
        top_n: number of keywords to extract per document.
    Returns:
        List of keyword lists, one per document.
    """
    if not texts:
        return []

    stop_words = list(stopwords.words("english"))
    # Add domain-specific and common low-value stop words
    stop_words.extend([
        "like", "would", "could", "also", "really", "think", "get", "got",
        "one", "much", "even", "going", "still", "way", "well", "know",
        "said", "say", "just", "dont", "doesnt", "didnt", "cant", "im",
        "thats", "hes", "shes", "theyre", "its", "ive", "youre",
    ])

    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words=stop_words,
        min_df=2,
        max_df=0.95,
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",  # words with 3+ letters only
    )

    try:
        tfidf_matrix = tfidf.fit_transform(texts)
        feature_names = tfidf.get_feature_names_out()

        keywords_per_doc = []
        for i in range(tfidf_matrix.shape[0]):
            row = tfidf_matrix[i].toarray().flatten()
            top_indices = row.argsort()[-top_n:][::-1]
            keywords = [feature_names[idx] for idx in top_indices if row[idx] > 0]
            keywords_per_doc.append(keywords)

        return keywords_per_doc
    except ValueError:
        return [[] for _ in texts]


def preprocess():
    """Run the full preprocessing pipeline on raw posts in the database.

    Returns:
        List of cleaned post dicts.
    """
    ensure_nltk_data()

    print("[Preprocess] Loading raw posts from database...")
    raw_posts = get_raw_posts()
    print(f"[Preprocess] Loaded {len(raw_posts)} raw posts.")

    # Filter out irrelevant content
    filtered = [p for p in raw_posts if not is_irrelevant(p)]
    removed = len(raw_posts) - len(filtered)
    print(f"[Preprocess] {len(filtered)} posts after filtering ({removed} removed).")

    # Clean each post
    print("[Preprocess] Cleaning text...")
    cleaned_posts = []
    texts_for_keywords = []

    for post in filtered:
        title = remove_special_characters(remove_html_tags(post["title"]))
        content = remove_special_characters(remove_html_tags(post["content"]))

        combined_text = f"{title}. {content}".strip()
        if len(combined_text) < 15:
            continue

        cleaned_posts.append({
            "id": post["id"],
            "subreddit": post["subreddit"],
            "title": title,
            "cleaned_content": combined_text,
            "masked_author": mask_username(post["author"]),
            "num_comments": post["num_comments"],
            "created_at": convert_timestamp(post["created_utc"]),
            "keywords": [],
            "embedding": None,
            "cluster_id": None,
        })
        texts_for_keywords.append(combined_text)

    print(f"[Preprocess] {len(cleaned_posts)} posts after text cleaning.")

    # Extract keywords via TF-IDF
    print("[Preprocess] Extracting keywords...")
    all_keywords = extract_keywords(texts_for_keywords)
    for i, keywords in enumerate(all_keywords):
        cleaned_posts[i]["keywords"] = keywords

    # Store cleaned data in the database
    print("[Preprocess] Storing cleaned posts in database...")
    insert_cleaned_posts(cleaned_posts)
    print(f"[Preprocess] Done. {len(cleaned_posts)} cleaned posts stored.")

    return cleaned_posts


if __name__ == "__main__":
    init_db()
    preprocess()
