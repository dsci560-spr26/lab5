"""
scraper.py - Reddit data collection script.

Supports two modes:
  1. Live scraping via PRAW (Reddit API) with pagination for >1000 posts.
  2. Importing from an existing CSV file.

Usage:
  python scraper.py --num_posts 5000 --subreddit nbadiscussion
  python scraper.py --import_csv reddit.csv
"""

import argparse
import csv
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from database import init_db, insert_raw_posts


def scrape_reddit(subreddit_name, num_posts,
                  client_id=None, client_secret=None,
                  user_agent="dsci560_lab5_scraper/1.0"):
    """Scrape Reddit posts using PRAW with automatic pagination.

    Handles the Reddit API limit of 1000 posts per request by cycling
    through multiple sort methods and search queries.

    Args:
        subreddit_name: name of the subreddit to scrape.
        num_posts: total number of posts to fetch.
        client_id: Reddit API client ID (or set REDDIT_CLIENT_ID env var).
        client_secret: Reddit API client secret (or set REDDIT_CLIENT_SECRET env var).
        user_agent: user agent string for the API.
    Returns:
        List of post dicts.
    """
    import praw

    reddit = praw.Reddit(
        client_id=client_id or os.environ.get("REDDIT_CLIENT_ID", "YOUR_CLIENT_ID"),
        client_secret=client_secret or os.environ.get("REDDIT_CLIENT_SECRET", "YOUR_CLIENT_SECRET"),
        user_agent=user_agent
    )

    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    seen_ids = set()
    batch_size = min(num_posts, 1000)  # Reddit API max per request

    print(f"[Scraper] Fetching {num_posts} posts from r/{subreddit_name}...")

    # Strategy 1: cycle through different sort methods
    sort_methods = ["hot", "new", "top", "rising"]
    time_filters = ["all", "year", "month", "week"]

    for method in sort_methods:
        if len(posts) >= num_posts:
            break

        print(f"[Scraper] Fetching via '{method}' sort...")
        try:
            if method == "top":
                for tf in time_filters:
                    if len(posts) >= num_posts:
                        break
                    listing = subreddit.top(limit=batch_size, time_filter=tf)
                    for submission in listing:
                        if submission.id not in seen_ids:
                            posts.append(_submission_to_dict(submission, subreddit_name))
                            seen_ids.add(submission.id)
                        if len(posts) >= num_posts:
                            break
                    time.sleep(1)
            elif method == "hot":
                for submission in subreddit.hot(limit=batch_size):
                    if submission.id not in seen_ids:
                        posts.append(_submission_to_dict(submission, subreddit_name))
                        seen_ids.add(submission.id)
                    if len(posts) >= num_posts:
                        break
            elif method == "new":
                for submission in subreddit.new(limit=batch_size):
                    if submission.id not in seen_ids:
                        posts.append(_submission_to_dict(submission, subreddit_name))
                        seen_ids.add(submission.id)
                    if len(posts) >= num_posts:
                        break
            elif method == "rising":
                for submission in subreddit.rising(limit=batch_size):
                    if submission.id not in seen_ids:
                        posts.append(_submission_to_dict(submission, subreddit_name))
                        seen_ids.add(submission.id)
                    if len(posts) >= num_posts:
                        break

            print(f"[Scraper] Collected {len(posts)} posts so far.")
            time.sleep(1)
        except Exception as e:
            print(f"[Scraper] Error fetching '{method}': {e}")
            time.sleep(2)

    # Strategy 2: use search queries to get additional posts beyond 1000
    if len(posts) < num_posts:
        print(f"[Scraper] Using search to fetch more posts ({len(posts)}/{num_posts})...")
        search_terms = [
            "discussion", "question", "news", "analysis",
            "opinion", "game", "player", "trade", "stats", "draft"
        ]
        for term in search_terms:
            if len(posts) >= num_posts:
                break
            try:
                for submission in subreddit.search(term, limit=batch_size, sort="relevance"):
                    if submission.id not in seen_ids:
                        posts.append(_submission_to_dict(submission, subreddit_name))
                        seen_ids.add(submission.id)
                    if len(posts) >= num_posts:
                        break
                time.sleep(1)
            except Exception as e:
                print(f"[Scraper] Search error for '{term}': {e}")
                time.sleep(2)

    print(f"[Scraper] Fetched {len(posts)} posts total.")
    return posts[:num_posts]


def _submission_to_dict(submission, subreddit_name):
    """Convert a PRAW submission object to a dict."""
    return {
        "id": submission.id,
        "subreddit": subreddit_name,
        "title": submission.title,
        "content": submission.selftext or "",
        "author": str(submission.author) if submission.author else "[deleted]",
        "num_comments": submission.num_comments,
        "created_utc": str(submission.created_utc),
    }


def import_from_csv(csv_path):
    """Import posts from an existing CSV file.

    Args:
        csv_path: path to the CSV file.
    Returns:
        List of post dicts.
    """
    posts = []
    print(f"[Scraper] Importing from {csv_path}...")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            num_comments = row.get("num_comments", "0").strip()
            try:
                num_comments = int(num_comments)
            except (ValueError, TypeError):
                num_comments = 0

            posts.append({
                "id": row.get("id", ""),
                "subreddit": row.get("subreddit", ""),
                "title": row.get("title", ""),
                "content": row.get("content", ""),
                "author": row.get("author", ""),
                "num_comments": num_comments,
                "created_utc": row.get("created_utc", ""),
            })
    print(f"[Scraper] Imported {len(posts)} posts from CSV.")
    return posts


def main():
    parser = argparse.ArgumentParser(description="Reddit Data Collection")
    parser.add_argument("--num_posts", type=int, default=100,
                        help="Number of posts to fetch from Reddit")
    parser.add_argument("--subreddit", type=str, default="nbadiscussion",
                        help="Subreddit to scrape")
    parser.add_argument("--import_csv", type=str,
                        help="Path to CSV file to import instead of live scraping")
    args = parser.parse_args()

    init_db()

    if args.import_csv:
        posts = import_from_csv(args.import_csv)
    else:
        posts = scrape_reddit(args.subreddit, args.num_posts)

    count = insert_raw_posts(posts)
    print(f"[Scraper] Inserted {count} new posts into database.")


if __name__ == "__main__":
    main()
