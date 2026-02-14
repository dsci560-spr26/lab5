"""
pipeline.py - Automation script for periodic data collection, processing, and interactive querying.

This script:
  1. Accepts an interval (in minutes) as a command-line argument.
  2. Periodically runs: scrape -> preprocess -> cluster.
  3. When idle, provides an interactive prompt for keyword/message queries.
  4. Finds the best-matching cluster for a query and displays results with visualization.

Usage:
  python pipeline.py <interval_in_minutes>
  Example: python pipeline.py 5
"""

import sys
import os
import time
import threading
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from database import init_db, get_cleaned_posts_with_embeddings, insert_raw_posts
from scraper import import_from_csv, scrape_reddit
from preprocess import preprocess
from clustering import run_clustering, VECTOR_SIZE, MODEL_PATH, OUTPUT_DIR

from gensim.models.doc2vec import Doc2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Pipeline:
    """Manages periodic data updates and interactive cluster queries."""

    def __init__(self, interval_minutes, subreddit="nbadiscussion"):
        self.interval = interval_minutes * 60  # seconds
        self.subreddit = subreddit
        self.running = True
        self.model = None
        self.kmeans = None
        self.embeddings = None
        self.labels = None
        self.posts = None
        self.cluster_keywords = {}

    def update_cycle(self):
        """Execute one full pipeline cycle: scrape -> preprocess -> cluster."""
        print("\n" + "=" * 60)
        print("[Pipeline] Starting update cycle...")
        print("=" * 60)

        # Step 1: Fetch new data from Reddit
        try:
            print("\n[Pipeline] Step 1/3: Fetching data...")
            posts = scrape_reddit(self.subreddit, 100)
            if posts:
                insert_raw_posts(posts)
                print(f"[Pipeline] Stored {len(posts)} posts.")
            else:
                print("[Pipeline] No new posts fetched.")
        except Exception as e:
            print(f"[Pipeline] Scraping skipped (will use existing data): {e}")

        # Step 2: Preprocess all raw data
        try:
            print("\n[Pipeline] Step 2/3: Preprocessing data...")
            preprocess()
        except Exception as e:
            print(f"[Pipeline] Preprocessing error: {e}")
            return

        # Step 3: Generate embeddings and cluster
        try:
            print("\n[Pipeline] Step 3/3: Clustering...")
            result = run_clustering()
            if result:
                self.embeddings, self.labels, self.kmeans, self.model, self.cluster_keywords = result
                self.posts = get_cleaned_posts_with_embeddings()
                print("[Pipeline] Update cycle complete.")
            else:
                print("[Pipeline] Clustering returned no results.")
        except Exception as e:
            print(f"[Pipeline] Clustering error: {e}")

    def load_existing_model(self):
        """Attempt to load a previously trained Doc2Vec model and cluster data.

        Returns True if successful, False otherwise.
        """
        if not os.path.exists(MODEL_PATH):
            return False

        self.model = Doc2Vec.load(MODEL_PATH)
        self.posts = get_cleaned_posts_with_embeddings()

        if not self.posts:
            return False

        self.embeddings = np.array([
            np.frombuffer(p["embedding"], dtype=np.float32)
            for p in self.posts
        ])
        self.labels = np.array([p["cluster_id"] for p in self.posts])

        # Refit KMeans on the existing embeddings
        n_clusters = len(set(self.labels))
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(self.embeddings)

        print(f"[Pipeline] Loaded existing model: {len(self.posts)} posts, {n_clusters} clusters.")
        return True

    def find_matching_cluster(self, query):
        """Find the cluster that best matches the input query and display results.

        Args:
            query: keyword or message string from the user.
        """
        if self.model is None or self.kmeans is None:
            print("[Pipeline] No model loaded yet. Please wait for the first update cycle.")
            return

        # Infer an embedding vector for the query
        query_tokens = query.lower().split()
        query_vector = self.model.infer_vector(query_tokens).reshape(1, -1)

        # Predict the closest cluster
        cluster_id = self.kmeans.predict(query_vector)[0]

        # Collect posts from the matched cluster
        cluster_posts = [
            (self.posts[i], self.embeddings[i])
            for i in range(len(self.posts))
            if self.labels[i] == cluster_id
        ]

        if not cluster_posts:
            print(f"[Query] No posts found in cluster {cluster_id}.")
            return

        # Sort by distance to the query vector
        distances = [
            np.linalg.norm(emb - query_vector.flatten())
            for _, emb in cluster_posts
        ]
        sorted_results = sorted(zip(cluster_posts, distances), key=lambda x: x[1])

        # Display results
        print(f"\n{'=' * 60}")
        print(f"[Query] Best matching cluster: {cluster_id}")
        if cluster_id in self.cluster_keywords:
            print(f"[Query] Cluster keywords: {', '.join(self.cluster_keywords[cluster_id][:10])}")
        print(f"[Query] Total posts in cluster: {len(cluster_posts)}")
        print(f"{'=' * 60}")

        print("\nTop 5 closest posts:")
        for i, ((post, _), dist) in enumerate(sorted_results[:5]):
            keywords = json.loads(post["keywords"]) if isinstance(post["keywords"], str) else post.get("keywords", [])
            print(f"\n  [{i + 1}] (distance: {dist:.4f})")
            print(f"      Title: {post['title'][:80]}")
            print(f"      Content: {post['cleaned_content'][:150]}...")
            print(f"      Keywords: {', '.join(keywords[:5])}")

        # Generate visualization for the query
        self._visualize_query_result(query_vector.flatten(), cluster_id)

    def _visualize_query_result(self, query_vector, target_cluster):
        """Create a PCA scatter plot highlighting the matched cluster and query point.

        Args:
            query_vector: inferred embedding for the query.
            target_cluster: the cluster ID that matched.
        """
        all_vectors = np.vstack([self.embeddings, query_vector.reshape(1, -1)])
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(all_vectors)

        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot each cluster; highlight the target cluster
        for cid in set(self.labels):
            mask = self.labels == cid
            alpha = 0.7 if cid == target_cluster else 0.15
            size = 20 if cid == target_cluster else 8
            ax.scatter(
                reduced[:-1][mask, 0], reduced[:-1][mask, 1],
                alpha=alpha, s=size, label=f"Cluster {cid}",
            )

        # Plot the query point as a red star
        ax.scatter(
            reduced[-1, 0], reduced[-1, 1],
            c="red", s=200, marker="*", zorder=5,
            label="Query", edgecolors="black",
        )

        ax.set_title(f"Query Result - Matching Cluster {target_cluster}")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "query_result.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"\n[Query] Visualization saved to {output_path}")

    def _background_updater(self):
        """Background thread that triggers update cycles at the configured interval."""
        while self.running:
            time.sleep(self.interval)
            if self.running:
                self.update_cycle()

    def run(self):
        """Main entry point: run initial update, start background updater, and enter query loop."""
        init_db()

        # Run the initial update cycle
        print("[Pipeline] Running initial data processing...")
        self.update_cycle()

        # Start the periodic background updater
        updater = threading.Thread(target=self._background_updater, daemon=True)
        updater.start()
        print(f"\n[Pipeline] Background updater running every {self.interval // 60} minutes.")
        print("[Pipeline] Enter a keyword or message to search, or 'quit' to exit.\n")

        # Interactive query loop
        while self.running:
            try:
                query = input(">>> Enter query (or 'quit'): ").strip()
                if query.lower() in ("quit", "exit", "q"):
                    self.running = False
                    print("[Pipeline] Shutting down...")
                    break
                if query:
                    self.find_matching_cluster(query)
            except (EOFError, KeyboardInterrupt):
                self.running = False
                print("\n[Pipeline] Shutting down...")
                break


def main():
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <interval_in_minutes>")
        print("Example: python pipeline.py 5")
        sys.exit(1)

    try:
        interval = int(sys.argv[1])
    except ValueError:
        print("Error: interval must be an integer (minutes).")
        sys.exit(1)

    pipeline = Pipeline(interval_minutes=interval)
    pipeline.run()


if __name__ == "__main__":
    main()
