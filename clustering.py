"""
clustering.py - Document embedding, clustering, and visualization.

Steps:
  1. Train a Doc2Vec model on cleaned post texts to generate embeddings.
  2. Run K-Means clustering on the embedding vectors.
  3. Extract top keywords per cluster using TF-IDF.
  4. Identify messages closest to each cluster centroid.
  5. Visualize clusters with PCA and keyword charts.

Usage:
  python clustering.py
"""

import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from database import init_db, get_cleaned_posts, update_embeddings_and_clusters

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Configuration
VECTOR_SIZE = 100
NUM_CLUSTERS = 8
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "doc2vec.model")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def train_doc2vec(texts, vector_size=VECTOR_SIZE, epochs=20):
    """Train a Doc2Vec model on the input texts and return embeddings.

    Uses the PV-DM (Distributed Memory) algorithm.

    Args:
        texts: list of document strings.
        vector_size: dimensionality of the embedding vectors.
        epochs: number of training epochs.
    Returns:
        (embeddings_array, trained_model)
    """
    print("[Clustering] Preparing tagged documents for Doc2Vec...")
    tagged_docs = [
        TaggedDocument(words=text.lower().split(), tags=[str(i)])
        for i, text in enumerate(texts)
    ]

    print(f"[Clustering] Training Doc2Vec (vector_size={vector_size}, epochs={epochs})...")
    model = Doc2Vec(
        vector_size=vector_size,
        min_count=2,
        epochs=epochs,
        workers=4,
        dm=1,  # PV-DM (Distributed Memory)
    )
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(MODEL_PATH)

    embeddings = np.array([model.dv[str(i)] for i in range(len(texts))])
    print(f"[Clustering] Generated {len(embeddings)} embeddings of dimension {vector_size}.")
    return embeddings, model


def cluster_messages(embeddings, num_clusters=NUM_CLUSTERS):
    """Run K-Means clustering on embedding vectors.

    Args:
        embeddings: numpy array of shape (n_docs, vector_size).
        num_clusters: number of clusters (K).
    Returns:
        (labels_array, fitted_kmeans_model)
    """
    print(f"[Clustering] Running K-Means with K={num_clusters}...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(embeddings)

    cluster_counts = Counter(labels)
    for cid, count in sorted(cluster_counts.items()):
        print(f"  Cluster {cid}: {count} posts")

    return labels, kmeans


def extract_cluster_keywords(texts, labels, num_clusters, top_n=10):
    """Extract top keywords for each cluster using TF-IDF.

    Args:
        texts: list of document strings.
        labels: cluster label for each document.
        num_clusters: total number of clusters.
        top_n: number of keywords per cluster.
    Returns:
        Dict mapping cluster_id to list of keyword strings.
    """
    try:
        sw = list(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        sw = list(stopwords.words("english"))

    sw.extend([
        "like", "would", "could", "also", "really", "think", "get", "got",
        "one", "much", "even", "going", "still", "way", "well", "know",
        "just", "dont", "doesnt", "didnt", "cant", "im", "game", "team",
        "player", "players", "season", "nba", "thats", "hes", "theyre",
    ])

    cluster_keywords = {}
    for cid in range(num_clusters):
        cluster_texts = [texts[i] for i in range(len(texts)) if labels[i] == cid]
        if not cluster_texts:
            cluster_keywords[cid] = []
            continue

        tfidf = TfidfVectorizer(
            stop_words=sw,
            max_features=1000,
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
        )
        try:
            matrix = tfidf.fit_transform(cluster_texts)
            feature_names = tfidf.get_feature_names_out()
            avg_scores = matrix.mean(axis=0).A1
            top_indices = avg_scores.argsort()[-top_n:][::-1]
            cluster_keywords[cid] = [feature_names[i] for i in top_indices]
        except ValueError:
            cluster_keywords[cid] = []

    return cluster_keywords


def find_closest_to_centroid(embeddings, labels, kmeans, texts, post_ids, n=3):
    """Find the n messages closest to each cluster centroid.

    Args:
        embeddings: numpy array of embeddings.
        labels: cluster labels.
        kmeans: fitted KMeans model.
        texts: document texts.
        post_ids: post IDs corresponding to each document.
        n: number of closest messages to return per cluster.
    Returns:
        Dict mapping cluster_id to list of {id, text, distance} dicts.
    """
    results = {}
    for cid in range(kmeans.n_clusters):
        mask = labels == cid
        if not mask.any():
            results[cid] = []
            continue

        cluster_embeddings = embeddings[mask]
        cluster_indices = np.where(mask)[0]
        centroid = kmeans.cluster_centers_[cid]

        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        closest_idx = distances.argsort()[:n]

        results[cid] = [
            {
                "id": post_ids[cluster_indices[i]],
                "text": texts[cluster_indices[i]][:200],
                "distance": float(distances[i]),
            }
            for i in closest_idx
        ]
    return results


def visualize_clusters(embeddings, labels, cluster_keywords, output_dir=OUTPUT_DIR):
    """Generate PCA scatter plot and keyword bar charts for each cluster.

    Args:
        embeddings: numpy array of embeddings.
        labels: cluster label array.
        cluster_keywords: dict of cluster_id -> keyword list.
        output_dir: directory to save output images.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_clusters = max(labels) + 1

    # --- PCA scatter plot ---
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(14, 10))
    scatter = ax.scatter(
        reduced[:, 0], reduced[:, 1],
        c=labels, cmap="tab10", alpha=0.5, s=10,
    )

    # Annotate cluster centers with top keywords
    for cid in range(num_clusters):
        mask = labels == cid
        if not mask.any():
            continue
        cx, cy = reduced[mask].mean(axis=0)
        kw_text = ", ".join(cluster_keywords.get(cid, [])[:5])
        ax.annotate(
            f"C{cid}: {kw_text}", (cx, cy),
            fontsize=7, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    ax.set_title("Reddit Post Clusters (PCA Visualization)")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    plt.colorbar(scatter, label="Cluster ID")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "clusters.png"), dpi=150)
    plt.close()
    print(f"[Clustering] Scatter plot saved to {output_dir}/clusters.png")

    # --- Keyword bar charts per cluster ---
    ncols = min(4, num_clusters)
    nrows = (num_clusters + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if num_clusters == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for cid in range(num_clusters):
        kws = cluster_keywords.get(cid, [])[:8]
        if kws and cid < len(axes):
            y_pos = range(len(kws))
            axes[cid].barh(
                y_pos, list(range(len(kws), 0, -1)),
                tick_label=kws, color=plt.cm.tab10(cid / 10),
            )
            axes[cid].set_title(f"Cluster {cid} ({(labels == cid).sum()} posts)")
            axes[cid].invert_yaxis()

    # Hide unused subplot axes
    for i in range(num_clusters, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Top Keywords per Cluster", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cluster_keywords.png"), dpi=150)
    plt.close()
    print(f"[Clustering] Keyword chart saved to {output_dir}/cluster_keywords.png")


def run_clustering(num_clusters=NUM_CLUSTERS):
    """Execute the full clustering pipeline.

    Returns:
        Tuple of (embeddings, labels, kmeans, model, cluster_keywords) or None on failure.
    """
    print("[Clustering] Loading cleaned posts from database...")
    posts = get_cleaned_posts()

    if not posts:
        print("[Clustering] No cleaned posts found. Run preprocess.py first.")
        return None

    texts = [p["cleaned_content"] for p in posts]
    post_ids = [p["id"] for p in posts]

    print(f"[Clustering] Processing {len(texts)} posts...")

    # Step 1: Generate Doc2Vec embeddings
    embeddings, model = train_doc2vec(texts)

    # Step 2: K-Means clustering
    labels, kmeans = cluster_messages(embeddings, num_clusters)

    # Step 3: Extract keywords per cluster
    cluster_keywords = extract_cluster_keywords(texts, labels, num_clusters)
    print("\n[Clustering] Cluster Keywords:")
    for cid, kws in cluster_keywords.items():
        print(f"  Cluster {cid}: {', '.join(kws)}")

    # Step 4: Find representative messages (closest to centroid)
    closest = find_closest_to_centroid(embeddings, labels, kmeans, texts, post_ids)
    print("\n[Clustering] Messages Closest to Centroid:")
    for cid, msgs in closest.items():
        print(f"\n  --- Cluster {cid} ---")
        for m in msgs:
            print(f"    [{m['id']}] {m['text'][:100]}...")

    # Step 5: Update database with embeddings and cluster assignments
    print("\n[Clustering] Saving embeddings and cluster IDs to database...")
    updates = [
        (embeddings[i].tolist(), int(labels[i]), post_ids[i])
        for i in range(len(post_ids))
    ]
    update_embeddings_and_clusters(updates)

    # Step 6: Visualize
    visualize_clusters(embeddings, labels, cluster_keywords)

    print("\n[Clustering] Done!")
    return embeddings, labels, kmeans, model, cluster_keywords


if __name__ == "__main__":
    init_db()
    run_clustering()
