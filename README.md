# DSCI-560 Lab 5: Reddit Data Collection, Preprocessing, Clustering & Automation

## Team Information

- **Team Name:** [Your Team Name]
- **Members:** [Member Names]

---

## Project Overview

This project implements a full pipeline for collecting Reddit forum data, preprocessing and cleaning the text, clustering posts by content similarity using Doc2Vec embeddings and K-Means, and automating the entire workflow with an interactive query interface.

### Data Source

- **Subreddit:** r/nbadiscussion (and related subreddits)
- **Total Posts Collected:** 5,004
- **Posts After Cleaning:** 4,970

---

## Project Structure

```
lab5/
├── venv/                  # Python 3.13 virtual environment
├── reddit.csv             # Raw scraped data (CSV)
├── reddit.db              # SQLite database (generated)
├── doc2vec.model          # Trained Doc2Vec model (generated)
├── requirements.txt       # Python dependencies
├── database.py            # Database setup and utility functions
├── scraper.py             # Reddit data collection (PRAW + CSV import)
├── preprocess.py          # Data cleaning and keyword extraction
├── clustering.py          # Doc2Vec embedding, K-Means clustering, visualization
├── pipeline.py            # Automation script with interactive query
├── README.md              # This file
└── output/                # Generated visualizations
    ├── clusters.png           # PCA scatter plot of clusters
    ├── cluster_keywords.png   # Top keywords per cluster
    └── query_result.png       # Query result visualization
```

---

## Setup Instructions

### Prerequisites

- Python 3.13 (tested; Python 3.10+ should also work)
- macOS / Linux

### 1. Create Virtual Environment

```bash
cd lab5
python3.13 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| praw | Reddit API wrapper for live scraping |
| pandas | Data manipulation |
| nltk | Natural language processing (tokenization, stopwords) |
| scikit-learn | TF-IDF vectorization, K-Means clustering, PCA |
| gensim | Doc2Vec document embedding |
| matplotlib | Data visualization |
| beautifulsoup4 | HTML tag removal |
| numpy | Numerical computation |
| wordcloud | Word cloud generation |

### 3. NLTK Data (Auto-downloaded)

The preprocessing script automatically downloads required NLTK data (`punkt`, `punkt_tab`, `stopwords`) on first run.

### 4. Reddit API Credentials (Optional - for live scraping)

To scrape Reddit live, set the following environment variables:

```bash
export REDDIT_CLIENT_ID="your_client_id"
export REDDIT_CLIENT_SECRET="your_client_secret"
```

You can obtain these by creating an app at https://www.reddit.com/prefs/apps/.

If no credentials are set, the system will use the pre-collected CSV data instead.

---

## How to Run

### Quick Start (Full Pipeline)

```bash
source venv/bin/activate

# Step 1: Import existing CSV data into SQLite
python scraper.py --import_csv reddit.csv

# Step 2: Preprocess the data
python preprocess.py

# Step 3: Run clustering and generate visualizations
python clustering.py

# Step 4: Start the automation pipeline (update every 5 minutes)
python pipeline.py 5
```

### Running Individual Scripts

#### Data Collection (`scraper.py`)

```bash
# Import from existing CSV file
python scraper.py --import_csv reddit.csv

# Live scrape from Reddit (requires API credentials)
python scraper.py --num_posts 500 --subreddit nbadiscussion

# Scrape 5000+ posts (handles API pagination automatically)
python scraper.py --num_posts 5000 --subreddit nbadiscussion
```

The scraper handles Reddit's API limit of 1,000 posts per request by:
1. Cycling through multiple sort methods (hot, new, top, rising).
2. Using multiple time filters for top posts (all, year, month, week).
3. Performing search queries with various keywords to fetch additional posts.

#### Data Preprocessing (`preprocess.py`)

```bash
python preprocess.py
```

This script:
- Removes HTML tags using BeautifulSoup.
- Strips URLs, markdown formatting, and special characters.
- Filters out promoted/ad content and deleted posts.
- Converts Unix timestamps to readable datetimes.
- Masks usernames with SHA-256 hashes for privacy (e.g., `user_a3f8b2c1`).
- Extracts top-10 keywords per post using TF-IDF scoring.

#### Clustering (`clustering.py`)

```bash
python clustering.py
```

This script:
- Trains a Doc2Vec model (PV-DM algorithm, 100-dimensional vectors, 20 epochs).
- Runs K-Means clustering with K=8 clusters.
- Extracts top keywords per cluster using TF-IDF.
- Identifies the 3 posts closest to each cluster centroid.
- Generates PCA visualizations saved to the `output/` directory.

#### Automation Pipeline (`pipeline.py`)

```bash
python pipeline.py <interval_in_minutes>
# Example: update every 5 minutes
python pipeline.py 5
```

This script:
1. Runs an initial full cycle (scrape -> preprocess -> cluster).
2. Starts a background thread that repeats the cycle at the specified interval.
3. Provides an interactive prompt (`>>>`) where you can type keywords or messages.
4. Finds the best-matching cluster for your query and displays the top 5 closest posts.
5. Generates a visualization highlighting the matched cluster (`output/query_result.png`).

Example interaction:
```
>>> Enter query (or 'quit'): lebron three pointer
[Query] Best matching cluster: 5
[Query] Cluster keywords: lebron, year, teams, ball, games, best, shot, three, league, time
[Query] Total posts in cluster: 232

Top 5 closest posts:
  [1] (distance: 3.2145)
      Title: The Celtics have performed better with Brown off the court...
      ...
```

---

## Database Schema

The project uses SQLite (`reddit.db`) with two tables:

### `raw_posts` — Original scraped data

| Column | Type | Description |
|--------|------|-------------|
| id | TEXT (PK) | Reddit post ID |
| subreddit | TEXT | Source subreddit |
| title | TEXT | Post title |
| content | TEXT | Post body text |
| author | TEXT | Original username |
| num_comments | INTEGER | Comment count |
| created_utc | TEXT | Unix timestamp |

### `cleaned_posts` — Preprocessed data with embeddings

| Column | Type | Description |
|--------|------|-------------|
| id | TEXT (PK) | Reddit post ID |
| subreddit | TEXT | Source subreddit |
| title | TEXT | Cleaned title |
| cleaned_content | TEXT | Cleaned and combined text |
| masked_author | TEXT | SHA-256 hashed username |
| num_comments | INTEGER | Comment count |
| created_at | TEXT | Human-readable timestamp |
| keywords | TEXT | JSON list of TF-IDF keywords |
| embedding | BLOB | Doc2Vec vector (float32 array) |
| cluster_id | INTEGER | Assigned K-Means cluster |

---

## Design Decisions

### Why SQLite over MySQL?

SQLite was chosen for portability and zero-configuration setup. It stores everything in a single file (`reddit.db`), making the project easy to share and reproduce without running a separate database server. The dataset size (~5,000 posts) is well within SQLite's capabilities.

### Why Doc2Vec for Embeddings?

Doc2Vec (PV-DM) was chosen because:
- It is specifically designed for document-level embeddings, unlike Word2Vec which only produces word vectors.
- It captures semantic meaning of entire posts, not just individual words.
- It is recommended in the assignment specification.
- The gensim implementation is well-tested and efficient.

Configuration: 100-dimensional vectors, 20 training epochs, minimum word count of 2.

### Why K-Means for Clustering?

K-Means is straightforward, scalable, and produces easily interpretable results. With K=8 clusters, the algorithm produces distinct topic groups (NBA, politics, finance, movies, etc.) that can be verified by examining cluster keywords and centroid-nearest posts.

### Preprocessing Choices

- **HTML removal:** BeautifulSoup safely parses and extracts text from any HTML fragments.
- **Username masking:** SHA-256 hashing preserves referential integrity (same user = same hash) while protecting privacy.
- **Keyword extraction:** TF-IDF with domain-specific stopwords produces meaningful, distinguishing keywords per post and per cluster.
- **Filtering:** Posts with `[deleted]`, `[removed]`, promoted content, or fewer than 10 characters are excluded.

### Handling Reddit API Limits (>1000 Posts)

The scraper handles the 1,000-post API limit by:
1. Iterating through sort methods: `hot`, `new`, `top` (with multiple time filters), `rising`.
2. Using search queries with domain-relevant keywords.
3. Deduplicating posts by ID across all methods.
4. Adding `time.sleep()` between requests to respect rate limits.

---

## Output Visualizations

- **`output/clusters.png`** — PCA 2D projection of all posts colored by cluster, with keyword annotations at cluster centers.
- **`output/cluster_keywords.png`** — Horizontal bar charts showing top 8 keywords per cluster with post counts.
- **`output/query_result.png`** — Generated during interactive queries, highlighting the matched cluster and query point.
