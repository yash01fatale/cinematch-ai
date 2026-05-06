"""
build_model.py — CineMatch AI · Model Builder
Builds TF-IDF similarity matrix from TMDB 5000 dataset.
Run once before launching the app: python build_model.py
"""

import pandas as pd
import ast
import pickle
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Config ───────────────────────────────────────────────────────────────────
MOVIES_CSV  = "dataset/tmdb_5000_movies.csv"
CREDITS_CSV = "dataset/tmdb_5000_credits.csv"
OUTPUT_DIR  = "model_data"
MAX_FEATURES = 10_000   # richer vocabulary → better recs

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Load & Merge ─────────────────────────────────────────────────────────────
print("📂  Loading datasets …")
movies  = pd.read_csv(MOVIES_CSV)
credits = pd.read_csv(CREDITS_CSV)
movies  = movies.merge(credits, on="title")

print(f"✅  Loaded {len(movies):,} movies")

# ─── Helpers ──────────────────────────────────────────────────────────────────
def parse_list(text, n=None):
    """Parse JSON-like list strings → space-joined names."""
    try:
        items = [i["name"].replace(" ", "") for i in ast.literal_eval(text)]
        return " ".join(items[:n] if n else items)
    except Exception:
        return ""

def get_director(text):
    try:
        for person in ast.literal_eval(text):
            if person.get("job") == "Director":
                return person["name"].replace(" ", "")
    except Exception:
        pass
    return ""

def clean_text(text):
    """Lowercase and remove punctuation."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^a-z0-9 ]", "", text.lower())

# ─── Feature Engineering ──────────────────────────────────────────────────────
print("🔧  Engineering features …")

movies["genres"]   = movies["genres"].apply(parse_list)
movies["keywords"] = movies["keywords"].apply(parse_list)
movies["cast"]     = movies["cast"].apply(lambda x: parse_list(x, n=5))
movies["director"] = movies["crew"].apply(get_director)
movies["overview"] = movies["overview"].fillna("").apply(clean_text)

# Weight important fields by repeating them
movies["tags"] = (
    movies["overview"] + " "
    + (movies["genres"] + " ") * 2          # genres × 2
    + movies["keywords"] + " "
    + movies["cast"] + " "
    + (movies["director"] + " ") * 3        # director × 3
).str.lower().str.strip()

# Keep only what we need
movies = movies[["title", "tags", "vote_average", "vote_count", "release_date"]].dropna(subset=["tags"])

# Popularity score for trending section
movies["popularity_score"] = (
    movies["vote_average"].fillna(0) * movies["vote_count"].fillna(0)
).rank(pct=True)

print(f"✅  Feature matrix ready — {len(movies):,} movies")

# ─── TF-IDF + Cosine Similarity ───────────────────────────────────────────────
print("🧠  Fitting TF-IDF vectoriser …")
tfidf  = TfidfVectorizer(stop_words="english", max_features=MAX_FEATURES, ngram_range=(1, 2))
matrix = tfidf.fit_transform(movies["tags"])

print("🔄  Computing cosine similarity … (may take ~30 s)")
similarity = cosine_similarity(matrix)

# ─── Persist ──────────────────────────────────────────────────────────────────
# Save to model_data/ (canonical) AND root (backward-compat with old layout)
for dest_dir in [OUTPUT_DIR, "."]:
    pickle.dump(movies,     open(os.path.join(dest_dir, "movies.pkl"),     "wb"))
    pickle.dump(similarity, open(os.path.join(dest_dir, "similarity.pkl"), "wb"))

print(f"\n🎉  Model saved to '{OUTPUT_DIR}/' and project root")
print("    → movies.pkl")
print("    → similarity.pkl")
print("\nRun:  streamlit run app.py")