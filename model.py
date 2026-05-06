"""
model.py — CineMatch AI · Inference Layer
Handles recommendation logic and TMDB poster/metadata fetching.
"""

import pickle
import os
import sys
import requests
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

# ─── Locate model artefacts (supports old root layout + new model_data/) ──────
def _find_file(filename: str) -> str:
    """Search common locations for a pickle file, return path or raise."""
    candidates = [
        os.path.join("model_data", filename),   # new layout (build_model.py output)
        filename,                                 # legacy root layout
        os.path.join(os.path.dirname(__file__), "model_data", filename),
        os.path.join(os.path.dirname(__file__), filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"\n\n❌  Could not find '{filename}'.\n"
        "    Please run:  python build_model.py\n"
        "    This builds the similarity matrix and saves the required pickle files.\n"
    )


try:
    movies = pickle.load(open(_find_file("movies.pkl"), "rb"))
    similarity = pickle.load(open(_find_file("similarity.pkl"), "rb"))

except FileNotFoundError:

    print("⚠️ Model files not found. Building model...")

    import subprocess

    subprocess.run([sys.executable, "build_model.py"], check=True)
    
    movies = pickle.load(open(_find_file("movies.pkl"), "rb"))
    similarity = pickle.load(open(_find_file("similarity.pkl"), "rb"))
# Normalised title lookup for fast matching
_title_lower = movies["title"].str.lower()

API_KEY  = os.getenv("TMDB_API_KEY") or os.getenv("API_KEY")
_BASE    = "https://api.themoviedb.org/3"
_IMG_BASE = "https://image.tmdb.org/t/p/w500"

# ─── TMDB Helpers ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=512)
def _tmdb_search(title: str) -> dict | None:
    """Search TMDB and return the first result dict (cached)."""
    if not API_KEY:
        return None
    try:
        resp = requests.get(
            f"{_BASE}/search/movie",
            params={"api_key": API_KEY, "query": title},
            timeout=5,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return results[0] if results else None
    except Exception:
        return None


def fetch_poster(movie_title: str) -> str:
    """Return full poster URL or empty string."""
    result = _tmdb_search(movie_title)
    if result and result.get("poster_path"):
        return _IMG_BASE + result["poster_path"]
    return ""


def fetch_movie_meta(movie_title: str) -> dict:
    """Return a dict with poster, year, rating, overview."""
    result = _tmdb_search(movie_title) or {}
    poster = (_IMG_BASE + result["poster_path"]) if result.get("poster_path") else ""
    year   = (result.get("release_date") or "")[:4]
    rating = round(result.get("vote_average", 0), 1)
    overview = result.get("overview", "")
    return {
        "poster":   poster,
        "year":     year,
        "rating":   rating,
        "overview": overview,
    }


# ─── Recommendation Engine ────────────────────────────────────────────────────
def recommend(movie_title: str, n: int = 10) -> list[dict]:
    """
    Return up to *n* recommended movies as list of dicts:
      {title, poster, year, rating, overview, score}
    Returns [] if movie not found in dataset.
    """
    title_lower = movie_title.strip().lower()
    mask        = _title_lower == title_lower

    if not mask.any():
        return []

    idx       = movies[mask].index[0]
    distances = similarity[idx]

    top_indices = sorted(
        enumerate(distances), key=lambda x: x[1], reverse=True
    )[1: n + 1]

    recommendations = []
    for i, score in top_indices:
        title = movies.iloc[i]["title"]
        meta  = fetch_movie_meta(title)
        recommendations.append({
            "title":    title,
            "score":    round(float(score), 3),
            **meta,
        })
    return recommendations


def get_trending(n: int = 10) -> list[str]:
    """Return top-n movie titles ranked by popularity score (falls back to row order)."""
    if "popularity_score" in movies.columns:
        return movies.sort_values("popularity_score", ascending=False).head(n)["title"].tolist()
    # Fallback: first n rows (already sorted by TMDB popularity in the CSV)
    return movies.head(n)["title"].tolist()