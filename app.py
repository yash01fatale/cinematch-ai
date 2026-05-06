"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        CineMatch AI  ·  app.py                             ║
║                  Main Streamlit frontend — startup showcase                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  FOUNDER  : Yash Yuvraj Fatale                                              ║
║  PRODUCT  : CineMatch AI — NLP-powered movie recommendation engine          ║
║  STACK    : Streamlit · Scikit-learn · TMDB API · Python 3.10+              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  QUICK-START FOR NEW CONTRIBUTORS / REVIEWERS                               ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  1. pip install -r requirements.txt                                         ║
║  2. Create a .env file:  TMDB_API_KEY=your_key_here                         ║
║     → Get a free key at https://www.themoviedb.org/settings/api             ║
║  3. python build_model.py          ← run ONCE to build the AI model         ║
║  4. streamlit run app.py           ← launch the app                         ║
║                                                                             ║
║  FILES IN THIS PROJECT                                                      ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  app.py          → THIS FILE. Streamlit UI + layout + styling               ║
║  model.py        → Recommendation logic + TMDB API calls                    ║
║  build_model.py  → One-time script: builds TF-IDF matrix → .pkl files      ║
║  dataset/        → tmdb_5000_movies.csv + tmdb_5000_credits.csv             ║
║  model_data/     → Auto-created by build_model.py (movies.pkl etc.)         ║
║  .env            → Your secret TMDB_API_KEY goes here (never commit!)       ║
║                                                                             ║
║  WHAT TO CUSTOMISE                                                          ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  • FOUNDER_INFO dict below      → your name, bio, GitHub, LinkedIn          ║
║  • HERO section (search ①)      → tagline + subtitle text                   ║
║  • STATS BAR (search ④)         → update numbers if you retrain             ║
║  • Brand accent colour          → find-replace  #d4a05a  with your hex      ║
║  • Trending count               → change get_trending(10) → more/fewer      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import streamlit as st
import pandas as pd
from model import recommend, fetch_movie_meta, get_trending


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  ← must be the FIRST Streamlit call — do not move this block
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CineMatch AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ══════════════════════════════════════════════════════════════════════════════
# FOUNDER / TEAM INFO  ← UPDATE THIS SECTION with your real details
# ══════════════════════════════════════════════════════════════════════════════
FOUNDER_INFO = {
    "name":       "Yash Yuvraj Fatale",
    "role":       "Founder & Lead Engineer",
    "bio": (
        "Engineering student passionate about building AI products that solve "
        "real problems. CineMatch is applied NLP in action — turning raw movie "
        "metadata into a personalised discovery experience anyone can use."
    ),
    # Replace with your real URLs  (or set to "" to hide that link)
    "github":     "https://github.com/yashfatale",
    "linkedin":   "https://linkedin.com/in/yashfatale",
    "email":      "yash@cinematch.ai",
    "avatar_url": "",   # direct URL to photo; leave "" to show golden initials
}

# Add co-founders / contributors here (or leave list empty)
TEAM = [
    # {
    #     "name":   "Jane Doe",
    #     "role":   "ML Engineer",
    #     "github": "https://github.com/janedoe",
    # },
]


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS  — @st.cache_data means they run only ONCE per session
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_titles() -> list[str]:
    """
    Loads every movie title from the TMDB CSV into the search dropdown.
    If you move the CSV, update the path string below.
    """
    df = pd.read_csv("dataset/tmdb_5000_movies.csv")
    return sorted(df["title"].dropna().unique().tolist())


@st.cache_data
def load_trending() -> list[dict]:
    """
    Fetches the top-10 highest-rated films and their TMDB posters.
    To show more films, change get_trending(10) → get_trending(15).
    """
    titles = get_trending(10)
    return [{"title": t, **fetch_movie_meta(t)} for t in titles]


all_titles = load_titles()   # loaded once; used by the selectbox


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — cinematic dark-luxury aesthetic
# ══════════════════════════════════════════════════════════════════════════════
# Brand accent : #d4a05a  (warm gold — find-replace to re-theme)
# Background   : #080a0f  (near-black)
# Card surface : #0d1018
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: #080a0f !important;
    color: #e8e0d5 !important;
    font-family: 'DM Sans', sans-serif;
}

/* Hide Streamlit default chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

.block-container {
    max-width: 1380px !important;
    padding: 0 2.5rem 5rem !important;
}

/* ── HERO ── */
.hero {
    position: relative;
    text-align: center;
    padding: 5.5rem 1rem 3.5rem;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 90% 55% at 50% -5%, rgba(212,160,90,0.13) 0%, transparent 68%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(100,60,200,0.04) 0%, transparent 60%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #d4a05a;
    border: 1px solid rgba(212,160,90,0.3);
    padding: 0.32rem 1.1rem;
    border-radius: 100px;
    margin-bottom: 1.8rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(3.2rem, 8vw, 6.5rem);
    font-weight: 900;
    line-height: 1.03;
    letter-spacing: -0.025em;
    color: #f5ede0;
    margin-bottom: 1.1rem;
}
.hero-title em { color: #d4a05a; font-style: italic; }
.hero-sub {
    font-size: 1.05rem;
    font-weight: 300;
    color: #7a7068;
    max-width: 530px;
    margin: 0 auto 0.5rem;
    line-height: 1.75;
}

/* ── SELECTBOX ── */
[data-testid="stSelectbox"] > div {
    background: rgba(255,255,255,0.035) !important;
    border: 1px solid rgba(212,160,90,0.22) !important;
    border-radius: 12px !important;
}
[data-testid="stSelectbox"] > div:focus-within {
    border-color: rgba(212,160,90,0.55) !important;
    box-shadow: 0 0 0 3px rgba(212,160,90,0.08) !important;
}
[data-testid="stSelectbox"] label { display: none !important; }

/* ── BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #d4a05a 0%, #a87230 100%) !important;
    color: #080a0f !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 0.8rem 2.2rem !important;
    border: none !important;
    border-radius: 10px !important;
    transition: opacity 0.18s, transform 0.15s, box-shadow 0.18s !important;
    box-shadow: 0 4px 20px rgba(212,160,90,0.28) !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(212,160,90,0.38) !important;
}

/* ── SLIDER ── */
[data-testid="stSlider"] label {
    color: #6a6058 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}
[data-testid="stSlider"] input[type="range"] { accent-color: #d4a05a; }

/* ── STATS BAR ── */
.stats-bar {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    margin: 1.6rem 0 2.4rem;
    background: rgba(255,255,255,0.022);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    overflow: hidden;
}
.stat {
    flex: 1; min-width: 110px;
    text-align: center;
    padding: 1.4rem 1rem;
    border-right: 1px solid rgba(255,255,255,0.05);
    transition: background 0.2s;
}
.stat:last-child { border-right: none; }
.stat:hover { background: rgba(212,160,90,0.04); }
.stat-val {
    font-family: 'Playfair Display', serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: #d4a05a;
    line-height: 1;
}
.stat-lbl {
    font-size: 0.68rem;
    color: #4a4440;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.35rem;
}

/* ── SECTION HEADER ── */
.section-head {
    display: flex;
    align-items: baseline;
    gap: 0.8rem;
    margin: 3.8rem 0 1.6rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding-bottom: 0.9rem;
}
.section-head h2 {
    font-family: 'Playfair Display', serif;
    font-size: 1.65rem;
    font-weight: 700;
    color: #f0e8de;
    margin: 0;
}
.section-head h2 em { color: #d4a05a; font-style: italic; }
.section-tag {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #d4a05a;
    background: rgba(212,160,90,0.1);
    padding: 0.22rem 0.65rem;
    border-radius: 6px;
    border: 1px solid rgba(212,160,90,0.2);
}

/* ── MOVIE CARD ── */
.movie-card {
    background: #0d1018;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    overflow: hidden;
    transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
    height: 100%;
}
.movie-card:hover {
    transform: translateY(-7px) scale(1.01);
    border-color: rgba(212,160,90,0.45);
    box-shadow: 0 20px 50px rgba(0,0,0,0.55), 0 0 0 1px rgba(212,160,90,0.12);
}
.movie-card img {
    width: 100%; aspect-ratio: 2/3;
    object-fit: cover; display: block;
    background: #151820;
}
.card-body { padding: 0.85rem 0.95rem 1rem; }
.card-title {
    font-size: 0.87rem; font-weight: 500;
    color: #ddd5c8; line-height: 1.35;
    margin-bottom: 0.5rem;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.card-meta { display: flex; align-items: center; gap: 0.45rem; flex-wrap: wrap; }
.card-year   { font-size: 0.7rem; color: #4a4440; }
.card-rating { font-size: 0.7rem; font-weight: 500; color: #d4a05a; }
.card-score {
    margin-left: auto;
    font-size: 0.66rem; font-weight: 600;
    color: rgba(212,160,90,0.7);
    background: rgba(212,160,90,0.09);
    padding: 0.14rem 0.45rem;
    border-radius: 5px;
}

/* ── HOW-IT-WORKS CARD ── */
.how-card {
    background: rgba(255,255,255,0.022);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1.5rem 1.3rem;
    height: 100%;
    transition: border-color 0.2s, transform 0.2s;
}
.how-card:hover { border-color: rgba(212,160,90,0.3); transform: translateY(-3px); }
.how-icon  { font-size: 2rem; margin-bottom: 0.8rem; }
.how-title { font-weight: 600; font-size: 0.95rem; color: #e8e0d5; margin-bottom: 0.45rem; }
.how-body  { font-size: 0.8rem; color: #4a4440; line-height: 1.65; }

/* ── ONBOARDING BANNER ── */
.onboard-banner {
    background: linear-gradient(135deg, rgba(212,160,90,0.07) 0%, rgba(212,100,50,0.04) 100%);
    border: 1px solid rgba(212,160,90,0.18);
    border-radius: 14px;
    padding: 1.3rem 1.6rem;
    margin: 0 0 2rem;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    flex-wrap: wrap;
}
.onboard-icon  { font-size: 1.4rem; flex-shrink: 0; margin-top: 0.1rem; }
.onboard-title { font-weight: 600; font-size: 0.92rem; color: #d4a05a; margin-bottom: 0.25rem; }
.onboard-text  { font-size: 0.8rem; color: #5a5450; line-height: 1.65; }
.step-pill {
    display: inline-block;
    background: rgba(212,160,90,0.12);
    color: #d4a05a;
    font-size: 0.68rem; font-weight: 700;
    padding: 0.1rem 0.45rem;
    border-radius: 4px;
    margin-right: 0.3rem;
}

/* ── TEAM SECTION ── */
.team-section {
    margin: 4rem 0 0;
    padding: 2.5rem;
    background: rgba(255,255,255,0.018);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px;
    position: relative;
    overflow: hidden;
}
.team-section::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(212,160,90,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.founder-avatar {
    width: 72px; height: 72px;
    border-radius: 50%;
    background: linear-gradient(135deg, #d4a05a, #6a3d10);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem; font-weight: 700;
    color: #080a0f; flex-shrink: 0;
    border: 2px solid rgba(212,160,90,0.35);
}
.founder-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem; font-weight: 700;
    color: #f0e8de;
}
.founder-role {
    font-size: 0.72rem; font-weight: 500;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: #d4a05a; margin-top: 0.1rem;
}
.founder-bio {
    font-size: 0.87rem; color: #5a5450;
    line-height: 1.7; margin-top: 0.6rem;
    max-width: 560px;
}
.social-link {
    display: inline-flex; align-items: center; gap: 0.4rem;
    font-size: 0.78rem; font-weight: 500;
    color: #6a6058; text-decoration: none;
    padding: 0.3rem 0.75rem;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    transition: color 0.18s, border-color 0.18s;
}
.social-link:hover { color: #d4a05a; border-color: rgba(212,160,90,0.3); }

/* ── DIVIDER ── */
.divider { border: none; border-top: 1px solid rgba(255,255,255,0.055); margin: 3.5rem 0; }

/* ── FOOTER ── */
.footer { text-align: center; padding: 3rem 1rem 1.5rem; }
.footer-logo {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem; font-weight: 700;
    color: #d4a05a; letter-spacing: 0.04em;
    margin-bottom: 0.5rem;
}
.footer-copy { font-size: 0.72rem; color: #2e2a26; letter-spacing: 0.05em; line-height: 1.8; }
.footer-copy a { color: #4a4440; text-decoration: none; }
.footer-copy a:hover { color: #d4a05a; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

FALLBACK_POSTER = "https://placehold.co/300x450/0d1018/1e2028?text=🎬"


def render_card(movie: dict, show_score: bool = False) -> str:
    """
    Returns the HTML string for one movie card.

    Args:
        movie      : dict with keys → title, poster, year, rating, score
        show_score : True on the recommendations grid (shows "XX% match" badge)
    """
    poster  = movie.get("poster") or FALLBACK_POSTER
    title   = movie.get("title", "Unknown")
    year    = movie.get("year", "")
    rating  = movie.get("rating", 0)
    score   = movie.get("score")

    year_html   = f'<span class="card-year">{year}</span>'   if year   else ""
    rating_html = f'<span class="card-rating">★ {rating}</span>' if rating else ""
    score_html  = (
        f'<span class="card-score">{score:.0%} match</span>'
        if show_score and score else ""
    )

    return f"""
<div class="movie-card">
  <img src="{poster}" alt="{title}" loading="lazy"
       onerror="this.onerror=null;this.src='{FALLBACK_POSTER}'">
  <div class="card-body">
    <div class="card-title">{title}</div>
    <div class="card-meta">{year_html}{rating_html}{score_html}</div>
  </div>
</div>"""


def render_grid(movies_list: list[dict], cols: int = 5, show_score: bool = False):
    """
    Lays out a row-based grid of movie cards.

    Args:
        movies_list : output of recommend() or load_trending()
        cols        : cards per row (5 fits nicely on wide layout)
        show_score  : pass True only for the AI recommendations section
    """
    for i in range(0, len(movies_list), cols):
        row = movies_list[i: i + cols]
        for col, movie in zip(st.columns(cols, gap="small"), row):
            with col:
                st.markdown(render_card(movie, show_score), unsafe_allow_html=True)


def _initials(name: str) -> str:
    """'Yash Yuvraj Fatale' → 'YF'"""
    parts = name.strip().split()
    return (parts[0][0] + (parts[-1][0] if len(parts) > 1 else "")).upper()


# ══════════════════════════════════════════════════════════════════════════════
# ① HERO — first impression, brand identity, one-line value proposition
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-badge">✦ NLP · Cosine Similarity · TMDB · 4,803 Films</div>
  <div class="hero-title">
    Find Your Next<br><em>Favourite Film</em>
  </div>
  <div class="hero-sub">
    CineMatch reads the DNA of a movie — plot, genre, cast, director —
    and surfaces films you'll actually love, not just what's trending.
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ② ONBOARDING BANNER — guides first-time users through the 3-step flow
#    Remove or comment-out this block once the app is publicly well-known.
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="onboard-banner">
  <div class="onboard-icon">👋</div>
  <div>
    <div class="onboard-title">New here? Here's how CineMatch works in 3 steps</div>
    <div class="onboard-text">
      <span class="step-pill">1</span>
        Pick any movie you already love from the search box below.<br>
      <span class="step-pill">2</span>
        Choose how many recommendations you want — 5, 10, or 15.<br>
      <span class="step-pill">3</span>
        Hit <strong style="color:#d4a05a">Get Recommendations</strong> and our AI
        instantly finds films with matching plot, genre, cast &amp; director DNA.<br>
      <span style="color:#2e2a26;font-size:0.73rem">
        Scroll down to browse trending picks and learn how the model works.
      </span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ③ SEARCH + CONTROLS
#    Selectbox  — searchable dropdown of all 4,803 titles
#    Slider     — how many recs: 5 / 10 / 15
#    Button     — triggers the AI recommendation engine
# ══════════════════════════════════════════════════════════════════════════════
_, col_center, _ = st.columns([1, 3, 1])
with col_center:
    selected_movie = st.selectbox(
        "Pick a movie",
        options=all_titles,
        # Change "The Dark Knight" to any other default title you prefer
        index=all_titles.index("The Dark Knight") if "The Dark Knight" in all_titles else 0,
        label_visibility="collapsed",
        placeholder="🔍  Search or scroll 4,803 movies …",
    )

st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

ctrl_l, ctrl_m, ctrl_r = st.columns([1.4, 1, 1.4])
with ctrl_l:
    top_n = st.slider("Recommendations", min_value=5, max_value=15, value=10, step=5)
with ctrl_m:
    st.markdown("<div style='padding-top:1.6rem'></div>", unsafe_allow_html=True)
    recommend_clicked = st.button("✦  Get Recommendations", use_container_width=True)
# ctrl_r intentionally empty — keeps button visually centred


# ══════════════════════════════════════════════════════════════════════════════
# ④ STATS BAR — social proof + model transparency
#    Update numbers here if you retrain on a larger / newer dataset.
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="stats-bar">
  <div class="stat"><div class="stat-val">4,803</div><div class="stat-lbl">Films Indexed</div></div>
  <div class="stat"><div class="stat-val">10K</div>  <div class="stat-lbl">TF-IDF Features</div></div>
  <div class="stat"><div class="stat-val">5+</div>   <div class="stat-lbl">Signals / Film</div></div>
  <div class="stat"><div class="stat-val">NLP</div>  <div class="stat-lbl">Core Technology</div></div>
  <div class="stat"><div class="stat-val">TMDB</div> <div class="stat-lbl">Data Source</div></div>
  <div class="stat"><div class="stat-val">&lt;1s</div><div class="stat-lbl">Query Latency</div></div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ⑤ RECOMMENDATION RESULTS
#    Only renders when the button above is pressed.
#    recommend() lives in model.py → returns list of dicts:
#      [ {title, poster, year, rating, overview, score}, ... ]
# ══════════════════════════════════════════════════════════════════════════════
if recommend_clicked:
    with st.spinner("Analysing cinematic DNA …"):
        results = recommend(selected_movie, n=top_n)

    if not results:
        st.warning(
            f"⚠️  **'{selected_movie}'** wasn't found in our dataset. "
            "Try a different spelling or choose another title from the dropdown."
        )
    else:
        st.markdown(f"""
        <div class="section-head">
          <h2>Because you liked <em>{selected_movie}</em></h2>
          <span class="section-tag">{len(results)} Picks</span>
        </div>
        """, unsafe_allow_html=True)

        render_grid(results, cols=5, show_score=True)

        # Explainer note — helps non-technical users understand the score
        st.markdown("""
        <p style="font-size:0.72rem;color:#2a2520;text-align:right;
                  margin-top:0.7rem;padding-right:0.3rem">
          Match % = cosine similarity · higher score = closer cinematic DNA
        </p>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ⑥ TRENDING — always-visible section, top-rated films from the dataset
#    Change get_trending(10) in load_trending() (top of file) to show more.
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div class="section-head">
  <h2>Trending Right Now</h2>
  <span class="section-tag">Top Rated</span>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading trending films …"):
    trending = load_trending()

render_grid(trending, cols=5)


# ══════════════════════════════════════════════════════════════════════════════
# ⑦ HOW IT WORKS — pipeline explainer (valuable for demos & investor pitches)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div class="section-head">
  <h2>How It Works</h2>
  <span class="section-tag">Under the Hood</span>
</div>
""", unsafe_allow_html=True)

PIPELINE_STEPS = [
    ("🗂️", "Data Ingestion",
     "TMDB 5,000 movies merged with full cast & crew credits — 4,803 cleaned titles "
     "with genre, keyword, overview, and director metadata."),
    ("🔬", "Feature Engineering",
     "Plot overview, genres (×2 weight), keywords, top-5 cast, and director (×3 weight) "
     "are blended into one weighted tag string per film."),
    ("🧠", "TF-IDF Vectorisation",
     "A 10,000-dimension sparse matrix captures term importance across the corpus, "
     "with bigrams for richer semantic coverage."),
    ("🎯", "Cosine Similarity",
     "Nearest-neighbour lookup in semantic space surfaces films with matching cinematic "
     "DNA — not just shared genre labels."),
]

for col, (icon, title, body) in zip(st.columns(4, gap="medium"), PIPELINE_STEPS):
    with col:
        st.markdown(f"""
        <div class="how-card">
          <div class="how-icon">{icon}</div>
          <div class="how-title">{title}</div>
          <div class="how-body">{body}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ⑧ FOUNDER / ABOUT — branding + social links
#    All content is driven by FOUNDER_INFO at the top of this file.
#    Just update that dict — nothing else to change here.
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div class="section-head">
  <h2>Built By</h2>
  <span class="section-tag">The Team</span>
</div>
""", unsafe_allow_html=True)

# Avatar: photo if URL provided, else golden initials
avatar_html = (
    f'<img src="{FOUNDER_INFO["avatar_url"]}" '
    f'style="width:72px;height:72px;border-radius:50%;object-fit:cover;'
    f'border:2px solid rgba(212,160,90,0.35);">'
    if FOUNDER_INFO.get("avatar_url")
    else f'<div class="founder-avatar">{_initials(FOUNDER_INFO["name"])}</div>'
)

# Social links: dynamically built from FOUNDER_INFO keys
_SOCIAL_META = {
    "github":   ("⌥", "GitHub"),
    "linkedin": ("in", "LinkedIn"),
    "email":    ("✉",  "Email"),
}
social_html = ""
for key, (icon, label) in _SOCIAL_META.items():
    val = FOUNDER_INFO.get(key, "")
    if val:
        href = f"mailto:{val}" if key == "email" else val
        social_html += f'<a href="{href}" target="_blank" class="social-link">{icon} {label}</a>'

st.markdown(f"""
<div class="team-section">
  <div style="display:flex;align-items:flex-start;gap:1.4rem;flex-wrap:wrap;">
    {avatar_html}
    <div style="flex:1;min-width:220px;">
      <div class="founder-name">{FOUNDER_INFO["name"]}</div>
      <div class="founder-role">{FOUNDER_INFO["role"]}</div>
      <div class="founder-bio">{FOUNDER_INFO["bio"]}</div>
      <div style="display:flex;gap:0.6rem;flex-wrap:wrap;margin-top:1rem;">
        {social_html}
      </div>
    </div>
    <div style="flex-shrink:0;text-align:right;padding-top:0.2rem;">
      <div style="font-family:'Playfair Display',serif;font-size:2.2rem;
                  font-weight:900;color:rgba(212,160,90,0.1);line-height:1;">
        Cine<br>Match
      </div>
      <div style="font-size:0.62rem;color:#2a2520;letter-spacing:0.14em;
                  text-transform:uppercase;margin-top:0.25rem;">v1.0 · 2025</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Optional co-founder / contributor cards (populated from TEAM list at top)
if TEAM:
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    for col, member in zip(st.columns(min(len(TEAM), 4), gap="medium"), TEAM):
        with col:
            gh = (f"<a href='{member['github']}' target='https://github.com/yash01fatale' class='social-link' "
                  f"style='margin-top:0.6rem'>⌥ GitHub</a>") if member.get("github") else ""
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);
                        border-radius:12px;padding:1.1rem;text-align:center;">
              <div style="width:48px;height:48px;border-radius:50%;
                          background:linear-gradient(135deg,#d4a05a,#6a3d10);
                          display:flex;align-items:center;justify-content:center;
                          font-family:'Playfair Display',serif;font-weight:700;
                          font-size:1.1rem;color:#080a0f;margin:0 auto 0.6rem;">
                {_initials(member["name"])}
              </div>
              <div style="font-weight:600;font-size:0.88rem;color:#ddd5c8">{member["name"]}</div>
              <div style="font-size:0.7rem;color:#4a4440;margin-top:0.15rem">{member.get("role","")}</div>
              {gh}
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ⑨ FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="footer">
  <div class="footer-logo">🎬 CineMatch AI</div>
  <div class="footer-copy">
    Built with
    <a href="https://streamlit.io" target="_blank">Streamlit</a> ·
    <a href="https://scikit-learn.org" target="_blank">Scikit-learn</a> ·
    <a href="https://www.themoviedb.org" target="_blank">TMDB API</a><br>
    © 2025 {FOUNDER_INFO["name"]} · All rights reserved ·
    <a href="mailto:{FOUNDER_INFO.get('email','')}" target="_blank">Contact</a>
  </div>
</div>
""", unsafe_allow_html=True)