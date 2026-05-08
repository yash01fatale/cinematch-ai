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
║  model_data/     → Auto-created by build_model.py                          ║
║  .env            → Your secret TMDB_API_KEY goes here (never commit!)       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import streamlit as st
import pandas as pd
from model import recommend, fetch_movie_meta, get_trending, fetch_trailer


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  ← must be the FIRST Streamlit call
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CineMatch AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ══════════════════════════════════════════════════════════════════════════════
# FOUNDER / TEAM INFO
# ══════════════════════════════════════════════════════════════════════════════
FOUNDER_INFO = {
    "name":       "Yash Yuvraj Fatale",
    "role":       "Founder & Lead Engineer",
    "bio": (
        "Engineering student passionate about building AI products that solve "
        "real problems. CineMatch is applied NLP in action — turning raw movie "
        "metadata into a personalised discovery experience anyone can use."
    ),
    "github":     "https://github.com/yashfatale",
    "linkedin":   "https://linkedin.com/in/yashfatale",
    "email":      "yash@cinematch.ai",
    "avatar_url": "",
}

TEAM = []


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_titles() -> list[str]:
    df = pd.read_csv("dataset/tmdb_5000_movies.csv")
    return sorted(df["title"].dropna().unique().tolist())


@st.cache_data
def load_trending() -> list[dict]:
    titles = get_trending(10)
    return [{"title": t, **fetch_movie_meta(t)} for t in titles]


all_titles = load_titles()


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — Dark luxury cinema aesthetic
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;0,700;1,300;1,400;1,600&family=Outfit:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --gold: #C9A84C;
    --gold2: #E8C97A;
    --gold-dim: rgba(201,168,76,0.10);
    --gold-border: rgba(201,168,76,0.22);
    --bg: #05080F;
    --bg2: #0A0D16;
    --bg3: #0F1420;
    --bg4: #141928;
    --border: rgba(255,255,255,0.07);
    --text: #F0EAE0;
    --text2: #8A8070;
    --text3: #4A4540;
    --radius: 14px;
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Outfit', sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

.block-container {
    max-width: 1400px !important;
    padding: 0 2.5rem 6rem !important;
}

/* ── TOPBAR ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.2rem 0 1.2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0;
    position: sticky;
    top: 0;
    z-index: 99;
    background: rgba(5,8,15,0.90);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
}
.topbar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.45rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: 0.02em;
}
.logo-pulse {
    width: 9px; height: 9px;
    border-radius: 50%;
    background: var(--gold);
    box-shadow: 0 0 10px var(--gold);
    animation: pulse-anim 2.2s infinite;
}
@keyframes pulse-anim {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:.4; transform:scale(0.85); }
}
.topbar-nav {
    display: flex;
    align-items: center;
    gap: 2.5rem;
}
.topbar-nav a {
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: var(--text3);
    text-decoration: none;
    transition: color .2s;
}
.topbar-nav a:hover { color: var(--gold); }
.topbar-badge {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #05080F;
    background: var(--gold);
    padding: 0.45rem 1.1rem;
    border-radius: 7px;
}

/* ── HERO ── */
.hero {
    padding: 6rem 0 4rem;
    display: grid;
    grid-template-columns: 1fr 380px;
    gap: 3rem;
    align-items: center;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--gold);
    background: var(--gold-dim);
    border: 1px solid var(--gold-border);
    padding: 0.38rem 1rem;
    border-radius: 100px;
    margin-bottom: 2.2rem;
}
.hero-eyebrow-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: var(--gold);
    animation: pulse-anim 2.2s infinite;
}
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(3.5rem, 6vw, 6.5rem);
    font-weight: 300;
    line-height: 0.96;
    letter-spacing: -0.025em;
    color: var(--text);
    margin-bottom: 1.8rem;
}
.hero-title em  { font-style: italic; color: var(--gold); }
.hero-title strong { font-weight: 700; display: block; }
.hero-sub {
    font-size: 1rem;
    font-weight: 300;
    color: var(--text2);
    max-width: 480px;
    line-height: 1.85;
    margin-bottom: 0;
}

/* Right panel – floating suggestion cards */
.hero-panel {
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.float-card {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 13px 15px;
    display: flex;
    align-items: center;
    gap: 12px;
    transition: transform .3s, border-color .3s;
}
.float-card:hover {
    transform: translateX(-5px);
    border-color: var(--gold-border);
}
.float-card:nth-child(2) { margin-left: 20px; }
.float-card:nth-child(3) { margin-left: 10px; }
.float-thumb {
    width: 38px; height: 54px;
    border-radius: 6px;
    flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
    color: rgba(201,168,76,0.3);
}
.float-info { flex: 1; min-width: 0; }
.float-title {
    font-size: 0.82rem; font-weight: 500;
    color: var(--text);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    margin-bottom: 3px;
}
.float-meta { font-size: 0.68rem; color: var(--text3); }
.float-score {
    font-size: 0.68rem; font-weight: 600;
    color: var(--gold);
    background: var(--gold-dim);
    border: 1px solid var(--gold-border);
    padding: 2px 8px; border-radius: 5px;
    white-space: nowrap;
}

/* ── SEARCH CARD ── */
.search-card {
    background: var(--bg2);
    border: 1px solid var(--gold-border);
    border-radius: 20px;
    padding: 2rem 2.5rem 2.5rem;
    position: relative;
    overflow: hidden;
    margin: 0 0 2.5rem;
}
.search-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
}
.search-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 1rem;
}

/* ── SELECTBOX override ── */
[data-testid="stSelectbox"] > div {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(201,168,76,0.22) !important;
    border-radius: 12px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.95rem !important;
    color: var(--text) !important;
}
[data-testid="stSelectbox"] > div:focus-within {
    border-color: rgba(201,168,76,0.5) !important;
    box-shadow: 0 0 0 3px rgba(201,168,76,0.07) !important;
}
[data-testid="stSelectbox"] label { display: none !important; }
[data-testid="stSelectbox"] svg { color: var(--gold) !important; }

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, #C9A84C 0%, #8a6820 100%) !important;
    color: #05080F !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.83rem !important;
    letter-spacing: 0.09em !important;
    text-transform: uppercase !important;
    padding: 0.78rem 2rem !important;
    border: none !important;
    border-radius: 10px !important;
    transition: opacity .18s, transform .15s, box-shadow .18s !important;
    box-shadow: 0 4px 22px rgba(201,168,76,0.25) !important;
}
.stButton > button:hover {
    opacity: 0.87 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(201,168,76,0.35) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── SLIDER ── */
[data-testid="stSlider"] label {
    color: var(--text3) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.10em !important;
    text-transform: uppercase !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-testid="stSlider"] input[type="range"] {
    accent-color: var(--gold) !important;
}
[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: var(--gold) !important;
    font-family: 'Outfit', sans-serif !important;
}

/* ── STATS RIBBON ── */
.stats-ribbon {
    display: flex;
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
    margin: 2rem 0;
    background: var(--bg2);
}
.stat-item {
    flex: 1;
    text-align: center;
    padding: 1.3rem 0.8rem;
    border-right: 1px solid var(--border);
    transition: background .2s;
}
.stat-item:last-child { border-right: none; }
.stat-item:hover { background: var(--bg3); }
.stat-num {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.8rem;
    font-weight: 600;
    color: var(--gold);
    line-height: 1;
}
.stat-lbl {
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: var(--text3);
    margin-top: 0.3rem;
}

/* ── SECTION HEADER ── */
.sec-head {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    margin: 3.5rem 0 1.8rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
}
.sec-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem;
    font-weight: 600;
    color: var(--text);
    margin: 0;
}
.sec-title em { color: var(--gold); font-style: italic; }
.sec-badge {
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--gold);
    background: var(--gold-dim);
    border: 1px solid var(--gold-border);
    padding: 0.22rem 0.7rem;
    border-radius: 6px;
}

/* ── MOVIE CARD ── */
.movie-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
    height: 100%;
    position: relative;
}
.movie-card:hover {
    transform: translateY(-8px) scale(1.015);
    border-color: rgba(201,168,76,0.4);
    box-shadow: 0 24px 60px rgba(0,0,0,0.65), 0 0 0 1px rgba(201,168,76,0.1);
}
.movie-card img {
    width: 100%;
    aspect-ratio: 2/3;
    object-fit: cover;
    display: block;
    background: var(--bg3);
}
.card-overlay {
    position: absolute;
    top: 0; left: 0; right: 0;
    bottom: 42%;
    background: linear-gradient(to bottom, transparent 60%, rgba(5,8,15,0.85) 100%);
    pointer-events: none;
    opacity: 0;
    transition: opacity .25s;
}
.movie-card:hover .card-overlay { opacity: 1; }
.card-body {
    padding: 0.85rem 0.95rem 1rem;
}
.card-title {
    font-size: 0.84rem;
    font-weight: 500;
    color: var(--text);
    line-height: 1.35;
    margin-bottom: 0.45rem;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.card-meta {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    flex-wrap: wrap;
}
.card-year   { font-size: 0.68rem; color: var(--text3); }
.card-dot    { width: 3px; height: 3px; border-radius: 50%; background: var(--text3); }
.card-rating { font-size: 0.68rem; font-weight: 500; color: var(--gold); }
.card-score  {
    margin-left: auto;
    font-size: 0.64rem; font-weight: 600;
    color: var(--gold);
    background: var(--gold-dim);
    border: 1px solid var(--gold-border);
    padding: 0.14rem 0.45rem;
    border-radius: 5px;
}
.match-badge-top {
    position: absolute;
    top: 9px; right: 9px;
    background: rgba(5,8,15,0.82);
    border: 1px solid var(--gold-border);
    color: var(--gold);
    font-size: 0.62rem;
    font-weight: 600;
    padding: 3px 7px;
    border-radius: 5px;
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
}

/* ── HOW IT WORKS ── */
.how-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.6rem 1.4rem;
    height: 100%;
    transition: border-color .2s, transform .2s;
    position: relative;
    overflow: hidden;
}
.how-card::after {
    content: attr(data-step);
    position: absolute;
    top: -0.6rem; right: 1rem;
    font-family: 'Cormorant Garamond', serif;
    font-size: 5.5rem;
    font-weight: 700;
    color: rgba(201,168,76,0.05);
    line-height: 1;
    pointer-events: none;
    user-select: none;
}
.how-card:hover {
    border-color: rgba(201,168,76,0.28);
    transform: translateY(-4px);
}
.how-icon {
    width: 42px; height: 42px;
    background: var(--gold-dim);
    border: 1px solid var(--gold-border);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
    margin-bottom: 1rem;
}
.how-step-num {
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.4rem;
}
.how-title {
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.5rem;
}
.how-body {
    font-size: 0.78rem;
    color: var(--text3);
    line-height: 1.72;
}

/* ── TEAM / FOUNDER ── */
.team-section {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.5rem;
    position: relative;
    overflow: hidden;
}
.team-section::after {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(201,168,76,0.05) 0%, transparent 70%);
    pointer-events: none;
}
.founder-avatar {
    width: 76px; height: 76px;
    border-radius: 50%;
    background: linear-gradient(135deg, #C9A84C, #7A4E0A);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #05080F;
    flex-shrink: 0;
    border: 2px solid rgba(201,168,76,0.35);
}
.founder-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.45rem;
    font-weight: 600;
    color: var(--text);
}
.founder-role {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: var(--gold);
    margin-top: 2px;
}
.founder-bio {
    font-size: 0.85rem;
    color: var(--text2);
    line-height: 1.78;
    margin-top: 0.7rem;
    max-width: 530px;
}
.social-link {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 0.76rem;
    font-weight: 500;
    color: var(--text2);
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.35rem 0.85rem;
    text-decoration: none;
    transition: color .2s, border-color .2s;
}
.social-link:hover {
    color: var(--gold);
    border-color: var(--gold-border);
}

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    color: var(--text2) !important;
    font-size: 0.78rem !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-testid="stExpander"] summary:hover {
    color: var(--gold) !important;
}

/* ── VIDEO ── */
[data-testid="stVideo"] {
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── SPINNER ── */
[data-testid="stSpinner"] { color: var(--gold) !important; }
[data-testid="stSpinner"] > div { border-top-color: var(--gold) !important; }

/* ── WARNING / INFO ── */
[data-testid="stAlert"] {
    background: rgba(201,168,76,0.06) !important;
    border: 1px solid var(--gold-border) !important;
    border-radius: 10px !important;
    color: var(--text2) !important;
}

/* ── LINK BUTTON ── */
[data-testid="stLinkButton"] a {
    background: var(--bg4) !important;
    border: 1px solid var(--border) !important;
    color: var(--text2) !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-testid="stLinkButton"] a:hover {
    border-color: var(--gold-border) !important;
    color: var(--gold) !important;
}

/* ── DIVIDER ── */
.custom-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 3.5rem 0;
}

/* ── FOOTER ── */
.footer {
    text-align: center;
    padding: 3rem 1rem 1rem;
}
.footer-logo {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--gold);
    letter-spacing: 0.04em;
    margin-bottom: 0.5rem;
}
.footer-copy {
    font-size: 0.7rem;
    color: var(--text3);
    letter-spacing: 0.05em;
    line-height: 2;
}
.footer-copy a { color: var(--text3); text-decoration: none; transition: color .2s; }
.footer-copy a:hover { color: var(--gold); }

/* ── RESULT EXPLAINER NOTE ── */
.match-note {
    font-size: 0.68rem;
    color: var(--text3);
    text-align: right;
    margin-top: 0.6rem;
    padding-right: 0.2rem;
    font-style: italic;
}

/* ── MEMBER CARD ── */
.member-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(201,168,76,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(201,168,76,0.4); }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

FALLBACK_POSTER = "https://placehold.co/300x450/0A0D16/1e2028?text=🎬"

BG_GRADIENTS = [
    "linear-gradient(135deg,#1a1a2e,#16213e)",
    "linear-gradient(135deg,#2c1810,#3d2214)",
    "linear-gradient(135deg,#0d1b2a,#1b3a4b)",
    "linear-gradient(135deg,#1a1215,#2d1a22)",
    "linear-gradient(135deg,#0f1923,#1a2f3a)",
    "linear-gradient(135deg,#17120a,#2a2010)",
    "linear-gradient(135deg,#12160f,#1e2a18)",
    "linear-gradient(135deg,#1c1020,#2d1840)",
]


def _initials(name: str) -> str:
    """'Yash Yuvraj Fatale' → 'YF'"""
    parts = name.strip().split()
    return (parts[0][0] + (parts[-1][0] if len(parts) > 1 else "")).upper()


def render_card(movie: dict, show_score: bool = False, idx: int = 0) -> str:
    """Returns the HTML string for one polished movie card."""
    poster  = movie.get("poster") or FALLBACK_POSTER
    title   = movie.get("title", "Unknown")
    year    = movie.get("year", "")
    rating  = movie.get("rating", 0)
    score   = movie.get("score")

    year_html   = f'<span class="card-year">{year}</span><span class="card-dot"></span>' if year else ""
    rating_html = f'<span class="card-rating">★ {rating}</span>' if rating else ""
    score_html  = f'<span class="card-score">{score:.0%} match</span>' if (show_score and score) else ""
    badge_html  = f'<div class="match-badge-top">{score:.0%}</div>' if (show_score and score) else ""

    return f"""
<div class="movie-card">
  {badge_html}
  <img src="{poster}" alt="{title}" loading="lazy"
       onerror="this.onerror=null;this.src='{FALLBACK_POSTER}'">
  <div class="card-overlay"></div>
  <div class="card-body">
    <div class="card-title">{title}</div>
    <div class="card-meta">{year_html}{rating_html}{score_html}</div>
  </div>
</div>"""


def render_grid(movies_list: list[dict], cols: int = 5, show_score: bool = False):
    """Renders a responsive movie grid with trailers in expanders."""
    if not movies_list:
        st.warning("No movies found.")
        return

    for i in range(0, len(movies_list), cols):
        row_movies = movies_list[i:i + cols]
        columns = st.columns(len(row_movies), gap="medium")

        for col, movie in zip(columns, row_movies):
            with col:
                st.markdown(
                    render_card(movie, show_score, i),
                    unsafe_allow_html=True
                )
                with st.expander("🎬 Watch Trailer"):
                    st.write(movie.get("overview", "No overview available."))
                    st.markdown(f"""
⭐ **Rating:** {movie.get('rating', 'N/A')} &nbsp;·&nbsp; 📅 **Year:** {movie.get('year', 'N/A')}
""")
                    trailer_url = fetch_trailer(movie["title"])
                    if trailer_url:
                        st.video(trailer_url)
                        st.link_button("▶ Watch on YouTube", trailer_url, use_container_width=True)
                    else:
                        st.info("Trailer not available.")


# ══════════════════════════════════════════════════════════════════════════════
# ① TOP NAVIGATION BAR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="topbar">
  <div class="topbar-logo">
    <div class="logo-pulse"></div>
    CineMatch&nbsp;<span style="font-weight:300;color:#4A4540">AI</span>
  </div>
  <nav class="topbar-nav">
    <a href="#trending">Trending</a>
    <a href="#how-it-works">How It Works</a>
    <a href="#about">About</a>
  </nav>
  <span class="topbar-badge">Try Free</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ② HERO — two-column layout with floating suggestion cards
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-left">
    <div class="hero-eyebrow">
      <span class="hero-eyebrow-dot"></span>
      NLP · Cosine Similarity · TMDB · 4,803 Films
    </div>
    <h1 class="hero-title">
      <em>Discover</em> films<br>
      <strong>you'll actually love</strong>
    </h1>
    <p class="hero-sub">
      CineMatch reads the cinematic DNA of a movie — plot, genre, cast, director —
      and surfaces films that match your taste with scientific precision.
    </p>
  </div>
  <div class="hero-panel">
    <div class="float-card">
      <div class="float-thumb" style="background:linear-gradient(135deg,#1a1a2e,#16213e)">🎬</div>
      <div class="float-info">
        <div class="float-title">Blade Runner 2049</div>
        <div class="float-meta">2017 · ★ 8.0 · Sci-Fi</div>
      </div>
      <div class="float-score">98% match</div>
    </div>
    <div class="float-card">
      <div class="float-thumb" style="background:linear-gradient(135deg,#2c1810,#3d2214)">🎬</div>
      <div class="float-info">
        <div class="float-title">Inception</div>
        <div class="float-meta">2010 · ★ 8.8 · Thriller</div>
      </div>
      <div class="float-score">95% match</div>
    </div>
    <div class="float-card">
      <div class="float-thumb" style="background:linear-gradient(135deg,#0d1b2a,#1b3a4b)">🎬</div>
      <div class="float-info">
        <div class="float-title">Interstellar</div>
        <div class="float-meta">2014 · ★ 8.6 · Drama</div>
      </div>
      <div class="float-score">91% match</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ③ SEARCH CARD — selectbox + slider + button
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="search-card">
  <div class="search-label">✦ &nbsp;Find films like yours</div>
""", unsafe_allow_html=True)

_, col_center, _ = st.columns([0.5, 4, 0.5])
with col_center:
    selected_movie = st.selectbox(
        "Pick a movie",
        options=all_titles,
        index=all_titles.index("The Dark Knight") if "The Dark Knight" in all_titles else 0,
        label_visibility="collapsed",
        placeholder="🔍  Search or scroll 4,803 movies …",
    )

st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

ctrl_l, ctrl_m, ctrl_r = st.columns([1.6, 1, 1.6])
with ctrl_l:
    top_n = st.slider(
        "Number of recommendations",
        min_value=5, max_value=15, value=10, step=5,
        label_visibility="visible"
    )
with ctrl_m:
    st.markdown("<div style='padding-top:1.55rem'></div>", unsafe_allow_html=True)
    recommend_clicked = st.button("✦  Get Recommendations", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)  # close search-card


# ══════════════════════════════════════════════════════════════════════════════
# ④ STATS RIBBON
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="stats-ribbon">
  <div class="stat-item"><div class="stat-num">4,803</div><div class="stat-lbl">Films Indexed</div></div>
  <div class="stat-item"><div class="stat-num">10K</div>  <div class="stat-lbl">TF-IDF Features</div></div>
  <div class="stat-item"><div class="stat-num">5+</div>   <div class="stat-lbl">Signals / Film</div></div>
  <div class="stat-item"><div class="stat-num">NLP</div>  <div class="stat-lbl">Core Engine</div></div>
  <div class="stat-item"><div class="stat-num">TMDB</div> <div class="stat-lbl">Data Source</div></div>
  <div class="stat-item"><div class="stat-num">&lt;1s</div><div class="stat-lbl">Query Latency</div></div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ⑤ RECOMMENDATION RESULTS
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
        <div class="sec-head">
          <h2 class="sec-title">Because you liked <em>{selected_movie}</em></h2>
          <span class="sec-badge">{len(results)} Picks</span>
        </div>
        """, unsafe_allow_html=True)

        grid_cols = 5 if top_n >= 10 else top_n
        render_grid(results, cols=grid_cols, show_score=True)

        st.markdown(
            '<p class="match-note">Match % = cosine similarity · higher score = closer cinematic DNA</p>',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# ⑥ TRENDING
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="custom-divider"><a name="trending"></a>', unsafe_allow_html=True)
st.markdown("""
<div class="sec-head">
  <h2 class="sec-title">Trending <em>Right Now</em></h2>
  <span class="sec-badge">Top Rated</span>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading trending films …"):
    trending = load_trending()

render_grid(trending, cols=5)


# ══════════════════════════════════════════════════════════════════════════════
# ⑦ HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="custom-divider"><a name="how-it-works"></a>', unsafe_allow_html=True)
st.markdown("""
<div class="sec-head">
  <h2 class="sec-title">How It <em>Works</em></h2>
  <span class="sec-badge">Under the Hood</span>
</div>
""", unsafe_allow_html=True)

PIPELINE_STEPS = [
    ("🗂️", "01", "Data Ingestion",
     "TMDB 5,000 movies merged with full cast & crew credits — 4,803 cleaned titles "
     "with genre, keyword, overview, and director metadata."),
    ("🔬", "02", "Feature Engineering",
     "Plot overview, genres (×2 weight), keywords, top-5 cast, and director (×3 weight) "
     "are blended into one weighted tag string per film."),
    ("🧠", "03", "TF-IDF Vectorisation",
     "A 10,000-dimension sparse matrix captures term importance across the corpus, "
     "with bigrams for richer semantic coverage."),
    ("🎯", "04", "Cosine Similarity",
     "Nearest-neighbour lookup in semantic space surfaces films with matching cinematic "
     "DNA — not just shared genre labels."),
]

cols_how = st.columns(4, gap="medium")
for col, (icon, step, title, body) in zip(cols_how, PIPELINE_STEPS):
    with col:
        st.markdown(f"""
        <div class="how-card" data-step="{step}">
          <div class="how-icon">{icon}</div>
          <div class="how-step-num">Step {step}</div>
          <div class="how-title">{title}</div>
          <div class="how-body">{body}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ⑧ FOUNDER / ABOUT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="custom-divider"><a name="about"></a>', unsafe_allow_html=True)
st.markdown("""
<div class="sec-head">
  <h2 class="sec-title">Built <em>By</em></h2>
  <span class="sec-badge">The Team</span>
</div>
""", unsafe_allow_html=True)

# Avatar
avatar_html = (
    f'<img src="{FOUNDER_INFO["avatar_url"]}" '
    f'style="width:76px;height:76px;border-radius:50%;object-fit:cover;'
    f'border:2px solid rgba(201,168,76,0.35);">'
    if FOUNDER_INFO.get("avatar_url")
    else f'<div class="founder-avatar">{_initials(FOUNDER_INFO["name"])}</div>'
)

# Social links
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
        social_html += f'<a href="{href}" target="_blank" class="social-link">{icon} {label}</a> '

st.markdown(f"""
<div class="team-section">
  <div style="display:flex;align-items:flex-start;gap:1.6rem;flex-wrap:wrap">
    {avatar_html}
    <div style="flex:1;min-width:220px">
      <div class="founder-name">{FOUNDER_INFO["name"]}</div>
      <div class="founder-role">{FOUNDER_INFO["role"]}</div>
      <div class="founder-bio">{FOUNDER_INFO["bio"]}</div>
      <div style="display:flex;gap:0.55rem;flex-wrap:wrap;margin-top:1.1rem">
        {social_html}
      </div>
    </div>
    <div style="flex-shrink:0;text-align:right;padding-top:0.2rem">
      <div style="font-family:'Cormorant Garamond',serif;font-size:3rem;
                  font-weight:700;color:rgba(201,168,76,0.07);line-height:1;
                  letter-spacing:-0.02em">
        Cine<br>Match
      </div>
      <div style="font-size:0.6rem;color:#2a2520;letter-spacing:0.14em;
                  text-transform:uppercase;margin-top:4px">v1.0 · 2025</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Optional co-founder / contributor cards
if TEAM:
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    for col, member in zip(st.columns(min(len(TEAM), 4), gap="medium"), TEAM):
        with col:
            gh = (
                f"<a href='{member['github']}' target='_blank' class='social-link' "
                f"style='margin-top:0.6rem'>⌥ GitHub</a>"
            ) if member.get("github") else ""
            st.markdown(f"""
            <div class="member-card">
              <div class="founder-avatar" style="width:50px;height:50px;font-size:1.1rem;margin:0 auto 0.7rem">
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
<hr class="custom-divider">
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