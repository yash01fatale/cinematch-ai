# 🎬 CineMatch AI

<div align="center">

### AI-Powered Movie Recommendation Platform

Discover movies using Natural Language Processing, TF-IDF vectorisation, and cosine similarity.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TMDB](https://img.shields.io/badge/TMDB-01D277?style=for-the-badge)

</div>

---

# 🚀 Live Demo

👉 https://cinematch-ai.streamlit.app

---

# 📌 About The Project

CineMatch AI is a startup-inspired AI-powered movie recommendation system built using Natural Language Processing and Machine Learning.

The platform analyses:

- 🎭 Genres
- 📝 Movie overview
- 🎬 Director
- 🎥 Cast
- 🔑 Keywords

Using TF-IDF Vectorisation and Cosine Similarity, the system identifies movies with similar cinematic DNA and recommends personalised films instantly.

---

# ✨ Features

## 🎯 AI Recommendation Engine
- NLP-based recommendation system
- TF-IDF vectorisation
- Cosine similarity matching
- Weighted metadata processing

## 🎨 Premium UI/UX
- Startup-inspired cinematic design
- Responsive movie cards
- Trending movies section
- Interactive recommendations
- Modern dark luxury interface

## ⚡ Performance Optimisation
- Precomputed similarity matrix
- Pickle-based model caching
- Fast recommendation latency
- Efficient inference pipeline

## 🌍 Real-Time API Integration
- TMDB API integration
- Live movie posters
- Ratings and metadata
- Dynamic content loading

---

# 🧠 How It Works

└── screenshots/
    ├── home.png
    ├── trending.png
    └── recommendation.png

1. **Data Collection**: Movie metadata is collected from TMDB API, including genres, overview, director, cast, and keywords.
2. **Preprocessing**: The metadata is cleaned and combined into a single string for each movie.
3. **TF-IDF Vectorisation**: The combined metadata is transformed into a TF-ID
F matrix, which captures the importance of each term in the context of the movie dataset.
4. **Cosine Similarity Calculation**: A cosine similarity matrix is computed from the TF
-IDF matrix, which quantifies the similarity between movies based on their metadata.
5. **Model Persistence**: The movies data and similarity matrix are saved using pickle for fast
retrieval during inference.
6. **Recommendation Generation**: When a user selects a movie, the system retrieves the top
similar movies based on the cosine similarity scores and displays them in the UI.
7. **User Interface**: The Streamlit app provides a sleek, startup-inspired interface for users
to explore trending movies and receive personalised recommendations.
8. **Performance Optimisation**: By precomputing the similarity matrix and caching it, the
system ensures that recommendations are generated in under 1 second, providing a seamless user experience.
9. **Real-Time API Integration**: The app dynamically fetches movie posters and metadata from
TMDB API, ensuring that users see up-to-date information and visuals.
10. **Continuous Improvement**: The model can be retrained with new data to enhance recommendation
accuracy and adapt to changing user preferences.

---

⚙️ Installation Guide
1️⃣ Clone Repository
git clone https://github.com/yash01fatale/cinematch-ai.git
cd cinematch-ai
2️⃣ Create Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate
Linux / Mac
python3 -m venv venv
source venv/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Setup Environment Variables

Create a .env file:

TMDB_API_KEY=your_tmdb_api_key_here

Get free TMDB API key: 👉 https://www.themoviedb.org/settings/api

5️⃣ Build AI Model
python build_model.py

This generates:

movies.pkl
similarity.pkl
6️⃣ Run Application
streamlit run app.py

Application will launch at:

http://localhost:8501
🌍 Deployment

This project is deployed using Streamlit Cloud.

Deployment Steps
Push project to GitHub
Login to Streamlit Cloud
Create new app
Select repository
Add TMDB_API_KEY in Secrets
Deploy
📊 Dataset

Dataset Used:

👉 TMDB 5000 Movie Dataset

Source: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

📸 Screenshots
🏠 Home Screen

(Add screenshot here)

🎬 Recommendation Section

(Add screenshot here)

🔥 Trending Movies

(Add screenshot here)

👨‍💻 Developer Information
Yash Yuvraj Fatale
Founder & Lead Engineer — CineMatch AI

Computer Science Engineering student passionate about:

Artificial Intelligence
Machine Learning
Full Stack Development
Startup Product Engineering
NLP Applications
🔗 Connect With Me
GitHub: https://github.com/yash01fatale
LinkedIn: https://linkedin.com/in/yashfatale
Email: yash@cinematch.ai
🚀 Future Scope
User authentication
Personalized recommendations
Watchlist system
Deep learning embeddings
Sentiment-aware recommendations
Mobile application version
⭐ Support

If you like this project:

⭐ Star the repository 🍴 Fork the project 📢 Share with others

📜 License

This project is created for educational and portfolio purposes.

🎬 CineMatch AI

Built with ❤️ using AI & Machine Learning