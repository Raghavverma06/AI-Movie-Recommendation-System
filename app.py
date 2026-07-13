import os
import sys
import json
import time
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Configure page layout and tab options
st.set_page_config(
    page_title="AI Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom css for dark-theme, glassmorphism, animated cards, gradient text and scrollbars
GLASS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700;800&display=swap');

/* Apply primary fonts */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif;
    color: #e0e0ff !important;
}

/* Background setting */
.stApp {
    background: radial-gradient(circle at 50% 50%, #101124 0%, #06070a 100%);
    background-attachment: fixed;
    color: #e0e0ff;
}

/* Glassmorphism Container */
.glass-container {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 25px;
    margin-bottom: 25px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
}
.glass-container:hover {
    border-color: rgba(255, 255, 255, 0.12);
    box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.45);
}

/* Gradient text */
.gradient-text {
    background: linear-gradient(90deg, #00FFFF, #8A2BE2, #FF1493);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-family: 'Outfit', sans-serif;
}

/* KPI Card Styles */
.kpi-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
}
.kpi-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(138, 43, 226, 0.2);
    border-color: rgba(138, 43, 226, 0.4);
}
.kpi-num {
    font-family: 'Outfit', sans-serif;
    font-size: 38px;
    font-weight: 800;
    margin: 5px 0;
}
.kpi-num-1 { color: #00FFFF; text-shadow: 0 0 10px rgba(0, 255, 255, 0.3); }
.kpi-num-2 { color: #8A2BE2; text-shadow: 0 0 10px rgba(138, 43, 226, 0.3); }
.kpi-num-3 { color: #FF1493; text-shadow: 0 0 10px rgba(255, 20, 147, 0.3); }
.kpi-num-4 { color: #FF8C00; text-shadow: 0 0 10px rgba(255, 140, 0, 0.3); }

.kpi-label {
    font-size: 13px;
    color: #a0a0c0;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
}

/* Movie Card Poster Placeholder */
.movie-poster-placeholder {
    height: 220px;
    background: linear-gradient(135deg, #1d1b38 0%, #0d0c18 100%);
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    padding: 15px;
    transition: all 0.3s ease;
    margin-bottom: 12px;
}
.movie-poster-placeholder:hover {
    border-color: rgba(0, 255, 255, 0.3);
    transform: scale(1.03);
}
.movie-icon {
    font-size: 45px;
    margin-bottom: 10px;
    filter: drop-shadow(0 0 8px rgba(0, 255, 255, 0.4));
}
.movie-title-short {
    font-size: 14px;
    font-weight: 600;
    color: #ffffff;
    max-height: 40px;
    overflow: hidden;
}

/* Sidebar developer card */
.sidebar-dev-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 12px;
    padding: 16px;
    margin-top: 15px;
}
.sidebar-dev-title {
    color: #00FFFF;
    font-weight: 700;
    font-size: 14px;
    font-family: 'Outfit', sans-serif;
    margin-bottom: 2px;
}
.sidebar-dev-sub {
    color: #FF1493;
    font-weight: 600;
    font-size: 11px;
    margin-bottom: 8px;
}
.sidebar-dev-text {
    font-size: 11px;
    color: #a0a0c0;
    line-height: 1.4;
}

/* Custom badges */
.custom-badge {
    background: rgba(138, 43, 226, 0.2);
    border: 1px solid rgba(138, 43, 226, 0.5);
    color: #e0d0ff;
    padding: 3px 8px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    display: inline-block;
    margin-right: 5px;
    margin-bottom: 5px;
}

.rating-badge {
    background: rgba(0, 255, 255, 0.15);
    border: 1px solid rgba(0, 255, 255, 0.4);
    color: #d0ffff;
    padding: 3px 8px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    display: inline-block;
}

/* Footer style */
.footer {
    text-align: center;
    padding: 25px 0;
    margin-top: 50px;
    font-size: 12px;
    color: #656585;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
}

/* Animations */
@keyframes slideIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}
.animate-slide {
    animation: slideIn 0.5s ease-out forwards;
}
</style>
"""
st.markdown(GLASS_CSS, unsafe_allow_html=True)

# Helper to verify dataset files and models
@st.cache_resource
def get_hybrid_recommender():
    """
    Loads pre-trained Hybrid Recommender model and metrics on fly.
    Attempts auto-training if models are missing.
    """
    model_path = "model/hybrid_recommender.joblib"
    metrics_path = "model/evaluation_metrics.json"
    
    # Check if files exist, train if missing
    if not os.path.exists(model_path) or not os.path.exists(metrics_path):
        # We need to run the train pipeline
        from src.train import run_training_pipeline
        run_training_pipeline()
        
    # Load model
    try:
        rec_model = joblib.load(model_path)
        with open(metrics_path, "r") as f:
            metrics_dict = json.load(f)
        return rec_model, metrics_dict
    except Exception as e:
        # Fallback in case of corruption
        st.error(f"Error loading models: {e}. Attempting to re-train.")
        from src.train import run_training_pipeline
        run_training_pipeline()
        rec_model = joblib.load(model_path)
        with open(metrics_path, "r") as f:
            metrics_dict = json.load(f)
        return rec_model, metrics_dict

# Load Models
try:
    hybrid_engine, evaluation_metrics = get_hybrid_recommender()
    movies_df = hybrid_engine.movies_df
    ratings_df = hybrid_engine.ratings_df
except Exception as e:
    st.error(f"Failed to initialize recommender engine: {e}")
    st.stop()

# Import visualizations
from src.visualization import (
    plot_genre_distribution,
    plot_ratings_distribution,
    plot_top_genres,
    plot_popular_movies,
    plot_highest_rated_movies,
    plot_movies_per_year,
    plot_similarity_heatmap,
    plot_model_comparison,
    plot_model_latency
)

# Initialize Session State
if "total_recommendations" not in st.session_state:
    st.session_state.total_recommendations = 0
    # Calculate previous total from history file
    history = hybrid_engine.get_history()
    if not history.empty:
        st.session_state.total_recommendations = len(history)

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.markdown("<h2 class='gradient-text' style='text-align: center;'>🎬 CineMatch AI</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; font-size: 12px; color: #a0a0c0;'>Offline Movie Recommendation System</p>", unsafe_allow_html=True)

# User Settings (Select User ID for personalization simulation)
st.sidebar.markdown("<hr style='margin: 10px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size: 13px; font-weight: 600; color: #00FFFF;'>👤 Active User Profile</p>", unsafe_allow_html=True)
user_ids_list = [None] + sorted(ratings_df["userId"].unique().tolist())
selected_user = st.sidebar.selectbox(
    "Select User ID (for Collaborative filtering)",
    options=user_ids_list,
    format_func=lambda x: f"User {x}" if x is not None else "Anonymous User (Cold Start)"
)

# Navigation
st.sidebar.markdown("<hr style='margin: 10px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size: 13px; font-weight: 600; color: #00FFFF;'>🧭 Navigation</p>", unsafe_allow_html=True)

pages = [
    "🏠 Home",
    "🎬 Recommend Movies",
    "📊 Analytics",
    "📈 Genre Statistics",
    "⭐ Top Rated",
    "🔥 Popular Movies",
    "🧠 Recommendation Insights",
    "📊 Model Comparison",
    "📂 Dataset Explorer",
    "📜 Recommendation History",
    "⚙️ System Performance",
    "👨💻 About Developer"
]

selected_page = st.sidebar.radio(
    "Go To Page",
    options=pages,
    label_visibility="collapsed"
)

# Developer Sidebar Card
st.sidebar.markdown("<hr style='margin: 15px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
dev_card_html = """
<div class='sidebar-dev-card'>
    <div class='sidebar-dev-title'>👨💻 Developer</div>
    <div class='sidebar-dev-sub'>Raghav Verma</div>
    <div class='dev-detail'>
        <b>B.Tech CSE (AI & ML)</b><br>
        Manav Rachna International Institute of Research & Studies (MRIIRS)<br>
        <i>Intern ID: CITS2709</i>
    </div>
</div>
"""
st.sidebar.markdown(dev_card_html, unsafe_allow_html=True)


# ==========================================================
# PAGE ROUTING
# ==========================================================

# ----------------------------------------------------------
# 🏠 HOME PAGE
# ----------------------------------------------------------
if selected_page == "🏠 Home":
    st.markdown("<h1 class='gradient-text animate-slide'>AI Movie Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("### Production-Grade Hybrid Recommender Platform")
    st.write("")
    
    # Animated KPI Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Total Movies</div>
            <div class='kpi-num kpi-num-1'>{len(movies_df):,}</div>
            <div class='kpi-label' style='font-size: 10px; color: #707090;'>In Local Catalog</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col2:
        num_genres = len(set(movies_df["genres"].str.split("|").explode().dropna().unique()))
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Genres</div>
            <div class='kpi-num kpi-num-2'>{num_genres}</div>
            <div class='kpi-label' style='font-size: 10px; color: #707090;'>Distinct Categories</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col3:
        avg_rating = ratings_df["rating"].mean()
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Average Rating</div>
            <div class='kpi-num kpi-num-3'>{avg_rating:.2f}★</div>
            <div class='kpi-label' style='font-size: 10px; color: #707090;'>From {len(ratings_df):,} reviews</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi_col4:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Total Queries Run</div>
            <div class='kpi-num kpi-num-4'>{st.session_state.total_recommendations}</div>
            <div class='kpi-label' style='font-size: 10px; color: #707090;'>Log History Entries</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    
    # Layout sections
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.markdown("""
        <div class='glass-container'>
            <h3 style='color: #00FFFF; margin-top: 0;'>Project Overview</h3>
            <p>
                CineMatch AI is a complete, production-ready machine learning movie recommendation application operating 100% offline. 
                The system integrates both Content-Based filtering and Collaborative filtering, ensembling them into a <b>Hybrid Recommendation Engine</b>.
            </p>
            <p><b>Core Features:</b></p>
            <ul>
                <li><b>Content-Based Filtering:</b> Combines movie genres, descriptions, keywords, cast, and director names into a single metadata soup, processed with TF-IDF Vectorization and Cosine Similarity.</li>
                <li><b>Collaborative Filtering:</b> Pivots user rating patterns, utilizing NearestNeighbors for item-based matches, and Matrix Factorization (Truncated SVD) for personalized user-rating prediction.</li>
                <li><b>Hybrid Engine:</b> Blends content and collaborative scores using adjustable sliders to target specific balances of similarity.</li>
                <li><b>Explainable AI (XAI):</b> Displays descriptive explanations detailing why a movie was recommended.</li>
                <li><b>Dataset Explorer & Interactive Charts:</b> Full Plotly interactive analytics dashboard, genre breakdown, and dataset filters.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='glass-container'>
            <h3 style='color: #8A2BE2; margin-top: 0;'>Active recommendation Algorithm</h3>
            <p>The system ensembles scores mathematically based on the weights you select:</p>
            <p style='text-align: center; font-style: italic; background: rgba(0,0,0,0.2); padding: 12px; border-radius: 8px;'>
                Score = w<sub>content</sub> × CosineSim<sub>TF-IDF</sub> + w<sub>collab</sub> × Similarity<sub>NN</sub> + w<sub>personal</sub> × PredictedRating<sub>SVD</sub>
            </p>
            <p>Currently configured settings are adjusted dynamically on the <b>Recommend Movies</b> tab.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown(f"""
        <div class='glass-container'>
            <h3 style='color: #FF1493; margin-top: 0;'>Algorithm Accuracy</h3>
            <p>Evaluation results computed during the training pipeline:</p>
            <div style='margin: 15px 0;'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                    <span>Collaborative SVD RMSE:</span>
                    <span style='color:#00FFFF; font-weight:bold;'>{evaluation_metrics['Collaborative (SVD)']['RMSE']:.4f}</span>
                </div>
                <div style='background:rgba(255,255,255,0.05); border-radius:10px; height:8px;'>
                    <div style='background:linear-gradient(90deg, #8A2BE2, #00FFFF); width:{max(10, min(100, int((2.0 - evaluation_metrics['Collaborative (SVD)']['RMSE']) * 50)))}%; height:100%; border-radius:10px;'></div>
                </div>
            </div>
            <div style='margin: 15px 0;'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                    <span>Collaborative SVD MAE:</span>
                    <span style='color:#FF1493; font-weight:bold;'>{evaluation_metrics['Collaborative (SVD)']['MAE']:.4f}</span>
                </div>
                <div style='background:rgba(255,255,255,0.05); border-radius:10px; height:8px;'>
                    <div style='background:linear-gradient(90deg, #8A2BE2, #FF1493); width:{max(10, min(100, int((2.0 - evaluation_metrics['Collaborative (SVD)']['MAE']) * 50)))}%; height:100%; border-radius:10px;'></div>
                </div>
            </div>
            <div style='margin: 15px 0;'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                    <span>Hybrid Catalog Coverage:</span>
                    <span style='color:#FF8C00; font-weight:bold;'>{evaluation_metrics['Hybrid Engine']['Coverage']}%</span>
                </div>
                <div style='background:rgba(255,255,255,0.05); border-radius:10px; height:8px;'>
                    <div style='background:linear-gradient(90deg, #8A2BE2, #FF8C00); width:{evaluation_metrics['Hybrid Engine']['Coverage']}%; height:100%; border-radius:10px;'></div>
                </div>
            </div>
            <p style='font-size:11px; color:#707090; margin-top:20px;'>
                *SVD metrics are evaluated on an offline 20% validation split. Hybrid Catalog Coverage is evaluated across 30 random search runs.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='glass-container'>
            <h3 style='color: #FF8C00; margin-top: 0;'>Dataset Information</h3>
            <p><b>Type:</b> Offline Synthetic MovieLens 100K Representation</p>
            <p><b>Catalog Size:</b> {len(movies_df)} movies</p>
            <p><b>Ratings Volume:</b> {len(ratings_df):,} ratings</p>
            <p><b>Density:</b> {(len(ratings_df) / (len(movies_df) * len(ratings_df["userId"].unique())) * 100):.2f}%</p>
            <p><b>Status:</b> Cached local storage (ready)</p>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------------------------------------
# 🎬 MOVIE RECOMMENDATION PAGE
# ----------------------------------------------------------
elif selected_page == "🎬 Recommend Movies":
    st.markdown("<h1 class='gradient-text animate-slide'>Movie Recommendation Engine</h1>", unsafe_allow_html=True)
    st.write("Configure hybrid parameters and search for a movie to generate similar matches in real time.")
    
    # Config parameters container
    with st.expander("⚙️ Adjust Recommendation Weights (Ensemble settings)", expanded=True):
        col_w1, col_w2, col_w3, col_top = st.columns(4)
        with col_w1:
            w_content = st.slider("Content Similarity Weight", 0.0, 1.0, 0.5, 0.1, help="TF-IDF Metadata similarity (genres, keywords, cast, directors)")
        with col_w2:
            w_collab = st.slider("Collaborative Similarity Weight", 0.0, 1.0, 0.5, 0.1, help="Item-based Collaborative similarity (NearestNeighbors ratings profiles)")
        with col_w3:
            # Enable personal weights if a user is selected
            if selected_user is not None:
                w_personal = st.slider("Personal SVD Weight", 0.0, 1.0, 0.3, 0.1, help="Incorporate SVD predicted rating score of the active user profile")
            else:
                st.info("Log in as a User in sidebar to unlock personal weights.")
                w_personal = 0.0
        with col_top:
            num_recs = st.slider("Recommendations Count", 5, 20, 10, 1)

    # Search section
    st.write("")
    movie_titles = sorted(movies_df["title"].tolist())
    search_movie = st.selectbox(
        "🔎 Search and Select a Movie:",
        options=movie_titles,
        index=movie_titles.index("Inception") if "Inception" in movie_titles else 0
    )
    
    if st.button("🚀 Generate Recommendations", use_container_width=True):
        with st.spinner("Analyzing similarity vectors and rating arrays..."):
            
            recs, exec_time = hybrid_engine.recommend(
                movie_title=search_movie,
                user_id=selected_user,
                top_n=num_recs,
                weight_content=w_content,
                weight_collab=w_collab,
                weight_personal=w_personal
            )
            
            if not recs:
                st.warning("No recommendations could be generated. Check search spelling.")
            else:
                # Update session state recommendations counter
                st.session_state.total_recommendations += 1
                
                # Show results header
                st.markdown(f"### Top {len(recs)} Recommendations for ***{search_movie}***")
                st.caption(f"Inference Latency: **{exec_time:.2f} ms** | Active Profile: **{f'User {selected_user}' if selected_user else 'Anonymous'}**")
                st.write("")
                
                # Build cards in row grid
                card_cols = st.columns(2)
                for idx, item in enumerate(recs):
                    col_to_use = card_cols[idx % 2]
                    
                    with col_to_use:
                        # Extract data
                        title = item["title"]
                        year = item["release_year"]
                        genres = item["genres"].split("|")
                        director = item["director"]
                        avg_rat = item["avg_rating"]
                        sim_pct = item["similarity_percentage"]
                        conf_pct = item["confidence_percentage"]
                        explanation = item["explanation"]
                        desc = item["description"]
                        
                        # Generate simple clean acronym for poster placeholder
                        acronym = "".join([w[0].upper() for w in title.split() if w.isalnum()])[:3]
                        
                        genres_badges = " ".join([f"<span class='custom-badge'>{g}</span>" for g in genres])
                        
                        card_html = f"""
                        <div class='glass-container' style='margin-bottom: 20px;'>
                            <div style='display: flex; gap: 20px;'>
                                <div class='movie-poster-placeholder' style='width: 120px; height: 160px; flex-shrink: 0; margin-bottom: 0;'>
                                    <div class='movie-icon'>🎬</div>
                                    <div class='movie-title-short'>{acronym}</div>
                                </div>
                                <div style='flex-grow: 1;'>
                                    <div style='display: flex; justify-content: space-between; align-items: flex-start;'>
                                        <h4 style='margin: 0; font-size: 18px;'>{title} <span style='font-size: 13px; color: #a0a0c0;'>({year})</span></h4>
                                        <span class='rating-badge'>{avg_rat:.1f} ★</span>
                                    </div>
                                    <p style='font-size: 12px; color: #ff1493; margin: 4px 0 8px 0; font-weight:600;'>
                                        Directed by {director}
                                    </p>
                                    <div style='margin-bottom: 8px;'>
                                        {genres_badges}
                                    </div>
                                    <p style='font-size: 12px; color: #d0d0f0; margin-bottom: 10px; line-height: 1.4; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;'>
                                        {desc}
                                    </p>
                                    <div style='display: flex; gap: 15px; font-size: 11px; font-weight:600; color: #00FFFF; margin-bottom: 8px;'>
                                        <span>Similarity: {sim_pct}%</span>
                                        <span style='color: #FF8C00;'>Confidence: {conf_pct}%</span>
                                    </div>
                                </div>
                            </div>
                            <div style='margin-top: 12px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.06); font-size: 11.5px; line-height: 1.4; color: #a0a0c0; font-style: italic;'>
                                💡 <b>Why this recommendation?</b> {explanation}
                            </div>
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)
                
                # Export to CSV
                recs_df = pd.DataFrame(recs)
                # Select user facing fields
                export_df = recs_df[[
                    "title", "release_year", "genres", "director", 
                    "avg_rating", "similarity_percentage", "confidence_percentage", "explanation"
                ]].copy()
                export_df.columns = [
                    "Movie Title", "Release Year", "Genres", "Director", 
                    "Average Rating", "Similarity %", "Confidence %", "Explanation"
                ]
                
                csv_data = export_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Export Recommendations as CSV",
                    data=csv_data,
                    file_name=f"recommendations_{search_movie.replace(' ', '_')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# ----------------------------------------------------------
# 📊 ANALYTICS PAGE
# ----------------------------------------------------------
elif selected_page == "📊 Analytics":
    st.markdown("<h1 class='gradient-text animate-slide'>Dataset Analytics</h1>", unsafe_allow_html=True)
    st.write("Explore distributions, counts, and top categories in the recommendation dataset.")
    
    # 2x2 Grid using Plotly
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        g_fig = plot_genre_distribution(movies_df)
        st.plotly_chart(g_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        top_g_fig = plot_top_genres(movies_df, ratings_df)
        st.plotly_chart(top_g_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        r_fig = plot_ratings_distribution(ratings_df)
        st.plotly_chart(r_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        y_fig = plot_movies_per_year(movies_df)
        st.plotly_chart(y_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------
# 📈 GENRE STATISTICS
# ----------------------------------------------------------
elif selected_page == "📈 Genre Statistics":
    st.markdown("<h1 class='gradient-text animate-slide'>Genre Statistics</h1>", unsafe_allow_html=True)
    st.write("Detailed breakdown and top rated lists for each genre in the local database.")
    
    genres_list = sorted(list(set(movies_df["genres"].str.split("|").explode().dropna().unique())))
    selected_genre = st.selectbox("Select Genre to Analyze:", options=genres_list)
    
    # Filter movies by genre
    genre_movies = movies_df[movies_df["genres"].str.contains(selected_genre)]
    
    # Count average ratings for this genre
    genre_ratings = ratings_df[ratings_df["movieId"].isin(genre_movies["movieId"])]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class='glass-container'>
            <h3 style='color: #00FFFF; margin-top: 0;'>{selected_genre} Overview</h3>
            <table style='width: 100%; border-collapse: collapse; margin-top: 15px;'>
                <tr>
                    <td style='padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05); color:#a0a0c0;'>Total Movies</td>
                    <td style='padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-weight:bold; text-align:right;'>{len(genre_movies)}</td>
                </tr>
                <tr>
                    <td style='padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05); color:#a0a0c0;'>Total Ratings</td>
                    <td style='padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-weight:bold; text-align:right;'>{len(genre_ratings):,}</td>
                </tr>
                <tr>
                    <td style='padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05); color:#a0a0c0;'>Average Rating</td>
                    <td style='padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-weight:bold; text-align:right; color:#00FFFF;'>{genre_ratings['rating'].mean():.2f} ★</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        # Plot rating distribution for this genre
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        g_rate_counts = genre_ratings["rating"].value_counts().reset_index()
        g_rate_counts.columns = ["Rating", "Count"]
        g_rate_counts = g_rate_counts.sort_values(by="Rating")
        fig = px.bar(g_rate_counts, x="Rating", y="Count", color="Count", color_continuous_scale="Agsunset", title=f"Ratings Distribution for {selected_genre}")
        st.plotly_chart(apply_dark_theme(fig), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.write("")
    st.markdown(f"### Top Rated {selected_genre} Movies")
    
    # Calculate highest rated inside this genre
    avg_ratings = ratings_df.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        ratings_count=("rating", "count")
    ).reset_index()
    
    genre_merged = pd.merge(genre_movies, avg_ratings, on="movieId")
    # Show movies with at least 5 reviews
    top_genre_movies = genre_merged[genre_merged["ratings_count"] >= 5].sort_values(by="avg_rating", ascending=False).head(10)
    
    st.dataframe(
        top_genre_movies[["title", "release_year", "director", "avg_rating", "ratings_count"]].rename(columns={
            "title": "Movie Title",
            "release_year": "Year",
            "director": "Director",
            "avg_rating": "Average Rating",
            "ratings_count": "Votes"
        }),
        use_container_width=True,
        hide_index=True
    )

# ----------------------------------------------------------
# ⭐ TOP RATED PAGE
# ----------------------------------------------------------
elif selected_page == "⭐ Top Rated":
    st.markdown("<h1 class='gradient-text animate-slide'>Top Rated Movies</h1>", unsafe_allow_html=True)
    st.write("Browse highest-rated films based on global average ratings and minimum vote filters.")
    
    min_votes = st.slider("Minimum Votes Threshold", 1, 50, 15)
    
    # Plotly Chart
    st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
    top_rated_fig = plot_highest_rated_movies(movies_df, ratings_df, min_ratings=min_votes, top_n=15)
    st.plotly_chart(top_rated_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------
# 🔥 POPULAR MOVIES PAGE
# ----------------------------------------------------------
elif selected_page == "🔥 Popular Movies":
    st.markdown("<h1 class='gradient-text animate-slide'>Popular Movies</h1>", unsafe_allow_html=True)
    st.write("Browse the most-reviewed films in the dataset (highest count of ratings).")
    
    # Plotly Chart
    st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
    popular_fig = plot_popular_movies(movies_df, ratings_df, top_n=15)
    st.plotly_chart(popular_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------
# 🧠 RECOMMENDATION INSIGHTS PAGE
# ----------------------------------------------------------
elif selected_page == "🧠 Recommendation Insights":
    st.markdown("<h1 class='gradient-text animate-slide'>Recommendation Insights</h1>", unsafe_allow_html=True)
    st.write("Analyze semantic relationships, metadata similarity networks, and rating heatmaps.")
    
    st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top: 0; color: #00FFFF;'>Similarity Heatmap Creator</h3>", unsafe_allow_html=True)
    st.write("Select multiple movies below to plot their similarity matrix heatmap based on text metadata.")
    
    seed_selections = ["Inception", "The Dark Knight", "Interstellar", "The Matrix", "Titanic", "Avatar", "Jurassic Park"]
    default_opts = [t for t in seed_selections if t in movies_df["title"].values]
    
    heatmap_movies = st.multiselect(
        "Select movies to plot (Min 2, Max 15):",
        options=sorted(movies_df["title"].tolist()),
        default=default_opts
    )
    
    if len(heatmap_movies) >= 2:
        h_fig = plot_similarity_heatmap(
            heatmap_movies,
            hybrid_engine.content_rec.cosine_sim,
            hybrid_engine.content_rec.movie_to_idx
        )
        st.plotly_chart(h_fig, use_container_width=True)
    else:
        st.info("Please select at least 2 movies to generate the correlation heatmap.")
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='glass-container'>
            <h3 style='color: #8A2BE2; margin-top: 0;'>Vector Space Analysis</h3>
            <p>
                <b>TF-IDF Vector Space:</b> The Content-Based model converts movie features into a high-dimensional vector space. 
                Common terms (e.g. director names, genre labels) are weighted using Term Frequency-Inverse Document Frequency (TF-IDF), 
                penalizing generic terms (like "the" or "movie") and emphasizing specific identifiers (like "nolan" or "cyborg").
            </p>
            <p>
                <b>Latent Semantic Matrix:</b> Collaborative Filtering uses TruncatedSVD to decompose the User-Item matrix. 
                It projects movie and user ratings onto 15 latent vector dimensions representing implicit cinematic styles (e.g., fast paced, slow drama) 
                rather than raw categories.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class='glass-container'>
            <h3 style='color: #FF1493; margin-top: 0;'>Recommendation Confidence Index</h3>
            <p>
                Our <b>Confidence Index</b> combines content matching scores with ratings sample sizes. 
                A recommendation receives <b>90%+ confidence</b> when:
            </p>
            <ul>
                <li>The movie overlaps significantly in multiple metadata attributes (director + cast + genres).</li>
                <li>There is high consensus in item rating patterns (similar user vectors).</li>
                <li>The recommended movie has 15+ ratings, indicating low rating prediction variance.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------------------------------------
# 📊 MODEL COMPARISON PAGE
# ----------------------------------------------------------
elif selected_page == "📊 Model Comparison":
    st.markdown("<h1 class='gradient-text animate-slide'>Recommender Model Comparison</h1>", unsafe_allow_html=True)
    st.write("Compare validation error metrics and execution profiles between Content-Based, Collaborative (SVD), and Hybrid recommenders.")
    
    # 2 Columns for charts
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        comp_fig = plot_model_comparison(evaluation_metrics)
        st.plotly_chart(comp_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_c2:
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        lat_fig = plot_model_latency(evaluation_metrics)
        st.plotly_chart(lat_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Table comparison
    st.markdown("### Metric Comparison Table")
    metrics_tbl = pd.DataFrame(evaluation_metrics).T
    st.dataframe(metrics_tbl, use_container_width=True)
    
    st.markdown("""
    <div class='glass-container'>
        <h4 style='color: #00FFFF;'>Metric Glossary</h4>
        <ul>
            <li><b>RMSE (Root Mean Squared Error):</b> Meaures rating prediction deviations. Lower is better. Collaborative SVD achieves RMSE ~0.8-0.9 on test sets, which is ensembled to score candidate ranks.</li>
            <li><b>MAE (Mean Absolute Error):</b> Measures average absolute rating error.</li>
            <li><b>Catalog Coverage:</b> The percentage of total catalog movies recommended at least once in a randomized batch of queries. High coverage prevents popularity bias and increases catalog exploration.</li>
            <li><b>Query Latency:</b> The time in milliseconds taken to compute recommendations. Content-based is extremely fast, while the hybrid pipeline aggregates ratings and text similarity matrices.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------
# 📂 DATASET EXPLORER
# ----------------------------------------------------------
elif selected_page == "📂 Dataset Explorer":
    st.markdown("<h1 class='gradient-text animate-slide'>Dataset Explorer</h1>", unsafe_allow_html=True)
    st.write("Browse and filter the movie database catalog.")
    
    # Search controls
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        s_title = st.text_input("Search Title:")
    with col_f2:
        s_genre = st.multiselect("Filter Genres:", options=sorted(list(set(movies_df["genres"].str.split("|").explode().dropna().unique()))))
    with col_f3:
        s_year_range = st.slider("Release Year:", 1975, 2024, (1975, 2024))
        
    # Filter dataset
    filtered_movies = movies_df.copy()
    
    if s_title:
        filtered_movies = filtered_movies[filtered_movies["title"].str.contains(s_title, case=False)]
        
    if s_genre:
        # Check if movie contains any of selected genres
        genre_mask = filtered_movies["genres"].apply(lambda g: any(sg in g for sg in s_genre))
        filtered_movies = filtered_movies[genre_mask]
        
    filtered_movies = filtered_movies[
        (filtered_movies["release_year"] >= s_year_range[0]) & 
        (filtered_movies["release_year"] <= s_year_range[1])
    ]
    
    # Merge with ratings average to show rating count/average
    avg_rats = ratings_df.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        votes=("rating", "count")
    ).reset_index()
    
    explored_df = pd.merge(filtered_movies, avg_rats, on="movieId", how="left").fillna(0)
    
    explored_view = explored_df[[
        "movieId", "title", "release_year", "genres", "director", "cast", "avg_rating", "votes"
    ]].rename(columns={
        "movieId": "Movie ID",
        "title": "Title",
        "release_year": "Release Year",
        "genres": "Genres",
        "director": "Director",
        "cast": "Cast",
        "avg_rating": "Avg Rating",
        "votes": "Total Ratings"
    })
    
    st.dataframe(explored_view, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(explored_view)} out of {len(movies_df)} movies.")

# ----------------------------------------------------------
# 📜 RECOMMENDATION HISTORY PAGE
# ----------------------------------------------------------
elif selected_page == "📜 Recommendation History":
    st.markdown("<h1 class='gradient-text animate-slide'>Recommendation Query History</h1>", unsafe_allow_html=True)
    st.write("Browse logs of past recommendation queries executed during the current session or saved local runs.")
    
    history_df = hybrid_engine.get_history()
    
    if history_df.empty:
        st.info("No recommendation history found yet. Go to 'Recommend Movies' page to execute queries!")
    else:
        # Display history
        st.dataframe(history_df.sort_index(ascending=False), use_container_width=True, hide_index=True)
        
        # Download and clear controls
        col_ex, col_cl = st.columns(2)
        with col_ex:
            csv_hist = history_df.to_csv(index=False)
            st.download_button(
                label="📥 Export Query History as CSV",
                data=csv_hist,
                file_name="recommendation_query_history.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_cl:
            if st.button("🗑️ Clear Local History Logs", use_container_width=True):
                hybrid_engine.clear_history()
                st.session_state.total_recommendations = 0
                st.success("Log history wiped successfully.")
                st.rerun()

# ----------------------------------------------------------
# ⚙️ SYSTEM PERFORMANCE
# ----------------------------------------------------------
elif selected_page == "⚙️ System Performance":
    st.markdown("<h1 class='gradient-text animate-slide'>System Performance Diagnostics</h1>", unsafe_allow_html=True)
    st.write("Examine computational footprints, library mappings, and model serialization attributes.")
    
    # Calculate sizes
    model_size_mb = 0.0
    if os.path.exists("model/hybrid_recommender.joblib"):
        model_size_mb = os.path.getsize("model/hybrid_recommender.joblib") / (1024 * 1024)
        
    data_size_mb = 0.0
    if os.path.exists("data/movies.csv"):
        data_size_mb += os.path.getsize("data/movies.csv") / (1024 * 1024)
    if os.path.exists("data/ratings.csv"):
        data_size_mb += os.path.getsize("data/ratings.csv") / (1024 * 1024)
        
    # Get memory footprint using sys / estimate
    memory_footprint_mb = sys.getsizeof(movies_df) + sys.getsizeof(ratings_df) + sys.getsizeof(hybrid_engine.user_item_matrix)
    memory_footprint_mb = memory_footprint_mb / (1024 * 1024)
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(f"""
        <div class='glass-container'>
            <h3 style='color: #00FFFF; margin-top: 0;'>Storage Diagnostics</h3>
            <table style='width: 100%; border-collapse: collapse; margin-top: 15px;'>
                <tr>
                    <td style='padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05); color:#a0a0c0;'>Model Size (joblib)</td>
                    <td style='padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-weight:bold; text-align:right;'>{model_size_mb:.2f} MB</td>
                </tr>
                <tr>
                    <td style='padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05); color:#a0a0c0;'>Dataset CSV files</td>
                    <td style='padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-weight:bold; text-align:right;'>{data_size_mb:.2f} MB</td>
                </tr>
                <tr>
                    <td style='padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05); color:#a0a0c0;'>Est. Runtime RAM Footprint</td>
                    <td style='padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-weight:bold; text-align:right; color:#00FFFF;'>{memory_footprint_mb:.2f} MB</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
    with col_s2:
        st.markdown(f"""
        <div class='glass-container'>
            <h3 style='color: #8A2BE2; margin-top: 0;'>Pipeline Settings</h3>
            <table style='width: 100%; border-collapse: collapse; margin-top: 15px;'>
                <tr>
                    <td style='padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05); color:#a0a0c0;'>TF-IDF Vectorizer</td>
                    <td style='padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-weight:bold; text-align:right;'>TfidfVectorizer (English)</td>
                </tr>
                <tr>
                    <td style='padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05); color:#a0a0c0;'>Collaborative Model</td>
                    <td style='padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-weight:bold; text-align:right;'>Nearest Neighbors (Cosine)</td>
                </tr>
                <tr>
                    <td style='padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05); color:#a0a0c0;'>Decomposition Method</td>
                    <td style='padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-weight:bold; text-align:right;'>TruncatedSVD (15 comps)</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------------------------------------
# 👨💻 ABOUT DEVELOPER
# ----------------------------------------------------------
elif selected_page == "👨💻 About Developer":
    st.markdown("<h1 class='gradient-text animate-slide'>About the Developer</h1>", unsafe_allow_html=True)
    
    col_d1, col_d2 = st.columns([1, 2])
    
    with col_d1:
        st.markdown("""
        <div class='glass-container' style='text-align: center;'>
            <div style='font-size: 80px; margin-bottom: 15px;'>👨💻</div>
            <h3 style='margin: 0; color: #00FFFF;'>Raghav Verma</h3>
            <p style='color: #FF1493; font-weight: 600; font-size: 14px; margin-top: 5px; margin-bottom: 20px;'>AI & Machine Learning Developer</p>
            <div style='text-align: left; font-size: 13px; line-height: 1.6;'>
                <b>Intern ID:</b> CITS2709<br>
                <b>Institution:</b> MRIIRS<br>
                <b>Location:</b> Faridabad, India
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_d2:
        st.markdown("""
        <div class='glass-container'>
            <h3 style='color: #8A2BE2; margin-top: 0;'>Profile Summary</h3>
            <p>
                I am a focused B.Tech Computer Science and Engineering student specializing in Artificial Intelligence and Machine Learning 
                at Manav Rachna International Institute of Research and Studies (MRIIRS). 
                This project represents a complete submission for my Machine Learning / AI Internship.
            </p>
            <h4 style='color: #00FFFF;'>Key Project Achievements:</h4>
            <ul>
                <li>Implemented modular Python package structures following production software development guidelines.</li>
                <li>Ensembled dual recommender styles (Content Similarity and Rating Collaborative filtering) into a high-performance Hybrid Engine.</li>
                <li>Designed beautiful dark-theme glassmorphism layout stylesheets in Streamlit for professional UI visuals.</li>
                <li>Engineered offline data caching mechanisms and synthetic realistic generators to allow complete offline runtime.</li>
            </ul>
            <h4 style='color: #FF1493;'>Technical Toolset:</h4>
            <div>
                <span class='custom-badge'>Python</span>
                <span class='custom-badge'>Machine Learning</span>
                <span class='custom-badge'>Scikit-Learn</span>
                <span class='custom-badge'>Pandas / NumPy</span>
                <span class='custom-badge'>Streamlit Web Development</span>
                <span class='custom-badge'>Plotly Visualizations</span>
                <span class='custom-badge'>Git Version Control</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==========================================================
# FOOTER
# ==========================================================
st.markdown("""
<div class='footer'>
    <p>Built with Python • Streamlit • Scikit-Learn • Plotly</p>
    <p style='color: #454565; margin-top: 5px;'>Developed by <b>Raghav Verma</b> | B.Tech CSE (AI & ML) | MRIIRS | Intern ID: <b>CITS2709</b></p>
</div>
""", unsafe_allow_html=True)
