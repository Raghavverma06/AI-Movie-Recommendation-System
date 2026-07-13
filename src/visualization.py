import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any

# Custom Color Palette to match the dark premium styling
PALETTE = {
    "primary": "#8A2BE2",      # Blue Violet
    "secondary": "#00FFFF",    # Cyan
    "accent": "#FF1493",       # Deep Pink
    "background": "rgba(10, 10, 20, 0)", # Transparent backdrop
    "card_bg": "rgba(255, 255, 255, 0.05)",
    "grid": "rgba(255, 255, 255, 0.1)",
    "text": "#E0E0FF"
}

def apply_dark_theme(fig: go.Figure) -> go.Figure:
    """
    Applies unified dark theme styling to a Plotly figure.
    """
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PALETTE["background"],
        plot_bgcolor=PALETTE["background"],
        font=dict(color=PALETTE["text"], family="Inter, sans-serif"),
        title_font=dict(size=18, color=PALETTE["secondary"], family="Outfit, sans-serif"),
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(
            gridcolor=PALETTE["grid"],
            linecolor=PALETTE["grid"],
            tickfont=dict(color=PALETTE["text"])
        ),
        yaxis=dict(
            gridcolor=PALETTE["grid"],
            linecolor=PALETTE["grid"],
            tickfont=dict(color=PALETTE["text"])
        )
    )
    return fig

def plot_genre_distribution(movies_df: pd.DataFrame) -> go.Figure:
    """
    Plots the frequency of movies per genre.
    """
    # Split genres and count
    genres_series = movies_df["genres"].str.split("|").explode()
    genre_counts = genres_series.value_counts().reset_index()
    genre_counts.columns = ["Genre", "Count"]
    
    fig = px.bar(
        genre_counts, 
        x="Count", 
        y="Genre", 
        orientation="h",
        color="Count",
        color_continuous_scale="Electric",
        title="Movie Count by Genre"
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return apply_dark_theme(fig)

def plot_ratings_distribution(ratings_df: pd.DataFrame) -> go.Figure:
    """
    Plots the frequency of rating values (1.0 to 5.0).
    """
    rating_counts = ratings_df["rating"].value_counts().reset_index()
    rating_counts.columns = ["Rating", "Count"]
    rating_counts = rating_counts.sort_values(by="Rating")
    
    fig = px.bar(
        rating_counts, 
        x="Rating", 
        y="Count",
        color="Count",
        color_continuous_scale="Plasma",
        title="Distribution of Movie Ratings"
    )
    fig.update_xaxes(tickmode="linear", tick0=0.5, dtick=0.5)
    return apply_dark_theme(fig)

def plot_top_genres(movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> go.Figure:
    """
    Plots the average rating per movie genre.
    """
    # Explode genres
    exploded_movies = movies_df[["movieId", "genres"]].copy()
    exploded_movies["genre"] = exploded_movies["genres"].str.split("|")
    exploded_movies = exploded_movies.explode("genre")
    
    # Merge with ratings
    merged = pd.merge(exploded_movies, ratings_df, on="movieId")
    genre_ratings = merged.groupby("genre")["rating"].mean().reset_index()
    genre_ratings.columns = ["Genre", "Average Rating"]
    genre_ratings = genre_ratings.sort_values(by="Average Rating", ascending=False)
    
    fig = px.bar(
        genre_ratings,
        x="Genre",
        y="Average Rating",
        color="Average Rating",
        color_continuous_scale="Viridis",
        title="Average Rating by Genre"
    )
    fig.update_yaxes(range=[1.0, 5.0])
    return apply_dark_theme(fig)

def plot_popular_movies(movies_df: pd.DataFrame, ratings_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Plots the most popular movies based on number of ratings.
    """
    rating_counts = ratings_df["movieId"].value_counts().reset_index()
    rating_counts.columns = ["movieId", "Ratings Count"]
    
    merged = pd.merge(rating_counts, movies_df, on="movieId")
    top_movies = merged.head(top_n).sort_values(by="Ratings Count", ascending=True)
    
    fig = px.bar(
        top_movies,
        x="Ratings Count",
        y="title",
        orientation="h",
        color="Ratings Count",
        color_continuous_scale="Cividis",
        title=f"Top {top_n} Most Popular Movies (By Rating Count)"
    )
    return apply_dark_theme(fig)

def plot_highest_rated_movies(movies_df: pd.DataFrame, ratings_df: pd.DataFrame, min_ratings: int = 15, top_n: int = 10) -> go.Figure:
    """
    Plots the highest-rated movies with at least min_ratings votes.
    """
    stats = ratings_df.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        ratings_count=("rating", "count")
    ).reset_index()
    
    filtered_stats = stats[stats["ratings_count"] >= min_ratings]
    merged = pd.merge(filtered_stats, movies_df, on="movieId")
    top_rated = merged.sort_values(by="avg_rating", ascending=False).head(top_n)
    top_rated = top_rated.sort_values(by="avg_rating", ascending=True)
    
    fig = px.bar(
        top_rated,
        x="avg_rating",
        y="title",
        orientation="h",
        color="avg_rating",
        color_continuous_scale="Hot",
        title=f"Top {top_n} Highest Rated Movies (Min {min_ratings} Ratings)"
    )
    fig.update_xaxes(range=[1.0, 5.0])
    return apply_dark_theme(fig)

def plot_movies_per_year(movies_df: pd.DataFrame) -> go.Figure:
    """
    Plots the number of movies released per year.
    """
    year_counts = movies_df["release_year"].value_counts().reset_index()
    year_counts.columns = ["Year", "Count"]
    year_counts = year_counts.sort_values(by="Year")
    
    fig = px.line(
        year_counts,
        x="Year",
        y="Count",
        title="Movie Releases by Year",
        markers=True
    )
    fig.update_traces(line=dict(color=PALETTE["secondary"], width=2.5))
    return apply_dark_theme(fig)

def plot_similarity_heatmap(
    titles: List[str], 
    cosine_sim_matrix: np.ndarray, 
    movie_to_idx_dict: Dict[str, int]
) -> go.Figure:
    """
    Plots a similarity matrix heatmap for a list of movie titles.
    """
    indices = []
    valid_titles = []
    
    for t in titles:
        clean_t = t.strip().lower()
        if clean_t in movie_to_idx_dict:
            indices.append(movie_to_idx_dict[clean_t])
            valid_titles.append(t)
            
    if len(indices) < 2:
        # Returns empty figure if not enough valid titles
        fig = go.Figure()
        fig.update_layout(title="Not enough valid movies selected for heatmap.")
        return apply_dark_theme(fig)
        
    # Extract sub-matrix
    sub_matrix = cosine_sim_matrix[np.ix_(indices, indices)]
    
    fig = px.imshow(
        sub_matrix,
        labels=dict(x="Movie Title", y="Movie Title", color="Cosine Similarity"),
        x=valid_titles,
        y=valid_titles,
        color_continuous_scale="Purples",
        title="Content Similarity Heatmap"
    )
    return apply_dark_theme(fig)

def plot_model_comparison(metrics: Dict[str, Dict[str, float]]) -> go.Figure:
    """
    Generates a grouped bar chart comparing performance metrics across different recommendation algorithms.
    """
    # Create lists for DataFrame conversion
    models = list(metrics.keys())
    rmse_vals = [metrics[m]["RMSE"] for m in models]
    mae_vals = [metrics[m]["MAE"] for m in models]
    coverage_vals = [metrics[m]["Coverage"] for m in models]
    latency_vals = [metrics[m]["Latency_ms"] for m in models]
    
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=models, 
        y=rmse_vals, 
        name="RMSE (Lower is Better)",
        marker_color="#8A2BE2"
    ))
    fig.add_trace(go.Bar(
        x=models, 
        y=mae_vals, 
        name="MAE (Lower is Better)",
        marker_color="#FF1493"
    ))
    fig.add_trace(go.Bar(
        x=models, 
        y=[c/100.0 for c in coverage_vals], # Scale coverage to [0.0, 1.0] for double axis or side-by-side comparison
        name="Catalog Coverage (Ratio - Higher is Better)",
        marker_color="#00FFFF"
    ))
    
    fig.update_layout(
        barmode='group',
        title="Model Comparison: Rating Errors & Catalog Coverage",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return apply_dark_theme(fig)

def plot_model_latency(metrics: Dict[str, Dict[str, float]]) -> go.Figure:
    """
    Generates a bar chart comparing recommendation query latency in milliseconds.
    """
    models = list(metrics.keys())
    latencies = [metrics[m]["Latency_ms"] for m in models]
    
    fig = px.bar(
        x=models,
        y=latencies,
        color=latencies,
        color_continuous_scale="Mint",
        labels={"x": "Recommendation Algorithm", "y": "Query Inference Latency (ms)"},
        title="Recommendation Latency Benchmark (ms)"
    )
    fig.update_traces(width=0.4)
    return apply_dark_theme(fig)
