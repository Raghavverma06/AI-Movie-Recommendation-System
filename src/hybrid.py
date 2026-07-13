import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
from src.recommend import ContentBasedRecommender, CollaborativeRecommender

class HybridRecommender:
    """
    Ensembles Content-Based and Collaborative Filtering approaches.
    Provides detailed explanations, confidence scores, and historical query logs.
    """
    def __init__(self, data_dir: str = "data", outputs_dir: str = "outputs"):
        self.content_rec = ContentBasedRecommender()
        self.collab_rec = CollaborativeRecommender()
        self.movies_df = None
        self.ratings_df = None
        self.user_item_matrix = None
        self.history_path = os.path.join(outputs_dir, "recommendation_history.csv")
        os.makedirs(outputs_dir, exist_ok=True)
        
        # Initialize history file if it doesn't exist
        if not os.path.exists(self.history_path):
            pd.DataFrame(columns=[
                "timestamp", "queried_movie", "user_id", "weight_content", 
                "weight_collab", "recommended_titles", "algorithm_used", "execution_time_ms"
            ]).to_csv(self.history_path, index=False)

    def fit(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame, user_item_matrix: pd.DataFrame) -> 'HybridRecommender':
        """
        Fits both content-based and collaborative filtering sub-models.
        """
        self.movies_df = movies_df.copy()
        self.ratings_df = ratings_df.copy()
        self.user_item_matrix = user_item_matrix.copy()
        
        # Fit models
        self.content_rec.fit(self.movies_df)
        self.collab_rec.fit(self.user_item_matrix)
        
        # Pre-calculate movie rating stats
        self.mean_ratings = self.ratings_df.groupby("movieId")["rating"].mean().to_dict()
        self.count_ratings = self.ratings_df.groupby("movieId")["rating"].count().to_dict()
        
        return self

    def get_movie_details(self, title: str) -> Tuple[int, Dict[str, Any]]:
        """
        Helper to find movieId and details from a title (case insensitive, partial matching supported).
        """
        clean_title = title.strip().lower()
        titles = self.movies_df["title"].str.strip().str.lower().tolist()
        
        if clean_title in titles:
            idx = titles.index(clean_title)
            movie_row = self.movies_df.iloc[idx]
            return int(movie_row["movieId"]), movie_row.to_dict()
            
        # Try partial match
        for idx, row in self.movies_df.iterrows():
            if clean_title in row["title"].strip().lower():
                return int(row["movieId"]), row.to_dict()
                
        raise ValueError(f"Movie '{title}' not found in database.")

    def recommend(
        self, 
        movie_title: str, 
        user_id: int = None, 
        top_n: int = 10, 
        weight_content: float = 0.5, 
        weight_collab: float = 0.5,
        weight_personal: float = 0.0
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Hybrid recommendation engine blending:
        1. Content similarity (cosine similarity of text metadata)
        2. Collaborative similarity (cosine similarity of item rating histories)
        3. Personalized rating prediction (using SVD matrix factorization, if user_id is provided)
        
        Returns a list of top_n recommended movies with scores, confidence, and explanations,
        along with the total execution time (ms).
        """
        start_time = time.time()
        
        # 1. Look up query movie
        try:
            target_id, target_row = self.get_movie_details(movie_title)
        except ValueError as e:
            print(f"Error: {e}")
            return [], 0.0
            
        target_title = target_row["title"]
        target_genres = set(target_row["genres"].split("|"))
        target_director = target_row["director"]
        target_cast = set([actor.strip().lower() for actor in target_row["cast"].split(",")])
        target_keywords = set([kw.strip().lower() for kw in target_row["keywords"].split(",")])
        
        # 2. Get raw similarity arrays
        n_movies = len(self.movies_df)
        
        # Target index in the DataFrame (assumes index aligns with content recommender)
        target_idx_cb = self.content_rec.movie_to_idx.get(target_title.lower())
        
        # Get content-based cosine similarities
        if target_idx_cb is not None:
            cb_scores = self.content_rec.cosine_sim[target_idx_cb]
        else:
            cb_scores = np.zeros(n_movies)
            
        # Get collaborative similarity score using Nearest Neighbors vectors
        collab_scores = np.zeros(n_movies)
        target_idx_cf = self.collab_rec.movie_id_to_idx.get(target_id)
        
        if target_idx_cf is not None:
            # Reconstruct item similarity profile manually or query distances
            query_vec = self.collab_rec.item_vectors.iloc[target_idx_cf].values.reshape(1, -1)
            # Find neighbors for all items to get full similarity scores
            distances, indices = self.collab_rec.nn_model.kneighbors(query_vec, n_neighbors=n_movies)
            distances = distances.squeeze()
            indices = indices.squeeze()
            
            # Pre-map movie IDs to DataFrame indices for O(1) lookups
            movie_id_to_df_idx = {row["movieId"]: idx for idx, row in self.movies_df.iterrows()}
            
            # Map distances back to movies
            for idx, dist in zip(indices, distances):
                movie_id = self.collab_rec.idx_to_movie_id[idx]
                df_idx = movie_id_to_df_idx.get(movie_id)
                if df_idx is not None:
                    collab_scores[df_idx] = 1.0 - dist
                    
        # Normalized predicted SVD rating
        personal_scores = np.zeros(n_movies)
        movie_ids_list = self.movies_df["movieId"].tolist()
        
        if user_id is not None and user_id in self.user_item_matrix.index:
            u_idx = self.collab_rec.user_id_to_idx[user_id]
            for idx, m_id in enumerate(movie_ids_list):
                m_idx = self.collab_rec.movie_id_to_idx.get(m_id)
                if m_idx is not None:
                    predicted_rating = self.collab_rec.reconstructed_ratings[u_idx, m_idx]
                    personal_scores[idx] = (predicted_rating - 1.0) / 4.0
                else:
                    personal_scores[idx] = 0.5  # fallback neutral rating (3.0 stars)
                
        # 3. Calculate Hybrid Score
        hybrid_scores = []
        for idx, m_id in enumerate(movie_ids_list):
            if m_id == target_id:
                continue  # skip target movie itself
                
            cb_s = cb_scores[idx]
            cf_s = collab_scores[idx]
            pers_s = personal_scores[idx]
            
            # Weighted average
            total_weight = weight_content + weight_collab + (weight_personal if user_id is not None else 0.0)
            if total_weight > 0:
                h_score = (
                    weight_content * cb_s + 
                    weight_collab * cf_s + 
                    (weight_personal * pers_s if user_id is not None else 0.0)
                ) / total_weight
            else:
                h_score = 0.0
                
            hybrid_scores.append((idx, m_id, h_score, cb_s, cf_s, pers_s))
            
        # Sort and select top_n
        hybrid_scores = sorted(hybrid_scores, key=lambda x: x[2], reverse=True)[:top_n]
        
        # Use pre-calculated rating stats
        mean_ratings = self.mean_ratings
        count_ratings = self.count_ratings
        
        # 4. Generate Explanations & Confidence
        recommendations = []
        for idx, m_id, h_score, cb_s, cf_s, pers_s in hybrid_scores:
            row = self.movies_df.iloc[idx]
            rec_title = row["title"]
            rec_genres = set(row["genres"].split("|"))
            rec_director = row["director"]
            rec_cast = set([actor.strip().lower() for actor in row["cast"].split(",")])
            rec_keywords = set([kw.strip().lower() for kw in row["keywords"].split(",")])
            
            # Find overlaps for explanation
            genre_overlap = target_genres.intersection(rec_genres)
            cast_overlap = target_cast.intersection(rec_cast)
            kw_overlap = target_keywords.intersection(rec_keywords)
            director_match = (target_director == rec_director) and (target_director != "Independent Director")
            
            # Construct explanation
            explanation_parts = []
            if genre_overlap:
                explanation_parts.append(f"shares the genre(s) **{', '.join(list(genre_overlap))}**")
            if director_match:
                explanation_parts.append(f"was also directed by **{target_director}**")
            if cast_overlap:
                explanation_parts.append(f"stars **{', '.join([c.title() for c in cast_overlap])}** who also starred in *{target_title}*")
            if kw_overlap:
                explanation_parts.append(f"deals with themes of **{', '.join(list(kw_overlap))}**")
                
            # Collaborative insight
            if cf_s > 0.4:
                explanation_parts.append("is rated very similarly by viewers who enjoyed this movie")
                
            if user_id is not None and pers_s > 0.6:
                pred_val = 1.0 + (pers_s * 4.0)
                explanation_parts.append(f"our AI model predicts you would rate it a high **{pred_val:.1f}/5.0**")
                
            if not explanation_parts:
                explanation_parts.append("shares subtle stylistic features and thematic elements")
                
            explanation = "This movie is recommended because it " + ", and it ".join(explanation_parts) + "."
            
            # Confidence metric based on overlapping indicators
            # Combine content similarity, collaborative similarity, and ratings sample volume
            rating_vol_factor = min(count_ratings.get(m_id, 0), 50) / 50.0  # normalize volume up to 50 ratings
            confidence = (h_score * 0.7) + (rating_vol_factor * 0.3)
            confidence_percentage = int(np.clip(confidence * 100, 45, 99))
            
            recommendations.append({
                "movieId": int(m_id),
                "title": str(rec_title),
                "genres": str(row["genres"]),
                "director": str(rec_director),
                "cast": str(row["cast"]),
                "release_year": int(row["release_year"]),
                "score": float(h_score),
                "similarity_percentage": int(h_score * 100),
                "confidence_percentage": confidence_percentage,
                "content_score": float(cb_s),
                "collab_score": float(cf_s),
                "predicted_rating": float(1.0 + (pers_s * 4.0)) if user_id is not None else float(mean_ratings.get(m_id, 3.0)),
                "avg_rating": float(mean_ratings.get(m_id, 0.0)),
                "num_ratings": int(count_ratings.get(m_id, 0)),
                "explanation": explanation,
                "description": str(row["description"]),
                "keywords": str(row["keywords"])
            })
            
        elapsed_time_ms = (time.time() - start_time) * 1000
        
        # 5. Log Query to History CSV
        self._log_history(
            queried_movie=target_title,
            user_id=user_id,
            weight_content=weight_content,
            weight_collab=weight_collab,
            recommended_movies=[rec["title"] for rec in recommendations],
            algorithm="Hybrid (Content + Collab)",
            elapsed_time_ms=elapsed_time_ms
        )
        
        return recommendations, elapsed_time_ms

    def _log_history(
        self, 
        queried_movie: str, 
        user_id: int, 
        weight_content: float, 
        weight_collab: float, 
        recommended_movies: List[str],
        algorithm: str,
        elapsed_time_ms: float
    ):
        """
        Appends query information to outputs/recommendation_history.csv.
        """
        try:
            new_row = pd.DataFrame([{
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "queried_movie": queried_movie,
                "user_id": user_id if user_id is not None else "Anonymous",
                "weight_content": weight_content,
                "weight_collab": weight_collab,
                "recommended_titles": " | ".join(recommended_movies),
                "algorithm_used": algorithm,
                "execution_time_ms": round(elapsed_time_ms, 2)
            }])
            
            new_row.to_csv(self.history_path, mode="a", header=False, index=False)
        except Exception as e:
            print(f"Failed to log query history: {e}")
            
    def get_history(self) -> pd.DataFrame:
        """
        Reads and returns recommendation log history.
        """
        if os.path.exists(self.history_path):
            try:
                return pd.read_csv(self.history_path)
            except Exception as e:
                print(f"Error reading history file: {e}")
        return pd.DataFrame()
        
    def clear_history(self):
        """
        Clears query log history file.
        """
        try:
            pd.DataFrame(columns=[
                "timestamp", "queried_movie", "user_id", "weight_content", 
                "weight_collab", "recommended_titles", "algorithm_used", "execution_time_ms"
            ]).to_csv(self.history_path, index=False)
        except Exception as e:
            print(f"Failed to clear history: {e}")
