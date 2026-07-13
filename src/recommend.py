import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from typing import List, Tuple, Dict, Any

class ContentBasedRecommender:
    """
    Recommends movies based on content similarity.
    Uses TF-IDF Vectorization and Cosine Similarity on metadata (genres, cast, keywords, directors, etc).
    """
    def __init__(self, stop_words: str = "english"):
        self.vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 2))
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.movie_to_idx = {}
        self.idx_to_movie = {}

    def fit(self, movies_df: pd.DataFrame) -> 'ContentBasedRecommender':
        """
        Fits the TF-IDF vectorizer and computes the cosine similarity matrix.
        """
        # Ensure metadata_soup exists
        if "metadata_soup" not in movies_df.columns:
            raise ValueError("Input DataFrame must contain 'metadata_soup' column. Run preprocessing first.")
        
        self.tfidf_matrix = self.vectorizer.fit_transform(movies_df["metadata_soup"])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # Build index mappings
        self.movie_to_idx = {row["title"].strip().lower(): idx for idx, row in movies_df.iterrows()}
        self.idx_to_movie = {idx: row["title"] for idx, row in movies_df.iterrows()}
        return self

    def recommend(self, title: str, top_n: int = 10, movies_df: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """
        Finds movies similar to the given title based on content similarity.
        """
        clean_title = title.strip().lower()
        if clean_title not in self.movie_to_idx:
            # Fallback to fuzzy or partial match
            matched = [t for t in self.movie_to_idx.keys() if clean_title in t]
            if not matched:
                return []
            clean_title = matched[0]

        movie_idx = self.movie_to_idx[clean_title]
        sim_scores = list(enumerate(self.cosine_sim[movie_idx]))
        
        # Sort by similarity descending
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Fetch top_n excluding the queried movie itself
        recommendations = []
        for idx, score in sim_scores[1:top_n+1]:
            movie_title = self.idx_to_movie[idx]
            movie_info = {}
            if movies_df is not None:
                # Add full details
                info_row = movies_df.iloc[idx]
                movie_info = {
                    "movieId": int(info_row["movieId"]),
                    "title": str(info_row["title"]),
                    "genres": str(info_row["genres"]),
                    "director": str(info_row["director"]),
                    "cast": str(info_row["cast"]),
                    "release_year": int(info_row["release_year"]),
                    "score": float(score),
                    "description": str(info_row["description"]),
                    "keywords": str(info_row["keywords"])
                }
            else:
                movie_info = {
                    "title": movie_title,
                    "score": float(score)
                }
            recommendations.append(movie_info)
            
        return recommendations


class CollaborativeRecommender:
    """
    Recommends movies using collaborative ratings.
    - Uses NearestNeighbors on the rating profiles of movies (Item-based collaborative filtering).
    - Uses TruncatedSVD (Matrix Factorization) to predict ratings and handle sparsity.
    """
    def __init__(self, metric: str = "cosine", algorithm: str = "brute"):
        self.metric = metric
        self.algorithm = algorithm
        self.nn_model = NearestNeighbors(metric=metric, algorithm=algorithm)
        self.svd_model = None
        self.user_item_matrix = None
        self.item_vectors = None
        
        # Mappings
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}
        self.user_id_to_idx = {}
        self.idx_to_user_id = {}

    def fit(self, user_item_matrix: pd.DataFrame, n_components: int = 15) -> 'CollaborativeRecommender':
        """
        Fits NearestNeighbors and TruncatedSVD models on the user-item ratings matrix.
        Applies mean-centering to improve rating prediction quality.
        """
        self.user_item_matrix = user_item_matrix
        
        # Dimensions: users = rows, items = columns
        # Transpose: rows = items (movies), columns = users (ratings)
        self.item_vectors = user_item_matrix.T
        
        # Create index mappings
        self.movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(user_item_matrix.columns)}
        self.idx_to_movie_id = {idx: movie_id for idx, movie_id in enumerate(user_item_matrix.columns)}
        
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_item_matrix.index)}
        self.idx_to_user_id = {idx: user_id for idx, user_id in enumerate(user_item_matrix.index)}
        
        # 1. Fit NearestNeighbors on movie vectors for Item-Based similarity
        self.nn_model.fit(self.item_vectors)
        
        # 2. Fit Matrix Factorization (SVD) with Mean-Centering
        # Convert user-item matrix to float array
        ratings_matrix = user_item_matrix.values.copy()
        
        # Calculate mean rating for each user, ignoring zero entries (unrated items)
        user_means = np.zeros(ratings_matrix.shape[0])
        for i in range(ratings_matrix.shape[0]):
            user_ratings = ratings_matrix[i, :]
            rated_indices = user_ratings > 0
            if np.any(rated_indices):
                user_means[i] = np.mean(user_ratings[rated_indices])
            else:
                user_means[i] = 3.0  # Fallback mean rating
                
        # Subtract user mean from observed ratings, leave unrated (zeros) as zero
        centered_matrix = np.zeros_like(ratings_matrix)
        for i in range(ratings_matrix.shape[0]):
            rated_indices = ratings_matrix[i, :] > 0
            centered_matrix[i, rated_indices] = ratings_matrix[i, rated_indices] - user_means[i]
            
        # Fit TruncatedSVD on the centered rating matrix
        n_comp = min(n_components, user_item_matrix.shape[1] - 1, user_item_matrix.shape[0] - 1)
        self.svd_model = TruncatedSVD(n_components=n_comp, random_state=42)
        
        # Transform and reconstruct centered ratings
        latent_users = self.svd_model.fit_transform(centered_matrix)
        reconstructed_centered = np.dot(latent_users, self.svd_model.components_)
        
        # Add user means back to reconstruct predicted ratings
        self.reconstructed_ratings = np.zeros_like(reconstructed_centered)
        for i in range(ratings_matrix.shape[0]):
            self.reconstructed_ratings[i, :] = reconstructed_centered[i, :] + user_means[i]
            
        # Clip final predictions to valid rating scale
        self.reconstructed_ratings = np.clip(self.reconstructed_ratings, 1.0, 5.0)
        
        return self

    def recommend_similar_movies(self, movie_id: int, top_n: int = 10, movies_df: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """
        Finds movies with similar rating patterns using NearestNeighbors.
        """
        if movie_id not in self.movie_id_to_idx:
            return []
            
        movie_idx = self.movie_id_to_idx[movie_id]
        query_vector = self.item_vectors.iloc[movie_idx].values.reshape(1, -1)
        
        # Query NearestNeighbors (fetch top_n + 1 because the item itself will be closest with distance 0)
        distances, indices = self.nn_model.kneighbors(query_vector, n_neighbors=top_n + 1)
        
        distances = distances.squeeze()
        indices = indices.squeeze()
        
        recommendations = []
        for idx, dist in zip(indices[1:], distances[1:]):
            rec_movie_id = self.idx_to_movie_id[idx]
            sim_score = 1.0 - dist  # Cosine similarity is 1 - Cosine Distance
            
            movie_info = {}
            if movies_df is not None:
                info_rows = movies_df[movies_df["movieId"] == rec_movie_id]
                if not info_rows.empty:
                    info_row = info_rows.iloc[0]
                    movie_info = {
                        "movieId": int(rec_movie_id),
                        "title": str(info_row["title"]),
                        "genres": str(info_row["genres"]),
                        "director": str(info_row["director"]),
                        "cast": str(info_row["cast"]),
                        "release_year": int(info_row["release_year"]),
                        "score": float(sim_score),
                        "description": str(info_row["description"]),
                        "keywords": str(info_row["keywords"])
                    }
            else:
                movie_info = {
                    "movieId": int(rec_movie_id),
                    "score": float(sim_score)
                }
            recommendations.append(movie_info)
            
        return recommendations

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """
        Predicts the rating a user would give to a movie using Matrix Factorization.
        """
        if user_id not in self.user_id_to_idx or movie_id not in self.movie_id_to_idx:
            return 3.0  # Default fallback rating
            
        u_idx = self.user_id_to_idx[user_id]
        m_idx = self.movie_id_to_idx[movie_id]
        
        # Reconstructed rating
        predicted = self.reconstructed_ratings[u_idx, m_idx]
        return float(predicted)
