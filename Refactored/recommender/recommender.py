import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


# Manhwa Recommender System
class Recommender:
    """
    A Hybrid Recommendation Engine for Manhwas.
    Uses BERT embeddings (Cosine Similarity) for semantic emotional matching, 
    and categorical tags (Jaccard Similarity) for strict genre guardrails.
    """
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.df = None
        self.embeddings = None
        self._load_artifacts()
        
    
    def _load_artifacts(self):
        """Loads the DataFrame and NumPy matrices into memory."""
        data_path = self.artifacts_dir / "clean_manhwa_data.parquet"
        embeddings_path = self.artifacts_dir / "synopsis_embeddings.npy"

        try:
            self.df = pd.read_parquet(data_path)
            self.embeddings = np.load(embeddings_path)
            logger.info(f"Successfully loaded {len(self.df)} records. Embedding matrix shape: {self.embeddings.shape}")
        except FileNotFoundError as e:
            logger.error(f"Missing artifacts in '{self.artifacts_dir}'. Please run the data pipeline first.")
            raise e
        
        
    def _calculate_jaccard(self, genres_a: str, genres_b: str) -> float:
        """Calculates Jaccard Similarity between two comma-separated genre strings."""
        if pd.isna(genres_a) or pd.isna(genres_b):
            return 0.0
            
        # Clean, lowercase, and split into sets
        set_a = set([g.strip().lower() for g in str(genres_a).split(',') if g.strip()])
        set_b = set([g.strip().lower() for g in str(genres_b).split(',') if g.strip()])
        
        if not set_a or not set_b:
            return 0.0
            
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        
        return intersection / union
    

    def recommend(self, title: str, top_n: int = 5, cosine_weight: float = 0.6, jaccard_weight: float = 0.4) -> Dict:
        """
        Retrieves the top n most semantically similar Manhwas.
        
        Args:
            title (str): The search query (case-insensitive).
            top_n (int): Number of recommendations to return.
            
        Returns:
            Dict: A dictionary containing the recommended Manhwas or an error message.
        """
        # Check if the title is in the dataset
        # 1. Case-insensitive exact match
        match = self.df[self.df['title'].str.lower() == title.lower()]
        if match.empty:
            logger.error(f"Title '{title}' not found in the dataset.")
            return {"error": f"Title '{title}' not found in the dataset.", "status_code": 404}
        
        if self.df is None or self.embeddings is None:
            return {"error": "Recommender is not properly initialized with data.", "status_code": 503}
        
        query_idx = match.index[0]
        query_vector = self.embeddings[query_idx]
        actual_title = match.iloc[0]['title']
        query_genres = match.iloc[0]['genres']
        
        
        # 2. Vectorized Cosine Similarity
        # (A . B) / (||A|| * ||B||)
        dot_products = np.dot(self.embeddings, query_vector)
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vector)
        cosine_similarities = dot_products / (norms + 1e-9) # Add epsilon to prevent division by zero

        # 3. Get top 100 matches, excluding the query itself
        pool_size = min(100, len(self.df))
        candidate_indices = np.argsort(cosine_similarities)[-(pool_size + 1):][::-1]
        candidate_indices = candidate_indices[candidate_indices != query_idx]
        
        # Remove the searched title itself from the results
        candidate_indices = candidate_indices[candidate_indices != query_idx][:top_n]
        
        # 4. Hybrid Reranking using cosine and jaccard similarity
        results_pool = []
        for idx in candidate_indices:
            cand_genres = self.df.iloc[idx]['genres']
            cosine_score = float(cosine_similarities[idx])
            
            # Calculate Jaccard penalty
            jaccard_score = self._calculate_jaccard(query_genres, cand_genres)
            
            # The Hybrid Formula
            hybrid_score = (cosine_score * cosine_weight) + (jaccard_score * jaccard_weight)
            
            results_pool.append({
                "idx": idx,
                "title": self.df.iloc[idx]['title'],
                "genres": cand_genres,
                "hybrid_score": round(hybrid_score, 4),
                "cosine_score": round(cosine_score, 4),
                "jaccard_score": round(jaccard_score, 4),
                "synopsis": self.df.iloc[idx]['synopsis']
            })
        
        # 5. Sort by the new hybrid score and truncate to Top K
        results_pool.sort(key=lambda x: x['hybrid_score'], reverse=True)
        top_k_results = results_pool[:top_n]

        # Clean up the output to hide the internal indices
        final_results = []
        for rank, res in enumerate(top_k_results, 1):
            res["rank"] = rank
            del res["idx"]
            final_results.append(res)

        return {
            "query": actual_title,
            "query_genres": query_genres,
            "weights": {"cosine": cosine_weight, "jaccard": jaccard_weight},
            "results_count": len(final_results),
            "recommendations": final_results,
            "status_code": 200
        }