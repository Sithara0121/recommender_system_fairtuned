import numpy as np
import pandas as pd

class Reranker:
    def __init__(self, track_popularity: pd.Series, popularity_weight=0.7, diversity_weight=0.3):
        """
        track_popularity: pd.Series indexed by track_uri, values are popularity scores (e.g., play counts)
        popularity_weight: how much to penalize popularity (higher = more penalty)
        diversity_weight: how much to encourage diversity (higher = more diversity)
        """
        self.track_popularity = track_popularity
        self.popularity_weight = popularity_weight
        self.diversity_weight = diversity_weight

    def rerank(self, recommended_tracks: list, original_scores: list):
        """
        recommended_tracks: list of track_uris
        original_scores: list of floats (scores from hybrid model)

        Returns: reranked list of track_uris
        """
        # Normalize popularity (0 to 1)
        pop_scores = self.track_popularity.reindex(recommended_tracks).fillna(0)
        pop_norm = (pop_scores - pop_scores.min()) / (pop_scores.max() - pop_scores.min() + 1e-9)

        # Diversity: simple example - penalize repeated artists
        # For demo, let's assume we have a mapping track_uri -> artist_name
        # This mapping should be provided or loaded externally
        # Here we assume self.track_artist_map is a dict {track_uri: artist_name}
        
        # Calculate artist diversity penalty (repeat artists get higher penalty)
        artist_counts = {}
        diversity_penalty = []
        for track in recommended_tracks:
            artist = self.track_artist_map.get(track, "unknown")
            count = artist_counts.get(artist, 0)
            diversity_penalty.append(count)
            artist_counts[artist] = count + 1
        diversity_penalty = np.array(diversity_penalty)
        diversity_norm = (diversity_penalty - diversity_penalty.min()) / (diversity_penalty.max() - diversity_penalty.min() + 1e-9)

        # Combine scores: higher original score is better, penalize popularity and diversity penalty
        combined_score = np.array(original_scores) - self.popularity_weight * pop_norm - self.diversity_weight * diversity_norm

        # Get reranked indices (descending)
        reranked_indices = np.argsort(-combined_score)

        reranked_tracks = [recommended_tracks[i] for i in reranked_indices]
        return reranked_tracks

    def set_artist_map(self, track_artist_map):
        self.track_artist_map = track_artist_map
