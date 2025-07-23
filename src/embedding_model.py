# src/embedding_model.py

from gensim.models import Word2Vec
import pandas as pd
import numpy as np


class EmbeddingRecommender:
    def __init__(self, vector_size=64, window=5, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None

    def prepare_training_data(self, df: pd.DataFrame) -> list:
        """
        Group track URIs by playlist_id for Word2Vec training.
        """
        playlists = df.groupby('playlist_id')['track_uri'].apply(list).tolist()
        return playlists

    def train(self, playlists):
     self.model = Word2Vec(
        sentences=playlists,
        vector_size=self.vector_size,
        window=self.window,
        min_count=self.min_count,
        sg=1,  # skip-gram
        workers=4,
        epochs=10
     )

    def recommend_similar_tracks(self, track_uri: str, top_n: int = 10) -> list:
        """
        Recommend tracks similar to a given track.
        """
        if track_uri not in self.model.wv:
            return []
        similar = self.model.wv.most_similar(track_uri, topn=top_n)
        return [uri for uri, _ in similar]

    def recommend_for_playlist(self, playlist_tracks: list, top_n: int = 10) -> list:
        """
        Recommend tracks for a playlist by averaging its track vectors.
        """
        vectors = []
        for track in playlist_tracks:
            if track in self.model.wv:
                vectors.append(self.model.wv[track])

        if not vectors:
            return []

        mean_vec = np.mean(vectors, axis=0)
        similar = self.model.wv.similar_by_vector(mean_vec, topn=top_n + len(playlist_tracks))
        
        # Filter out existing tracks
        recommended = [uri for uri, _ in similar if uri not in playlist_tracks]
        return recommended[:top_n]
