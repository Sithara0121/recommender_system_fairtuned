# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import linear_kernel
# import numpy as np

# class ContentRecommender:
#     def __init__(self):
#         self.track_metadata = None
#         self.tfidf_matrix = None
#         self.track_idx_map = {}
#         self.rev_track_idx_map = {}

#     def prepare(self, df_tracks: pd.DataFrame):
#         # Combine relevant textual features into one string per track
#         df_tracks['combined_features'] = (
#             df_tracks['artist_name'].fillna('') + ' ' +
#             df_tracks['album_name'].fillna('') + ' ' +
#             # df_tracks['genres'].fillna('') + ' ' +
#             df_tracks['track_name'].fillna('')
#         )

#         self.track_metadata = df_tracks.reset_index(drop=True)
#         self.track_idx_map = {uri: idx for idx, uri in enumerate(self.track_metadata['track_uri'])}
#         self.rev_track_idx_map = {idx: uri for uri, idx in self.track_idx_map.items()}

#         # Vectorize combined text features using TF-IDF
#         vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
#         self.tfidf_matrix = vectorizer.fit_transform(self.track_metadata['combined_features'])

#     def recommend_tracks(self, playlist_track_uris, top_n=10):
#         # Map playlist track URIs to indices
#         idxs = [self.track_idx_map[uri] for uri in playlist_track_uris if uri in self.track_idx_map]

#         if not idxs:
#             print("No valid tracks found in playlist for content-based recommendation.")
#             return []

#         # Average TF-IDF vectors of playlist tracks
#         playlist_vec = self.tfidf_matrix[idxs].mean(axis=0)
#         playlist_vec = np.asarray(playlist_vec)  # ✅ convert np.matrix to np.ndarray
#         cosine_sim = linear_kernel(playlist_vec, self.tfidf_matrix).flatten()


#         # Sort by similarity and exclude tracks already in playlist
#         similar_idxs = cosine_sim.argsort()[::-1]

#         recommendations = []
#         for idx in similar_idxs:
#             track_uri = self.rev_track_idx_map[idx]
#             if track_uri not in playlist_track_uris:
#                 recommendations.append(track_uri)
#             if len(recommendations) >= top_n:
#                 break

#         return recommendations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

class ContentRecommender:
    def __init__(self):
        self.track_metadata = None
        self.tfidf_matrix = None
        self.track_idx_map = {}
        self.rev_track_idx_map = {}

    def prepare(self, df_tracks: pd.DataFrame):
        # Combine relevant textual features into one string per track
        df_tracks['combined_features'] = (
            df_tracks['artist_name'].fillna('') + ' ' +
            df_tracks['album_name'].fillna('') + ' ' +
            df_tracks['track_name'].fillna('')
        )

        # Normalize track URIs in metadata
        df_tracks['track_uri'] = df_tracks['track_uri'].str.lower().str.strip()

        self.track_metadata = df_tracks.reset_index(drop=True)
        self.track_idx_map = {uri: idx for idx, uri in enumerate(self.track_metadata['track_uri'])}
        self.rev_track_idx_map = {idx: uri for uri, idx in self.track_idx_map.items()}

        # Vectorize combined text features using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = vectorizer.fit_transform(self.track_metadata['combined_features'])

    def recommend_tracks(self, playlist_track_uris, top_n=10):
        # Normalize incoming URIs
        playlist_track_uris = [uri.lower().strip() for uri in playlist_track_uris]

        # Map URIs to indices
        idxs = [self.track_idx_map[uri] for uri in playlist_track_uris if uri in self.track_idx_map]

        if not idxs:
            print("⚠️ No valid tracks found in playlist for content-based recommendation.")
            return []

        # Average TF-IDF vectors of playlist tracks
        playlist_vec = self.tfidf_matrix[idxs].mean(axis=0)
        playlist_vec = np.asarray(playlist_vec)
        cosine_sim = linear_kernel(playlist_vec, self.tfidf_matrix).flatten()

        # Sort by similarity and exclude existing playlist tracks
        similar_idxs = cosine_sim.argsort()[::-1]

        recommendations = []
        for idx in similar_idxs:
            track_uri = self.rev_track_idx_map[idx]
            if track_uri not in playlist_track_uris:
                recommendations.append(track_uri)
            if len(recommendations) >= top_n:
                break

        return recommendations

