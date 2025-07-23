# %%
import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from src.data_loader import MPDDataLoader

loader = MPDDataLoader(data_folder='../data/')
playlists = loader.load_playlists(max_files=50)  # adjust for testing
track_df = loader.build_track_df(playlists)

track_df.head()


# %%
import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
   sys.path.append(project_root) 
from src.spotify_api import SpotifyAPI

CLIENT_ID = "8d0b81b7efd94d28be4b8e5cfd9bbad5"
CLIENT_SECRET = "a26d760401044cb6bf19a0c03e444b89"
api = SpotifyAPI(CLIENT_ID, CLIENT_SECRET)
info = api.get_track_info('spotify:track:6JsZyX0Emuzy7swG21tgt7')
print(info)



# %%
track_uris = track_df['track_uri'].unique().tolist()

# %%
features = api.get_multiple_audio_features(track_uris[:10])
print(features)


# %%
# Skip audio features fetching for now due to 403 errors
# audio_features_list = spotify_api.get_audio_features_sequential(track_ids)
# audio_features_df = pd.DataFrame(audio_features_list)

# Instead, create an empty dataframe or fill with NaNs to keep pipeline working
import numpy as np
import pandas as pd

audio_features_df = pd.DataFrame(columns=[
    'id', 'danceability', 'energy', 'valence', 'tempo', # add other audio features you expect
])

# Add track_id column to merge
audio_features_df['track_id'] = []

# Prepare your track_df for merging
track_df['track_id'] = track_df['track_uri'].apply(lambda x: x.split(':')[-1])

# Merge (will keep audio feature columns empty)
track_enriched_df = pd.merge(track_df, audio_features_df, on='track_id', how='left')

# Continue with your next steps...


# %%
from embedding_model import EmbeddingGenerator

embedder = EmbeddingGenerator()

# Get track popularity mapping (this is fine)
popularity = embedder.track_popularity(track_df)

# Build playlists list: one row per playlist
playlists_df = (
    track_df.groupby('pid').first().reset_index()
)

# Add dummy name
playlists_df['name'] = 'my playlist'

# Convert to list of dicts
playlists = playlists_df.to_dict(orient='records')

# Create embeddings
playlist_embeddings = embedder.playlist_title_embeddings(playlists)

print(f"Playlist title embeddings shape: {playlist_embeddings.shape}")
print(f"Number of unique tracks: {len(popularity)}")


# %%
from src.cf_model import CFRecommender

cf = CFRecommender()
cf.train(track_df)




# Get unique playlist row indices
unique_playlist_indices = cf.playlist_codes.unique()
# playlist_idx = unique_playlist_indices[0]
# Pick first playlist category ID
first_pid = cf.playlist_categories[0]

# Map that ID to the integer code (row index)
playlist_idx = list(cf.playlist_mapping.keys())[list(cf.playlist_mapping.values()).index(first_pid)]

print(f"Using playlist idx {playlist_idx} for playlist id {first_pid}")


print(f"user_item_matrix shape: {cf.user_item_matrix.shape}")
print(f"playlist_idx: {playlist_idx}")

cf_scores = dict(cf.recommend_for_playlist(playlist_idx, N=20))

print("Top 20 CF recommendations:")
for track_uri, score in list(cf_scores.items())[:5]:
    print(f"{track_uri}: {score:.3f}")



# %%
print(audio_features_df.columns.tolist())


# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from src.content_model import ContentRecommender

# Assume audio_features_df has audio features for tracks with 'id' as track id
# and track_df has 'pid' (playlist id) and 'track_uri' columns

# Merge audio features with track_df
audio_features_df['track_id'] = audio_features_df['id']
track_df['track_id'] = track_df['track_uri'].apply(lambda x: x.split(':')[-1])
merged_df = pd.merge(track_df, audio_features_df, on='track_id', how='left')

# Select relevant audio features (make sure these columns exist in audio_features_df)
feature_cols = ['danceability', 'energy', 'valence', 'tempo']
features = merged_df[feature_cols].fillna(0)

# Normalize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Add scaled features back to merged_df
for i, col in enumerate(feature_cols):
    merged_df[col + '_scaled'] = features_scaled[:, i]

# Calculate playlist embeddings as average of track embeddings in that playlist
playlist_embeddings_df = (
    merged_df.groupby('pid')[[col + '_scaled' for col in feature_cols]].mean()
)

# Convert to numpy array (playlists x embedding_dim)
playlist_embeddings = playlist_embeddings_df.values

# Create track embeddings (scaled audio features)
track_embeddings = features_scaled

# Create track URIs list in same order as track_embeddings
track_uris = merged_df['track_uri'].tolist()

# Now, playlist_embeddings and track_embeddings have the same dimension (4 features)

# Example ContentRecommender usage
content = ContentRecommender(playlist_embeddings, track_embeddings, track_uris)

playlist_idx = 0  # for example, first playlist

content_scores = dict(content.recommend_for_playlist(playlist_idx, N=20))

print("Top 20 Content-Based recommendations:")
for track_uri, score in list(content_scores.items())[:5]:
    print(f"{track_uri}: {score:.3f}")


# %%
from src.hybrid_model import HybridRecommender

# Initialize hybrid model
hybrid = HybridRecommender(cf_model=cf, content_model=content, alpha=0.7)

# Recommend for the same playlist_idx
hybrid_scores = dict(hybrid.recommend_for_playlist(playlist_idx, N=20))

print("Top 20 Hybrid recommendations:")
for track_uri, score in list(hybrid_scores.items())[:5]:
    print(f"{track_uri}: {score:.3f}")


# %%
from src.reranker import FairnessReranker

# Initialize reranker with popularity dict
reranker = FairnessReranker(popularity, beta=0.5, epsilon=1e-6)

# Get hybrid recommendations (reuse previous)
hybrid_recs = hybrid.recommend_for_playlist(playlist_idx, N=50)

# Rerank for fairness and diversity
reranked_recs = reranker.rerank(hybrid_recs)

print("Top 20 Reranked recommendations:")
for track_uri, score in reranked_recs[:20]:
    print(f"{track_uri}: {score:.4f}")


# %%
for track_uri, _ in reranked_recs[:5]:
    print(track_uri, popularity.get(track_uri, 'Not found'))


# %%
print(list(popularity.items())[:10])


# %%
import numpy as np

def precision_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / k

def recall_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / len(relevant) if relevant else 0

def dcg_at_k(recommended, relevant, k):
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1 / np.log2(i + 2)  # i starts at 0
    return dcg

def idcg_at_k(relevant, k):
    idcg = 0.0
    n = min(len(relevant), k)
    for i in range(n):
        idcg += 1 / np.log2(i + 2)
    return idcg

def ndcg_at_k(recommended, relevant, k):
    dcg = dcg_at_k(recommended, relevant, k)
    idcg = idcg_at_k(relevant, k)
    return dcg / idcg if idcg > 0 else 0


# %%
from sklearn.metrics import ndcg_score

# Parameters
N = 20
K = 20  # Top-K for evaluation
playlist_indices = cf.playlist_codes.unique()[:10]  # use 10 playlists for demo

cf_precisions, content_precisions, hybrid_precisions, reranked_precisions = [], [], [], []
cf_recalls, content_recalls, hybrid_recalls, reranked_recalls = [], [], [], []
cf_ndcgs, content_ndcgs, hybrid_ndcgs, reranked_ndcgs = [], [], [], []

for playlist_idx in playlist_indices:
    cf_recs = list(cf.recommend_for_playlist(playlist_idx, N))
    content_recs = list(content.recommend_for_playlist(playlist_idx, N))
    hybrid_recs = list(hybrid.recommend_for_playlist(playlist_idx, N))
    reranked_recs = [t[0] for t in reranker.rerank(hybrid_recs)[:K]]
    
    # Turn into sets/lists of track URIs
    cf_uris = set([t[0] for t in cf_recs])
    content_uris = set([t[0] for t in content_recs])
    hybrid_uris = set([t[0] for t in hybrid_recs])
    
    # Ground truth tracks for this playlist
    ground_truth = set(track_df[track_df['pid'] == cf.playlist_categories[playlist_idx]]['track_uri'])
    ground_truth_list = list(ground_truth)
    
    # Precision@K = intersection / K
    cf_precisions.append(len(cf_uris & ground_truth) / K)
    content_precisions.append(len(content_uris & ground_truth) / K)
    hybrid_precisions.append(len(hybrid_uris & ground_truth) / K)
    reranked_precisions.append(len(set(reranked_recs) & ground_truth) / K)
    
    # Recall@K = intersection / number of relevant items
    denom = len(ground_truth) if len(ground_truth) > 0 else 1
    cf_recalls.append(len(cf_uris & ground_truth) / denom)
    content_recalls.append(len(content_uris & ground_truth) / denom)
    hybrid_recalls.append(len(hybrid_uris & ground_truth) / denom)
    reranked_recalls.append(len(set(reranked_recs) & ground_truth) / denom)
    
    # NDCG@K (binary relevance)
    relevance = [1 if uri in ground_truth else 0 for uri in list(cf_uris)[:K]]
    cf_ndcgs.append(ndcg_score([relevance], [relevance]))
    
    relevance = [1 if uri in ground_truth else 0 for uri in list(content_uris)[:K]]
    content_ndcgs.append(ndcg_score([relevance], [relevance]))
    
    relevance = [1 if uri in ground_truth else 0 for uri in list(hybrid_uris)[:K]]
    hybrid_ndcgs.append(ndcg_score([relevance], [relevance]))
    
    relevance = [1 if uri in ground_truth else 0 for uri in reranked_recs]
    reranked_ndcgs.append(ndcg_score([relevance], [relevance]))

# Compute average metrics
print(f"CF Precision@{K}: {np.mean(cf_precisions):.3f}, Recall@{K}: {np.mean(cf_recalls):.3f}, NDCG@{K}: {np.mean(cf_ndcgs):.3f}")
print(f"Content Precision@{K}: {np.mean(content_precisions):.3f}, Recall@{K}: {np.mean(content_recalls):.3f}, NDCG@{K}: {np.mean(content_ndcgs):.3f}")
print(f"Hybrid Precision@{K}: {np.mean(hybrid_precisions):.3f}, Recall@{K}: {np.mean(hybrid_recalls):.3f}, NDCG@{K}: {np.mean(hybrid_ndcgs):.3f}")
print(f"Reranked Precision@{K}: {np.mean(reranked_precisions):.3f}, Recall@{K}: {np.mean(reranked_recalls):.3f}, NDCG@{K}: {np.mean(reranked_ndcgs):.3f}")


# %%
alphas = [0.3, 0.5, 0.7, 0.9]
alpha_results = {}

for alpha in alphas:
    print(f"\nTesting alpha={alpha}")
    hybrid = HybridRecommender(cf_model=cf, content_model=content, alpha=alpha)
    hybrid_scores = dict(hybrid.recommend_for_playlist(playlist_idx, N=20))
    
    # Evaluate hybrid_scores using your evaluation function (replace with yours)
    # Here, dummy numbers just for example:
    eval_result = {
        'precision@20': np.random.uniform(0.10, 0.20),
        'recall@20': np.random.uniform(0.08, 0.15),
        'ndcg@20': np.random.uniform(0.09, 0.18)
    }
    alpha_results[alpha] = eval_result
    print(f"Results: {eval_result}")

# Visualize
plt.figure(figsize=(8,5))
for metric in ['precision@20', 'recall@20', 'ndcg@20']:
    scores = [alpha_results[a][metric] for a in alphas]
    plt.plot(alphas, scores, label=metric)

plt.xlabel('Alpha')
plt.ylabel('Score')
plt.title('Hyperparameter Tuning for Hybrid alpha')
plt.legend()
plt.show()


# %%
betas = [0.3, 0.5, 0.7, 0.9]
beta_results = {}

for beta in betas:
    print(f"\nTesting beta={beta}")
    reranker = FairnessReranker(popularity, beta=beta, epsilon=1e-6)
    reranked_recs = reranker.rerank(list(hybrid_scores.items()))
    
    # Evaluate reranked_recs using your evaluation function
    # Again, dummy numbers as placeholders:
    eval_result = {
        'precision@20': np.random.uniform(0.10, 0.20),
        'recall@20': np.random.uniform(0.08, 0.15),
        'ndcg@20': np.random.uniform(0.09, 0.18)
    }
    beta_results[beta] = eval_result
    print(f"Results: {eval_result}")

# Visualize
plt.figure(figsize=(8,5))
for metric in ['precision@20', 'recall@20', 'ndcg@20']:
    scores = [beta_results[b][metric] for b in betas]
    plt.plot(betas, scores, label=metric)

plt.xlabel('Beta')
plt.ylabel('Score')
plt.title('Hyperparameter Tuning for Reranker beta')
plt.legend()
plt.show()


# %%
import pickle

# Save embeddings
with open('playlist_embeddings.pkl', 'wb') as f:
    pickle.dump(playlist_embeddings, f)

with open('track_embeddings.pkl', 'wb') as f:
    pickle.dump(track_embeddings, f)

# Save recommendations
import pandas as pd
pd.DataFrame(list(cf_scores.items()), columns=['track_uri', 'score']).to_csv('cf_recommendations.csv', index=False)
pd.DataFrame(list(content_scores.items()), columns=['track_uri', 'score']).to_csv('content_recommendations.csv', index=False)
pd.DataFrame(list(hybrid_scores.items()), columns=['track_uri', 'score']).to_csv('hybrid_recommendations.csv', index=False)
pd.DataFrame(reranked_recs, columns=['track_uri', 'score']).to_csv('reranked_recommendations.csv', index=False)

print("✅ Embeddings and recommendations saved!")



