# src/eda.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_playlist_length_distribution(df: pd.DataFrame):
    playlist_lengths = df.groupby("playlist_id").size()
    plt.figure(figsize=(10, 5))
    sns.histplot(playlist_lengths, bins=50, kde=False)
    plt.title("Playlist Length Distribution")
    plt.xlabel("Number of Tracks")
    plt.ylabel("Number of Playlists")
    plt.grid(True)
    plt.show()

def plot_top_tracks(df: pd.DataFrame, top_n=20):
    top_tracks = df['track_name'].value_counts().head(top_n)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_tracks.values, y=top_tracks.index, palette="viridis")
    plt.title(f"Top {top_n} Most Frequent Tracks")
    plt.xlabel("Occurrences")
    plt.ylabel("Track Name")
    plt.tight_layout()
    plt.show()

def plot_track_popularity_long_tail(df: pd.DataFrame):
    track_freq = df['track_uri'].value_counts()
    plt.figure(figsize=(10, 5))
    sns.histplot(track_freq, bins=100, log_scale=(True, True))
    plt.title("Track Popularity Distribution (Log Scale)")
    plt.xlabel("Track Occurrence Count")
    plt.ylabel("Number of Tracks")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def get_cold_start_track_stats(df: pd.DataFrame):
    track_counts = df['track_uri'].value_counts()
    cold_tracks = track_counts[track_counts == 1]
    print(f"Cold-start tracks (appearing only once): {len(cold_tracks)}")
    print(f"Total unique tracks: {len(track_counts)}")
    print(f"Percentage of cold-start tracks: {len(cold_tracks) / len(track_counts) * 100:.2f}%")
