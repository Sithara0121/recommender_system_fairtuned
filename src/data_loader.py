# src/data_loader.py

import os
import json
import pandas as pd
from tqdm import tqdm

def load_mpd_dataset(root_dir: str = ".", num_files: int = 100) -> pd.DataFrame:
    """
    Load Spotify MPD dataset from JSON slices inside 'data' folder.

    Args:
        root_dir (str): Path to the root project folder (where data/ exists).
        num_files (int): Number of JSON files to load (default 100 for dev).

    Returns:
        pd.DataFrame: Flattened playlist-track DataFrame.
    """
    data_dir = os.path.join(root_dir, "data")
    all_data = []

    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".json")])
    files = files[:num_files]  # Load only first `num_files` JSON files

    for file in tqdm(files, desc="Loading MPD JSON slices"):
        file_path = os.path.join(data_dir, file)
        with open(file_path, "r") as f:
            slice_data = json.load(f)
            playlists = slice_data["playlists"]
            for playlist in playlists:
                pid = playlist["pid"]
                pname = playlist["name"]
                for track in playlist["tracks"]:
                    all_data.append({
                        "playlist_id": pid,
                        "playlist_name": pname,
                        "track_uri": track.get("track_uri"),
                        "track_name": track.get("track_name"),
                        "artist_uri": track.get("artist_uri"),
                        "artist_name": track.get("artist_name"),
                        "album_uri": track.get("album_uri"),
                        "album_name": track.get("album_name"),
                    })

    df = pd.DataFrame(all_data)
    print(f"✅ Loaded {len(df)} track entries from {len(files)} files.")
    return df
