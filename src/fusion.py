# src/fusion.py

import numpy as np

def fuse_scores(cf_scores, content_scores, alpha=0.7):
    """
    Weighted sum fusion. alpha controls importance of CF.
    cf_scores, content_scores: dict {track_uri: score}
    """
    fused = {}
    all_tracks = set(cf_scores) | set(content_scores)
    for track in all_tracks:
        cf = cf_scores.get(track, 0)
        cont = content_scores.get(track, 0)
        fused[track] = alpha * cf + (1 - alpha) * cont
    return fused
