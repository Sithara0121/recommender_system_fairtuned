# src/evaluation.py

import math

def precision_at_k(recommended, ground_truth, k):
    recommended = recommended[:k]
    hits = sum(1 for track in recommended if track in ground_truth)
    return hits / k

def recall_at_k(recommended, ground_truth, k):
    recommended = recommended[:k]
    hits = sum(1 for track in recommended if track in ground_truth)
    return hits / len(ground_truth) if ground_truth else 0

def ndcg_at_k(recommended, ground_truth, k):
    recommended = recommended[:k]
    dcg = 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(ground_truth), k)))

    for i, track in enumerate(recommended):
        if track in ground_truth:
            dcg += 1.0 / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0
