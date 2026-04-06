"""
BOOKIFY — Classic Collaborative Filtering Evaluation Script
===========================================================
Evaluates the classic item-based collaborative filtering model
using the exact same test dataset split and metrics as the NCF model
for a fair head-to-head comparison.
"""
import pickle
import json
import time
import numpy as np

TOP_K = 10
RELEVANT_THRESHOLD = 0.6  # normalized rating above this = "relevant"

def compute_ranking_metrics(sim_matrix, test_users, test_books, test_ratings, n_books, train_user_items):
    # Group test data by user
    user_items = {}
    for i in range(len(test_users)):
        uid = test_users[i]
        bid = test_books[i]
        rating = test_ratings[i]
        if uid not in user_items:
            user_items[uid] = []
        user_items[uid].append((bid, rating))
    
    precisions, ndcgs, hits = [], [], []
    eval_users = 0
    
    for uid, items in user_items.items():
        if len(items) < 2:
            continue
        relevant = set(bid for bid, r in items if r >= RELEVANT_THRESHOLD)
        if not relevant:
            continue
        
        # User history from train
        history = train_user_items.get(uid, [])
        if not history:
            continue
            
        # Predict scores for all books based on history
        # Simple weighted sum of similarities
        scores = np.zeros(n_books)
        for h_bid, h_rating in history:
            scores += sim_matrix[h_bid] * h_rating
            
        # Get top-K predictions (that are not in user history)
        history_bids = set(h[0] for h in history)
        # Avoid recommending already read books
        scores[list(history_bids)] = -1e9
        top_k_items = np.argsort(scores)[::-1][:TOP_K]
        
        # Rankings...
        hits_in_top_k = len(set(top_k_items) & relevant)
        precisions.append(hits_in_top_k / TOP_K)
        hits.append(1.0 if hits_in_top_k > 0 else 0.0)
        
        dcg = 0.0
        for rank, item in enumerate(top_k_items):
            if item in relevant:
                dcg += 1.0 / np.log2(rank + 2)
        ideal_hits = min(len(relevant), TOP_K)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        eval_users += 1
        
    return {
        f'precision_at_{TOP_K}': float(np.mean(precisions)) if precisions else 0.0,
        f'ndcg_at_{TOP_K}': float(np.mean(ndcgs)) if ndcgs else 0.0,
        f'hit_rate_at_{TOP_K}': float(np.mean(hits)) if hits else 0.0,
        'num_evaluated_users': eval_users,
    }

def main():
    print("=" * 60)
    print("BOOKIFY — Classic CF Evaluation")
    print("=" * 60)
    
    print("📦 Loading data...")
    pt = pickle.load(open('pt.pkl', 'rb'))
    sim_matrix = pickle.load(open('similarity_scores.pkl', 'rb'))
    
    ratings_matrix = pt.values
    n_books, n_users = ratings_matrix.shape
    
    user_indices, book_indices, ratings = [], [], []
    for bi in range(n_books):
        for ui in range(n_users):
            r = ratings_matrix[bi, ui]
            if r != 0:
                user_indices.append(ui)
                book_indices.append(bi)
                ratings.append(r)
                
    ratings = np.array(ratings)
    r_min, r_max = ratings.min(), ratings.max()
    if r_max > r_min:
        ratings_norm = (ratings - r_min) / (r_max - r_min)
    else:
        ratings_norm = np.zeros_like(ratings)
        
    n = len(ratings_norm)
    perm = np.random.RandomState(42).permutation(n)
    test_size = int(n * 0.15)
    val_size = int(n * 0.15)
    train_size = n - test_size - val_size
    
    test_idx = perm[:test_size]
    train_idx = perm[test_size + val_size:]
    
    train_user_items = {}
    for i in train_idx:
        uid = user_indices[i]
        if uid not in train_user_items:
            train_user_items[uid] = []
        train_user_items[uid].append((book_indices[i], ratings_norm[i]))
        
    test_user_list = [user_indices[i] for i in test_idx]
    test_book_list = [book_indices[i] for i in test_idx]
    test_rating_list = ratings_norm[test_idx].tolist()
    
    print("📈 Computing ranking metrics...")
    start_time = time.time()
    metrics = compute_ranking_metrics(
        sim_matrix, test_user_list, test_book_list, test_rating_list,
        n_books, train_user_items
    )
    
    elapsed = time.time() - start_time
    print(f"   Precision@{TOP_K}: {metrics[f'precision_at_{TOP_K}']:.4f}")
    print(f"   NDCG@{TOP_K}:      {metrics[f'ndcg_at_{TOP_K}']:.4f}")
    print(f"   Hit Rate@{TOP_K}:   {metrics[f'hit_rate_at_{TOP_K}']:.4f}")
    print(f"   Evaluated users:  {metrics['num_evaluated_users']}")
    print(f"   Time: {elapsed:.1f}s")
    
    # Save classic accuracy
    with open('classic_accuracy.json', 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == '__main__':
    main()
