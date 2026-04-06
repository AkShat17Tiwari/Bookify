"""
BOOKIFY — Neural Collaborative Filtering (NCF) Training Script
================================================================
Trains an NCF model on user-book ratings from pt.pkl, extracts learned
book embeddings, and precomputes a cosine-similarity matrix that can be
used as a drop-in replacement for the original similarity_scores.pkl.

Produces model_accuracy.json with comprehensive accuracy metrics.

Usage:
    python train_ncf.py
"""

import pickle
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity

# ── Hyperparameters ──
EMBED_DIM = 64
HIDDEN_LAYERS = [128, 64, 32]
EPOCHS = 100
BATCH_SIZE = 256
LR = 0.001
LR_MIN = 1e-6
PATIENCE = 10        # early-stopping patience
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
TOP_K = 10           # for Precision@K, NDCG@K, Hit Rate@K
RELEVANT_THRESHOLD = 0.6  # normalized rating above this = "relevant"

# ── Dataset ──
class RatingDataset(Dataset):
    def __init__(self, user_ids, book_ids, ratings):
        self.user_ids = torch.LongTensor(user_ids)
        self.book_ids = torch.LongTensor(book_ids)
        self.ratings = torch.FloatTensor(ratings)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.book_ids[idx], self.ratings[idx]


# ── NCF Model ──
class NCF(nn.Module):
    def __init__(self, n_users, n_books, embed_dim, hidden_layers):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.book_embed = nn.Embedding(n_books, embed_dim)

        # MLP layers
        layers = []
        input_dim = embed_dim * 2
        for h in hidden_layers:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(0.2))
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())  # Output in [0, 1]
        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.book_embed.weight)

    def forward(self, user_ids, book_ids):
        u = self.user_embed(user_ids)
        b = self.book_embed(book_ids)
        x = torch.cat([u, b], dim=1)
        return self.mlp(x).squeeze()


def compute_ranking_metrics(model, test_users, test_books, test_ratings,
                            n_books, device, top_k=TOP_K):
    """Compute Precision@K, NDCG@K, Hit Rate@K on the test set."""
    model.eval()

    # Group test data by user
    user_items = {}
    for i in range(len(test_users)):
        uid = test_users[i]
        bid = test_books[i]
        rating = test_ratings[i]
        if uid not in user_items:
            user_items[uid] = []
        user_items[uid].append((bid, rating))

    precisions = []
    ndcgs = []
    hits = []

    with torch.no_grad():
        for uid, items in user_items.items():
            if len(items) < 2:
                continue

            # Get relevant items (rating above threshold)
            relevant = set(bid for bid, r in items if r >= RELEVANT_THRESHOLD)
            if not relevant:
                continue

            # Score all books for this user
            all_book_ids = torch.arange(n_books, device=device)
            user_tensor = torch.full((n_books,), uid, dtype=torch.long, device=device)
            scores = model(user_tensor, all_book_ids).cpu().numpy()

            # Get top-K predictions
            top_k_items = np.argsort(scores)[::-1][:top_k]

            # Precision@K
            hits_in_top_k = len(set(top_k_items) & relevant)
            precision = hits_in_top_k / top_k
            precisions.append(precision)

            # Hit Rate@K (at least one relevant item in top-K)
            hit = 1.0 if hits_in_top_k > 0 else 0.0
            hits.append(hit)

            # NDCG@K
            dcg = 0.0
            for rank, item in enumerate(top_k_items):
                if item in relevant:
                    dcg += 1.0 / np.log2(rank + 2)
            # Ideal DCG
            ideal_hits = min(len(relevant), top_k)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)

    return {
        f'precision_at_{top_k}': float(np.mean(precisions)) if precisions else 0.0,
        f'ndcg_at_{top_k}': float(np.mean(ndcgs)) if ndcgs else 0.0,
        f'hit_rate_at_{top_k}': float(np.mean(hits)) if hits else 0.0,
        'num_evaluated_users': len(precisions),
    }


def main():
    start_time = time.time()

    print("=" * 60)
    print("BOOKIFY — Neural Collaborative Filtering Training")
    print("=" * 60)

    # ── Load data ──
    print("\n📦 Loading data...")
    pt = pickle.load(open('pt.pkl', 'rb'))
    print(f"   Pivot table shape: {pt.shape}")

    # Extract non-zero ratings as (user_idx, book_idx, rating) triplets
    # Note: pt has books as rows, users as columns
    ratings_matrix = pt.values
    book_names = list(pt.index)
    user_ids_raw = list(pt.columns)

    n_books = len(book_names)
    n_users = len(user_ids_raw)
    print(f"   Books: {n_books}, Users: {n_users}")

    # Create user/book index mappings
    user_to_idx = {uid: i for i, uid in enumerate(user_ids_raw)}
    book_to_idx = {bname: i for i, bname in enumerate(book_names)}

    # Extract triplets (only non-zero ratings)
    user_indices, book_indices, ratings = [], [], []
    for bi in range(n_books):
        for ui in range(n_users):
            r = ratings_matrix[bi, ui]
            if r != 0:
                user_indices.append(ui)
                book_indices.append(bi)
                ratings.append(r)

    print(f"   Total ratings: {len(ratings):,}")
    print(f"   Rating range: {min(ratings):.1f} – {max(ratings):.1f}")

    # Normalize ratings to [0, 1]
    ratings = np.array(ratings)
    r_min, r_max = ratings.min(), ratings.max()
    if r_max > r_min:
        ratings_norm = (ratings - r_min) / (r_max - r_min)
    else:
        ratings_norm = np.zeros_like(ratings)

    # ── Train/Val/Test split ──
    n = len(ratings_norm)
    perm = np.random.RandomState(42).permutation(n)
    test_size = int(n * TEST_SPLIT)
    val_size = int(n * VAL_SPLIT)
    train_size = n - test_size - val_size

    test_idx = perm[:test_size]
    val_idx = perm[test_size:test_size + val_size]
    train_idx = perm[test_size + val_size:]

    train_ds = RatingDataset(
        [user_indices[i] for i in train_idx],
        [book_indices[i] for i in train_idx],
        ratings_norm[train_idx]
    )
    val_ds = RatingDataset(
        [user_indices[i] for i in val_idx],
        [book_indices[i] for i in val_idx],
        ratings_norm[val_idx]
    )
    test_ds = RatingDataset(
        [user_indices[i] for i in test_idx],
        [book_indices[i] for i in test_idx],
        ratings_norm[test_idx]
    )

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

    print(f"   Train: {len(train_ds):,}, Val: {len(val_ds):,}, Test: {len(test_ds):,}")

    # ── Model ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Device: {device}")

    model = NCF(n_users, n_books, EMBED_DIM, HIDDEN_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, min_lr=LR_MIN
    )
    criterion = nn.MSELoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    print(f"   Architecture: User/Book Embed({EMBED_DIM}) → MLP{HIDDEN_LAYERS} → Sigmoid → 1")

    # ── Training ──
    print(f"\n🚀 Training for up to {EPOCHS} epochs (patience={PATIENCE})...\n")
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    train_losses = []
    val_losses = []

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0
        for users, bks, rats in train_dl:
            users, bks, rats = users.to(device), bks.to(device), rats.to(device)
            preds = model(users, bks)
            loss = criterion(preds, rats)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(rats)
        train_loss /= len(train_ds)
        train_losses.append(train_loss)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for users, bks, rats in val_dl:
                users, bks, rats = users.to(device), bks.to(device), rats.to(device)
                preds = model(users, bks)
                loss = criterion(preds, rats)
                val_loss += loss.item() * len(rats)
        val_loss /= len(val_ds)
        val_losses.append(val_loss)

        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        bar = "█" * int(30 * epoch / EPOCHS) + "░" * (30 - int(30 * epoch / EPOCHS))
        print(f"   Epoch {epoch:3d}/{EPOCHS} [{bar}] "
              f"train={train_loss:.4f}  val={val_loss:.4f}  lr={current_lr:.2e}", end="")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(" ✓ best")
        else:
            patience_counter += 1
            print(f" (patience {patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print(f"\n   ⏹ Early stopping at epoch {epoch}")
                break

    # Load best weights
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    # ── Test Set Evaluation ──
    print("\n📊 Evaluating on test set...")

    # RMSE and MAE on test set
    all_preds = []
    all_true = []
    test_loss = 0

    with torch.no_grad():
        for users, bks, rats in test_dl:
            users, bks, rats = users.to(device), bks.to(device), rats.to(device)
            preds = model(users, bks)
            loss = criterion(preds, rats)
            test_loss += loss.item() * len(rats)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(rats.cpu().numpy())

    test_loss /= len(test_ds)
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    # Convert back to original scale for RMSE/MAE
    preds_original = all_preds * (r_max - r_min) + r_min
    true_original = all_true * (r_max - r_min) + r_min

    rmse = float(np.sqrt(np.mean((preds_original - true_original) ** 2)))
    mae = float(np.mean(np.abs(preds_original - true_original)))

    # Normalized RMSE (as percentage of rating range)
    nrmse = rmse / (r_max - r_min) if r_max > r_min else 0.0

    print(f"   Test MSE (normalized): {test_loss:.4f}")
    print(f"   RMSE (original scale): {rmse:.4f}")
    print(f"   MAE  (original scale): {mae:.4f}")
    print(f"   Normalized RMSE: {nrmse:.4f} ({nrmse*100:.1f}%)")

    # Ranking metrics
    print(f"\n📈 Computing ranking metrics (Precision@{TOP_K}, NDCG@{TOP_K}, Hit Rate@{TOP_K})...")
    test_user_list = [user_indices[i] for i in test_idx]
    test_book_list = [book_indices[i] for i in test_idx]
    test_rating_list = ratings_norm[test_idx].tolist()

    ranking_metrics = compute_ranking_metrics(
        model, test_user_list, test_book_list, test_rating_list,
        n_books, device, TOP_K
    )

    print(f"   Precision@{TOP_K}: {ranking_metrics[f'precision_at_{TOP_K}']:.4f}")
    print(f"   NDCG@{TOP_K}:      {ranking_metrics[f'ndcg_at_{TOP_K}']:.4f}")
    print(f"   Hit Rate@{TOP_K}:   {ranking_metrics[f'hit_rate_at_{TOP_K}']:.4f}")
    print(f"   Evaluated users:  {ranking_metrics['num_evaluated_users']}")

    # ── Compute overall accuracy (1 - NRMSE, clamped to 0-100%) ──
    accuracy_pct = max(0, min(100, (1 - nrmse) * 100))

    # ── Extract book embeddings ──
    print("\n📊 Extracting book embeddings...")
    book_embeddings = model.book_embed.weight.detach().cpu().numpy()
    print(f"   Shape: {book_embeddings.shape}")

    # ── Compute similarity matrix ──
    print("📐 Computing cosine similarity matrix...")
    sim_matrix = cosine_similarity(book_embeddings)
    print(f"   Shape: {sim_matrix.shape}")
    print(f"   Range: {sim_matrix.min():.4f} – {sim_matrix.max():.4f}")

    # ── Save outputs ──
    print("\n💾 Saving outputs...")
    pickle.dump(book_embeddings, open('ncf_book_embeddings.pkl', 'wb'))
    print("   ✓ ncf_book_embeddings.pkl")

    pickle.dump(sim_matrix, open('ncf_similarity_scores.pkl', 'wb'))
    print("   ✓ ncf_similarity_scores.pkl")

    # ── Save accuracy report ──
    elapsed = time.time() - start_time

    accuracy_report = {
        'model': 'Neural Collaborative Filtering (NCF)',
        'architecture': f'User/Book Embed({EMBED_DIM}) → MLP{HIDDEN_LAYERS} → Sigmoid',
        'total_parameters': total_params,
        'training': {
            'epochs_run': len(train_losses),
            'best_val_loss': float(best_val_loss),
            'final_train_loss': float(train_losses[-1]),
            'early_stopped': patience_counter >= PATIENCE,
            'train_losses': [float(l) for l in train_losses],
            'val_losses': [float(l) for l in val_losses],
        },
        'dataset': {
            'total_books': n_books,
            'total_users': n_users,
            'total_ratings': len(ratings),
            'rating_range': [float(r_min), float(r_max)],
            'train_size': len(train_ds),
            'val_size': len(val_ds),
            'test_size': len(test_ds),
        },
        'accuracy': {
            'rmse': round(rmse, 4),
            'mae': round(mae, 4),
            'nrmse': round(nrmse, 4),
            'accuracy_pct': round(accuracy_pct, 2),
            'test_mse_normalized': round(test_loss, 4),
        },
        'ranking': {
            f'precision_at_{TOP_K}': round(ranking_metrics[f'precision_at_{TOP_K}'], 4),
            f'ndcg_at_{TOP_K}': round(ranking_metrics[f'ndcg_at_{TOP_K}'], 4),
            f'hit_rate_at_{TOP_K}': round(ranking_metrics[f'hit_rate_at_{TOP_K}'], 4),
            'num_evaluated_users': ranking_metrics['num_evaluated_users'],
        },
        'training_time_seconds': round(elapsed, 1),
    }

    with open('model_accuracy.json', 'w') as f:
        json.dump(accuracy_report, f, indent=2)
    print("   ✓ model_accuracy.json")

    # ── Quick sanity check ──
    print("\n🔍 Sanity check — Top 3 similar to first book:")
    test_book = book_names[0]
    sims = sim_matrix[0]
    top_idx = np.argsort(sims)[::-1][1:4]
    for rank, idx in enumerate(top_idx, 1):
        print(f"   #{rank}: {book_names[idx]} (similarity: {sims[idx]:.4f})")

    print(f"\n{'=' * 60}")
    print(f"✅ Training complete!")
    print(f"   Accuracy: {accuracy_pct:.1f}%")
    print(f"   RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    print(f"   Precision@{TOP_K}: {ranking_metrics[f'precision_at_{TOP_K}']:.4f}")
    print(f"   NDCG@{TOP_K}: {ranking_metrics[f'ndcg_at_{TOP_K}']:.4f}")
    print(f"   Hit Rate@{TOP_K}: {ranking_metrics[f'hit_rate_at_{TOP_K}']:.4f}")
    print(f"   Time: {int(elapsed//60)}m {int(elapsed%60)}s")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
