"""
BOOKIFY — Neural Collaborative Filtering (NCF) Training Script
================================================================
Trains an NCF model on user-book ratings from pt.pkl, extracts learned
book embeddings, and precomputes a cosine-similarity matrix that can be
used as a drop-in replacement for the original similarity_scores.pkl.

Usage:
    python train_ncf.py
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity

# ── Hyperparameters ──
EMBED_DIM = 64
HIDDEN_LAYERS = [128, 64, 32]
EPOCHS = 50
BATCH_SIZE = 256
LR = 0.001
PATIENCE = 7       # early-stopping patience
VAL_SPLIT = 0.2

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
            layers.append(nn.Dropout(0.2))
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.book_embed.weight)

    def forward(self, user_ids, book_ids):
        u = self.user_embed(user_ids)
        b = self.book_embed(book_ids)
        x = torch.cat([u, b], dim=1)
        return self.mlp(x).squeeze()


def main():
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

    # Extract triplets
    user_indices, book_indices, ratings = [], [], []
    for bi in range(n_books):
        for ui in range(n_users):
            r = ratings_matrix[bi, ui]
            if r != 0:
                user_indices.append(ui)
                book_indices.append(bi)
                ratings.append(r)

    print(f"   Total ratings: {len(ratings)}")
    print(f"   Rating range: {min(ratings):.1f} – {max(ratings):.1f}")

    # Normalize ratings to [0, 1]
    ratings = np.array(ratings)
    r_min, r_max = ratings.min(), ratings.max()
    ratings_norm = (ratings - r_min) / (r_max - r_min)

    # ── Train/Val split ──
    n = len(ratings_norm)
    perm = np.random.RandomState(42).permutation(n)
    val_size = int(n * VAL_SPLIT)
    train_idx, val_idx = perm[val_size:], perm[:val_size]

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

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

    print(f"   Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # ── Model ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Device: {device}")

    model = NCF(n_users, n_books, EMBED_DIM, HIDDEN_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    print(f"   Architecture: User/Book Embed({EMBED_DIM}) → MLP{HIDDEN_LAYERS} → 1")

    # ── Training ──
    print(f"\n🚀 Training for up to {EPOCHS} epochs (patience={PATIENCE})...\n")
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

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

        bar = "█" * int(30 * epoch / EPOCHS) + "░" * (30 - int(30 * epoch / EPOCHS))
        print(f"   Epoch {epoch:3d}/{EPOCHS} [{bar}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}", end="")

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
    model.eval()

    # ── Extract book embeddings ──
    print("\n📊 Extracting book embeddings...")
    book_embeddings = model.book_embed.weight.detach().cpu().numpy()
    print(f"   Shape: {book_embeddings.shape}")

    # ── Compute similarity matrix ──
    print("📐 Computing cosine similarity matrix...")
    sim_matrix = cosine_similarity(book_embeddings)
    print(f"   Shape: {sim_matrix.shape}")
    print(f"   Range: {sim_matrix.min():.4f} – {sim_matrix.max():.4f}")

    # ── Save ──
    print("\n💾 Saving outputs...")
    pickle.dump(book_embeddings, open('ncf_book_embeddings.pkl', 'wb'))
    print("   ✓ ncf_book_embeddings.pkl")

    pickle.dump(sim_matrix, open('ncf_similarity_scores.pkl', 'wb'))
    print("   ✓ ncf_similarity_scores.pkl")

    # ── Quick sanity check ──
    print("\n🔍 Sanity check — Top 3 similar to first book:")
    test_book = book_names[0]
    sims = sim_matrix[0]
    top_idx = np.argsort(sims)[::-1][1:4]
    for rank, idx in enumerate(top_idx, 1):
        print(f"   #{rank}: {book_names[idx]} (similarity: {sims[idx]:.4f})")

    print("\n" + "=" * 60)
    print("✅ Training complete! Files saved.")
    print("=" * 60)


if __name__ == '__main__':
    main()
