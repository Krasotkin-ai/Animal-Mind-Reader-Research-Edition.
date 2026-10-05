import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class NeuralTransformer(nn.Module):
    def __init__(self, n_units, n_classes, d_model=128, nhead=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(n_units, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True,
            dim_feedforward=d_model * 4, dropout=0.1, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes)
        )

    def forward(self, x, padding_mask=None):
        x = self.proj(x)
        z = self.encoder(x, src_key_padding_mask=padding_mask)
        if padding_mask is None:
            pooled = z.mean(dim=1)
        else:
            valid = (~padding_mask).unsqueeze(-1)
            pooled = (z * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return self.head(pooled)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int64)
    mask = d["mask"].astype(bool)

    tr, te = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NeuralTransformer(X.shape[-1], len(d["labels"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    train_ds = TensorDataset(
        torch.from_numpy(X[tr]),
        torch.from_numpy(y[tr]),
        torch.from_numpy(mask[tr])
    )
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model.train()
    for epoch in range(args.epochs):
        losses = []
        for xb, yb, mb in loader:
            xb, yb = xb.to(device), yb.to(device)
            # Transformer expects True where positions should be ignored.
            padding = ~mb.to(device)
            opt.zero_grad()
            logits = model(xb, padding)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        print(f"epoch {epoch+1:03d} loss={np.mean(losses):.4f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_units": X.shape[-1],
            "n_classes": len(d["labels"]),
            "labels": d["labels"].tolist(),
        },
        out,
    )
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
