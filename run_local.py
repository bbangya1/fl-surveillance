import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import set_seed, get_device, ensure_dir
from datasets import load_cifar, split_indices, SynthHealthDataset, make_client_dist
from model import build_model
from train import train_one_epoch, evaluate


@torch.no_grad()
def eval_loss(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0.0
    total = 0
    for batch in dataloader:
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        bs = y.size(0)
        total_loss += loss.item() * bs
        total += bs
    return total_loss / max(1, total), total  # (avg_loss, num_examples)


def write_one_row_metrics_csv(path, loss, acc, macro_f1):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "round", "loss", "acc", "macro_f1"])
        w.writerow([int(time.time()), 0, float(loss), float(acc), float(macro_f1)])


def write_clients_csv(path, rows):
    """
    rows: list of dict with keys:
      cid, num_examples, loss, acc, macro_f1
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "cid", "num_examples", "loss", "acc", "macro_f1"])
        ts = int(time.time())
        for r in rows:
            w.writerow([ts, r["cid"], r["num_examples"], r["loss"], r["acc"], r["macro_f1"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--out_dir", default="./outputs")
    ap.add_argument("--vendor_ckpt", default="./outputs/central_vendor.pt")
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=50)  # rounds * local_epochs
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)

    # perf/stability knobs
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--pin_memory", action="store_true", help="Enable pin_memory (default: False)")

    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)
    device = get_device()

    train_base, test_base = load_cifar(args.data_dir)
    client_splits = split_indices(len(train_base), args.clients, args.seed)
    test_splits = split_indices(len(test_base), args.clients, args.seed + 999)

    criterion = nn.CrossEntropyLoss()

    client_rows = []

    # Weighted sums for global summary
    total_examples = 0
    sum_loss = 0.0
    sum_acc = 0.0
    sum_f1 = 0.0

    for cid in range(args.clients):
        dist = make_client_dist(cid, args.seed)
        ds_train = SynthHealthDataset(train_base, client_splits[cid], dist, seed=args.seed + cid)
        ds_test = SynthHealthDataset(test_base, test_splits[cid], dist, seed=args.seed + 100 + cid)

        dl_train = DataLoader(
            ds_train,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        dl_test = DataLoader(
            ds_test,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )

        model = build_model(num_classes=6).to(device)
        model.load_state_dict(torch.load(args.vendor_ckpt, map_location="cpu"))
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

        for _ in range(args.epochs):
            train_one_epoch(model, dl_train, opt, device)

        m = evaluate(model, dl_test, device)
        loss, n = eval_loss(model, dl_test, device, criterion)

        client_rows.append(
            {"cid": cid, "num_examples": n, "loss": float(loss), "acc": float(m["acc"]), "macro_f1": float(m["macro_f1"])}
        )

        print(f"[Local-only][Client {cid}] loss={loss:.4f} acc={m['acc']:.4f} f1={m['macro_f1']:.4f}")

        # Weighted aggregation
        total_examples += n
        sum_loss += n * float(loss)
        sum_acc += n * float(m["acc"])
        sum_f1 += n * float(m["macro_f1"])

    # Weighted global averages
    avg_loss = sum_loss / max(1, total_examples)
    avg_acc = sum_acc / max(1, total_examples)
    avg_f1 = sum_f1 / max(1, total_examples)

    # Also compute std across clients for reporting (unweighted)
    accs = np.array([r["acc"] for r in client_rows], dtype=float)
    f1s = np.array([r["macro_f1"] for r in client_rows], dtype=float)

    print(
        f"[Local-only][Summary] "
        f"loss={avg_loss:.4f} acc_mean={avg_acc:.4f} acc_std={accs.std():.4f} "
        f"f1_mean={avg_f1:.4f} f1_std={f1s.std():.4f}"
    )

    # Write CSVs
    local_csv = os.path.join(args.out_dir, "local_metrics.csv")
    local_clients_csv = os.path.join(args.out_dir, "local_clients.csv")

    write_one_row_metrics_csv(local_csv, loss=avg_loss, acc=avg_acc, macro_f1=avg_f1)
    write_clients_csv(local_clients_csv, client_rows)

    print(f"Saved local-only summary metrics: {local_csv}")
    print(f"Saved local-only per-client metrics: {local_clients_csv}")


if __name__ == "__main__":
    main()
