import argparse
import csv
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import set_seed, get_device, ensure_dir
from datasets import load_cifar, split_indices, SynthHealthDataset, make_vendor_dist
from model import build_model
from train import train_one_epoch, evaluate


@torch.no_grad()
def eval_loss(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0.0
    total = 0
    for batch in dataloader:
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=False)
        y = y.to(device, non_blocking=False)
        logits = model(x)
        loss = criterion(logits, y)
        bs = y.size(0)
        total_loss += loss.item() * bs
        total += bs
    return total_loss / max(1, total)


def write_single_row_csv(path, loss, acc, macro_f1):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "round", "loss", "acc", "macro_f1"])
        w.writerow([int(time.time()), 0, float(loss), float(acc), float(macro_f1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--out_dir", default="./outputs")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)

    # Stability knobs (recommended defaults for WSL2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--pin_memory", action="store_true", help="Enable pin_memory (default: False)")

    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)
    device = get_device()

    train_base, test_base = load_cifar(args.data_dir)

    # Vendor dataset uses only a subset to simulate "vendor-curated" data
    vendor_train_idx = split_indices(len(train_base), 1, args.seed)[0]
    vendor_test_idx = split_indices(len(test_base), 1, args.seed)[0]

    vendor_dist = make_vendor_dist(args.seed)
    ds_train = SynthHealthDataset(train_base, vendor_train_idx, vendor_dist, seed=args.seed)
    ds_test = SynthHealthDataset(test_base, vendor_test_idx, vendor_dist, seed=args.seed + 1)

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
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    last_train_loss = None
    last_test_loss = None
    last_metrics = None

    for e in range(args.epochs):
        last_train_loss = train_one_epoch(model, dl_train, opt, device)
        last_metrics = evaluate(model, dl_test, device)
        last_test_loss = eval_loss(model, dl_test, device, criterion)
        print(
            f"[Central][Epoch {e+1}/{args.epochs}] "
            f"train_loss={last_train_loss:.4f} test_loss={last_test_loss:.4f} "
            f"acc={last_metrics['acc']:.4f} f1={last_metrics['macro_f1']:.4f}"
        )

    ckpt = f"{args.out_dir}/central_vendor.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"Saved vendor-central model: {ckpt}")

    # Save vendor metrics (single row, round=0)
    vendor_csv = os.path.join(args.out_dir, "vendor_metrics.csv")
    write_single_row_csv(
        vendor_csv,
        loss=last_test_loss if last_test_loss is not None else 0.0,
        acc=last_metrics["acc"] if last_metrics else 0.0,
        macro_f1=last_metrics["macro_f1"] if last_metrics else 0.0,
    )
    print(f"Saved vendor metrics: {vendor_csv}")


if __name__ == "__main__":
    main()
