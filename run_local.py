import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import set_seed, get_device, ensure_dir
from datasets import load_cifar, split_indices, SynthHealthDataset, make_client_dist, make_vendor_dist
from model import build_model
from train import train_one_epoch, evaluate

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
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)
    device = get_device()

    train_base, test_base = load_cifar(args.data_dir)
    client_splits = split_indices(len(train_base), args.clients, args.seed)
    test_splits = split_indices(len(test_base), args.clients, args.seed + 999)

    results = []
    for cid in range(args.clients):
        dist = make_client_dist(cid, args.seed)
        ds_train = SynthHealthDataset(train_base, client_splits[cid], dist, seed=args.seed + cid)
        ds_test = SynthHealthDataset(test_base, test_splits[cid], dist, seed=args.seed + 100 + cid)

        dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=False)
        dl_test = DataLoader(ds_test, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=False)

        model = build_model(num_classes=6).to(device)
        model.load_state_dict(torch.load(args.vendor_ckpt, map_location="cpu"))
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

        for e in range(args.epochs):
            train_one_epoch(model, dl_train, opt, device)

        m = evaluate(model, dl_test, device)
        results.append((cid, m["acc"], m["macro_f1"]))
        print(f"[Local-only][Client {cid}] acc={m['acc']:.4f} f1={m['macro_f1']:.4f}")

    accs = np.array([r[1] for r in results])
    f1s = np.array([r[2] for r in results])
    print(f"[Local-only][Summary] acc_mean={accs.mean():.4f} acc_std={accs.std():.4f} "
          f"f1_mean={f1s.mean():.4f} f1_std={f1s.std():.4f}")

if __name__ == "__main__":
    main()
