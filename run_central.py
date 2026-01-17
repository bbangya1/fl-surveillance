import argparse
import torch
from torch.utils.data import DataLoader
from utils import set_seed, get_device, ensure_dir
from datasets import load_cifar, split_indices, SynthHealthDataset, make_vendor_dist
from model import build_model
from train import train_one_epoch, evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--out_dir", default="./outputs")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
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

    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=False)
    dl_test = DataLoader(ds_test, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=False)

    model = build_model(num_classes=6).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    for e in range(args.epochs):
        loss = train_one_epoch(model, dl_train, opt, device)
        metrics = evaluate(model, dl_test, device)
        print(f"[Central][Epoch {e+1}/{args.epochs}] loss={loss:.4f} acc={metrics['acc']:.4f} f1={metrics['macro_f1']:.4f}")

    ckpt = f"{args.out_dir}/central_vendor.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"Saved vendor-central model: {ckpt}")

if __name__ == "__main__":
    main()
