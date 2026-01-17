import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import flwr as fl

from utils import set_seed, get_device
from datasets import load_cifar, split_indices, SynthHealthDataset, make_client_dist
from model import build_model
from train import train_one_epoch, evaluate

def get_parameters(model):
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, params):
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    new_state = {k: torch.tensor(v) for k, v in zip(keys, params)}
    model.load_state_dict(new_state, strict=True)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: int, args):
        self.cid = cid
        self.args = args
        self.device = torch.device("cpu")

        train_base, test_base = load_cifar(args.data_dir)
        splits = split_indices(len(train_base), args.clients, args.seed)
        test_splits = split_indices(len(test_base), args.clients, args.seed + 999)

        dist = make_client_dist(cid, args.seed)
        self.ds_train = SynthHealthDataset(train_base, splits[cid], dist, seed=args.seed + cid)
        self.ds_test = SynthHealthDataset(test_base, test_splits[cid], dist, seed=args.seed + 100 + cid)

        self.dl_train = DataLoader(self.ds_train, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=False)
        self.dl_test = DataLoader(self.ds_test, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=False)

        self.model = build_model(num_classes=6).to(self.device) 

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        opt = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9)
        for _ in range(self.args.local_epochs):
            train_one_epoch(self.model, self.dl_train, opt, self.device)
        return get_parameters(self.model), len(self.ds_train), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        m = evaluate(self.model, self.dl_test, self.device)
        # Flower expects: loss, num_examples, metrics
        # loss not used here -> set 0
        return 0.0, len(self.ds_test), {"acc": m["acc"], "macro_f1": m["macro_f1"]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid", type=int, required=True)
    ap.add_argument("--server", default="127.0.0.1:8080")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed + args.cid)
    client = FlowerClient(args.cid, args)
    fl.client.start_numpy_client(server_address=args.server, client=client)

if __name__ == "__main__":
    main()
