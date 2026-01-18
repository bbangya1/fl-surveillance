import argparse
import csv
import os
import time

import torch
import flwr as fl

from utils import set_seed, ensure_dir
from model import build_model


def get_parameters(model):
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, params):
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    new_state = {k: torch.tensor(v) for k, v in zip(keys, params)}
    model.load_state_dict(new_state, strict=True)


class FedAvgSaveFinalAndCSV(fl.server.strategy.FedAvg):
    """FedAvg strategy that (1) stores latest aggregated params and (2) logs per-round metrics to CSV."""

    def __init__(self, *args, out_dir: str = "./outputs", **kwargs):
        super().__init__(*args, **kwargs)
        self.out_dir = out_dir
        self.latest_parameters = None
        self.csv_path = os.path.join(out_dir, "metrics.csv")
        self._csv_initialized = False

    def _ensure_csv(self):
        ensure_dir(self.out_dir)
        if self._csv_initialized:
            return
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestamp", "round", "loss", "acc", "macro_f1"])
        self._csv_initialized = True

    def _append_csv(self, server_round: int, loss: float, acc: float, macro_f1: float):
        self._ensure_csv()
        ts = int(time.time())
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([ts, server_round, loss, acc, macro_f1])

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        # aggregated is either (Parameters, metrics) or None
        if aggregated is None:
            return None
        params, metrics = aggregated
        self.latest_parameters = params
        return params, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        """
        Flower collects evaluate results from clients:
        results: list of (client_proxy, EvaluateRes)
        EvaluateRes has: loss, num_examples, metrics (dict)
        We'll compute weighted averages and log to CSV.
        """
        aggregated = super().aggregate_evaluate(server_round, results, failures)

        # Compute weighted averages ourselves (version-safe)
        total = 0
        sum_loss = 0.0
        sum_acc = 0.0
        sum_f1 = 0.0

        for _, res in results:
            n = int(getattr(res, "num_examples", 0) or 0)
            loss = float(getattr(res, "loss", 0.0) or 0.0)
            metrics = getattr(res, "metrics", {}) or {}

            acc = float(metrics.get("acc", 0.0) or 0.0)
            f1 = float(metrics.get("macro_f1", 0.0) or 0.0)

            total += n
            sum_loss += n * loss
            sum_acc += n * acc
            sum_f1 += n * f1

        if total > 0:
            avg_loss = sum_loss / total
            avg_acc = sum_acc / total
            avg_f1 = sum_f1 / total
        else:
            avg_loss, avg_acc, avg_f1 = 0.0, 0.0, 0.0

        # Write one row per round
        self._append_csv(server_round, avg_loss, avg_acc, avg_f1)

        return aggregated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./outputs")
    ap.add_argument("--vendor_ckpt", default="./outputs/central_vendor.pt")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--addr", default="0.0.0.0:8080")
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    # Initial global params = vendor-central model (실제 배포 초기 모델 가정)
    model = build_model(num_classes=6)
    model.load_state_dict(torch.load(args.vendor_ckpt, map_location="cpu"))
    init_params = fl.common.ndarrays_to_parameters(get_parameters(model))

    strategy = FedAvgSaveFinalAndCSV(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.clients,
        min_evaluate_clients=args.clients,
        min_available_clients=args.clients,
        initial_parameters=init_params,
        out_dir=args.out_dir,
        # NOTE: we still keep these for Flower's History and console,
        # but CSV logging is handled in aggregate_evaluate for robustness.
        fit_metrics_aggregation_fn=lambda metrics: {},
        evaluate_metrics_aggregation_fn=None,
    )

    fl.server.start_server(
        server_address=args.addr,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    # Save final global model parameters (Flower version-safe)
    final_params = getattr(strategy, "latest_parameters", None)
    if final_params is None:
        final_params = getattr(strategy, "initial_parameters", None)

    if final_params is None:
        raise RuntimeError("Could not obtain final parameters from strategy.")

    ndarrays = fl.common.parameters_to_ndarrays(final_params)
    set_parameters(model, ndarrays)

    out = f"{args.out_dir}/fedavg_global.pt"
    torch.save(model.state_dict(), out)
    print(f"Saved FedAvg global model: {out}")
    print(f"Saved per-round metrics: {os.path.join(args.out_dir, 'metrics.csv')}")


if __name__ == "__main__":
    main()
