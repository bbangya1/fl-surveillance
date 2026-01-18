import argparse
import numpy as np
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

    def fit_metrics_agg(metrics):
        # metrics: list of (num_examples, metrics_dict)
        # Here we can aggregate client metrics if needed
        return {}

    def eval_metrics_agg(metrics):
        # aggregate evaluation metrics from clients each round
        # compute weighted mean
        total = sum(n for n, _ in metrics)
        acc = sum(n * m.get("acc", 0.0) for n, m in metrics) / max(1, total)
        f1 = sum(n * m.get("macro_f1", 0.0) for n, m in metrics) / max(1, total)
        return {"acc": acc, "macro_f1": f1}
    
    # This strategy configuration ensures all clients participate in each round
    # strategy = fl.server.strategy.FedAvg(
    #     fraction_fit=1.0,
    #     fraction_evaluate=1.0,
    #     min_fit_clients=args.clients,
    #     min_evaluate_clients=args.clients,
    #     min_available_clients=args.clients,
    #     initial_parameters=init_params,
    #     fit_metrics_aggregation_fn=fit_metrics_agg,
    #     evaluate_metrics_aggregation_fn=eval_metrics_agg,
    # )
    
    # This strategy configuration allows partial client participation each round
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.3,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=args.clients,
        initial_parameters=init_params,
        fit_metrics_aggregation_fn=fit_metrics_agg,
        evaluate_metrics_aggregation_fn=eval_metrics_agg,
    )


    hist = fl.server.start_server(
        server_address=args.addr,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    # Save final global model parameters
    final_params = hist.parameters_centralized
    ndarrays = fl.common.parameters_to_ndarrays(final_params)
    set_parameters(model, ndarrays)
    out = f"{args.out_dir}/fedavg_global.pt"
    torch.save(model.state_dict(), out)
    print(f"Saved FedAvg global model: {out}")

if __name__ == "__main__":
    main()
