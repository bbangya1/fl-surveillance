import argparse
import os
import csv
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt


def read_metrics_csv(path: str) -> Dict[str, List[float]]:
    """
    Expected columns:
      timestamp, round, loss, acc, macro_f1
    """
    rounds: List[int] = []
    loss: List[float] = []
    acc: List[float] = []
    f1: List[float] = []

    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rounds.append(int(row["round"]))
            loss.append(float(row["loss"]))
            acc.append(float(row["acc"]))
            f1.append(float(row["macro_f1"]))

    return {"round": rounds, "loss": loss, "acc": acc, "macro_f1": f1}


def safe_label(name: str) -> str:
    return name.strip().replace("_", "-")


def ensure_round_alignment(
    metrics: Dict[str, List[float]],
    target_rounds: List[int],
) -> Dict[str, List[float]]:
    """
    If metrics has only 1 round entry (e.g., vendor/local baseline),
    expand it to match target_rounds by repeating the same y-values.
    Otherwise return as-is.

    Assumption: baseline CSV contains a single summary row (round=0).
    """
    if len(metrics["round"]) <= 1 and len(target_rounds) > 1:
        v_loss = metrics["loss"][0] if metrics["loss"] else 0.0
        v_acc = metrics["acc"][0] if metrics["acc"] else 0.0
        v_f1 = metrics["macro_f1"][0] if metrics["macro_f1"] else 0.0

        return {
            "round": list(target_rounds),
            "loss": [v_loss] * len(target_rounds),
            "acc": [v_acc] * len(target_rounds),
            "macro_f1": [v_f1] * len(target_rounds),
        }
    return metrics


def plot_single(metrics: Dict[str, List[float]], out_png: str, title: str) -> None:
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    x = metrics["round"]

    # --- Loss ---
    plt.figure()
    plt.plot(x, metrics["loss"])
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title(f"{title} - Loss")
    plt.grid(True)
    plt.tight_layout()
    loss_png = out_png.replace(".png", "_loss.png")
    plt.savefig(loss_png, dpi=150)
    plt.close()

    # --- Accuracy ---
    plt.figure()
    plt.plot(x, metrics["acc"])
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title(f"{title} - Accuracy")
    plt.grid(True)
    plt.tight_layout()
    acc_png = out_png.replace(".png", "_acc.png")
    plt.savefig(acc_png, dpi=150)
    plt.close()

    # --- Macro-F1 ---
    plt.figure()
    plt.plot(x, metrics["macro_f1"])
    plt.xlabel("Round")
    plt.ylabel("Macro-F1")
    plt.title(f"{title} - Macro-F1")
    plt.grid(True)
    plt.tight_layout()
    f1_png = out_png.replace(".png", "_macro_f1.png")
    plt.savefig(f1_png, dpi=150)
    plt.close()

    # Overview
    plt.figure()
    plt.plot(x, metrics["loss"], label="loss")
    plt.plot(x, metrics["acc"], label="acc")
    plt.plot(x, metrics["macro_f1"], label="macro_f1")
    plt.xlabel("Round")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_compare(
    series: List[Tuple[str, Dict[str, List[float]]]],
    out_png: str,
    metric_key: str,
    title: str,
) -> None:
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    # line styles only (no explicit colors)
    styles = ["-", "--", "-.", ":"]

    plt.figure()
    for i, (name, m) in enumerate(series):
        x = m["round"]
        y = m[metric_key]
        plt.plot(x, y, styles[i % len(styles)], label=safe_label(name))
    plt.xlabel("Round")
    ylabel = "Loss" if metric_key == "loss" else ("Accuracy" if metric_key == "acc" else "Macro-F1")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fedavg_csv", default="./outputs/metrics.csv", help="FedAvg metrics.csv path")
    ap.add_argument("--vendor_csv", default="./outputs/vendor_metrics.csv", help="Vendor-only CSV path")
    ap.add_argument("--local_csv", default="./outputs/local_metrics.csv", help="Local-only CSV path")
    ap.add_argument("--out_dir", default="./outputs", help="Output directory for PNGs")
    ap.add_argument("--title_prefix", default="FL Surveillance", help="Plot title prefix")
    args = ap.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    fed_rounds: List[int] = []

    # 1) Single FedAvg plot: metrics.csv -> metrics.png
    fed = None
    if os.path.exists(args.fedavg_csv):
        fed = read_metrics_csv(args.fedavg_csv)
        fed_rounds = fed["round"]
        plot_single(
            fed,
            out_png=os.path.join(out_dir, "metrics.png"),
            title=f"{args.title_prefix} - FedAvg",
        )
        print(f"[plot] Saved: {os.path.join(out_dir, 'metrics.png')} (+ _loss/_acc/_macro_f1)")
    else:
        print(f"[plot] WARNING: FedAvg CSV not found: {args.fedavg_csv}")

    # 2) Compare Vendor/Local/FedAvg (if available)
    series: List[Tuple[str, Dict[str, List[float]]]] = []

    if os.path.exists(args.vendor_csv):
        vendor = read_metrics_csv(args.vendor_csv)
        if fed_rounds:
            vendor = ensure_round_alignment(vendor, fed_rounds)
        series.append(("Vendor-only", vendor))
    else:
        print(f"[plot] WARNING: Vendor CSV not found: {args.vendor_csv}")

    if os.path.exists(args.local_csv):
        local = read_metrics_csv(args.local_csv)
        if fed_rounds:
            local = ensure_round_alignment(local, fed_rounds)
        series.append(("Local-only", local))
    else:
        print(f"[plot] WARNING: Local CSV not found: {args.local_csv}")

    if fed is not None:
        series.append(("FedAvg", fed))

    if len(series) >= 2:
        plot_compare(
            series,
            out_png=os.path.join(out_dir, "compare_loss.png"),
            metric_key="loss",
            title=f"{args.title_prefix} - Compare (Loss)",
        )
        plot_compare(
            series,
            out_png=os.path.join(out_dir, "compare_acc.png"),
            metric_key="acc",
            title=f"{args.title_prefix} - Compare (Accuracy)",
        )
        plot_compare(
            series,
            out_png=os.path.join(out_dir, "compare_macro_f1.png"),
            metric_key="macro_f1",
            title=f"{args.title_prefix} - Compare (Macro-F1)",
        )
        print("[plot] Saved comparison PNGs: compare_loss.png / compare_acc.png / compare_macro_f1.png")
    else:
        print("[plot] NOTE: Need at least 2 CSVs to create comparison plots.")


if __name__ == "__main__":
    main()
