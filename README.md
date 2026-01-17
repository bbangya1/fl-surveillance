# Federated Learning-Based Model Improvement for Video Surveillance Systems

This repository provides a **reproducible experimental framework** for studying **Federated Learning (FL)** in video surveillance systems where customer video data cannot be centrally collected due to **privacy, security, and legal constraints**.

As part of a **master’s thesis**, the project evaluates whether FL can improve model performance across heterogeneous customer domains without sharing raw data. Using **PyTorch** and **Flower**, we compare three training paradigms under **non-IID client data distributions**:

- **Vendor-only centralized training**
- **Local-only client training**
- **FedAvg-based federated learning**

---

## Motivation

Commercial video surveillance AI is commonly trained on **vendor-curated datasets** and deployed to customer sites (firmware/edge apps). In real deployments, model performance often degrades due to domain gaps (lighting, camera angles, background dynamics, environment-specific artifacts).

However, central collection of customer video data is usually infeasible due to:

- Privacy concerns
- Security policies
- Regulatory and legal restrictions

This repository explores FL as a practical alternative to enable collaborative model improvement without raw data sharing.

---

## Research Questions

1. How does FL compare to vendor-only centralized training in surveillance-like domain shifts?
2. Does FL reduce performance variance across heterogeneous clients?
3. How does FL compare to local-only training (no aggregation)?
4. Is FL a feasible training paradigm under realistic constraints?

---

## Experimental Overview

### Training Paradigms

| Approach | Description |
|---|---|
| Vendor-only | Centralized training using vendor-distributed data only |
| Local-only | Each client fine-tunes the vendor model using its own data |
| Federated (FedAvg) | Clients collaboratively train a global model via parameter aggregation |

### Dataset Simulation (Privacy-Friendly)

- Base dataset: **CIFAR-10**
- Original CIFAR-10 labels are **ignored**
- Each image is transformed into one of six **synthetic image health states** (6-class classification):
  - `Normal`
  - `Blacked`
  - `Glared`
  - `Blurred`
  - `Blocked`
  - `Tilted`

To simulate real-world conditions:

- **Vendor distribution**: relatively clean and mild degradations
- **Client distributions**: non-IID with:
  - **Label skew** (different dominant failure modes)
  - **Severity skew** (different degradation intensities)

### Model and Metrics

- Model: **ResNet-18 (6-class head)**
- Optimizer: SGD (momentum=0.9)
- Loss: Cross-Entropy
- Metrics:
  - Accuracy
  - Macro-F1

---

## Repository Structure

```text
.
├── client_app.py         # Flower client implementation
├── server_app.py         # Flower server (FedAvg)
├── run_central.py        # Vendor-only centralized training
├── run_local.py          # Local-only client training
├── datasets.py           # Synthetic dataset & non-IID simulation
├── model.py              # Model definition
├── train.py              # Training & evaluation utilities
├── utils.py              # Common utilities
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── data/                 # Dataset storage (mounted volume)
└── outputs/              # Trained models & results
```

---

## Test Environment Setup

### System Requirements

- **Operating System**
  - Linux (Ubuntu recommended)
  - Windows with **WSL2**
- **Docker & Docker Compose**
- **Memory**
  - Minimum: 8 GB
  - Recommended: **16 GB or higher** (for 10 FL clients)
- **GPU (Optional)**
  - NVIDIA GPU with CUDA support (recommended for centralized training)

### WSL2 Configuration (Recommended)

When running multiple federated learning clients on **WSL2**, memory exhaustion may occur unless resource limits are explicitly configured.

Create or edit the file on Windows:

- Path: `C:\Users\<username>\.wslconfig`

Add the following configuration:

```ini
[wsl2]
memory=12GB
processors=8
swap=16GB
```

Apply changes by restarting WSL:

```powershell
wsl --shutdown
```

### Python Dependencies (Optional: Non-Docker Setup)

If you prefer to run experiments without Docker:

```bash
python3 -m pip install -r requirements.txt
```

---

## Running Experiments (Docker-Based)

> **Important**  
> Federated learning experiments require an initial vendor-trained model.  
> Make sure `outputs/central_vendor.pt` is created before starting FL.

### Step 1: Vendor-Only Centralized Training

Generate the vendor-initial model using centralized training.

```bash
docker compose run --rm server \
  python3 run_central.py --epochs 20 --out_dir ./outputs
```

After completion, the following file will be created:

```text
outputs/central_vendor.pt
```

### Step 2: Federated Learning (Server + Clients)

Start the Flower server and federated clients:

```bash
docker compose up --build
```

- The server listens on port `8080`
- Clients connect via Docker internal networking
- Each client runs in an isolated container with a fixed memory limit

To monitor server logs:

```bash
docker compose logs -f server
```

To stop all containers:

```bash
docker compose down
```

---

## Local-Only Training Baseline (Optional)

To evaluate local-only training without global aggregation, run:

```bash
python3 run_local.py \
  --vendor_ckpt ./outputs/central_vendor.pt \
  --clients 10 \
  --epochs 50
```

Each client trains independently starting from the vendor-initial model.

---

## Stability and Scalability Notes

To ensure stable execution, especially on WSL2:

- Configure all `DataLoader` instances with:
  - `num_workers = 0`
  - `pin_memory = False`
- Run each federated client in a Docker container with a memory limit
- Optionally enable partial participation in `server_app.py`:
  - `fraction_fit < 1.0`
  - `min_fit_clients < total clients`

These settings reflect realistic federated learning deployments and help prevent out-of-memory (OOM) failures.

---

## Expected Outcomes

This experimental framework enables:

- Direct comparison of **Vendor-only**, **Local-only**, and **Federated (FedAvg)** training
- Analysis of performance variance across heterogeneous client domains
- Practical evaluation of federated learning feasibility in privacy-sensitive video surveillance systems

---

## License

This project is intended for **academic and research purposes only**.

---

## Citation

Citation information will be added after thesis publication.

```text
(To be added)
```

---

## Future Work

- Round-wise federated metrics logging (CSV)
- Convergence visualization and analysis
- Personalized federated learning extensions
