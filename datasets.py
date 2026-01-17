from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import functional as F
from PIL import Image, ImageEnhance, ImageFilter

HEALTH_CLASSES = ["normal", "blacked", "glared", "blurred", "blocked", "tilted"]

@dataclass
class DistConfig:
    class_probs: np.ndarray  # shape (6,)
    severity: Dict[str, Tuple[float, float]]  # per class severity range

def _apply_health_transform(img: Image.Image, cls: str, sev: float) -> Image.Image:
    # sev in [0,1]
    if cls == "normal":
        return img

    if cls == "blacked":
        # reduce brightness
        factor = 1.0 - 0.85 * sev  # 1 -> ~0.15
        return ImageEnhance.Brightness(img).enhance(factor)

    if cls == "glared":
        # increase brightness + slight contrast
        b = 1.0 + 1.2 * sev
        c = 1.0 + 0.3 * sev
        out = ImageEnhance.Brightness(img).enhance(b)
        out = ImageEnhance.Contrast(out).enhance(c)
        return out

    if cls == "blurred":
        radius = 0.2 + 3.0 * sev
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    if cls == "blocked":
        # random rectangle mask
        w, h = img.size
        rng = np.random.default_rng(int(sev * 1e6) % (2**32 - 1))
        bw = int((0.2 + 0.5 * sev) * w)
        bh = int((0.2 + 0.5 * sev) * h)
        x0 = int(rng.integers(0, max(1, w - bw)))
        y0 = int(rng.integers(0, max(1, h - bh)))
        out = img.copy()
        # draw black block
        for x in range(x0, x0 + bw):
            for y in range(y0, y0 + bh):
                out.putpixel((x, y), (0, 0, 0))
        return out

    if cls == "tilted":
        angle = (5 + 35 * sev) * (1 if sev >= 0.5 else -1)
        return img.rotate(angle, resample=Image.BILINEAR, expand=False)

    return img

class SynthHealthDataset(Dataset):
    """
    Wrap CIFAR10 images and assign synthetic 'image health' labels + transforms.
    Labels/transforms are precomputed for reproducibility.
    """
    def __init__(
        self,
        base: CIFAR10,
        indices: np.ndarray,
        dist: DistConfig,
        seed: int = 42,
    ):
        self.base = base
        self.indices = indices
        self.dist = dist
        self.rng = np.random.default_rng(seed)

        probs = dist.class_probs / dist.class_probs.sum()
        self.assigned = self.rng.choice(len(HEALTH_CLASSES), size=len(indices), p=probs)

        self.sev = np.zeros(len(indices), dtype=np.float32)
        for i, cidx in enumerate(self.assigned):
            cname = HEALTH_CLASSES[cidx]
            lo, hi = dist.severity[cname]
            self.sev[i] = float(self.rng.uniform(lo, hi))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        img, _ = self.base[idx]  # ignore original label
        cname = HEALTH_CLASSES[int(self.assigned[i])]
        sev = float(self.sev[i])

        img = _apply_health_transform(img, cname, sev)
        # to tensor
        x = F.to_tensor(img)  # [0,1]
        y = int(self.assigned[i])
        return x, y

def make_vendor_dist(seed: int = 42) -> DistConfig:
    # Vendor distribution: mostly normal + mild issues, low severity
    probs = np.array([0.60, 0.08, 0.08, 0.10, 0.08, 0.06], dtype=np.float32)
    sev = {
        "normal": (0.0, 0.0),
        "blacked": (0.05, 0.25),
        "glared": (0.05, 0.25),
        "blurred": (0.05, 0.25),
        "blocked": (0.05, 0.25),
        "tilted": (0.05, 0.25),
    }
    return DistConfig(class_probs=probs, severity=sev)

def make_client_dist(client_id: int, seed: int = 42) -> DistConfig:
    # Non-IID: label skew via Dirichlet + severity skew via client-specific scaling
    rng = np.random.default_rng(seed + client_id * 1000)
    alpha = np.array([0.8, 0.6, 0.6, 0.6, 0.6, 0.6])  # tends to skew
    probs = rng.dirichlet(alpha).astype(np.float32)

    # Severity skew: each client has different typical severity
    base = 0.15 + 0.6 * (client_id / 9.0)  # client 0 mild -> client 9 harsher
    sev = {
        "normal": (0.0, 0.0),
        "blacked": (max(0.05, base - 0.10), min(0.95, base + 0.10)),
        "glared": (max(0.05, base - 0.10), min(0.95, base + 0.10)),
        "blurred": (max(0.05, base - 0.10), min(0.95, base + 0.10)),
        "blocked": (max(0.05, base - 0.10), min(0.95, base + 0.10)),
        "tilted": (max(0.05, base - 0.10), min(0.95, base + 0.10)),
    }
    return DistConfig(class_probs=probs, severity=sev)

def load_cifar(data_dir: str = "./data"):
    train = CIFAR10(root=data_dir, train=True, download=True)
    test = CIFAR10(root=data_dir, train=False, download=True)
    return train, test

def split_indices(n: int, num_clients: int, seed: int = 42) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return np.array_split(idx, num_clients)
