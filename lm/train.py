"""
Training loop.

- AdamW optimizer, flat learning rate
- Cosine LR warmup then flat (or just flat if warmup=0)
- Gradient clipping
- MPS / CUDA / CPU device auto-selection
- Periodic eval on validation set
- Checkpoint saving
"""

import time
import numpy as np
import torch
from dataclasses import dataclass, field
from pathlib import Path

from .model import GPT, ModelConfig


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"

    batch_size: int = 32
    context_length: int = 512
    max_steps: int = 10_000
    eval_interval: int = 500
    eval_steps: int = 50
    checkpoint_interval: int = 1_000

    lr: float = 3e-4
    grad_clip: float = 1.0


def _get_batch(data: np.ndarray, cfg: TrainConfig, device: torch.device):
    ix = torch.randint(len(data) - cfg.context_length, (cfg.batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + cfg.context_length].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + cfg.context_length + 1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def _eval(model: GPT, data: np.ndarray, cfg: TrainConfig, device: torch.device) -> float:
    model.eval()
    losses = [
        model(*_get_batch(data, cfg, device))[1].item()
        for _ in range(cfg.eval_steps)
    ]
    model.train()
    return sum(losses) / len(losses)


def train(model_cfg: ModelConfig, train_cfg: TrainConfig):
    device = get_device()
    print(f"Device: {device}")

    data_dir = Path(train_cfg.data_dir)
    train_data = np.fromfile(data_dir / "train.bin", dtype=np.uint16)
    val_data = np.fromfile(data_dir / "val.bin", dtype=np.uint16)
    print(f"Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")

    model = GPT(model_cfg).to(device)
    print(f"Parameters: {model.num_params():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    ckpt_dir = Path(train_cfg.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)

    model.train()
    t0 = time.time()

    for step in range(train_cfg.max_steps):
        x, y = _get_batch(train_data, train_cfg, device)
        _, loss = model(x, y)
        loss.backward()

        if train_cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

        optimizer.step()
        optimizer.zero_grad()

        if step % 100 == 0:
            dt = time.time() - t0
            print(f"step {step:5d} | loss {loss.item():.4f} | {dt:.1f}s")
            t0 = time.time()

        if step % train_cfg.eval_interval == 0:
            val_loss = _eval(model, val_data, train_cfg, device)
            print(f"  val loss: {val_loss:.4f}")

        if step > 0 and step % train_cfg.checkpoint_interval == 0:
            _save(model, optimizer, step, model_cfg, train_cfg, ckpt_dir / f"ckpt_{step:06d}.pt")

    _save(model, optimizer, train_cfg.max_steps, model_cfg, train_cfg, ckpt_dir / "ckpt_final.pt")
    print("Done.")
    return model


def _save(model, optimizer, step, model_cfg, train_cfg, path):
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_cfg": model_cfg,
        "train_cfg": train_cfg,
    }, path)
    print(f"  checkpoint -> {path}")
