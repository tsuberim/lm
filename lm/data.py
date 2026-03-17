"""
Data pipeline: download -> tokenize -> save to flat binary (uint16).

The binary format is just a flat array of token ids, concatenated across
all documents with EOT tokens as separators. Training reads random chunks.
"""

import numpy as np
from pathlib import Path
from datasets import load_dataset
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich import print

from .tokenizer import BPETokenizer


def prepare(
    dataset_name: str = "roneneldan/TinyStories",
    data_dir: str = "data",
    vocab_size: int = 8192,
    tokenizer_sample_size: int = 20_000,
):
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)

    tok_path = data_dir / "tokenizer.json"
    train_path = data_dir / "train.bin"
    val_path = data_dir / "val.bin"

    # 1. Download
    print(f"[bold]Loading[/bold] [cyan]{dataset_name}[/cyan]...")
    ds = load_dataset(dataset_name)

    # 2. Train tokenizer on a sample
    if tok_path.exists():
        print(f"[dim]Tokenizer exists at {tok_path}, loading.[/dim]")
        tok = BPETokenizer.load(tok_path)
    else:
        print(f"[bold]Training tokenizer[/bold] on {tokenizer_sample_size:,} examples (vocab_size={vocab_size})...")
        texts = ds["train"].select(range(tokenizer_sample_size))["text"]
        tok = BPETokenizer()
        tok.train(texts, vocab_size=vocab_size)
        tok.save(tok_path)
        print(f"[green]Tokenizer saved[/green]: vocab_size={tok.vocab_size}")

    # 3. Tokenize and save
    splits = {"train": train_path, "validation": val_path}
    for split, path in splits.items():
        if path.exists():
            print(f"[dim]{path} exists, skipping.[/dim]")
            continue

        data = ds[split]
        all_tokens = []

        with Progress(
            TextColumn(f"[bold cyan]Tokenizing {split}[/]"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} docs"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(split, total=len(data))
            for example in data:
                all_tokens.extend(tok.encode(example["text"], add_eot=True))
                progress.advance(task)

        arr = np.array(all_tokens, dtype=np.uint16)
        arr.tofile(path)
        print(f"[green]Saved[/green] {len(arr):,} tokens -> {path}")

    return tok
