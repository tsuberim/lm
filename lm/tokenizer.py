"""
BPE tokenizer — Rust core, Python wrapper with progress display.

The heavy lifting (train, encode, decode) runs in Rust via lm._tokenizer.
This module adds rich progress bars and re-exports the same API.
"""

from pathlib import Path
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from lm._tokenizer import BPETokenizer as _BPETokenizer


class BPETokenizer:
    def __init__(self):
        self._tok = _BPETokenizer()

    def train(self, texts: list[str], vocab_size: int = 8192):
        n_merges = vocab_size - 256 - 1

        with Progress(
            TextColumn("[bold cyan]BPE merges[/]"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("merging", total=n_merges)

            def on_merge(step: int, total: int, merged: bytes):
                progress.advance(task)

            self._tok.train(texts, vocab_size, on_merge)

    def encode(self, text: str, add_eot: bool = False) -> list[int]:
        return self._tok.encode(text, add_eot)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size

    @property
    def eot_id(self) -> int:
        return self._tok.eot_id

    def save(self, path: str | Path):
        self._tok.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        tok = cls.__new__(cls)
        tok._tok = _BPETokenizer.load(str(path))
        return tok
