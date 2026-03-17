"""
BPE tokenizer trained from scratch.

Algorithm:
  1. Start with a vocab of all 256 bytes.
  2. Count adjacent token pair frequencies across the corpus.
  3. Merge the most frequent pair into a new token.
  4. Repeat until vocab_size is reached.
"""

import json
import collections
from pathlib import Path


class BPETokenizer:
    EOT = "<|endoftext|>"

    def __init__(self):
        # Populated by train() or load()
        self.merges: list[tuple[int, int]] = []  # ordered: merge[i] -> token id 256+i
        self.vocab: dict[int, bytes] = {}        # id -> byte string
        self.eot_id: int = -1

    def train(self, texts: list[str], vocab_size: int = 8192, verbose: bool = True):
        assert vocab_size > 257, "vocab_size must be > 257 (256 bytes + EOT)"
        n_merges = vocab_size - 256 - 1  # reserve one slot for EOT

        # Represent corpus as lists of byte ids
        corpus: list[list[int]] = [list(t.encode("utf-8")) for t in texts]

        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = []

        for step in range(n_merges):
            # Count all adjacent pairs
            counts: dict[tuple[int, int], int] = collections.Counter()
            for seq in corpus:
                for a, b in zip(seq, seq[1:]):
                    counts[(a, b)] += 1
            if not counts:
                break

            best = max(counts, key=counts.__getitem__)
            new_id = 256 + step
            self.merges.append(best)
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]

            # Apply merge in-place
            new_corpus = []
            for seq in corpus:
                new_seq, i = [], 0
                while i < len(seq):
                    if i < len(seq) - 1 and seq[i] == best[0] and seq[i + 1] == best[1]:
                        new_seq.append(new_id)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                new_corpus.append(new_seq)
            corpus = new_corpus

            if verbose and (step + 1) % 500 == 0:
                freq = counts[best]
                merged = self.vocab[new_id]
                print(f"  merge {step+1}/{n_merges}: {merged!r} (freq={freq:,})")

        self.eot_id = len(self.vocab)
        self.vocab[self.eot_id] = self.EOT.encode()

    def encode(self, text: str, add_eot: bool = False) -> list[int]:
        tokens = list(text.encode("utf-8"))
        for i, (a, b) in enumerate(self.merges):
            new_id = 256 + i
            out, j = [], 0
            while j < len(tokens):
                if j < len(tokens) - 1 and tokens[j] == a and tokens[j + 1] == b:
                    out.append(new_id)
                    j += 2
                else:
                    out.append(tokens[j])
                    j += 1
            tokens = out
        if add_eot:
            tokens.append(self.eot_id)
        return tokens

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in ids if i != self.eot_id)
        return raw.decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, path: str | Path):
        data = {
            "merges": self.merges,
            "vocab": {str(k): list(v) for k, v in self.vocab.items()},
            "eot_id": self.eot_id,
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        tok = cls()
        data = json.loads(Path(path).read_text())
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        tok.eot_id = data["eot_id"]
        return tok
