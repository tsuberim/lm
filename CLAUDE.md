# CLAUDE.md

## Project

Building a language model from scratch in Python. The goal is clarity and correctness over performance — every piece should be understandable and well-motivated.

## Principles

- Prefer numpy/pure Python for core components; reach for PyTorch only when needed
- No HuggingFace or high-level LM libraries — implement from first principles
- Keep modules small and focused; one concept per file
- Prefer explicit math over clever abstractions

## Structure (evolving)

- `cli.py` — CLI entry point (train, eval, generate)
- `docs/` — design notes, architecture decisions, experiment logs
