# CLAUDE.md

## Project

Building a small, practical language model from scratch in Python — targeting **format translation** (e.g. JSON↔YAML, NL→SQL) and **classification** tasks. Efficient and useful, not a demo.

## Principles

- PyTorch for tensor ops; no HuggingFace `transformers` or high-level LM libraries
- Implement all core components from scratch: tokenizer, attention, transformer blocks, training loop
- Architecture: **encoder-decoder transformer** (better fit for translation than decoder-only)
- Optimize for small model size and fast inference — this will run on modest hardware
- Keep modules small and focused; one concept per file

## Structure (evolving)

- `cli.py` — CLI entry point (train, eval, generate)
- `lm/` — model source (tokenizer, model, trainer, data)
- `docs/` — architecture decisions, experiment logs
