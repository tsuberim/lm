# CLAUDE.md

## Project

Building a language model from scratch in Python — for learning and fun, but with a real target: tasks where small models can genuinely compete. Think format translation (JSON↔YAML, NL→regex) and classification (intent, sentiment, NER). Not a toy demo, but also not pretending to be GPT.

## Principles

- PyTorch for tensor ops; no HuggingFace `transformers` or high-level LM libraries
- Implement all core components from scratch: tokenizer, attention, transformer blocks, training loop
- Architecture: **decoder-only transformer** (GPT-style, causal self-attention)
- Tasks are defined at inference time via prompting — no hardcoded task heads
- Optimize for small model size and fast inference — this will run on modest hardware
- Keep modules small and focused; one concept per file

## Structure (evolving)

- `cli.py` — CLI entry point (train, eval, generate)
- `lm/` — model source (tokenizer, model, trainer, data)
- `docs/` — architecture decisions, experiment logs
