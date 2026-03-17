# lm

A language model built from scratch — no shortcuts, no magic, just math and code.

## Goals

- Build a transformer-based LM from first principles — for the joy of it
- Target tasks where small models are genuinely useful: format translation, classification, NL→structured output
- Keep the code readable and the math explicit
- Document the journey in [`docs/`](docs/)

## Quickstart

```bash
pip install -r requirements.txt
python cli.py --help
```

## Docs

Full documentation and design notes at [GitHub Pages](https://tsuberim.github.io/lm).

## Structure

```
lm/
├── cli.py          # Entry point
├── docs/           # Design docs + GH Pages
├── requirements.txt
└── CLAUDE.md       # AI collaboration notes
```
