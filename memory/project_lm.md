---
name: project_lm
description: Goals and architecture decisions for the lm project (language model from scratch)
type: project
---

Building a small, practical language model from scratch (no HuggingFace transformers).

Target tasks: format translation (JSON↔YAML, NL→SQL, etc.) and text classification.

Architecture decisions:
- Encoder-decoder transformer (better fit than decoder-only for translation)
- BPE tokenizer, implemented from scratch
- PyTorch for tensor ops; no high-level LM libraries
- Optimize for small size and fast inference on modest hardware

**Why:** User explicitly wants from-scratch implementation, not fine-tuning, for a practical/production-oriented small LM.
**How to apply:** Don't suggest HuggingFace or pretrained model fine-tuning. Implement all components ourselves.
