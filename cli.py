import typer
from rich.console import Console

app = typer.Typer(help="lm — small language model from scratch")
console = Console()


@app.command()
def prepare(
    dataset: str = typer.Option("roneneldan/TinyStories", help="HuggingFace dataset"),
    data_dir: str = typer.Option("data", help="Output directory"),
    vocab_size: int = typer.Option(8192, help="BPE vocabulary size"),
    tokenizer_sample: int = typer.Option(20_000, help="Texts to train tokenizer on"),
):
    """Download dataset and prepare tokenized binary files."""
    from lm.data import prepare as run_prepare
    run_prepare(dataset, data_dir, vocab_size, tokenizer_sample)


@app.command()
def train(
    data_dir: str = typer.Option("data", help="Directory with prepared data"),
    checkpoint_dir: str = typer.Option("checkpoints", help="Where to save checkpoints"),
    # Model
    d_model: int = typer.Option(512, help="Hidden dimension"),
    n_layers: int = typer.Option(8, help="Transformer layers"),
    n_heads: int = typer.Option(8, help="Attention heads"),
    d_ff: int = typer.Option(2048, help="FFN hidden dimension"),
    context_length: int = typer.Option(512, help="Context length"),
    # Training
    batch_size: int = typer.Option(32, help="Batch size"),
    max_steps: int = typer.Option(10_000, help="Training steps"),
    lr: float = typer.Option(3e-4, help="Learning rate"),
):
    """Train the language model."""
    from lm.model import ModelConfig
    from lm.train import TrainConfig, train as run_train
    from lm.tokenizer import BPETokenizer

    tok = BPETokenizer.load(f"{data_dir}/tokenizer.json")

    model_cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        context_length=context_length,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
    )
    train_cfg = TrainConfig(
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        batch_size=batch_size,
        context_length=context_length,
        max_steps=max_steps,
        lr=lr,
    )
    run_train(model_cfg, train_cfg)


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Prompt to continue"),
    checkpoint: str = typer.Option("checkpoints/ckpt_final.pt", help="Model checkpoint"),
    max_tokens: int = typer.Option(200, help="Max new tokens"),
    temperature: float = typer.Option(1.0, help="Sampling temperature (lower = more focused)"),
    data_dir: str = typer.Option("data", help="Directory with tokenizer"),
):
    """Generate text from a prompt."""
    import torch
    from lm.model import GPT
    from lm.tokenizer import BPETokenizer
    from lm.train import get_device

    device = get_device()
    tok = BPETokenizer.load(f"{data_dir}/tokenizer.json")

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = GPT(ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ids = tok.encode(prompt)
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    out = model.generate(x, max_new_tokens=max_tokens, temperature=temperature)
    console.print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    app()
