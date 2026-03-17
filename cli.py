import typer
from rich import print

app = typer.Typer(help="lm — language model from scratch")


@app.command()
def train(
    data: str = typer.Argument(..., help="Path to training data"),
    epochs: int = typer.Option(10, help="Number of training epochs"),
    lr: float = typer.Option(1e-3, help="Learning rate"),
):
    """Train the language model."""
    print(f"[bold]Training[/bold] on [cyan]{data}[/cyan] for {epochs} epochs @ lr={lr}")
    raise NotImplementedError("Training not implemented yet")


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Prompt to continue"),
    max_tokens: int = typer.Option(100, help="Max tokens to generate"),
    checkpoint: str = typer.Option(None, help="Path to model checkpoint"),
):
    """Generate text from a prompt."""
    print(f"[bold]Generating[/bold] from: [italic]{prompt}[/italic]")
    raise NotImplementedError("Generation not implemented yet")


@app.command()
def eval(
    data: str = typer.Argument(..., help="Path to eval data"),
    checkpoint: str = typer.Option(None, help="Path to model checkpoint"),
):
    """Evaluate model perplexity on a dataset."""
    print(f"[bold]Evaluating[/bold] on [cyan]{data}[/cyan]")
    raise NotImplementedError("Eval not implemented yet")


if __name__ == "__main__":
    app()
