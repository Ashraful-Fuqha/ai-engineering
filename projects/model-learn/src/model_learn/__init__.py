def main():
    try:
        from model_learn.cli import main as cli_main
    except ImportError:
        from src.model_learn.cli import main as cli_main
    cli_main()

__all__ = ["main"]
