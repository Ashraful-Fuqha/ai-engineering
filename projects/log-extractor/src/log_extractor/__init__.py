def main():
    try:
        from log_extractor.cli import main as cli_main
    except ImportError:
        from src.log_extractor.cli import main as cli_main
    cli_main()

__all__ = ["main"]

