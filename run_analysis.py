"""Command-line entry point for the Phoenix crime-hotspot forecasting project."""
import argparse
import json
from pathlib import Path
from src.forecasting import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the local source incident CSV")
    parser.add_argument("--output-dir", default="results/generated", help="Output directory for derived CSVs")
    args = parser.parse_args()
    summary = run_pipeline(Path(args.input), Path(args.output_dir))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
